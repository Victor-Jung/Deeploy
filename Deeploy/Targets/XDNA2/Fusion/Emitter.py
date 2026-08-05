# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Dedicated FusionRegion -> MLIR-AIE emitter for XDNA2.

Builds the ``aie.device`` module straight from the IR — no synthesized graph, no
Deeploy tiler/deployer. This slice covers single-core elementwise chains
(Slice 1: one kernel; Slice 2: temporal chain with an L1-scratch interior).

Mapping used here:
* graph inputs  -> shim->core ObjectFifos      (STREAM)
* graph outputs -> core->shim ObjectFifos      (STREAM)
* interior edges (TRANSIENT) -> core-local ``aie.buffer`` scratch (never to DRAM)
* all kernels on ONE core; each tile iteration chains their ``func.call``s
* tile size from the region's loop; flat element count from tensor shape

Multi-core placement (Slice 3) needs an explicit physical-placement signal in the
IR and is intentionally not handled yet.
"""

from __future__ import annotations

import aie.ir as ir
from aie.dialects import aie as aie_d
from aie.dialects import aiex as aiex_d
from aie.dialects import arith as arith_d
from aie.dialects import func as func_d
from aie.dialects import scf as scf_d
from aie.extras.context import mlir_mod_ctx

from Deeploy.FusionExtension.Derivations import resolve
from Deeploy.FusionExtension.IR import FusionRegion, Residency

_COL, _CORE_ROW, _SHIM_ROW = 0, 2, 0


def _flat(shape, params) -> int:
    n = 1
    for d in shape:
        n *= resolve(d, params)
    return n


def _ordered_interior(region: FusionRegion):
    consumed, seen, out = region.input_names(), set(), []
    for k in region.kernels:
        for t in k.outputs:
            if t.name in consumed and t.name not in seen:
                seen.add(t.name)
                out.append(t.name)
    return out


def emit(region: FusionRegion) -> str:
    params = region.params
    gin, gout, interior = region.graph_inputs(), region.graph_outputs(), _ordered_interior(region)
    refs = {t.name: t for k in region.kernels for t in list(k.inputs) + list(k.outputs)}

    # Residency contract: fail loud on cases this emitter would silently mishandle.
    # Supported model = STREAM boundary I/O + TRANSIENT (L1 scratch) interiors only.
    interior_set = set(interior)
    for name, edge in region.edges.items():
        if edge.residency == Residency.RESIDENT:
            raise NotImplementedError(f"tensor '{name}' is RESIDENT; accumulator/invariant lowering "
                                      f"is not implemented yet")
        if edge.residency == Residency.TRANSIENT and name not in interior_set:
            raise NotImplementedError(f"tensor '{name}' is TRANSIENT but not an interior edge (produced "
                                      f"and consumed in-region); a region input/output cannot be TRANSIENT")
        if name in interior_set and edge.residency != Residency.TRANSIENT:
            raise NotImplementedError(f"interior tensor '{name}' has residency {edge.residency.name}; only "
                                      f"TRANSIENT (L1 scratch) interiors are supported")

    total = _flat(refs[gin[0]].shape, params)
    for n in gin + gout:
        assert _flat(refs[n].shape, params) == total, "elementwise emitter needs uniform tensor size"
    tile_elems = resolve(region.loops[0].tile, params)
    assert total % tile_elems == 0, f"tile {tile_elems} must divide total {total}"
    num_tiles = total // tile_elems

    with mlir_mod_ctx() as ctx:

        @aie_d.device(aie_d.AIEDevice.npu2)
        def _dev():
            bf16 = ir.BF16Type.get()
            i32 = ir.IntegerType.get_signless(32)
            tile_ty = ir.MemRefType.get((tile_elems,), bf16)
            flat_ty = ir.MemRefType.get((total,), bf16)
            sub_ty = aie_d.ObjectFifoSubviewType.get(tile_ty)
            Consume, Produce = aie_d.ObjectFifoPort.Consume, aie_d.ObjectFifoPort.Produce

            compute = aie_d.tile(_COL, _CORE_ROW)
            shim = aie_d.tile(_COL, _SHIM_ROW)

            in_fifo = {n: f"{n}_in" for n in gin}
            out_fifo = {n: f"{n}_out" for n in gout}
            for n in gin:
                aie_d.object_fifo(in_fifo[n], shim, [compute], 2, tile_ty)
            for n in gout:
                aie_d.object_fifo(out_fifo[n], compute, [shim], 2, tile_ty)

            scratch = {n: aie_d.buffer(compute, tile_ty, name = f"scratch_{n}") for n in interior}

            declared = set()
            for k in region.kernels:
                if k.kernel_symbol in declared:
                    continue
                if not k.kernel_obj:
                    raise ValueError(f"kernel '{k.kernel_symbol}' (node '{k.node}') has no kernel_obj; "
                                     f"set BoundKernel.kernel_obj to the link .o")
                declared.add(k.kernel_symbol)
                argtys = [tile_ty] * (len(k.inputs) + len(k.outputs)) + [i32]
                aie_d.external_func(k.kernel_symbol, argtys, link_with = k.kernel_obj)

            @aie_d.core(compute)
            def _core():
                for _ in scf_d.for_(0, 0x7FFFFFFFFFFFFFFF, 1):
                    for _ in scf_d.for_(0, num_tiles, 1):
                        acq = {}
                        for n in gin:
                            a = aie_d.objectfifo_acquire(sub_ty, Consume, in_fifo[n], 1)
                            acq[n] = aie_d.objectfifo_subview_access(tile_ty, a, 0)
                        for n in gout:
                            a = aie_d.objectfifo_acquire(sub_ty, Produce, out_fifo[n], 1)
                            acq[n] = aie_d.objectfifo_subview_access(tile_ty, a, 0)
                        size_c = arith_d.constant(i32, tile_elems)

                        def operand(name):
                            return acq[name] if name in acq else scratch[name]

                        for k in region.kernels:
                            call_args = [operand(t.name) for t in k.inputs] + [operand(t.name) for t in k.outputs
                                                                              ] + [size_c]
                            func_d.call([], k.kernel_symbol, call_args)

                        for n in gin:
                            aie_d.objectfifo_release(Consume, in_fifo[n], 1)
                        for n in gout:
                            aie_d.objectfifo_release(Produce, out_fifo[n], 1)
                        scf_d.yield_([])
                    scf_d.yield_([])

            @aiex_d.runtime_sequence(*([flat_ty] * (len(gin) + len(gout))))
            def _seq(*args):
                arg_of = {n: args[i] for i, n in enumerate(gin + gout)}
                dims = [aie_d.bd_dim_layout(size = 1, stride = 0)] * 3 + [aie_d.bd_dim_layout(size = total, stride = 1)]
                in_tasks, out_tasks = [], []
                for n in gin:
                    task = aiex_d.dma_configure_task_for(in_fifo[n])
                    with ir.InsertionPoint(task.body.blocks.append()):
                        aie_d.dma_bd(arg_of[n], offset = 0, len = total, dimensions = dims, burst_length = 0)
                        aie_d.end()
                    aiex_d.dma_start_task(task)
                    in_tasks.append(task)
                for n in gout:
                    task = aiex_d.dma_configure_task_for(out_fifo[n], issue_token = True)
                    with ir.InsertionPoint(task.body.blocks.append()):
                        aie_d.dma_bd(arg_of[n], offset = 0, len = total, dimensions = dims, burst_length = 0)
                        aie_d.end()
                    aiex_d.dma_start_task(task)
                    out_tasks.append(task)
                for t in out_tasks:
                    aiex_d.dma_await_task(t)
                for t in in_tasks:
                    aiex_d.dma_free_task(t)

        module = ctx.module
        assert module.operation.verify(), "fusion emitter: MLIR verification failed"
        return str(module)
