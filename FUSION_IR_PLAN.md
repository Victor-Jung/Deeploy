# Fusion IR — Implementation Plan (E2E-per-slice)

> Companion to `FUSION_IR_NOTES.md` (design handoff). This file is the build plan.
> Branch: `fusion-ir`. Rewritten so **every step lands a hardware-verified E2E slice**.

## Status

- **Slices 0, 1, 2 — DONE, HW-verified** (dedicated emitter, 0 errors on NPU):
  - Slice 0: collection blocker fixed (`DeeployTypes.py` raw-strings + slim `FusionExtension/__init__`);
    emitter skeleton stood up. `Deeploy/Targets/XDNA2/Fusion/Emitter.py`.
  - Slice 1: single-node Relu, schedule-driven (no synthetic graph / tiler) → NPU, 0/65536 errors.
  - Slice 2: temporal `relu(A+B)` on one core, interior `s` = L1 `aie.buffer` scratch → NPU, 0/65536.
  - Tests (renamed for what they test, not the slice number): `test_fusion_single_relu.py`,
    `test_fusion_temporal_add_relu.py` (`-m xdna2`, IR-as-input: author region → pickle + numpy golden →
    runner → NPU numeric check). Derivation oracle (`test_fusion_derivations.py`, 5 T0) still green.
    All 7 pass on a cold cache.
  - Superseded `XDNA2Lowering.py` + `FusionDeployer.py` removed. New: `FusionExtension/Reference.py`
    (numpy golden), `DeeployTest/fusionTestUtils.py`, rewritten `generateFusionNetwork_xdna2.py`.
  - Review round applied: `BoundKernel.kernel_obj` now carries the link `.o` (removed the emitter's
    hardcoded op→obj table); emitter enforces a fail-loud residency contract (see deferred blockers B6).
- **Next:** Slice 3 (pipeline, two cores + core→core ObjectFifo) — needs an explicit physical-placement
  signal in the IR (today the emitter hardwires one core); then Slice 4 (GEMM + accumulator).

## Deferred emitter blockers (from the slice-0/1/2 review)

The current emitter is hardwired to elementwise / current kernels. These are known and deferred; most
resolve when the **CP tiler is integrated** and when more of this info moves **into the IR**. The IR
must be rich enough to express them — validating that is the point of the next batch of examples.

- **B1 — kernel ABI hardcoded to `(memref×N, i32 size)`.** `external_func` decl + call assume all
  operands are same-shaped memrefs then one i32 element count. Breaks for GEMV/GEMM
  `(i32 m, i32 row_offset, A, B, C)`. Fix ties to tiler integration; the IR should carry a per-kernel
  ABI descriptor. *(Deferred — tiler.)*
- **B3 — one global flat tile type** shared by every fifo/scratch, guarded by "all tensors same size".
  Breaks for MatMul (A[M,K],B[K,N],Y[M,N] differ). Per-edge tile shape from `access` + `Loop.tile`.
  *(Deferred — tiler; belongs in the IR.)*
- **B4 — global size from `gin[0]`.** Same root cause as B3; the region must carry per-tensor tile
  geometry rather than the emitter assuming uniformity. *(Deferred — tiler; belongs in the IR.)*
- **B5 — dtype hardcoded to bf16** in the emitter though `TensorRef.dtype` knows better. Should be read
  from the IR tensor declaration. *(Deferred — encode/consume dtype from the IR.)*
- **B7 — runtime-seq arg order (`gin+gout`) is an implicit contract** with the host testbench; the
  emitter (first-appearance) and generate script (npz name integer) can disagree silently. Needs one
  canonical region-derived ordering + assert. *(Deferred — on the blocker list.)*
- **B8 — no check that `Loop.tile` suits the kernel's vector factor** → cryptic aiecc failure. Resolves
  when the CP tiler owns tile-size legality. *(Deferred — tiler.)*

## Locked decisions

1. **E2E gate = hardware numeric run, every step.** A slice is "done" only when a hand-authored
   `FusionRegion` is emitted to MLIR, the MLIR passes the built-in verifier, is built to an xclbin,
   runs on the NPU, and matches a numpy golden within tolerance. No slice counts on MLIR-verify alone.
2. **Dedicated IR→MLIR emitter.** A new emitter builds the `aie.device` module directly from the
   `FusionRegion` (loops → `scf.for`, edges → ObjectFifos / L1 scratch, maps → tiles/placement,
   `Loop.tile` → tile sizes). We do **not** synthesize a graph, and we do **not** use Deeploy's
   tiler / memory scheduler / stock deployer. We **reuse** the build+run harness (CMake → aiecc.py →
   xclbin + `npu_insts` → XRT host testbench → numeric compare) and the existing AIE kernels.

Why bypassing the tiler is fine: the `aie` dialect + `aiecc.py` own low-level L1 allocation
(ObjectFifo buffers, core scratch), so we only need tile *sizes* (which the IR carries) and topology
(which the region specifies) — not Deeploy's memory scheduler.

## Principles

- **Walking skeleton, then features.** Slice 1 is the smallest schedule-driven single node; each later
  slice adds exactly one lowering capability and its own HW test. Order follows the natural difficulty
  ramp: single node → temporal chain (one core) → pipeline (multi-core) → GEMM/accumulator → FFN.
- **The region drives codegen.** From Slice 1 on, the emitter reads the region — no decorative IR. A
  slice's HW test is the proof its feature actually shapes the running program.
- **Numpy golden per region.** A target-agnostic reference evaluator computes expected outputs from the
  region's ops, so every slice has a reference without hand-written expected data.
- **Derivation oracle stays in parallel.** `test_fusion_derivations.py` (T0, pure Python) keeps
  validating the cost model (`offchip_bytes`, `legality`, accumulator/footprint). It is not an E2E
  slice; it is the numeric oracle that later picks *which* schedule to lower.

## Module layout

```
Deeploy/FusionExtension/            # target-agnostic; SLIM __init__ (no backend imports)
  IR.py                             # dataclasses + enums (done)
  Derivations.py                    # cost model / legality (done)
  Reference.py                      # NEW: FusionRegion -> numpy golden (op table: Relu, Add, Mul, MatMul, ...)

Deeploy/Targets/XDNA2/Fusion/       # XDNA2-specific lowering
  Emitter.py                        # NEW: FusionRegion -> aie.ir.Module (the dedicated emitter)
  __init__.py

DeeployTest/
  generateFusionNetwork_xdna2.py    # region -> Emitter -> network.mlir (+verify) + testinputs/outputs.h
  deeployRunner_xdna2_fusion.py     # generate -> build -> run on NPU -> numeric check (single-core for now)
  test_fusion_slice1_singlenode.py  # xdna2 HW pytest, one per slice
  test_fusion_slice2_temporal.py
  ...
  test_fusion_derivations.py        # T0 oracle (unchanged)
```

Two cleanups this layout resolves: (a) XDNA2-specific lowering moves **out** of the agnostic
`FusionExtension` package into `Targets/XDNA2/Fusion/`; (b) `FusionExtension/__init__.py` is slimmed to
import only `IR` + `Derivations`, so the pure-IR layer no longer drags in the deployer stack. The
emitter borrows emission idioms from the existing `Targets/XDNA2/CodeTransformationPasses/MLIR*Pass.py`.

The current `FusionExtension/XDNA2Lowering.py` + `FusionDeployer.py` (synthetic-graph + stock-deployer
approach) are **superseded** by the emitter and will be removed once Slice 1 lands.

---

## Slice 0 — Unblock + emitter skeleton (prep, folded into Slice 1's PR)

- **Fix the cold-compile collection blocker.** `DeeployTypes.py:675,678` has a pre-existing invalid
  escape (`re.sub('\.', …)`); under root `pytest.ini`'s `filterwarnings=error` a fresh compile
  escalates the `SyntaxWarning` to a `SyntaxError`, breaking fusion pytest collection on a cold cache.
  Raw-string it (`r'\.'`) and slim `FusionExtension/__init__.py`.
- **Emitter skeleton:** `FusionRegion → mlir.ir.Module` with `aie.device(npu2)`, plus
  `module.operation.verify()` and MLIR text-out. No feature yet.
- **Done-when:** fusion pytest collects on a cold cache; skeleton emits a verifiable empty device.

## Slice 1 — Single node, schedule-driven, HW-verified

- **Goal:** the emitter lowers a single-op region (`Relu`) with **no** synthetic graph / tiler.
- **Emitter capability:** 1 core from `placement`; input/output ObjectFifos (shim↔core) sized from
  `edges` + `Loop.tile`; one `aie.core` with `scf.for` over `numTiles` calling `relu_bf16`;
  `runtime_sequence` with shim `dma_bd`s; `external_func` decl. Tile size from the IR.
- **Reference:** `Reference.py` numpy `relu`.
- **HW test:** `test_fusion_slice1_singlenode.py` (`-m xdna2`) — generate → verify → build → NPU →
  0 errors vs golden.
- **Done-when:** single-Relu region runs on NPU with 0 errors, produced by the dedicated emitter.
- **Kernels:** `relu_bf16` (exists).

## Slice 2 — Multi-node temporal fusion (one core, kernel chain, L1 scratch), HW-verified

- **Goal:** `Y = relu(A + B)` as two kernels on **one** core; interior stays in L1.
- **Emitter capability:** two ops placed on the same core (`placement` → same engine); the interior
  edge (`TRANSIENT`, "L1") becomes a core-local scratch `memref`; one `scf.for` body issues two
  `func.call`s (`add_bf16` into scratch, `relu_bf16` scratch→out); one shared input/interior/output
  buffering. No interior ObjectFifo, no interior shim DMA.
- **Reference:** numpy `relu(A + B)`.
- **HW test:** `test_fusion_slice2_temporal.py` — verify (one `aie.core`, two `func.call`s, scratch
  memref, no interior fifo) → build → NPU → 0 errors.
- **Done-when:** the fused chain runs on NPU and matches `relu(A+B)`; interior never leaves the core.
- **Kernels:** `add_bf16`, `relu_bf16` (exist).

## Slice 3 — Multi-node pipeline fusion (two cores, core→core ObjectFifo), HW-verified

- **Goal:** same `Y = relu(A + B)` but split across **two** cores, streamed.
- **Emitter capability:** two ops on distinct cores (`placement` → two engines); the interior
  `TRANSIENT` edge becomes a **core→core ObjectFifo** (producer core → consumer core); no shim DMA for
  the interior tensor; each core keeps its own `scf.for`.
- **Reference:** numpy `relu(A + B)`.
- **HW test:** `test_fusion_slice3_pipeline.py` — verify (2 `aie.core`, one core→core objectfifo, no
  interior `dma_bd`) → build → NPU → 0 errors.
- **Done-when:** two-core pipeline runs on NPU, matches, interior streams core→core (measurably no
  interior DRAM traffic).
- **Kernels:** `add_bf16`, `relu_bf16` (exist).

## Slice 4 — Single GEMM (K-reduction + RESIDENT accumulator), HW-verified

- **Goal:** `Y = A @ B`, reduction over K, accumulator lowering.
- **Emitter capability:** `REDUCE` loop → K `scf.for`; `RESIDENT` output whose access omits the reduce
  axis → accumulator: allocate + zero-init outside the K loop, accumulate across it, drain after;
  `SpatialMap` → core placement.
- **Reference:** numpy `A @ B` (bf16 tolerance, ~40 ULP).
- **HW test:** `test_fusion_slice4_gemm.py` — verify (K `scf.for`, accumulator zero+drain) → build →
  NPU → within-tolerance vs golden.
- **Done-when:** single GEMM runs on NPU within tolerance.
- **Kernels:** `matmul_bf16` — **does not exist on this branch**; cherry-pick the bf16 matvec kernel +
  tiling pattern from `pr/spatial-mapping-beta` as the base, or add a small matmul kernel.

## Slice 5 — FFN temporal chain `Gemm → act → Gemm` (small dims), HW-verified

- **Goal:** the headline — combine the accumulator (Slice 4) with a `TRANSIENT` interior (Slice 2),
  scheme-D shape at small dims.
- **Emitter capability:** `matmul_acc` accumulate-into-RESIDENT across the shared chunk loop; interior
  `H`/`Ha` kept `TRANSIENT`; verifies the cost model's `2TD + 2DF·(T/64R)` prediction against measured
  DRAM traffic.
- **Reference:** numpy FFN.
- **HW test:** `test_fusion_slice5_ffn.py` — verify + build + NPU within tolerance; assert interior
  tensors emit no DRAM traffic.
- **Done-when:** FFN scheme-D runs on NPU within tolerance and hits the predicted traffic.
- **Kernels:** `matmul_bf16`, `matmul_acc_bf16`, an activation kernel (`relu_bf16`/`gelu_bf16`).

---

## Known gaps mapped to slices (from design review)

| Gap | Description | Slice |
|-----|-------------|-------|
| collection blocker | `DeeployTypes` escape-seq + `filterwarnings=error` breaks cold-cache collection | 0 |
| R1/R3 | interior core→core ObjectFifo creation + both-sided binding | 3 |
| R2 | interior edge as an on-chip mover/scratch, not `None` | 2 (scratch) / 3 (fifo) |
| R5 | REDUCE-into-RESIDENT accumulator (no analog today) | 4 |
| ① | `offchip_bytes` boundary-I/O term (already applied in Derivations) | done |
| ② | `CARRIED`/kernel-combinator/drain-hook for flash-attention | future |

## Out of scope (for now)
- Deeploy tiler / memory scheduler (emitter owns tiling from the IR).
- Phase 1 "elect-only" path (regions hand-authored until the emitter is proven; then derive
  `BoundKernel`s from real bindings).
- Auto group formation / schedule search (manual regions first; the derivation oracle scores them).
- Flash-attention online-softmax (`CARRIED` extension).
- Graduating `FusionRegion` to an MLIR dialect.

## Open decisions (surface when hit)
- **Slice 3 interior fifo depth / placement** (adjacent cores vs via memtile) — decide when lowering it.
- **Slice 4 GEMM kernel source** — cherry-pick the sibling-branch matvec vs write a fresh matmul.
- **Multi-core in the runner** — `deeployRunner_xdna2_fusion.py` is single-core; Slice 3 needs it to
  activate ≥2 cores (add a `--num-*`-style knob or derive the array shape from the region).
