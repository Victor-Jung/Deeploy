# Fusion IR for XDNA2 — Handoff Context

> **For the receiving Claude instance:** This is a design/context handoff for ongoing work on
> operator-fusion support in Deeploy targeting the AMD XDNA2 NPU. Read it fully before acting.
> It records decisions already made, findings from a codebase analysis, the IR design, a worked
> example, and the concrete next steps. Nothing here is committed to the backend yet unless noted.

## 1. Goal

Add a way to **describe, explore, and compile multi-layer operator-fusion schemes** for the XDNA2
NPU using Deeploy as the backend. The near-term deliverable is a small **Python fusion IR** (to be
hand-authored / searched over) that lowers into Deeploy's existing XDNA2 MLIR-AIE code generation.
Longer term the IR may graduate to an MLIR dialect once its abstractions are frozen.

## 2. Hardware model (XDNA2 / AIE-ML)

- Array of **64×64 output-stationary PE tiles**, physically **8×4 = 32 tiles** (usable as 8×4 or a
  4×4 sub-grid). Per-tile L1 ~64KB; **memtiles** (L2) ~512KB each; shim tiles to DRAM.
- GEMM dataflow (when it exists): each PE holds a 64×64 INT32/accumulator output block, K is
  streamed. Inputs multicast across columns (**input reuse = #cols C**), weights multicast down
  rows (**weight reuse = #rows R**). One "wave" of an R×C region produces a (64R)×(64C) output block.
- **Off-chip transfer formula** for one GEMM `Out[M,N]=A[M,K]@B[K,N]`, tiled with reuse:
  `input = M*N*K/(64*C)`, `weight = M*N*K/(64*R)`, total `= (M*N*K/64)*(1/R + 1/C)`.
  Per-mega-block operand cost `= 64*K*(R+C)`.

## 3. Key finding: where "temporal" lives on XDNA2

Deeploy has TWO backends. The **PULP path emits C tile-loops** (`TilingCodeGeneration`, double-buffer
`switch(TILING_I%2)`). **XDNA2 does NOT use that** — `XDNA2Deployer.generateMLIR()` builds an AIE/IRON
MLIR module. On XDNA2 the temporal aspect is:

- **intra-core time** = `scf.for 0..numTiles` inside each `@aie_d.core` (`MLIRComputeCorePass`), with
  `objectfifo_acquire`/`release` around **one** `func.call`.
- **inter-core / host time** = **ObjectFifo topology + depths** + the runtime-sequence's 3 phases
  (configure-all DMAs → await-all outputs → free-all inputs). No explicit inter-node ordering is
  emitted; ObjectFifo back-pressure + the infinite per-core outer loop ARE the pipeline.
- **tile sizes** come from the OR-Tools tiler (which is already multi-node capable — see §5).

So the thing the ONNX graph fails to encode is not "tiles" per se but **how a group of nodes shares a
tile loop and where each interior edge lives** (L1 scratch vs core→core ObjectFifo). That is what the
fusion IR must add.

## 4. Two native fusion shapes on this hardware

1. **Temporal fusion (kernel chain, ONE core):** group nodes on the same PE; interior edge = L1
   `memref` scratch; multiple `func.call`s inside one `scf.for`. Needs an `MLIRComputeCorePass`
   change (emit a chain + scratch). Good for epilogue / elementwise chains.
2. **Pipeline fusion (core→core streaming):** group nodes on different PEs; interior edge = a
   **core→core ObjectFifo** (direct or via memtile); each core keeps its own `scf.for`; streaming by
   back-pressure. **Nearly free** — it's exactly what `fifoRegistry` + `objectfifo_link` already do
   for Split/Concat. **Do this first.**

## 5. Deeploy codebase findings (seams)

**Core IR** (`Deeploy/DeeployTypes.py`):
- Graph = onnx_graphsurgeon + `NetworkContext` (symbol table of `VariableBuffer`s, keyed by tensor name).
- Per-edge annotations on buffers: `_memoryLevel`, `_dataMoverEngine`, `_logicalParent`/`_chunkOffset`
  (serialized to ONNX doc_string). Per-node: `node.attrs['engine']` (placement), `['mapping']`.
- `ExecutionBlock` (MCU) / `MLIRExecutionBlock` (XDNA2) = the executable unit; holds ordered code
  snippets + `patternMemoryConstraint`.
- `DataMoverEngine` (~line 2377) is a near-empty **stub** — natural seam to make data movement first-class.
- Frontend flow: `parse` (NodeParser → `operatorRepresentation`) → `typeCheck` (NodeTypeChecker elects
  a `NodeBinding` = template + TileConstraint) → `bind` (materializes the ExecutionBlock).
  **Election is already separable from materialization** — this is the reuse seam (§7).

**Tiling** (`Deeploy/TilingExtension/`):
- Tile sizes from an **OR-Tools CP model** (`TilerModel`); per-op geometry in
  `TileConstraint.addGeometricalConstraint` (these are the affine "linear dependency laws").
- **The solver is already multi-node/pattern aware**: geometric constraints of nodes in one pattern
  are auto-unified via shared tile-size variables. `PatternMemoryConstraints` = list of nodes; has an
  `intermediate` tensor bucket; can express on-chip-resident / transient / single-level tensors.
- **Codegen is single-node-locked** by three literal asserts `"Only layerwise supported for now!"`
  (`TilingCodeGeneration.py:250`, `TilingVariableReplacement.py:116,217`) — but that's the C path.
- Temporal data structures: `TilingSchedule` (list of per-tile `{tensor: HyperRectangle}`),
  produced by each op's `serializeTilingSolution`. Not kept as IR after codegen.

**XDNA2 backend** (`Deeploy/Targets/XDNA2/`):
- `generateMLIR()` builds AIE dialect: `MLIRObjectFifoPass` (data movement = ObjectFifos, depth 2),
  `MLIRComputeCorePass` (per-core loop + one `func.call`), `MLIRRuntimeSequencePass` (host DMA `dma_bd`),
  `MLIRDistributeLinkPass`/`MLIRJoinLinkPass` (Split/Concat via `objectfifo_link`).
- **Placement = engine coloring**: `node.attrs['engine']="AIE_c{c}r{r}"` → `aie.tile(c,r)`.
- **On-chip "skip the shim" already exists**: `fifoRegistry[(logical_parent,col,row)]` lets a consumer
  reuse a memtile↔core fifo (used by Split/Concat). Generalize this to interior fused edges.
- **"No host DMA for this port" already exists**: `argIndexMap[key]=None` skip in
  `MLIRRuntimeSequencePass` / `_resolveDmaPlacement`.
- `_checkDataMoverInvariants` (Deployer.py ~153) **forbids `_dataMoverEngine=None`** — must relax to
  mark interior/on-chip tensors.
- **`HybridElementwiseSpatialSplitPass` is data-parallel spatial split of ONE op, NOT fusion.**
- **Supported ops today**: Add, Mul, Gelu, Relu, SiLU, Tanh, LayerNorm (elementwise/row-wise, bf16),
  + Split/Concat (pure data movement). **No GEMM/MatMul/Conv kernel, no K-reduction tile constraint,
  no accumulate-across-PEs** (JoinLink concatenates, does not reduce).

## 6. Architectural decisions taken

- **Do NOT round-trip the fusion schedule back through ONNX.** ONNX can't carry loop nests / residency /
  fifo topology; serializing back to it either loses the schedule or bloats ONNX into a fake IR (and if
  the IR is MLIR, it becomes MLIR→ONNX→MLIR-AIE, a strict downgrade).
- **Keep ONNX as the *source* subgraph only.** Hand off to Deeploy at its **mid-level scheduled IR**
  (`PatternMemoryConstraints` + `MLIRExecutionBlock`), not at ONNX.
- **Tiling becomes a subroutine of exploration** (drive the multi-node tiler from the IR), because
  fusion feasibility IS a tile-size question. Don't let Deeploy re-decide tile size after fusion.
- **Reuse boundary**: run Deeploy `parse`+`typeCheck` to get per-node **tiling-agnostic bound-kernel
  descriptors**; the IR owns materialization (loops/fifos/placement) at the group level.
- **Descriptors are tiling-agnostic** (decided): they carry kernel symbol + parsed attrs + typed I/O +
  a *reference* to the TileConstraint, but NO tile sizes. The IR/tiler resolves tiling.
- Present a committed scheme to Deeploy as a thin **`FusedLayer`** that plugs into `layerBinding` so the
  existing emission driver + AIE passes work. IR references bindings (composition), does not subclass
  `ONNXLayer`.
- **Durable split**: kernel/binding layer (ONNX op → AIE kernel symbol + tiling geometry + types) is
  stable metadata that survives a future move to an MLIR dialect; only the *structure* layer (the IR)
  changes representation.

## 7. The IR (Python), tiling-agnostic descriptors + fusion region

```python
# ---- Layer 1: tiling-agnostic descriptors (from Deeploy elect-only path) ----
@dataclass
class TensorRef:
    name: str; shape: tuple; dtype: str          # e.g. ("T","D"), "bf16"

@dataclass
class BoundKernel:
    node: str; op: str
    kernel_symbol: str                            # AIE external_func symbol
    operator_repr: dict                           # parsed attrs (NodeParser)
    inputs: list[TensorRef]; outputs: list[TensorRef]
    tile_constraint: type                         # reference to a Deeploy TileConstraint (NOT solved)

# ---- Layer 2: the fusion IR (structure) ----
class LoopKind(Enum):   MARCH=auto(); REDUCE=auto(); PARALLEL=auto()
class Residency(Enum):  STREAM=auto(); TRANSIENT=auto(); RESIDENT=auto()

@dataclass
class Loop:          axis:str; extent:str; tile:str; kind:LoopKind   # tile symbolic (R, KF, C)
@dataclass
class EdgeSchedule:  tensor:str; access:dict; residency:Residency; level:str  # access: axis->dim idx
@dataclass
class SpatialMap:    node:str; row_axis:str; col_axis:str; stationary:str; core_range:tuple
@dataclass
class FusionRegion:
    kernels:list; loops:list; placement:dict; edges:dict; maps:dict; params:dict
```

### Derivation rules (pure functions over a FusionRegion)
1. **Legality**: every `REDUCE` axis must be absent from all output `access` maps (summed away); no
   `MARCH` axis may be a reduction axis of any kernel (else fusion barrier).
2. **Accumulator detection**: a `RESIDENT` tensor whose `access` omits a `REDUCE` axis = an accumulator;
   allocate+zero at the level just outside that loop, `+=` across it, drain after.
3. **DMA firing/count**: for each `STREAM` tensor, refetch factor = product of trip counts of loops
   OUTER to its compute-at level whose axis ∉ its `access`; bytes = refetch × footprint. TRANSIENT/
   RESIDENT emit no DRAM.
4. **Footprint**: peak over time of live RESIDENT + TRANSIENT tensors; capacity check per level.

## 8. Worked example: FFN `Gemm → Gelu → Gemm`, scheme D (the winning corner)

Structure: `X[T,D] --W1--> H[T,F] --gelu--> Ha[T,F] --W2--> Y[T,D]`, with `F=4D`.

**Scheme D** = keep the down-proj output `Y` as a RESIDENT accumulator, chunk the hidden dim `F`,
keep each `H`/`Ha` chunk TRANSIENT (never to DRAM), stream weights. Preserves weight reuse `R` (2-D
array mapping) AND kills the `H` round-trip.

Descriptors (⚠ = does not exist in backend yet):
- up:   Gemm  kernel `matmul_bf16` ⚠,       TileConstraint `GemmTileConstraint` ⚠
- act:  Gelu  kernel `gelu_bf16_vector` ✅,  TileConstraint `UnaryTileConstraint` ✅
- down: Gemm  kernel `matmul_acc_bf16` ⚠ (accumulating), `GemmTileConstraint` ⚠

FusionRegion (scheme D):
```python
loops = [ Loop("tok","T","64*R",MARCH),      # token panels
          Loop("fchunk","F","KF",REDUCE),     # hidden chunks = down-proj K (contraction)
          Loop("dgroup","D","64*C",PARALLEL)] # output column groups
placement = {"up_proj":"fchunk","act":"fchunk","down_proj":"dgroup"}
edges = {
  "X":  ("X",  {"tok":0},              RESIDENT,  "MEMTILE"),  # invariant to fchunk -> load once/panel
  "W1": ("W1", {"fchunk":1},           STREAM,    "DRAM"),
  "H":  ("H",  {"tok":0,"fchunk":1},   TRANSIENT, "L1"),
  "Ha": ("Ha", {"tok":0,"fchunk":1},   TRANSIENT, "L1"),
  "W2": ("W2", {"fchunk":0,"dgroup":1},STREAM,    "DRAM"),
  "Y":  ("Y",  {"tok":0,"dgroup":1},   RESIDENT,  "MEMTILE"), # omits REDUCE fchunk -> accumulator
}
maps = { "up_proj":   row=tok col=fchunk stationary=H,
         "down_proj": row=tok col=dgroup stationary=Y }
```
Derived off-chip: `2TD + 2DF*(T/64R)` (H traffic eliminated; weights streamed once per panel).

### The winning corner (FFN)
Keep weight reuse `R` and kill `H`; wins vs layer-by-layer when BOTH hold:
- **accumulator fits**: `size(Y_acc) = 64*R*D*4 bytes ≤ S` (on-chip budget) ⇒ `R ≤ S/(256*D)`
- **reuse not too degraded**: `D*(n-R)/(R*n) < 64`  (n = baseline weight reuse; equality/near-`R=n`
  gives a pure win = the saved H traffic `2TF`).
- Clean corner `R=n` needs `S ≥ 256*n*D`.
- **ViT-B (D=768): wins** (accumulator ~1.6MB at R=8). **DiT-XL (D=1152): borderline win.**
  **Llama (D=4096): loses** — R=8 accumulator is ~8MB, doesn't fit; forced R↓ loses weight reuse.
- Simpler **regime-1** (cache all of H in memtile, no REDUCE loop): flip `H`/`Ha` to RESIDENT/MEMTILE,
  drop the `fchunk` loop, `Y` becomes STREAM. Footprint flips `64R*D` → `64R*F` (4× larger). Use when H fits.

## 9. How to read a FusionRegion as pseudo-code

A FusionRegion is not meant to be a literal source program. It is a compact schedule
description. The safest way to read it is:

- `loops` describe the outer schedule axes.
- `placement` says which node runs at which schedule level or spatial region.
- `edges` describe where each tensor lives and whether it is streamed, transient,
  or resident.
- `maps` describe how a node is placed across the PE grid.
- `params` resolve symbolic sizes like `T`, `D`, `F`, `R`, `C`, `KF`.

Important interpretation rules:
- `MARCH` is time-like iteration over output panels.
- `REDUCE` is an accumulation axis. A tensor that is `RESIDENT` and omits a `REDUCE`
  axis is usually an accumulator.
- `PARALLEL` is spatial placement, basically a parfor.
- `TRANSIENT` means on-chip scratch; it should not be read as DRAM storage.
- `RESIDENT` means on-chip storage that persists across the relevant loop.

### Human reading of the FFN scheme-D example

A simple pseudo-code reading is:

```text
for each token panel:
  keep X and Y on-chip
  zero Y
  for each hidden chunk:
    in parallel across the PE grid:
      compute up projection
      apply gelu locally
      accumulate down projection into Y
  store Y
```

More detailed pseudo-code:

```text
# Symbolic dimensions
# T    = token count
# D    = model width
# F    = 4 * D
# R    = PE rows used for the wave
# C    = PE cols used for the wave
# KF   = 64 * C          # K-chunk size for the reduction axis
# tok_tile = 64 * R      # token panel size handled per wave
# d_tile   = 64 * C      # output column group size
# acc_tile  = 64 * R * D # resident Y accumulator footprint for one tok panel

for tok0 in range(0, T, tok_tile):
    # MEMORY CHOICE:
    # Y is RESIDENT in MEMTILE, so it is allocated once per tok panel
    # and updated across all fchunks.
    Yacc = resident_buffer(level="MEMTILE", shape=(tok_tile, D))
    zero(Yacc)

    # X is RESIDENT in MEMTILE too, but read-only for this whole tok panel.
    Xpanel = resident_input(level="MEMTILE", slice=X[tok0 : tok0 + tok_tile, :])

    for f0 in range(0, F, KF):
        # This loop is the REDUCE axis.
        # H and Ha are TRANSIENT: they exist only on-chip for this chunk,
        # then disappear. They do not go to DRAM.

        W1_chunk = stream_from_dram(W1[:, f0 : f0 + KF])
        W2_chunk = stream_from_dram(W2[f0 : f0 + KF, :])

        # Spatial mapping:
        # The region is executed over an R x C PE sub-grid.
        # up_proj and act use the same placement on the grid.
        # down_proj uses a spatial map over dgroup.
        for r in range(R):
            for c in range(C):
                # Each PE owns one 64x64 block in this wave.
                # row axis = tok, col axis = fchunk for up_proj/act
                # row axis = tok, col axis = dgroup for down_proj

                # ---------- PHASE 1: up projection ----------
                # input access:
                #   X uses {"tok": 0}
                #   W1 uses {"fchunk": 1}
                # meaning:
                #   X is indexed by the tok panel, not by fchunk
                #   W1 is indexed by the current hidden chunk
                H_block = matmul(
                    Xpanel[row_slice=64*r : 64*(r+1), full_D],
                    W1_chunk[full_D, col_slice=64*c : 64*(c+1)]
                )
                # H is TRANSIENT in L1
                store_transient(level="L1", name="H", row=r, col=c, value=H_block)

                # ---------- PHASE 2: gelu ----------
                # act reads H locally and writes Ha locally.
                # No DRAM traffic.
                Ha_block = gelu(load_transient("H", row=r, col=c))
                store_transient(level="L1", name="Ha", row=r, col=c, value=Ha_block)

                # ---------- PHASE 3: down projection ----------
                # down_proj is accumulation into Y:
                #   Y has access {"tok": 0, "dgroup": 1}
                #   it omits the REDUCE axis "fchunk"
                # so Y is an accumulator.
                #
                # The chunk of Ha participates in the K-reduction.
                # Each chunk contributes partial sums into the same Yacc tile.
                Yacc[row_slice=64*r : 64*(r+1),
                     col_slice=64*c : 64*(c+1)] += matmul_acc(
                        load_transient("Ha", row=r, col=c),
                        W2_chunk[row_slice=64*f0 : 64*f0 + KF,
                                 col_slice=64*c : 64*(c+1)]
                    )

    # After the REDUCE loop finishes, the accumulator is complete.
    writeback_to_dram(
        Y[tok0 : tok0 + tok_tile, :],
        Yacc
    )
```


## 10. What FFN forces onto the backend (not there yet)
1. `matmul_bf16` + `matmul_acc_bf16` AIE kernels + a `GemmTileConstraint` (K-reduction geometry).
2. `REDUCE`-into-`RESIDENT` lowering: accumulate across `fchunk` into persistent `Y`, fed by the
   up-proj producing each chunk (cross-node compute-at). No accumulate-across-tiles analog today.

## 11. Recommended next steps (in order)
1. **Elect-only path**: add a Deeploy entry that runs `parse`+`typeCheck` over a subgraph, populates the
   context, and returns per-node `BoundKernel` descriptors WITHOUT materializing ExecutionBlocks.
2. **Derivation functions**: implement `legality`, `offchip_bytes`, `residency_footprint` over a
   `FusionRegion` so scheme-D / regime-1 / thin-strip can be compared numerically.
3. **First lowering slice — pipeline-fuse an ELEMENTWISE chain** (`Add → LayerNorm → Gelu`, all exist):
   place nodes on distinct cores, mark the 2 interior edges interior, generalize `fifoRegistry` so
   compute→compute edges become core→core ObjectFifos, fire the `argIndexMap=None` skip, relax
   `_checkDataMoverInvariants`. No MLIRComputeCorePass change needed (each core still runs 1 kernel).
   Measure DRAM traffic vs layer-by-layer to confirm interior edges stopped round-tripping.
4. **Then** temporal fusion (kernel chain + L1 scratch, `MLIRComputeCorePass` change), then GEMM
   kernels + accumulator lowering for the FFN.

## 12. Open decisions
- Where the `FusionRegion` lives long-term (Python object vs MLIR dialect) — Python first, freeze later.
- Group formation: manual topology pass (like EngineColoringPass) first; auto-matching later.
- Exact `BoundKernel` field set (freeze it — it's the frontend↔IR interface).
- How to represent "interior / on-chip, no mover" without breaking `_checkDataMoverInvariants`.

## 13. Key file references
- Core IR / engines: `Deeploy/DeeployTypes.py` (`NetworkContext`, `VariableBuffer`, `NodeParser`,
  `NodeTypeChecker`, `NodeBinding`, `ExecutionBlock`, `DataMoverEngine`~2377).
- Tiling: `Deeploy/TilingExtension/{TileConstraint,TilerModel,MemoryConstraints,MemoryConstraintFlows,
  MemoryScheduler,TilingCodegen}.py`; single-node asserts in
  `CodeTransformationPasses/{TilingCodeGeneration.py:250,TilingVariableReplacement.py:116,217}`.
- XDNA2: `Deeploy/Targets/XDNA2/{Deployer,Platform,Tiler,Parsers,Bindings}.py`,
  `CodeTransformationPasses/MLIR*Pass.py`, `TopologyOptimizationPasses/*`, `TileConstraints/*`,
  `Templates/*`; kernels in `TargetLibraries/XDNA2/kernels/*.cc`.
- Branch: `pr/spatial-mapping-beta`.
