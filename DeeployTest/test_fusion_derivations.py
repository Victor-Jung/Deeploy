# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""T0 tests: derivation oracle over hand-authored FusionRegions (no hardware).

Encodes the three canonical schemes from FUSION_IR_NOTES.md and asserts the
derivation functions reproduce the analytical numbers. Runs anywhere (pure
Python), so it gates CI without an NPU.
"""

from Deeploy.FusionExtension.IR import BoundKernel, EdgeSchedule, FusionRegion, HardwareModel, Loop, LoopKind, \
    Residency, SpatialMap, TensorRef
from Deeploy.FusionExtension.Derivations import accumulator_bytes, is_accumulator, legality, offchip_elements


def _tref(name, shape, dtype = "bf16"):
    return TensorRef(name, shape, dtype)


# ---------------------------------------------------------------------------
# Case 1: elementwise Add -> Gelu.  Y = gelu(A + B), interior edge `s`.
# ---------------------------------------------------------------------------


def _elementwise(interior_residency):
    add = BoundKernel("add0", "Add", "add_bf16", {}, [_tref("A", ("T", "D")), _tref("B", ("T", "D"))],
                      [_tref("s", ("T", "D"))])
    gelu = BoundKernel("gelu0", "Gelu", "gelu_bf16", {}, [_tref("s", ("T", "D"))], [_tref("Y", ("T", "D"))])
    return FusionRegion(
        kernels = [add, gelu],
        loops = [Loop("tok", "T", "64", LoopKind.MARCH)],
        placement = {"add0": "tok", "gelu0": "tok"},
        edges = {
            "A": EdgeSchedule("A", {"tok": 0}, Residency.STREAM, "DRAM"),
            "B": EdgeSchedule("B", {"tok": 0}, Residency.STREAM, "DRAM"),
            "s": EdgeSchedule("s", {"tok": 0}, interior_residency, "L1"),
            "Y": EdgeSchedule("Y", {"tok": 0}, Residency.STREAM, "DRAM"),
        },
        params = {"T": 128, "D": 64},
    )


def test_elementwise_identity_vs_fused_traffic():
    T, D = 128, 64
    identity = _elementwise(Residency.STREAM)      # s round-trips to DRAM
    fused = _elementwise(Residency.TRANSIENT)       # s stays on-chip
    # identity: A(TD) + B(TD) + s(write TD + read TD) + Y(TD) = 5TD
    assert offchip_elements(identity) == 5 * T * D
    # fused: interior s eliminated -> 3TD (A,B,Y)
    assert offchip_elements(fused) == 3 * T * D
    # the win is exactly the eliminated round-trip of s
    assert offchip_elements(identity) - offchip_elements(fused) == 2 * T * D
    assert legality(identity) == [] and legality(fused) == []


# ---------------------------------------------------------------------------
# Case 2: single GEMM Y = A @ B, reduction over K.
# ---------------------------------------------------------------------------


def test_gemm_accumulator_and_legality():
    mm = BoundKernel("mm0", "MatMul", "matmul_bf16", {"M": "M", "K": "K", "N": "N"},
                     [_tref("A", ("M", "K")), _tref("B", ("K", "N"))], [_tref("Y", ("M", "N"))])
    region = FusionRegion(
        kernels = [mm],
        loops = [
            Loop("m", "M", "64", LoopKind.PARALLEL),
            Loop("n", "N", "64", LoopKind.PARALLEL),
            Loop("k", "K", "64", LoopKind.REDUCE),
        ],
        placement = {"mm0": "k"},
        edges = {
            "A": EdgeSchedule("A", {"m": 0, "k": 1}, Residency.STREAM, "DRAM"),
            "B": EdgeSchedule("B", {"k": 0, "n": 1}, Residency.STREAM, "DRAM"),
            "Y": EdgeSchedule("Y", {"m": 0, "n": 1}, Residency.RESIDENT, "L1"),
        },
        params = {"M": 128, "K": 256, "N": 128},
    )
    # Y omits the REDUCE axis k and is written -> accumulator; A/B include k -> not.
    assert is_accumulator(region, region.edges["Y"])
    assert not is_accumulator(region, region.edges["A"])
    assert legality(region) == []

    # Legality must reject a REDUCE axis leaking into an output access.
    bad = FusionRegion(kernels = region.kernels, loops = region.loops, placement = region.placement,
                       edges = {**region.edges, "Y": EdgeSchedule("Y", {"m": 0, "n": 1, "k": 2},
                                                                  Residency.RESIDENT, "L1")},
                       params = region.params)
    assert legality(bad), "a REDUCE axis in an output access must be flagged illegal"


# ---------------------------------------------------------------------------
# Case 3: FFN scheme-D  X --W1--> H --gelu--> Ha --W2--> Y   (F = 4D)
# reproduce off-chip = 2TD + 2DF*(T/64R) and the accumulator-fits corner.
# ---------------------------------------------------------------------------


def _ffn_scheme_d(T, D, R, C):
    F = 4 * D
    KF = 64 * C
    up = BoundKernel("up_proj", "MatMul", "matmul_bf16", {}, [_tref("X", ("T", "D")), _tref("W1", ("D", "F"))],
                     [_tref("H", ("T", "F"))])
    act = BoundKernel("act", "Gelu", "gelu_bf16", {}, [_tref("H", ("T", "F"))], [_tref("Ha", ("T", "F"))])
    down = BoundKernel("down_proj", "MatMul", "matmul_acc_bf16", {}, [_tref("Ha", ("T", "F")), _tref("W2", ("F", "D"))],
                       [_tref("Y", ("T", "D"))])
    return FusionRegion(
        kernels = [up, act, down],
        loops = [
            Loop("tok", "T", "64*R", LoopKind.MARCH),
            Loop("fchunk", "F", "KF", LoopKind.REDUCE),
            Loop("dgroup", "D", "64*C", LoopKind.PARALLEL),
        ],
        placement = {"up_proj": "fchunk", "act": "fchunk", "down_proj": "dgroup"},
        edges = {
            "X": EdgeSchedule("X", {"tok": 0}, Residency.RESIDENT, "MEMTILE"),
            "W1": EdgeSchedule("W1", {"fchunk": 1}, Residency.STREAM, "DRAM"),
            "H": EdgeSchedule("H", {"tok": 0, "fchunk": 1}, Residency.TRANSIENT, "L1"),
            "Ha": EdgeSchedule("Ha", {"tok": 0, "fchunk": 1}, Residency.TRANSIENT, "L1"),
            "W2": EdgeSchedule("W2", {"fchunk": 0, "dgroup": 1}, Residency.STREAM, "DRAM"),
            "Y": EdgeSchedule("Y", {"tok": 0, "dgroup": 1}, Residency.RESIDENT, "MEMTILE"),
        },
        maps = {
            "up_proj": SpatialMap("up_proj", "tok", "fchunk", "H", ("R", "C")),
            "down_proj": SpatialMap("down_proj", "tok", "dgroup", "Y", ("R", "C")),
        },
        params = {"T": T, "D": D, "F": F, "KF": KF, "R": R, "C": C},
    )


def test_ffn_scheme_d_offchip_matches_formula():
    T, D, R, C = 4096, 768, 8, 4  # ViT-B-ish, T chosen so T/(64R)=8 is integral
    F = 4 * D
    region = _ffn_scheme_d(T, D, R, C)
    expected = 2 * T * D + 2 * D * F * (T // (64 * R))
    assert offchip_elements(region) == expected
    # H / Ha never hit DRAM
    assert is_accumulator(region, region.edges["Y"])
    assert not is_accumulator(region, region.edges["X"])   # RESIDENT but read-only -> invariant, not accumulator
    assert legality(region) == []


def test_ffn_accumulator_fits_vit_not_llama():
    # Y accumulator panel = 64R x D x 4 bytes (dgroup is inner to the fchunk reduction -> full D).
    vit = _ffn_scheme_d(T = 4096, D = 768, R = 8, C = 4)
    llama = _ffn_scheme_d(T = 4096, D = 4096, R = 8, C = 4)
    acc_vit = accumulator_bytes(vit, "Y")
    acc_llama = accumulator_bytes(llama, "Y")
    assert acc_vit == 64 * 8 * 768 * 4        # 1.5 MB
    assert acc_llama == 64 * 8 * 4096 * 4     # 8 MB
    budget = 2 * 1024 * 1024                   # ~aggregate on-chip (4 memtiles x 512KB)
    assert acc_vit <= budget < acc_llama       # ViT-B fits, Llama does not (the notes' verdict)


# ---------------------------------------------------------------------------
# HardwareModel injection from a duck-typed platform.
# ---------------------------------------------------------------------------


def test_hardware_model_from_platform():

    class _Eng:

        def __init__(self, col, row = None):
            self.col = col
            self.row = row

    class _Level:

        def __init__(self, size):
            self.size = size

    class _Hier:
        memoryLevels = {"L3": _Level(1 << 30), "L2_c0": _Level(512 * 1024), "L1_c0r2": _Level(64000)}

    class _Plat:
        engines = [_Eng(c, r) for c in range(4) for r in (2, 3)]  # 4 cols x 2 rows
        memoryHierarchy = _Hier()

    hw = HardwareModel.from_platform(_Plat())
    assert hw.pe_cols == 4 and hw.pe_rows == 2
    assert hw.mem_capacity["L2_c0"] == 512 * 1024
