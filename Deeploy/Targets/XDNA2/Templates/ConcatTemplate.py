# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""XDNA2 MLIR template for ``Concat``.

Mirror of :class:`XDNA2SplitMemTileTemplate`: Concat is the layout-join
companion to Split, lowered by :class:`MLIRJoinLinkPass` into an
``aie.objectfifo.link`` join pattern. No compute kernel; ``emit()`` is a
no-op.

Input keys are ``data_in_1``..``data_in_N`` (matching ConcatParser);
output key is ``data_out``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from Deeploy.MLIRDataTypes import MLIRNodeTemplate

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import OperatorRepresentation


class XDNA2ConcatMemTileTemplate(MLIRNodeTemplate):
    """Mem-tile-hosted Concat — pure data movement, no compute kernel."""

    KERNEL_FN = ""
    KERNEL_OBJ = ""
    # INPUT_KEYS variadic per-instance; class default empty.
    INPUT_KEYS = []
    OUTPUT_KEYS = ['data_out']

    def __init__(self) -> None:
        super().__init__()

    def emit(self, operatorRepresentation: OperatorRepresentation, **kwargs) -> None:
        return  # no kernel call; link op is emitted by MLIRJoinLinkPass


referenceTemplate = XDNA2ConcatMemTileTemplate()
