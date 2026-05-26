# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Tile constraint for memtile-engine layout transforms (Split / Concat).

These nodes are not actually tiled, they describe a single bulk transfer
through a mem tile, lowered as one ``aie.objectfifo.link`` op. We still
have to register every input / output tensor's dimensions with the tiler
model so downstream ``addTensorNumOfEltToModel`` calls (used by the
solver to compute total transfer sizes) don't fault.

The constraint registers the tensors and pins each dimension to its full
shape (no tiling within these nodes).
"""

from typing import Dict

from Deeploy.DeeployTypes import NetworkContext
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel


def _portKeysFromOpRepr(parseDict: Dict) -> list:
    """Collect every operatorRepresentation key that names a tensor.

    Recognises Deeploy conventions used by ``SplitParser`` /
    ``ConcatParser``:

    * ``data_in`` (Split)
    * ``data_in_{i}`` for ``i >= 1`` (Concat)
    * ``data_out`` (Concat)
    * ``data_out_{i}`` for ``i >= 0`` (Split)

    Anything else (``axis``, ``size``, ...) is skipped — those aren't
    tensor names.
    """
    keys = []
    for k in parseDict:
        if k == "data_in" or k == "data_out":
            keys.append(k)
        elif k.startswith("data_in_") and k[len("data_in_"):].isdigit():
            keys.append(k)
        elif k.startswith("data_out_") and k[len("data_out_"):].isdigit():
            keys.append(k)
    return keys


class XDNA2MemTileLayoutTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        for portKey in _portKeysFromOpRepr(parseDict):
            tensorName = parseDict[portKey]
            tilerModel.addTensorDimToModel(ctxt, tensorName)
            buf = ctxt.lookup(tensorName)
            shape = buf.shape if not isinstance(buf.shape, int) else (buf.shape,)
            for dim, fullDim in enumerate(shape):
                dimVar = tilerModel.getTensorDimVar(tensorName = tensorName, dimIdx = dim)
                tilerModel.addConstraint(dimVar == int(fullDim))
        return tilerModel
