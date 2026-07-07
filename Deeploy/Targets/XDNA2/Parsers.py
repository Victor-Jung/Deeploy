# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

# XDNA2 reuses the Generic AddParser (see Platform.py).
# Add any XDNA2-specific parsers here as the platform grows.

from typing import Tuple

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NetworkContext, NodeParser
from Deeploy.Targets.Generic.Parsers import MatMulParser


class XDNA2GemvParser(MatMulParser):
    """Matrix-vector (GEMV) parser for XDNA2.

    Reuses the Generic MatMulParser but only matches the vector, bias-free
    case: exactly two inputs (matrix A=[M,N], vector x=[N,1]) and O == 1.
    Anything wider (a true matrix-matrix product) or with a bias is left to
    other mappers.
    """

    def parseNode(self, node: gs.Node) -> bool:
        return super().parseNode(node) and len(node.inputs) == 2

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)
        if ret:
            # GEMV marker: the second operand is a column vector (O == 1).
            ret = int(self.operatorRepresentation.get('O', 0)) == 1
        return newCtxt, ret


class XDNA2LayerNormParser(NodeParser):
    """Simplified LayerNorm parser for XDNA2.

    The XDNA2 kernel hardcodes gamma=1.0 and beta=0.0, so only the
    data input and output are registered.  Scale and bias tensors
    are validated to exist in the ONNX graph but are **not** added
    to ``operatorRepresentation``, keeping the tiling system simple
    (UnaryTileConstraint: 1 input, 1 output).
    """

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        return all([
            'epsilon' in node.attrs,
            len(node.inputs) == 3,
            len(node.outputs) >= 1,
        ])

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)

        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['size'] = int(np.prod(data_in.shape))
        self.operatorRepresentation['lastDimLength'] = int(data_in.shape[-1])
        self.operatorRepresentation['epsilon'] = node.attrs['epsilon']

        return ctxt, True
