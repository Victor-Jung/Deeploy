# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Sequence, Type

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.CommonExtensions.TypeCheckers.SignPropTypeChecker import SignPropTypeChecker
from Deeploy.DeeployTypes import OperatorRepresentation, VariableBuffer


class XDNA2AddChecker(SignPropTypeChecker):
    """Type checker for BF16 elementwise Add on XDNA2.

    Both inputs and the output are bfloat16_t pointers.
    """

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> Optional[List[int]]:
        # Float types do not have a meaningful nLevels — return 1 as a neutral value.
        return [1]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        # BF16 is a signed floating-point type.
        return [True]


class XDNA2MulChecker(SignPropTypeChecker):
    """Type checker for BF16 elementwise Mul on XDNA2.

    Both inputs and the output are bfloat16_t pointers.
    """

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> Optional[List[int]]:
        return [1]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        return [True]


class XDNA2SiLUChecker(SignPropTypeChecker):
    """Type checker for BF16 SiLU on XDNA2.

    Single input and output, both bfloat16_t pointers.
    """

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> Optional[List[int]]:
        return [1]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        return [True]


class XDNA2ReluChecker(SignPropTypeChecker):
    """Type checker for BF16 ReLU on XDNA2.

    Single input and output, both bfloat16_t pointers.
    """

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> Optional[List[int]]:
        return [1]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        return [True]


class XDNA2GeluChecker(SignPropTypeChecker):
    """Type checker for BF16 GELU on XDNA2.

    Single input and output, both bfloat16_t pointers.
    """

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> Optional[List[int]]:
        return [1]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        return [True]


class XDNA2TanhChecker(SignPropTypeChecker):
    """Type checker for BF16 Tanh on XDNA2.

    Single input and output, both bfloat16_t pointers.
    """

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> Optional[List[int]]:
        return [1]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        return [True]


class XDNA2SplitChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def typeInferOutput(self, ctxt, node, operatorRepresentation):
        outputType = self.output_types[0]
        for out in node.outputs:
            ctxt.annotateType(out.name, outputType)
        return ctxt

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> Optional[List[int]]:
        n = len([k for k in operatorRepresentation if k.startswith("data_out_")])
        return [1] * n

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        n = len([k for k in operatorRepresentation if k.startswith("data_out_")])
        return [True] * n


class XDNA2ConcatChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> Optional[List[int]]:
        return [1]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        return [True]


class XDNA2LayerNormChecker(SignPropTypeChecker):
    """Type checker for BF16 LayerNorm on XDNA2.

    Three inputs (data, weight, bias) and one output, all bfloat16_t pointers.
    The kernel hardcodes gamma=1, beta=0 but the ONNX node still carries
    scale/bias as graph-level inputs for proper type inference.
    """

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> Optional[List[int]]:
        return [1]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        return [True]
