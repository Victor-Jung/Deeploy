# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import bfloat16_t
from Deeploy.DeeployTypes import NodeBinding
from Deeploy.MLIRDataTypes import MLIRCodeTransformation
from Deeploy.Targets.XDNA2.CodeTransformationPasses.MLIRComputeCorePass import MLIRComputeCorePass
from Deeploy.Targets.XDNA2.CodeTransformationPasses.MLIRDistributeLinkPass import MLIRDistributeLinkPass
from Deeploy.Targets.XDNA2.CodeTransformationPasses.MLIRJoinLinkPass import MLIRJoinLinkPass
from Deeploy.Targets.XDNA2.CodeTransformationPasses.MLIRMemTileRuntimeSequencePass import \
    MLIRMemTileRuntimeSequencePass
from Deeploy.Targets.XDNA2.CodeTransformationPasses.MLIRGemvComputeCorePass import MLIRGemvComputeCorePass
from Deeploy.Targets.XDNA2.CodeTransformationPasses.MLIRGemvObjectFifoPass import MLIRGemvObjectFifoPass
from Deeploy.Targets.XDNA2.CodeTransformationPasses.MLIRGemvRuntimeSequencePass import MLIRGemvRuntimeSequencePass
from Deeploy.Targets.XDNA2.CodeTransformationPasses.MLIRObjectFifoPass import MLIRObjectFifoPass
from Deeploy.Targets.XDNA2.CodeTransformationPasses.MLIRRuntimeSequencePass import MLIRRuntimeSequencePass
from Deeploy.Targets.XDNA2.Templates import AddTemplate, ConcatTemplate, GeluTemplate, GemvTemplate, LayerNormTemplate, \
    MulTemplate, ReluTemplate, SiLUTemplate, SplitTemplate, TanhTemplate
from Deeploy.Targets.XDNA2.TypeCheckers import XDNA2AddChecker, XDNA2ConcatChecker, XDNA2GeluChecker, \
    XDNA2GemvChecker, XDNA2LayerNormChecker, XDNA2MulChecker, XDNA2ReluChecker, XDNA2SiLUChecker, XDNA2SplitChecker, \
    XDNA2TanhChecker

XDNA2Transformer = MLIRCodeTransformation(
    devicePasses = [
        MLIRObjectFifoPass(),
        MLIRComputeCorePass(),
    ],
    runtimeSequencePasses = [
        MLIRRuntimeSequencePass(),
    ],
)

XDNA2GemvTransformer = MLIRCodeTransformation(
    devicePasses = [
        MLIRGemvObjectFifoPass(),
        MLIRGemvComputeCorePass(),
    ],
    runtimeSequencePasses = [
        MLIRGemvRuntimeSequencePass(),
    ],
)

XDNA2SplitMemTileTransformer = MLIRCodeTransformation(
    devicePasses = [MLIRDistributeLinkPass()],
    runtimeSequencePasses = [MLIRMemTileRuntimeSequencePass()],
)

XDNA2ConcatMemTileTransformer = MLIRCodeTransformation(
    devicePasses = [MLIRJoinLinkPass()],
    runtimeSequencePasses = [MLIRMemTileRuntimeSequencePass()],
)

XDNA2AddBindings = [
    NodeBinding(
        XDNA2AddChecker([PointerClass(bfloat16_t), PointerClass(bfloat16_t)], [PointerClass(bfloat16_t)]),
        AddTemplate.referenceTemplate,
        XDNA2Transformer,
    )
]

XDNA2MulBindings = [
    NodeBinding(
        XDNA2MulChecker([PointerClass(bfloat16_t), PointerClass(bfloat16_t)], [PointerClass(bfloat16_t)]),
        MulTemplate.referenceTemplate,
        XDNA2Transformer,
    )
]

XDNA2SiLUBindings = [
    NodeBinding(
        XDNA2SiLUChecker([PointerClass(bfloat16_t)], [PointerClass(bfloat16_t)]),
        SiLUTemplate.referenceTemplate,
        XDNA2Transformer,
    )
]

XDNA2GeluBindings = [
    NodeBinding(
        XDNA2GeluChecker([PointerClass(bfloat16_t)], [PointerClass(bfloat16_t)]),
        GeluTemplate.referenceTemplate,
        XDNA2Transformer,
    )
]

XDNA2ReluBindings = [
    NodeBinding(
        XDNA2ReluChecker([PointerClass(bfloat16_t)], [PointerClass(bfloat16_t)]),
        ReluTemplate.referenceTemplate,
        XDNA2Transformer,
    )
]

XDNA2TanhBindings = [
    NodeBinding(
        XDNA2TanhChecker([PointerClass(bfloat16_t)], [PointerClass(bfloat16_t)]),
        TanhTemplate.referenceTemplate,
        XDNA2Transformer,
    )
]

XDNA2SplitMemTileBindings = [
    NodeBinding(
        XDNA2SplitChecker([PointerClass(bfloat16_t)], [PointerClass(bfloat16_t)]),
        SplitTemplate.referenceTemplate,
        XDNA2SplitMemTileTransformer,
    )
]

XDNA2ConcatMemTileBindings = [
    NodeBinding(
        XDNA2ConcatChecker([PointerClass(bfloat16_t)], [PointerClass(bfloat16_t)]),
        ConcatTemplate.referenceTemplate,
        XDNA2ConcatMemTileTransformer,
    )
]

XDNA2LayerNormBindings = [
    NodeBinding(
        XDNA2LayerNormChecker(
            [PointerClass(bfloat16_t), PointerClass(bfloat16_t),
             PointerClass(bfloat16_t)], [PointerClass(bfloat16_t)]),
        LayerNormTemplate.referenceTemplate,
        XDNA2Transformer,
    )
]

XDNA2GemvBindings = [
    NodeBinding(
        XDNA2GemvChecker([PointerClass(bfloat16_t), PointerClass(bfloat16_t)], [PointerClass(bfloat16_t)]),
        GemvTemplate.referenceTemplate,
        XDNA2GemvTransformer,
    )
]
