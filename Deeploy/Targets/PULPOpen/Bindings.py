# ----------------------------------------------------------------------
#
# File: PULPBindings.py
#
# Last edited: 10.03.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Authors:
# - Moritz Scherer, ETH Zurich
# - Victor Jung, ETH Zurichs
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
from functools import partial

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.CodeTransformationPasses.Closure import ClosureGeneration, MemoryAwareClosureGeneration
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import ArgumentStructGeneration, \
    MemoryManagementGeneration
from Deeploy.CommonExtensions.DataTypes import IntegerDataTypes, SignedIntegerDataTypes, float32_t, int8_t, int32_t, \
    uint8_t
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding, NodeTemplate
from Deeploy.FutureExtension.Bindings.AutoFutureBinding import AutoFutureBinding
from Deeploy.FutureExtension.CodeTransformationPasses.FutureCodeTransformation import FutureGeneration
from Deeploy.Targets.Generic.Templates import ConcatTemplate, DequantTemplate, FloatGELUTemplate, FloatGemmTemplate, \
    FloatLayernormTemplate, FloatMatMulTemplate, FloatMulTemplate, FloatReduceSumTemplate, FloatReluTemplate, \
    FloatSoftmaxTemplate, GatherTemplate, QuantTemplate, RQSiGELUTemplate, iHardswishTemplate
from Deeploy.Targets.Generic.TypeCheckers import ConcatChecker, ConvChecker, DequantChecker, GatherChecker, \
    GELUChecker, GEMMChecker, HardswishChecker, LayerNormChecker, MatMulChecker, MulChecker, QuantChecker, \
    ReduceMeanChecker, ReluChecker, RQAddChecker, RQHardswishChecker, SGDChecker, SliceChecker, SoftmaxChecker, \
    SoftmaxCrossEntropyLossChecker, TransposeChecker
from Deeploy.Targets.PULPOpen.CodeTransformationPasses.PULPClusterSynch import PULPSynchCoresPass
from Deeploy.Targets.PULPOpen.CodeTransformationPasses.PULPClusterTiling import PULPClusterTiling
from Deeploy.Targets.PULPOpen.CodeTransformationPasses.PULPL3Tiling import PULPL3Tiling
from Deeploy.Targets.PULPOpen.CodeTransformationPasses.PULPProfileUntiled import PULPProfileUntiled
from Deeploy.Targets.PULPOpen.DataTypes import PULPDMAFuture
from Deeploy.Targets.PULPOpen.Templates import ConvTemplate, FloatConvTemplate, FloatMaxPoolTemplate, \
    FloatSoftmaxTemplate, GEMMTemplate, MatrixVectorTemplate, MaxPool2DTemplate, MulTemplate, ReduceMeanTemplate, \
    RequantShiftTemplate, RQAddTemplate, RQSiHardswishTemplate, SGDTemplate, SliceTemplate, \
    SoftmaxCrossEntropyLossTemplate, TallGEMMTemplate, TransposeTemplate, UniformRequantShiftTemplate, \
    iRMSNormTemplate, iSoftmaxTemplate
from Deeploy.Targets.PULPOpen.TypeCheckers import PULPConvChecker, PULPLinearChecker, PULPMaxPoolChecker, \
    PULPRequantShiftChecker
from Deeploy.TilingExtension.CodeTransformationPasses.TilingVariableReplacement import TilingVariableReplacement

_clusterEntryClosureCallTemplate = NodeTemplate("""
// ${closureName} CLOSURE CALL
static struct pi_cluster_task cluster_task;

pi_cluster_task(&cluster_task, ${closureName}, &${closureStructArgName});
cluster_task.stack_size = 5000;
cluster_task.slave_stack_size = 3800;
pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
//pi_cluster_close(&cluster_dev);
""")

_clusterForkClosureCallTemplate = NodeTemplate("""
pi_cl_team_fork(NUM_CORES, (void*)${closureName}, &${closureStructArgName});
""")

FunctionCallClosure = partial(ClosureGeneration, closureSuffix = "_closure")
ClusterClosure = partial(ClosureGeneration,
                         closureSuffix = "_cluster_entry",
                         closureCallTemplate = _clusterEntryClosureCallTemplate)
ForkClosure = partial(ClosureGeneration,
                      closureSuffix = "_cluster_fork",
                      closureCallTemplate = _clusterForkClosureCallTemplate)

TilingCallClosure = partial(ClosureGeneration, closureSuffix = "_tiling_closure")
FunctionCallClosure = partial(ClosureGeneration, closureSuffix = "_closure")
ForkClosure = partial(ClosureGeneration,
                      closureSuffix = "_cluster_fork",
                      closureCallTemplate = _clusterForkClosureCallTemplate)

MemoryAwareClusterClosure = partial(MemoryAwareClosureGeneration,
                                    closureSuffix = "_cluster_entry",
                                    closureCallTemplate = _clusterEntryClosureCallTemplate,
                                    startRegion = "L2",
                                    endRegion = "L1")
MemoryAwareFunctionCallClosure = partial(MemoryAwareClosureGeneration,
                                         closureSuffix = "_closure",
                                         startRegion = "L2",
                                         endRegion = "L1")

L3MemoryAwareFunctionCallClosure = partial(ClosureGeneration, closureSuffix = "_closure_L3")

MemoryAwareForkTransformer = CodeTransformation([
    ArgumentStructGeneration(),
    ForkClosure(generateStruct = False),
    FutureGeneration(),
    ArgumentStructGeneration(),
    MemoryManagementGeneration("L1"),
    FunctionCallClosure(writeback = True),
    MemoryManagementGeneration("L2"),
    MemoryManagementGeneration()
])

ForkTransformer = CodeTransformation([
    TilingVariableReplacement("L1"),
    TilingCallClosure(writeback = False),
    PULPSynchCoresPass(),
    ForkClosure(writeback = False, generateStruct = True),
    PULPClusterTiling("L1"),
    ArgumentStructGeneration(),
    MemoryManagementGeneration("L1"),
    MemoryAwareFunctionCallClosure(writeback = False, generateStruct = True),
    TilingVariableReplacement("L2"),
    PULPL3Tiling("L2"),
    PULPProfileUntiled(),
    ArgumentStructGeneration(),
    L3MemoryAwareFunctionCallClosure(writeback = False),
    MemoryManagementGeneration("L3.*"),
    MemoryManagementGeneration("L2"),
    MemoryManagementGeneration(),
])

ClusterTransformer = CodeTransformation([
    TilingVariableReplacement("L1"),
    TilingCallClosure(writeback = False, generateStruct = True),
    PULPClusterTiling("L1"),
    ArgumentStructGeneration(),
    MemoryManagementGeneration("L1"),
    MemoryAwareFunctionCallClosure(writeback = False, generateStruct = True),
    TilingVariableReplacement("L2"),
    PULPL3Tiling("L2"),
    PULPProfileUntiled(),
    ArgumentStructGeneration(),
    L3MemoryAwareFunctionCallClosure(writeback = False),
    MemoryManagementGeneration("L2"),
    MemoryManagementGeneration("L3.*"),
    MemoryManagementGeneration(),
])

SimpleTransformer = CodeTransformation([
    MemoryManagementGeneration("L2"),
    MemoryManagementGeneration("L3.*"),
    MemoryManagementGeneration(),
])

PULPDMASliceBindings = [
    AutoFutureBinding(
        SliceChecker([
            PointerClass(type),
            PointerClass(uint8_t),
            PointerClass(uint8_t),
            PointerClass(uint8_t),
            PointerClass(uint8_t)
        ], [PULPDMAFuture(underlyingType = type)]), SliceTemplate.referenceTemplate, MemoryAwareForkTransformer)
    for type in IntegerDataTypes
]

PULPRQAddBindings = [
    NodeBinding(RQAddChecker([PointerClass(_type), PointerClass(_type2)], [PointerClass(_type3)]),
                RQAddTemplate.referenceTemplate, ForkTransformer)
    for _type in [int8_t, uint8_t]
    for _type2 in [int8_t, uint8_t]
    for _type3 in [int8_t, uint8_t]
]

PULPRQSConv2DBindings = [
    NodeBinding(
        PULPConvChecker([
            PointerClass(type1),
            PointerClass(int8_t),
            PointerClass(int32_t),
            PointerClass(int32_t),
            PointerClass(int32_t)
        ], [PointerClass(type2)]), ConvTemplate.PULPConv2D_8_Template, ForkTransformer)
    for type1, type2 in zip([int8_t, int8_t, uint8_t, uint8_t], [int8_t, uint8_t, int8_t, uint8_t])
]

PULPRQSDWConv2DBindings = [
    NodeBinding(
        PULPConvChecker([
            PointerClass(type1),
            PointerClass(int8_t),
            PointerClass(int32_t),
            PointerClass(int32_t),
            PointerClass(int32_t)
        ], [PointerClass(type2)]), ConvTemplate.PULPDWConv2D_8_Template, ForkTransformer)
    for type1, type2 in zip([int8_t, int8_t, uint8_t, uint8_t], [int8_t, uint8_t, int8_t, uint8_t])
]

PULPRQSGEMM_8_Binding = [
    NodeBinding(
        PULPLinearChecker([PointerClass(type1),
                           PointerClass(int8_t),
                           PointerClass(int32_t),
                           PointerClass(int32_t)], [PointerClass(type2)]), GEMMTemplate.PULPGEMM_8_Template,
        ForkTransformer) for type1, type2 in zip([int8_t, uint8_t, int8_t, uint8_t], [int8_t, uint8_t, uint8_t, int8_t])
]

PULPFloatGEMMBindings = [
    NodeBinding(
        GEMMChecker([PointerClass(float32_t), PointerClass(float32_t),
                     PointerClass(float32_t)], [PointerClass(float32_t)]), FloatGemmTemplate.referenceTemplate,
        ForkTransformer)
]

PULPFloatConv2DBindings = [
    NodeBinding(
        ConvChecker([PointerClass(float32_t), PointerClass(float32_t),
                     PointerClass(float32_t)], [PointerClass(float32_t)]), FloatConvTemplate.reference2DTemplate,
        ForkTransformer)
]

PULPRQSMatrixVecBindings = [
    NodeBinding(
        PULPLinearChecker([PointerClass(type1),
                           PointerClass(int8_t),
                           PointerClass(int32_t),
                           PointerClass(int32_t)], [PointerClass(type2)]), MatrixVectorTemplate.referenceTemplate,
        ForkTransformer) for type1, type2 in zip([int8_t], [int8_t])
]

PULPRQSTallGEMMBindings = [
    NodeBinding(
        PULPLinearChecker([PointerClass(type1),
                           PointerClass(int8_t),
                           PointerClass(int32_t),
                           PointerClass(int32_t)], [PointerClass(type2)]), TallGEMMTemplate.referenceTemplate,
        ForkTransformer) for type1, type2 in zip([int8_t], [int8_t])
]

PULPRQSGEMMBindings = PULPRQSGEMM_8_Binding

PULPMaxPool2DBindings = [
    NodeBinding(PULPMaxPoolChecker([PointerClass(type)], [PointerClass(type)]),
                MaxPool2DTemplate.PULPMaxPool2D_8_Template, ForkTransformer) for type in [int8_t, uint8_t]
] + [
    NodeBinding(PULPMaxPoolChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatMaxPoolTemplate.referenceTemplate, ForkTransformer)
]

PULPConv1DBinding = NodeBinding(
    PULPConvChecker(
        [PointerClass(int8_t), PointerClass(int8_t),
         PointerClass(int32_t),
         PointerClass(int32_t)], [PointerClass(int8_t)]), ConvTemplate.PULPConv1D_8_Template, ForkTransformer)

PULPDWConv1DBinding = NodeBinding(
    PULPConvChecker(
        [PointerClass(int8_t), PointerClass(int8_t),
         PointerClass(int32_t),
         PointerClass(int32_t)], [PointerClass(int8_t)]), ConvTemplate.PULPDWConv1D_8_Template, ForkTransformer)

PULPMatMulBindings = [
    NodeBinding(MatMulChecker([PointerClass(int8_t), PointerClass(int8_t)], [PointerClass(int32_t)]),
                GEMMTemplate.PULPMM_8_Template, ClusterTransformer)
] + [
    NodeBinding(MatMulChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatMatMulTemplate.referenceTemplate, ClusterTransformer)
]

PULPReduceMeanBindings = [
    NodeBinding(ReduceMeanChecker([PointerClass(type)], [PointerClass(type)]), ReduceMeanTemplate.referenceTemplate,
                ClusterTransformer) for type in IntegerDataTypes
]

PULPReduceSumBindings = [
    NodeBinding(ReduceMeanChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatReduceSumTemplate.referenceTemplate, ClusterTransformer)
]

PULPUniformRQSBindings = [
    NodeBinding(
        PULPRequantShiftChecker([PointerClass(type), PointerClass(int32_t),
                                 PointerClass(int32_t)], [PointerClass(int8_t)]),
        UniformRequantShiftTemplate.referenceTemplate, ForkTransformer) for type in IntegerDataTypes
]

PULPRQSBindings = [
    NodeBinding(
        PULPRequantShiftChecker([PointerClass(type), PointerClass(int32_t),
                                 PointerClass(int32_t)], [PointerClass(int8_t)]),
        RequantShiftTemplate.referenceTemplate, ForkTransformer) for type in IntegerDataTypes
] + [
    NodeBinding(
        PULPRequantShiftChecker([PointerClass(type), PointerClass(int32_t),
                                 PointerClass(int32_t)], [PointerClass(uint8_t)]),
        RequantShiftTemplate.referenceTemplate, ForkTransformer) for type in IntegerDataTypes
]

PULPSoftmaxBindings = [
    NodeBinding(SoftmaxChecker([PointerClass(_type)], [PointerClass(uint8_t)]), iSoftmaxTemplate.referenceTemplate,
                ForkTransformer) for _type in [int8_t, uint8_t]
] + [
    NodeBinding(SoftmaxChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatSoftmaxTemplate.referenceTemplate, ForkTransformer)
]

PULPSoftmaxGradBindings = [
    NodeBinding(SoftmaxChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatSoftmaxTemplate.referenceGradientTemplate, ForkTransformer)
]

PULPSoftmaxCrossEntropyLossBindings = [
    NodeBinding(
        SoftmaxCrossEntropyLossChecker([PointerClass(float32_t), PointerClass(type)], [PointerClass(float32_t)]),
        SoftmaxCrossEntropyLossTemplate.referenceTemplate, ForkTransformer) for type in IntegerDataTypes
]

PULPSoftmaxCrossEntropyLossGradBindings = [
    NodeBinding(
        SoftmaxCrossEntropyLossChecker([PointerClass(float32_t), PointerClass(type)], [PointerClass(float32_t)]),
        SoftmaxCrossEntropyLossTemplate.referenceGradientTemplate, ForkTransformer) for type in IntegerDataTypes
]

PULPSGDBindings = [
    NodeBinding(SGDChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                SGDTemplate.referenceTemplate, ForkTransformer)
]

PULPTransposeBindings = [
    NodeBinding(TransposeChecker([PointerClass(type)], [PointerClass(type)]), TransposeTemplate.referenceTemplate,
                ForkTransformer) for type in IntegerDataTypes
] + [
    NodeBinding(TransposeChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                TransposeTemplate.referenceTemplate, ForkTransformer)
]

PULPConcatBindings = [
    NodeBinding(ConcatChecker([PointerClass(type), PointerClass(type)], [PointerClass(type)]),
                ConcatTemplate.referenceTemplate, ClusterTransformer) for type in IntegerDataTypes
]

PULPiRMSNormBindings = [
    NodeBinding(LayerNormChecker([PointerClass(int8_t), PointerClass(int32_t)], [PointerClass(int8_t)]),
                iRMSNormTemplate.referenceTemplate, ForkTransformer)
]

PULPiHardswishBindings = [
    NodeBinding(HardswishChecker([PointerClass(int8_t)], [PointerClass(int32_t)]), iHardswishTemplate.referenceTemplate,
                ClusterTransformer)
]
PULPRQSiHardswishBindings = [
    NodeBinding(
        RQHardswishChecker([PointerClass(int8_t),
                            PointerClass(int32_t),
                            PointerClass(int32_t),
                            PointerClass(int32_t)], [PointerClass(int8_t)]), RQSiHardswishTemplate.referenceTemplate,
        ForkTransformer)
]

PULPiRQSGELUBindings = [
    NodeBinding(
        GELUChecker([PointerClass(int8_t),
                     PointerClass(int32_t),
                     PointerClass(int32_t),
                     PointerClass(int32_t)], [PointerClass(int8_t)]), RQSiGELUTemplate.referenceTemplate,
        ClusterTransformer)
]

PULPMulBindings = [
    NodeBinding(MulChecker([PointerClass(typeA), PointerClass(typeB)], [PointerClass(int32_t)]),
                MulTemplate.referenceTemplate, ForkTransformer)
    for typeA, typeB in itertools.product(SignedIntegerDataTypes, SignedIntegerDataTypes)
] + [
    NodeBinding(MulChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatMulTemplate.referenceTemplate, ForkTransformer)
]

PULPReluBinding = NodeBinding(ReluChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                              FloatReluTemplate.referenceTemplate, ForkTransformer)

PULPLayernormBinding = NodeBinding(
    LayerNormChecker(
        [PointerClass(float32_t), PointerClass(float32_t),
         PointerClass(float32_t)], [PointerClass(float32_t)]), FloatLayernormTemplate.referenceTemplate,
    ForkTransformer)

PULPFloatGELUBinding = NodeBinding(
    GELUChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
    FloatGELUTemplate.referenceTemplate, ForkTransformer)

PULPGatherBindings = [
    NodeBinding(GatherChecker([PointerClass(float32_t), PointerClass(type)], [PointerClass(float32_t)]),
                GatherTemplate.referenceTemplate, ForkTransformer) for type in IntegerDataTypes
]

BasicQuantBindings = [
    NodeBinding(QuantChecker([PointerClass(float32_t)], [PointerClass(int8_t)]), QuantTemplate.referenceTemplate,
                ForkTransformer),
]

BasicDequantBindings = [
    NodeBinding(DequantChecker([PointerClass(int8_t)], [PointerClass(float32_t)]), DequantTemplate.referenceTemplate,
                ForkTransformer),
] + [
    NodeBinding(DequantChecker([PointerClass(int32_t)], [PointerClass(float32_t)]), DequantTemplate.referenceTemplate,
                ForkTransformer),
]
