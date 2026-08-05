# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.FusionExtension.IR import BoundKernel, EdgeSchedule, FusionRegion, Loop, LoopKind, Residency, TensorRef

region = FusionRegion(
    kernels = [BoundKernel(
        node = 'relu0',
        op = 'Relu',
        kernel_symbol = 'relu_bf16',
        operator_repr = {'size': 65536},
        inputs = [TensorRef('input_0', (256, 256), 'bf16')],
        outputs = [TensorRef('output_0', (256, 256), 'bf16')],
        kernel_obj = 'relu.o',
    )],
    loops = [Loop('elem', '65536', '4096', LoopKind.MARCH)],
    placement = {'relu0': 'elem'},
    edges = {'input_0': EdgeSchedule('input_0', {'elem': 0}, Residency.STREAM, 'DRAM'), 'output_0': EdgeSchedule('output_0', {'elem': 0}, Residency.STREAM, 'DRAM')},
    maps = {},
    params = {},
)
