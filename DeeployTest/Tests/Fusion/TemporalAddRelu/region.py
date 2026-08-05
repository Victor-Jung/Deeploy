# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.FusionExtension.IR import BoundKernel, EdgeSchedule, FusionRegion, Loop, LoopKind, Residency, TensorRef

region = FusionRegion(
    kernels = [BoundKernel(
        node = 'add0',
        op = 'Add',
        kernel_symbol = 'eltwise_add_bf16_vector',
        operator_repr = {'size': 65536},
        inputs = [TensorRef('A', (256, 256), 'bf16'), TensorRef('B', (256, 256), 'bf16')],
        outputs = [TensorRef('s', (256, 256), 'bf16')],
        kernel_obj = 'add.o',
    ), BoundKernel(
        node = 'relu0',
        op = 'Relu',
        kernel_symbol = 'relu_bf16',
        operator_repr = {'size': 65536},
        inputs = [TensorRef('s', (256, 256), 'bf16')],
        outputs = [TensorRef('Y', (256, 256), 'bf16')],
        kernel_obj = 'relu.o',
    )],
    loops = [Loop('elem', '65536', '2048', LoopKind.MARCH)],
    placement = {'add0': 'elem', 'relu0': 'elem'},
    edges = {
        'A': EdgeSchedule('A', {'elem': 0}, Residency.STREAM, 'DRAM'),
        'B': EdgeSchedule('B', {'elem': 0}, Residency.STREAM, 'DRAM'),
        's': EdgeSchedule('s', {'elem': 0}, Residency.TRANSIENT, 'L1'),
        'Y': EdgeSchedule('Y', {'elem': 0}, Residency.STREAM, 'DRAM'),
    },
    maps = {},
    params = {},
)
