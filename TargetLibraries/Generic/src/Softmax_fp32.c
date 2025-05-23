/* =====================================================================
 * Title:        Softmax_fp32.c
 * Description:
 *
 * $Date:        23.01.2025
 *
 * ===================================================================== */
/*
 * Copyright (C) 2022 ETH Zurich and University of Bologna.
 *
 * - Run Wang, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "DeeployBasicMath.h"
#include <math.h>

void Softmax_fp32_fp32(float32_t *input, float32_t *output, int32_t size,
                       int32_t last_dim_length) {

  int32_t batch_size = size / last_dim_length;

  for (int b = 0; b < batch_size; b++) {
    float32_t max_val = -inf;
    float sum = 0.0f;

    for (int i = 0; i < last_dim_length; i++) {
      if (input[b * last_dim_length + i] > max_val) {
        max_val = input[b * last_dim_length + i];
      }
    }

    for (int i = 0; i < last_dim_length; i++) {
      output[b * last_dim_length + i] =
          expf(input[b * last_dim_length + i] - max_val);
      sum += output[b * last_dim_length + i];
    }

    for (int i = 0; i < last_dim_length; i++) {
      output[b * last_dim_length + i] /= sum;
    }
  }
}

void SoftmaxGrad_fp32_fp32_fp32(float32_t *upstream_grad,
                                float32_t *softmax_output,
                                float32_t *softmax_gradient, int32_t size,
                                int32_t last_dim_length) {

  int32_t batch_size = size / last_dim_length;

  for (int b = 0; b < batch_size; b++) {

    float32_t weighted_sum = 0.0f;

    for (int i = 0; i < last_dim_length; i++) {
      int idx = b * last_dim_length + i;
      weighted_sum += upstream_grad[idx] * softmax_output[idx];
    }

    for (int i = 0; i < last_dim_length; i++) {
      int idx = b * last_dim_length + i;
      softmax_gradient[idx] =
          softmax_output[idx] * (upstream_grad[idx] - weighted_sum);
    }
  }
}