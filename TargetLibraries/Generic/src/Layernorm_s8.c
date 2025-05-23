/* =====================================================================
 * Title:        Layernorm_s8.c
 * Description:
 *
 * $Date:        19.12.2022
 *
 * ===================================================================== */
/*
 * Copyright (C) 2022 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Moritz Scherer, ETH Zurich
 * - Philip Wiese, ETH Zurich
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

// Taken from PULP-DSP - Moritz Scherer is original author
void _plp_sqrt_q32(const int32_t *__restrict__ pSrc, const uint32_t fracBits,
                   int32_t *__restrict__ pRes) {

  int32_t number = *pSrc;
  int32_t root = 0;

  int32_t start = 0;
  int32_t end = 46342; // smallest integer that is larger than sqrt(0x7FFFFFFF)
  int32_t mid;

  if (number > 0) {

    while (start <= end) {

      mid = (start + end) >> 1;

      if (((mid * mid) >> fracBits) == number) {
        root = mid;
        break;
      }

      if (((mid * mid) >> fracBits) < number) {
        start = mid + 1;
        root = mid;
      } else {
        end = mid - 1;
      }
    }

    *pRes = root;

  } else {
    *pRes = 0;
  }
}

void Layernorm_s8_s8(int8_t *data_in, int8_t *data_out, int32_t *weight,
                     int32_t *bias, int32_t input_offset, int32_t size,
                     int32_t lastDimLength, int32_t log2D) {

  int32_t mean;
  int32_t sum;
  int32_t std;
  int16_t temp;

  for (int lastDimStart = 0; lastDimStart < size;
       lastDimStart += lastDimLength) {
    sum = 0;
    mean = 0;
    for (int j = 0; j < lastDimLength; j++) {
      mean += data_in[lastDimStart + j] + input_offset;
    }
    mean = mean / lastDimLength;
    for (int j = 0; j < lastDimLength; j++) {
      temp = (int16_t)(data_in[lastDimStart + j] + input_offset - mean);
      sum += temp * temp;
    }
    sum = sum / lastDimLength;
    sum += 1;
    _plp_sqrt_q32(&sum, 0, &std);

    for (int j = 0; j < lastDimLength; j++) {
      data_out[lastDimStart + j] =
          (int8_t)((((((int64_t)data_in[lastDimStart + j]) + input_offset -
                      mean) *
                     weight[j]) /
                        (std) +
                    bias[j]) >>
                   log2D);
    }
  }
}