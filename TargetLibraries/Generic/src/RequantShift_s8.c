/* =====================================================================
 * Title:        RequantShift_s8.c
 * Description:
 *
 * Date:         19.12.2022
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

void RequantShift_s8_s8_NHWC(int8_t *data_in, int32_t size, int32_t *mul,
                             int32_t *add, int8_t *data_out, int32_t log2D,
                             int32_t channels, int32_t input_offset,
                             int32_t output_offset, int8_t output_min,
                             int8_t output_max, bool rounding) {
  int32_t intermediate;
  int8_t out;
  for (int i = 0; i < size; i++) {
    intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
                   add[i % channels];
    if (rounding && log2D > 0)
      intermediate += 1 << (log2D - 1);
    intermediate = (intermediate >> log2D) + output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_s16_s8_NHWC(int16_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, int8_t *data_out, int32_t log2D,
                              int32_t channels, int32_t input_offset,
                              int32_t output_offset, int8_t output_min,
                              int8_t output_max, bool rounding) {
  int32_t intermediate;
  int8_t out;
  for (int i = 0; i < size; i++) {
    intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
                   add[i % channels];
    if (rounding && log2D > 0)
      intermediate += 1 << (log2D - 1);
    intermediate = (intermediate >> log2D) + output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_s32_s8_NHWC(int32_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, int8_t *data_out, int32_t log2D,
                              int32_t channels, int32_t input_offset,
                              int32_t output_offset, int8_t output_min,
                              int8_t output_max, bool rounding) {
  int32_t intermediate;
  int8_t out;
  for (int i = 0; i < size; i++) {
    intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
                   add[i % channels];
    if (rounding && log2D > 0)
      intermediate += 1 << (log2D - 1);
    intermediate = (intermediate >> log2D) + output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_s8_s8_NCHW(int8_t *data_in, int32_t size, int32_t *mul,
                             int32_t *add, int8_t *data_out, int32_t log2D,
                             int32_t HW, int32_t input_offset,
                             int32_t output_offset, int8_t output_min,
                             int8_t output_max, bool rounding) {
  int32_t intermediate;
  int8_t out;
  for (int i = 0; i < size; i++) {
    intermediate =
        ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
    if (rounding && log2D > 0)
      intermediate += 1 << (log2D - 1);
    intermediate = (intermediate >> log2D) + output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_s16_s8_NCHW(int16_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, int8_t *data_out, int32_t log2D,
                              int32_t HW, int32_t input_offset,
                              int32_t output_offset, int8_t output_min,
                              int8_t output_max, bool rounding) {
  int32_t intermediate;
  int8_t out;
  for (int i = 0; i < size; i++) {

    intermediate = (int32_t)data_in[i];

#ifdef DEEPLOY_PULP_PLATFORM
    // SCHEREMO: PULP specific hack
    // SCHEREMO: Need to trigger a HW loop with at least 3 nops
#pragma nounroll
    for (int j = 0; j < 3; j++) {
      asm volatile("nop" ::);
    }
#endif

    intermediate = (intermediate + input_offset) * mul[i / HW] + add[i / HW];
    if (rounding && log2D > 0)
      intermediate += 1 << (log2D - 1);
    intermediate = (intermediate >> log2D) + output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_s32_s8_NCHW(int32_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, int8_t *data_out, int32_t log2D,
                              int32_t HW, int32_t input_offset,
                              int32_t output_offset, int8_t output_min,
                              int8_t output_max, bool rounding) {
  int32_t intermediate;
  int8_t out;
  for (int i = 0; i < size; i++) {
    intermediate = (int32_t)data_in[i];

#ifdef DEEPLOY_PULP_PLATFORM
    // SCHEREMO: PULP specific hack
    // SCHEREMO: Need to trigger a HW loop with at least 3 nops
#pragma nounroll
    for (int j = 0; j < 3; j++) {
      asm volatile("nop" ::);
    }
#endif

    intermediate = (intermediate + input_offset) * mul[i / HW] + add[i / HW];
    if (rounding && log2D > 0)
      intermediate += 1 << (log2D - 1);
    intermediate = (intermediate >> log2D) + output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}
