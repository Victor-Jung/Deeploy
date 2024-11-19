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
 * - Victor Jung, ETH Zurich
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

#include "DeeployPULPMath.h"
#include "pmsis.h"

void PULPRequantShift_u8_s8_NHWC(uint8_t *data_in, int32_t size, int32_t *mul,
                             int32_t *add, int8_t *data_out, int32_t log2D,
                             int32_t channels, int32_t input_offset,
                             int32_t output_offset, int8_t output_min,
                             int8_t output_max, bool rounding) {
  int32_t intermediate;
  int8_t out;
  for (int i = 0; i < size; i++) {
    intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
                   add[i % channels];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void PULPRequantShift_u16_s8_NHWC(uint16_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, int8_t *data_out, int32_t log2D,
                              int32_t channels, int32_t input_offset,
                              int32_t output_offset, int8_t output_min,
                              int8_t output_max, bool rounding) {
  int32_t intermediate;
  int8_t out;
  for (int i = 0; i < size; i++) {
    intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
                   add[i % channels];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void PULPRequantShift_u32_s8_NHWC(uint32_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, int8_t *data_out, int32_t log2D,
                              int32_t channels, int32_t input_offset,
                              int32_t output_offset, int8_t output_min,
                              int8_t output_max, bool rounding) {
  int32_t intermediate;
  int8_t out;
  for (int i = 0; i < size; i++) {
    intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
                   add[i % channels];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void PULPRequantShift_u8_s8_NCHW(uint8_t *data_in, int32_t size, int32_t *mul,
                             int32_t *add, int8_t *data_out, int32_t log2D,
                             int32_t HW, int32_t input_offset,
                             int32_t output_offset, int8_t output_min,
                             int8_t output_max, bool rounding) {
  int32_t intermediate;
  int8_t out;
  for (int i = 0; i < size; i++) {
    intermediate =
        ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void PULPRequantShift_u16_s8_NCHW(uint16_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, int8_t *data_out, int32_t log2D,
                              int32_t HW, int32_t input_offset,
                              int32_t output_offset, int8_t output_min,
                              int8_t output_max, bool rounding) {
  int32_t intermediate;
  int8_t out;
  for (int i = 0; i < size; i++) {
    intermediate =
        ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void PULPRequantShift_u32_s8_NCHW(uint32_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, int8_t *data_out, int32_t log2D,
                              int32_t HW, int32_t input_offset,
                              int32_t output_offset, int8_t output_min,
                              int8_t output_max, bool rounding) {
  int32_t intermediate;
  int8_t out;
  for (int i = 0; i < size; i++) {
    intermediate =
        ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void PULPRequantShift_u8_u8_NHWC(uint8_t *data_in, int32_t size, int32_t *mul,
                             int32_t *add, uint8_t *data_out, int32_t log2D,
                             int32_t channels, int32_t input_offset,
                             int32_t output_offset, uint8_t output_min,
                             uint8_t output_max, bool rounding) {
  int32_t intermediate;
  uint8_t out;
  for (int i = 0; i < size; i++) {
    intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
                   add[i % channels];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void PULPRequantShift_u16_u8_NHWC(uint16_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, uint8_t *data_out, int32_t log2D,
                              int32_t channels, int32_t input_offset,
                              int32_t output_offset, uint8_t output_min,
                              uint8_t output_max, bool rounding) {
  int32_t intermediate;
  uint8_t out;
  for (int i = 0; i < size; i++) {
    intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
                   add[i % channels];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP((uint32_t)intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void PULPRequantShift_u32_u8_NHWC(uint32_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, uint8_t *data_out, int32_t log2D,
                              int32_t channels, int32_t input_offset,
                              int32_t output_offset, uint8_t output_min,
                              uint8_t output_max, bool rounding) {
  int32_t intermediate;
  uint8_t out;
  for (int i = 0; i < size; i++) {
    intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
                   add[i % channels];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP((uint32_t)intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void PULPRequantShift_u8_u8_NCHW(uint8_t *data_in, int32_t size, int32_t *mul,
                             int32_t *add, uint8_t *data_out, int32_t log2D,
                             int32_t HW, int32_t input_offset,
                             int32_t output_offset, uint8_t output_min,
                             uint8_t output_max, bool rounding) {
  int32_t intermediate;
  uint8_t out;
  for (int i = 0; i < size; i++) {
    intermediate =
        ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP((uint32_t)intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void PULPRequantShift_u16_u8_NCHW(uint16_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, uint8_t *data_out, int32_t log2D,
                              int32_t HW, int32_t input_offset,
                              int32_t output_offset, uint8_t output_min,
                              uint8_t output_max, bool rounding) {
  int32_t intermediate;
  uint8_t out;
  for (int i = 0; i < size; i++) {
    intermediate =
        ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP((uint32_t)intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void PULPRequantShift_u32_u8_NCHW(uint32_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, uint8_t *data_out, int32_t log2D,
                              int32_t HW, int32_t input_offset,
                              int32_t output_offset, uint8_t output_min,
                              uint8_t output_max, bool rounding) {
  int32_t intermediate;
  uint8_t out;
  for (int i = 0; i < size; i++) {
    intermediate =
        ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP((uint32_t)intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void PULPRequantShift_s8_u8_NHWC(int8_t *data_in, int32_t size, int32_t *mul,
                             int32_t *add, uint8_t *data_out, int32_t log2D,
                             int32_t channels, int32_t input_offset,
                             int32_t output_offset, uint8_t output_min,
                             uint8_t output_max, bool rounding) {
  int32_t intermediate;
  uint8_t out;
  for (int i = 0; i < size; i++) {
    intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
                   add[i % channels];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void PULPRequantShift_s16_u8_NHWC(int16_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, uint8_t *data_out, int32_t log2D,
                              int32_t channels, int32_t input_offset,
                              int32_t output_offset, uint8_t output_min,
                              uint8_t output_max, bool rounding) {
  int32_t intermediate;
  uint8_t out;
  for (int i = 0; i < size; i++) {
    intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
                   add[i % channels];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void PULPRequantShift_s32_u8_NHWC(int32_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, uint8_t *data_out, int32_t log2D,
                              int32_t channels, int32_t input_offset,
                              int32_t output_offset, uint8_t output_min,
                              uint8_t output_max, bool rounding) {
  int32_t intermediate;
  uint8_t out;
  for (int i = 0; i < size; i++) {
    intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
                   add[i % channels];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void PULPRequantShift_s8_u8_NCHW(int8_t *data_in, int32_t size, int32_t *mul,
                             int32_t *add, uint8_t *data_out, int32_t log2D,
                             int32_t HW, int32_t input_offset,
                             int32_t output_offset, uint8_t output_min,
                             uint8_t output_max, bool rounding) {
  int32_t intermediate;
  uint8_t out;
  for (int i = 0; i < size; i++) {
    intermediate =
        ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void PULPRequantShift_s16_u8_NCHW(int16_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, uint8_t *data_out, int32_t log2D,
                              int32_t HW, int32_t input_offset,
                              int32_t output_offset, uint8_t output_min,
                              uint8_t output_max, bool rounding) {
  int32_t intermediate;
  uint8_t out;
  for (int i = 0; i < size; i++) {
    intermediate =
        ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void PULPRequantShift_s32_u8_NCHW(int32_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, uint8_t *data_out, int32_t log2D,
                              int32_t HW, int32_t input_offset,
                              int32_t output_offset, uint8_t output_min,
                              uint8_t output_max, bool rounding) {
  int32_t intermediate;
  uint8_t out;
  for (int i = 0; i < size; i++) {
    intermediate =
        ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void PULPRequantShift_s32_s8_NCHW(int32_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, int8_t *data_out, int32_t log2D,
                              int32_t HW, int32_t input_offset,
                              int32_t output_offset, int8_t output_min,
                              int8_t output_max, bool rounding) {

  
  int8_t core_id = pi_core_id();
  int8_t log2Core = log2(NUM_CORES);
  int16_t chunk = (HW >> log2Core) + ((HW & (NUM_CORES - 1)) != 0);
  int16_t chunk_start = MIN(chunk * core_id, HW);
  int16_t chunk_stop = MIN(chunk_start + chunk, HW + 1);

  uint16_t channels = size/HW;
  int32_t intermediate;
  int8_t out;

  for (int j = 0; j < channels; j++) {
    for (int i = chunk_start; i < chunk_stop; i++) {
      intermediate = (int32_t)data_in[i + j*HW];

      intermediate = (intermediate + input_offset) * mul[j] + add[j];
      intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                    output_offset;
      out = (int8_t)CLAMP(intermediate, output_min, output_max);
      data_out[i + j*HW] = out;
    }
  }
}