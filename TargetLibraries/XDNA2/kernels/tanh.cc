// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All
// rights reserved. SPDX-License-Identifier: Apache-2.0

#include <aie_kernels/aie_kernel_utils.h>

#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

void tanh_bf16_vectorized(bfloat16 *restrict input_vector,
                          bfloat16 *restrict output_vector,
                          const int32_t vector_size) {
  event0();

  int num_elems = vector_size;
  auto it_in = aie::begin_restrict_vector<16>((bfloat16 *)input_vector);
  auto it_out = aie::begin_restrict_vector<16>((bfloat16 *)output_vector);

  aie::vector<bfloat16, 16> input;
  aie::accum<accfloat, 16> acc;
  aie::vector<bfloat16, 16> output;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(64)
  for (int i = 0; i < num_elems; i += 16) {
    input = *it_in++;

    acc.from_vector(input, 0);
    auto tanh_x = aie::tanh<bfloat16>(acc.to_vector<float>());

    *it_out++ = tanh_x;
  }

  event1();

  return;
}

extern "C" {

void tanh_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
               int input_size) {
  tanh_bf16_vectorized(input, output, input_size);
}

} // extern "C"
