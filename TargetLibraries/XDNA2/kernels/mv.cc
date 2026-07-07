// SPDX-FileCopyrightText: Copyright (C) 2026 ETH Zurich and University of
// Bologna. Adapted from the IRON generic matrix-vector kernel
// (Copyright (C) 2025 Advanced Micro Devices, Inc.).
// SPDX-License-Identifier: Apache-2.0
//
// BF16 matrix-vector (GEMV) microkernel for XDNA2 (AIE2p).
//
// This is the row-major, native-bf16 kernel used by IRON's production `gemv`
// operator (IRON/aie_kernels/generic/mv.cc), NOT the older "32-bit-word
// transposed" aie2 example kernel. It expects A in plain row-major order, so
// no dataflow transpose is required on the A ObjectFifo.
//
// matvec_vectorized computes `m` output rows in one call, reducing over the
// full K (= DIM_K) dimension:  c[row] = sum_k A[row,k] * b[k]. The reduction
// is done as an fp32 (accfloat) partial-sum vector followed by a horizontal
// reduce_add per row; the result is cast back to bf16 on store. The kernel
// OVERWRITES c (it does not accumulate), so no zero-init kernel is needed and
// the full K row band must reside in L1.
//
// Compile-time parameters (baked into mv.o under Deeploy's flag-less kernel
// build; override with -DDIM_K / -DVEC_SIZE to match a different K):
//   DIM_K    – reduction dimension K (length of the input vector / matrix row)
//   VEC_SIZE – SIMD chunk size r; must divide DIM_K and satisfy
//              DIM_K >= 2*VEC_SIZE (the pipelined loop assumes >= 2 iterations).
//
// The runtime `m` (output rows) and `row_offset` (into c) are passed as i32
// arguments, matching IRON's matvec_vectorized_bf16_bf16 ABI.

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>
#include <aie_kernels/aie_kernel_utils.h>

#ifndef DIM_K
#define DIM_K 32
#endif

#ifndef VEC_SIZE
#define VEC_SIZE 16
#endif

void matvec_scalar(uint32_t m, uint32_t k, const bfloat16 *__restrict a,
                   const bfloat16 *__restrict b, bfloat16 *__restrict c) {
  for (uint32_t row = 0; row < m; row++) {
    float acc = 0;
    for (uint32_t i = 0; i < k; i++) {
      acc += a[row * k + i] * b[i];
    }
    c[row] = static_cast<bfloat16>(acc);
  }
}

/*
Matrix-vector multiplication kernel

 - m: Number of output rows == number of rows in the input matrix
 - k: Number of columns in the input matrix == length of the input vector
 - a: Pointer to the input matrix, stored in row-major order
 - b: Pointer to the input vector
 - c: Pointer to the output vector
 - r: Vector size; data from the matrix and vector will be loaded in and
      processed in chunks of this size
*/
template <uint32_t r, uint32_t k>
void matvec_vectorized(uint32_t m, const bfloat16 *__restrict a,
                       const bfloat16 *__restrict b, bfloat16 *__restrict c) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  bfloat16 *c_end = c + m;
  const bfloat16 *b_end = b + k;
  for (; c < c_end; c++) {
    aie::accum acc = aie::zeros<accfloat, r>();
    // The following pragma enables pipelining the zero-overhead loop, but it
    // assumes there are at least two iterations, i.e. k >= 2*r. It will break
    // the code if that is not the case!
    AIE_LOOP_MIN_ITERATION_COUNT(k / VEC_SIZE)
    for (const bfloat16 *__restrict b_cur = b; b_cur < b_end;
         b_cur += r, a += r) {
      aie::vector<bfloat16, r> a_vec = aie::load_v<r>(a);
      aie::vector<bfloat16, r> b_vec = aie::load_v<r>(b_cur);
      acc = aie::mac(acc, a_vec, b_vec);
    }
    *c = static_cast<bfloat16>(aie::reduce_add(acc.template to_vector<float>()));
  }
}

// ---------------------------------------------------------------------------
// extern "C" entry points (IRON ABI: runtime m + row_offset, then A, b, c)
//
// The row_offset parameter writes the output to c + row_offset, which lets the
// caller compute several m-row sub-tiles into one larger acquired C tile
// without pointer arithmetic in the MLIR. Deeploy currently drives one full
// C tile per call (row_offset == 0), but the ABI is kept identical to IRON.
// ---------------------------------------------------------------------------
extern "C" {

void matvec_scalar_bf16_bf16(uint32_t m, uint32_t row_offset,
                             const bfloat16 *__restrict a_in,
                             const bfloat16 *__restrict b_in,
                             bfloat16 *__restrict c_out) {
  c_out += row_offset;
  matvec_scalar(m, DIM_K, a_in, b_in, c_out);
}

void matvec_vectorized_bf16_bf16(uint32_t m, uint32_t row_offset,
                                 const bfloat16 *__restrict a_in,
                                 const bfloat16 *__restrict b_in,
                                 bfloat16 *__restrict c_out) {
  c_out += row_offset;
  matvec_vectorized<VEC_SIZE, DIM_K>(m, a_in, b_in, c_out);
}

} // extern "C"
