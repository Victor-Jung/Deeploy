/*
 * pulp_nn_add_i8_i8_i8.c
 * Georg Rutishauser <georgr@iis.ee.ethz.ch>
 * Victor Jung <jungvi@iis.ee.ethz.ch>
 *
 * Copyright (C) 2018-2020 University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DeeploySnitchMath.h"

void __attribute__((noinline))
snitch_nn_add_i8_i8_i8(int8_t *pIn1, int8_t *pIn2, int8_t *pOut,
                       int32_t in1_mul, int32_t in1_add, uint16_t in1_shift,
                       int32_t in2_mul, int32_t in2_add, uint16_t in2_shift,
                       int32_t out_mul, int32_t out_add, uint16_t out_shift,
                       uint16_t dim_im_in_x, uint16_t dim_im_in_y,
                       uint16_t ch_im_in, int out_requant_flag) {
  int core_id = snrt_global_compute_core_idx();
  int n_cores = snrt_global_compute_core_num();

  if (dim_im_in_y < n_cores) {
    n_cores = dim_im_in_y;
  }

  int Log2Core = INT_LOG2(n_cores);
  int chunck = (dim_im_in_y >> Log2Core) + ((dim_im_in_y & (n_cores - 1)) != 0);

  int32_t in1_rq1, in1_rq2, in1_rq3, in1_rq4, in2_rq1, in2_rq2, in2_rq3,
      in2_rq4;
  int32_t sum1, sum2, sum3, sum4;
  int32_t sum_out1, sum_out2, sum_out3, sum_out4;
  int32_t out1, out2, out3, out4, sum_int1, sum_int2, sum_int3, sum_int4;

  int ch_im_in1_r = ch_im_in >> 0;
  int ch_im_in2_r = ch_im_in >> 0;
  int ch_im_out_r = ch_im_in >> 0;

  int start = MIN(chunck * core_id, dim_im_in_y);
  int stop = MIN(start + chunck, dim_im_in_y);

  int8_t *target1 = pIn1 + start * ch_im_in1_r * dim_im_in_x;
  int8_t *target2 = pIn2 + start * ch_im_in2_r * dim_im_in_x;
  int8_t *pOutBuffer = pOut + start * ch_im_out_r * dim_im_in_x;

  int a = 0;
  int b = 0;

  int8_t *target1_ext = &a;
  int8_t *target2_ext = &b;

  for (int i = 0; i < (((stop - start) * ch_im_out_r * dim_im_in_x) >> 2);
       i++) {
    target1_ext = target1;
    target1 += 4;

    target2_ext = target2;
    target2 += 4;
#ifdef ADD_VERBOSE
    printf("core %d - in1 it0 before requant: %d\n", core_id, *(target1_ext));
    printf("core %d - in2 it0 before requant: %d\n", core_id, *(target2_ext));
#endif
    in1_rq1 = ((*(target1_ext)) * in1_mul + in1_add) >> in1_shift;
    in2_rq1 = ((*(target2_ext)) * in2_mul + in2_add) >> in2_shift;
    sum1 = clips8(in1_rq1) + clips8(in2_rq1);
#ifdef ADD_VERBOSE
    printf("core %d - in1_rq1 it0 after requant: %d\nclipped in1_rq1: %d\n",
           core_id, in1_rq1, clips8(in1_rq1));
    printf("core %d - in2_rq1 it0 after requant: %d\nclipped in2_rq1: %d\n",
           core_id, in2_rq1),
        clips8(in2_rq1);
    printf("core %d - sum1: %d\n", core_id, sum1);
#endif
#ifdef ADD_VERBOSE
    printf("core %d - in1 it1 before requant: %d\n", core_id,
           *(target1_ext + 1));
    printf("core %d - in2 it1 before requant: %d\n", core_id,
           *(target2_ext + 1));
#endif
    in1_rq2 = ((*(target1_ext + 1)) * in1_mul + in1_add) >> in1_shift;
    in2_rq2 = ((*(target2_ext + 1)) * in2_mul + in2_add) >> in2_shift;
    sum2 = clips8(in1_rq2) + clips8(in2_rq2);
#ifdef ADD_VERBOSE
    printf("core %d - in1_rq2 it1 after requant: %d\nclipped in1_rq2: %d\n",
           core_id, in1_rq2, clips8(in1_rq2));
    printf("core %d - in2_rq2 it1 after requant: %d\nclipped in2_rq2: %d\n",
           core_id, in2_rq2),
        clips8(in2_rq2);
    printf("core %d - sum2: %d\n", core_id, sum2);
#endif
#ifdef ADD_VERBOSE
    printf("core %d - in1 it2 before requant: %d\n", core_id,
           *(target1_ext + 2));
    printf("core %d - in2 it2 before requant: %d\n", core_id,
           *(target2_ext + 2));
#endif
    in1_rq3 = ((*(target1_ext + 2)) * in1_mul + in1_add) >> in1_shift;
    in2_rq3 = ((*(target2_ext + 2)) * in2_mul + in2_add) >> in2_shift;
    sum3 = clips8(in1_rq3) + clips8(in2_rq3);
#ifdef ADD_VERBOSE
    printf("core %d - in1_rq3 it2 after requant: %d\nclipped in1_rq3: %d\n",
           core_id, in1_rq3, clips8(in1_rq3));
    printf("core %d - in2_rq3 it2 after requant: %d\nclipped in2_rq3: %d\n",
           core_id, in2_rq3),
        clips8(in2_rq3);
    printf("core %d - sum3: %d\n", core_id, sum3);
#endif
#ifdef ADD_VERBOSE
    printf("core %d - in1 it3 before requant: %d\n", core_id,
           *(target1_ext + 3));
    printf("core %d - in2 it3 before requant: %d\n", core_id,
           *(target2_ext + 3));
#endif
    in1_rq4 = ((*(target1_ext + 3)) * in1_mul + in1_add) >> in1_shift;
    in2_rq4 = ((*(target2_ext + 3)) * in2_mul + in2_add) >> in2_shift;
    sum4 = clips8(in1_rq4) + clips8(in2_rq4);
#ifdef ADD_VERBOSE
    printf("core %d - in1_rq4 it3 after requant: %d\nclipped in1_rq4: %d\n",
           core_id, in1_rq4, clips8(in1_rq4));
    printf("core %d - in2_rq4 it3 after requant: %d\nclipped in2_rq4: %d\n",
           core_id, in2_rq4),
        clips8(in2_rq4);
    printf("core %d - sum4: %d\n", core_id, sum4);
#endif

    if (out_requant_flag) {
      sum1 = (sum1 * out_mul + out_add) >> out_shift;
#ifdef ADD_VERBOSE
      printf("core %d - requantized sum1: %d\n", core_id, sum1);
#endif
      sum2 = (sum2 * out_mul + out_add) >> out_shift;
#ifdef ADD_VERBOSE
      printf("core %d - requantized sum2: %d\n", core_id, sum2);
#endif
      sum3 = (sum3 * out_mul + out_add) >> out_shift;
#ifdef ADD_VERBOSE
      printf("core %d - requantized sum3: %d\n", core_id, sum3);
#endif
      sum4 = (sum4 * out_mul + out_add) >> out_shift;
#ifdef ADD_VERBOSE
      printf("core %d - requantized sum4: %d\n", core_id, sum4);
#endif
    }
    out1 = clips8(sum1);
#ifdef ADD_VERBOSE
    printf("core %d - out1 clipped: %d\n", core_id, out1);
#endif
    out2 = clips8(sum2);
#ifdef ADD_VERBOSE
    printf("core %d - out2 clipped: %d\n", core_id, out2);
#endif
    out3 = clips8(sum3);
#ifdef ADD_VERBOSE
    printf("core %d - out3 clipped: %d\n", core_id, out3);
#endif
    out4 = clips8(sum4);
#ifdef ADD_VERBOSE
    printf("core %d - out4 clipped: %d\n", core_id, out4);
#endif

    *pOutBuffer = (int8_t)out1;
    pOutBuffer++;
    *pOutBuffer = (int8_t)out2;
    pOutBuffer++;
    *pOutBuffer = (int8_t)out3;
    pOutBuffer++;
    *pOutBuffer = (int8_t)out4;
    pOutBuffer++;
  }
  // SCHEREMO: Cleanup leftovers, not doing it with this codebase for sub-byte
  // formats
  for (int i = 0; i < (((stop - start) * ch_im_out_r * dim_im_in_x) % 4); i++) {
    in1_rq1 = ((*(target1)) * in1_mul + in1_add) >> in1_shift;
    in2_rq1 = ((*(target2)) * in2_mul + in2_add) >> in2_shift;

// SCHEREMO: Maybe it's just LLVM, but unless I hack 3 non-unrolled nops in
// here, stuff fails
#pragma nounroll
    for (int j = 0; j < 3; j++) {
      asm volatile("nop" ::);
    }

    target1++;
    target2++;
    sum1 = clips8(in1_rq1) + clips8(in2_rq1);
    if (out_requant_flag) {
      sum1 = (sum1 * out_mul + out_add) >> out_shift;
    }

    out1 = clips8(sum1);
    *pOutBuffer = (int8_t)out1;
    pOutBuffer++;
  }
}
