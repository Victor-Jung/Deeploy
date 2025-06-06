/* =====================================================================
 * Title:        Layernorm.h
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

#ifndef __DEEPLOY_BASIC_MATH_LAYERNORM_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_LAYERNORM_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/*
 *
 */

/******************************************************************************/
/*                             Layernorm (8bit)                               */
/******************************************************************************/

void Layernorm_s8_s8(int8_t *data_in, int8_t *data_out, int32_t *weight,
                     int32_t *bias, int32_t input_offset, int32_t size,
                     int32_t lastDimLength, int32_t log2D);

void Layernorm_fp32_fp32(float32_t *data_in, float32_t *data_out,
                         float32_t *scale, float32_t *bias, float32_t epsilon,
                         int32_t size, int32_t lastDimLength);

#endif //__DEEPLOY_BASIC_MATH_LAYERNORM_KERNEL_HEADER_
