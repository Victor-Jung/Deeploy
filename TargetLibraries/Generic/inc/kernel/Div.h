/* =====================================================================
 * Title:        Div.h
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

#ifndef __DEEPLOY_BASIC_MATH_DIV_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_DIV_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/*
 * This file implements the element-wise binary division.
 */

/******************************************************************************/
/*                              Division (32bit)                              */
/******************************************************************************/

void Div_s32_s32(int32_t *data_in_nom, int32_t *data_in_denom, int32_t size_nom,
                 int32_t size_denom, int32_t nomStep, int32_t denomStep,
                 int32_t *data_out, int32_t Delta, int32_t eps, int32_t eta);

void Div_fp32_fp32_fp32(float32_t *data_in_1, float32_t *data_in_2,
                        float32_t *data_out, int32_t size);

#endif //__DEEPLOY_BASIC_MATH_DIV_KERNEL_HEADER_
