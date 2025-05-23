/* =====================================================================
 * Title:        CycleCounter.h
 * Description:
 *
 * Date:         06.12.2022
 *
 * ===================================================================== */

/*
 * Copyright (C) 2022 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Philip Wiese (wiesep@iis.ee.ethz.ch), ETH Zurich
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

#ifndef __DEEPLOY_MATH_CYCLE_HEADER_
#define __DEEPLOY_MATH_CYCLE_HEADER_

#include <stdint.h>

// Resets the internal cycle and instruction counter to zero
void ResetTimer(void);

// Starts the internal cycle and instruction counter
void StartTimer(void);

// Stops the internal cycle and instruction counter
void StopTimer(void);

// Returns the current number of cycles according to the internal cycle counter
uint32_t getCycles(void);

// Returns the current number of instructions according to the internal
// instructions counter
uint32_t getInstr(void);

#endif //__DEEPLOY_MATH_CYCLE_HEADER_
