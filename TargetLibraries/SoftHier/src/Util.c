/* =====================================================================
 * Title:        Util.c
 * Description:
 *
 * Date:         06.05.2025
 *
 * ===================================================================== */

/*
 * Copyright (C) 2025 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Bowen Wang <bowwang@iis.ee.ethz.ch>, ETH Zurich
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

#include "DeeploySoftHierMath.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

int deeploy_log(const char *__restrict fmt, ...) {
  va_list args;
  va_start(args, fmt);
  int ret = vprintf(fmt, args);
  va_end(args);
  return ret;
}

void *deeploy_malloc(const size_t size) {
  return (void *)flex_hbm_malloc(size);
}
void deeploy_free(void *const ptr) { flex_hbm_free(ptr); }
