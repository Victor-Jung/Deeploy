/* =====================================================================
 * Title:        Div_s32.c
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

void Div_s32_s32(int32_t *data_in_nom, int32_t *data_in_denom, int32_t size_nom,
                 int32_t __attribute__((unused)) size_denom, int32_t nomStep,
                 int32_t denomStep, int32_t *data_out, int32_t Delta,
                 int32_t eps, int32_t eta) {

  int32_t innerMostIter = denomStep;
  int32_t secondIter = nomStep / innerMostIter;
  int32_t thirdIter = size_nom / secondIter;
  int64_t nom;
  int32_t sgnNom = 0;
  int64_t denom;

  for (int i = 0; i < thirdIter; i++) {
    for (int k = 0; k < innerMostIter; k++) {
      denom = data_in_denom[i * innerMostIter + k];
      denom = ((eta * denom) + eps);
      for (int j = 0; j < secondIter; j++) {
        nom =
            data_in_nom[i * secondIter * innerMostIter + j * innerMostIter + k];
        nom = (Delta * eta * nom);
        sgnNom = (nom >= 0) - (nom < 0);
        data_out[i * secondIter * innerMostIter + j * innerMostIter + k] =
            (int32_t)((nom + sgnNom * (denom >> 1)) / denom);
      }
    }
  }
}