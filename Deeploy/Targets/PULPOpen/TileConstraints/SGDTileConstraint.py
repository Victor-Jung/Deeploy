# ----------------------------------------------------------------------
#
# File: AddTileConstraint.py
#
# Last edited: 21.03.2025
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Run Wang, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from Deeploy.Targets.Generic.TileConstraints.BOPTileConstraint import BOPTileConstraint


class SGDTileConstraint(BOPTileConstraint):

    dataIn1Name = 'weight'
    dataIn2Name = 'grad'
    dataOutName = 'weight_updated'
