# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
dynamic operator
"""
from __future__ import absolute_import as _abs

from .unsorted_segment_sum import unsorted_segment_sum
from . import gather_nd
from . import gather_v2
from .scatter_nd import scatter_nd
from .scatter_add import scatter_add
from .scatter_update import scatter_update
from .scatter_sub import scatter_sub
from .equal import equal
from .relu import relu
from .add import add
from .floor_mod import floor_mod
from .mul import mul
from .reduce_sum import reduce_sum
from .reduce_sum_d import reduce_sum_d
from .reduce_max_d import reduce_max_d
from .reduce_mean_d import reduce_mean_d
from .conv2d import conv2d
from .dynamic_atomic_addr_clean import dynamic_atomic_addr_clean
from . import sparse_apply_ftrl_d
from .div import div
from .sqrt import sqrt
from .square import square
from .sparse_apply_proximal_adagrad_d import sparse_apply_proximal_adagrad_d
from .maximum import maximum
from .minimum import minimum
from .add_n import add_n
from .greater_equal import greater_equal
from .less import less
from .less_equal import less_equal
from .floor_div import floor_div
from .tile_d import tile_d
from .logical_or import logical_or
from .real_div import real_div
from .reciprocal import reciprocal
from .neg import neg
from .concat_d import concat_d
from .concat_v2_d import concat_v2_d
from .strided_slice import strided_slice
from .slice import slice
from .cast import cast
from .exp import exp
from .leaky_relu_grad import leaky_relu_grad
from .log1p import log1p
from .sigmoid_grad import sigmoid_grad
from .sqrt_grad import sqrt_grad
from .zeros_like import zeros_like
from .conv2d_backprop_input import conv2d_backprop_input
from .sub import sub
from .transpose_d import transpose_d
from .unpack import unpack
from .pad_d import pad_d
from .split_d import split_d
from .strided_slice_grad import strided_slice_grad
from .fill import fill
