# Copyright 2019 Huawei Technologies Co., Ltd
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
segment_max_d
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

# the threshold for default stack space
FIRST_DIM_SIZE_THRESHOLD = 2000
# shape limit for segment_max_d
SHAPE_SIZE_LIMIT = 2147483648


def _check_segment_ids(shape, segment_ids):
    """
    Check segment_max_d parameter segment_ids.

    Parameters
    ----------
    shape: tuple or list
        shape of Tensor

    segment_ids :
        should be the size of the first dimension
        must sorted and need not cover all values in the full range of
        valid values
        must be positive integer
        len(segment_ids) == shape[0]

    Returns
    -------
        None
    """
    if sorted(segment_ids) != list(segment_ids):
        raise RuntimeError("segment_ids must be sorted(from small to large)")

    if segment_ids[0] < 0:
        raise RuntimeError("segment_ids must be positive integer")

    for val in segment_ids:
        if not isinstance(val, int):
            raise RuntimeError("segment_ids must be positive integer")

    if len(segment_ids) != shape[0]:
        raise RuntimeError("len(segment_ids) == shape[0]")

# pylint: disable = locally-disabled,invalid-name,unused-argument,no-member
@fusion_manager.register("segment_max_d")
def segment_max_d_compute(x, y, segment_ids, kernel_name="segment_max_d"):
    """
    compute for tf_segment_max_cce
    """
    res = te.lang.cce.unsorted_segment_max(x, segment_ids, segment_ids[-1] + 1)

    return res


@util.check_input_type(dict, dict, (list, tuple), str)
def segment_max_d(x, y, segment_ids, kernel_name="segment_max_d"):
    """
    Operation and Schedule for segment_max


    Parameters
    ----------
    x : dict
        shape and dtype of input
    y: dict
        shape and dtype of output
    segment_ids : list
        should be the size of the first dimension
    kernel_name: str
        kernel name, default value is "segment_max_d"

    Returns
    -------
        None
    """
    shape = x.get("shape")
    dtype = x.get("dtype")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)

    check_list = ["float16", "float32", "int32"]
    if dtype.lower() not in check_list:
        raise RuntimeError(
            "segment_max only support float16, float32, int32")

    # when shape[0] > first_dim_size_threshold,
    # default stack space may not be enough, we need to prompt the user
    if shape[0] > FIRST_DIM_SIZE_THRESHOLD:
        print("Default stack space may not be enough.\
         You shall increase the stack space.")

    dtype = dtype.lower()

    _check_segment_ids(shape, segment_ids)

    input_data = tvm.placeholder(shape, name="input_data", dtype=dtype)
    with tvm.target.cce():
        res = segment_max_d_compute(input_data, y, segment_ids, kernel_name)
        sch = generic.auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": [input_data, res]}
    te.lang.cce.cce_build_code(sch, config)
