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
batch_to_space_d
"""
from impl.batch_to_space_nd_d import batch_to_space_nd_d_compute
from te import tvm
from te.platform.cce_build import build_config
from topi.cce import util
from te.utils.op_utils import *

DIM_CNT = 5
CROPS_LEN = 2


# pylint: disable=locally-disabled,invalid-name
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_INT,
                 (REQUIRED_ATTR_LIST_INT, REQUIRED_ATTR_LIST_LIST_INT), KERNEL_NAME)
def batch_to_space_d(x, y, block_size, crops, kernel_name="batch_to_space_d"):
    """
    batch_to_space for tensor.

    Parameters
    ----------
    x: dict
        the dict of input tensor.
    y: dict
        the dict of output tensor.
    block_size: int
        the size of block.
    crops: list or tuple
        2-D with shape [2, 2], crops[i] = [crop_start, crop_end].
    kernel_name: str
        cce kernel name, default value is "batch_to_space".

    Returns
    -------
    None.
    """
    input_shape = x.get("shape")
    input_dtype = x.get("dtype").lower()
    if len(crops) == 4:
        crops = [[crops[0], crops[1]], [crops[2], crops[3]]]
    check_shape(input_shape, param_name="x")
    check_list = {"float16", "float32"}
    check_dtype(input_dtype, check_list, param_name="x")

    if len([x for x in input_shape if isinstance(x, int) and x > 0])\
            != len(input_shape):
        raise RuntimeError("input_shape should be positive integer")

    if len(input_shape) != DIM_CNT:
        raise RuntimeError("the length of input_shape must be 5,\
        while it is: %d" % len(input_shape))

    if not (len(crops) == CROPS_LEN and len(crops[0]) == CROPS_LEN
            and len(crops[1]) == CROPS_LEN):
        raise RuntimeError("shape of crops should be 2*2")

    if not (isinstance(crops[0][0], int) and crops[0][0] >= 0
            and isinstance(crops[0][1], int) and crops[0][1] >= 0
            and isinstance(crops[1][0], int) and crops[1][0] >= 0
            and isinstance(crops[1][1], int) and crops[1][1] >= 0):
        raise RuntimeError("crops  must be >= 0")

    batch_size = input_shape[0]
    if batch_size % (block_size * block_size) != 0:
        raise RuntimeError("batch_size  should be divisible by\
        the square of block_size")
    output_shape = (input_shape[0] // block_size // block_size, input_shape[1],
                    input_shape[2] * block_size - crops[0][0] - crops[0][1],
                    input_shape[3] * block_size - crops[1][0] - crops[1][1],
                    input_shape[4])
    check_shape(output_shape, param_name="y")

    block_shape = [block_size, block_size]
    data = tvm.placeholder(input_shape, name="data", dtype=input_dtype)

    res = batch_to_space_nd_d_compute(data, y, block_shape, crops, kernel_name)

    sch = tvm.create_schedule(res.op)

    with build_config:
        tvm.build(sch, [data, res], "cce", name=kernel_name)
