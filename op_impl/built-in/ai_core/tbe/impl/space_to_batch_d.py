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
space_to_batch_d
"""
from te import tvm
from te.platform.cce_build import build_config
from topi.cce import util
from impl.space_to_batch_nd_d import space_to_batch_nd_d_compute
from te.utils.op_utils import *


# pylint: disable=invalid-name,unused-argument
def _check_param(x, y, paddings, block_size, kernel_name):
    """check the parameters including shape, dtype, block_shape, paddings
    and kernel_name.
    """
    shape = x.get("shape")
    dtype = x.get("dtype").lower()
    dtype_list = ("float16", "float32")
    check_shape(shape, param_name="x")
    check_dtype(dtype, dtype_list, param_name="x")

    if len(shape) != 5:
        raise RuntimeError(
            "the shape of image_input should be 5, but got: %d" % len(shape))
    if block_size < 2:
        raise RuntimeError("the attr block_size must be greater than one")

    _check_padding(paddings)

    padding_shape = (shape[0], shape[1],
                     shape[2] + paddings[0][0] + paddings[0][1],
                     shape[3] + paddings[1][0] + paddings[1][1], shape[4])
    check_shape(padding_shape, param_name="paddings")

    padding_height, padding_width = padding_shape[2], padding_shape[3]
    if padding_height % block_size != 0 or padding_width % block_size != 0:
        raise RuntimeError(
            "both height_pad and width_pad must be divisible by block_size")

    output_shape = (padding_shape[0] * block_size * block_size,
                    padding_shape[1], padding_shape[2] // block_size,
                    padding_shape[3] // block_size, padding_shape[4])
    check_shape(output_shape, param_name="y")


def _check_padding(paddings):
    """
    check the paddings
    """
    if len(paddings) != 2 or len(paddings[0]) != 2 or len(paddings[1]) != 2:
        raise RuntimeError("the shape of paddings should be 2x2")

    def _check_padding_val(val):
        """
        check the padding value
        """
        if not (isinstance(val, int) and val >= 0):
            raise RuntimeError("paddings should be integer and must be >= 0")

    _check_padding_val(paddings[0][0])
    _check_padding_val(paddings[0][1])
    _check_padding_val(paddings[1][0])
    _check_padding_val(paddings[1][1])


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_INT,
                 (REQUIRED_ATTR_LIST_INT, REQUIRED_ATTR_LIST_LIST_INT), KERNEL_NAME)
def space_to_batch_d(x,
                     y,
                     block_size,
                     paddings,
                     kernel_name="space_to_batch_d"):
    """
    the main function of space_to_batch_d

    Parameters
    ----------
    x: dict,shape and datatype,datatype supports float16,float32
    y: dict,shape and datatype,datatype supports float16,float32
    block_size: must be greater than one. It indicates the block size
    paddings: (tuple, list),the padding of the input with zeros across the
              spatial dimensions as follows:
              paddings = [[pad_top, pad_bottom], [pad_left, pad_right]]
    kernel_name: cce kernel name, default value is "space_to_batch_d"
    Returns
    -------
    None
    """
    if len(paddings) == 4:
        paddings = [[paddings[0], paddings[1]], [paddings[2], paddings[3]]]

    _check_param(x, y, paddings, block_size, kernel_name)

    input_shape = x.get("shape")
    input_dtype = x.get("dtype").lower()
    block_shape = [block_size, block_size]

    data = tvm.placeholder(input_shape, name="data", dtype=input_dtype)
    res = space_to_batch_nd_d_compute(data, y, block_shape, paddings,
                                      kernel_name)
    sch = tvm.create_schedule(res.op)
    with build_config:
        tvm.build(sch, [data, res], "cce", name=kernel_name)
