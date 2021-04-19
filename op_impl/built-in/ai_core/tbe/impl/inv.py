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
inv
"""
from __future__ import absolute_import

from functools import reduce as functools_reduce
from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import *

# define a scalar , value = 1
SCALAR_ONE = 1


# pylint: disable=locally-disabled,unused-argument
@fusion_manager.register("inv")
def inv_compute(input_x, output_y, kernel_name="inv"):
    """
    compute inv

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: TVM tensor
        the placeholder of output data
    kernel_name: str
        kernel name, default value is "inv"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    dtype = input_x.dtype
    shape = te.lang.cce.util.shape_to_list(input_x.shape)

    temp_const = tvm.const(SCALAR_ONE, dtype=dtype)
    temp_tensor = te.lang.cce.broadcast(temp_const, shape, dtype)
    res = te.lang.cce.vdiv(temp_tensor, input_x)

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def inv(input_x, output_y, kernel_name="inv"):
    """
    algorithm: inv
    calculating data's reciprocal, y = 1 / x

    Parameters
    ----------
    input_x: dict
        shape and dtype of input, only support float16, float32, int32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "inv"

    Returns
    -------
    None
    """
    shape_input = input_x.get("shape")
    dtype_input = input_x.get("dtype")

    check_shape(shape_input, param_name="input_x")

    dtype_input = dtype_input.lower()
    check_list = ("float16", "float32", "int32")
    check_dtype(dtype_input, check_list, param_name="input_x")

    shape_input = util.shape_refine(shape_input)
    shape_input = (functools_reduce(lambda x, y: x*y, shape_input[:]),)
    data_input = tvm.placeholder(shape_input,
                                 name="data_input",
                                 dtype=dtype_input)

    res = inv_compute(data_input, output_y, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}
    te.lang.cce.cce_build_code(sch, config)
