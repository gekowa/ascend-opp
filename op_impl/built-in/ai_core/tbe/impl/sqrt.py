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
sqrt
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from functools import reduce as reduceIns
from te import platform as tbe_platform
from te.utils.op_utils import *

# shape limit for aicore equals 2**31
SHAPE_SIZE_LIMIT = 2147483648


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("sqrt")
def sqrt_compute(input_data, output_data, kernel_name="sqrt"):
    """
    calculating data sqrt,y= x**0.5,mini not support vsqrt, use exp(0.5*log(x))

    Parameters
    ----------
    input_data: TVM tensor
        the placeholder of input data
    output_data: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is sqrt

    Returns
    -------
    result: TVM tensor
        the result of sqrt
    """
    dtype = input_data.dtype
    has_improve_precision = False
    if dtype == "float16" and\
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vsqrt",
                                                    "float32"):
        input_data = te.lang.cce.cast_to(input_data, "float32")
        has_improve_precision = True
    result = te.lang.cce.vsqrt(input_data)

    if has_improve_precision:
        result = te.lang.cce.cast_to(result, "float16")

    return result


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def sqrt(input_x, output_y, kernel_name="sqrt"):
    """
    algorithm: sqrt
    calculating data sqrt,y= x**0.5, mini not support vsqrt, use exp(0.5*log(x))

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is sqrt

    Returns
    -------
    None
    """
    input_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()

    check_shape(input_shape, param_name="input_x")
    check_dtype(input_dtype, ("float16", "float32"), param_name="input_x")

    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x*y, input_shape)
    input_data = tvm.placeholder(fuseshape, name="input_data",
                                 dtype=input_dtype)
    result = sqrt_compute(input_data, output_y, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(result)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [input_data, result]}

    te.lang.cce.cce_build_code(sch, config)
