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
tanh
"""
from __future__ import absolute_import
from functools import reduce as reduceIns

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform

from topi import generic
from topi.cce import util
from te.utils.op_utils import *

# shape size limit for aicore is 2**31
SHAPE_SIZE_LIMIT = 2147483648



# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("tanh")
def tanh_compute(input_x, output_y, kernel_name="tanh"):
    """
    algorithm: tanh
    calculating data's tanh,y= (e^(2x)-1)/(e^(2x)+1)

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is tanh

    Returns
    -------
    res : tvm.tensor
        the result of tanh
    """
    input_dtype = input_x.dtype
    # positive min float32 value
    MIN_FP_DATA = 2 ** (-126)
    CONST_DTYPE = input_dtype
    # positive min float16 value
    if input_dtype == "float16":
        MIN_FP_DATA = 2 ** (-14)

    has_improve_precision = False

    if input_dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp", "float32"):
        input_x = te.lang.cce.cast_to(input_x, "float32")
        has_improve_precision = True
        CONST_DTYPE = "float32"

    input_abs = te.lang.cce.vabs(input_x)
    power_val = te.lang.cce.vmuls(input_abs, tvm.const(-2, CONST_DTYPE))
    exp_val = te.lang.cce.vexp(power_val)

    up_val_tmp = te.lang.cce.vmul(exp_val, input_x)
    up_val = te.lang.cce.vsub(input_x, up_val_tmp)

    input_x_tmp = te.lang.cce.vadds(input_abs, MIN_FP_DATA)
    down_val_tmp = te.lang.cce.vadds(exp_val, tvm.const(1, CONST_DTYPE))
    down_val = te.lang.cce.vmul(down_val_tmp, input_x_tmp)

    res = te.lang.cce.vdiv(up_val, down_val)

    if has_improve_precision:
        res = te.lang.cce.cast_to(res, "float16")

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def tanh(input_x, output_y, kernel_name="tanh"):
    """
    algorithm: tanh
    calculating data's tanh,y= (e^(2x)-1)/(e^(2x)+1)

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is tanh

    Returns
    -------
    None
    """
    input_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()

    check_shape(input_shape, param_name="input_x")

    check_list = ("float16", "float32")
    check_dtype(input_dtype, check_list, param_name="input_x")
    # fuse single axis
    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x * y, input_shape)

    data = tvm.placeholder(fuseshape, name="data", dtype=input_dtype)
    res = tanh_compute(data, output_y, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, res]}

    te.lang.cce.cce_build_code(sch, config)
