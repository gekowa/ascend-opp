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
exp
"""
import math
from functools import reduce as reduceIns
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import *


def isclose(valuex, valuey, rel_tol=1e-08, abs_tol=0.0):
    """
    determines whether the values of two floating-point numbers are close or equal
    """
    return math.isclose(valuex, valuey, rel_tol=rel_tol, abs_tol=abs_tol)

# pylint: disable=locally-disabled,unused-argument,too-many-arguments
@fusion_manager.register("exp")
def exp_compute(input_x, output_y, base=-1.0, scale=1.0, shift=0.0, kernel_name="exp"):
    """
    algorithm: exp
    calculating data's exp
    if base == -1:
       y = exp(shift + scale * x)
    if base > 0:
       y = exp((shift+scale*x)*ln(base))

    Parameters
    ----------
    input_x : TVM tensor, the placeholders of input data
    output_y : dict, shape and dtype of output, should be same shape and type as input
    base: (optional, default -1 for a value of e the base gamma
    scale: (optional, default 1) the scale alpha
    shift: (optional, default 0) the shift beta
    kernel_name : str, kernel name, default value is "exp"

    Returns
    -------
    res : the result of compute
    """
    input_x_dtype = input_x.dtype
    if isclose(scale, 1.0) and isclose(shift, 0.0):
        input_x_vadds = input_x
    else:
        scale_const = tvm.const(scale, dtype=input_x_dtype)
        shift_const = tvm.const(shift, dtype=input_x_dtype)
        input_x_vmuls = te.lang.cce.vmuls(input_x, scale_const)
        input_x_vadds = te.lang.cce.vadds(input_x_vmuls, shift_const)
    if base > 0:
        base_const = tvm.const(math.log(base), dtype=input_x_dtype)
        input_x_bases = te.lang.cce.vmuls(input_x_vadds, base_const)
        res = te.lang.cce.vexp(input_x_bases)

    # base is -1 value
    else:
        res = te.lang.cce.vexp(input_x_vadds)

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, OPTION_ATTR_FLOAT, OPTION_ATTR_FLOAT, OPTION_ATTR_FLOAT, KERNEL_NAME)
def exp(input_x, output_y, base=-1.0, scale=1.0, shift=0.0, kernel_name="exp"):
    """
    algorithm: exp
        calculating data's exp
    if base == -1:
       y = exp(shift + scale * x)
    if base > 0:
       y = exp((shift+scale*x)*ln(base))

    Parameters
    ----------
    input_x : dict,shape and dtype of input, only support float16,float32
    output_y: dict,shape and dtype of output, should be same shape and type as input
    base: (optional, default -1 for a value of e the base gamma
    scale: (optional, default 1) the scale alpha
    shift: (optional, default 0) the shift beta
    kernel_name : str, kernel name, default value is "exp"

    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype")

    check_shape(shape, param_name="input_x")

    # input_x' dtype check, only supports fp16 and fp32
    check_list = ("float16", "float32")
    input_dtype = dtype.lower()
    check_dtype(input_dtype, check_list, param_name="input_x")

    if base <= 0 and (not isclose(base, -1.0)):
        error_info = {}
        error_info['errCode'] = 'E80000'
        error_info['param_name'] = 'base'
        error_info['op_name'] = 'exp'
        error_info['expect_value'] = "strictly positive or -1"
        error_info['real_value'] = base
        raise RuntimeError("In op[%s], the parameter[%s] should be [%s], but actually is [%s]."
                           % (error_info['op_name'], error_info['param_name'],
                              error_info['expect_value'], error_info['real_value']))
    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x*y, shape)
    data_input = tvm.placeholder(fuseshape, name="data_input", dtype=input_dtype)

    res = exp_compute(data_input, output_y, base, scale, shift, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}
    te.lang.cce.cce_build_code(sch, config)
