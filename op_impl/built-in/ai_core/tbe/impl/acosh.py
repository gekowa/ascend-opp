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
acosh

  Op_description :
    Computes inverse hyperbolic cosine of x element-wise

    # acosh(
    #   input_data,
    #   output_res,
    #   kernel_name="cce_acosh")

  Supportive_dtype_format :
    ['float16', 'float32']
    ['ALL']

  Constraint :
    [1] All : shape size limit is 2147483648.
"""
from te import tvm
import te.lang.cce
from te.platform.cce_conf import api_check_support
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_shape
from te.utils.op_utils import check_op_params
from te.utils.op_utils import refine_shape_axes
from te.utils.op_utils import *
import topi
from topi.cce import util

CONST_NEG_ONE = -1.0


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("acosh")
def acosh_compute(input_data, output_res, kernel_name="acosh"):
    """
    do element-wise acosh compute
    f(x) = log(x+sqrt(x^2-1)),  for all inputs

    Parameters:
    ----------
    input_data: the placeholder of data input

    output_res : the dict of output

    kernel_name : cce kernel name, default value is "acosh"

    Returns : A Tensor. Has the same type as input_data.
    -------
    """
    data = input_data

    input_dtype = data.dtype.lower()
    if input_dtype == "float16" and \
       api_check_support("te.lang.cce.vadd", "float32"):
        data = te.lang.cce.cast_to(data, "float32")

    res = te.lang.cce.vmul(data, data)
    res = te.lang.cce.vadds(res, tvm.const(CONST_NEG_ONE, data.dtype))
    res = te.lang.cce.vsqrt(res, 1)
    res = te.lang.cce.vadd(res, data)
    res = te.lang.cce.vlog(res, 1)

    if input_dtype == "float16":
        res = te.lang.cce.cast_to(res, "float16")

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def acosh(input_data, output_res, kernel_name="acosh"):
    """
    calculating data's acosh,y= log(x+sqrt(x^(2)-1))

    Parameters
    ----------
    input_data: the dict of input, only support float16, float32

    output_res : the dict of output

    kernel_name : cce kernel name, default value is "cce_acosh"

    Returns
    -------
    None

    """

    shape_input = input_data.get("shape")
    dtype_input = input_data.get("dtype")
    check_shape(shape_input, param_name="input_data")

    check_list = ("float16", "float32")
    check_dtype(dtype_input, check_list, param_name="input_data")
    shape_input, _ = refine_shape_axes(shape_input, [])

    input_dtype = dtype_input.lower()
    data = tvm.placeholder(shape_input, dtype=input_dtype, name="data_input")

    res = acosh_compute(data, output_res, kernel_name)

    with tvm.target.cce():
        sch = topi.generic.auto_schedule(res)

    config = {"name": kernel_name,
              "print_ir": False,
              "tensor_list": (data, res),
              "bool_storage_as_1bit": False}

    te.lang.cce.cce_build_code(sch, config)
