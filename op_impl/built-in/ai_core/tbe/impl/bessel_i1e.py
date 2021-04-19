# Copyright 2018 Huawei Technologies Co., Ltd
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
bessel_i1e

  Op_description :
    Computes the Bessel i0e function of `x` element-wise

    # bessel_i1e(
    #   x,
    #   y,
    #   kernel_name="bessel_i1e")

  Supportive_dtype_format :
    ['float16', 'float32']
    ['ALL']

  Constraint :
    [1] All : shape size limit is 2147483648.
"""
from impl.util import util_compute
from te import tvm
import te.lang.cce
from te.platform.cce_conf import api_check_support
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_shape
from te.utils.op_utils import refine_shape_axes
from te.utils.op_utils import *
import topi
from topi.cce import util

ITR_BEFORE = (0.5, 0.87890594, 0.51498869, 0.15084934, 0.02658773, 0.00301532, 0.00032411)

ITR_AFTER = (0.39894228, -0.03988024, -0.00362018,
             0.00163801, -0.01031555, 0.02282967,
             -0.02895312, 0.01787654, -0.00420059)

LEN_BEFORE = 7
LEN_AFTER = 9
CONST_LIMIT = 15.0/4


def _before_res_compute(abs_data, const_limit):
    """
    Algrithm:
    t = x / 3.75
    I1(x) = e^-x*x*(0.5 + 0.87890594t^2 + 0.51498869t^4 + 0.15084934t^6
                    + 0.02658773t^8 + 0.00301532t^10 + 0.00032411t^12)

    Parameters
    ----------
    abs_data: the placeholder of data input

    Returns
    -------
    A tensor of bessel_i1e(x)

    """

    data = te.lang.cce.vdiv(abs_data, const_limit)
    data_square = te.lang.cce.vmul(data, data)

    before_res = te.lang.cce.vmuls(data_square, tvm.const(ITR_BEFORE[LEN_BEFORE - 1]))
    before_res = te.lang.cce.vadds(before_res, ITR_BEFORE[LEN_BEFORE - 2])
    for index in reversed(range(LEN_BEFORE - 2)):
        before_res = te.lang.cce.vmul(before_res, data_square)
        before_res = te.lang.cce.vadds(before_res, ITR_BEFORE[index])

    tensor_exp = te.lang.cce.vexp(abs_data)
    before_res = te.lang.cce.vdiv(before_res, tensor_exp)
    before_res = te.lang.cce.vmul(before_res, abs_data)

    return before_res


def _after_res_compute(abs_data, const_limit):
    """
    Algrithm:
    t = 3.75 / x
    I1(x) = (1 / sqrt(x))*(0.39894228 - 0.03988024t - 0.00362018t^2
                           + 0.00163801t^3 - 0.01031555t^4 + 0.02282967t^5
                           - 0.02895312t^6 + 0.01787654t^7 - 0.00420059t^8)

    Parameters
    ----------
    abs_data: the placeholder of data input

    Returns
    -------
    A tensor of bessel_i1e(x)

    """

    data = te.lang.cce.vdiv(const_limit, abs_data)

    after_res = te.lang.cce.vmuls(data, tvm.const(ITR_AFTER[LEN_AFTER - 1]))
    after_res = te.lang.cce.vadds(after_res, ITR_AFTER[LEN_AFTER - 2])
    for index in reversed(range(LEN_AFTER - 2)):
        after_res = te.lang.cce.vmul(after_res, data)
        after_res = te.lang.cce.vadds(after_res, ITR_AFTER[index])

    tensor_sqrt = te.lang.cce.vsqrt(abs_data, 1)

    after_res = te.lang.cce.vdiv(after_res, tensor_sqrt)

    return after_res


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@fusion_manager.register("bessel_i1e")
def bessel_i1e_compute(x, y, kernel_name="bessel_i1e"):
    """
    Algrithm:
    I0 = 1 + ( (z/2) / (1!) )^2 + ((z/2)^2 / (2!))^2 + ... + ((z/2)^n / (n!)) ^2
    I0e = I0 / exp(x)
    I1e = I0e * z / (2*(k+1))
    u = 4 * v^2
    Ive = (1 - (u-1)/(8*z) + (u-1)*(u-9)/(2! * (8*z)^2) - (u-1)*(u-9)*(u-25)/(3!*(8*z)^3))
          /sqrt(2*pi*z)

    Parameters
    ----------
    x: the placeholder of data input

    y: the dict of output

    kernel_name: cce kernel name, default value is "bessel_i1e"

    Returns
    -------
    A tensor. Has the same type as x.
    """

    shape_input = x.shape
    dtype_input = x.dtype

    # chose the type of data in begin
    if dtype_input == "float16" and \
       api_check_support("te.lang.cce.vadd", "float32"):
        x = te.lang.cce.cast_to(x, "float32")

    abs_data = te.lang.cce.vabs(x)

    broad_const_limit = te.lang.cce.broadcast(tvm.const(CONST_LIMIT, x.dtype), shape_input)
    before_res = _before_res_compute(abs_data, broad_const_limit)
    after_res = _after_res_compute(abs_data, broad_const_limit)

    if abs_data.dtype == before_res.dtype and \
            api_check_support("te.lang.cce.vcmpsel", abs_data.dtype):
        res = te.lang.cce.vcmpsel(abs_data,
                                  broad_const_limit,
                                  'lt',
                                  before_res,
                                  after_res)
    else:
        select_index = te.lang.cce.vcmp(abs_data, broad_const_limit, 'lt')
        res = te.lang.cce.vsel(select_index, before_res, after_res)

    data_sign = util_compute.sign(x)
    res = te.lang.cce.vmul(res, data_sign)

    if dtype_input == "float16":
        res = te.lang.cce.cast_to(res, "float16")

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def bessel_i1e(x, y, kernel_name="bessel_i1e"):
    """
    Algrithm: calculating data's bessel

    Parameters
    ----------
    x: the dict of input, only support float16, float32

    y : the dict of output

    kernel_name : cce kernel name, default value is "bessel_i1e"

    Returns
    -------
    None
    """

    shape_input = x.get("shape")
    dtype_input = x.get("dtype")

    check_shape(shape_input, param_name="x")
    shape_input, _ = refine_shape_axes(shape_input, [])

    check_list = ("float16", "float32")
    check_dtype(dtype_input, check_list, param_name="x")

    input_dtype = dtype_input.lower()
    data = tvm.placeholder(shape_input, dtype=input_dtype, name="data_input")

    res = bessel_i1e_compute(data, y, kernel_name)

    with tvm.target.cce():
        sch = topi.generic.auto_schedule(res)

    config = {"name": kernel_name,
              "print_ir": False,
              "tensor_list": (data, res),
              "bool_storage_as_1bit": False}
    te.lang.cce.cce_build_code(sch, config)
