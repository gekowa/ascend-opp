"""
Copyright (C) Huawei Technologies Co., Ltd 2020-2020. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

lp_norm
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils import op_utils

_CONST_INF = 2147483647
_CONST_EPSILON = 1e-12


@fusion_manager.register("lp_norm")
def lp_norm_inf_compute(abs_x, y, p, axes, keepdim, kernel_name):
    """
    Compute norm for p = "inf" or p = "-inf"
    When p equals inf, lp_norm equals the max absolute value of elements;
    when -inf, lp_norm equals the min absolute value of elements.
    """
    if (p == "inf") or (p == _CONST_INF):
        res = te.lang.cce.reduce_max(abs_x, axis=axes, keepdims=keepdim)
    else:
        # p is "-inf"
        res = te.lang.cce.reduce_min(abs_x, axis=axes, keepdims=keepdim)
    return res


def lp_norm0_compute(abs_x, y, axes, keepdim, kernel_name):
    """
    Compute norm for p = 0.
    When p = 0, lp_norm equals the number of nonzero-elements
    """
    zero_tensor = te.lang.cce.vmuls(abs_x, tvm.const(0, dtype="float32"))
    one_tensor = te.lang.cce.vadds(zero_tensor, tvm.const(1, dtype="float32"))
    ele_tensor = te.lang.cce.vcmpsel(abs_x, zero_tensor, 'ne', one_tensor, zero_tensor)
    res = te.lang.cce.sum(ele_tensor, axis=axes, keepdims=keepdim)
    return res


def lp_norm1_compute(abs_x, y, axes, keepdim, kernel_name):
    """
    Compute norm for p = 1.
    When p = 1, lp_norm equals the sum of elements' absolute value
    """
    res = te.lang.cce.sum(abs_x, axis=axes, keepdims=keepdim)
    return res


def lp_norm2_compute(abs_x, y, axes, keepdim, kernel_name):
    """
    Compute norm for p = 2.
    For precision considering, separate it from lp_norm_compute without using vlog.
    """
    pow_x = te.lang.cce.vmul(abs_x, abs_x)
    sum_pow = te.lang.cce.sum(pow_x, axis=axes, keepdims=keepdim)
    res = te.lang.cce.vsqrt(sum_pow, priority_flag=1)
    return res


def lp_norm_compute(abs_x, y, p, axes, keepdim, kernel_name):
    """
    Compute norm for p >= 3.
    When p equals other int value, lp_norm = pow(sum(pow(abs(input),p)),1/p).
    """
    prod_x = abs_x
    for p_ix in range(1, p):
        prod_x = te.lang.cce.vmul(prod_x, abs_x)
    sum_prod_x = te.lang.cce.sum(prod_x, axis=axes, keepdims=keepdim)
    # extraction can be transformed like x^p = y --> x = exp(log(y)/p)
    log_sum_x = te.lang.cce.vlog(sum_prod_x)
    zero_tensor = te.lang.cce.vmuls(log_sum_x, tvm.const(0, dtype="float32"))
    p_tensor = te.lang.cce.vadds(zero_tensor, tvm.const(p, dtype="float32"))
    div_log_x = te.lang.cce.vdiv(log_sum_x, p_tensor)
    exp_div_x = te.lang.cce.vexp(div_log_x)
    return exp_div_x


def lp_norm(x, y, p=2, axes=None, keepdim=False, epsilon=1e-12, kernel_name="lp_norm"):
    """
    Computes norm for p equals 0, 1, 2, -inf, inf, or other integers.
    Parameters
    ----------
    x: tensor
       The input tensor.
       Required.
    y: tensor
       The output tensor.
       Required.
    p: int, inf, -inf
       The order of norm.
       Optional. Default: 2.
    axes: int list, None.
          The dimension on which the norm will be applied. None means all dimensions will be applied.
          Optional. Default: None.
    keepdim: bool
             Whether the output tensors should have dim keeped or not.
             Optional. Default: False
    epsilon: float
             The number used for safe considering as norm usually served as denominator.
             Optional. Default: 1e-12
    kernel_name: str
                 Kernel name.
                 Optional. Default: "lp_norm".
    Returns
    -------
    None
    """
    util.check_kernel_name(kernel_name)
    xtype_list = ["float16", "float32"]
    x_type = x.get("dtype").lower()
    x_shape = x.get("shape")
    op_utils.check_dtype(x_type, xtype_list)
    op_utils.check_shape(x_shape)
    p_inf_list = ("inf", "-inf")

    no_shape = len(x_shape)
    if isinstance(axes, int):
        axes = [axes]
    if axes is None:
        axes = [i for i in range(no_shape)]
    if len(axes) == 0:
        axes = [i for i in range(no_shape)]
    input_data = tvm.placeholder(x_shape, dtype=x_type, name="input_data")
    f_data = te.lang.cce.cast_to(input_data, "float32")
    abs_data = te.lang.cce.vabs(f_data)
    if (p in p_inf_list) or (p == _CONST_INF) or (p == -_CONST_INF - 1):
        res = lp_norm_inf_compute(abs_data, y, p, axes, keepdim, kernel_name)
    elif p == 0:
        res = lp_norm0_compute(abs_data, y, axes, keepdim, kernel_name)
    elif p == 1:
        res = lp_norm1_compute(abs_data, y, axes, keepdim, kernel_name)
    elif p == 2:
        res = lp_norm2_compute(abs_data, y, axes, keepdim, kernel_name)
    else:
        res = lp_norm_compute(abs_data, y, p, axes, keepdim, kernel_name)

    if epsilon > _CONST_EPSILON:
        std_no = tvm.const(float(epsilon), dtype="float32")
        res = te.lang.cce.vmaxs(res, std_no)

    res = te.lang.cce.cast_to(res, dtype=x_type)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_data, res]}
    te.lang.cce.cce_build_code(schedule, config)
