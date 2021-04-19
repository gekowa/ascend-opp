#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

fast_gelu grad
"""
from __future__ import absolute_import

import operator

import te.lang.cce
from te import tvm
from topi import generic
from functools import reduce as reduce_ins
from te.utils.op_utils import *

CONST_1 = 1

# pylint: disable=locally-disabled,too-many-arguments,unused-argument,no-member
@fusion_manager.register("fast_gelu_grad")
def fast_gelu_grad_compute(input_dy, input_x, output_z,
                           kernel_name="fast_gelu_grad"):
    """
    algorithm: fast_gelu_grad
    calculating: dy*res'
    res' = div_up/div_down
    div_up = e^(-1.702x) + 1.702xe^(-1.702x) + e^(1.702(x-|x|))
    div_down = (e^(-1.702x)+1)^2

    Parameters
    ----------
    input_dy : dict
        shape and dtype of dy input, only support float16, float32
    input_x : dict
        shape and dtype of x input, only support float16, float32
    output_z: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is fast_gelu_grad

    Returns
    -------
    A TVM tensor same as input placeholders.
    """
    attr = 1.702
    dtype = input_x.dtype
    attr_opp = 0 - attr
    const_1 = tvm.const(attr_opp, dtype)
    const_2 = tvm.const(attr, dtype)
    const_3 = tvm.const(CONST_1, dtype)

    # e^(-1.702x)
    abs_x = te.lang.cce.vabs(input_x)
    mul_abs_x = te.lang.cce.vmuls(abs_x, const_1)
    exp_x = te.lang.cce.vexp(mul_abs_x)

    # 1.702xe^(-1.702x)
    add_2 = te.lang.cce.vmul(input_x, exp_x)
    add_2 = te.lang.cce.vmuls(add_2, const_2)

    # e^(1.702(x-|x|))
    pn_x = te.lang.cce.vsub(input_x, abs_x)
    mul_pn_x = te.lang.cce.vmuls(pn_x, const_2)
    exp_pn_x = te.lang.cce.vexp(mul_pn_x)

    #  e^(-1.702x) + 1.702xe^(-1.702x) + e^(1.702(x-|x|))
    div_up = te.lang.cce.vadd(exp_x, add_2)
    div_up = te.lang.cce.vadd(div_up, exp_pn_x)

    # (e^(-1.702x)+1)^2
    div_down_i = te.lang.cce.vadds(exp_x, const_3)
    div_down = te.lang.cce.vmul(div_down_i, div_down_i)

    result_temp = te.lang.cce.vdiv(div_up, div_down)
    result = te.lang.cce.vmul(input_dy, result_temp)
    return result


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT,
                 KERNEL_NAME)
def fast_gelu_grad(input_dy, input_x, output_z,
                   kernel_name="fast_gelu_grad"):
    """
    algorithm: fast_gelu_grad
    calculating: dy*res'
    res' = div_up/div_down
    div_up = e^(-1.702x) + 1.702xe^(-1.702x) + e^(1.702(x-|x|))
    div_down = (e^(-1.702x)+1)^2

    Parameters
    ----------
    input_dy : dict
        shape and dtype of dy input, only support float16, float32
    input_x : dict
        shape and dtype of x input, only support float16, float32
    output_z: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is fast_gelu_grad

    Returns
    -------
    none.
    """
    shape_dy = input_dy.get("shape")
    shape_x = input_x.get("shape")

    check_shape(shape_dy, param_name="input_dy")
    check_shape(shape_x, param_name="input_x")
    input_dtype = input_dy.get("dtype").lower()
    check_list = ("float16", "float32")
    check_dtype(input_dtype, check_list, param_name="input_dy")
    shape_dy = list(shape_dy)
    shape_x = list(shape_x)
    if not operator.eq(shape_dy, shape_x):
        raise RuntimeError("all input shape must be equal")

    fuseshape = [1]
    fuseshape[0] = reduce_ins(lambda x, y: x * y, shape_dy)
    data_dy = tvm.placeholder(fuseshape, name="data_dy", dtype=input_dtype)
    data_x = tvm.placeholder(fuseshape, name="data_x", dtype=input_dtype)
    res = fast_gelu_grad_compute(data_dy, data_x, output_z, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_dy, data_x, res]}

    te.lang.cce.cce_build_code(sch, config)
