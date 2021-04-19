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

fast_gelu
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from topi import generic
from functools import reduce as reduce_ins
from te.utils.op_utils import *

# const value
CONST_1 = 1

# pylint: disable=locally-disabled,too-many-arguments,unused-argument,no-member
@fusion_manager.register("fast_gelu")
def fast_gelu_compute(input_x, output_y, kernel_name="fast_gelu"):
    """
    mathematical formula of fast_gelu(x):
    fast_gelu(x) = xe^(0.851x)(x-|x|)/(1+e^(-1.702|x|))
    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input input_x
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is fast_gelu

    Returns
    -------
     A TVM tensor same as input placeholders.
    """
    attr = 1.702
    dtype = input_x.dtype.lower()
    attr_opp = 0 - attr
    attr_half = attr / 2
    const_0 = tvm.const(attr_opp, dtype)
    const_1 = tvm.const(CONST_1, dtype)
    abs_x = te.lang.cce.vabs(input_x)
    mul_abs_x = te.lang.cce.vmuls(abs_x, const_0)
    exp_abs_x = te.lang.cce.vexp(mul_abs_x)
    div_down = te.lang.cce.vadds(exp_abs_x, const_1)

    const_2 = tvm.const(attr_half, dtype)
    pn_x = te.lang.cce.vsub(input_x, abs_x)
    mul_pn_x = te.lang.cce.vmuls(pn_x, const_2)
    exp_pn_x = te.lang.cce.vexp(mul_pn_x)
    div_up = te.lang.cce.vmul(input_x, exp_pn_x)

    result = te.lang.cce.vdiv(div_up, div_down)

    return result


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def fast_gelu(input_x, output_y, kernel_name="fast_gelu"):
    """
    mathematical formula of fast_gelu(x):
    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is fast_fast_gelu

    Returns
    -------
    none.
    """
    attr = 1.702
    shape = input_x.get("shape")
    check_shape(shape, param_name="input_x")

    check_list = ("float16", "float32")
    input_dtype = input_x.get("dtype").lower()
    check_dtype(input_dtype, check_list, param_name="input_x")

    fuseshape = [1]
    fuseshape[0] = reduce_ins(lambda x, y: x * y, shape)
    data = tvm.placeholder(fuseshape, name="data", dtype=input_dtype)
    result = fast_gelu_compute(data, output_y, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(result)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, result]}

    te.lang.cce.cce_build_code(sch, config)

