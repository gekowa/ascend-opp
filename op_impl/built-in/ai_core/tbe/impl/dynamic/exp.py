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

exp
"""
import math
from functools import reduce as reduceIns
import te.lang.dynamic
from te import tvm
from topi import generic
from te.platform.shape_classifier import classify
from te.platform.shape_classifier import Mode
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import OPTION_ATTR_FLOAT
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_op_params
from te.utils.op_utils import variable_shape
from te import platform as tbe_platform
from te.utils.error_manager import error_manager_vector


def isclose(valuex, valuey, rel_tol=1e-08, abs_tol=0.0):
    """
    determines whether the values of two floating-point numbers are close or equal
    """
    return math.isclose(valuex, valuey, rel_tol=rel_tol, abs_tol=abs_tol)


# pylint: disable=locally-disabled,unused-argument,too-many-arguments
def exp_compute(input_x, output_y, base=-1.0, scale=1.0, shift=0.0,
                kernel_name="exp"):
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
    api_check = tbe_platform.cce_conf.api_check_support("te.lang.dynamic.vexp",
                                                        "float32")
    if (not api_check) and (input_x.dtype == "float32"):
        input_x = te.lang.dynamic.cast_to(input_x, "float16")
    if isclose(scale, 1.0) and isclose(shift, 0.0):
        input_x_vadds = input_x
    else:
        scale_const = tvm.const(scale, dtype=input_x_dtype)
        shift_const = tvm.const(shift, dtype=input_x_dtype)
        input_x_vmuls = te.lang.dynamic.vmuls(input_x, scale_const)
        input_x_vadds = te.lang.dynamic.vadds(input_x_vmuls, shift_const)
    if base > 0:
        base_const = tvm.const(math.log(base), dtype=input_x.dtype)
        input_x_bases = te.lang.dynamic.vmuls(input_x_vadds, base_const)
        res = te.lang.dynamic.vexp(input_x_bases)
    # base is -1 value
    else:
        res = te.lang.dynamic.vexp(input_x_vadds)
    if input_x.dtype != input_x_dtype:
        res = te.lang.dynamic.cast_to(res, input_x_dtype)
    return res


@te.op.register_operator("Exp")
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, OPTION_ATTR_FLOAT,
                 OPTION_ATTR_FLOAT, OPTION_ATTR_FLOAT, KERNEL_NAME)
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
    dtype = input_x.get("dtype")
    # input_x' dtype check, only supports fp16 and fp32
    check_list = ("float16", "float32")
    input_dtype = dtype.lower()
    check_dtype(input_dtype, check_list, param_name="input_x")
    if base <= 0 and (not isclose(base, -1.0)):
        expect_value = "strictly positive or -1"
        real_value = "base < 0 or base notequal with -1"
        error_manager_vector.raise_err_input_value_invalid(
            kernel_name, "base", expecte_value, real_value)
    ins = classify([input_x], Mode.ELEWISE)
    schedules, tensors = [], []
    for (input_x,) in ins:
        with te.op.compute():
            shape_x = variable_shape([input_x])
            fuseshape = [1]
            fuseshape[0] = reduceIns(lambda x, y: x * y, shape_x[0])
            data_input = tvm.placeholder(fuseshape, name="data_input",
                                         dtype=input_dtype)
            res = exp_compute(data_input, output_y, base, scale, shift,
                              kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = generic.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    te.lang.dynamic.build(schedules, config)
