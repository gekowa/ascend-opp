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

sqrt_grad
"""

import operator
from functools import reduce as reduce_ins
import te.lang.dynamic
from te import platform as tbe_platform
from te import tvm
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_elewise_shape_range
from te.platform.shape_classifier import classify
from te.platform.shape_classifier import Mode
from topi import generic
from te.utils.op_utils import check_op_params
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import variable_shape
from te.utils.op_utils import broadcast_shapes
from te.utils.op_utils import refine_shapes_for_broadcast
from te.utils.error_manager import error_manager_vector

# General limitation of the reduce size for input shape: 2**30
SHAPE_LIMIT = 1 << 30


# pylint: disable=locally-disabled,too-many-arguments,unused-argument

def sqrt_grad_compute(x, dx, out, kernel_name="sqrt_grad"):
    """
    algorithm: sqrt_grad_compute
    output_grad = input_grad/(2*input)

    Parameters
    ----------
    x: a tensor of input data

    dx : a tensor of grad

    Returns
    -------
    output data

    """

    dtype = x.dtype.lower()
    mul_support = tbe_platform.cce_conf.api_check_support("te.lang.cce.vmuls", "float32")
    if dtype == "float32" and not mul_support:
        error_detail = "not support dtype(float32) on this platform!"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "x", 'dx', error_detail)
    const_val_half = tvm.const(0.5, dtype)
    div_val = te.lang.dynamic.vdiv(dx, x)
    res = te.lang.dynamic.vmuls(div_val, const_val_half)
    return res


@te.op.register_operator("SqrtGrad")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def sqrt_grad(x, dx, out, kernel_name="sqrt_grad"):
    """
    algorithm: sqrt_grad_cce

    Parameters
    ----------
    x : dict of data: dict

    dx : dict of data_grad: dict

    out : dict of output: dict

    kernel_name : cce kernel name, default value is "sqrt_grad": str

    Returns
    -------
    None

    """
    x_dtype = x.get("dtype").lower()
    dx_dtype = dx.get("dtype").lower()
    check_list = ("float16", "float32")
    check_dtype(x_dtype, check_list, param_name="x")
    check_dtype(dx_dtype, check_list, param_name="dx")
    check_elewise_shape_range([x, dx], support_broadcast=False)
    if x_dtype != dx_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "x", "dx",
                                                              x_dtype, dx_dtype)
    ins = classify([x, dx], Mode.ELEWISE)
    schedules, tensors = [], []
    for (x, dx) in ins:
        with te.op.compute():
            x_shape, dx_shape = variable_shape([x, dx], support_broadcast=False)
            x_shape, dx_shape = refine_shapes_for_broadcast(x_shape, dx_shape)
            tensor_x = tvm.placeholder(x_shape, x_dtype, "tensor_x")
            tensor_dx = tvm.placeholder(dx_shape, dx_dtype, "tensor_dx")
            res = sqrt_grad_compute(tensor_x, tensor_dx, out, kernel_name)
            tensors.append([tensor_x, tensor_dx, res])
        with tvm.target.cce():
            sch = generic.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    te.lang.dynamic.build(schedules, config)
