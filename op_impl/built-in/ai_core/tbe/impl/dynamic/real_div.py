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

real_div
"""
from __future__ import absolute_import

import te.lang.dynamic
from te import tvm
from topi import generic
from te.platform.shape_classifier import classify
from te.platform.shape_classifier import Mode
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_op_params
from te.utils.op_utils import variable_shape
from te.utils.op_utils import check_elewise_shape_range
from te.utils.op_utils import broadcast_shapes
from te.utils.op_utils import refine_shapes_for_broadcast
from te.utils.error_manager import error_manager_vector

# General limitation of the reduce size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648


# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=unused-argument,invalid-name

def real_div_compute(x1, x2, y, kernel_name="real_div"):
    """
    calculating data's realdiv, c = a / b

    Parameters
    ----------
    x1: TVM tensor
        the placeholder of first input data
    x2: TVM tensor
        the placeholder of second input data
    y: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is real_div

    Returns
    -------
    res : output of the data's divide
    """
    shape_x = te.lang.dynamic.shape_to_list(x1.shape)
    shape_y = te.lang.dynamic.shape_to_list(x2.shape)
    shape_x, shape_y, shape_max = broadcast_shapes(shape_x, shape_y)
    data_x = te.lang.dynamic.broadcast(x1, shape_max)
    data_y = te.lang.dynamic.broadcast(x2, shape_max)
    res = te.lang.dynamic.vdiv(data_x, data_y)

    return res


@te.op.register_operator("RealDiv")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def real_div(x1, x2, y, kernel_name="real_div"):
    """
    algorithm: real_div
    calculating data's real_div, c = a / b

    Parameters
    ----------
    x1 : dict
        shape and dtype of first input, only support float16, float32, int32
    x2 : dict
        shape and dtype of second input, only support float16, float32, int32
    y: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name : str
        cce kernel name, default value is real_div

    Returns
    -------
    None
    """

    x_dtype = x1.get("dtype").lower()
    y_dtype = x2.get("dtype").lower()
    check_list = ("float16", "float32")
    check_dtype(x_dtype, check_list, param_name="input_x")
    check_dtype(y_dtype, check_list, param_name="input_y")
    check_elewise_shape_range([x1, x2], support_broadcast=True)
    if x_dtype != y_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "x1", "x2",
                                                              x_dtype, y_dtype)
    ins = classify([x1, x2], Mode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (x1, x2) in ins:
        with te.op.compute():
            x_shape, y_shape = variable_shape([x1, x2], support_broadcast=True)
            x_shape, y_shape = refine_shapes_for_broadcast(x_shape, y_shape)
            tensor_x = tvm.placeholder(x_shape, x_dtype, "tensor_x")
            tensor_y = tvm.placeholder(y_shape, y_dtype, "tensor_y")
            res = real_div_compute(tensor_x, tensor_y, y, kernel_name)

            tensors.append([tensor_x, tensor_y, res])
        with tvm.target.cce():
            sch = generic.auto_schedule(res)
        schedules.append(sch)
    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    te.lang.dynamic.build(schedules, config)
