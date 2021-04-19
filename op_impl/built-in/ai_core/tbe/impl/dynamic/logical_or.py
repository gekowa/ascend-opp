#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

logical_or
"""
import te.lang.dynamic
from te import tvm
from te.platform.shape_classifier import classify
from te.platform.shape_classifier import Mode
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_op_params
from te.utils.op_utils import check_elewise_shape_range
from te.utils.op_utils import variable_shape
from te.utils.op_utils import broadcast_shapes
from te.utils.op_utils import refine_shapes_for_broadcast
from topi import generic


# pylint: disable=unused-argument,invalid-name
# pylint: disable=locally-disabled,too-many-arguments,too-many-locals
def logical_or_compute(x1, x2, y, kernel_name="logical_or"):
    """
    algorithm : logical_or_compute
    calculating the value of x1 OR x2 element-wise

    Parameters
    ----------
    x1 : the placeholders of x1

    x2 : the placeholders of x2

    y : the dict of y

    kernel_name : string, cce kernel name, default value is "logical_or"

    Returns
    -------
    result res
    """
    _, _, shape_max = broadcast_shapes(te.lang.dynamic.shape_to_list(x1.shape),
                                       te.lang.dynamic.shape_to_list(x2.shape),
                                       param_name_input1="x1",
                                       param_name_input2="x2")
    x1 = te.lang.dynamic.cast_to(x1, "float16")
    x2 = te.lang.dynamic.cast_to(x2, "float16")
    x1 = te.lang.dynamic.broadcast(x1, shape_max)
    x2 = te.lang.dynamic.broadcast(x2, shape_max)
    res = te.lang.dynamic.vmax(x1, x2)
    res = te.lang.dynamic.cast_to(res, "int8")

    return res


@te.op.register_operator("LogicalOr")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def logical_or(x1, x2, y, kernel_name="logical_or"):
    """
    algorithm : logical_or
    calculating the value of x1 OR x2 element-wise

    Parameters
    ----------
    x1 : the dict of x1,
         include shape and dtype,
         dtype support int8, the value only support 0, 1

    x2 : the dict of x2,
         include shape and dtype,
         dtype support int8, the value only support 0, 1

    y : the dict of y, include shape and dtype

    kernel_name : string, cce kernel name, default value is "logical_or"

    Returns
    -------
    None
    """

    dtype_x1 = x1.get("dtype").lower()
    dtype_x2 = x2.get("dtype").lower()
    if dtype_x1 == "bool" or dtype_x2 == "bool":
        dtype_x1 = "int8"
        dtype_x2 = "int8"

    check_tuple = ("int8",)
    check_dtype(dtype_x1, check_tuple, param_name="x1")
    check_dtype(dtype_x2, check_tuple, param_name="x2")
    check_elewise_shape_range([x1, x2], support_broadcast=True)

    ins = classify([x1, x2], Mode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (x1, x2) in ins:
        with te.op.compute():
            shape_x1, shape_x2 = variable_shape([x1, x2],
                                                support_broadcast=True)
            shape_x1, shape_x2 = refine_shapes_for_broadcast(shape_x1, shape_x2)
            data_x1 = tvm.placeholder(shape_x1, name="data_x1", dtype=dtype_x1)
            data_x2 = tvm.placeholder(shape_x2, name="data_x2", dtype=dtype_x2)
            res = logical_or_compute(data_x1, data_x2, y, kernel_name)

            tensors.append((data_x1, data_x2, res))
        with tvm.target.cce():
            schedule = generic.auto_schedule(res)
        schedules.append(schedule)

    config = {"print_ir": False, "need_build": False, "name": kernel_name,
              "tensor_list": tensors}
    te.lang.dynamic.build(schedules, config)
