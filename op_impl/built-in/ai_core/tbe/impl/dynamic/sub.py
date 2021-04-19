#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright 2018 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

sub
"""
from __future__ import absolute_import

import te.lang.dynamic
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from te.platform.shape_classifier import classify
from te.platform.shape_classifier import Mode
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import check_op_params
from te.utils.op_utils import variable_shape
from te.utils.op_utils import broadcast_shapes
from te.utils.op_utils import refine_shapes_for_broadcast
from te.utils.error_manager import error_manager_vector


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
def sub_compute(input_x, input_y, output_z, kernel_name="sub"):
    """
    calculating data's sub, c = a - b

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of first input data
    input_y: TVM tensor
        the placeholder of second input data
    output_z: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is sub

    Returns
    -------
    res : output of the data's sub
    """
    shape_x = te.lang.dynamic.shape_to_list(input_x.shape)
    shape_y = te.lang.dynamic.shape_to_list(input_y.shape)

    shape_x, shape_y, shape_max = broadcast_shapes(shape_x, shape_y,
                                                   param_name_input1="input_x",
                                                   param_name_input2="input_y")
    input_x = te.lang.dynamic.broadcast(input_x, shape_max)
    input_y = te.lang.dynamic.broadcast(input_y, shape_max)
    res = te.lang.dynamic.vsub(input_x, input_y)

    return res


@te.op.register_operator("Sub")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def sub(input_x, input_y, output_z, kernel_name="sub"):
    """
    do element-wise sub operation between two input tensors

    Parameters:
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32,int32
    input_y : dict
        shape and dtype of input, only support float16, float32,int32
    output_z: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : kernel name, default value is "sub"

    Returns
    -------
    None
    """

    check_list = ["float16", "float32", "int32"]
    x_dtype = input_x.get("dtype").lower()
    y_dtype = input_x.get("dtype").lower()
    if not x_dtype in check_list or not y_dtype in check_list:
        error_detal = "sub only support float16, float32, int32"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name,
                                                               "input_x",
                                                               "input_y",
                                                               error_detal)

    ins = classify([input_x, input_y], Mode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (x1, x2) in ins:
        with te.op.compute():
            x_shape, y_shape = variable_shape([x1, x2], support_broadcast=True)
            x_shape, y_shape = refine_shapes_for_broadcast(x_shape,
                                                           y_shape)
            data1 = tvm.placeholder(x_shape, x_dtype, "data1")
            data2 = tvm.placeholder(y_shape, y_dtype, "data2")
            res = sub_compute(data1, data2, output_z, kernel_name)
            tensors.append([data1, data2, res])
        with tvm.target.cce():
            sch = generic.auto_schedule(res)
        schedules.append(sch)
    # build
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensors}
    te.lang.dynamic.build(schedules, config)
