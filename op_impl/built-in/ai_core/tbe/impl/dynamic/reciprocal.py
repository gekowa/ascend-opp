#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

reciprocal
"""

import json
from te import tvm
from topi import generic
import te.lang.dynamic
from te import platform as tbe_platform
from functools import reduce as reduceIns
from te.platform.shape_classifier import classify
from te.platform.shape_classifier import Mode
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_op_params
from te.utils.op_utils import variable_shape

SHAPE_SIZE_LIMIT = 2147483648  # shape limit


# pylint: disable=redefined-builtin,unused-argument
def op_select_format(input_x, output_y, kernel_name="reciprocal"):
    """
    Get support format according to input_x
    """
    shape = input_x.get("shape")
    shape_len = len(shape)
    format = input_x.get("ori_format")
    format_4d_list = ["NCHW", "NHWC", "HWCN"]
    support_format = "ND,ND,NCHW,NCHW,NHWC,NHWC,HWCN,HWCN"
    ini_dict = {"input0": {"name": "x", "format": "ND",
                           "dtype": "float,float16"},
                "output0": {"name": "y", "format": "ND",
                            "dtype": "float,float16"}}

    # whether support format NC1HWC0、FRACTAL_Z、C1HWNCoC0
    if shape_len == 4 and format in format_4d_list:
        if format == "NCHW":
            n_dim = shape[0]
            c_dim = shape[1]
        if format == "NHWC":
            n_dim = shape[0]
            c_dim = shape[3]
        if format == "HWCN":
            n_dim = shape[3]
            c_dim = shape[2]
        # whether support format NC1HWC0
        if c_dim % 16 == 0:
            support_format += ("," + "NC1HWC0") * 2
        # whether support format FRACTAL_Z and C1HWNCoC0
        if n_dim % 16 == 0 and c_dim % 16 == 0:
            support_format += ("," + "FRACTAL_Z") * 2
            support_format += ("," + "C1HWNCoC0") * 2

    ini_dict["input0"]["format"] = support_format
    ini_dict["input0"]["dtype"] = "float,float16," * \
                                  (len(support_format.split(
                                      ",")) // 2 - 1) + "float,float16"
    ini_dict["output0"]["format"] = support_format
    ini_dict["output0"]["dtype"] = "float,float16," * \
                                   (len(support_format.split(
                                       ",")) // 2 - 1) + "float,float16"

    return json.dumps(ini_dict, indent=4)


def reciprocal_compute(input_x, output_y, kernel_name="reciprocal"):
    if tbe_platform.cce_conf.api_check_support("te.lang.dynamic.vdiv",
                                               "float32"):
        dtype = input_x.dtype
        shape = te.lang.dynamic.shape_to_list(input_x.shape)
        if dtype == "float16":
            input_x = te.lang.dynamic.cast_to(input_x, "float32")
        data_one = te.lang.dynamic.broadcast(tvm.const(1, "float32"), shape)
        res = te.lang.dynamic.vdiv(data_one, input_x)
        if dtype == "float16":
            res = te.lang.dynamic.cast_to(res, "float16")
    else:
        res = te.lang.dynamic.vrec(input_x)

    return res


@te.op.register_operator("Reciprocal")
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def reciprocal(input_x, output_y, kernel_name="reciprocal"):
    """
    algorithm: reciprocal

    calculating data's reciprocal,y= 1 / x

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is reciprocal

    Returns
    -------
    None
    """

    dtype = input_x.get("dtype")
    check_list = ("float16", "float32")
    input_dtype = dtype.lower()
    check_dtype(input_dtype, check_list, param_name="input_x")
    schedules, tensors = [], []
    ins = classify([input_x], Mode.ELEWISE)
    for (input_x,) in ins:
        with te.op.compute():
            x_shape = variable_shape([input_x])
            fuse_shape = [1]
            fuse_shape[0] = reduceIns(lambda x, y: x * y, x_shape[0])
            data_input = tvm.placeholder(fuse_shape, dtype=input_dtype,
                                         name="data_input")
            res = reciprocal_compute(data_input, output_y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = generic.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name,
              "print_ir": False,
              "tensor_list": tensors
              }
    te.lang.dynamic.build(schedules, config)
