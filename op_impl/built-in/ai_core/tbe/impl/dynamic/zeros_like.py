#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

zeros_like
"""

import te.lang.dynamic
from te import tvm
from topi import generic
from functools import reduce as functools_reduce
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_op_params
from te.utils.op_utils import variable_shape
from te.platform.shape_classifier import classify
from te.platform.shape_classifier import Mode


# pylint: disable=locally-disabled,invalid-name,unused-argument
def zeros_like_compute(x, y, kernel_name="zeros_like"):
    """
    Enter a tensor, output a tensor of all zero,
    you can specify the output data type

    Parameters
    ----------
    x: TVM tensor
        the placeholder of input data
    y: TVM tensor
        the placeholder of output data
    kernel_name : str
        cce kernel name, default value is "zeros_like"

    Returns
    -------
    res: TVM tensor
        the result of zeros_like_compute
    """
    src_dtype = x.dtype.lower()
    dst_type = src_dtype
    src_type_list = ("int8", "uint8")
    dst_type_list = ("int8", "uint8")
    if src_dtype in src_type_list:
        src_dtype = "float16"
    zero = tvm.const(0, dtype=src_dtype)
    zero_src = te.lang.dynamic.broadcast(zero, x.shape)
    if src_dtype in dst_type_list:
        zero_src = te.lang.dynamic.cast_to(zero_src, dst_type,
                                           f1628IntegerFlag=True)
    else:
        zero_src = te.lang.dynamic.cast_to(zero_src, dst_type)
    return zero_src


@te.op.register_operator("ZerosLike")
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def zeros_like(x, y, kernel_name="zeros_like"):
    """
    output a tensor of all zero, you can specify the output type

    Parameters
    ----------
    x: dict
        shape and dtype of input, only support float16, float32,
        int32,int8,uint8,bool
    y: dict
        shape and dtype of output data
    kernel_name: str
        cce kernel name, default value is "zeros_like"

    Returns
    ------
    None
    """
    dtype_x = x.get("dtype")
    check_list_src = ("float16", "float32", "int32", "int8", "uint8", "bool")
    src_dtype = dtype_x.lower()
    check_dtype(src_dtype, check_list_src, param_name="x")
    schedules, tensors = [], []
    ins = classify([x], Mode.ELEWISE)
    for (input_x,) in ins:
        with te.op.compute():
            shape_x = variable_shape([input_x])
            shape_x = (functools_reduce(lambda x, y: x * y, shape_x[0]),)
            x_input = tvm.placeholder(shape_x, name="x_input", dtype=src_dtype)
            res = zeros_like_compute(x_input, y, kernel_name=kernel_name)
            tensors.append([x_input, res])
        with tvm.target.cce():
            sch = generic.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    te.lang.dynamic.build(schedules, config)
