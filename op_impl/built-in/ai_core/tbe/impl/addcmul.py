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

addcmul
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import broadcast_shapes


SHAPE_SIZE_LIMIT = 2147483648


@fusion_manager.register("addcmul")
def addcmul_compute(input_data, x1, x2, shape_max, y, value, kernel_name="addcmul"):
    dtype = input_data.dtype

    input_data = te.lang.cce.broadcast(input_data, shape_max)
    x1 = te.lang.cce.broadcast(x1, shape_max)
    x2 = te.lang.cce.broadcast(x2, shape_max)

    vmul_val = te.lang.cce.vmul(x1, x2)
    value_val = tvm.const(value, dtype)
    vmuls_val = te.lang.cce.vmuls(vmul_val, value_val)
    res = te.lang.cce.vadd(input_data, vmuls_val)

    return res


@util.check_input_type(dict, dict, dict, dict, float, str)
def addcmul(input_data, x1, x2, y, value=1.0, kernel_name="addcmul"):
    """
    algorithm: addcmul
    calculating data's addcmul, y = input_data + value * (x1 * x2)

    Parameters
    ----------
    input_data : dict
        shape and dtype of first input, only support float16, float32, int32, int8, uint8
    x1 : dict
        shape and dtype of second input, only support float16, float32, int32, int8, uint8
    x2 : dict
        shape and dtype of third input, only support float16, float32, int32, int8, uint8
    y: dict
        shape and dtype of output, should be broadcast shape and type as input
    value: float
        scaling coefficient, default value is 1.0
    kernel_name : str
        cce kernel name, default value is addcmul

    Returns
    -------
    None
    """
    shape_input = input_data.get("shape")
    shape_x1 = x1.get("shape")
    shape_x2 = x2.get("shape")
    dtype_input = input_data.get("dtype").lower()
    dtype_x1 = x1.get("dtype").lower()
    dtype_x2 = x2.get("dtype").lower()

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_input)
    util.check_shape_size(shape_input, SHAPE_SIZE_LIMIT)
    util.check_shape_rule(shape_x1)
    util.check_shape_size(shape_x1, SHAPE_SIZE_LIMIT)
    util.check_shape_rule(shape_x2)
    util.check_shape_size(shape_x2, SHAPE_SIZE_LIMIT)

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    util.check_dtype_rule(dtype_input, check_list)
    util.check_dtype_rule(dtype_x1, check_list)
    util.check_dtype_rule(dtype_x2, check_list)
    if dtype_input != dtype_x1 or dtype_input != dtype_x2:
        raise RuntimeError("the type of input_data, x1, x2 must be same")

    shape_x1, shape_x2, shape_max1 = broadcast_shapes(shape_x1, shape_x2)
    util.check_tensor_shape_size(shape_max1)
    shape_input, _, shape_max = broadcast_shapes(shape_input, shape_max1)
    util.check_tensor_shape_size(shape_max)
    shape_x1, _, _ = broadcast_shapes(shape_x1, shape_max)
    shape_x2, _, _ = broadcast_shapes(shape_x2, shape_max)

    data_input = tvm.placeholder(shape_input, dtype=dtype_input, name="data_input")
    data_x1 = tvm.placeholder(shape_x1, dtype=dtype_x1, name="data_x1")
    data_x2 = tvm.placeholder(shape_x2, dtype=dtype_x2, name="data_x2")
    res = addcmul_compute(data_input, data_x1, data_x2, shape_max, y, value, kernel_name="addcmul")

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    tensor_list = [data_input, data_x1, data_x2, res]

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    te.lang.cce.cce_build_code(schedule, config)
