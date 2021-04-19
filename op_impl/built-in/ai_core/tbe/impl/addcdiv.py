"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

addcdiv
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import broadcast_shapes

SHAPE_SIZE_LIMIT = 2147483648


@fusion_manager.register("addcdiv")
def addcdiv_compute(data_x1, data_x2, data_x3, shape_max, alpha, kernel_name="addcdiv"):

    data_x1 = te.lang.cce.broadcast(data_x1, shape_max)
    data_x2 = te.lang.cce.broadcast(data_x2, shape_max)
    data_x3 = te.lang.cce.broadcast(data_x3, shape_max)

    div_val = te.lang.cce.vdiv(data_x2, data_x3)    # 执行input_x / input_y
    alpha_val = tvm.const(alpha, data_x1.dtype)
    muls_val = te.lang.cce.vmuls(div_val, alpha_val)
    res = te.lang.cce.vadd(data_x1, muls_val)

    return res


@util.check_input_type(dict, dict, dict, dict, float, str)
def addcdiv(x1, x2, x3, y=None, alpha=1.0, kernel_name="addcdiv"):

    check_list = ("float16", "float32")

    shape_x1 = x1.get("shape")
    dtype_x1 = x1.get("dtype").lower()

    shape_x2 = x2.get("shape")
    dtype_x2 = x2.get("dtype").lower()

    shape_x3 = x3.get("shape")
    dtype_x3 = x3.get("dtype").lower()

    util.check_shape_rule(shape_x1)    # 校验算子的shape，维度数需要大于等于1、小于等于8
    util.check_shape_size(shape_x1, SHAPE_SIZE_LIMIT)    # 校验算子第一个输入shape大小
    util.check_dtype_rule(dtype_x1, check_list)    # 校验算子的输入数据类型

    util.check_shape_rule(shape_x2)
    util.check_shape_size(shape_x2, SHAPE_SIZE_LIMIT)
    util.check_dtype_rule(dtype_x2, check_list)

    util.check_shape_rule(shape_x3)
    util.check_shape_size(shape_x3, SHAPE_SIZE_LIMIT)
    util.check_dtype_rule(dtype_x3, check_list)

    if dtype_x1 != dtype_x2 or dtype_x1 != dtype_x3:
        raise RuntimeError("the type of x1, x2, x3 must be the same!")

    util.check_kernel_name(kernel_name)    # 校验算子的kernel_name

    # 取shape_x1,shape_x2,shape_x3中每个维度的大值赋给shape_max
    shape_x2, shape_x3, shape_max = broadcast_shapes(shape_x2, shape_x3)
    util.check_tensor_shape_size(shape_max)     # 对shape_max进行校验
    shape_x1, _, shape_max = broadcast_shapes(shape_x1, shape_max)
    util.check_tensor_shape_size(shape_max)     # 对shape_max进行校验
    shape_x2, _, _ = broadcast_shapes(shape_x2, shape_max)    # 将input_x的shape广播为shape_max
    shape_x3, _, _ = broadcast_shapes(shape_x3, shape_max)    # 将input_y的shape广播为shape_max

    data_x1 = tvm.placeholder(shape_x1, name="data_x1", dtype=dtype_x1)
    data_x2 = tvm.placeholder(shape_x2, name="data_x2", dtype=dtype_x2)
    data_x3 = tvm.placeholder(shape_x3, name="data_x3", dtype=dtype_x3)

    res = addcdiv_compute(data_x1, data_x2, data_x3, shape_max, alpha, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x1, data_x2, data_x3, res]}

    te.lang.cce.cce_build_code(schedule, config)
