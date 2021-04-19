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

tensor_equal
"""

import te.lang.cce
from te import tvm
from topi import generic
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import check_op_params, check_dtype, check_shape, refine_shape_axes

NUM_ONE = 1.0
NUM_ZERO = 0.0

# define a scalar exponent, value is -126,minimun num of float32 exponent
SCALAR_MIN_EXP_FP32 = -126
# define a scalar exponent, value is 50
SCALAR_MUL_EXP_FP32 = 50
# define a scalar exponent, value is 26
SCALAR_MUL2_EXP_FP32 = 26
# define a scalar exponent, value is -24,minimun num of float32 exponent
SCALAR_MIN_EXP_FP16 = -24
# define a scalar exponent, value is 12
SCALAR_MUL_EXP_FP16 = 12
# define a scalar, minimun num of float32 2^SCALAR_MIN_EXP_FP32
SCALAR_MIN_FP32 = 2 ** SCALAR_MIN_EXP_FP32
# define a scalar, value is 2^SCALAR_MUL_EXP_FP32
SCALAR_MUL_FP32 = 2 ** SCALAR_MUL_EXP_FP32
# define a scalar, value is 2^SCALAR_MUL2_EXP_FP32
SCALAR_MUL2_FP32 = 2 ** SCALAR_MUL2_EXP_FP32
# define a scalar, minimun num of float16 2^SCALAR_MIN_EXP_FP16
SCALAR_MIN_FP16 = 2 ** SCALAR_MIN_EXP_FP16
# define a scalar, value is 2^SCALAR_MUL_EXP_FP16
SCALAR_MUL_FP16 = 2 ** SCALAR_MUL_EXP_FP16
# define a scalar, value is 1
SCALAR_ONE = 1


@fusion_manager.register("tensor_equal")
def tensor_equal_compute_use_sub(input_x, input_y, output_y, kernel_name="tensor_equal"):
    '''
    True if two tensors have the same size and elements, False otherwise
    :param input_x: TVM tensor
            input tenser x
    :param input_y: TVM tensor
            input tensor y
    :param kernel_name: str
            kernel name, default value is "tensor_equal"
    :return:output_z
            output tensor with True or False
    '''
    dtype_x = input_x.dtype
    dtype_y = input_y.dtype
    shape_x = te.lang.cce.util.shape_to_list(input_x.shape)
    shape_y = te.lang.cce.util.shape_to_list(input_y.shape)

    x_axis_list = []
    for i in range(len(shape_x)):
        x_axis_list.append(i)

    if shape_x != shape_y or dtype_x != dtype_y:
        # general Falae result and return
        scalar_zero = tvm.const(NUM_ZERO, dtype=dtype_x)
        zero_res = te.lang.cce.vmuls(input_x, scalar_zero)
        zero_res = te.lang.cce.cast_to(zero_res, "int8", True)
        res = te.lang.cce.reduce_min(zero_res, x_axis_list, False)
        res = te.lang.cce.cast_to(res, "int8", True)
        return res

    if dtype_x == "float32":
        scalar_min = tvm.const(SCALAR_MIN_FP32, dtype="float32")
        scalar_mul = tvm.const(SCALAR_MUL_FP32, dtype="float32")
        scalar_mul1 = tvm.const(SCALAR_MUL2_FP32, dtype="float32")
        scalar_one = tvm.const(-1*SCALAR_ONE, dtype="float32")
    else:
        scalar_min = tvm.const(SCALAR_MIN_FP16, dtype="float16")
        scalar_mul = tvm.const(SCALAR_MUL_FP16, dtype="float16")
        scalar_one = tvm.const(-1*SCALAR_ONE, dtype="float16")
    if dtype_x in ("int8", "uint8"):
        input_x = te.lang.cce.cast_to(input_x, "float16")
        input_y = te.lang.cce.cast_to(input_y, "float16")

    res_vsub = te.lang.cce.vsub(input_x, input_y)
    res_vabs = te.lang.cce.vabs(res_vsub)
    res_min = te.lang.cce.vmins(res_vabs, scalar_min)
    res_vmul = te.lang.cce.vmuls(res_min, scalar_mul)
    res_vmul1 = te.lang.cce.vmuls(res_vmul, scalar_mul)

    if dtype_x == "float32":
        res_vmul2 = te.lang.cce.vmuls(res_vmul1, scalar_mul1)
        res_vsub1 = te.lang.cce.vadds(res_vmul2, scalar_one)
        res_vabs1 = te.lang.cce.vabs(res_vsub1)
    else:
        res_vsub1 = te.lang.cce.vadds(res_vmul1, scalar_one)
        res_vabs1 = te.lang.cce.vabs(res_vsub1)

    res = te.lang.cce.cast_to(res_vabs1, "int8", True)
    res = te.lang.cce.reduce_min(res, x_axis_list, True)
    res = te.lang.cce.cast_to(res, "int8", True)

    return res


@check_op_params(dict, dict, dict, str)
def tensor_equal(input_x, input_y, output_z, kernel_name="tensor_equal"):
    '''
    True if two tensors have the same size and elements, False otherwise

    :param input_x: dict
                input tenser x
    :param input_y: dict
                input tensor y
    :param kernel_name: str
                  kernel name, default value is "tensor_equal"
    :return: none
    '''

    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype")
    shape_y = input_y.get("shape")
    dtype_y = input_y.get("dtype")

    check_shape(shape_x)
    check_shape(shape_y)

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    check_dtype(dtype_x, check_list)
    check_dtype(dtype_y, check_list)

    shape_x = list(shape_x)
    shape_x, _ = refine_shape_axes(shape_x, [])
    data_input_x = tvm.placeholder(shape_x, name="data_input_x", dtype=dtype_x)
    shape_y, _ = refine_shape_axes(shape_y, [])
    data_input_y = tvm.placeholder(shape_y, name="data_input_y", dtype=dtype_y)

    # use vsub method compute equal result
    res = tensor_equal_compute_use_sub(data_input_x, data_input_y, output_z, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input_x, data_input_y, res],
              "bool_storage_as_1bit": False}

    te.lang.cce.cce_build_code(schedule, config)
