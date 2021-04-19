"""
Copyright (C) Huawei Technologies Co., Ltd 2020-2020. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

hardtanh_grad
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils import op_utils

INF_FP32_VAL = 1e-10


@fusion_manager.register("hardtanh_grad")
def hardtanh_grad_compute(input_result, input_grad, output_y, min_val=-1.0, max_val=1.0, kernel_name="hardtanh_grad"):
    """
    calculating data

    Parameters
    ----------
    input_result : TVM tensor
        the placeholder of input_x
    input_grad : TVM tensor
        the placeholder of input_y
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "hardtanh_grad"

    Returns
    -------
    output tensor
    """

    """
    Please refer to the TE DSL Manual, And code here with TE DSL.
    """
    in_data_type = input_result.dtype.lower()
    f_min = tvm.const(min_val, dtype="float32")
    f_max = tvm.const(max_val, dtype="float32")

    max_tensor = te.lang.cce.broadcast(f_max, input_result.shape, output_dtype="float32")
    min_tensor = te.lang.cce.broadcast(f_min, input_result.shape, output_dtype="float32")

    if in_data_type == "float16":
        in_grad_float = te.lang.cce.cast_to(input_grad, "float32", False)
        in_result_float = te.lang.cce.cast_to(input_result, "float32", False)
        tmp_min = te.lang.cce.vmins(in_result_float, f_max)
    else:
        tmp_min = te.lang.cce.vmins(input_result, f_max)

    # control value in maximum & minimum
    tmp_max = te.lang.cce.vmaxs(tmp_min, f_min)
    sub_max = te.lang.cce.vsub(tmp_max, max_tensor)
    sub_min = te.lang.cce.vsub(tmp_max, min_tensor)
    mul_max_min = te.lang.cce.vmul(sub_max, sub_min)

    add_inf = te.lang.cce.vadds(mul_max_min, tvm.const(INF_FP32_VAL, dtype="float32"))
    div_res = te.lang.cce.vdiv(mul_max_min, add_inf)

    if in_data_type == "float16":
        res_float32 = te.lang.cce.vmul(div_res, in_grad_float)
        return te.lang.cce.cast_to(res_float32, in_data_type, False)
    else:
        return te.lang.cce.vmul(div_res, input_grad)


@op_utils.check_op_params(dict, dict, dict, float, float, str)
def hardtanh_grad(result, grad, y, min_val, max_val, kernel_name="hardtanh_grad"):
    """
    calculating data

    Parameters
    ----------
    result : dict
        shape and dtype of input
    grad : dict
        shape and dtype of input, should be same shape and type as result
    y : dict
        shape and dtype of input, should be same shape and type as input
    min_val:
        minimum value of the linear region range.
    max_val:
        maximum value of the linear region range.
    kernel_name : str
        kernel name, default value is "hardtanh_grad"

    Returns
    -------
    None
    """

    result_shape = result.get("shape")
    result_dtype = (result.get("dtype")).lower()
    grad_shape = grad.get("shape")
    grad_dtype = (grad.get("dtype")).lower()

    """
    operator check
    """
    util.check_shape_rule(result_shape)
    util.check_tensor_shape_size(result_shape)
    util.check_shape_rule(grad_shape)
    util.check_tensor_shape_size(grad_shape)
    util.check_kernel_name(kernel_name)

    check_tuple = ("float16", "float32")
    util.check_dtype_rule(grad_dtype, check_tuple)
    util.check_dtype_rule(result_dtype, check_tuple)

    if grad_dtype != result_dtype:
        raise RuntimeError("grad datatype %s and result datatype %s should be equal!" % (grad_dtype, result_dtype))

    if result_shape != grad_shape:
        raise RuntimeError("grad shape %s and result shape %s should be equal!" % (grad_shape, result_shape))

    """
    operator compute, invoke hardtanh_grad_compute
    """
    input_result = tvm.placeholder(result_shape, name="input_result", dtype=result_dtype)
    input_grad = tvm.placeholder(grad_shape, name="input_grad", dtype=grad_dtype)
    res = hardtanh_grad_compute(input_result, input_grad, y, min_val, max_val, kernel_name)

    """
    auto schedule
    """
    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    """
    operator build
    """
    config = {"name": kernel_name,
              "tensor_list": [input_result, input_grad, res]}

    te.lang.cce.cce_build_code(schedule, config)
