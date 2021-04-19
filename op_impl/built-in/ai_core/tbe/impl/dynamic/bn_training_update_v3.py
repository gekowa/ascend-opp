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

dynamic bn_training_update_v3
"""

import te
import te.lang.dynamic
from te import tvm
from te.platform import log
from te.platform import operation
from topi import generic
from te.utils.op_utils import check_op_params
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_format
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import REQUIRED_ATTR_FLOAT
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import variable_shape
from impl.util import fusion_util


def _check_shape_5hd(shape_x, shape_sum, shape_square_sum,
                     shape_scale, shape_offset):
    if len(shape_x) != 5 or len(shape_sum) != 5 \
            or len(shape_square_sum) != 5 or len(shape_scale) != 5 \
            or len(shape_offset) != 5:
        raise RuntimeError(
            "The data format is 5HD, "
            "but some input's shape length is not 5")

    dim_c1 = shape_x[1]
    dim_c0 = shape_x[4]

    if shape_sum[1] != dim_c1 or shape_sum[4] != dim_c0:
        raise RuntimeError(
            "Dimension C must be equal, but %s and %s"
            % (str(shape_x), str(shape_sum)))
    if shape_square_sum[1] != dim_c1 or shape_square_sum[4] != dim_c0:
        raise RuntimeError(
            "Dimension C must be equal, but %s and %s"
            % (str(shape_x), str(shape_square_sum)))
    if shape_scale[1] != dim_c1 or shape_scale[4] != dim_c0:
        raise RuntimeError(
            "Dimension C must be equal, but %s and %s"
            % (str(shape_x), str(shape_scale)))
    if shape_offset[1] != dim_c1 or shape_offset[4] != dim_c0:
        raise RuntimeError(
            "Dimension C must be equal, but %s and %s"
            % (str(shape_x), str(shape_offset)))


def _check_dtype(dtype_x, dtype_sum, dtype_square_sum,
                 dtype_scale, dtype_offset):
    check_dtype(dtype_x, ("float16", "float32"))
    check_dtype(dtype_sum, ("float32",))
    check_dtype(dtype_square_sum, ("float32",))
    check_dtype(dtype_scale, ("float32",))
    check_dtype(dtype_offset, ("float32",))


# 'pylint: disable=unused-argument,invalid-name,too-many-arguments,too-many-locals
def bn_training_update_v3_compute(x, sum, square_sum, scale, offset,
                                  y, batch_mean, batch_variance,
                                  reserve_1, reserve_2, epsilon,
                                  kernel_name="bn_training_update_v3"):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: TVM tensor
        contains sum data
    square_sum: TVM tensor
        contains square_sum data
    scale: TVM tensor
        contains scale data
    offset: TVM tensor
        contains offset data
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of batch_mean, A `Tensor`.
        One of the result which is called save mean.
    batch_variance: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_1: dict
        dict of batch_mean, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_2: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_variance`.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update_v3"

    Returns
    -------
    res: TVM tensor list
        the result of bn_training_update_v3 compute
    """
    shape_x = list(x.shape)

    # runtime tiling: "NCHW" or "NC1HWC0" reduce [0, 2, 3]
    # num = shape_x[0] * shape_x[2] * shape_x[3]
    # num_rec = 1.0/num
    # if num == 1: batch_var_scaler = 0.0
    # else: batch_var_scaler = float(num)/(num - 1)
    num_rec = operation.var("num_rec", dtype="float32")
    batch_var_scaler = operation.var("batch_var_scaler", dtype="float32")

    # compute the saved mean of x
    save_mean_reduce = te.lang.dynamic.vmuls(sum, num_rec)

    # compute the saved variance of x
    variance_div = te.lang.dynamic.vmuls(square_sum, num_rec)
    variance_square = te.lang.dynamic.vmul(save_mean_reduce, save_mean_reduce)
    save_variance_reduce = te.lang.dynamic.vsub(variance_div, variance_square)

    # compute the oefficient of y
    multiplier_add = te.lang.dynamic.vadds(save_variance_reduce, epsilon)
    multiplier_sqrt = te.lang.dynamic.vsqrt(multiplier_add)
    multiplier_div = te.lang.dynamic.vdiv(scale, multiplier_sqrt)
    multiplier = te.lang.dynamic.broadcast(multiplier_div, shape_x)

    addend_mul = te.lang.dynamic.vmul(multiplier_div, save_mean_reduce)
    addend_sub = te.lang.dynamic.vsub(offset, addend_mul)
    addend = te.lang.dynamic.broadcast(addend_sub, shape_x)

    # compute the batch normalization of x
    if x.dtype == "float16":
        x = te.lang.dynamic.cast_to(x, "float32")
        res_y = te.lang.dynamic.vadd(te.lang.dynamic.vmul(multiplier, x), addend)
        res_y = te.lang.dynamic.cast_to(res_y, "float16")
    else:
        res_y = te.lang.dynamic.vadd(te.lang.dynamic.vmul(multiplier, x), addend)

    # compute batch_mean and batch_var
    res_batch_mean = te.lang.dynamic.vmuls(sum, num_rec)
    res_batch_var = te.lang.dynamic.vmuls(save_variance_reduce, batch_var_scaler)

    res = [res_y, res_batch_mean, res_batch_var,
           save_mean_reduce, save_variance_reduce]

    return res


@te.op.register_fusion_compute("BnTrainingUpdate")
def bn_training_update_v3_fusion_compute(x, sum, square_sum, scale, offset,
                                         y, batch_mean, batch_variance,
                                         reserve_1, reserve_2, epsilon,
                                         kernel_name="bn_training_update_v3"):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: TVM tensor
        contains sum data
    square_sum: TVM tensor
        contains square_sum data
    scale: TVM tensor
        contains scale data
    offset: TVM tensor
        contains offset data
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of batch_mean, A `Tensor`.
        One of the result which is called save mean.
    batch_variance: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_1: dict
        dict of batch_mean, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_2: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_variance`.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update_v3"

    Returns
    -------
    res: TVM tensor list
        the result of bn_training_update_v3 compute
    """
    fusion_util.check_fusion_input([x, sum])
    dict_x = fusion_util.extract_dict(x)
    dict_sum = fusion_util.extract_dict(sum)
    shape_x, shape_sum = fusion_util.normalize_shape([dict_x, dict_sum])

    in_x = fusion_util.create_placeholder(x, shape_x)
    in_sum = fusion_util.create_placeholder(sum, shape_sum)
    in_sqrsum = fusion_util.create_placeholder(square_sum, shape_sum)
    in_scale = fusion_util.create_placeholder(scale, shape_sum)
    in_offset = fusion_util.create_placeholder(offset, shape_sum)
    res = bn_training_update_v3_compute(in_x, in_sum, in_sqrsum,
                                        in_scale, in_offset,
                                        y, batch_mean, batch_variance,
                                        reserve_1, reserve_2,
                                        epsilon, kernel_name=kernel_name)


    return {"op_placeholder": [in_x, in_sum, in_sqrsum, in_scale, in_offset],
            "op_res": list(res)}


@te.op.register_operator("BnTrainingUpdate")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT,
                 REQUIRED_INPUT, REQUIRED_INPUT,
                 REQUIRED_OUTPUT, REQUIRED_OUTPUT, REQUIRED_OUTPUT,
                 REQUIRED_OUTPUT, REQUIRED_OUTPUT,
                 REQUIRED_ATTR_FLOAT, KERNEL_NAME)
def bn_training_update_v3(x, sum, square_sum, scale, offset,
                          y, batch_mean, batch_variance, reserve_1, reserve_2,
                          epsilon, kernel_name="bn_training_update_v3"):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization.

    Parameters
    ----------
    x: dict
        dict of input, A 5HD Tensor for input data.
    sum: dict
        dict of sum, A 5HD Tensor for sum.
        The output of batch_normalization_forward_training_reduce.
    square_sum: dict
        dict of square_sum, A 5HD Tensor for square_sum.
        The output of batch_normalization_forward_training_reduce.
    scale: dict
        dict of scale, A 5HD Tensor for mean.
    offset: dict
        dict of offset, A 5HD Tensor for variance.
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of batch_mean, A `Tensor`.
        One of the result which is called save mean.
    batch_variance: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_1: dict
        dict of batch_mean, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_2: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_variance`.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update_v3"

    Returns
    -------
    None
    """
    dtype_x = x.get("dtype").lower()
    dtype_sum = sum.get("dtype").lower()
    dtype_sqrsum = square_sum.get("dtype").lower()
    dtype_scale = scale.get("dtype").lower()
    dtype_offset = offset.get("dtype").lower()

    shape_x = x.get("shape")
    shape_sum = sum.get("shape")
    shape_sqrsum = square_sum.get("shape")
    shape_scale = scale.get("shape")
    shape_offset = offset.get("shape")

    data_format = x.get("format").upper()
    origin_format = x.get("ori_format").upper()

    # check dtype
    _check_dtype(dtype_x, dtype_sum, dtype_sqrsum,
                 dtype_scale, dtype_offset)

    # check format
    check_list = ("NC1HWC0", "NCHW")
    check_format(data_format, check_list, param_name="x")
    if data_format == "NCHW" and origin_format not in ("NCHW",):
        raise RuntimeError("The origin format only supports "
                           "NCHW when format is NCHW")

    # check shape
    if data_format == "NC1HWC0":
        _check_shape_5hd(shape_x, shape_sum, shape_sqrsum,
                         shape_scale, shape_offset)
        shape_list = [1, 1, 1, 1, 1]
        shape_list[1] = shape_x[1]
        shape_list[4] = shape_x[4]
        shape_sum = shape_list
    else:
        shape_list = [1, 1, 1, 1]
        shape_list[1] = shape_x[1]
        shape_sum = shape_list

    # get dynamic shape
    shape_x, shape_sum = variable_shape([x, sum])
    log.debug("input_x shape: " + str(shape_x))
    log.debug("input_sum shape: " + str(shape_sum))

    # compute
    with te.op.compute():
        in_x = tvm.placeholder(shape_x, name="x", dtype=dtype_x)
        in_sum = tvm.placeholder(shape_sum, name="sum", dtype=dtype_sum)
        in_sqrsum = tvm.placeholder(shape_sum, name="sqrsum", dtype=dtype_sum)
        in_scale = tvm.placeholder(shape_sum, name="scale", dtype=dtype_sum)
        in_offset = tvm.placeholder(shape_sum, name="offset", dtype=dtype_sum)
        res = bn_training_update_v3_compute(in_x, in_sum, in_sqrsum,
                                            in_scale, in_offset,
                                            y, batch_mean, batch_variance,
                                            reserve_1, reserve_2,
                                            epsilon, kernel_name=kernel_name)

    # schedule
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    # build
    tensor_list = [in_x, in_sum, in_sqrsum, in_scale, in_offset] + list(res)
    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    te.lang.dynamic.build(sch, config)
