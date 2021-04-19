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

dynamic bn_training_reduce
"""
import te
import te.lang.dynamic
from te import tvm
from te.platform import log
from topi import generic
from te.utils.op_utils import check_op_params
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_format
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import variable_shape
from impl.util import fusion_util


# 'pylint: disable=unused-argument,invalid-name
def bn_training_reduce_compute(x, sum, square_sum,
                               kernel_name="bn_training_reduce"):
    """
    algorithm: part of fused_batch_norm_v2
    The first step of batch_norm
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: dict
        dict of sum, A `Tensor`. Sum of x.
    square_sum: dict
        dict of square_sum, A `Tensor`. Square sum of x.
    kernel_name: str
        kernel name, default value is "bn_training_reduce"

    Returns
    -------
    res: TVM tensor list
        the result of bn_training_reduce compute
    """
    dtype = x.dtype.lower()
    if dtype == "float16":
        x = te.lang.dynamic.cast_to(x, "float32")

    # format "NCHW" or "NC1HWC0"
    axis = [0, 2, 3]
    log.debug("input shape: " + str(x.shape))
    log.debug("reduce axis: " + str(axis))

    square_x = te.lang.dynamic.vmul(x, x)
    sum_x, square_sum_x = te.lang.dynamic.tuple_sum([x, square_x], axis, True)
    res = [sum_x, square_sum_x]

    return res


@te.op.register_fusion_compute("BnTrainingReduce")
def bn_training_reduce_fusion_compute(x, sum, square_sum,
                                      kernel_name="bn_training_reduce"):
    """
    algorithm: part of fused_batch_norm_v2
    The first step of batch_norm
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: dict
        dict of sum, A `Tensor`. Sum of x.
    square_sum: dict
        dict of square_sum, A `Tensor`. Square sum of x.
    kernel_name: str
        kernel name, default value is "bn_training_reduce"

    Returns
    -------
    res: TVM tensor list
        the result of bn_training_reduce compute
    """
    fusion_util.check_fusion_input([x])
    dict_x = fusion_util.extract_dict(x)
    shape_x = fusion_util.normalize_shape([dict_x])[0]

    data_input = fusion_util.create_placeholder(x, shape_x)
    res = bn_training_reduce_compute(data_input, sum, square_sum,
                                     kernel_name=kernel_name)

    return {"op_placeholder": [data_input], "op_res": list(res)}


@te.op.register_operator("BnTrainingReduce")
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT,
                 REQUIRED_OUTPUT, KERNEL_NAME)
def bn_training_reduce(x, sum, square_sum,
                       kernel_name="bn_training_reduce"):
    """
    algorithm: part of fused_batch_norm_v2
    The first step of batch_norm
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: dict
        dict of input, A 5HD Tensor for input data.
    sum: dict
        dict of sum, A `Tensor`. Sum of x.
    square_sum: dict
        dict of square_sum, A `Tensor`. Square sum of x.
    kernel_name: str
        kernel name, default value is "bn_training_reduce"

    Returns
    -------
    None
    """
    data_format = x.get("format").upper()
    origin_format = x.get("ori_format").upper()
    dtype = x.get("dtype").lower()

    # check and format
    check_list = ("NC1HWC0", "NCHW")
    check_format(data_format, check_list, param_name="x")
    if data_format == "NCHW" and origin_format not in ("NCHW",):
        raise RuntimeError("The origin format only supports "
                           "NCHW when format is NCHW")

    # check dtype
    check_list = ("float16", "float32")
    check_dtype(dtype, check_list, param_name="x")

    # get dynamic shape, x.get("shape"), x.get("range")
    shape_x = variable_shape([x])[0]

    # compute
    with te.op.compute():
        data_input = tvm.placeholder(shape_x, name="data_input", dtype=dtype)
        res = bn_training_reduce_compute(data_input, sum, square_sum,
                                         kernel_name=kernel_name)

    # schedule
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    # build
    tensor_list = [data_input] + list(res)
    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    te.lang.dynamic.build(sch, config)
