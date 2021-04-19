# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
in_training_reduce_v2
"""
from __future__ import division

import math

import te.lang.cce
import te.platform.cce_params as cce_params
import te.platform.cce_emitinsn_params as cce_emitinsn_params

from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.platform import cce_util
from te.platform.cce_build import build_config

from topi import generic
from te.utils.op_utils import *
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


# pylint: disable=locally-disabled,unused-argument,invalid-name
# pylint: disable=locally-disabled,redefined-builtin
def op_select_format(x, sum, square_sum,
                     kernel_name="in_training_reduce_v2"):
    """
    select format dynamically
    """
    input0 = gen_param(classify="input0", name="x",
                       datatype="float16,float",
                       format="NC1HWC0,NC1HWC0")
    output0 = gen_param(classify="output0", name="sum",
                        datatype="float,float",
                        format="NC1HWC0,NC1HWC0")
    output1 = gen_param(classify="output1", name="square_sum",
                        datatype="float,float",
                        format="NC1HWC0,NC1HWC0")

    param_list = [input0, output0, output1]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def _check_input(data_format, shape):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    data_format: str
        data format of data
    origin_format: str
        origin format of data

    Returns
    -------
    None
    """
    check_format(data_format.upper(), ("NC1HWC0",), param_name="x")
    check_shape(shape, min_rank=5, max_rank=5, param_name="x")


def _reduce_compute(x):
    """
    algorithm: part of instance_norm_v2
    The first step of instance_norm
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: dict
        dict of sum, A `Tensor`. Sum of x.

    Returns
    -------
    res: TVM tensor list
        the result of in_training_reduce compute
    """
    axis = [2, 3]

    square_x = te.lang.cce.vmul(x, x)
    sum_x, square_sum_x = te.lang.cce.tuple_sum([x, square_x], axis, True)
    res = [sum_x, square_sum_x]

    return res


@fusion_manager.register("in_training_reduce_v2")
def in_training_reduce_compute(x, kernel_name="in_training_reduce_v2"):
    """
    algorithm: part of instance_norm_v2
    The first step of instance_norm_v2
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    kernel_name: str
        kernel name, default value is "in_training_reduce_v2"

    Returns
    -------
    res: TVM tensor list
        the result of in_training_reduce_v2 compute
    """
    if x.dtype == "float16":
        x = te.lang.cce.cast_to(x, "float32")
    res = _reduce_compute(x)
    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_OUTPUT,
                 KERNEL_NAME)
def in_training_reduce_v2(x, sum, square_sum,
                          kernel_name="in_training_reduce_v2"):
    """
    algorithm: part of instance_norm_v2
    The first step of instance_norm
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: dict
        dict of input, A Tensor for input data.
    sum: dict
        dict of sum, A `Tensor`. Sum of x.
    square_sum: dict
        dict of square_sum, A `Tensor`. Square sum of x.
    kernel_name: str
        kernel name, default value is "in_training_reduce_v2"

    Returns
    -------
    None
    """

    shape_x = x.get("shape")
    dtype_x = x.get("dtype")

    check_shape(shape_x, param_name="x")
    check_dtype(dtype_x.lower(), ("float16", "float32"), param_name="x")

    data_format = x.get("format")
    _check_input(data_format, shape_x)

    x_input = tvm.placeholder(shape_x, name="x_input",
                              dtype=dtype_x.lower())

    res = in_training_reduce_compute(x_input, kernel_name=kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)
    tensor_list = [x_input] + list(res)

    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    te.lang.cce.cce_build_code(sch, config)
