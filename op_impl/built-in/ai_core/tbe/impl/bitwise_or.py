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
bitwise_or
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import refine_shapes_for_broadcast
from te.utils.op_utils import *

# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=too-many-locals,invalid-name
@fusion_manager.register("bitwise_or")
def bitwise_or_compute(placeholders, shape_x, shape_y):
    """
    calculating data's element_or, c = a | b

    Parameters
    ----------
    placeholders : tuple of data
    shape_x: list of int
            shape of input_x
    shape_y: list of int
            shape of input_y

    Returns
    -------
    res : z of the data's bitwise_or
    """
    data_x = placeholders[0]
    data_y = placeholders[1]
    shape_x, shape_y, shape_max = broadcast_shapes(shape_x,
                                                   shape_y,
                                                   param_name_input1="x1",
                                                   param_name_input2="x2")
    data_x_broadcast = te.lang.cce.broadcast(data_x, shape_max)
    data_y_broadcast = te.lang.cce.broadcast(data_y, shape_max)
    res = te.lang.cce.vor(data_x_broadcast, data_y_broadcast)

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def bitwise_or(x1, x2, y, kernel_name="bitwise_or",):
    """
    algorithm: bitwise_or
    calculating data's bitwise_or, c = a | b

    Parameters
    ----------
    x1: dict
              shape and dtype of data_1
    x2: dict
              shape and dtype of data_2
    y: dict
              shape and dtype of y
    kernel_name : string
                  cce kernel name, default value is "bitwise_or"

    Returns
    -------
    None
    """
    shape_x = x1.get("shape")
    shape_y = x2.get("shape")
    dtype_x = x1.get("dtype")
    dtype_y = x2.get("dtype")

    check_shape(shape_x, param_name="x1")
    check_shape(shape_y, param_name="x2")

    check_tuple = ("int16", "uint16", "int32")
    input_data_type = dtype_x.lower()
    check_dtype(input_data_type, check_tuple, param_name="x1")

    if dtype_x != dtype_y:
        raise RuntimeError("The type of input must be the same")

    shape_x, shape_y, shape_max = broadcast_shapes(shape_x,
                                                   shape_y,
                                                   param_name_input1="x1",
                                                   param_name_input2="x2")
    shape_x, shape_y = refine_shapes_for_broadcast(shape_x, shape_y)

    if input_data_type == "int32":
        input_data_type = "int16"
        shape_x.append(2)
        shape_y.append(2)

    data_x = tvm.placeholder(shape_x, dtype=input_data_type, name="data_x")
    data_y = tvm.placeholder(shape_y, dtype=input_data_type, name="data_y")
    res = bitwise_or_compute((data_x, data_y), shape_x, shape_y)
    y = {'shape': res.shape, 'dtype': input_data_type}

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": (data_x, data_y, res)}

    te.lang.cce.cce_build_code(schedule, config)
