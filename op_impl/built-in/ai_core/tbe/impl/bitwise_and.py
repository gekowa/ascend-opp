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
bitwise_and
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import refine_shapes_for_broadcast
from te.utils.op_utils import *

# pylint: disable=unused-argument,invalid-name
@fusion_manager.register("bitwise_and")
def bitwise_and_compute(x1, x2, y, kernel_name="bitwise_and"):
    """
    calculating data's bitwise and
    res = x & y

    Parameters
    ----------
    x1 : tvm tensor
              input data x1
    x2 : tvm tensor
              input data x2
    y : dict
               the shape and dtype of the tensor y
    kernel_name : string
                  cce kernel name, default value is "bitwise_and"

    Returns
    -------
    res : output of the data's bitwise and
    """
    shape_x = te.lang.cce.util.shape_to_list(x1.shape)
    shape_y = te.lang.cce.util.shape_to_list(x2.shape)
    shape_x, shape_y, shape_max = broadcast_shapes(shape_x,
                                                   shape_y,
                                                   param_name_input1="x1",
                                                   param_name_input2="x2")

    data_x = te.lang.cce.broadcast(x1, shape_max)
    data_y = te.lang.cce.broadcast(x2, shape_max)

    res = te.lang.cce.vand(data_x, data_y)

    return res


def _check_parameters(x1, x2, y, kernel_name):
    """
    check the input parameters
    return the shape and data type of x1 and x2
    """

    shape_x = x1.get("shape")
    shape_y = x2.get("shape")
    dtype_x = x1.get("dtype").lower()
    dtype_y = x2.get("dtype").lower()
    dtype_z = y.get("dtype").lower()

    check_shape(shape_x, param_name="x1")
    check_shape(shape_y, param_name="x2")

    check_tuple = ("int16", "uint16", "int32")
    check_dtype(dtype_x, check_tuple, param_name="x1")
    check_dtype(dtype_y, check_tuple, param_name="x2")
    check_dtype(dtype_z, check_tuple, param_name="y")
    if dtype_x != dtype_y:
        raise RuntimeError(
            "two input type must be the same")

    return shape_x, shape_y, dtype_x


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def bitwise_and(x1, x2, y, kernel_name="bitwise_and"):
    """
    algorithm: bitwise_and
    computes the bitwise and of `x1` and `x2`

    Parameters
    ----------
    x1 : dict
              the shape and dtype of the tensor x1, only support int16,uint16
    x2 : dict
              the shape and dtype of the tensor x2, only support int16,uint16
    y : dict
              the shape and dtype of the tensor y, only support int16,uint16
    kernel_name : string
                  cce kernel name, default value is "bitwise_and"

    Returns
    -------
    None
    """
    shape_x, shape_y, dtype = _check_parameters(x1, x2, y, kernel_name)
    shape_x, shape_y, shape_max = broadcast_shapes(shape_x,
                                                   shape_y,
                                                   param_name_input1="x1",
                                                   param_name_input2="x2")
    shape_x, shape_y = refine_shapes_for_broadcast(shape_x, shape_y)

    if dtype == "int32":
        dtype = "int16"
        shape_x.append(2)
        shape_y.append(2)

    data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype)
    data_y = tvm.placeholder(shape_y, name="data_y", dtype=dtype)

    res = bitwise_and_compute(data_x, data_y, y, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": (data_x, data_y, res)}
    te.lang.cce.cce_build_code(schedule, config)
