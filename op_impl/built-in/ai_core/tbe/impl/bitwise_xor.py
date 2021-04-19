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
bitwise_xor
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import refine_shapes_for_broadcast
from te.utils.op_utils import *

# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=invalid-name,too-many-locals
@fusion_manager.register("bitwise_xor")
def bitwise_xor_compute(x1, x2, y, kernel_name="bitwise_xor"):
    """
    calculating data's bitwise xor
    (x&y)|!(x|y)

    Parameters
    ----------
    x1 : tvm tensor
              input data x
    x2 : tvm tensor
              input data y
    y : dict
               the shape and dtype of the tensor
    kernel_name : string
                  cce kernel name, default value is "bitwise_and"

    Returns
    -------
    result : y of the data's bitwise xor
    """
    shape_x = te.lang.cce.util.shape_to_list(x1.shape)
    shape_y = te.lang.cce.util.shape_to_list(x2.shape)
    shape_x, shape_y, shape_max = broadcast_shapes(shape_x,
                                                   shape_y,
                                                   param_name_input1="x1",
                                                   param_name_input2="x2")

    data_x = te.lang.cce.broadcast(x1, shape_max)
    data_y = te.lang.cce.broadcast(x2, shape_max)

    data_and = te.lang.cce.vand(data_x, data_y)
    data_not = te.lang.cce.vnot(data_and)
    data_or = te.lang.cce.vor(data_x, data_y)
    result = te.lang.cce.vand(data_or, data_not)

    return result


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def bitwise_xor(x1, x2, y, kernel_name="bitwise_xor"):
    """
    algorithm: bitwise_xor
    calculating: gradient of bitwise_xor

    Parameters
    ----------
    x1 : dict
              the shape and dtype of the tensor x1
    x2 : dict
              the shape and dtype of the tensor x2
    y :  dict
              the shape and dtype of the tensor y
    kernel_name : string
                  cce kernel name, default value is "bitwise_xor"
    Returns
    -------
    None
    """
    shape_x = x1.get("shape")
    shape_y = x2.get("shape")
    dtype_x = x1.get("dtype").lower()
    dtype_y = x2.get("dtype").lower()

    check_shape(shape_x, param_name="x1")
    check_shape(shape_y, param_name="x2")

    check_tuple = ("int16", "uint16", "int32")
    input_data_type = dtype_x.lower()
    check_dtype(input_data_type, check_tuple, param_name="x1")

    if dtype_x != dtype_y:
        raise RuntimeError("two input type must be the same")

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

    result = bitwise_xor_compute(data_x, data_y, y, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(result)

    config = {
        "name": kernel_name,
        "tensor_list": [data_x, data_y, result]}
    te.lang.cce.cce_build_code(sch, config)
