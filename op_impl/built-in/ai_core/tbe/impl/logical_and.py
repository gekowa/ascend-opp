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
logical_and
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import refine_shapes_for_broadcast
from te.utils.op_utils import *

# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=too-many-locals,invalid-name
@fusion_manager.register("logical_and")
def logical_and_compute(x1, x2, y, kernel_name="logical_and"):
    """
    calculating data

    Parameters
    ----------
    x1 : TVM tensor
        the placeholder of x1
    x2: TVM tensor
        the placeholder of x2
    y : dict
        dict of output_y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "logical_and"

    Returns
    -------
    output tensor
    """
    shape_x = te.lang.cce.util.shape_to_list(x1.shape)
    shape_y = te.lang.cce.util.shape_to_list(x2.shape)
    _, _, shape_max = broadcast_shapes(shape_x, shape_y, param_name_input1="x1", param_name_input2="x2")

    x1 = te.lang.cce.cast_to(x1, "float16")
    x2 = te.lang.cce.cast_to(x2, "float16")

    data_x = te.lang.cce.broadcast(x1, shape_max)
    data_y = te.lang.cce.broadcast(x2, shape_max)

    res = te.lang.cce.vmul(data_x, data_y)

    res = te.lang.cce.cast_to(res, "int8", True)

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def logical_and(x1, x2, y, kernel_name="logical_and"):
    """
    calculating data

    Parameters
    ----------
    x1 : dict
        shape and dtype of input, only support float16, float32
    x2 : dict
        shape and dtype of input, only support float16, float32
    y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "logical_and"

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

    if dtype_x != dtype_y:
        raise RuntimeError("The type of input must be the same")

    input_data_type = dtype_x.lower()
    check_tuple = ("int8",)
    check_dtype(input_data_type, check_tuple, param_name="x1")

    shape_x, shape_y, _ = broadcast_shapes(shape_x, shape_y, param_name_input1="x1", param_name_input2="x2")
    shape_x, shape_y = refine_shapes_for_broadcast(shape_x, shape_y)
    data_x = tvm.placeholder(shape_x, dtype=dtype_x, name="data_x")
    data_y = tvm.placeholder(shape_y, dtype=dtype_y, name="data_y")

    res = logical_and_compute(data_x, data_y, y, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": (data_x, data_y, res)}

    te.lang.cce.cce_build_code(sch, config)
