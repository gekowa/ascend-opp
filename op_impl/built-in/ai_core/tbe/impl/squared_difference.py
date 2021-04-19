# Copyright 2020 Huawei Technologies Co., Ltd
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
squared_difference
"""
import te.lang.cce
from te import tvm
from topi import generic

from topi.cce import util
from te.utils.op_utils import refine_shapes_for_broadcast
from te.utils.op_utils import *

SHAPE_SIZE_LIMIT = 2147483648

# pylint: disable=locally-disabled,too-many-locals,invalid-name
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def squared_difference(x1, x2, y, kernel_name="squared_difference"):
    """
    algorithm: squared_difference

    calculating data's tf_squared_difference,y= (x - y) * (x - y)

    Parameters
    ----------
    x2 : dict
        shape and dtype of y input, only support float16, float32
    input_dy : dict
        shape and dtype of dy input, only support float16, float32
    output_x: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is squared_difference

    Returns
    -------
    None
    """
    shape_x = x1.get("shape")
    shape_y = x2.get("shape")
    check_shape(shape_x, param_name="x1")
    check_shape(shape_y, param_name="x2")

    check_list = ["float16", "float32", "int32"]
    dtype = x1.get("dtype").lower()

    if not dtype in check_list:
        raise RuntimeError(
            "tf_squared_difference_cce only support float16, float32, int32")

    shape_x, shape_y, shape_max = broadcast_shapes(shape_x, shape_y, param_name_input1="x1", param_name_input2="x2")

    shape_x, shape_y = refine_shapes_for_broadcast(shape_x, shape_y)
    data_x = tvm.placeholder(shape_x, dtype=dtype, name="data_x")
    data_y = tvm.placeholder(shape_y, dtype=dtype, name="data_y")

    with tvm.target.cce():
        shape_x, shape_y, shape_max = broadcast_shapes(shape_x, shape_y, param_name_input1="x1", param_name_input2="x2")
        data_x_tmp = te.lang.cce.broadcast(data_x, shape_max)
        data_y_tmp = te.lang.cce.broadcast(data_y, shape_max)
        data_sub = te.lang.cce.vsub(data_x_tmp, data_y_tmp)
        res = te.lang.cce.vmul(data_sub, data_sub)
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_x, data_y, res]}

    te.lang.cce.cce_build_code(sch, config)
