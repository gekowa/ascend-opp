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
dynamic div
"""
import te.lang.dynamic
from te import platform as tbe_platform
from te import tvm
from te.platform.shape_classifier import classify
from te.platform.shape_classifier import Mode
from topi import generic
from te.utils.op_utils import check_op_params
from te.utils.op_utils import check_dtype
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import check_elewise_shape_range
from te.utils.op_utils import variable_shape
from te.utils.op_utils import refine_shapes_for_broadcast
from te.utils.op_utils import broadcast_shapes
from te.utils.op_utils import OP_ERROR_CODE_018


def div_compute(input_x, input_y, output_z, kernel_name="div"):
    """
    div compute
    calculating data's div, res =x / y

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    input_y: TVM tensor
        the placeholder of input_y
    output_div: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "div"

    Returns
    -------
    res: TVM tensor
        the result of div compute
    """
    x_shape = te.lang.dynamic.shape_to_list(input_x.shape)
    y_shape = te.lang.dynamic.shape_to_list(input_y.shape)
    x_shape, y_shape, z_shape = broadcast_shapes(x_shape, y_shape,
                                                 param_name_input1="input_x",
                                                 param_name_input2="input_y")
    dtype_x = input_x.dtype
    int_list = ("int8", "uint8", "int32")
    if tbe_platform.cce_conf.api_check_support("te.lang.dynamic.vdiv",
                                               "float32"):
        input_x = te.lang.dynamic.cast_to(input_x, "float32")
        input_y = te.lang.dynamic.cast_to(input_y, "float32")
    input_x = te.lang.dynamic.broadcast(input_x, z_shape)
    input_y = te.lang.dynamic.broadcast(input_y, z_shape)
    res = te.lang.dynamic.vdiv(input_x, input_y)

    if dtype_x in int_list:
        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") == "Ascend310":
            res = te.lang.dynamic.cast_to(res, "float16")
        res = te.lang.dynamic.floor(res)

    res = te.lang.dynamic.cast_to(res, dtype_x)

    return res


@te.op.register_operator("Div")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def div(input_x, input_y, output_z, kernel_name="div"):
    """
    algorithm: div
    calculating data's div, res =x / yq


    Parameters
    ----------
    input_x: dict
        dict with keys(shape and dtype) of input_x
    input_y: dict
        dict with keys(shape and dtype) of input_y
    output_div: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "div"

    Returns
    -------
    None
    """

    # check dtype
    x_dtype = input_x.get("dtype").lower()
    y_dtype = input_y.get("dtype").lower()
    check_list = ("float16", "float32", "int8", "uint8", "int32")
    check_dtype(x_dtype, check_list, param_name="input_x")
    check_dtype(y_dtype, check_list, param_name="input_y")
    check_elewise_shape_range([input_x, input_y], support_broadcast=True)

    if x_dtype != y_dtype:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_018
        error_info['op_name'] = 'div'
        error_info['param_name1'] = 'x_dtype'
        error_info['param_name2'] = 'y_dtype'
        error_info['param1_dtype'] = str(x_dtype)
        error_info['param2_dtype'] = str(y_dtype)
        raise RuntimeError(error_info,
                           "In op[%s], the parameter[%s][%s] are not equal in "
                           "dtype with dtype[%s][%s]." % (
                               error_info['op_name'],
                               error_info['param_name1'],
                               error_info['param_name2'],
                               error_info['param1_dtype'],
                               error_info['param2_dtype']))

    ins = classify([input_x, input_y], Mode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (input_x, input_y) in ins:
        with te.op.compute():
            x_shape, y_shape = variable_shape([input_x, input_y],
                                              support_broadcast=True)
            x_shape, y_shape = refine_shapes_for_broadcast(x_shape, y_shape)
            tensor_x = tvm.placeholder(x_shape, x_dtype, "tensor_x")
            tensor_y = tvm.placeholder(y_shape, y_dtype, "tensor_y")
            res = div_compute(tensor_x, tensor_y, output_z, kernel_name)

            tensors.append([tensor_x, tensor_y, res])
        with tvm.target.cce():
            sch = generic.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    te.lang.dynamic.build(schedules, config)
