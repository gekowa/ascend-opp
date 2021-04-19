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
dynamic floordiv
"""
import te.lang.dynamic
from te import platform as tbe_platform
from te import tvm
from te.platform.shape_classifier import classify
from te.platform.shape_classifier import Mode
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_op_params
from te.utils.op_utils import check_elewise_shape_range
from te.utils.op_utils import variable_shape
from te.utils.op_utils import refine_shapes_for_broadcast
from te.utils.op_utils import broadcast_shapes
from te.utils.op_utils import OP_ERROR_CODE_018
from topi import generic


# pylint: disable=locally-disabled,unused-argument
def floor_div_compute(input_x, input_y, output_z, kernel_name='floor_div'):
    """
       floordiv compute
       calculating data's floordiv, res =floor(x / y)

       Parameters
       ----------
       input_x: TVM tensor
           the placeholder of input_x
       input_y: TVM tensor
           the placeholder of input_y
       output_z: dict
           dict of output
       kernel_name: str
           kernel name, default value is "floor_div"

       Returns
       -------
       res: TVM tensor
           the result of floordiv compute
    """
    dtype_x = input_x.dtype
    input_x_shape = te.lang.dynamic.shape_to_list(input_x.shape)
    input_y_shape = te.lang.dynamic.shape_to_list(input_y.shape)
    input_x_shape, input_y_shape, shape_broad = \
        broadcast_shapes(input_x_shape, input_y_shape,
                         param_name_input1="input_x",
                         param_name_input2="input_y")

    if dtype_x != "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.dynamic.vdiv", "float32"):
        input_x = te.lang.dynamic.cast_to(input_x, 'float32')
        input_y = te.lang.dynamic.cast_to(input_y, 'float32')

        input_x = te.lang.dynamic.broadcast(input_x, shape_broad)
        input_y = te.lang.dynamic.broadcast(input_y, shape_broad)
    else:
        input_x = te.lang.dynamic.broadcast(input_x, shape_broad)
        input_y = te.lang.dynamic.broadcast(input_y, shape_broad)

    res = te.lang.dynamic.vdiv(input_x, input_y)

    if dtype_x != "float16" and tbe_platform.cce_conf.get_soc_spec(
            "SOC_VERSION") == "Ascend310":
        res = te.lang.dynamic.cast_to(res, "float16")

    res = te.lang.dynamic.floor(res)

    res = te.lang.dynamic.cast_to(res, dtype_x)

    return res


@te.op.register_operator("FloorDiv")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def floor_div(input_x, input_y, output_z, kernel_name="floor_div"):
    """
      algorithm: floordiv
      calculating data's floordiv, res =floor(x / y)

      Parameters
      ----------
      input_x: dict
      input_y: dict
      output_z: dict
      kernel_name: str, default value is "floor_div"

      Returns
      -------
      None
    """
    # check dtype of input_x/input_y
    input_dtype_x = input_x.get("dtype").lower()
    input_dtype_y = input_y.get("dtype").lower()
    check_list = ('int8', 'uint8', 'int32', 'float16', 'float32')
    check_dtype(input_dtype_x, check_list, param_name="input_x")
    check_dtype(input_dtype_y, check_list, param_name="input_y")
    check_elewise_shape_range([input_x, input_y], support_broadcast=True)
    if input_dtype_x != input_dtype_y:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_018
        error_info['op_name'] = 'floor_div'
        error_info['param_name1'] = 'input_dtype_x'
        error_info['param_name2'] = 'input_dtype_y'
        error_info['param1_dtype'] = str(input_dtype_x)
        error_info['param2_dtype'] = str(input_dtype_y)
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
            tensor_x = tvm.placeholder(x_shape, input_dtype_x, "tensor_x")
            tensor_y = tvm.placeholder(y_shape, input_dtype_y, "tensor_y")
            res = floor_div_compute(tensor_x, tensor_y, output_z, kernel_name)

            tensors.append([tensor_x, tensor_y, res])
        with tvm.target.cce():
            sch = generic.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    te.lang.dynamic.build(schedules, config)
