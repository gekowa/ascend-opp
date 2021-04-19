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
dynamic equal
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

# define a scalar, value = 2**(-126), minimun num of float32 2**(-126)
SCALAR_MIN_FP32 = 2 ** (-126)
# define a scalar, value = 2**(50)
SCALAR_MUL_FP32 = 2 ** 50
# define a scalar, value = 2**(26)
SCALAR_MUL2_FP32 = 2 ** 26
# define a scalar, value = 2**(-24), minimun num of float16 2**(-24)
SCALAR_MIN_FP16 = 2 ** (-24)
# define a scalar, value = 2**(12)
SCALAR_MUL_FP16 = 2 ** 12
# define a scalar, value = 1
SCALAR_ONE = 1


# 'pylint: disable=unused-argument,invalid-name
def equal_compute(input_x, input_y, output_z, kernel_name="equal"):
    """
    compute for equal

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x, has shape, dtype and range attributes
    input_y: TVM tensor
        the placeholder of input_y, has shape, dtype and range attributes
    output_z: dict
        dict info of output_z
    kernel_name: str
        cce kernel name, default value is "equal"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    dtype_x = input_x.dtype
    shape_x = te.lang.dynamic.shape_to_list(input_x.shape)
    shape_y = te.lang.dynamic.shape_to_list(input_y.shape)
    shape_x, shape_y, shape_broad = broadcast_shapes(shape_x, shape_y,
                                                     param_name_input1="input_x",
                                                     param_name_input2="input_y")

    if dtype_x == "float32":
        scalar_min = tvm.const(SCALAR_MIN_FP32, dtype="float32")
        scalar_mul = tvm.const(SCALAR_MUL_FP32, dtype="float32")
        scalar_mul1 = tvm.const(SCALAR_MUL2_FP32, dtype="float32")
        scalar_one = tvm.const(-1 * SCALAR_ONE, dtype="float32")
    else:
        scalar_min = tvm.const(SCALAR_MIN_FP16, dtype="float16")
        scalar_mul = tvm.const(SCALAR_MUL_FP16, dtype="float16")
        scalar_one = tvm.const(-1 * SCALAR_ONE, dtype="float16")

    if dtype_x in ("int8", "uint8"):
        input_x = te.lang.dynamic.cast_to(input_x, "float16")
        input_y = te.lang.dynamic.cast_to(input_y, "float16")

    x_brod = te.lang.dynamic.broadcast(input_x, shape_broad)
    y_brod = te.lang.dynamic.broadcast(input_y, shape_broad)

    res_vsub = te.lang.dynamic.vsub(x_brod, y_brod)
    if tbe_platform.cce_conf.api_check_support("te.lang.dynamic.vabs",
                                               res_vsub.dtype):
        res_vabs = te.lang.dynamic.vabs(res_vsub)
    else:
        res_vsub = te.lang.dynamic.cast_to(res_vsub, "float32")
        res_vabs = te.lang.dynamic.vabs(res_vsub)
    res_min = te.lang.dynamic.vmins(res_vabs, scalar_min)
    res_vmul = te.lang.dynamic.vmuls(res_min, scalar_mul)
    res_vmul1 = te.lang.dynamic.vmuls(res_vmul, scalar_mul)

    if dtype_x == "float32":
        res_vmul2 = te.lang.dynamic.vmuls(res_vmul1, scalar_mul1)
        res_vsub1 = te.lang.dynamic.vadds(res_vmul2, scalar_one)
        res_vabs1 = te.lang.dynamic.vabs(res_vsub1)
    else:
        res_vsub1 = te.lang.dynamic.vadds(res_vmul1, scalar_one)
        res_vabs1 = te.lang.dynamic.vabs(res_vsub1)

    res = te.lang.dynamic.cast_to(res_vabs1, "int8", True)
    return res


@te.op.register_operator("Equal")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def equal(input_x, input_y, output_z, kernel_name="equal"):
    """
    Returns the truth value of (x = y) element-wise

    Parameters
    ----------
    input_x: dict
        dict{"shape":tuple or list,"dtype":str, range: tuple or list}
        shape of data
        the data type, src_dtype equals dst_dtype, support fp16,fp32,int32,uint8,int8
    input_y: dict
        dict{"shape":tuple or list,"dtype":str, range: tuple or list}
        shape of data
        the data type, src_dtype equals dst_dtype, support fp16,fp32,int32,uint8,int8
    output_z: dict, reserved field
        dict with keys(shape and dtype) of output
    kernel_name: str
        cce kernel name, default value is "equal"

    Returns
    -------
    None
    """

    # check input tensor data_type
    x_dtype = input_x.get("dtype").lower()
    y_dtype = input_y.get("dtype").lower()
    check_list = ("float16", "float32", "int32", "uint8", "int8")
    check_dtype(x_dtype, check_list, param_name="input_x")
    check_dtype(y_dtype, check_list, param_name="input_y")
    check_elewise_shape_range([input_x, input_y], support_broadcast=True)
    if x_dtype != y_dtype:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_018
        error_info['op_name'] = 'equal'
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
            res = equal_compute(tensor_x, tensor_y, output_z, kernel_name)
            tensors.append([tensor_x, tensor_y, res])
        with tvm.target.cce():
            sch = generic.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    te.lang.dynamic.build(schedules, config)
