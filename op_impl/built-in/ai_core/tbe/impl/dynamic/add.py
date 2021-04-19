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
dynamic add
"""
from __future__ import absolute_import

import te.lang.dynamic
from impl.util import fusion_util
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
from te.utils.op_utils import OP_ERROR_CODE_009
from te.utils.op_utils import OP_ERROR_CODE_018
from topi import generic

# General limitation of the reduce size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648
SIZE_SIXTEEN = 16


def scalar2tensor_one(shape):
    """
    if the input_shape is [],convert the input_shape to [1]
    ----------
    shape: shape of input tensor

    Returns
    -------
    list:[1]
    """
    if isinstance(shape, (list, tuple)):
        if not shape:
            return [1]
    return shape


def _can_division_sixteen(shape):
    if shape[-1] == 0 or shape[-2] == 0:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_009
        error_info['op_name'] = 'add'
        error_info['rule_desc'] = "The last two dim shape can not be zero at" \
                                  "the same time."
        error_info['param_name1'] = 'shape[-1]'
        error_info['param_name2'] = 'shape[-2]'
        error_info['param1_value'] = str(shape[-1])
        error_info['param2_value'] = str(shape[-2])
        raise RuntimeError(error_info, "Op[%s] has rule: %s, "
                           "but [%s] is [%s], [%s] is [%s]." % (
                           error_info['op_name'], error_info['rule_desc'],
                           error_info['param_name1'],
                           error_info['param1_value'],
                           error_info['param_name2'],
                           error_info['param2_value']))

    if shape[-1] % SIZE_SIXTEEN == 0 and shape[-2] % SIZE_SIXTEEN == 0:
        return True

    return False


def _add_check_format(x, y):
    format_pattern = 0
    shape1 = x.get("shape")
    shape2 = y.get("shape")
    list_format = [x.get("format"), y.get("format")]
    shape1 = scalar2tensor_one(shape1)
    shape2 = scalar2tensor_one(shape2)
    check_list = [["FRACTAL_NZ", "ND"], ["ND", "FRACTAL_NZ"]]
    if list_format == check_list[0] and (
            len(shape2) != 1 or (len(shape2) == 1 and shape2[0] != 1)):
        format_pattern = 1
    elif list_format == check_list[1] and (
            len(shape1) != 1 or (len(shape1) == 1 and shape1[0] != 1)):
        format_pattern = 2

    return format_pattern


def _infer_shape(format_pattern, x, y):
    shape_x = x.get("shape")
    shape_y = y.get("shape")
    shape_x = scalar2tensor_one(shape_x)
    shape_y = scalar2tensor_one(shape_y)

    if format_pattern == 1:
        shape_x, shape_y, shape_max = broadcast_shapes(shape_x, shape_y,
                                                       param_name_input1="input_x",
                                                       param_name_input2="input_y")

        if shape_y[-2] == 1 and shape_y[-1] == shape_x[-1]:
            shape_y.append(1)
            shape_y.append(1)
            shape_y[-3] = 1
            shape_y[-1] = shape_x[-1]
            shape_y[-4] = shape_x[-4]

        elif shape_y[-2] == shape_x[-2] and shape_y[-1] == 1:
            shape_y.append(1)
            shape_y.append(1)
            shape_y[-4] = 1
            shape_y[-2] = shape_x[-2]
            shape_y[-3] = shape_x[-3]

        elif shape_y[-2] == shape_y[-1] == 1:
            shape_y.append(1)
            shape_y.append(1)

    elif format_pattern == 2:
        shape_x, shape_y, shape_max = broadcast_shapes(shape_x, shape_y,
                                                       param_name_input1="input_x",
                                                       param_name_input2="input_y")
        if shape_x[-2] == 1 and shape_x[-1] == shape_y[-1]:
            shape_x.append(1)
            shape_x.append(1)
            shape_x[-3] = 1
            shape_x[-1] = shape_y[-1]
            shape_x[-4] = shape_y[-4]

        elif shape_x[-2] == shape_y[-2] and shape_x[-1] == 1:
            shape_x.append(1)
            shape_x.append(1)
            shape_x[-4] = 1
            shape_x[-2] = shape_y[-2]
            shape_x[-3] = shape_y[-3]

        elif shape_x[-2] == shape_x[-1] == 1:
            shape_x.append(1)
            shape_x.append(1)

    return shape_x, shape_y


@te.op.register_fusion_compute("Add")
def add_fusion_compute(input_x, input_y, output_z, kernel_name="add"):
    fusion_util.check_fusion_input([input_x, input_y])

    dict_x = fusion_util.extract_dict(input_x)
    dict_y = fusion_util.extract_dict(input_y)
    shape_x, shape_y = fusion_util.normalize_shape([dict_x, dict_y])
    ph_x = fusion_util.create_placeholder(input_x, shape_x)
    ph_y = fusion_util.create_placeholder(input_y, shape_y)

    res = add_compute(ph_x, ph_y, output_z, kernel_name)

    return {"op_placeholder": [ph_x, ph_y], "op_res": [res]}


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
def add_compute(input_x, input_y, output_z, kernel_name="add"):
    """
    calculating data's add, c = a + b

    Parameters
    ----------
    input_x:
    left input, may be dict or tensor

    input_y:
    left input, may be dict or tensor

    output_z: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is add

    Returns
    -------
    res : output of the data's add
    """
    shape_x = te.lang.dynamic.shape_to_list(input_x.shape)
    shape_y = te.lang.dynamic.shape_to_list(input_y.shape)

    shape_x, shape_y, shape_max = broadcast_shapes(shape_x, shape_y,
                                                   param_name_input1="input_x",
                                                   param_name_input2="input_y")

    input_x = te.lang.dynamic.broadcast(input_x, shape_max)
    input_y = te.lang.dynamic.broadcast(input_y, shape_max)
    res = te.lang.dynamic.vadd(input_x, input_y)

    return res


@te.op.register_operator("Add")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def add(input_x, input_y, output_z, kernel_name="add"):
    """
    algorithm: add
    calculating data's add, c = a + b

    Parameters
    ----------
    input_x : dict
       including shape, dtype and range, only support float16, float32, int32
    input_y : dict
       including shape, dtype and range, only support float16, float32, int32
    output_z: dict
       shape should be broadcast shape of input, and type equals to input
    kernel_name : str
       cce kernel name, default value is add

    Returns
    -------
    None
    """

    # check input tensor data_type
    x_dtype = input_x.get("dtype").lower()
    y_dtype = input_y.get("dtype").lower()
    check_list = ("float16", "float32", "int32")
    check_dtype(x_dtype, check_list, param_name="input_x")
    check_dtype(y_dtype, check_list, param_name="input_y")
    check_elewise_shape_range([input_x, input_y], support_broadcast=True)
    if x_dtype != y_dtype:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_018
        error_info['op_name'] = 'add'
        error_info['param_name1'] = 'x_dtype'
        error_info['param_name2'] = 'y_dtype'
        error_info['param1_dtype'] = str(x_dtype)
        error_info['param2_dtype'] = str(y_dtype)
        raise RuntimeError(error_info,
                           "In op[%s], the parameter[%s][%s] are not equal in"
                           "dtype with dtype[%s][%s]" % (error_info['op_name'],
                                                         error_info[
                                                             'param_name1'],
                                                         error_info[
                                                             'param_name2'],
                                                         error_info[
                                                             'param1_dtype'],
                                                         error_info[
                                                             'param2_dtype']))

    # format_pattern = 1  Nz and vector
    # format_pattern = 2  vector and Nz
    # format_pattern = 0  Nz scalar  Nz Nz  ND ND
    format_pattern = _add_check_format(input_x, input_y)

    # infer shape for supporting add
    shape_x, shape_y = _infer_shape(format_pattern, input_x, input_y)
    shape_x = scalar2tensor_one(shape_x)
    shape_y = scalar2tensor_one(shape_y)

    # normalize shape
    input_x["shape"] = shape_x
    input_y["shape"] = shape_y

    ins = classify([input_x, input_y], Mode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (input_x, input_y) in ins:
        with te.op.compute():
            shape_x, shape_y = variable_shape([input_x, input_y],
                                              support_broadcast=True)
            shape_x, shape_y = refine_shapes_for_broadcast(shape_x, shape_y)
            data_x = tvm.placeholder(shape_x, name="data_1", dtype=x_dtype)
            data_y = tvm.placeholder(shape_y, name="data_2", dtype=y_dtype)
            res = add_compute(data_x, data_y, output_z, kernel_name)

            tensors.append((data_x, data_y, res))
        with tvm.target.cce():
            schedule = generic.auto_schedule(res)
        schedules.append(schedule)

    config = {"print_ir": False, "name": kernel_name,
              "tensor_list": tensors}

    te.lang.dynamic.build(schedules, config)
