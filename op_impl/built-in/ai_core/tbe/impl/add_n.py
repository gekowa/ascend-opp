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
add_n
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from functools import reduce as reduceIns
from te.platform.cce_conf import api_check_support
from te.utils.op_utils import *

SHAPE_SIZE_LIMIT = 2147483648  # General limitation of the reduce size for input shape: 2**31


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("add_n")
def add_n_compute_for_fusion(datas, output, tensor_num, kernel_name="add_n"):
    """
    calculating data's adds, z = a + b + c...

    Parameters
    ----------
    datas : list of placeholders
        all input data
    output : dict
        dict of output
    tensor_num:
        nums of input
    kernel_name : string
        cce kernel name, default value is add_n

    Returns
    -------
    res : output of the data's add_n
    """
    res = datas[0]
    for i, data_n in enumerate(datas):
        if i == 0:
            continue
        res = te.lang.cce.vadd(res, data_n)

    return res


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
def add_n_compute(datas, output, tensor_num, kernel_name="add_n"):
    """
    calculating data's adds, z = a + b + c...

    Parameters
    ----------
    datas : list of placeholders
        all input data
    output : dict
        dict of output
    tensor_num:
        nums of input
    kernel_name : string
        cce kernel name, default value is add_n

    Returns
    -------
    res : output of the data's add_n
    """

    data_type = datas[0].dtype
    has_covert_float32 = (data_type == "float16" and
                          api_check_support("te.lang.cce.vadd", "float32"))

    first_data = datas[0] if not has_covert_float32 else\
        te.lang.cce.cast_to(datas[0], "float32")

    res = first_data
    for i, data_n in enumerate(datas):
        if i == 0:
            continue
        temp_data = data_n if not has_covert_float32 else \
            te.lang.cce.cast_to(data_n, "float32")
        res = te.lang.cce.vadd(res, temp_data)

    if has_covert_float32:
        res = te.lang.cce.cast_to(res, "float16")
    return res


@check_op_params(DYNAMIC_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_INT, KERNEL_NAME)
def add_n(inputs, output, tensor_num, kernel_name="add_n"):
    """
    algorithm: add_n
    calculating data's adds, z = a + b + c...

    Parameters
    ----------
    inputs : list or tuple of dict
        A list of Tensor objects, each with same shape and type.
    output : dict
        dict of output
    tensor_num:
        nums of input
    kernel_name : string
        cce kernel name, default value is add_n

    Returns
    -------
    None
    """

    input_num = len(inputs)
    if input_num < 2:
        errorInfo = {}
        errorInfo['errCode'] = 'E80002'
        errorInfo['param_name'] = 'inputs'
        errorInfo['excepted_value'] = '1'
        errorInfo['actual_value'] = input_num
        raise RuntimeError(errorInfo, "the parameter[%s] should be more than [%s], but actually is [%s]."
                           % (errorInfo['param_name'], errorInfo['excepted_value'], errorInfo['actual_value']))

    if input_num != tensor_num:
        errorInfo = {}
        errorInfo['errCode'] = 'E80000'
        errorInfo['param_name'] = 'inputs'
        errorInfo['excepted_value'] = tensor_num
        errorInfo['actual_value'] = input_num
        raise RuntimeError(errorInfo, "the parameter[%s] should be [%s], but actually is [%s]."
                           % (errorInfo['param_name'], errorInfo['excepted_value'], errorInfo['actual_value']))

    shape_0 = inputs[0].get("shape")
    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x*y, shape_0)

    check_list = ("float16", "float32", "int32")
    data = []
    for i, input_dict in enumerate(inputs):
        shape_input = input_dict.get("shape")
        if list(shape_0) != list(shape_input):
            errorInfo = {}
            errorInfo['errCode'] = 'E80012'
            errorInfo['param_name'] = 'shape_input'
            errorInfo['excepted_value'] = list(shape_input)
            errorInfo['actual_value'] = list(shape_0)
            raise RuntimeError(errorInfo, "the num of dimensions of input[%s] should be [%s], but actually is [%s]."
                               % (errorInfo['param_name'], errorInfo['excepted_value'], errorInfo['actual_value']))
        check_shape(shape_input, param_name="inputs")
        dtype_input = input_dict.get("dtype").lower()
        check_dtype(dtype_input, check_list, param_name="inputs")
        data.append(tvm.placeholder(fuseshape,
                                    name="data_%d" % i,
                                    dtype=dtype_input))

    res = add_n_compute(data, output, tensor_num, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    data.append(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": data}

    te.lang.cce.cce_build_code(schedule, config)
