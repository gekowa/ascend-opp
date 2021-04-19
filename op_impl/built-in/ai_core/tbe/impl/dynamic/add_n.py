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
dynamic add_n
"""
from functools import reduce as reduceIns
from te.platform.cce_conf import api_check_support

import te.lang.dynamic
from te import tvm
from te.platform.shape_classifier import classify
from te.platform.shape_classifier import Mode
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import DYNAMIC_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import REQUIRED_ATTR_INT
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_shape
from te.utils.op_utils import check_op_params
from te.utils.op_utils import variable_shape
from te.utils.op_utils import OP_ERROR_CODE_012
from te.utils.op_utils import OP_ERROR_CODE_017
from te.utils.op_utils import OP_ERROR_CODE_018
from topi import generic


def add_n_compute(datas, output, tensor_num, kernel_name="add_n"):
    """
    calculating data's adds, z = a + b + c...

    Parameters
    ----------
    datas : list of placeholders, all input data
    output : dict, dict of output
    tensor_num: nums of input
    kernel_name : string
        cce kernel name, default value is add_n

    Returns
    -------
    res : output of the data's add_n
    """
    data_type = datas[0].dtype
    has_covert_float32 = (data_type == "float16" and api_check_support(
        "te.lang.dynamic.vadd", "float32"))

    first_data = datas[
        0] if not has_covert_float32 else te.lang.dynamic.cast_to(datas[0],
                                                                  "float32")
    res = first_data

    for i, data_i in enumerate(datas):
        if i == 0:
            continue
        tmp_data = data_i if not has_covert_float32 else \
            te.lang.dynamic.cast_to(data_i, "float32")
        res = te.lang.dynamic.vadd(res, tmp_data)

    if has_covert_float32:
        res = te.lang.dynamic.cast_to(res, "float16")

    return res


@te.op.register_operator("AddN")
@check_op_params(DYNAMIC_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_INT,
                 KERNEL_NAME)
def add_n(inputs, output, tensor_num, kernel_name="add_n"):
    """
    algorithm: add_n
    calculating data's adds, z = a + b + c...

    Parameters
    ----------
    inputs : list or tuple of dict
        A list of Tensor objects,
        each with same shape, range and dtype of first input,
        only support float16, float32, int32.
    output : dict
        shape, range and dtype of output,
        should be broadcast shape and type as input.
    tensor_num:
        nums of input
    kernel_name : string
        cce kernel name, default value is add_n

    Returns
    -------
    None
    """
    # check inputs num
    input_num = len(inputs)
    if input_num < 2:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_012
        error_info['op_name'] = 'add_n'
        error_info['param_name'] = 'input_num'
        error_info['max_value'] = '8'
        error_info['min_value'] = '2'
        error_info['real_value'] = str(input_num)
        raise RuntimeError(error_info,
                           "In op[%s], the num of dimensions of input[%s] "
                           "should be in the range of [%s, %s], but actually "
                           "is [%s]." % (
                               error_info['op_name'], error_info['param_name'],
                               error_info['min_value'],
                               error_info['max_value'],
                               error_info['real_value']))
    if input_num != tensor_num:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_017
        error_info['op_name'] = 'add_n'
        error_info['param_name1'] = 'input_num'
        error_info['param_name2'] = 'tensor_num'
        error_info['param1_shape'] = str(input_num)
        error_info['param2_shape'] = str(tensor_num)
        raise RuntimeError(error_info,
                           "In op[%s], the parameter[%s][%s] is not match with"
                           "the parameter[%s][%s],it should be the same." % (
                               error_info['op_name'],
                               error_info['param_name1'],
                               error_info['param1_shape'],
                               error_info['param_name2'],
                               error_info['param2_shape']))

    dtype_0 = inputs[0].get("dtype").lower()
    for index in range(0, tensor_num):
        shape_input = inputs[index].get("shape")
        check_shape(shape_input, param_name="inputs")
        dtype_input = inputs[index].get("dtype").lower()
        check_list = ("float16", "float32", "int32")
        check_dtype(dtype_input, check_list, param_name="inputs")
        if dtype_input != dtype_0:
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_018
            error_info['op_name'] = 'add_n'
            error_info['param_name1'] = 'dtype_input'
            error_info['param_name2'] = 'dtype_0'
            error_info['param1_dtype'] = str(dtype_input)
            error_info['param2_dtype'] = str(dtype_0)
            raise RuntimeError(error_info, "In op[%s], the parameter"
                                           "[%s][%s] are not equal in "
                                           "dtype with dtype[%s][%s]." % (
                                   error_info['op_name'],
                                   error_info['param_name1'],
                                   error_info['param_name2'],
                                   error_info['param1_dtype'],
                                   error_info['param2_dtype']))

    ins = classify(inputs, Mode.ELEWISE)
    schedules, tensors = [], []
    for inputs in ins:
        with te.op.compute():
            shape_normlize = variable_shape(inputs)
            fuse_shape = [1]
            datas = []
            for (i, input_dict), shape_i in zip(enumerate(inputs),
                                                shape_normlize):
                fuse_shape[0] = reduceIns(lambda x, y: x * y, shape_i)
                datas.append(tvm.placeholder(fuse_shape, name="data_%d" % i,
                                             dtype=dtype_0))

            # add_n_compute
            res = add_n_compute(datas, output, kernel_name)

            tensors.append(datas)
        with tvm.target.cce():
            sch = generic.auto_schedule(res)
        schedules.append(sch)

    # build
    datas.append(res)
    config = {"name": kernel_name, "tensor_list": tensors}
    te.lang.dynamic.build(schedules, config)
