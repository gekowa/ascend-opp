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
dynamic relu
"""
from __future__ import absolute_import

import te.lang.dynamic
from te import tvm
from te import platform as tbe_platform
from functools import reduce as reduceIns
from te.platform.shape_classifier import classify
from te.platform.shape_classifier import Mode
from te.utils.op_utils import check_op_params
from te.utils.op_utils import check_dtype
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import variable_shape
from impl.util import fusion_util
from topi import generic

# const value
CONST_ZERO = 0


@te.op.register_fusion_compute("Relu")
def relu_fusion_compute(x, y, kernel_name="relu"):
    fusion_util.check_fusion_input([x])

    dict_x = fusion_util.extract_dict(x)
    shape_x = fusion_util.normalize_shape([dict_x])[0]
    ph_x = fusion_util.create_placeholder(x, shape_x)

    res = relu_compute(ph_x, y, kernel_name)

    return {"op_placeholder": [ph_x], "op_res": [res]}


def relu_compute(x, y, kernel_name="relu"):
    """
    Algrithm : relu(x) = max(x, 0)

    Parameters
    ----------
    x: the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name

    Returns
    -------
    res : result of relu
    """
    inp_dtype = x.dtype
    shape = x.shape
    compatible_dtype = x.dtype

    if inp_dtype == 'int8' and tbe_platform.cce_conf.api_check_support(
            'te.lang.dynamic.cast_to', 's82f16'):
        x = te.lang.dynamic.cast_to(x, 'float16')
        compatible_dtype = 'float16'
    if tbe_platform.cce_conf.api_check_support('te.lang.dynamic.vrelu',
                                               compatible_dtype):
        data_res = te.lang.dynamic.vrelu(x)
    else:
        tensor_zero = te.lang.dynamic.broadcast(
            tvm.const(CONST_ZERO, compatible_dtype), shape)
        data_res = te.lang.dynamic.vmax(x, tensor_zero)

    data_res = te.lang.dynamic.cast_to(data_res, inp_dtype)

    return data_res


@te.op.register_operator("Relu")
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def relu(x, y, kernel_name="relu"):
    """
    Algrithm: relu(x) = max(x, 0)

    Parameters
    ----------
    Algorithm: relu

    Parameters:

    x: dynamic input, include shape, dtype and range

    y: the dict of output

    kernel_name: kernel name, must be string, default value is "relu".

    Returns
    -------
    None
    """

    # check input tensor data_type
    dtype_x = x.get("dtype").lower()
    check_list = ("float16", "float32", "int8", "int32")
    check_dtype(dtype_x, check_list, param_name="x")

    ins = classify([x], Mode.ELEWISE)
    schedules, tensors = [], []
    for (x,) in ins:
        with te.op.compute():
            shape_x = variable_shape([x])

            fuse_shape = [1]
            fuse_shape[0] = reduceIns(lambda x, y: x * y, shape_x[0])

            input_data = tvm.placeholder(fuse_shape, name="input_data",
                                         dtype=dtype_x)
            res = relu_compute(input_data, y, kernel_name)

            tensors.append([input_data, res])
        with tvm.target.cce():
            sch = generic.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}

    te.lang.dynamic.build(schedules, config)
