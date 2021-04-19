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
sigmoid
"""
from functools import reduce as reduceIns

import te.lang.cce
from te import platform as tbe_platform
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils import op_utils
from topi import generic


# pylint: disable=locally-disabled,unused-argument,too-many-locals
@fusion_manager.register("sigmoid")
def sigmoid_compute(x, y, kernel_name="sigmoid"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of x
    y : dict
        dict of y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "sigmoid"

    Returns
    -------
    output tensor
    """
    data_input = x
    dtype = x.dtype
    exp_support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.vexp", "float32")
    mul_support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.vmuls", "float32")
    if dtype == "float32" and not mul_support:
        error_info = {}
        error_info['errCode'] = 'E80008'
        error_info['param_name'] = 'x'
        error_info['op_name'] = 'sigmoid'
        error_info['expect_value'] = "float16"
        error_info['real_value'] = dtype
        raise RuntimeError(
            error_info, "In op[%s], the parameter[%s]'s dtype "
            "should be [%s], but actually is [%s]." %
            (error_info['op_name'], error_info['param_name'],
             error_info['expect_value'], error_info['real_value']))

    const_num_neg_one = tvm.const(-1, dtype=dtype)
    const_num_one = tvm.const(1, dtype=dtype)
    tmp_negative = te.lang.cce.vmuls(data_input, const_num_neg_one)
    if dtype == "float32" and not exp_support:
        tmp_negative = te.lang.cce.cast_to(tmp_negative, "float16")
    tmp_exp = te.lang.cce.vexp(tmp_negative)
    if dtype == "float32" and not exp_support:
        tmp_exp = te.lang.cce.cast_to(tmp_exp, "float32")
    tmp_sum = te.lang.cce.vadds(tmp_exp, const_num_one)
    if dtype == "float32":
        inp_shape = tmp_sum.shape
        tensor_one = te.lang.cce.broadcast(tvm.const(1, dtype), inp_shape)
        tmp_rec = te.lang.cce.vdiv(tensor_one, tmp_sum)
    else:
        tmp_rec = te.lang.cce.vrec(tmp_sum)
    return tmp_rec


@op_utils.check_op_params(op_utils.REQUIRED_INPUT, op_utils.REQUIRED_OUTPUT,
                          op_utils.KERNEL_NAME)
def sigmoid(x, y, kernel_name="sigmoid"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        dict of x, include keys(shape and dtype)
    y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "sigmoid"

    Returns
    -------
    None
    """
    shape = x.get("shape")
    dtype = x.get("dtype")
    op_utils.check_shape(shape, param_name="x")
    input_dtype = dtype.lower()
    check_list = ("float16", "float32")
    op_utils.check_dtype(dtype, check_list, param_name="x")

    fused_shape = [reduceIns(lambda a, b: a * b, shape[:])]
    data_input = tvm.placeholder(fused_shape,
                                 name="data_input",
                                 dtype=input_dtype)

    res = sigmoid_compute(data_input, y, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_input, res]}
    te.lang.cce.cce_build_code(sch, config)
