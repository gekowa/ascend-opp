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
elu
  Op_description :
    do element-wise elu operation.

    # elu(
    #   x,
    #   y,
    #   kernel_name='cce_elu')

  Supportive_dtype_format :
    ["float16", "float32"]
    ['ND', 'NCHW', 'NHWC', 'NC1HWC0']

  Constraint :
    [1] All : shape size limit is 2147483648

"""
from functools import reduce as reduceIns
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
import topi
from topi.cce import util
from te import platform as tbe_platform
from te.utils.op_utils import *

# shape limit 2**31
SHAPE_SIZE_LIMIT = 2147483648
NUM_ZERO = 0.0
NUM_ONE_NEG = -1.0


def _elu_computer_performance(data, alpha, dtype):
    scalar_one_neg = tvm.const(NUM_ONE_NEG, dtype)

    negative_data = te.lang.cce.vmuls(data, scalar_one_neg)
    negative_data = te.lang.cce.vrelu(negative_data)
    negative_data = te.lang.cce.vmuls(negative_data, scalar_one_neg)
    positive_data = te.lang.cce.vrelu(data)

    exp_res = te.lang.cce.vexp(negative_data)
    exp_res = te.lang.cce.vadds(exp_res, scalar_one_neg)

    res = te.lang.cce.vmuls(exp_res, tvm.const(alpha, dtype))
    res = te.lang.cce.vadd(positive_data, res)

    return res


def _elu_computer_precision(data, alpha, dtype):
    scalar_zero = tvm.const(NUM_ZERO, dtype)
    negative_data = te.lang.cce.vmins(data, scalar_zero)
    positive_data = te.lang.cce.vmaxs(data, scalar_zero)

    exp_res = te.lang.cce.vexp(negative_data)
    exp_res = te.lang.cce.vadds(exp_res, tvm.const(NUM_ONE_NEG, dtype))

    res = te.lang.cce.vaxpy(exp_res, positive_data, tvm.const(alpha, dtype))

    return res

# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@fusion_manager.register("elu")
def elu_compute(x, y, alpha, kernel_name="elu"):
    """
    do element-wise elu compute
    f(x) = max(min(alpha(e^x - 1), 0), x),  in cloud scene, for all inputs
    f(x) = max(min(alpha(e^x - 1), 0), x),  in mini scene, for x <= TAYLOR_THRESHOLD or x >= 0
    f(x) = fifth taylor computer,    in mini scene, for TAYLOR_THRESHOLD < x < 0

    Parameters:
    ----------
    x: the placeholder of data input

    alpha: float, coefficient when input tensor is less than zero

    y: the dict of output

    kernel_name : cce kernel name, default value is "elu"

    Returns : A Tensor. Has the same type as data_input.
    -------
    """

    data = x
    dtype = data.dtype

    has_improve_precision = False
    if tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp", "float32"):
        has_improve_precision = True
    if dtype.lower() == "float16" and has_improve_precision:
        data = te.lang.cce.cast_to(data, "float32")
        cvt_dtype = "float32"
    else:
        cvt_dtype = dtype

    if has_improve_precision:
        res = _elu_computer_precision(data, alpha, cvt_dtype)
    else:
        res = _elu_computer_performance(data, alpha, cvt_dtype)

    if dtype.lower() == "float16" and has_improve_precision:
        res = te.lang.cce.cast_to(res, "float16")

    return res


# pylint: disable=invalid-name
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, OPTION_ATTR_FLOAT, KERNEL_NAME)
def elu(x, y, alpha=1.0, kernel_name="elu"):
    """
    do element-wise elu operation

    Parameters:
    ----------
    x: the dict of input, only support float16, float32

    alpha: float, coefficient when input tensor is less than zero.

    output_res : the dict of output

    kernel_name : cce kernel name, default value is "elu"

    Returns
    -------
    None
    """

    shape_input = x.get("shape")
    dtype_input = x.get("dtype")
    input_dtype = dtype_input.lower()

    check_shape(shape_input, param_name="x")

    check_list = ("float16", "float32")
    check_dtype(dtype_input, check_list, param_name="x")

    if not tbe_platform.cce_conf.api_check_support("te.lang.cce.sum", "float32") and dtype_input == "float32":
        error_info = {}
        error_info['errCode'] = 'E80008'
        error_info['param_name'] = 'x'
        error_info['op_name'] = 'elu'
        error_info['expect_value'] = "float16"
        error_info['real_value'] = dtype_input
        raise RuntimeError(error_info, "In op[%s], the parameter[%s]'s dtype "
                                       "should be [%s], but actually is [%s]."
                           % (error_info['op_name'], error_info['param_name'], \
                              error_info['expect_value'], error_info['real_value']))

    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x*y, shape_input)
    data_input = tvm.placeholder(fuseshape, name="data_input", dtype=input_dtype)

    res = elu_compute(data_input, y, alpha, kernel_name)

    with tvm.target.cce():
        auto_sch = topi.generic.auto_schedule(res)

    config = {"name": kernel_name,
              "print_ir": False,
              "tensor_list": [data_input, res],
              "bool_storage_as_1bit": False}
    te.lang.cce.cce_build_code(auto_sch, config)
