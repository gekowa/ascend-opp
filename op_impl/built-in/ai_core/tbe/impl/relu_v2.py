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
relu_v2

  Op_description :
    Algrithm: relu_v2(x) = x and 1 when x > 0 , else 0, 0

    # relu_v2(
    #   x,
    #   y,
    #   mask,
    #   kernel_name='relu_v2')

  Supportive_dtype_format :
    ['float16', 'float32', 'int8', 'int32', 'uint8']
    ['ND', 'NCHW', 'NHWC', 'NC1HWC0']

  Constraint :
    [1] All : the last dim of `x` must be mutiply of 8.
    [2] All : shape size limit is 2147483648.
"""
# noinspection PyInterpreter
from te import tvm
import te.lang.cce
from te.platform.cce_conf import api_check_support
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_shape
from topi import generic
from topi.cce import util
from te.utils.op_utils import *

# const value
CONST_ZERO = 0


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@fusion_manager.register("relu_v2")
def relu_v2_compute(x, y, mask, kernel_name="relu_v2_cce"):
    """
    Algrithm : relu_v2(x) = x and 1 when x > 0 , else 0, 0

    Parameters
    ----------
    x: the placeholder of data input

    y : the dict of output

    mask : the dict of output

    kernel_name : cce kernel name

    Returns
    -------
    res : result of relu_v2_res

    mask: result of relu_v2_mask
    """

    inp_dtype = x.dtype
    shape = x.shape
    compatible_dtype = x.dtype

    if inp_dtype == 'int8' and api_check_support('te.lang.cce.cast_to',
                                                 's82f16'):
        x = te.lang.cce.cast_to(x, 'float16')
        compatible_dtype = 'float16'
    if api_check_support('te.lang.cce.vrelu', compatible_dtype):
        data_res = te.lang.cce.vrelu(x)
    else:
        tensor_zero = te.lang.cce.broadcast(tvm.const(CONST_ZERO,
                                                      compatible_dtype),
                                            shape)
        data_res = te.lang.cce.vmax(x, tensor_zero)

    data_res = te.lang.cce.cast_to(data_res, inp_dtype)
    mask = te.lang.cce.vcmp(x, CONST_ZERO, "gt", "bit")

    return data_res, mask


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def relu_v2(x, y, mask, kernel_name="relu_v2"):
    """
    Algrithm: relu_v2(x) = x and 1 when x > 0 , else 0, 0

    Parameters
    ----------
    Algorithm: relu_v2

    Parameters:

    x: the dict of input data, support float16, float32, int8, int32, uint8

    y: the dict of output

    mask: the dict of mask_output

    kernel_name: cce kernel name, default value is "relu_v2".

    Returns
    -------
    None
    """

    shape = x.get("shape")
    dtype = x.get("dtype")

    check_shape(shape, param_name="x")

    if shape[-1] % 8 != 0:
        raise RuntimeError(
            "the last axis if shape must be dive by 8")

    check_list = ("float16", "float32", "int8", "int32", "uint8")
    check_dtype(dtype, check_list, param_name="x")

    dtype = dtype.lower()
    input_data = tvm.placeholder(shape, dtype, "input_data")

    with tvm.target.cce():
        res, res_mask = relu_v2_compute(input_data, y, mask, kernel_name)
        sch = generic.auto_schedule([res, res_mask])

    config = {"name": kernel_name,
              "tensor_list": [input_data, res, res_mask],
              "print_ir": False
              }

    te.lang.cce.cce_build_code(sch, config)
