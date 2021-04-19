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
rsqrt

  Op_description :
    Computes reciprocal of square root of x element-wise

    # rsqrt(
    #   x,
    #   y,
    #   kernel_name="rsqrt_cce")

  Supportive_dtype_format :
    ['float16', 'float32']
    ['ALL']

  Constraint :
    [1] All : shape size limit is 2147483648.
"""
from te import tvm
import te.lang.cce
from te.platform.cce_conf import api_check_support
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_shape
from te.utils.op_utils import refine_shape_axes
from topi import generic
from topi.cce import util
from te.utils.op_utils import *

# const value
CONST_ONE = 1.0


# pylint: disable=locally-disabled,too-many-arguments,
# pylint: disable=unused-argument,invalid-name
@fusion_manager.register("rsqrt")
def rsqrt_compute(x, y, kernel_name="rsqrt_cce"):
    """
    Algrithm : rsqrt(x) = 1 / sqrt(x)  where x > 0

    Parameters
    ----------
    x: the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name

    Returns
    -------
    res : result of rsqrt
    """

    inp_dtype = x.dtype

    if inp_dtype == "float16" and api_check_support("te.lang.cce.vadd",
                                                    "float32"):
        x = te.lang.cce.cast_to(x, "float32")

    data_res = _compute(x)

    if inp_dtype == "float16":
        data_res = te.lang.cce.cast_to(data_res, "float16")

    return data_res


def _compute(data_input):
    """
    Algrithm: rsqrt(x) = 1 / sqrt(x)

    Parameters
    ----------
    data_input: the placeholder of data input

    Returns
    -------
    data_res :  return of rsqrt
    """

    inp_shape = data_input.shape
    data_sqrt = te.lang.cce.vsqrt(data_input, 1)
    tesor_one = te.lang.cce.broadcast(tvm.const(CONST_ONE, data_input.dtype),
                                      inp_shape)
    result = te.lang.cce.vdiv(tesor_one, data_sqrt)

    return result


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def rsqrt(x, y, kernel_name="rsqrt_cce"):
    """
    Algrithm: rsqrt(x) = 1 / sqrt(x)  where x > 0

    Parameters
    ----------
    Algorithm: rsqrt

    Parameters:

    x: the dict of input data, support float16, float32

    y: the dict of output

    kernel_name: cce kernel name, default value is "rsqrt_cce".

    Returns
    -------
    None
    """

    shape = x.get("shape")
    dtype = x.get("dtype")

    check_shape(shape, param_name="x")
    shape, _ = refine_shape_axes(shape, [])

    check_list = ("float16", "float32")
    check_dtype(dtype, check_list, param_name="x")

    dtype = dtype.lower()
    input_data = tvm.placeholder(shape, dtype, "input_data")

    with tvm.target.cce():
        res = rsqrt_compute(input_data, y, kernel_name)
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_data, res],
              "print_ir": False,
             }

    te.lang.cce.cce_build_code(sch, config)
