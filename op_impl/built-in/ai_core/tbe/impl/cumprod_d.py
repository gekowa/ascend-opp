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
cumprod_d
"""
from topi.cce import util
from impl.cum_computer import get_computer_by_ctype
from te.utils.op_utils import *


# the computer type
PROD_TYPE = "prod"


# pylint: disable=locally-disabled, unused-argument,invalid-name
# pylint: disable=locally-disabled, too-many-arguments, not-callable
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, OPTION_ATTR_INT, OPTION_ATTR_BOOL,
                 OPTION_ATTR_BOOL, KERNEL_NAME)
def cumprod_d(x, y, axis=0, exclusive=False, reverse=False,
              kernel_name="cumprod_d"):
    """
    Compute the cumulative product of the input tensor along `axis`.

    Parameters
    ----------
    x: dict, shape and dtype, dtype must be in ('float16','float32','int32',
    'int8','uint8')
    y: a dict of output
    axis: a number of int32(default:0), cumulative axis, must be in the range
    [-rank(x), rank(x))
    exclusive: if `True`, perform exclusive cumprod
    reverse: a `bool` (default: False)
    kernel_name: kernel name

    Returns
    -------
    tik_instance: tik_instance

    """
    shape = x.get("shape")
    if axis < 0:
        axis = len(shape) + axis
    check_param(x, axis, kernel_name)
    cumprod_template = get_computer_by_ctype(x, axis, kernel_name, PROD_TYPE)
    cumprod_template.set_ext_params(exclusive, reverse)

    return cumprod_template.get_tik_instance()


def check_param(input_x, axis, kernel_name):
    """
    check the parameters is valid, if one is invalid,then raise error

    Parameters
    ----------
    input_x: dict,shape and datatype
    axis: cumulative axis
    kernel_name: kernel_name
    Returns
    -------
    None
    """
    input_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()

    check_shape(input_shape, param_name="input_x")
    check_dtype(input_dtype, ("float16", "float32", "int32", "int8",
                                        "uint8"), param_name="input_x")

    if axis < len(input_shape)*(-1) or axis >= len(input_shape):
        raise RuntimeError("axis must be in the range [%d, %d). but is %d " % (
            len(input_shape)*(-1), len(input_shape), axis))
