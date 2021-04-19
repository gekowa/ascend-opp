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
cumulativelogsumexp_d
"""
from topi.cce import util
from impl.cum_computer import get_computer_by_ctype
from te.utils.op_utils import check_op_params
from te.utils.op_utils import check_dtype
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import OPTION_OUTPUT
from te.utils.op_utils import REQUIRED_ATTR_INT
from te.utils.op_utils import OPTION_ATTR_BOOL
from te.utils.op_utils import KERNEL_NAME

# the computer type
COMPUTE_TYPE = "logsumexp"


# pylint: disable=locally-disabled, unused-argument,invalid-name
# pylint: disable=locally-disabled, too-many-arguments, not-callable
@check_op_params(REQUIRED_INPUT, OPTION_OUTPUT, REQUIRED_ATTR_INT,
                 OPTION_ATTR_BOOL, OPTION_ATTR_BOOL, KERNEL_NAME)
def cumulative_logsumexp_d(x, y, axis, exclusive=False, reverse=False,
                           kernel_name="cumulative_logsumexp_d"):
    """
    Compute the cumulative logsumexp of the input tensor along `axis`.

    Parameters
    ----------
    x: dict, shape and dtype, must be in ('float16', 'float32', 'float64')
    y: the dict of output
    axis: a number of int32 or int 64, cumulative axis, must be in the range
    [-rank(x), rank(x))
    exclusive: if `True`, perform exclusive cumsum
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

    cumlogsumexp_template = get_computer_by_ctype(
        x, axis, kernel_name, COMPUTE_TYPE)
    cumlogsumexp_template.set_ext_params(exclusive, reverse)

    return cumlogsumexp_template.get_tik_instance()


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

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(input_shape)
    util.check_tensor_shape_size(input_shape)
    check_dtype(input_dtype, ("float16", "float32"))

    if axis < len(input_shape) * (-1) or axis >= len(input_shape):
        raise RuntimeError("axis must be in the range [%d, %d). but is %d "
                           % (len(input_shape) * (-1), len(input_shape), axis))
