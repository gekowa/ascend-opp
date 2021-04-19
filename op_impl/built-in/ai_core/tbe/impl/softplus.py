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
softplus
"""
from te import tvm
from te import platform as tbe_platform
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from functools import reduce as functools_reduce
from te.utils.op_utils import *

# define a scalar, value = 1
SCALAR_ONE = 1
NEG_LN_2 = - 0.69314718055994530941723212145818
NEG_ONE = -1

# pylint: disable=locally-disabled,unused-argument,too-many-locals
@fusion_manager.register("softplus")
def softplus_compute(input_x, y, kernel_name="softplus"):
    """
    Compute for softplus.
    The compute: "log(exp(x) + 1)".

    Parameters
    ----------
    input_x: TVM tensor
        data of input.
        source data type, support "float16", "float32".
    y: TVM tensor
        data of output.
    kernel_name: str
        kernel name, default value is "softplus".

    Returns
    -------
    res: TVM tensor
        output data and has the same type as `features`.
    """
    dtype = input_x.dtype
    if dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp",
                                                    "float32"):
        input_x = te.lang.cce.cast_to(input_x, "float32")

    positive_part = te.lang.cce.vmaxs(input_x, tvm.const(0, dtype="float32"))
    negative_part = te.lang.cce.vmins(input_x, tvm.const(0, dtype="float32"))

    # calculate positive part softplus
    pos_to_neg = te.lang.cce.vmuls(positive_part, tvm.const(NEG_ONE, dtype="float32"))
    exp_pos = te.lang.cce.vexp(pos_to_neg)
    exp_add_one = te.lang.cce.vadds(exp_pos, SCALAR_ONE)
    log_pos = te.lang.cce.vlog(exp_add_one)
    res_positive = te.lang.cce.vadd(log_pos, positive_part)

    #calculate positive part softplus
    exp_neg = te.lang.cce.vexp(negative_part)
    add_one = te.lang.cce.vadds(exp_neg, SCALAR_ONE)
    res_negative = te.lang.cce.vlog(add_one)

    res_tmp = te.lang.cce.vadd(res_positive, res_negative)
    res = te.lang.cce.vadds(res_tmp, NEG_LN_2)

    if dtype == "float16":
        res = te.lang.cce.cast_to(res, "float16")

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def softplus(x, y, kernel_name="softplus"):
    """
    Compute for softplus.

    Parameters
    ----------
    x: dict
        data of input.
        source data type, support "float16", "float32".
    y: dict
        data of output.
    kernel_name: str
        kernel name, default value is "softplus".

    Returns
    -------
    None
    """
    shape_input = x.get("shape")
    dtype_input = x.get("dtype")

    check_shape(shape_input, param_name="x")

    check_list = ("float16", "float32")
    check_dtype(dtype_input.lower(), check_list, param_name="x")

    shape = util.shape_refine(shape_input)
    input_dtype = dtype_input.lower()
    shape_x = (functools_reduce(lambda x, y: x*y, shape[:]),)
    data = tvm.placeholder(shape_x, name="data", dtype=input_dtype)

    res = softplus_compute(data, y, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data, res]}
    te.lang.cce.cce_build_code(sch, config)

