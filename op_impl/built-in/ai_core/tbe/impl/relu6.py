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
relu6
f(x) = min(max(0,x), 6)
"""
from functools import reduce as reduce_ins

import te.lang.cce
import topi
from te import platform as tbe_platform
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils import op_utils
from topi.cce import util


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("relu6")
def relu6_compute(input_x, output_y, kernel_name="relu6"):
    """
    compute of relu6

    Parameters
    ----------
    input_data: TVM tensor
        the placeholder of first input data
    output_y: dict
        shape and dtype of output,should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is "relu6"

    Returns
    -------
    compute result of relu6
    """
    tmp_res = te.lang.cce.vmaxs(input_x, tvm.const(0, input_x.dtype))
    final_res = te.lang.cce.vmins(tmp_res, tvm.const(6, input_x.dtype))

    return final_res


@op_utils.check_op_params(op_utils.REQUIRED_INPUT, op_utils.REQUIRED_OUTPUT,
                          op_utils.KERNEL_NAME)
def relu6(input_x, output_y, kernel_name="relu6"):
    """
       f(x)= 6(x >= 6)
       f(x)= 0(x <= 0)
       f(x)= x(0<x<6)

    Parameters
    ----------
    input_x : dict
        shape and dtype of input_x
    output_y : dict
        shape and dtype of output_y, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is "relu6"

    Returns
    ------
    None
    """
    input_shape = util.scalar2tensor_one(input_x.get("shape"))
    input_dtype = input_x.get("dtype").lower()
    op_utils.check_shape(input_shape, param_name="input_x")

    vmaxs_support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.vmaxs", "float32")
    if input_dtype == "float32" and not vmaxs_support:
        raise RuntimeError(
            "Input dtype is float32, but do not support on the platform")

    # check input tensor data_type
    check_list = ("int32", "float16", "float32")
    op_utils.check_dtype(input_dtype, check_list, param_name="input_x")

    input_shape = [reduce_ins(lambda x, y: x * y, input_shape[:])]
    input_data = tvm.placeholder(input_shape,
                                 name="input_data",
                                 dtype=input_dtype)
    final_res = relu6_compute(input_data, output_y, kernel_name="relu6")

    with tvm.target.cce():
        auto_sch = topi.generic.auto_schedule(final_res)

    config = {"name": kernel_name, "tensor_list": (input_data, final_res)}
    te.lang.cce.cce_build_code(auto_sch, config)
