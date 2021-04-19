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
dynamic div
"""
from functools import reduce as reduceIns
from te.platform.shape_classifier import classify
from te.platform.shape_classifier import Mode
import te.lang.dynamic
from te import platform as tbe_platform
from te import tvm
from topi import generic
from te.utils.op_utils import check_op_params
from te.utils.op_utils import check_dtype
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import variable_shape


def sqrt_compute(input_data, output_data, kernel_name="sqrt"):
    """
    calculating data sqrt,y= x**0.5,mini not support vsqrt, use exp(0.5*log(x))

    Parameters
    ----------
    input_data: TVM tensor
        the placeholder of input data
    output_data: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is sqrt

    Returns
    -------
    result: TVM tensor
        the result of sqrt
    """
    dtype = input_data.dtype
    has_improve_precision = False
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.dynamic.vsqrt", "float32"):
        input_data = te.lang.dynamic.cast_to(input_data, "float32")
        has_improve_precision = True
    result = te.lang.dynamic.vsqrt(input_data)

    if has_improve_precision:
        result = te.lang.dynamic.cast_to(result, "float16")

    return result


@te.op.register_operator("Sqrt")
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def sqrt(input_x, output_y, kernel_name="sqrt"):
    """
    algorithm: sqrt
    calculating data sqrt,y= x**0.5, mini not support vsqrt, use exp(0.5*log(x))

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is sqrt

    Returns
    -------
    None
    """

    # check dtype
    x_dtype = input_x.get("dtype").lower()
    check_list = ("float16", "float32")
    check_dtype(x_dtype, check_list, param_name="input_x")

    ins = classify([input_x], Mode.ELEWISE)
    schedules, tensors = [], []
    for (input_x,) in ins:
        with te.op.compute():
            # shape
            x_shape = variable_shape([input_x])
            fuseshape = [1]
            fuseshape[0] = reduceIns(lambda x, y: x * y, x_shape[0])
            # div_compute
            input_data = tvm.placeholder(fuseshape, name="input_data",
                                         dtype=x_dtype)
            res = sqrt_compute(input_data, output_y, kernel_name)

            tensors.append([input_data, res])
        with tvm.target.cce():
            sch = generic.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    te.lang.dynamic.build(schedules, config)
