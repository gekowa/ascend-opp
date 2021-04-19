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
dynamic square
"""
import te.lang.dynamic
from te import tvm
from topi import generic
from functools import reduce as reduceIns
from te.platform.shape_classifier import classify
from te.platform.shape_classifier import Mode
from te.utils.op_utils import check_op_params
from te.utils.op_utils import check_dtype
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import variable_shape


def square_compute(input_x, output_y, kernel_name="square"):
    """
    algorithm: square
    calculating data's square,y= x*x

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is square

    Returns
    -------
    res : tvm.tensor
        the result of square
    """
    res = te.lang.dynamic.vmul(input_x, input_x)
    return res


@te.op.register_operator("Square")
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def square(input_x, output, kernel_name="square"):
    """
    algorithm: square
    calculating data's square,y= x*x

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32, int32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "square"

    Returns
    -------
    None
    """

    # check dtype
    x_dtype = input_x.get("dtype").lower()
    check_list = ("float16", "float32", "int32")
    check_dtype(x_dtype, check_list, param_name="input_x")

    ins = classify([input_x], Mode.ELEWISE)
    schedules, tensors = [], []
    for (input_x,) in ins:
        with te.op.compute():
            # shape
            x_shape = variable_shape([input_x])
            fuseshape = [1]
            fuseshape[0] = reduceIns(lambda x, y: x * y, x_shape[0])
            # square_compute
            data_x = tvm.placeholder(fuseshape, x_dtype, name="data_x")
            res = square_compute(data_x, output, kernel_name)

            tensors.append((data_x, res))
        with tvm.target.cce():
            sch = generic.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    te.lang.dynamic.build(schedules, config)
