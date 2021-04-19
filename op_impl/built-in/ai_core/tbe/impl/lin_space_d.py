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
lin_space_d
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import *


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=too-many-locals
@fusion_manager.register("lin_space_d")
def lin_space_d_compute(input_assist, input_start, input_stop, input_num,
                        output_op, kernel_name="lin_space"):
    """
    algorithm: linspace
    Description of calculating process with TE api,
    the computational formula is as follows.
    step = (stop - start)/(num - 1)
    res = assist * step + start

    Parameters
    ----------
    input_assist: TVM tensor
        contains assist data
    input_start: TVM tensor
        contains start data
    input_stop: TVM tensor
        contains stop data
    input_num: TVM tensor
        contains num data
    output_op: dict
        dict of output, which contains shape and dtype
    kernel_name: str
        cce kernel name, default value is "lin_space"

    Returns
    -------
    res: TVM tensor
        the result of linspace compute
    """
    num_float = te.lang.cce.cast_to(input_num, "float32")
    num_divided = te.lang.cce.vadds(num_float, -1.0)

    step_divider = te.lang.cce.vsub(input_stop, input_start)
    step = te.lang.cce.vdiv(step_divider, num_divided)

    res_temp = te.lang.cce.vmul(input_assist,
                                te.lang.cce.broadcast(step,
                                                      input_assist.shape))
    res = te.lang.cce.vadd(res_temp,
                           te.lang.cce.broadcast(input_start,
                                                 input_assist.shape))

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT,
                 REQUIRED_OUTPUT, KERNEL_NAME)
def lin_space_d(input_assist, input_start, input_stop,
                input_num, output_op, kernel_name="lin_space_d"):
    """
    algorithm: linspace
    Generates values in an interval.
    A sequence of 'num' evenly-spaced values are generated beginning at 'start'.
    If 'num' > 1, the values in the sequence increase by 'stop-start / num-1',
    so that the last one is exactly 'stop'.
    For example:
    linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]

    Parameters
    ----------
    input_assist: dict
        dict of input, which contains shape and dtype
    input_start: dict
        dict of start, which contains shape and dtype
    input_stop: dict
        dict of stop, which contains shape and dtype
    input_num: dict
        dict of num, which contains shape and dtype
    output_op: dict
        dict of output, which contains shape and dtype
    kernel_name: str
        kernel name, default value is "lin_space"

    Returns
    -------
    None
    """
    shape_assist = input_assist.get("shape")
    shape_start = input_start.get("shape")
    shape_stop = input_stop.get("shape")
    shape_num = input_num.get("shape")
    dtype_input = input_start.get("dtype")
    dtype_input_stop = input_stop.get("dtype")
    dtype_input_assist = input_assist.get("dtype")
    dtype_num = input_num.get("dtype")
    check_shape(shape_assist, param_name="input_assist")

    check_dtype(dtype_input_assist.lower(), ("float32",), param_name="input_assist")
    check_dtype(dtype_input.lower(), ("float32",), param_name="input_start")
    check_dtype(dtype_input_stop.lower(), ("float32",), param_name="input_stop")
    check_dtype(dtype_num.lower(), ("int32",), param_name="input_num")

    # check shape of start, stop and num, must be (1,)
    shape_start = tuple(shape_start)
    shape_stop = tuple(shape_stop)
    shape_num = tuple(shape_num)
    if shape_start != (1,) or shape_stop != (1,) or shape_num != (1,):
        raise RuntimeError(
            "lin_space only support rank=1 while shape "
            "of start or stop or num is not (1,)")

    # check shape of assist, only support 1dim
    if len(shape_assist) != 1:
        raise RuntimeError(
            "lin_space only support rank=1 while length of assist shape is %d"
            % (len(shape_assist)))

    assist_input = tvm.placeholder(shape_assist, name="assist_input",
                                   dtype=dtype_input.lower())
    start_input = tvm.placeholder(shape_start, name="start_input",
                                  dtype=dtype_input.lower())
    stop_input = tvm.placeholder(shape_stop, name="stop_input",
                                 dtype=dtype_input.lower())
    num_input = tvm.placeholder(shape_num, name="num_input",
                                dtype=dtype_num.lower())

    res = lin_space_d_compute(assist_input, start_input, stop_input,
                              num_input, output_op, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [assist_input,
                              start_input,
                              stop_input,
                              num_input,
                              res]}
    te.lang.cce.cce_build_code(sch, config)
