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
gn_training_reduce
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from te.utils.op_utils import *
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


# pylint: disable=locally-disabled,unused-argument,invalid-name
# pylint: disable=locally-disabled,redefined-builtin
def op_select_format(x, sum, square_sum, num_groups,
                     kernel_name="gn_training_reduce"):
    """
    select format dynamically
    """
    input0 = gen_param(classify="input0", name="x",
                       datatype="float16,float,float16,float",
                       format="NCHW,NHWC,NCHW,NHWC")
    output0 = gen_param(classify="output0", name="sum",
                        datatype="float,float,float,float",
                        format="ND,ND,ND,ND")
    output1 = gen_param(classify="output1", name="square_sum",
                        datatype="float,float,float,float",
                        format="ND,ND,ND,ND")

    param_list = [input0, output0, output1]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def _shape_check(shape_x, data_format, num_groups):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    data_format: str
        data format of input x
    num_groups: int
        groups of channel
    Returns
    -------
    None
    """
    if data_format == "NCHW":
        c_index_ = 1
    elif data_format == "NHWC":
        c_index_ = 3
    else:
        check_format(data_format, ("NCHW", "NHWC"), param_name="x")

    check_shape(shape_x, min_rank=4, max_rank=4, param_name="x")
    if shape_x[c_index_] % num_groups != 0:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_009
        error_info['op_name'] = OP_NAME
        error_info['param_name'] = "channel and num_groups"
        error_info['rule_desc'] = "num_groups must divide C channel"
        error_info['param_value'] = "{} and {}".format(
            shape_x[c_index_], num_groups)
        raise RuntimeError(error_info,
                           "Op[%s] has rule: %s, but [%s] is [%s]." \
                           % (error_info['op_name'],
                              error_info['rule_desc'],
                              error_info['param_name'],
                              error_info['param_value']))


@fusion_manager.register("gn_training_reduce")
def gn_training_reduce_compute(x, data_format, kernel_name="gn_training_reduce"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of input_x
    data_format: str
        format string of input x
    kernel_name : str
        kernel name, default value is "gn_training_reduce"

    Returns
    -------
    output tensor
    """
    if data_format == "NCHW":
        reduce_axis = [2, 3, 4]
    else:
        reduce_axis = [1, 2, 4]
    dtype = x.dtype
    if dtype == "float16":
        x = te.lang.cce.cast_to(x, "float32")
    square_x = te.lang.cce.vmul(x, x)
    sum_x, square_sum_x = te.lang.cce.tuple_sum([x, square_x], reduce_axis, True)
    res = [sum_x, square_sum_x]
    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_OUTPUT,
                 OPTION_ATTR_INT, KERNEL_NAME)
def gn_training_reduce(x, sum, square_sum, num_groups=2, kernel_name="gn_training_reduce"):
    """
    calculating data

    Parameters
    ----------
    x: dict
        dict of input, A 5HD Tensor for input data.
    sum: dict
        dict of sum, A `Tensor`. Sum of x.
    square_sum: dict
        dict of square_sum, A `Tensor`. Square sum of x.
    num_groups: int
        A integer value indicates the group in channel.
    kernel_name : str
        kernel name, default value is "gn_training_reduce"

    Returns
    -------
    None
    """
    shape_x = x.get("shape")
    dtype_x = x.get("dtype")
    data_format = x.get("format")
    input_dtype = dtype_x.lower()

    _shape_check(shape_x, data_format, num_groups)
    check_dtype(input_dtype, ("float16", "float32"), param_name="x")

    # Reshape NCHW -> N[GD]HW
    if data_format == "NCHW":
        shape_x = [shape_x[0], num_groups, shape_x[1] // num_groups, shape_x[2], shape_x[3]]

    # Reshape NHWC -> NHW[GD]
    elif data_format == "NHWC":
        shape_x = [shape_x[0], shape_x[1], shape_x[2], num_groups, shape_x[3] // num_groups]

    x_input = tvm.placeholder(shape_x, name="x_input", dtype=input_dtype)

    res = gn_training_reduce_compute(x_input, data_format, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    tensor_list = [x_input] + list(res)

    config = {"name": kernel_name,
              "tensor_list": tensor_list}

    te.lang.cce.cce_build_code(schedule, config)
