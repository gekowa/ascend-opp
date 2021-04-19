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
clip_by_norm_no_div_sum
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import *

SHAPE_SIZE_LIMIT = 2147483648


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
# pylint: disable=locally-disabled,redefined-builtin,too-many-locals,unused-variable
def select_compute(condition, x1, x2, y, kernel_name="select"):
    """
    compute for select

    Parameters
    ----------
    condition: TVM tensor
        the placeholder of input condition
    x1: TVM tensor
        the placeholder of input x1
    x2: TVM tensor
        the placeholder of input x2
    y: dict
        dict of y
    kernel_name: str
        cce kernel name, default value is "select"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    res = te.lang.cce.vsel(condition, x1, x2)
    return res


def greater_compute(x, y, z, kernel_name="greater"):
    """
    if x is greater than y, then return 1, else return 0.

    Parameters:
    ----------
    x : Tensor
        input data_x
    y : Tensor
        input data_y
    z : dict
        shape and dtype of output data_z
    kernel_name : str
        cce kernel name, default value is "greater"

    Returns
    -------
    the result
    """
    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    shape_y = te.lang.cce.util.shape_to_list(y.shape)
    dtype = x.dtype.lower()
    shape_x, shape_y, shape = broadcast_shapes(shape_x, shape_y, param_name_input1="x", param_name_input2="y")
    data_x = te.lang.cce.broadcast(x, shape)
    data_y = te.lang.cce.broadcast(y, shape)

    res = te.lang.cce.vcmp(data_x, data_y, 'gt', 'bool')

    return res


def maximum_compute(input_x, input_y, output_z, kernel_name="maximum"):
    """
    calculating data maximum

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
    shape1 = te.lang.cce.util.shape_to_list(input_x.shape)
    shape2 = te.lang.cce.util.shape_to_list(input_y.shape)
    shape1 = util.scalar2tensor_one(shape1)

    shape2 = util.scalar2tensor_one(shape2)

    shape1, shape2, shape_max = broadcast_shapes(shape1, shape2, param_name_input1="select1_result", param_name_input2="maximum_ones")

    data1_tmp1 = te.lang.cce.broadcast(input_x, shape_max)
    data2_tmp1 = te.lang.cce.broadcast(input_y, shape_max)
    res = te.lang.cce.vmax(data1_tmp1, data2_tmp1)
    return res


@fusion_manager.register("clip_by_norm_no_div_sum")
def clip_by_norm_no_div_sum_compute(data_input_x,
                                    data_greater_zeros,
                                    data_select_ones,
                                    data_maximum_ones,
                                    y,
                                    kernel_name="clip_by_norm_no_div_sum"):
    """
    calculating data

    Parameters
    ----------
    Input and output of fusion graph

    Returns
    -------
    output tensor
    """

    # greater
    greater_result = greater_compute(data_input_x, data_greater_zeros,
                                     {}, kernel_name)

    # select
    select_result = select_compute(greater_result, data_input_x,
                                   data_select_ones, {}, kernel_name)

    # sqrt
    sqrt_result = te.lang.cce.vsqrt(select_result)

    # select1
    select1_result = select_compute(greater_result, sqrt_result,
                                    data_input_x, {}, kernel_name)

    res = maximum_compute(select1_result, data_maximum_ones, {}, kernel_name)

    return res

@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT,
                 REQUIRED_OUTPUT, KERNEL_NAME)
def clip_by_norm_no_div_sum(x, greater_zeros, select_ones, maximum_ones, y,
                            kernel_name="clip_by_norm_no_div_sum"):
    """
    calculating data

    Parameters
    ----------
    Input and output of fusion graph

    Returns
    -------
    None
    """
    shape_x = x.get("shape")
    dtype_x = x.get("dtype").lower()
    shape_greater_zeros = greater_zeros.get("shape")
    shape_select_ones = select_ones.get("shape")
    shape_maximum_ones = maximum_ones.get("shape")

    shape_x, shape_greater_zeros, shape_greater_max = \
        broadcast_shapes(shape_x, shape_greater_zeros, param_name_input1="x", param_name_input2="greater_zeros")
    shape_x, shape_select_ones, shape_select_max = \
        broadcast_shapes(shape_x, shape_select_ones, param_name_input1="x", param_name_input2="select_ones")
    shape_x, shape_maximum_ones, shape_maximum_max = \
        broadcast_shapes(shape_x, shape_maximum_ones, param_name_input1="x", param_name_input2="maximum_ones")

    check_shape(shape_x, param_name="x")

    data_input_x = tvm.placeholder(shape_x,
                                   name="data_input_x",
                                   dtype=dtype_x)
    data_greater_zeros = tvm.placeholder(shape_greater_zeros,
                                         name="data_greater_zeros",
                                         dtype=dtype_x)
    data_select_ones = tvm.placeholder(shape_select_ones,
                                       name="data_select_ones",
                                       dtype=dtype_x)
    data_maximum_ones = tvm.placeholder(shape_maximum_ones,
                                        name="data_maximum_ones",
                                        dtype=dtype_x)

    res = clip_by_norm_no_div_sum_compute(data_input_x,
                                          data_greater_zeros,
                                          data_select_ones,
                                          data_maximum_ones,
                                          y,
                                          kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "bool_storage_as_1bit": False,
              "tensor_list": [data_input_x, data_greater_zeros,
                              data_select_ones, data_maximum_ones, res]}

    te.lang.cce.cce_build_code(sch, config)
