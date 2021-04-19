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
reduce_max_d
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from impl.reduce_max_d_tik import reduce_max_d_tik
from te.utils.op_utils import *

NoneType = type(None)


# pylint: disable=unused-argument,invalid-name,unexpected-keyword-arg
def reduce_max_d_compute(x, y, axis, keepdims, kernel_name="reduce_max_d"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of input_x
    y : dict
        dict of output_y, include keys(shape and dtype)
    axis: list
        list axis to reduce
    keepdims: bool
        if true, retains reduced dimensions with length 1,
        default value is None
    kernel_name : str
        kernel name, default value is "reduce_max_d"

    Returns
    -------
    output tensor
    """
    shape = te.lang.cce.util.shape_to_list(x.shape)
    shape_len = len(shape)
    if not axis:
        axis = range(shape_len)
    if hasattr(axis, 'index'):
        axis = list(axis)

    inp_dtype = x.dtype

    res_tmp = te.lang.cce.reduce_max(x, axis=axis,
                                     keepdims=keepdims,
                                     priority_flag=True)
    res = te.lang.cce.cast_to(res_tmp, inp_dtype, f1628IntegerFlag=True)
    return res


# pylint: disable=invalid-name
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, (REQUIRED_ATTR_INT, REQUIRED_ATTR_LIST_INT),
                 OPTION_ATTR_BOOL, KERNEL_NAME)
def reduce_max_d(x, y, axis, keepdims=False, kernel_name="reduce_max_d"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        shape and dtype of input
    y : dict
        shape and dtype of output, should be same shape and type as input
    axis: list
        the first axis to reduce,may be negative to index from the end
        (e.g., -1 for the last axis).
        axis may be int or list(e.g. [1,2])
    keepdims: bool
        if true, retains reduced dimensions with length 1,
        default value is None
    kernel_name : str
        kernel name, default value is "reduce_max_d"

    Returns
    -------
    None
    """
    shape = x.get("shape")
    dtype = x.get("dtype")
    input_dtype = dtype.lower()

    check_shape(shape, param_name="x")

    check_list = ["float16", "float32", "int8", "uint8", "int32"]
    check_dtype(input_dtype, check_list, param_name="x")

    shape_len = len(shape)

    if not axis:
        axis = range(shape_len)

    if hasattr(axis, 'index'):
        axis = list(axis)

    axis = util.axis_check(shape_len, axis)


    # Shape should not be modified in 5HD mode
    # 5HD Special param for 5hd schedule
    is_5hdc = util.check_and_init_5hdc_reduce_support(x, axis)
    if not is_5hdc:
        shape, axis = util.shape_refine(list(shape), axis)
        shape, axis = util.simplify_axis_shape(shape, axis)
    shape_len = len(shape)
    x["shape"] = shape
    if input_dtype in ("float32", "int32") and len(axis) == 1 \
            and ((axis[0] == (shape_len - 1)) or (axis[0] == -1)):
        reduce_max_d_tik(x, y, axis[0], kernel_name)
    else:
        data_input = tvm.placeholder(shape,
                                     name="data_input_" + kernel_name,
                                     dtype=input_dtype)
        res = reduce_max_d_compute(data_input, y, axis,
                                   keepdims, kernel_name)

        if is_5hdc:
            res.ori_shape = x["ori_shape"]
            res.ori_format = x["ori_format"]
        with tvm.target.cce():
            sch = generic.auto_schedule(res)

        config = {"name": kernel_name, "tensor_list": [data_input, res]}

        te.lang.cce.cce_build_code(sch, config)
