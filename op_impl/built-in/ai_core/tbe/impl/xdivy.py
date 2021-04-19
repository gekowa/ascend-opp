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
xdivy
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import refine_shapes_for_broadcast
from te import platform as tbe_platform
from topi import generic
from topi.cce import util
from te.utils.op_utils import *
# define a scalar , value = 1
SCALAR_ONE = 1
# minimun num of float32 2**(-126)
MININUM_NUM_FLOAT = 2**(-126)
# minimun num of float16 2**(-24)
MININUM_NUM_HALF = 2**(-24)
# max num of float32 is 2**126, but cce can only support 2**62,
# so use 62/62/2 to adaptor 149
MAX_ONE_CONST_FLOAT = 2**62
MAX_TWO_CONST_FLOAT = 2**2
# max num of float16 is 2**24, but cce can only support 2**12,
# so use 12/12 to adaptor 24
MAX_CONST_HALF = 2**12

# pylint: disable=locally-disabled,too-many-locals,unused-argument
@fusion_manager.register("xdivy")
def xdivy_compute(input_x, input_y, output_z, kernel_name="xdivy"):
    """
    xdivy compute
    calculating data's xdivy,return 0 if x==0 and x/y otherwise, elementwise

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    input_y: TVM tensor
        the placeholder of input_y
    output_z: dict
        dict with keys(shape and dtype) of output
    kernel_name : str
        kernel name, default value is "xdivy"

    Returns
    -------
    res: TVM tensor
        the result of xdivy compute
    """
    input_data1 = te.lang.cce.util.shape_to_list(input_x.shape)
    input_data2 = te.lang.cce.util.shape_to_list(input_y.shape)
    shape_list = broadcast_shapes(input_data1, input_data2,
                                  param_name_input1="input_x",
                                  param_name_input2="input_y")
    dtype = input_x.dtype

    broadcast_x = te.lang.cce.broadcast(input_x, shape_list[2])
    broadcast_y = te.lang.cce.broadcast(input_y, shape_list[2])
    broadcast_one = te.lang.cce.broadcast(tvm.const(SCALAR_ONE, dtype),
                                          shape_list[2], dtype)

    abs_x = te.lang.cce.vabs(broadcast_x)
    abs_y = te.lang.cce.vabs(broadcast_y)
    add_x_y = te.lang.cce.vadd(abs_x, abs_y)

    if dtype == "float32":
        data_min = te.lang.cce.broadcast(tvm.const(MININUM_NUM_FLOAT,
                                                   dtype=dtype),
                                         shape_list[2], dtype)
    elif dtype == "float16":
        data_min = te.lang.cce.broadcast(tvm.const(MININUM_NUM_HALF,
                                                   dtype=dtype),
                                         shape_list[2], dtype)

    zero_x_y = te.lang.cce.vmin(add_x_y, data_min)

    if dtype == "float32":
        data_mul1 = te.lang.cce.vmuls(zero_x_y, tvm.const(MAX_ONE_CONST_FLOAT,
                                                          dtype=dtype))
        data_mul2 = te.lang.cce.vmuls(data_mul1, tvm.const(MAX_ONE_CONST_FLOAT,
                                                           dtype=dtype))
        mul_data = te.lang.cce.vmuls(data_mul2, tvm.const(MAX_TWO_CONST_FLOAT,
                                                          dtype=dtype))
    elif dtype == "float16":
        data_mul1 = te.lang.cce.vmuls(zero_x_y, tvm.const(MAX_CONST_HALF,
                                                          dtype=dtype))
        mul_data = te.lang.cce.vmuls(data_mul1, tvm.const(MAX_CONST_HALF,
                                                          dtype=dtype))

    sub_x_y_zero = te.lang.cce.vsub(mul_data, broadcast_one)
    abs_x_y_zero = te.lang.cce.vabs(sub_x_y_zero)
    input_y_revised = te.lang.cce.vadd(broadcast_y, abs_x_y_zero)

    has_improve_precision = False
    if dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv",
                                                    "float32"):
        broadcast_x = te.lang.cce.cast_to(broadcast_x, "float32")
        input_y_revised = te.lang.cce.cast_to(input_y_revised, "float32")
        has_improve_precision = True

    res = te.lang.cce.vdiv(broadcast_x, input_y_revised)

    if has_improve_precision:
        res = te.lang.cce.cast_to(res, dtype)

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def xdivy(input_x, input_y, output_z, kernel_name="xdivy"):
    """
    algorithm: xdivy
    calculating data's xdivy,return 0 if x==0 and x/y otherwise, elementwise

    Parameters
    ----------
    input_x: dict
        dict with keys(shape and dtype) of input_x
    input_y: dict
        dict with keys(shape and dtype) of input_y
    output_z: dict
        dict with keys(shape and dtype) of output
    kernel_name : str
        kernel name, default value is "xdivy"

    Returns
    -------
    None
    """
    shape_x = input_x.get("shape")
    dtype = input_x.get("dtype")
    shape_y = input_y.get("shape")
    dtype_y = input_y.get("dtype")

    util.compare_tensor_dict_key(input_x, input_y, "dtype")
    check_shape(shape_x, param_name="input_x")
    check_shape(shape_y, param_name="input_y")
    shape_list = broadcast_shapes(shape_x, shape_y, param_name_input1="input_x",
                                  param_name_input2="input_y")
    input_dtype = dtype.lower()
    input_dtype_y = dtype_y.lower()
    check_list = ("float16", "float32")
    check_dtype(input_dtype, check_list, param_name="input_x")
    check_dtype(input_dtype_y, check_list, param_name="input_y")

    reshape_x, reshape_y = refine_shapes_for_broadcast(shape_list[0],
                                                       shape_list[1])
    data_x = tvm.placeholder(reshape_x, dtype=input_dtype, name="data_x")
    data_y = tvm.placeholder(reshape_y, dtype=input_dtype, name="data_y")

    res = xdivy_compute(data_x, data_y, output_z, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, data_y, res]}
    te.lang.cce.cce_build_code(sch, config)
