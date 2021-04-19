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
rsqrt_grad
"""
from __future__ import absolute_import

from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import refine_shapes_for_broadcast
from topi import generic
from topi.cce import util
from te.utils.op_utils import *

# define a scalar, value = -0.5
SCALAR = -0.5

# pylint: disable=locally-disabled,unused-argument
@fusion_manager.register("rsqrt_grad")
def rsqrt_grad_compute(input_y, input_dy, output_z, kernel_name="rsqrt_grad"):
    """
    compute for rsqrt_grad

    Parameters
    ----------
    input_y: TVM tensor
        the placeholder of input_y
    input_dy: TVM tensor
        the placeholder of input_dy
    output_z: dict
        dict info of output_z
    kernel_name: str
        cce kernel name, default value is "rsqrt_grad"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    dtype_input_y = input_y.dtype
    rsqrt_const = tvm.const(SCALAR, dtype=dtype_input_y)
    if dtype_input_y in ("int8", "float16"):
        rsqrt_const = tvm.const(SCALAR, dtype="float32")
        input_y = te.lang.cce.cast_to(input_y, "float32")
        input_dy = te.lang.cce.cast_to(input_dy, "float32")
    res_vmul = te.lang.cce.vmul(input_y, input_y)
    res_vmul1 = te.lang.cce.vmul(res_vmul, input_y)
    res_vmul2 = te.lang.cce.vmul(res_vmul1, input_dy)
    res = te.lang.cce.vmuls(res_vmul2, rsqrt_const)
    if dtype_input_y in ("int8", "int32", "float16"):
        res = te.lang.cce.cast_to(res, dtype_input_y, f1628IntegerFlag=True)
    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def rsqrt_grad(input_y, input_dy, output_z, kernel_name="rsqrt_grad"):
    """
    calculate the backpropagation of rsqrt operation
    rsqrt: y = 1 / sqrt（x）
    rsqrt_grad: -1/2 * y**3 *dy

    Parameters
    ----------
    input_y: dict
        dict of input_y, include keys(shape and dtype)
    input_dy: dict
        dict of input_dy, include keys(shape and dtype)
    output_z: dict
        dict of  output
    kernel_name: str
        cce kernel name, default value is "rsqrt_grad"

    Returns
    -------
    None
    """
    shape_input_y = input_y.get("shape")
    dtype_input_y = input_y.get("dtype")
    shape_input_dy = input_dy.get("shape")
    dtype_input_dy = input_dy.get("dtype")

    check_shape(shape_input_y, param_name="input_y")
    check_shape(shape_input_dy, param_name="input_dy")
    util.compare_tensor_dict_key(input_y, input_dy, "shape")

    check_list = ("float16", "float32", "int32", "int8")
    dtype_input_y = dtype_input_y.lower()
    check_dtype(dtype_input_y, check_list, param_name="input_y")
    dtype_input_dy = dtype_input_dy.lower()
    check_dtype(dtype_input_dy, check_list, param_name="input_dy")
    util.compare_tensor_dict_key(input_y, input_dy, "dtype")
    reshape_y, reshape_dy = refine_shapes_for_broadcast(shape_input_y,
                                                        shape_input_dy)

    data_input_y = tvm.placeholder(reshape_y,
                                   name="data_input_y",
                                   dtype=dtype_input_y)
    data_input_dy = tvm.placeholder(reshape_dy,
                                    name="data_input_dy",
                                    dtype=dtype_input_dy)

    res = rsqrt_grad_compute(data_input_y, data_input_dy, output_z, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input_y, data_input_dy, res]}
    te.lang.cce.cce_build_code(sch, config)
