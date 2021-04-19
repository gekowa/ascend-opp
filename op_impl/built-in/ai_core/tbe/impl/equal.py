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
equal
"""
from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi import generic
from te.utils.op_utils import refine_shapes_for_broadcast
from topi.cce import util
from te.utils.op_utils import *

# define a scalar, value = 2**(-126), minimun num of float32 2**(-126)
SCALAR_MIN_FP32 = 2**(-126)
# define a scalar, value = 2**(50)
SCALAR_MUL_FP32 = 2**(50)
# define a scalar, value = 2**(26)
SCALAR_MUL2_FP32 = 2**(26)
# define a scalar, value = 2**(-24), minimun num of float16 2**(-24)
SCALAR_MIN_FP16 = 2**(-24)
# define a scalar, value = 2**(12)
SCALAR_MUL_FP16 = 2**(12)
# define a scalar, value = 1
SCALAR_ONE = 1

# limit of input shape
MAX_SHAPE_NUM = 10000000

# pylint: disable=locally-disabled,unused-argument,too-many-locals
@fusion_manager.register("equal")
def equal_compute(input_x, input_y, output_z, kernel_name="equal"):
    """
    compute for equal

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    input_y: TVM tensor
        the placeholder of input_y
    output_z: dict
        dict info of output_z
    kernel_name: str
        cce kernel name, default value is "equal"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    dtype_x = input_x.dtype
    shape_x = te.lang.cce.util.shape_to_list(input_x.shape)
    shape_y = te.lang.cce.util.shape_to_list(input_y.shape)
    shape_x, shape_y, shape_broadcast = broadcast_shapes(shape_x, shape_y, param_name_input1="input_x", param_name_input2="input_y")

    if dtype_x == "float32":
        scalar_min = tvm.const(SCALAR_MIN_FP32, dtype="float32")
        scalar_mul = tvm.const(SCALAR_MUL_FP32, dtype="float32")
        scalar_mul1 = tvm.const(SCALAR_MUL2_FP32, dtype="float32")
        scalar_one = tvm.const(-1*SCALAR_ONE, dtype="float32")
    else:
        scalar_min = tvm.const(SCALAR_MIN_FP16, dtype="float16")
        scalar_mul = tvm.const(SCALAR_MUL_FP16, dtype="float16")
        scalar_one = tvm.const(-1*SCALAR_ONE, dtype="float16")
    if dtype_x in ("int8", "uint8"):
        input_x = te.lang.cce.cast_to(input_x, "float16")
        input_y = te.lang.cce.cast_to(input_y, "float16")

    input_x = te.lang.cce.broadcast(input_x, shape_broadcast)
    input_y = te.lang.cce.broadcast(input_y, shape_broadcast)

    res_vsub = te.lang.cce.vsub(input_x, input_y)
    res_vabs = te.lang.cce.vabs(res_vsub)
    res_min = te.lang.cce.vmins(res_vabs, scalar_min)
    res_vmul = te.lang.cce.vmuls(res_min, scalar_mul)
    res_vmul1 = te.lang.cce.vmuls(res_vmul, scalar_mul)

    if dtype_x == "float32":
        res_vmul2 = te.lang.cce.vmuls(res_vmul1, scalar_mul1)
        res_vsub1 = te.lang.cce.vadds(res_vmul2, scalar_one)
        res_vabs1 = te.lang.cce.vabs(res_vsub1)
    else:
        res_vsub1 = te.lang.cce.vadds(res_vmul1, scalar_one)
        res_vabs1 = te.lang.cce.vabs(res_vsub1)

    res = te.lang.cce.cast_to(res_vabs1, "int8", True)

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def equal(input_x, input_y, output_z, kernel_name="equal"):
    """
    Returns the truth value of (x = y) element-wise

    Parameters
    ----------
    input_x: dict
        dict of input_x, include keys(shape and dtype)
    input_y: dict
        dict of input_y, include keys(shape and dtype)
    output_z: dict
        dict of  output
    kernel_name: str
        cce kernel name, default value is "equal"

    Returns
    -------
    None
    """
    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype")
    shape_y = input_y.get("shape")
    dtype_y = input_y.get("dtype")
    shape_x, shape_y, shape_broadcast = broadcast_shapes(shape_x, shape_y, param_name_input1="input_x", param_name_input2="input_y")

    check_shape(shape_x, param_name="input_x")
    check_shape(shape_y, param_name="input_y")

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    dtype_x = dtype_x.lower()
    check_dtype(dtype_x, check_list, param_name="input_x")
    dtype_y = dtype_y.lower()
    check_dtype(dtype_y, check_list, param_name="input_y")
    util.compare_tensor_dict_key(input_x, input_y, "dtype")

    shape_x = list(shape_x)
    shape_y = list(shape_y)
    shape_x, shape_y = refine_shapes_for_broadcast(shape_x, shape_y)
    data_input_x = tvm.placeholder(shape_x, name="data_input_x", dtype=dtype_x)
    data_input_y = tvm.placeholder(shape_y, name="data_input_y", dtype=dtype_y)

    res = equal_compute(data_input_x, data_input_y, output_z, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input_x, data_input_y, res]}
    te.lang.cce.cce_build_code(sch, config)
