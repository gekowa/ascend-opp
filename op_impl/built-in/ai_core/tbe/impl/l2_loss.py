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
l2_loss
"""
import te.lang.cce
from te import tvm
from topi import generic
from topi.cce import util
from te.utils.op_utils import *

SHAPE_SIZE_LIMIT = 2147483648  # shape limit

# pylint: disable=invalid-name
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def l2_loss(x, y, kernel_name="l2_loss"):
    """
    Reduce a tensor on a certain axis, and scale output with coeff

    Parameters
    ----------
    shape : shape of data

    dtype : source data type, only support float16, float32

    kernel_name : kernel name, default value is "l2_loss"

    Returns
    -------
    None

    """
    shape = x.get("shape")
    dtype = x.get("dtype")

    check_shape(shape, param_name="x")

    check_list = ["float16", "float32"]
    if not dtype.lower() in check_list:
        raise RuntimeError(
            "l2_loss only support float16 float32")

    shape, axis = util.simplify_axis_shape(shape, range(len(shape)))

    inp_dtype = dtype.lower()
    data_input = tvm.placeholder(shape, name="data_input", dtype=inp_dtype)

    coeff_sqrt = tvm.const(1.0 / (2**(0.5)), dtype=inp_dtype)

    data_mul = te.lang.cce.vmuls(data_input, coeff_sqrt)
    data_sqr = te.lang.cce.vmul(data_mul, data_mul)
    res = te.lang.cce.sum(data_sqr, axis)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}
    te.lang.cce.cce_build_code(sch, config)
