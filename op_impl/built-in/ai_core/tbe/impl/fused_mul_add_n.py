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
fused_mul_add_n
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.utils import op_utils
from topi import generic


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
def mul_add_n_compute(data_x, data_y, data_z):
    """
    fused mul+add_n, output = x * z + y
    res : output of the data's mul+add_n
    """
    data_z = te.lang.cce.broadcast(data_z, data_x.shape)
    res = te.lang.cce.vmul(data_x, data_z)
    res = te.lang.cce.vadd(data_y, res)
    return res


@op_utils.check_op_params(op_utils.REQUIRED_INPUT, op_utils.REQUIRED_INPUT,
                          op_utils.REQUIRED_INPUT, op_utils.REQUIRED_OUTPUT,
                          op_utils.KERNEL_NAME)
def fused_mul_add_n(input_x,
                    input_y,
                    input_z,
                    output,
                    kernel_name="fused_mul_add_n"):
    """
    algorithm: fused mul+add_n
    calculating output = input_x * input_z + input_y

    Parameters
    ----------
    input_x : dict of input_x, tensor
    input_y: dict of input_y, tensor
    input_z: dict of input_z, scalar
    output : dict of output

    kernel_name : string
        cce kernel name, default value is fused_mul_add_n

    Returns
    -------
    None
    """

    check_list = ("float16", "float32", "int32", "int16")
    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype")
    op_utils.check_shape(shape_x, param_name="input_x")
    op_utils.check_dtype(dtype_x, check_list, param_name="input_x")
    shape_y = input_y.get("shape")
    dtype_y = input_y.get("dtype")
    op_utils.check_shape(shape_y, param_name="input_y")
    op_utils.check_dtype(dtype_y, check_list, param_name="input_y")
    dtype_z = input_z.get("dtype")
    shape_z = [1 for i in range(len(shape_x))]
    op_utils.check_shape(shape_z, param_name="input_z")
    op_utils.check_dtype(dtype_z, check_list, param_name="input_z")

    data_x = tvm.placeholder(shape_x, name="input_x", dtype=dtype_x)
    data_y = tvm.placeholder(shape_y, name="input_y", dtype=dtype_y)
    data_z = tvm.placeholder(shape_z, name="input_z", dtype=dtype_z)

    res = mul_add_n_compute(data_x, data_y, data_z)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    tensor_list = [data_x, data_y, data_z, res]

    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }

    te.lang.cce.cce_build_code(schedule, config)
