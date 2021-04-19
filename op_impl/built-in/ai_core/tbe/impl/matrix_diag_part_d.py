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
matrix_diag_part_d
"""
from __future__ import absolute_import

from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import *

# define a scaler, value = -2
SCALER_NEGATIVE_TWO = -2


# pylint: disable=locally-disabled,unused-argument
@fusion_manager.register("matrix_diag_part_d")
def matrix_diag_part_d_compute(input_diagonal, input_help, output_diagonal,
                               kernel_name="matrix_diag_part_d"):
    """
    compute for matrix_diag_part_d

    Parameters
    ----------
    input_diagonal: TVM tensor
        the placeholder of input diagonal
    input_help: TVM tensor
        the placeholder of input help
    output_diagonal: dict
        dict of output_diagonal
    kernel_name: str
        cce kernel name, default value is "matrix_diag_part_d"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    shape_input_diagonal = te.lang.cce.util.shape_to_list(input_diagonal.shape)
    dtype_input_diagonal = input_diagonal.dtype

    res_vmul = te.lang.cce.vmul(input_diagonal, input_help)
    if shape_input_diagonal[-2] < shape_input_diagonal[-1]:
        if dtype_input_diagonal == "int32":
            res_vmul = te.lang.cce.cast_to(res_vmul, "float32")
        res = te.lang.cce.sum(res_vmul, -1)
        if dtype_input_diagonal == "int32":
            res = te.lang.cce.cast_to(res, "int32")
    else:
        res = te.lang.cce.sum(res_vmul, SCALER_NEGATIVE_TWO)

    if dtype_input_diagonal in ("int8", "uint8"):
        res = te.lang.cce.cast_to(res, dtype_input_diagonal,
                                  f1628IntegerFlag=True)

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def matrix_diag_part_d(input_diagonal, input_help,
                       output_diagonal, kernel_name="matrix_diag_part_d"):
    """
    Returns the batched diagonal part of a batched tensor

    Parameters
    ----------
    input_diagonal: dict
        dict of input_diagonal, include keys(shape and dtype)
    input_help: dict
        dict of help Matrix, Its Diagonal Line value is 1 else value is 0
    output_diagonal: dict
        dict of output
    kernel_name: str
        cce kernel name, default value is "matrix_diag_part_d"

    Returns
    -------
    None
    """
    shape_input_diagonal = input_diagonal.get("shape")
    dtype_input_diagonal = input_diagonal.get("dtype")
    shape_input_help = input_help.get("shape")
    dtype_input_help = input_help.get("dtype")

    check_shape(shape_input_diagonal, param_name="input_diagonal")
    check_shape(shape_input_help, param_name="input_help")

    if len(shape_input_diagonal) < 2:
        raise RuntimeError("Input tensors of rank>=2 are supported!")
    if list(shape_input_diagonal) != list(shape_input_help):
        raise RuntimeError("the shape of data must be equal!")

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    dtype_input_diagonal = dtype_input_diagonal.lower()
    check_dtype(dtype_input_diagonal, check_list, param_name="input_diagonal")
    dtype_input_help = dtype_input_help.lower()
    check_dtype(dtype_input_help, check_list, param_name="input_help")

    data_input_diagonal = tvm.placeholder(shape_input_diagonal,
                                          name="data_input_diagonal",
                                          dtype=dtype_input_diagonal)
    data_input_help = tvm.placeholder(shape_input_help, name="data_input_help",
                                      dtype=dtype_input_help)

    res = matrix_diag_part_d_compute(data_input_diagonal, data_input_help,
                                     output_diagonal, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input_diagonal, data_input_help, res]}
    te.lang.cce.cce_build_code(sch, config)
