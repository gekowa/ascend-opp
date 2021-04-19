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


import te.lang.dynamic
from te import tvm
from topi import generic
from functools import reduce as functools_reduce
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_op_params
from te.utils.op_utils import variable_shape
from te.platform.shape_classifier import classify
from te.platform.shape_classifier import Mode


def fill_compute(dims, value, dtype, kernel_name="fill"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of input
    value : a number of float or int
    dtype : string
        the type of input
    kernel_name : str
        kernel name, default value is "fills"

    Returns
    -------
    res: TVM tensor
        the calculation results
    """

    res = te.lang.dynamic.broadcast(value, dims)

    return res


@te.op.register_operator("Fill")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def fill(dims, value, y, kernel_name="fill"):
    """
    do  fill operation

    Parameters:
    ----------
    dims : the dict of input
    value :  the dict of input
    y:  the dict of output
    kernel_name : cce kernel name, default value is "fill"

    Returns
    -------
    None
    """
    # get the shape and dtype
    shape = value.get("shape")
    dtype = value.get("dtype").lower()
    dtype_dims = dims.get("dtype").lower()
    dims["shape"] = [-1]
    dims['range'] = [[1, None]]

    # check whether dtypes are right
    check_list = ("int32", "float16", "float32")
    check_dtype(dtype, check_list)

    schedules, tensors = [], []

    with te.op.compute():
        shape_dim = variable_shape([dims])
        x_input = tvm.placeholder(shape, name="x_input", dtype=dtype)
        dim_input = tvm.placeholder(shape_dim[0], name="dim_input", dtype=dtype_dims)

        res = fill_compute(shape_dim[0], x_input, y, kernel_name=kernel_name)
        tensors.append([dim_input, x_input, res])
    with tvm.target.cce():
        sch = generic.auto_schedule(res)
    schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}

    te.lang.dynamic.build(schedules, config)
    te.op.add_compile_info("_use_special_pattern", False)