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
inplace_sub_d
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import *


# pylint: disable = locally-disabled,invalid-name
# pylint: disable = too-many-arguments,unused-argument,no-member
@fusion_manager.register("inplace_sub_d")
def inplace_sub_d_compute(x, v, y, indices, kernel_name="inplace_sub_d"):
    """
    inplace_sub_d compute process

    Parameters
    ----------
    x : TVM tensor
        the placeholder of x
    v : TVM tensor.
        the placeholder of v
    y : dict
        dict with keys(shape and dtype) of output
    indices : a vector.
        indices into the left-most dimension of x
    kernel_name : str
        kernel name, default value is "inplace_sub_d_d"

    Returns
    -------
    output tensor
    """

    res = te.lang.cce.inplace_sub(x, indices, v)

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_LIST_INT,
                 KERNEL_NAME)
def inplace_sub_d(x, v, y, indices, kernel_name="inplace_sub_d"):
    """
    algorithm: inplacea_add_d

    Parameters
    ----------
    x : TVM tensor
        the placeholder of x
    v : TVM tensor.
        the placeholder of v
    y : dict
        dict with keys(shape and dtype) of output
    indices : a vector.
        indices into the left-most dimension of x
    kernel_name : str
        kernel name, default value is "inplace_sub_d"

    Returns
    -------
    None
    """
    check_tuple = ("float16", "float32", "int32")

    shape_x = x.get("shape")
    shape_v = v.get("shape")

    check_shape(shape_x, param_name="x")
    check_shape(shape_v, param_name="v")
    check_dtype(x.get("dtype").lower(), check_tuple, param_name="x")
    check_dtype(v.get("dtype").lower(), check_tuple, param_name="v")
    indices = list(indices)

    if len(shape_x) != len(shape_v):
        raise RuntimeError("The number of dimension x must be"
                           " same as dimension v")

    if shape_v[0] != len(indices):
        raise RuntimeError("The length of rank 0 of tensor v must"
                           " be the same as length of indices")

    for i in range(1, len(shape_v)):
        if shape_x[i] != shape_v[i]:
            raise RuntimeError("The length of each rank of tensor x must "
                               "be the same as length of each rank of "
                               "tensor v except the first dimension")

    for i, _ in enumerate(indices):
        indices[i] = (indices[i] % shape_x[0] + shape_x[0]) % shape_x[0]

    data_x = tvm.placeholder(shape_x, name="data_x",
                             dtype=x.get("dtype").lower())
    data_v = tvm.placeholder(shape_v, name="data_v",
                             dtype=v.get("dtype").lower())

    res = inplace_sub_d_compute(data_x, data_v, y,
                                indices, kernel_name="inplace_sub_d")

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, data_v, res]}

    te.lang.cce.cce_build_code(sch, config)
