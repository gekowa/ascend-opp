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
fill

  Op_description :
    This operation creates a tensor of shape `dims` and fills it with `value`.

    # fill(
    #   x,
    #   y,
    #   value,
    #   kernel_name='fill'
    # )

  Supportive_dtype_format :
    ['int32', 'float32', 'float16']
    all format

  Constraint :
    [1] All : shape size limit is 2147483648.
"""
from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from functools import reduce as reduceIns
from te.utils.op_utils import *


@fusion_manager.register("fills")
def fills_compute(x, value, dtype, kernel_name="fills"):
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
    res = te.lang.cce.broadcast(tvm.const(value, dtype=dtype), x.shape)
    with tvm.tag_scope("elewise_binary_phony"):
        res = te.tvm.compute(res.shape,
                             lambda *indices: res[indices] + x[indices],
                             name="elewise_binary_phony_output")

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_FLOAT, KERNEL_NAME)
def fills(x, y, value, kernel_name="fills"):
    """
    do  fill operation

    Parameters:
    ----------
    x : the dict of output
    y :  the dict of output
    value:  scalar  value,
    kernel_name : cce kernel name, default value is "fill"

    Returns
    -------
    None
    """
    # get the shape and dtype
    shape = x.get("shape")
    dtype = x.get("dtype").lower()

    # check whether dtypes are right
    check_list = ("int32", "float16", "float32")
    check_dtype(dtype, check_list)

    # fuse shapes
    shape = util.shape_refine(shape)
    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x * y, shape)
    data_x = tvm.placeholder(fuseshape, name="data_x", dtype=dtype)

    res = fills_compute(data_x, value, dtype)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": (data_x, res),
        "print_ir": False
    }
    te.lang.cce.cce_build_code(sch, config)
