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
pack
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from impl.concat_v2_d import concat_v2_d
from te.utils.op_utils import *


# pylint: disable = locally-disabled,invalid-name,too-many-arguments
# pylint: disable = unused-argument,no-member

@check_op_params(DYNAMIC_INPUT, REQUIRED_OUTPUT, OPTION_ATTR_INT, KERNEL_NAME)
def check_supported(x, y, axis, kernel_name="pack"):
    """
    support aicpu route
    """
    if axis == -1 or axis == len((x[0].get("shape"))):
        return False
    return True


def pack(x, y, axis, kernel_name="pack"):
    """
    algorithm: pack
    Concatenates tensors along one dimension.
    Parameters
    ----------
    x : A list of `dict`.dict include keys shape and dtype
    y: dict of output_data, dict include keys shape and dtype
    axis : int, in the range [-rank(values), rank(values)
    kernel_name : cce kernel name, default value is "pack"
    Returns
    -------
    None
    """
    check_list = ("int8", "int16", "int32", "int64", "uint8",
                  "uint16", "uint32", "uint64", "float16", "float32")
    data = []
    for i, input_dict in enumerate(x):
        shape_input = input_dict.get("shape")
        check_shape(shape_input, param_name="x")
        check_dtype(input_dict.get("dtype").lower(), check_list, param_name="x")
        input_dtype = (input_dict.get("dtype")).lower()
        data.append(tvm.placeholder(shape_input, name="data_%d" % i,
                                    dtype=input_dtype))

    if axis < -len((x[0].get("shape")))-1 or axis > len((x[0].get("shape"))):
        raise RuntimeError(
            "pack axis must be in [-%d , %d), "
            "actual is %d" % (len(x[0].get("shape"))+1,
                              len(x[0].get("shape"))+1, axis))

    if axis < -1:
        axis = axis + 1
    concat_v2_d(x, y, axis, kernel_name)
