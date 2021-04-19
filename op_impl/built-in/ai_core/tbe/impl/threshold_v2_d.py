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
threshold_v2_d
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from functools import reduce as reduceIns
from te.utils.op_utils import check_op_params
from te.utils.op_utils import check_dtype
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import REQUIRED_ATTR_FLOAT
from te.utils.op_utils import KERNEL_NAME


@fusion_manager.register("threshold_v2_d")
# pylint: disable=invalid-name
def threshold_v2_d_compute(x, y, threshold, value,
                           kernel_name="threshold_v2_d_cce"):
    """
    Thresholds each element of the input Tensor
    y = (x > threshold) ? x : value

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    threshold : float
        scale value to threshold at
    value : float
        scale value to replace with
    kernel_name : str
        kernel name, default value is "threshold_v2_d_cce"

    Returns
    -------
    output tensor
    """
    dtype_x = x.dtype

    threshold = tvm.const(threshold, dtype_x)
    value = tvm.const(value, dtype_x)

    data_res = te.lang.cce.vcmpsel(x, threshold, operation='gt', slhs=x, srhs=value)
    return data_res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_FLOAT,
                 REQUIRED_ATTR_FLOAT, KERNEL_NAME)
# pylint: disable=invalid-name
def threshold_v2_d(x, y, threshold, value, kernel_name="threshold_v2_d_cce"):
    """
    Thresholds each element of the input Tensor
    y = (x > threshold) ? x : value

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    threshold : float
        scale value to threshold at
    value : float
        scale value to replace with
    kernel_name : str
        kernel name, default value is "threshold_v2_d_cce"

    Returns
    -------
    output tensor
    """

    # get the shape and dtype
    shape_x = x.get("shape")
    dtype_x = x.get("dtype").lower()

    # check whether dtypes are right
    check_list = ("float16", "float32", "int8", "uint8", "int32")
    check_dtype(dtype_x, check_list)

    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x*y, shape_x)

    data_x = tvm.placeholder(shape=fuseshape, name="data_x", dtype=dtype_x)
    res = threshold_v2_d_compute(data_x, y, threshold, value, kernel_name)
    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_x, res]}
    te.lang.cce.cce_build_code(schedule, config)
