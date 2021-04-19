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
relu_grad_v2
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import *


# pylint: disable=locally-disabled,unused-argument
@fusion_manager.register("relu_grad_v2")
def relu_grad_v2_compute(gradients, mask, backprops,
                         kernel_name="relu_grad_v2"):
    """
    calculate the backpropagation of relu operation
    output_backprops = input_gradients*1(input_features>0) or 0(input_features<=0).

    Parameters
    ----------
    gradients: TVM tensor
        input tensor of grad
    mask: TVM tensor
        input tensor of relu output
    backprops: dict
        output dict of relu grad
    kernel_name: str
        cce kernel name, default value is "relu_grad_v2"

    Returns
    -------
    res: TVM tensor
        the result of relu_grad_compute
    """
    dtype = gradients.dtype
    trans_type = dtype

    # need cast int8 or uint8 to float16
    if dtype in ("int8", "uint8"):
        gradients = te.lang.cce.cast_to(gradients, "float16")
        trans_type = "float16"

    result = te.lang.cce.vsel(mask, gradients, tvm.const(0, trans_type))

    # cast int8 or uint8 back
    if dtype in ("int8", "uint8"):
        result = te.lang.cce.cast_to(result, dtype, f1628IntegerFlag=True)

    return result


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def relu_grad_v2(gradients, mask, backprops, kernel_name="relu_grad_v2"):
    """
    calculate the backpropagation of relu operation
    output_backprops = input_gradients*1(input_features>0) or 0(input_features<=0).
    support dtype:float16,float32,int32,int8,uint8

    Parameters
    ----------
    gradients: dict
        dict of grad
    mask: dict
        dict of relu output mask
    backprops: dict
        output of relu grad
    kernel_name: str
        cce kernel name, default value is "relu_grad_v2"

    Returns
    -------
    None
    """
    shape_input_gradients = gradients.get("shape")
    shape_input_features = mask.get("shape")

    check_shape(shape_input_gradients, param_name="gradients")
    check_shape(shape_input_features, param_name="mask")

    dtype_input_gradients = gradients.get("dtype").lower()
    dtype_input_features = mask.get("dtype").lower()

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    check_dtype(dtype_input_gradients, check_list, param_name="gradients")
    check_dtype(dtype_input_features, ("uint8"), param_name="mask")

    shape_in = list(shape_input_gradients)

    # make sure the last dim of input feature is 8's multipules
    if shape_in[-1] % 8 != 0:
        shape_in[-1] = (shape_in[-1] + 7) // 8 * 8

    data_input_gradients = tvm.placeholder(tuple(shape_in),
                                           name="data_input_gradients",
                                           dtype=dtype_input_gradients)
    data_input_features = tvm.placeholder(shape_input_features,
                                          name="data_input_features",
                                          dtype=dtype_input_features)

    res = relu_grad_v2_compute(data_input_gradients, data_input_features,
                               backprops, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input_gradients, data_input_features, res]}
    te.lang.cce.cce_build_code(sch, config)
