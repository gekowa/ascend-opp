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
relu_grad
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import refine_shapes_for_broadcast
from topi import generic
from topi.cce import util
from te.utils.op_utils import *


# pylint: disable=locally-disabled,unused-argument
@fusion_manager.register("relu_grad")
def relu_grad_compute(input_gradients, input_features, output_backprops,
                      kernel_name="relu_grad"):
    """
    calculate the backpropagation of relu operation
    output_backprops = input_gradients*1(input_features>0) or 0(input_features<=0).

    Parameters
    ----------
    input_gradients: TVM tensor
        input tensor of grad
    input_features: TVM tensor
        input tensor of relu output
    output_backprops: dict
        output dict of relu grad
    kernel_name: str
        cce kernel name, default value is "relu_grad"

    Returns
    -------
    res: TVM tensor
        the result of relu_grad_compute
    """
    dtype = input_gradients.dtype
    trans_type = dtype
    shape_input_gradients = te.lang.cce.util.shape_to_list(input_gradients.shape)
    shape_input_features = te.lang.cce.util.shape_to_list(input_features.shape)
    shape = shape_input_gradients

    # need cast int8 or uint8 to float16
    if dtype in ("int8", "uint8"):
        input_gradients = te.lang.cce.cast_to(input_gradients, "float16")
        input_features = te.lang.cce.cast_to(input_features, "float16")
        trans_type = "float16"

    # broadcast in case the input shapes are not same
    if list(shape_input_gradients) != list(shape_input_features):
        shape_input_gradients, shape_input_features, shape = \
            broadcast_shapes(shape_input_gradients, shape_input_features,
                             param_name_input1="input_gradients",
                             param_name_input2="input_features")
        input_gradients = te.lang.cce.broadcast(input_gradients, shape,
                                                trans_type)
        input_features = te.lang.cce.broadcast(input_features, shape,
                                               trans_type)

    derivative_relu = te.lang.cce.calculate_one_or_zero(input_features,
                                                        shape, trans_type)

    result = te.lang.cce.vmul(input_gradients, derivative_relu)

    # cast int8 or uint8 back
    if dtype in ("int8", "uint8"):
        result = te.lang.cce.cast_to(result, dtype, f1628IntegerFlag=True)

    return result


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def relu_grad(input_gradients, input_features, output_backprops,
              kernel_name="relu_grad"):
    """
    calculate the backpropagation of relu operation
    output_backprops = input_gradients*1(input_features>0) or 0(input_features<=0).
    support dtype:float16,float32,int32,int8,uint8

    Parameters
    ----------
    input_gradients: dict
        the backpropagated gradients to the corresponding relu operation
    input_features: dict
        the features passed as output of relu operation
    output_backprops: dict
        the output of relu back propagation
    kernel_name: str
        cce kernel name, default value is "relu_grad"

    Returns
    -------
    None
    """
    shape_input_gradients = input_gradients.get("shape")
    shape_input_features = input_features.get("shape")

    util.compare_tensor_dict_key(input_gradients, input_features, "dtype")
    check_shape(shape_input_gradients, param_name="input_gradients")
    check_shape(shape_input_features, param_name="input_features")

    if list(shape_input_gradients) != list(shape_input_features):
        shape_input_gradients, shape_input_features, shape_max = \
            broadcast_shapes(shape_input_gradients, shape_input_features,
                             param_name_input1="input_gradients",
                             param_name_input2="input_features")

    dtype_input_gradients = input_gradients.get("dtype").lower()
    dtype_input_features = input_features.get("dtype").lower()

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    check_dtype(dtype_input_gradients, check_list, param_name="input_gradients")
    check_dtype(dtype_input_features, check_list, param_name="input_features")

    shape_input_gradients, shape_input_features = \
        refine_shapes_for_broadcast(shape_input_gradients,
                                    shape_input_features)
    data_input_gradients = tvm.placeholder(shape_input_gradients,
                                           name="data_input_gradients",
                                           dtype=dtype_input_gradients)
    data_input_features = tvm.placeholder(shape_input_features,
                                          name="data_input_features",
                                          dtype=dtype_input_features)

    res = relu_grad_compute(data_input_gradients, data_input_features,
                            output_backprops, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input_gradients, data_input_features, res]}
    te.lang.cce.cce_build_code(sch, config)
