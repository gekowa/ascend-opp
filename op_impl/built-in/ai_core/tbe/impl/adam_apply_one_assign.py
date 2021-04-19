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
adam_apply_one_assign
"""
import uuid
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import *


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=locally-disabled,too-many-locals,unused-variable
def shape_broadcast(data_1, data_2):
    """
    broadcast the two input

    Parameters
    ----------
    data_1: TVM tensor
        the placeholder of first input data
    data_2: TVM tensor
        the placeholder of second input data
    output_z: dict
        shape and dtype of output, should be broadcast shape and type as input

    Returns
    -------
    res : output of the data's divide
    """
    shape_x = te.lang.cce.util.shape_to_list(data_1.shape)
    shape_y = te.lang.cce.util.shape_to_list(data_2.shape)
    if shape_x != shape_y:
        shape_x, shape_y, shape_max = broadcast_shapes(shape_x, shape_y, param_name_input1="data_1", param_name_input2="data_2")
        data_1 = te.lang.cce.broadcast(data_1, shape_max)
        data_2 = te.lang.cce.broadcast(data_2, shape_max)

    return data_1, data_2


@fusion_manager.register("adam_apply_one_assign")
def adam_apply_one_assign_compute(data_input0, data_input1, data_input2, data_input3,
                           data_input4, data_input_mul, data_input_mul1,
                           data_input_mul2, data_input_mul3, data_input_add2,
                           output0, output1, output2,
                           kernel_name="adam_apply_one_assign"):
    """
    apply one adam calculation function

    Parameters
    ----------
    data_input0: TVM tensor
         the input tensor of square and mul_1
    data_input1: TVM tensor
         the input tensor of mul_2
    data_input2: TVM tensor
         the input tensor of mul_0
    data_input3: TVM tensor
         the input tensor of sub
    data_input4: TVM tensor
         the input tensor of mul_4
    data_input_mul: TVM tensor
         the input tensor of mul_0
    data_input_mul1: TVM tensor
         the input tensor of mul_1
    data_input_mul2: TVM tensor
         the input tensor of mul_2
    data_input_mul3: TVM tensor
         the input tensor of mul_3
    data_input_add2: TVM tensor
         the input tensor of mul_3
    output0: TVM tensor
         the output tensor of add_1
    output1: TVM tensor
         the output tensor of add_0
    output2: TVM tensor
         the output tensor of sub
    kernel_name : str
        kernel name, default value is "adam_apply_one_assign"

    Returns
    -------
    output tensor
    """

    # square
    data_input0, data_input0 = shape_broadcast(data_input0, data_input0)
    square_result = te.lang.cce.vmul(data_input0, data_input0)

    # mul_3
    square_result, data_input_mul3 = shape_broadcast(square_result,
                                                     data_input_mul3)
    mul_3_result = te.lang.cce.vmul(square_result, data_input_mul3)

    # mul_2
    data_input1, data_input_mul2 = shape_broadcast(data_input1,
                                                   data_input_mul2)
    mul_2_result = te.lang.cce.vmul(data_input1, data_input_mul2)

    # add_1
    mul_3_result, mul_2_result = shape_broadcast(mul_3_result, mul_2_result)
    output0 = te.lang.cce.vadd(mul_2_result, mul_3_result)

    # sqrt
    sqrt_result = te.lang.cce.vsqrt(output0)

    # add_2
    data_input_add2, sqrt_result = shape_broadcast(data_input_add2,
                                                   sqrt_result)
    add_2_result = te.lang.cce.vadd(sqrt_result, data_input_add2)

    # mul_0
    data_input2, data_input_mul = shape_broadcast(data_input2, data_input_mul)
    mul_0_result = te.lang.cce.vmul(data_input2, data_input_mul)

    # mul_1
    data_input0, data_input_mul1 = shape_broadcast(data_input0,
                                                   data_input_mul1)
    mul_1_result = te.lang.cce.vmul(data_input0, data_input_mul1)

    # add
    mul_0_result, mul_1_result = shape_broadcast(mul_0_result, mul_1_result)
    output1 = te.lang.cce.vadd(mul_0_result, mul_1_result)

    # truediv
    add_2_result, output1 = shape_broadcast(add_2_result, output1)
    truediv_result = te.lang.cce.vdiv(output1, add_2_result)

    # mul_4
    truediv_result, data_input4 = shape_broadcast(truediv_result, data_input4)
    mul_4_result = te.lang.cce.vmul(truediv_result, data_input4)

    # sub
    mul_4_result, data_input3 = shape_broadcast(mul_4_result, data_input3)
    output2 = te.lang.cce.vsub(data_input3, mul_4_result)

    res = [output0, output1, output2]

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT,
                 REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT,
                 REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_OUTPUT,
                 REQUIRED_OUTPUT, KERNEL_NAME)
def adam_apply_one_assign(input0, input1, input2, input3, input4,
                   mul0_x, mul1_x, mul2_x, mul3_x, add2_y,
                   output0, output1, output2, kernel_name="adam_apply_one_assign"):
    """
    function: For bert fuse

    Parameters
    ----------
    input0: dict
         the dict of input of square and mul_1,
         and dtype supports 'float16', 'float32'
    input1: dict
         the dict of input of mul_2, and dtype supports 'float16', 'float32'
    input2: dict
         the dict of input of mul, and dtype supports 'float16', 'float32'
    input3: dict
         the dict of input of sub, and dtype supports 'float16', 'float32'
    input4: dict
         the dict of input of mul_4, and dtype supports 'float16', 'float32'
    mul0_x: dict
         the dict of input of mul_0, and dtype supports 'float16', 'float32'
    mul1_x: dict
         the dict of input of mul_1, and dtype supports 'float16', 'float32'
    mul2_x: dict
         the dict of input of mul_2, and dtype supports 'float16', 'float32'
    mul3_x: dict
         the dict of input of mul_3, and dtype supports 'float16', 'float32'
    add2_y: dict
         the dict of input of add_2, and dtype supports 'float16', 'float32'
    output0: dict
         the dict of output of add_1, and dtype supports 'float16', 'float32'
    output1: dict
         the dict of output of add_0, and dtype supports 'float16', 'float32'
    output2: dict
         the dict of output of sub, and dtype supports 'float16', 'float32'
    kernel_name: str
        cce kernel name, default value is adam_apply_one_assign

    Returns
    -------
    None
    """
    shape_input0 = util.scalar2tensor_one(input0.get("shape"))
    shape_input1 = util.scalar2tensor_one(input1.get("shape"))
    shape_input2 = util.scalar2tensor_one(input2.get("shape"))
    shape_input3 = util.scalar2tensor_one(input3.get("shape"))
    shape_input4 = util.scalar2tensor_one(input4.get("shape"))
    shape_mul0_x = util.scalar2tensor_one(mul0_x.get("shape"))
    shape_mul1_x = util.scalar2tensor_one(mul1_x.get("shape"))
    shape_mul2_x = util.scalar2tensor_one(mul2_x.get("shape"))
    shape_mul3_x = util.scalar2tensor_one(mul3_x.get("shape"))
    shape_add2_y = util.scalar2tensor_one(add2_y.get("shape"))

    dtype_input0 = input0.get("dtype").lower()
    dtype_input1 = input1.get("dtype").lower()
    dtype_input2 = input2.get("dtype").lower()
    dtype_input3 = input3.get("dtype").lower()
    dtype_input4 = input4.get("dtype").lower()
    dtype_mul0_x = mul0_x.get("dtype").lower()
    dtype_mul1_x = mul1_x.get("dtype").lower()
    dtype_mul2_x = mul2_x.get("dtype").lower()
    dtype_mul3_x = mul3_x.get("dtype").lower()
    dtype_add2_y = add2_y.get("dtype").lower()


    shape_input0, shape_mul3_x, shape_max_mul3 = \
        broadcast_shapes(shape_input0, shape_mul3_x, param_name_input1="input0", param_name_input2="mul3_x")
    shape_input1, shape_mul2_x, shape_max_mul2 = \
        broadcast_shapes(shape_input1, shape_mul2_x, param_name_input1="input1", param_name_input2="mul2_x")
    shape_input1, shape_add2_y, shape_max_add2 = \
        broadcast_shapes(shape_input1, shape_add2_y, param_name_input1="input1", param_name_input2="add2_y")
    shape_input1, shape_input4, shape_max_mul4 = \
        broadcast_shapes(shape_input1, shape_input4, param_name_input1="input1", param_name_input2="input4")
    shape_input1, shape_input3, shape_max_sub = \
        broadcast_shapes(shape_input1, shape_input3, param_name_input1="input1", param_name_input2="input3")
    shape_input2, shape_mul0_x, shape_max_mul0 = \
        broadcast_shapes(shape_input2, shape_mul0_x, param_name_input1="input2", param_name_input2="mul0_x")
    shape_input0, shape_mul1_x, shape_max_mul1 = \
        broadcast_shapes(shape_input0, shape_mul1_x, param_name_input1="input0", param_name_input2="mul1_x")

    if kernel_name == "adam_apply_one_assign":
        kernel_name += \
            str(uuid.uuid4()) + str(uuid.uuid4()) + str(uuid.uuid4()) + str(uuid.uuid4())
        kernel_name = kernel_name.replace('-', 'Z')
    data_input0 = tvm.placeholder(shape_input0,
                                  name="data_input0__" + kernel_name,
                                  dtype=dtype_input0)
    data_input1 = tvm.placeholder(shape_input1,
                                  name="data_input1__" + kernel_name,
                                  dtype=dtype_input1)
    data_input2 = tvm.placeholder(shape_input2,
                                  name="data_input2__" + kernel_name,
                                  dtype=dtype_input2)
    data_input3 = tvm.placeholder(shape_input3,
                                  name="data_input3__" + kernel_name,
                                  dtype=dtype_input3)
    data_input4 = tvm.placeholder(shape_input4,
                                  name="data_input4__" + kernel_name,
                                  dtype=dtype_input4)
    data_input_mul = tvm.placeholder(shape_mul0_x,
                                     name="data_input_mul__" + kernel_name,
                                     dtype=dtype_mul0_x)
    data_input_mul1 = tvm.placeholder(shape_mul1_x,
                                      name="data_input_mul1__" + kernel_name,
                                      dtype=dtype_mul1_x)
    data_input_mul2 = tvm.placeholder(shape_mul2_x,
                                      name="data_input_mul2__" + kernel_name,
                                      dtype=dtype_mul2_x)
    data_input_mul3 = tvm.placeholder(shape_mul3_x,
                                      name="data_input_mul3__" + kernel_name,
                                      dtype=dtype_mul3_x)
    data_input_add2 = tvm.placeholder(shape_add2_y,
                                      name="data_input_add2__" + kernel_name,
                                      dtype=dtype_add2_y)

    res = adam_apply_one_assign_compute(data_input0, data_input1, data_input2,
                                 data_input3, data_input4, data_input_mul,
                                 data_input_mul1, data_input_mul2,
                                 data_input_mul3, data_input_add2,
                                 output0, output1, output2, kernel_name)

    # Fusion with assign
    util.reuse_input_as_output(data_input1, res[0], kernel_name)
    util.reuse_input_as_output(data_input2, res[1], kernel_name)
    util.reuse_input_as_output(data_input3, res[2], kernel_name)

    inputlist = [data_input0, data_input1, data_input2, data_input3,
                 data_input4, data_input_mul, data_input_mul1,
                 data_input_mul2, data_input_mul3, data_input_add2]

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": list(inputlist) + list(res),
              "dummy_placeholder": True}

    te.lang.cce.cce_build_code(sch, config)
