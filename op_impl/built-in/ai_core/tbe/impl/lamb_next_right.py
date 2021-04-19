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
lamb_next_right
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import *


# pylint: disable=locally-disabled,unused-variable,unused-argument
# pylint: disable=locally-disabled,too-many-locals,too-many-arguments
# pylint: disable=locally-disabled,invalid-name
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
        shape_x, shape_y, shape_max = broadcast_shapes(shape_x, shape_y,
                                                       param_name_input1="data_1",
                                                       param_name_input2="data_2")
        data_1 = te.lang.cce.broadcast(data_1, shape_max)
        data_2 = te.lang.cce.broadcast(data_2, shape_max)

    return data_1, data_2


@fusion_manager.register("lamb_next_right")
def lamb_next_right_compute(data_input_square, data_input_mul2, data_mul2_x,
                            data_mul3_x, data_truediv1_recip, data_add2_y,
                            y1, y2, kernel_name="lamb_next_right"):
    """
    calculating data

    Parameters
    ----------
    data_input_square: TVM tensor
        the input tensor of square
    data_input_mul2: TVM tensor
        the input tensor of mul_2
    data_mul2_x: TVM tensor
        the input tensor of mul_2
    data_mul3_x: TVM tensor
        the input tensor of mul_3
    data_truediv1_recip: TVM tensor
        the input tensor of truediv_1
    data_add2_y: TVM tensor
        the input tensor of add_2
    y1: TVM tensor
        the output tensor of add_1
    y2: TVM tensor
        the output tensor of add_2
    kernel_name : str
        kernel name, default value is "lamb_next_right"

    Returns
    -------
    output tensor
    """

    # square
    data_input_square, data_input_square = \
        shape_broadcast(data_input_square, data_input_square)
    square_result = te.lang.cce.vmul(data_input_square, data_input_square)

    # mul_3
    square_result, data_mul3_x = shape_broadcast(square_result,
                                                 data_mul3_x)
    mul_3_result = te.lang.cce.vmul(square_result, data_mul3_x)

    # mul_2
    data_input_mul2, data_mul2_x = shape_broadcast(data_input_mul2,
                                                   data_mul2_x)
    mul_2_result = te.lang.cce.vmul(data_input_mul2, data_mul2_x)

    # add_1
    mul_3_result, mul_2_result = shape_broadcast(mul_3_result, mul_2_result)
    output0 = te.lang.cce.vadd(mul_2_result, mul_3_result)

    # truediv_1-->vmul
    data_truediv1_recip, output0 = shape_broadcast(data_truediv1_recip, output0)
    truediv_1_result = te.lang.cce.vmul(output0, data_truediv1_recip)

    # sqrt
    sqrt_result = te.lang.cce.vsqrt(truediv_1_result)

    # add_2
    sqrt_result, data_add2_y = shape_broadcast(sqrt_result,
                                               data_add2_y)
    output1 = te.lang.cce.vadd(data_add2_y, sqrt_result)

    res = [output0, output1]

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT,
                 REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_OUTPUT,
                 KERNEL_NAME)
def lamb_next_right(input_square, input_mul2, mul2_x, mul3_x,
                    truediv1_recip, add2_y,
                    y1, y2, kernel_name="lamb_next_right"):
    """
    calculating data

    Parameters
    ----------
    input_square: dict
        the dict of input of square, and dtype supports 'float16', 'float32'
    input_mul2: dict
        the dict of input of mul_2, and dtype supports 'float16', 'float32'
    mul2_x: dict
        the dict of input of mul_2, and dtype supports 'float16', 'float32'
    mul3_x: dict
        the dict of input of mul_3, and dtype supports 'float16', 'float32'
    truediv1_recip: dict
        the dict of input of truediv1, and dtype supports 'float16', 'float32'
    add2_y: dict
        the dict of input of add_2, and dtype supports 'float16', 'float32'
    y1: dict
        the dict of output of add_1, and dtype supports 'float16', 'float32'
    y2: dict
        the dict of output of add_2, and dtype supports 'float16', 'float32'
    kernel_name: str
        kernel name, default value is lamb_next_right

    Returns
    -------
    None
    """
    shape_input_square = util.scalar2tensor_one(input_square.get("shape"))
    shape_input_mul2 = util.scalar2tensor_one(input_mul2.get("shape"))
    shape_mul2_x = util.scalar2tensor_one(mul2_x.get("shape"))
    shape_mul3_x = util.scalar2tensor_one(mul3_x.get("shape"))
    shape_truediv1_recip = util.scalar2tensor_one(truediv1_recip.get("shape"))
    shape_add2_y = util.scalar2tensor_one(add2_y.get("shape"))

    dtype_input_square = input_square.get("dtype").lower()
    dtype_input_mul2 = input_mul2.get("dtype").lower()
    dtype_mul2_x = mul2_x.get("dtype").lower()
    dtype_mul3_x = mul3_x.get("dtype").lower()
    dtype_truediv1_recip = truediv1_recip.get("dtype").lower()
    dtype_add2_y = add2_y.get("dtype").lower()


    # broadcast
    shape_input_square, shape_mul3_x, shape_max_mul3 = \
        broadcast_shapes(shape_input_square, shape_mul3_x, param_name_input1="input_square",
                         param_name_input2="mul3_x")
    shape_input_mul2, shape_mul2_x, shape_max_mul2 = \
        broadcast_shapes(shape_input_mul2, shape_mul2_x, param_name_input1="input_mul2",
                         param_name_input2="mul2_x")
    shape_max_mul2, shape_max_mul3, shape_max_add1 = \
        broadcast_shapes(shape_max_mul2, shape_max_mul3, param_name_input1="shape_max_mul2",
                         param_name_input2="shape_max_mul3")
    shape_max_add1, shape_truediv1_recip, shape_max_truediv1 = \
        broadcast_shapes(shape_max_add1, shape_truediv1_recip, param_name_input1="shape_max_add1",
                         param_name_input2="truediv1_recip")
    shape_max_truediv1, shape_add2_y, shape_max_add2 = \
        broadcast_shapes(shape_max_truediv1, shape_add2_y, param_name_input1="shape_max_truediv1",
                         param_name_input2="add2_y")

    data_input_square = tvm.placeholder(shape_input_square,
                                        name="data_input_square",
                                        dtype=dtype_input_square)
    data_input_mul2 = tvm.placeholder(shape_input_mul2,
                                      name="data_input_mul2",
                                      dtype=dtype_input_mul2)
    data_mul2_x = tvm.placeholder(shape_mul2_x,
                                  name="data_mul2_x",
                                  dtype=dtype_mul2_x)
    data_mul3_x = tvm.placeholder(shape_mul3_x,
                                  name="data_mul3_x",
                                  dtype=dtype_mul3_x)
    data_truediv1_recip = tvm.placeholder(shape_truediv1_recip,
                                          name="data_truediv1_recip",
                                          dtype=dtype_truediv1_recip)
    data_add2_y = tvm.placeholder(shape_add2_y,
                                  name="data_add2_y",
                                  dtype=dtype_add2_y)

    res = lamb_next_right_compute(data_input_square, data_input_mul2,
                                  data_mul2_x, data_mul3_x,
                                  data_truediv1_recip, data_add2_y,
                                  y1, y2, kernel_name)

    inputlist = [data_input_square, data_input_mul2, data_mul2_x,
                 data_mul3_x, data_truediv1_recip, data_add2_y]

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": list(inputlist) + list(res)}

    te.lang.cce.cce_build_code(sch, config)
