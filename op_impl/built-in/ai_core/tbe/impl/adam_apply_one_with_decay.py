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
adam_apply_one_with_decay
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te import platform as tbe_platform
from te.utils.op_utils import *

# shape size limit
SHAPE_SIZE_LIMIT = 2**30


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=locally-disabled,too-many-locals,too-many-statements
# pylint: disable=locally-disabled,invalid-name,too-many-locals
def square_compute(x, kernel_name="square"):
    """
    calculating data's square,y= x*x

    Parameters
    ----------
    x: TVM tensor
        the placeholder of input data
    kernel_name: str
        cce kernel name, default value is "square"

    Returns
    -------
    res: the result of square
    """
    res = te.lang.cce.vmul(x, x)
    return res


def mul_compute(x1, x2, kernel_name="mul"):
    """
   calculating data's element-wise mul, c = a .* b

   Parameters
   ----------
   x1: TVM tensor
       the placeholder of first input data
   x2: TVM tensor
       the placeholder of second input data
   kernel_name: str
       cce kernel name, default value is "mul"

   Returns
   -------
   res: output of the data's element-wise mul
   """
    shape_x1 = te.lang.cce.util.shape_to_list(x1.shape)
    shape_x2 = te.lang.cce.util.shape_to_list(x2.shape)

    shape_x1, shape_x2, shape_max = broadcast_shapes(shape_x1, shape_x2, param_name_input1="x1", param_name_input2="x2")
    data_x1 = te.lang.cce.broadcast(x1, shape_max)
    data_x2 = te.lang.cce.broadcast(x2, shape_max)
    res = te.lang.cce.vmul(data_x1, data_x2)

    return res


def add_compute(x1, x2, kernel_name="add"):
    """
   calculating data's element-wise add, c = a + b

   Parameters
   ----------
   x1: TVM tensor
       the placeholder of first input data
   x2: TVM tensor
       the placeholder of second input data
   kernel_name: str
       cce kernel name, default value is "add"

   Returns
   -------
   res: output of the data's add
   """
    shape_x1 = te.lang.cce.util.shape_to_list(x1.shape)
    shape_x2 = te.lang.cce.util.shape_to_list(x2.shape)

    shape_x1, shape_x2, shape_max = broadcast_shapes(shape_x1, shape_x2, param_name_input1="x1", param_name_input2="x2")
    data_x1 = te.lang.cce.broadcast(x1, shape_max)
    data_x2 = te.lang.cce.broadcast(x2, shape_max)
    res = te.lang.cce.vadd(data_x1, data_x2)

    return res


def sqrt_compute(x, kernel_name="sqrt"):
    """
    calculating data sqrt,y= x**0.5, mini not support vsqrt, use exp(0.5*log(x))

    Parameters
    ----------
    x: TVM tensor
        the placeholder of input data
    kernel_name: str
        cce kernel name, default value is sqrt

    Returns
    -------
    res:  the result of sqrt
    """
    input_dtype = x.dtype
    has_improve_precision = False
    if input_dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vsqrt",
                                                    "float32"):
        x = te.lang.cce.cast_to(x, "float32")
        has_improve_precision = True

    res = te.lang.cce.vsqrt(x)

    if has_improve_precision:
        res = te.lang.cce.cast_to(res, "float16")

    return res


def true_div_compute(x1, x2, kernel_name="true_div"):
    """
    calculating data's realdiv, y = x1 / x2

    Parameters
    ----------
    x1: TVM tensor
        the placeholder of first input data
    x2: TVM tensor
        the placeholder of second input data
    kernel_name: str
        cce kernel name, default value is "true_div"

    Returns
    -------
    res: output of the data's divide
    """
    shape_x1 = te.lang.cce.util.shape_to_list(x1.shape)
    shape_x2 = te.lang.cce.util.shape_to_list(x2.shape)

    shape_x1, shape_x2, shape_max = broadcast_shapes(shape_x1, shape_x2, param_name_input1="x1", param_name_input2="x2")
    data_x1 = te.lang.cce.broadcast(x1, shape_max)
    data_x2 = te.lang.cce.broadcast(x2, shape_max)

    res = te.lang.cce.vdiv(data_x1, data_x2)

    return res


def sub_compute(x1, x2, kernel_name="sub"):
    """
   calculating data's sub, c = a - b

   Parameters
   ----------
   x1: TVM tensor
       the placeholder of first input data
   x2: TVM tensor
       the placeholder of second input data
   kernel_name: str
       cce kernel name, default value is "sub"

   Returns
   -------
   res : output of the data's sub
   """
    shape_x1 = te.lang.cce.util.shape_to_list(x1.shape)
    shape_x2 = te.lang.cce.util.shape_to_list(x2.shape)

    shape_x1, shape_x2, shape_max = broadcast_shapes(shape_x1, shape_x2, param_name_input1="x1", param_name_input2="x2")
    data_x1 = te.lang.cce.broadcast(x1, shape_max)
    data_x2 = te.lang.cce.broadcast(x2, shape_max)
    res = te.lang.cce.vsub(data_x1, data_x2)

    return res


def _check_broadcast_shape(input0, input1, input2, input3, input4,
                           const_mul_x, const_mul1_x, const_mul2_x,
                           const_mul3_x, const_mul4_x, add2_y):
    """
    check broadcast shape

    Parameters
    ----------
    all inputs: dict
        the dict of inputs

    Returns
    -------
    the list of inputs shape after broadcast
    """
    shape0 = input0.get("shape")
    check_shape(shape0, param_name="input0")

    shape1 = input1.get("shape")
    check_shape(shape1, param_name="input1")

    shape2 = input2.get("shape")
    check_shape(shape2, param_name="input2")

    shape3 = input3.get("shape")
    check_shape(shape3, param_name="input3")

    shape4 = input4.get("shape")
    check_shape(shape4, param_name="input4")

    shapecm0 = const_mul_x.get("shape")
    check_shape(shapecm0, param_name="const_mul_x")

    shapecm1 = const_mul1_x.get("shape")
    check_shape(shapecm1, param_name="const_mul1_x")

    shapecm2 = const_mul2_x.get("shape")
    check_shape(shapecm2, param_name="const_mul2_x")

    shapecm3 = const_mul3_x.get("shape")
    check_shape(shapecm3, param_name="const_mul3_x")

    shapecm4 = const_mul4_x.get("shape")
    check_shape(shapecm4, param_name="const_mul4_x")

    shapey = add2_y.get("shape")
    check_shape(shapey, param_name="add2_y")

    # broadcast mul_3 shape
    shape0, shapecm3, shape_max_03 = broadcast_shapes(shape0, shapecm3, param_name_input1="input0", param_name_input2="const_mul3_x")
    # broadcast mul_2 shape
    shape1, shapecm2, shape_max_02 = broadcast_shapes(shape1, shapecm2, param_name_input1="input1", param_name_input2="const_mul2_x")
    # broadcast add_1 shape
    shape_max_02, shape_max_03, shape_max_add1 = broadcast_shapes(
        shape_max_02, shape_max_03, param_name_input1="shape_max_02", param_name_input2="shape_max_03")
    # broadcast add_2 shape
    shapey, shape_max_add1, shape_max_add2 = broadcast_shapes(
        shapey, shape_max_add1, param_name_input1="add2_y", param_name_input2="shape_max_add1")

    # broadcast mul_0 shape
    shape2, shapecm0, shape_max_20 = broadcast_shapes(shape2, shapecm0, param_name_input1="input2", param_name_input2="const_mul_x")
    # broadcast mul_1 shape
    shape0, shapecm1, shape_max_01 = broadcast_shapes(shape0, shapecm1, param_name_input1="input0", param_name_input2="const_mul1_x")
    # broadcast add_0 shape
    shape_max_20, shape_max_01, shape_max_add0 = broadcast_shapes(
        shape_max_20, shape_max_01, param_name_input1="shape_max_20", param_name_input2="shape_max_01")

    # broadcast truediv_0 shape
    shape_max_add0, shape_max_add2, shape_max_truediv = broadcast_shapes(
        shape_max_add0, shape_max_add2, param_name_input1="shape_max_add0", param_name_input2="shape_max_add2")

    # broadcast mul_4 shape
    shape3, shapecm4, shape_max_34 = broadcast_shapes(shape3, shapecm4, param_name_input1="input3", param_name_input2="const_mul4_x")
    # broadcast add_3 shape
    shape_max_34, shape_max_truediv, shape_max_add3 = broadcast_shapes(
        shape_max_34, shape_max_truediv, param_name_input1="shape_max_34", param_name_input2="shape_max_truediv")

    # broadcast mul_5 shape
    shape4, shape_max_add3, shape_max_4add3 = broadcast_shapes(
        shape4, shape_max_add3, param_name_input1="input4", param_name_input2="shape_max_add3")
    # broadcast sub_0 shape
    shape3, shape_max_4add3, shape_max_sub = broadcast_shapes(
        shape3, shape_max_4add3, param_name_input1="input3", param_name_input2="shape_max_4add3")

    return shape0, shape1, shape2, shape3, shape4,\
           shapecm0, shapecm1, shapecm2, shapecm3, shapecm4, shapey


@fusion_manager.register("adam_apply_one_with_decay")
def adam_apply_one_with_decay_compute(input0, input1, input2, input3, input4,
                                      const_mul_x, const_mul1_x, const_mul2_x,
                                      const_mul3_x, const_mul4_x, add2_y):
    """
    calculating data

    Parameters
    ----------
    input0: TVM tensor
        the placeholder of input0
    input1: TVM tensor
        the placeholder of input1
    input2: TVM tensor
        the placeholder of input2
    input3: TVM tensor
        the placeholder of input3
    input4: TVM tensor
        the placeholder of input4
    const_mul_x: TVM tensor
        the placeholder of const_mul_x
    const_mul1_x: TVM tensor
        the placeholder of const_mul1_x
    const_mul2_x: TVM tensor
        the placeholder of const_mul2_x
    const_mul3_x: TVM tensor
        the placeholder of const_mul3_x
    const_mul4_x: TVM tensor
        the placeholder of const_mul4_x
    add2_y: TVM tensor
        the placeholder of add2_y

    Returns
    -------
    y0: TVM tensor
        the tensor of y0
    y1: TVM tensor
        the tensor of y1
    y2: TVM tensor
        the tensor of y2
    """
    square_0 = square_compute(input0, kernel_name="square")
    mul_3 = mul_compute(square_0, const_mul3_x, kernel_name="mul_3")
    mul_2 = mul_compute(input1, const_mul2_x, kernel_name="mul_2")

    y0 = add_compute(mul_2, mul_3, kernel_name="add_1")

    sqrt_0 = sqrt_compute(y0, kernel_name="sqrt")
    add_2 = add_compute(sqrt_0, add2_y, kernel_name="add_2")
    mul_0 = mul_compute(input2, const_mul_x, kernel_name="mul_0")
    mul_1 = mul_compute(input0, const_mul1_x, kernel_name="mul_1")

    y1 = add_compute(mul_0, mul_1, kernel_name="add_0")

    truediv_0 = true_div_compute(y1, add_2, kernel_name="truediv")
    mul_4 = mul_compute(input3, const_mul4_x, kernel_name="mul_4")
    add_3 = add_compute(truediv_0, mul_4, kernel_name="add_3")
    mul_5 = mul_compute(add_3, input4, kernel_name="mul_5")

    y2 = sub_compute(input3, mul_5, kernel_name="sub")

    return y0, y1, y2


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT,
                 REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT,
                 REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT,
                 REQUIRED_OUTPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def adam_apply_one_with_decay(input0,
                              input1,
                              input2,
                              input3,
                              input4,
                              const_mul_x,
                              const_mul1_x,
                              const_mul2_x,
                              const_mul3_x,
                              const_mul4_x,
                              add2_y,
                              output0,
                              output1,
                              output2,
                              kernel_name="adam_apply_one_with_decay"):
    """
    calculating data

    Parameters
    ----------
    input0: dict
        shape and dtype of input0
    input1: dict
        shape and dtype of input1
    input2: dict
        shape and dtype of input2
    input3: dict
        shape and dtype of input3
    input4: dict
        shape and dtype of input4
    const_mul_x: dict
        shape and dtype of const_mul_x
    const_mul1_x: dict
        shape and dtype of const_mul1_x
    const_mul2_x: dict
        shape and dtype of const_mul2_x
    const_mul3_x: dict
        shape and dtype of const_mul3_x
    const_mul4_x: dict
        shape and dtype of const_mul4_x
    add2_y: dict
        shape and dtype of add2_y
    output0: dict
        shape and dtype of output0
    output1: dict
        shape and dtype of output1
    output2: dict
        shape and dtype of output2
    kernel_name: str
        kernel name, default value is "adam_apply_one_with_decay"

    Returns
    -------
    None
    """
    dtype0 = input0.get("dtype")
    dtype1 = input1.get("dtype")
    dtype2 = input2.get("dtype")
    dtype3 = input3.get("dtype")
    dtype4 = input4.get("dtype")
    dtypecm0 = const_mul_x.get("dtype")
    dtypecm1 = const_mul1_x.get("dtype")
    dtypecm2 = const_mul2_x.get("dtype")
    dtypecm3 = const_mul3_x.get("dtype")
    dtypecm4 = const_mul4_x.get("dtype")
    dtypey = add2_y.get("dtype")

    shape0, shape1, shape2, shape3, shape4,\
    shapecm0, shapecm1, shapecm2, shapecm3, shapecm4, \
    shapey = _check_broadcast_shape(input0, input1, input2, input3, input4,
                                    const_mul_x, const_mul1_x, const_mul2_x,
                                    const_mul3_x, const_mul4_x, add2_y)


    input_place0 = tvm.placeholder(shape0, name="input0", dtype=dtype0)
    input_place1 = tvm.placeholder(shape1, name="input1", dtype=dtype1)
    input_place2 = tvm.placeholder(shape2, name="input2", dtype=dtype2)
    input_place3 = tvm.placeholder(shape3, name="input3", dtype=dtype3)
    input_place4 = tvm.placeholder(shape4, name="input4", dtype=dtype4)

    input_cm0 = tvm.placeholder(shapecm0, name="const_mul_x", dtype=dtypecm0)
    input_cm1 = tvm.placeholder(shapecm1, name="const_mul1_x", dtype=dtypecm1)
    input_cm2 = tvm.placeholder(shapecm2, name="const_mul2_x", dtype=dtypecm2)
    input_cm3 = tvm.placeholder(shapecm3, name="const_mul3_x", dtype=dtypecm3)
    input_cm4 = tvm.placeholder(shapecm4, name="const_mul4_x", dtype=dtypecm4)

    input_y = tvm.placeholder(shapey, name="add2_y", dtype=dtypey)

    y1, y2, y3 = adam_apply_one_with_decay_compute(
        input_place0, input_place1, input_place2, input_place3, input_place4,
        input_cm0, input_cm1, input_cm2, input_cm3, input_cm4, input_y)

    with tvm.target.cce():
        sch = generic.auto_schedule([y1, y2, y3])

    config = {
        "name":
            kernel_name,
        "tensor_list": (input_place0, input_place1, input_place2, input_place3,
                        input_place4, input_cm0, input_cm1, input_cm2,
                        input_cm3, input_cm4, input_y, y1, y2, y3)
    }

    te.lang.cce.cce_build_code(sch, config)
