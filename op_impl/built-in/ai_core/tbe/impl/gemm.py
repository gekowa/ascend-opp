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
gemm
"""
from __future__ import absolute_import

from math import ceil
# pylint: disable=import-error
import te.lang.cce
import te.platform.cce_params as cce
from te.utils.error_manager import error_manager_util as err_man
from te import tvm
from topi import generic
from topi.cce import util

from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json

ALPHA_BETA_SHAPE = [1]
NoneType = type(None)
MAX_INT32_LENGTH = 2147483647


def op_select_format(input_x1, input_x2, # pylint: disable=too-many-arguments
                     alpha, beta, bias=None, output_y=None, trans_a=False,
                     trans_b=False, kernel_name="gemm"):
    """
    select format dynamically
    """
    def _select_format(params):
        input_x1 = params[0]
        input_x2 = params[1]
        shape_b = input_x2.get("ori_shape")
        format_a = input_x1.get("format")
        format_b = input_x2.get("format")
        format_c = bias.get("format")
        need_transdata = False
        if set([format_a, format_b, format_c]) & \
                set(["FRACTAL_NZ", "FRACTAL_Z"]):
            need_transdata = True
        else:
            if trans_b:
                b_n = shape_b[0]
            else:
                b_n = shape_b[1]
            if b_n % cce.BLOCK_OUT != 0:
                need_transdata = True

        if need_transdata:
            input0 = gen_param(
                classify="input0",
                name="a",
                datatype="float16,float16,int8,int8",
                format="FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ",
            )
            input1 = gen_param(
                classify="input1",
                name="b",
                datatype="float16,float16,int8,int8",
                format="FRACTAL_NZ,FRACTAL_NZ,FRACTAL_Z,FRACTAL_Z",
            )
            input2 = gen_param(
                classify="input2",
                name="c",
                datatype="float32,float16,int32,float32",
                format="FRACTAL_NZ,FRACTAL_NZ,ND,FRACTAL_NZ",
            )
            output0 = gen_param(
                classify="output0",
                name="y",
                datatype="float32,float16,int32,float32",
                format="FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ",
            )
        else:
            input0 = gen_param(
                classify="input0",
                name="a",
                datatype="float16,float16,int8,int8",
                format="ND,ND,ND,ND",
            )
            input1 = gen_param(
                classify="input1",
                name="b",
                datatype="float16,float16,int8,int8",
                format="ND,ND,ND,ND",
            )
            input2 = gen_param(
                classify="input2",
                name="c",
                datatype="float32,float16,int32,float32",
                format="ND,ND,ND,ND",
            )
            output0 = gen_param(
                classify="output0",
                name="y",
                datatype="float32,float16,int32,float32",
                format="ND,ND,ND,ND",
            )
        input3 = gen_param(
            classify="input3",
            name="alpha",
            datatype="float32,float16,int32,float32",
            format="ND,ND,ND,ND",
        )
        input4 = gen_param(
            classify="input4",
            name="beta",
            datatype="float32,float16,int32,float32",
            format="ND,ND,ND,ND",
        )
        return [input0, input1, input2, input3, input4, output0]

    params = [input_x1, input_x2, alpha, beta, bias, output_y, trans_a,
              trans_b, kernel_name]
    param_list = _select_format(params)
    return get_dynamic_param_in_json(param_list)


# pylint: disable=locally-disabled,too-many-arguments,too-many-branches, too-many-statements, too-many-locals,
def _shape_check(
        shape_a, shape_b, shape_bias, src_dtype,
        trans_a, trans_b, alpha_dtype, beta_dtype, dst_dtype,
):
    """
    Check the given input if legal

    Parameters:
    shape_a: list or tuple
            Shape of the first tensor a with rank > 1
    shape_b:  list or tuple
            Shape of the second tensor b with the same type with a,
            and shape_a, shape_b must be 2 dims
    shape_bias: list or tuple
            Shape of bias, only support the input data format with ND
    src_dtype: str
            The data type of input, support "float32", "float16"
    trans_a: bool
            If True, shape_a == transposed before multiplication
    trans_b: bool
            If True, shape_b == transposed before multiplication

    Returns None
    """

    if alpha_dtype != beta_dtype:
        args_dict = {
            "errCode": "E60002",
            "attr_name": "dtype",
            "param1_name": "alpha",
            "param1_value": "{}".format(alpha_dtype),
            "param2_name": "beta",
            "param2_value": "{}".format(beta_dtype)
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
    if alpha_dtype != dst_dtype:
        args_dict = {
            "errCode": "E60002",
            "attr_name": "dtype",
            "param1_name": "alpha",
            "param1_value": "{}".format(alpha_dtype),
            "param2_name": "y",
            "param2_value": "{}".format(dst_dtype)
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))

    if src_dtype == "int8":
        if dst_dtype not in ["int32", "float32"]:
            args_dict = {
                "errCode": "E60003",
                "a_dtype": src_dtype,
                "expected_dtype_list": "int32,float32",
                "out_dtype": "{}".format(dst_dtype)
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
    elif src_dtype == "float16":
        if dst_dtype not in ["float16", "float32"]:
            args_dict = {
                "errCode": "E60003",
                "a_dtype": src_dtype,
                "expected_dtype_list": "float16,float32",
                "out_dtype": "{}".format(dst_dtype)
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))

    src_dtype = src_dtype.lower()

    check_list = ("float16", "int8")

    if src_dtype not in check_list:
        args_dict = {
            "errCode": "E60005",
            "param_name": "a",
            "expected_dtype_list": "{}".format(check_list),
            "dtype": "{}".format(src_dtype)
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))

    if len(shape_a) != 2 and len(shape_a) != 4:
        args_dict = {
            "errCode": "E60006",
            "param_name": "a",
            "expected_length": "2 or 4",
            "length": "{}".format(len(shape_a))
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))

    if len(shape_b) != 2 and len(shape_b) != 4:
        args_dict = {
            "errCode": "E60006",
            "param_name": "b",
            "expected_length": "2 or 4",
            "length": "{}".format(len(shape_b))
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))

    if len(shape_a) == 2 and len(shape_b) == 2:
        if trans_a:
            km_shape = shape_a[0]
        else:
            km_shape = shape_a[1]

        if trans_b:
            kn_shape = shape_b[1]
        else:
            kn_shape = shape_b[0]

        if km_shape != kn_shape:
            args_dict = {
                "errCode": "E60009",
                "a_1d": km_shape,
                "b_0d": kn_shape
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))


def _get_bias_element(shape_bias_element):
    bias_length = shape_bias_element
    if bias_length % 16 == 0:
        return bias_length
    bias_length = (bias_length // 16) * 16 + 16
    return bias_length


def _get_bias(shape_bias):
    for index, value in enumerate(shape_bias):
        shape_bias[index] = _get_bias_element(value)
    return shape_bias


def _get_input_shape_a(shape_x, dtype):
    dim_a = shape_x[0]
    dim_b = shape_x[1]
    res = list()
    block_in = cce.BLOCK_IN

    if dtype == "float16":
        block_reduce = cce.BLOCK_REDUCE
    else:
        block_reduce = cce.BLOCK_REDUCE_INT8

    res.append(ceil(dim_a/block_in)*block_in)
    res.append(ceil(dim_b/block_reduce)*block_reduce)
    return res


def _get_input_shape_b(shape_x, dtype):
    dim_a = shape_x[0]
    dim_b = shape_x[1]
    res = list()
    block_out = cce.BLOCK_OUT

    if dtype == "float16":
        block_reduce = cce.BLOCK_REDUCE
    else:
        block_reduce = cce.BLOCK_REDUCE_INT8

    res.append(ceil(dim_a/block_reduce)*block_reduce)
    res.append(ceil(dim_b/block_out)*block_out)
    return res


def _bias_check(input_x1, input_x2, bias, trans_a, trans_b):
    if input_x1["ori_format"] == "ND" and input_x2["ori_format"] == \
            "ND" and bias["ori_format"] == "ND":
        shape_a = list(input_x1["ori_shape"])
        shape_b = list(input_x2["ori_shape"])
        shape_bias = list(bias["ori_shape"])

        if trans_a:
            a_m = shape_a[1]
        else:
            a_m = shape_a[0]

        if trans_b:
            b_n = shape_b[0]
        else:
            b_n = shape_b[1]
        if shape_bias != [a_m, b_n]:
            args_dict = {
                "errCode": "E60000",
                "param_name": "c shape",
                "expected_value": str([a_m, b_n]),
                "input_value": "{}".format(shape_bias)
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
    else:
        shape_a = list(input_x1["shape"])
        shape_b = list(input_x2["shape"])
        shape_bias = list(bias["shape"])
        if len(shape_bias) == 2:
            shape_bias = [ceil(shape_bias[1]/cce.BLOCK_OUT), ceil(shape_bias[0]/cce.BLOCK_IN)]
        else:
            shape_bias = shape_bias[:2]
        if input_x2["dtype"] == "int8" and shape_bias != [shape_b[1], shape_a[1]]:
            args_dict = {
                "errCode": "E60000",
                "param_name": "c shape",
                "expected_value": str([shape_a[1], shape_b[1]]),
                "input_value": "{}".format(shape_bias)
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
        if input_x2["dtype"] == "float16" and shape_bias != [shape_b[0], shape_a[1]]:
            args_dict = {
                "errCode": "E60000",
                "param_name": "c shape",
                "expected_value": str([shape_a[1], shape_b[0]]),
                "input_value": "{}".format(shape_bias)
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))


# pylint: disable=locally-disabled,too-many-arguments, too-many-locals, too-many-statements
@util.check_input_type(dict, dict, dict, dict, dict, dict, bool, bool, str)
def gemm(input_x1, input_x2, bias, alpha, beta, output_y=None, trans_a=False,
         trans_b=False, kernel_name="gemm"):
    """
    calculating  matrix multiplication with bias, C = alpha*A*B + beta*bias, support input
    data with Nz format.

    Parameters:
    input_x1: dict
            shape and dtype of tensor_a
    input_x2: dict
            shape and dtype of tensor_b
    alpha: shape and dtype of alpha
    beta: shape and dtype of beta
    bias: dict
            Shape of bias, support the input data format with Nz/ND in different scenes
    trans_a:
            whether transpose a
            only support false
    trans_b:
            whether transpose b
            only support false
    Returns
    -------
    None
    """
    if output_y is None:
        output_y = {}

    # 当ab不都为ND格式时，由外层处理transpose
    if input_x1.get("format") != "ND" or input_x2.get("format") != "ND":
        trans_a = False
        trans_b = False

    shape_a = input_x1.get("ori_shape")
    shape_b = input_x2.get("ori_shape")
    src_dtype = input_x1.get("dtype").lower()
    b_dtype = input_x2.get("dtype").lower()
    dst_dtype = output_y.get("dtype").lower()

    if shape_a is not None:
        if len(shape_a) < 2:
            shape_a = input_x1.get("shape")

    if shape_b is not None:
        if len(shape_b) < 2:
            shape_b = input_x2.get("shape")

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_a)
    util.check_shape_rule(shape_b)

    alpha_dtype = alpha.get("dtype")
    beta_dtype = beta.get("dtype")

    shape_bias = bias.get("ori_shape")
    if bias.get("format") == "ND" and bias.get("ori_format") != "ND":
        shape_bias = bias.get("shape")
    shape_bias = list(shape_bias)

    if src_dtype != b_dtype:
        args_dict = {
            "errCode": "E60002",
            "attr_name": "dtype",
            "param1_name": "a",
            "param1_value": "{}".format(src_dtype),
            "param2_name": "b",
            "param2_value": "{}".format(b_dtype)
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))

    _bias_check(input_x1, input_x2, bias, trans_a, trans_b)

    _shape_check(
        shape_a, shape_b, shape_bias, src_dtype,
        trans_a, trans_b, alpha_dtype, beta_dtype, dst_dtype,
    )

    if bias.get("format") != "ND" and len(shape_bias) == 2:
        shape_bias = _get_bias(shape_bias)

    if len(shape_a) == 2:
        if input_x1.get("format") != "ND":
            shape_a = _get_input_shape_a(list(shape_a), src_dtype)
        if input_x1.get("format") == "FRACTAL_NZ":
            shape_a = [shape_a[1], shape_a[0]]
            trans_a = bool(1 - trans_a)
    elif len(shape_a) == 4:
        trans_a = bool(1 - trans_a)

    if len(shape_b) == 2:
        if input_x2.get("format") != "ND":
            shape_b = _get_input_shape_b(list(shape_b), src_dtype)
        if input_x2.get("format") == "FRACTAL_NZ":
            shape_b = [shape_b[1], shape_b[0]]
            trans_b = bool(1 - trans_b)
    elif len(shape_b) == 4:
        if input_x2.get("format") == "FRACTAL_NZ":
            trans_b = bool(1 - trans_b)

    if bias is None or not bool(bias):
        args_dict = {
            "errCode": "E60108",
            "reason": "unsupport c is None"
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))

    if len(shape_a) == 2:
        m_shape = shape_a[0]
        km_shape = shape_a[1]
    if len(shape_b) == 2:
        kn_shape = shape_b[0]
        n_shape = shape_b[1]

    if src_dtype == "float16":
        block_reduce = cce.BLOCK_REDUCE
    else:
        block_reduce = cce.BLOCK_REDUCE_INT8

    block_in = cce.BLOCK_IN
    block_out = cce.BLOCK_OUT

    if len(shape_a) == 2:
        if trans_a:
            shape_a_temp = (
                m_shape // block_reduce, km_shape // block_in, block_in,
                block_reduce
            )
        else:
            shape_a_temp = (
                m_shape // block_in, km_shape // block_reduce, block_in,
                block_reduce
            )
        if input_x1.get("format") == "FRACTAL_NZ":
            format_a = "FRACTAL_NZ"
        else:
            shape_a_temp = shape_a
            format_a = "ND"
    elif len(shape_a) == 4:
        if input_x1.get("format") == "FRACTAL_NZ":
            shape_a_temp = shape_a
            format_a = "FRACTAL_NZ"

    if len(shape_b) == 2:
        if trans_b:
            shape_b_temp = (
                kn_shape // block_out, n_shape // block_reduce, block_reduce,
                block_out
            )
        else:
            shape_b_temp = (
                kn_shape // block_reduce, n_shape // block_out, block_out,
                block_reduce
            )
        if input_x2.get("format") == "FRACTAL_Z":
            format_b = "fractal"
        elif input_x2.get("format") == "FRACTAL_NZ":
            format_b = "FRACTAL_NZ"
        else:
            shape_b_temp = shape_b
            format_b = "ND"
    elif len(shape_b) == 4:
        if input_x2.get("format") == "FRACTAL_Z":
            shape_b_temp = shape_b
            format_b = "fractal"
        elif input_x2.get("format") == "FRACTAL_NZ":
            shape_b_temp = shape_b
            format_b = "FRACTAL_NZ"

    # 获取Nz格式的bias shape
    if bias.get("format") != "ND" and len(shape_bias) != 4:
        shape_bias_temp = (
            shape_bias[1] // block_out, shape_bias[0] // block_in, block_in,
            block_out,
            )
    else:
        shape_bias_temp = shape_bias

    def _gemm_local_compute():
        tensor_a = tvm.placeholder(shape_a_temp, name='tensor_a',
                                   dtype=src_dtype)
        tensor_b = tvm.placeholder(shape_b_temp, name='tensor_b',
                                   dtype=src_dtype)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name='tensor_alpha',
                                       dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name='tensor_beta',
                                      dtype=alpha_dtype)
        tensor_bias = tvm.placeholder(shape_bias_temp, name='tensor_bias',
                                      dtype=dst_dtype)
        result = te.lang.cce.gemm(
            tensor_a, tensor_b, tensor_alpha, tensor_beta, trans_a, trans_b,
            format_a=format_a, format_b=format_b, dst_dtype=dst_dtype,
            tensor_bias=tensor_bias,
            kernel_name=kernel_name
        )

        with tvm.target.cce():
            schedule = generic.auto_schedule(result)

        tensor_list = [tensor_a, tensor_b, tensor_bias,
                       tensor_alpha, tensor_beta, result]
        config = {"print_ir": False,
                  "name": kernel_name,
                  "tensor_list": tensor_list,
                  }
        te.lang.cce.cce_build_code(schedule, config)

    def _is_larger_than_int32(input_tensor):
        m_bit_ratio = {"float16": 2, "float32": 4, "int8": 1, "int32": 4}
        res = 1
        input_shape = input_tensor.get("ori_shape")
        for axls in input_shape:
            res *= axls
        res *= m_bit_ratio.get(input_tensor.get("dtype").lower())
        if res > MAX_INT32_LENGTH:
            return True
        return False

    _gemm_local_compute()


