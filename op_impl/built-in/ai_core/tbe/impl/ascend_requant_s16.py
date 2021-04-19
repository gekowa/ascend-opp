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
ascend_requant_s16
"""
from functools import reduce as function_reduce
from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

NONETYPE = type(None)


# pylint: disable=invalid-name,unused-argument,unnecessary-lambda
# pylint: disable=too-many-arguments,too-many-locals
@fusion_manager.register("ascend_requant_s16")
def ascend_requant_s16_compute(x, req_scale, x1, y, y1, dual_output, relu_flag,
                               kernel_name='ascend_requant_s16'):
    """
    int16 -> int8

    Parameters:
     ----------
    x : the placeholder of input

    req_scale: the placeholder of req_scale

    x1: the placeholder of x1

    y : the dict of output.

    y1 : the dict of output1.

    dual_output : the sqrt mode when true return 2 result,
                  default value is False

    relu_flag : the relu mode when true the result to do relu,
                default value is False

    kernel_name : cce kernel name, default value is "ascend_requant_s16"

    Returns:

    res : the result of ascend_requant_s16 which is list
    -------
    None
    """
    x_shape = x.shape
    x_shape_list = te.lang.cce.util.shape_to_list(x_shape)
    align_shape = x_shape_list.copy()

    ori_shape_req = req_scale.op.attrs['ori_shape']
    ori_shape_req_list = te.lang.cce.util.shape_to_list(ori_shape_req)
    req_dim = function_reduce(lambda x, y: x * y, ori_shape_req_list[:])
    tensor_flag = False
    if req_dim > 1:
        tensor_flag = True

    c1_index = 1
    if _is_nz_format(x):
        c1_index = len(x_shape) - 4

    align_shape[c1_index] = (align_shape[c1_index] + 1) // 2 * 2
    res_s16, res_ub = _s16_to_s8_normal_compute(x, x1, req_scale, x_shape,
                                                align_shape, c1_index,
                                                tensor_flag, relu_flag)

    res = _format_transfer(align_shape, res_ub, c1_index)
    if _is_nz_format(x):
        res = tvm.compute(align_shape, lambda *i: res[i],
                          name='res', tag='requant_s16_NZ')

    if dual_output:
        return [res, res_s16]

    return [res]


def _is_nz_format(x):
    """
    check is nz format
    """
    tensor_format = "NC1HWC0"
    if x.op.attrs:
        if 'format' in x.op.attrs:
            # NZ format,UB convergence scenario, input shape ..C1,N1,N0,C0
            tensor_format = x.op.attrs['format']
    if tensor_format == "FRACTAL_NZ":
        return True

    return False


def _s16_to_s8_normal_compute(x, x1, req_scale, x_shape, align_shape, c1_index,
                              tensor_flag, relu_flag):
    """
    generate s16_to_s8 compute
    """
    if x1 is not None:
        if relu_flag:
            res_s16 = tvm.compute(x_shape, lambda *indices: tvm.relu(
                x(*indices) + x1(*indices)), name="res_s16",
                                  tag="requant_s16_vaddrelu")
        else:
            res_s16 = tvm.compute(x_shape,
                                  lambda *indices: x(*indices) + x1(*indices),
                                  name="res_s16", tag="requant_s16_vadd")
    else:
        if relu_flag:
            res_s16 = tvm.compute(x_shape, lambda *indices: tvm.relu(
                x(*indices)), name="res_s16",
                                  tag="requant_s16_relu")
        else:
            res_s16 = tvm.compute(x_shape, lambda *indices: x(*indices),
                                  name="res_s16", tag="requant_s16")
    x_shape_list = te.lang.cce.util.shape_to_list(x_shape)
    if tensor_flag:
        res_ub = tvm.compute(align_shape,
                             _deq_cast_compute(res_s16, req_scale,
                                               align_shape, c1_index,
                                               tensor_flag, x_shape_list),
                             name='s16_to_s8', tag="requant_s16_vector")
    else:
        res_ub = tvm.compute(align_shape,
                             _deq_cast_compute(res_s16, req_scale,
                                               align_shape, c1_index,
                                               tensor_flag, x_shape_list),
                             name='s16_to_s8', tag="requant_s16_scale")
    return res_s16, res_ub


def _deq_cast_compute(res_s16, req_scale,
                      align_shape, c1_index, tensor_flag, x_shape_list):
    """
    generate lambda func
    """
    n_dim = len(align_shape)
    c0_index = n_dim - 1

    def lambda_func(*indice):
        new_indice = [0] * 5
        if tensor_flag:
            new_indice[4] = indice[c0_index]
            new_indice[1] = indice[c1_index]

        return tvm.select(indice[c1_index] < x_shape_list[c1_index],
                          tvm.conv_vdeq(res_s16(*indice),
                                        req_scale(*new_indice)).astype("int8"),
                          tvm.const(0, dtype="int8"))

    return lambda_func


def _format_compute(tensor, trans_shape, c1_index):
    """
    generate lambda func
    """
    n_dim = len(trans_shape)
    c0_index = n_dim - 1

    def lambda_func(*indice):
        new_indice = [0] * n_dim
        for i in range(n_dim):
            if i == c0_index:
                new_indice[i] = (indice[c1_index] * 32 +
                                 indice[c0_index]) % 16
            elif i == c1_index:
                new_indice[i] = (indice[c1_index] * 32 +
                                 indice[c0_index]) // 16
            else:
                new_indice[i] = indice[i]
        return tensor(*new_indice)

    return lambda_func


def _format_transfer(shape, x, c1_index):
    """
    C0 from 16 to 32 for NC1HWC0
    """
    trans_shape = shape[:]
    trans_shape[c1_index] = trans_shape[c1_index] // 2
    trans_shape[-1] = trans_shape[-1] * 2
    res = tvm.compute(trans_shape,
                      _format_compute(x, trans_shape, c1_index),
                      name='data_transfer',
                      tag="requant_s16_data_transfer")
    return res


@util.check_input_type((dict), (dict), (dict, NONETYPE), (dict),
                       (dict, NONETYPE), bool, bool, str)
def ascend_requant_s16(x, req_scale, x1, y, y1, dual_output=False,
                       relu_flag=False, kernel_name='ascend_requant_s16'):
    """
    int16 -> int8

    Parameters:
    ----------
    x : the dict of input

    req_scale: the dict of requant num

    x1: the dict of elewise num

    y : the dict of output.

    y1: the dict of output1

    dual_output : the sqrt mode when true return 2 result,
                  default value is False

    relu_flag : the relu mode when true the result to do relu,
                default value is False

    kernel_name : cce kernel name, default value is "ascend_requant_s16"

    Returns:
    -------
    None
    """

    shape_x = x.get("shape")
    format_x = x.get("format")
    dtype_x = x.get("dtype")

    shape_req = req_scale.get("shape")
    format_req = req_scale.get("format")
    dtype_req = req_scale.get("dtype")

    check_list = [("int16",), ("uint64",)]
    format_list = ["NC1HWC0", "FRACTAL_NZ"]
    util.check_dtype_rule(dtype_x, check_list[0])
    util.check_dtype_rule(dtype_req, check_list[1])

    if format_x not in format_list:
        raise RuntimeError("x only support [NC1HWC0, FRACTAL_NZ]")

    if format_x == "NC1HWC0":
        if len(shape_x) != 5:
            raise ValueError(
                "x shape must of length 5 when format is NC1HWC0")

    if format_x == "FRACTAL_NZ":
        if len(shape_x) < 4:
            raise RuntimeError(
                "x shape length must >= 4 when format is FRACTAL_NZ")

    if len(shape_req) != 5:
        raise ValueError(
            "req_scale shape must of length 5")

    if format_req != "NC1HWC0":
        raise ValueError(
            "req_scale only support NC1HWC0")

    if shape_req[0] != 1 or shape_req[2] != 1 or shape_req[3] != 1:
        raise RuntimeError(
            "req_scale shape must be 1 in n,h,w")

    if format_x == "NC1HWC0":
        # n, C1, H*W, C0
        shape_x = [shape_x[0], shape_x[1], shape_x[2] * shape_x[3], shape_x[4]]

    ori_shape_req = req_scale.get("ori_shape")
    attr = {"ori_shape": ori_shape_req}
    input_x = tvm.placeholder(shape_x, dtype_x, "x")
    input_req = tvm.placeholder(shape_req,
                                name="req_scale",
                                dtype=dtype_req,
                                attrs=attr)
    if x1:
        input_x1 = tvm.placeholder(shape_x, "int16", "x1")
    else:
        input_x1 = None

    with tvm.target.cce():
        res = ascend_requant_s16_compute(input_x, input_req, input_x1, y, y1,
                                         dual_output, relu_flag, kernel_name)
        generic.auto_schedule(res)
