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
ascend_dequant_s16
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
@fusion_manager.register("ascend_dequant_s16")
def ascend_dequant_s16_compute(x0, deq_scale, x1, y, relu_flag=False,
                               kernel_name='ascend_dequant_s16'):
    """
    int32 -> int16

    Parameters:
     ----------
    x : the placeholder of input

    deq: the placeholder of requant num

    x1:  the placeholder of add input tensor

    y: the dict of output

    relu_flag : the relu mode when true the result to do relu,
                default value is False

    kernel_name : cce kernel name, default value is "ascend_dequant_s16"

    Returns:

    res : the result of ascend_dequant_s16
    -------
    None
    """

    x0_shape = x0.shape
    x0_shape_list = te.lang.cce.util.shape_to_list(x0_shape)
    align_shape = x0_shape_list.copy()

    ori_shape_deq = deq_scale.op.attrs['ori_shape']
    ori_shape_deq_list = te.lang.cce.util.shape_to_list(ori_shape_deq)
    deq_dim = function_reduce(lambda x, y: x * y, ori_shape_deq_list[:])
    tensor_flag = False
    if deq_dim > 1:
        tensor_flag = True

    c1_index = 1
    if _is_nz_format(x0):
        c1_index = len(x0_shape) - 4

    align_shape[-2] = (align_shape[-2] + 15) // 16 * 16
    res_ub = _s32_to_s16_normal_compute(x0, deq_scale, x1, align_shape,
                                        c1_index, tensor_flag, relu_flag)

    if _is_nz_format(x0):
        res = tvm.compute(align_shape, lambda *i: res_ub[i],
                          name='res', tag='dequant_s16_NZ')
        return res

    res_shape = te.lang.cce.util.shape_to_list(res_ub.shape)
    res_shape[-2] = x0.shape[-2]
    res = tvm.compute(res_shape, lambda *indice: res_ub(*indice),
                      name='dequant_s16_remove_pad',
                      tag="dequant_s16_remove_pad")

    return res


def _is_nz_format(x0):
    """
    check is nz format
    """
    tensor_format = "NC1HWC0"
    if x0.op.attrs:
        if 'format' in x0.op.attrs:
            # NZ format,UB convergence scenario, input shape ..C1,N1,N0,C0
            tensor_format = x0.op.attrs['format']
    if tensor_format == "FRACTAL_NZ":
        return True

    return False


def _s32_to_s16_normal_compute(x0, deq_scale, x1, align_shape, c1_index,
                               tensor_flag, relu_flag):
    """
    generate s32_to_s16 compute
    """
    if tensor_flag:
        res_ub = tvm.compute(align_shape,
                             _deq_cast_compute(x0, deq_scale, x1,
                                               align_shape, c1_index,
                                               tensor_flag, relu_flag),
                             name='s32_to_s16', tag="dequant_s16_vector")
    else:
        res_ub = tvm.compute(align_shape,
                             _deq_cast_compute(x0, deq_scale, x1,
                                               align_shape, c1_index,
                                               tensor_flag, relu_flag),
                             name='s32_to_s16', tag="dequant_s16_scale")

    return res_ub


def _deq_cast_compute(x0, deq_scale, x1,
                      align_shape, c1_index, tensor_flag, relu_flag):
    """
    generate lambda func
    """
    n_dim = len(align_shape)
    c0_index = n_dim - 1

    def lambda_func(*indice):
        deq_indice = [0] * 5
        x1_indice = [0] * 5
        x1_indice[4] = indice[c0_index]
        x1_indice[1] = indice[c1_index]
        if tensor_flag:
            deq_indice[4] = indice[c0_index]
            deq_indice[1] = indice[c1_index]

        if x1 is not None:
            if tensor_flag:
                func = tvm.vdeq_cast(x0(*indice),
                                     deq_scale(*deq_indice),
                                     "int16",
                                     do_relu=relu_flag) + x1(*x1_indice)
            else:
                func = tvm.deq_cast(x0(*indice),
                                    deq_scale(*deq_indice),
                                    "int16") + x1(*x1_indice)
        else:
            if tensor_flag:
                func = tvm.vdeq_cast(x0(*indice),
                                     deq_scale(*deq_indice),
                                     "int16",
                                     do_relu=relu_flag)
            else:
                func = tvm.deq_cast(x0(*indice),
                                    deq_scale(*deq_indice),
                                    "int16")
        return func

    return lambda_func


@util.check_input_type((dict), (dict), (dict, NONETYPE), (dict), bool, str)
def ascend_dequant_s16(x0, deq_scale, x1, y, relu_flag=False,
                       kernel_name='ascend_dequant_s16'):
    """
    int32 -> int16

    Parameters:
    ----------
    x0 : the dict of input

    deq_scale: the dict of dequant num

    x1 : the input of add tensor

    y : the dict of output.

    relu_flag : the relu mode when true the result to do relu

    kernel_name : cce kernel name, default value is "ascend_dequant_s16"

    Returns:
    -------
    None
    """

    shape_x0 = x0.get("shape")
    format_x0 = x0.get("format")
    dtype_x0 = x0.get("dtype")

    shape_deq = deq_scale.get("shape")
    format_deq = deq_scale.get("format")
    dtype_deq = deq_scale.get("dtype")

    check_list = [("int32",), ("uint64",), ("int16",)]
    format_list = ["NC1HWC0", "FRACTAL_NZ"]
    util.check_dtype_rule(dtype_x0, check_list[0])
    util.check_dtype_rule(dtype_deq, check_list[1])

    if format_x0 not in format_list:
        raise RuntimeError("x0 only support [NC1HWC0, FRACTAL_NZ]")

    if format_x0 == "NC1HWC0":
        if len(shape_x0) != 5:
            raise ValueError(
                "x0 shape must of length 5 when format is NC1HWC0")

    if format_x0 == "FRACTAL_NZ":
        if len(shape_x0) < 4:
            raise RuntimeError(
                "x0 shape length must >= 4 when format is FRACTAL_NZ")

    if len(shape_deq) != 5:
        raise ValueError(
            "deq_scale shape must of length 5")

    if format_deq != "NC1HWC0":
        raise ValueError(
            "deq_scale only support NC1HWC0")

    if shape_deq[0] != 1 or shape_deq[2] != 1 or shape_deq[3] != 1:
        raise RuntimeError(
            "deq_scale shape must be 1 in n,h,w")

    if format_x0 == "NC1HWC0":
        # n, C1, H*W, C0
        shape_x0 = [shape_x0[0], shape_x0[1], shape_x0[2] * shape_x0[3],
                    shape_x0[4]]

    ori_shape_deq = deq_scale.get("ori_shape")
    attr = {"ori_shape": ori_shape_deq}
    input_x0 = tvm.placeholder(shape_x0, dtype_x0, "x0")
    input_deq = tvm.placeholder(shape_deq,
                                name="deq_scale",
                                dtype=dtype_deq,
                                attrs=attr)
    input_x1 = None
    if x1:
        shape_bias = x1.get("shape")
        input_x1 = tvm.placeholder(shape_bias, "int16", "x1")

    with tvm.target.cce():
        res = ascend_dequant_s16_compute(input_x0, input_deq, input_x1,
                                         relu_flag, kernel_name)
        generic.auto_schedule(res)
