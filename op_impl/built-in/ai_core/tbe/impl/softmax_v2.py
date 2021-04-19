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
softmax_v2
"""
# pylint: ungrouped-imports
# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=invalid-name,unused-variable,too-many-locals
# pylint: disable=too-many-statements,unnecessary-lambda
# pylint: disable=unidiomatic-typecheck,ungrouped-imports
# pylint: disable=too-many-lines,too-many-branches
from __future__ import absolute_import

import math
import te.lang.cce
import topi
import te.platform.cce_params as cce
from te import tvm
from te import platform as tbe_platform
from te.platform.cce_build import build_config
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import *
from topi import generic
from topi.cce import util
from impl.util import util_frac_z as fz
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json

# General limitation of the size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648
# 1/4 UB size

UB_SIZE_LIMIT = \
    tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
UB_SIZE_LIMIT = UB_SIZE_LIMIT / 4


def select_nd_to_5d(dtype, shape_x_ori, axis):
    length_x_ori = len(shape_x_ori)
    if not isinstance(axis, int):
        axis = list(axis)
    else:
        axis = [axis]
    nd_to_5d = 0
    if ((dtype == "float16" and shape_x_ori[-1] % 16 != 0)
        or (dtype == "float32" and shape_x_ori[-1] % 8 !=
            0)) and (length_x_ori == 3 or length_x_ori == 4):
        if (axis[0] == -1 and len(axis) == 1):
            nd_to_5d = 1
        else:
            nd_to_5d = 0
    else:
        nd_to_5d = 0

    return nd_to_5d


def check_axis_is_last(shape_x_ori, axis):
    length_x_ori = len(shape_x_ori)
    if not isinstance(axis, int):
        axis = list(axis)
    else:
        axis = [axis]
    axis_is_last = 0
    if (length_x_ori == 2):
        if (axis[0] == -1 or axis[0] == 1):
            axis_is_last = 1
        else:
            axis_is_last = 0
    else:
        axis_is_last = 0
    return axis_is_last


def op_select_format(input_x, output_y, axis=-1, kernel_name="softmax_v2"):
    """
    select format dynamically
    """
    shape_x_ori = util.scalar2tensor_one(input_x.get("ori_shape"))
    length_x_ori = len(shape_x_ori)
    tbe_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    dtype = input_x.get("dtype").lower()
    if length_x_ori == 2:
        if check_axis_is_last(shape_x_ori, axis):
            if tbe_product in ("Hi3796CV300ES", "Hi3796CV300CS"):
                input0 = gen_param(classify="input0", name="x",
                                datatype="float16,float16,float16",
                                format="FRACTAL_NZ,NC1HWC0,ND")
                output0 = gen_param(classify="output0", name="y",
                                    datatype="float16,float16,float16",
                                    format="FRACTAL_NZ,NC1HWC0,ND")
            if tbe_product in ("Ascend610", "Ascend710",):
                input0 = gen_param(classify="input0", name="x",
                                datatype="float16,float16,float16,float",
                                format="FRACTAL_NZ,NC1HWC0,ND,ND")
                output0 = gen_param(classify="output0", name="y",
                                    datatype="float16,float16,float16,float",
                                    format="FRACTAL_NZ,NC1HWC0,ND,ND")
            if tbe_product in ("Ascend910", "Ascend310",):
                input0 = gen_param(classify="input0", name="x",
                                datatype="float16,float16,float16,float,"
                                            "float",
                                format="FRACTAL_NZ,NC1HWC0,ND,ND,"
                                        "NC1HWC0")
                output0 = gen_param(classify="output0", name="y",
                                    datatype="float16,float16,float16,float,"
                                            "float",
                                    format="FRACTAL_NZ,NC1HWC0,ND,ND,"
                                        "NC1HWC0")
        else:
            if tbe_product in ("Hi3796CV300ES", "Hi3796CV300CS"):
                input0 = gen_param(classify="input0", name="x",
                                datatype="float16,float16",
                                format="NC1HWC0,ND")
                output0 = gen_param(classify="output0", name="y",
                                    datatype="float16,float16",
                                    format="NC1HWC0,ND")
            if tbe_product in ("Ascend610", "Ascend710",):
                input0 = gen_param(classify="input0", name="x",
                                datatype="float16,float16,float",
                                format="NC1HWC0,ND,ND")
                output0 = gen_param(classify="output0", name="y",
                                    datatype="float16,float16,float",
                                    format="NC1HWC0,ND,ND")
            if tbe_product in ("Ascend910", "Ascend310",):
                input0 = gen_param(classify="input0", name="x",
                                datatype="float16,float16,float,"
                                            "float",
                                format="NC1HWC0,ND,ND,"
                                        "NC1HWC0")
                output0 = gen_param(classify="output0", name="y",
                                    datatype="float16,float16,float,"
                                            "float",
                                    format="NC1HWC0,ND,ND,"
                                        "NC1HWC0")
    elif length_x_ori > 2 and (shape_x_ori[-1] % 16 != 0 or \
                               shape_x_ori[-2] % 16 != 0):
        if tbe_product in ("Hi3796CV300ES", "Hi3796CV300CS"):
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float16",
                               format="NC1HWC0,ND")
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,float16",
                                format="NC1HWC0,ND")
        if tbe_product in ("Ascend610", "Ascend710",):
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float16,float",
                               format="NC1HWC0,ND,ND")
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,float16,float",
                                format="NC1HWC0,ND,ND")
        if tbe_product in ("Ascend910",):
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float16,float,"
                                        "float",
                               format="NC1HWC0,ND,ND,"
                                      "NC1HWC0")
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,float16,float,"
                                         "float",
                                format="NC1HWC0,ND,ND,"
                                       "NC1HWC0")
        if tbe_product in ("Ascend310",):
            if select_nd_to_5d(dtype, shape_x_ori, axis):
                input0 = gen_param(classify="input0", name="x",
                                   datatype="float16,"
                                            "float",
                                   format="NC1HWC0,"
                                          "NC1HWC0")
                output0 = gen_param(classify="output0", name="y",
                                    datatype="float16,"
                                             "float",
                                    format="NC1HWC0,"
                                           "NC1HWC0")
            else:
                input0 = gen_param(classify="input0", name="x",
                                   datatype="float16,float16,float,"
                                            "float",
                                   format="NC1HWC0,ND,ND,"
                                          "NC1HWC0")
                output0 = gen_param(classify="output0", name="y",
                                    datatype="float16,float16,float,"
                                             "float",
                                    format="NC1HWC0,ND,ND,"
                                           "NC1HWC0")
    else:
        if tbe_product in ("Hi3796CV300ES", "Hi3796CV300CS"):
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float16,float16",
                               format="FRACTAL_NZ,NC1HWC0,ND")
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,float16,float16",
                                format="FRACTAL_NZ,NC1HWC0,ND")
        if tbe_product in ("Ascend610", "Ascend710",):
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float,float16,float16,float",
                               format="FRACTAL_NZ,FRACTAL_NZ,NC1HWC0,ND,ND")
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,float,float16,float16,float",
                                format="FRACTAL_NZ,FRACTAL_NZ,NC1HWC0,ND,ND")
        if tbe_product in ("Ascend910", "Ascend310",):
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float,float16,float16,float,"
                                        "float",
                               format="FRACTAL_NZ,FRACTAL_NZ,NC1HWC0,ND,ND,"
                                      "NC1HWC0")
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,float,float16,float16,float,"
                                         "float",
                                format="FRACTAL_NZ,FRACTAL_NZ,NC1HWC0,ND,ND,"
                                       "NC1HWC0")
    param_list = [input0, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def _broadcast_nz(tensor, shape):
    broadcast_axes = []
    src_shape = te.lang.cce.util.shape_to_list(tensor.shape)
    for i, _ in enumerate(shape):
        if shape[i] != src_shape[i]:
            broadcast_axes.append(i)
    if len(broadcast_axes) == 2 and \
            broadcast_axes[1] - broadcast_axes[0] != 1 and \
            broadcast_axes[1] + 1 == len(shape):
        temp_shape = src_shape[:-1] + [shape[-1]]
        tensor = te.lang.cce.broadcast(tensor, temp_shape)
    tensor = te.lang.cce.broadcast(tensor, shape)
    return tensor


@fusion_manager.register("softmax_v2")
def softmax_v2_compute(input_x, output_y, axis=-1, kernel_name="softmax_v2"):
    """
    calculating data's softmax, produces an output tensor with shape
    the result equals the sum of the x power of e over the sum of
    the x power of e

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    axis : int or list or tuple
       the data's axis, range == [-d, d-1]
    kernel_name: str
        cce kernel name, default value is softmax_v2

    Returns
    -------
    output: TVM tensor
        the result of softmax
    """

    shape = te.lang.cce.util.shape_to_list(input_x.shape)
    dtype = input_x.dtype
    axis = list(axis)
    last_dim = len(input_x.shape) - 1
    vcmax_flag = False

    for i in axis:
        if (i == -1) or (i == last_dim):
            vcmax_flag = True

    if dtype == "float32" and vcmax_flag and \
            not tbe_platform.cce_conf.api_check_support(
                "te.lang.cce.reduce_max", "float32"):
        data_max_input = te.lang.cce.cast_to(input_x, "float16")
        data_max_output = te.lang.cce.reduce_max(data_max_input,
                                                 axis=axis, keepdims=True)
        data_max = te.lang.cce.cast_to(data_max_output, "float32")
    else:
        data_max = te.lang.cce.reduce_max(input_x, axis=axis, keepdims=True)

    data_max = _broadcast_nz(data_max, shape)
    data_subtrac = te.lang.cce.vsub(input_x, data_max)

    has_improve_precision = False
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.vexp", "float32"):
        data_subtrac = te.lang.cce.cast_to(data_subtrac, "float32")
        has_improve_precision = True
    data_exp = te.lang.cce.vexp(data_subtrac)

    tbe_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if data_exp.dtype == "float16" and tbe_product in ("Ascend310",):
        data_exp = te.lang.cce.cast_to(data_exp, "float32")
        has_improve_precision = True

    data_expsum = te.lang.cce.sum(data_exp, axis, keepdims=True)
    data_expsum = _broadcast_nz(data_expsum, shape)
    output = te.lang.cce.vdiv(data_exp, data_expsum)
    if has_improve_precision and dtype == "float16":
        output = te.lang.cce.cast_to(output, "float16")

    return output


def buffer_mapping(schedule, ops):
    """
    set buffer scope
    """
    if not ops:
        raise RuntimeError("operation list is empty")
    for i_op in ops:
        schedule[i_op].set_scope(cce.scope_ubuf)


def align(schedule, ops, pad_param, factor=16, offset=0):
    """
    determine if aligning needs to be enabled
    """
    if not ops:
        raise RuntimeError("operation list is empty")
    length = len(ops)
    if length <= 3:
        # no op need aligning
        return
    for i in range(0, length-1):
        shape_len = len(ops[i].shape)
        if shape_len > 1:
            if ops[i].shape[1].value == 1 and pad_param[1] == 15:
                if ops[i].shape[4].value == 1:
                    schedule[ops[i]].storage_align(ops[i].op.axis[shape_len - 2],
                                                   factor, offset)
                if ops[i].op.name == "res_vonv_fp32_max":
                    schedule[ops[i]].storage_align(ops[i].op.axis[shape_len - 2],
                                                   factor, offset)
            else:
                if ops[i].shape[4].value == 1:
                    schedule[ops[i]].storage_align(ops[i].op.axis[shape_len - 3],
                                                   factor, offset)
                if ops[i].op.name == "res_vonv_fp32_max":
                    schedule[ops[i]].storage_align(ops[i].op.axis[shape_len - 3],
                                                   factor, offset)


def align_nz(schedule, ops, pad_param, factor=16, offset=0):
    """
    determine if aligning needs to be enabled
    """
    if not ops:
        raise RuntimeError("operation list is empty")
    length = len(ops)
    if length <= 3:
        # no op need aligning
        return
    for i in range(0, length-1):
        shape_len = len(ops[i].shape)
        if shape_len > 1:
            if ops[i].shape[0].value == 1 and pad_param[1] == 15:
                if ops[i].shape[3].value == 1:
                    schedule[ops[i]].storage_align(ops[i].op.axis[shape_len - 1],
                                                   factor, offset)
                if ops[i].op.name == "res_vonv_fp32_max":
                    schedule[ops[i]].storage_align(ops[i].op.axis[shape_len - 1],
                                                   factor, offset)
            else:
                if ops[i].shape[3].value == 1:
                    schedule[ops[i]].storage_align(ops[i].op.axis[shape_len - 2],
                                                   factor, offset)
                if ops[i].op.name == "res_vonv_fp32_max":
                    schedule[ops[i]].storage_align(ops[i].op.axis[shape_len - 2],
                                                   factor, offset)


def multicore_factor_calculate(shape):
    """
    the compute produce, calculate multicore information
    """
    if not shape:
        raise RuntimeError("input shape is empty")

    device_core_num = \
        tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)

    split_axis = 0
    split_size = 0
    if shape[0] >= device_core_num:
        npart_n = device_core_num
        npart_h = 1
        npart_w = 1
        split_axis = 1
        split_size = math.ceil(shape[0] / device_core_num)
    elif device_core_num // shape[0] <= shape[2]:
        npart_n = shape[0]
        npart_h = device_core_num // shape[0]
        npart_w = 1
        split_axis = 2
        split_size = math.ceil(shape[2] / (device_core_num // shape[0]))
    elif device_core_num // shape[0] // shape[2] <= shape[3]:
        npart_n = shape[0]
        npart_h = shape[2]
        npart_w = (device_core_num // shape[0] // shape[2])
        split_axis = 3
        split_size = math.ceil(shape[3] / (device_core_num // shape[0] // shape[2]))
    else:
        npart_n = shape[0]
        npart_h = shape[2]
        npart_w = shape[3]
        split_axis = 4
        split_size = 1
    return npart_n, npart_h, npart_w, split_axis, split_size


def multicore_factor_calculate_nz(shape):
    """
    the compute produce, calculate multicore information
    """
    if not shape:
        raise RuntimeError("input shape is empty")

    device_core_num = \
        tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)

    split_axis = 0
    split_size = 0
    if shape[1] >= device_core_num:
        npart_n1 = device_core_num
        npart_n0 = 1
        split_axis = 1
        split_size = math.ceil(shape[1] / device_core_num)
    elif device_core_num // shape[1] <= shape[2]:
        npart_n1 = shape[1]
        npart_n0 = device_core_num // shape[1]
        split_axis = 2
        split_size = math.ceil(shape[2] / (device_core_num // shape[1]))
    else:
        npart_n1 = shape[1]
        npart_n0 = shape[2]
        split_axis = 3
        split_size = 1
    return npart_n1, npart_n0, split_axis, split_size


def tiling_factor_calculate(shape, split_axis_0, split_size, use_fp32):
    """
    do tiling calculate
    """
    if not shape:
        raise RuntimeError("input shape is empty")

    if use_fp32:
        temp = UB_SIZE_LIMIT // (shape[1] * shape[4] * 4)
    else:
        temp = UB_SIZE_LIMIT // (shape[1] * shape[4] * 2)

    split_flag = False
    split_axis = 0
    split_factor = 0
    if split_axis_0 == 1:
        if temp >= split_size * shape[2] * shape[3]:
            # no split
            split_flag = False
        elif temp < split_size * shape[2]*shape[3] and temp >= shape[2]*shape[3]:
            # split on n.inner
            split_flag = True
            split_axis = 0
            split_factor = int(temp // (shape[2] * shape[3]))
        elif temp < shape[2] * shape[3] and temp >= shape[3]:
            # split on h
            split_flag = True
            split_axis = 2
            split_factor = int(temp // shape[3])
        elif temp < shape[3]:
            # split on w
            split_flag = True
            split_axis = 3
            split_factor = int(temp)
    if split_axis_0 == 2:
        if temp >= split_size * shape[3]:
            # no split
            split_flag = False
        elif temp < shape[2] * shape[3] and temp >= shape[3]:
            # split on h
            split_flag = True
            split_axis = 2
            split_factor = int(temp // shape[3])
        elif temp < shape[3]:
            # split on w
            split_flag = True
            split_axis = 3
            split_factor = int(temp)
    if split_axis_0 == 3:
        if temp >= split_size:
            # no split
            split_flag = False
        else:
            # split on w
            split_flag = True
            split_axis = 3
            split_factor = int(temp)
    if split_axis_0 == 4:
        # no split
        split_flag = False

    return split_flag, split_axis, split_factor


def tiling_factor_calculate_nz(shape, split_axis_0, split_size, use_fp32):
    """
    do tiling calculate
    """
    if not shape:
        raise RuntimeError("input shape is empty")

    if use_fp32:
        temp = UB_SIZE_LIMIT // (shape[0] * shape[3] * 4)
    else:
        temp = UB_SIZE_LIMIT // (shape[0] * shape[3] * 2)

    split_flag = False
    split_axis = 0
    split_factor = 0
    if split_axis_0 == 1:
        if temp >= split_size * shape[2]:
            # no split
            split_flag = False
        elif temp < split_size * shape[2] and temp >= shape[2]:
            # split on n.inner
            split_flag = True
            split_axis = 1
            split_factor = int(temp // shape[2])
        elif temp < shape[3]:
            # split on w
            split_flag = True
            split_axis = 2
            split_factor = int(temp)
    if split_axis_0 == 2:
        if temp >= split_size:
            # no split
            split_flag = False
        else:
            # split on w
            split_flag = True
            split_axis = 2
            split_factor = int(temp)
    if split_axis_0 == 3:
        # no split
        split_flag = False

    return split_flag, split_axis, split_factor


def ops_integrate(schedule, ops, axis):
    """
    determine if integrating needs to be enabled
    """
    if not ops:
        raise RuntimeError("operation list is empty")
    length = len(ops)
    if length < 2:
        # no op need integrating
        return
    integration_op = schedule[ops[-1]]
    for i in range(0, length-1):
        schedule[ops[i]].compute_at(integration_op, axis)


def double_buf(schedule, ops):
    """
    determine if double buffer needs to be enabled
    """
    if not ops:
        raise RuntimeError("operation list is empty")
    length = len(ops)
    if length < 2:
        # no op need double buffer
        return
    for i in range(0, length-1):
        schedule[ops[i]].double_buffer()


def emit_axis_collect(ops, pad_param, instructions, last_axis):
    """
    the compute produce, emit axis information
    """
    if not ops:
        raise RuntimeError("operation list is empty")
    if len(ops) != len(instructions):
        raise RuntimeError("length of operations and instructions "
                           "does not match")
    axis_list = []
    length = len(ops)
    for i in range(0, length-1):
        if pad_param[1] == 15:
            axis_list += [ops[i].op.axis[1]]
        else:
            if instructions[i] == 'vector_adds' and \
                    ops[i].op.name == 'res_sub':
                axis_list += [ops[i].op.axis[1]]
            elif instructions[i] == 'vector_muls' and \
                    ops[i].op.name == "res_mul":
                axis_list += [ops[i].op.axis[1]]
            else:
                axis_list += [ops[i].op.axis[0]]
    axis_list += [last_axis]
    return axis_list


def emit_nz_axis_collect(ops, pad_param, instructions, last_axis):
    """
    the compute produce, emit axis information
    """
    if not ops:
        raise RuntimeError("operation list is empty")
    if len(ops) != len(instructions):
        raise RuntimeError("length of operations and instructions "
                           "does not match")
    axis_list = []
    length = len(ops)
    for i in range(0, length-1):
        axis_list += [ops[i].op.axis[0]]
    axis_list += [last_axis]
    return axis_list


def axis_reorder(schedule, ops, instructions):
    """
    the compute produce, reorder axis
    """
    if not ops:
        raise RuntimeError("operation list is empty")
    if len(ops) != len(instructions):
        raise RuntimeError("length of operations and instructions "
                           "does not match")
    length = len(ops)
    for i in range(0, length):
        if instructions[i] == 'vector_adds' or instructions[i] == 'vector_muls':
            schedule[ops[i]].reorder(ops[i].op.axis[0], ops[i].op.axis[2],
                                     ops[i].op.axis[3], ops[i].op.axis[1],
                                     ops[i].op.axis[4])


def axis_reorder_nz(schedule, ops, instructions):
    """
    the compute produce, reorder axis
    """
    if not ops:
        raise RuntimeError("operation list is empty")
    if len(ops) != len(instructions):
        raise RuntimeError("length of operations and instructions "
                           "does not match")
    length = len(ops)
    for i in range(0, length):
        if instructions[i] == 'vector_adds' or instructions[i] == 'vector_muls':
            schedule[ops[i]].reorder(ops[i].op.axis[1], ops[i].op.axis[2],
                                     ops[i].op.axis[0], ops[i].op.axis[3])


def instructions_replace(schedule, ops, axes, instructions):
    """
    the compute produce, replace instructions
    """
    if not ops:
        raise RuntimeError("operation list is empty")
    if len(ops) != len(instructions):
        raise RuntimeError("length of operations and instructions "
                           "does not match")
    length = len(ops)
    for i in range(0, length):
        schedule[ops[i]].emit_insn(axes[i], instructions[i])


def check_isusefp32(shape, dtype):
    """
    check compute wheather ues fp32
    """
    use_fp32 = True
    if dtype == "float32":
        use_fp32 = True
        return use_fp32
    tbe_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if tbe_product in ("Hi3796CV300ES", "Hi3796CV300CS"):
        use_fp32 = False
        return use_fp32
    if shape[1] * shape[4] * 4 > UB_SIZE_LIMIT:
        use_fp32 = False
        return use_fp32
    return use_fp32


def check_nz_isusefp32(shape, dtype):
    """
    check nz format compute wheather ues fp32
    """
    use_fp32 = True
    if dtype == "float32":
        use_fp32 = True
        return use_fp32
    tbe_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if tbe_product in ("Hi3796CV300ES", "Hi3796CV300CS"):
        use_fp32 = False
        return use_fp32
    if shape[0] * shape[3] * 4 > UB_SIZE_LIMIT:
        use_fp32 = False
        return use_fp32
    return use_fp32


def compute_nopad_fp32(tensor_in, shape):
    """
    the compute produce, handling the scenes without padding
    """
    # preparing
    reduce_shape = (shape[0], 1, shape[2], shape[3], 1)
    op_list = []
    instruction_list = []

    # move data from gm to ub
    tensor_in_ub = \
        tvm.compute(shape, lambda *i: tensor_in(*i), name="tensor_in_ub")
    op_list += [tensor_in_ub]
    instruction_list += ['dma_copy']

    if tensor_in.dtype == "float32":
        # vcov fp32tofp16
        tensor_in_ub_fp16 = te.lang.cce.cast_to(tensor_in_ub, "float16")
        tensor_in_ub_fp16 = tvm.compute(shape,
                                        lambda *i: topi.cast(tensor_in_ub(*i), "float16"),
                                        name='res_vonv_fp16_tensor')
        op_list += [tensor_in_ub_fp16]
        instruction_list += ['vector_conv']

        # reduce max
        i = tvm.reduce_axis((0, shape[1]), "c1_max")
        j = tvm.reduce_axis((0, shape[4]), "c0_max")
        res_max = tvm.compute(reduce_shape,
                              lambda n, c1, h, w, c0:
                              tvm.max(tensor_in_ub_fp16[n, i, h, w, j],
                                      axis=[i, j]), name="res_max")
        op_list += [res_max]
        instruction_list += ['vector_reduce_max']

        res_max_fp32 = te.lang.cce.cast_to(res_max, "float32")
        res_max_fp32 = tvm.compute(shape,
                                   lambda *i: topi.cast(res_max(*i), "float32"),
                                   name='res_vonv_fp32_max')
        op_list += [res_max_fp32]
        instruction_list += ['vector_conv']

        # sub
        minus = tvm.const(-1, 'float32')
        res_minus = tvm.compute(reduce_shape,
                                lambda *i: minus * res_max_fp32(*i), name="res_minus")
        op_list += [res_minus]
        instruction_list += ['vector_muls']

        res_sub_fp32 = tvm.compute(shape, lambda n, c1, h, w, c0:
        tensor_in_ub[n, c1, h, w, c0] + res_minus[n, 0, h, w, 0],
                                   name="res_sub")
        op_list += [res_sub_fp32]
        instruction_list += ['vector_adds']

        res_sub = te.lang.cce.cast_to(res_sub_fp32, "float16")
        res_sub = tvm.compute(shape,
                              lambda *i: topi.cast(res_sub_fp32(*i), "float16"),
                              name='res_vonv_fp16_sub')
        op_list += [res_sub]
        instruction_list += ['vector_conv']

    else:
        # reduce max
        i = tvm.reduce_axis((0, shape[1]), "c1_max")
        j = tvm.reduce_axis((0, shape[4]), "c0_max")
        res_max = tvm.compute(reduce_shape,
                              lambda n, c1, h, w, c0:
                              tvm.max(tensor_in_ub[n, i, h, w, j], axis=[i, j]),
                              name="res_max")
        op_list += [res_max]
        instruction_list += ['vector_reduce_max']

        # sub
        minus = tvm.const(-1, 'float16')
        res_minus = tvm.compute(reduce_shape, lambda *i: minus * res_max(*i), name="res_minus")
        op_list += [res_minus]
        instruction_list += ['vector_muls']

        res_sub = tvm.compute(shape,
                              lambda n, c1, h, w, c0:
                              tensor_in_ub[n, c1, h, w, c0] + res_minus[n, 0, h, w, 0],
                              name="res_sub")
        op_list += [res_sub]
        instruction_list += ['vector_adds']

    if not tbe_platform.cce_conf.intrinsic_check_support("Intrinsic_exp",
                                                         "float32"):
        # exp
        res_exp = tvm.compute(shape, lambda *i: tvm.exp(res_sub(*i)), name="res_exp")
        op_list += [res_exp]
        instruction_list += ['vector_exp']

        # vcov fp16tofp32
        res_exp_fp32 = te.lang.cce.cast_to(res_exp, "float32")
        res_exp_fp32 = tvm.compute(shape,
                                   lambda *i: topi.cast(res_exp(*i), "float32"),
                                   name='res_vonv_fp32_exp')
        op_list += [res_exp_fp32]
        instruction_list += ['vector_conv']
    else:
        # vcov fp16tofp32
        res_sub_fp32 = te.lang.cce.cast_to(res_sub, "float32")
        res_sub_fp32 = tvm.compute(shape, lambda *i: topi.cast(res_sub(*i), "float32"),
                                   name='res_vonv_fp32_exp')
        op_list += [res_sub_fp32]
        instruction_list += ['vector_conv']

        # exp
        res_exp_fp32 = tvm.compute(shape, lambda *i: tvm.exp(res_sub_fp32(*i)), name="res_exp")
        op_list += [res_exp_fp32]
        instruction_list += ['vector_exp']


    # reduce sum
    ii = tvm.reduce_axis((0, shape[1]), "c1_sum")
    jj = tvm.reduce_axis((0, shape[4]), "c0_sum")
    res_sum = tvm.compute(reduce_shape,
                          lambda n, c1, h, w, c0:
                          tvm.sum(res_exp_fp32[n, ii, h, w, jj], axis=[ii, jj]),
                          name="res_sum")
    op_list += [res_sum]
    instruction_list += ['vector_reduce_sum']

    # rec
    res_rec = tvm.compute(reduce_shape, lambda *i: 1/(res_sum(*i)), name="res_rec")
    op_list += [res_rec]
    instruction_list += ['vector_rec']

    #loop 1  do newton iteration
    # vmlu
    res_mul_newton = tvm.compute(reduce_shape,
                                 lambda *i: res_rec(*i) * res_sum(*i), name="res_mul_newton")
    op_list += [res_mul_newton]
    instruction_list += ['vector_mul']

    # vmuls
    const1 = tvm.const(-1, 'float32')
    res_muls_newton = tvm.compute(reduce_shape,
                                  lambda *i: const1 * res_mul_newton(*i), name="res_muls_newton")
    op_list += [res_muls_newton]
    instruction_list += ['vector_muls']

    # vadds
    const2 = tvm.const(2, 'float32')
    res_adds_newton = tvm.compute(reduce_shape,
                                  lambda *i: const2 + res_muls_newton(*i), name="res_adds_newton")
    op_list += [res_adds_newton]
    instruction_list += ['vector_adds']

    # vmlu
    res_mul_newton = tvm.compute(reduce_shape,
                                 lambda *i: res_rec(*i) * res_adds_newton(*i),
                                 name="res_mul_newton")
    op_list += [res_mul_newton]
    instruction_list += ['vector_mul']

    # mul
    res_mul = tvm.compute(shape,
                          lambda n, c1, h, w, c0:
                          res_exp_fp32[n, c1, h, w, c0] * res_mul_newton[n, 0, h, w, 0],
                          name="res_mul")
    op_list += [res_mul]
    instruction_list += ['vector_muls']

    if tensor_in.dtype == "float16":
        res_mul_fp16 = te.lang.cce.cast_to(res_mul, "float16")
        res_mul_fp16 = tvm.compute(shape,
                                   lambda *i: topi.cast(res_mul(*i), "float16"),
                                   name='res_vonv_fp16')
        op_list += [res_mul_fp16]
        instruction_list += ['vector_conv']

        # move data from ub to gm
        res = tvm.compute(shape, lambda *i: res_mul_fp16(*i), name="result_out")
        op_list += [res]
        instruction_list += ['dma_copy']
    else:
        # move data from ub to gm
        res = tvm.compute(shape, lambda *i: res_mul(*i), name="result_out")
        op_list += [res]
        instruction_list += ['dma_copy']

    # schedule
    schedule = tvm.create_schedule([res.op])
    return schedule, op_list, instruction_list


def compute_nz_nopad_fp32(tensor_in, shape):
    """
    the compute produce, handling the scenes without padding
    """
    # preparing
    reduce_shape = (1, shape[1], shape[2], 1)
    op_list = []
    instruction_list = []

    # move data from gm to ub
    tensor_in_ub = \
        tvm.compute(shape, lambda *i: tensor_in(*i), name="tensor_in_ub")
    op_list += [tensor_in_ub]
    instruction_list += ['dma_copy']

    if tensor_in.dtype == "float32":
        # vcov fp32tofp16
        tensor_in_ub_fp16 = te.lang.cce.cast_to(tensor_in_ub, "float16")
        tensor_in_ub_fp16 = tvm.compute(shape,
                                        lambda *i: topi.cast(tensor_in_ub(*i), "float16"),
                                        name='res_vonv_fp16_tensor')
        op_list += [tensor_in_ub_fp16]
        instruction_list += ['vector_conv']

        # reduce max
        i = tvm.reduce_axis((0, shape[0]), "c1_max")
        j = tvm.reduce_axis((0, shape[3]), "c0_max")
        res_max = tvm.compute(reduce_shape,
                              lambda c1, n1, n0, c0:
                              tvm.max(tensor_in_ub_fp16[i, n1, n0, j],
                                      axis=[i, j]), name="res_max")
        op_list += [res_max]
        instruction_list += ['vector_reduce_max']

        res_max_fp32 = te.lang.cce.cast_to(res_max, "float32")
        res_max_fp32 = tvm.compute(shape,
                                   lambda *i: topi.cast(res_max(*i), "float32"),
                                   name='res_vonv_fp32_max')
        op_list += [res_max_fp32]
        instruction_list += ['vector_conv']

        # sub
        minus = tvm.const(-1, 'float32')
        res_minus = tvm.compute(reduce_shape,
                                lambda *i: minus * res_max_fp32(*i), name="res_minus")
        op_list += [res_minus]
        instruction_list += ['vector_muls']

        res_sub_fp32 = tvm.compute(shape, lambda c1, n1, n0, c0:
        tensor_in_ub[c1, n1, n0, c0] + res_minus[0, n1, n0, 0],
                                   name="res_sub")
        op_list += [res_sub_fp32]
        instruction_list += ['vector_adds']

        res_sub = te.lang.cce.cast_to(res_sub_fp32, "float16")
        res_sub = tvm.compute(shape,
                              lambda *i: topi.cast(res_sub_fp32(*i), "float16"),
                              name='res_vonv_fp16_sub')
        op_list += [res_sub]
        instruction_list += ['vector_conv']

    else:
        # reduce max
        i = tvm.reduce_axis((0, shape[0]), "c1_max")
        j = tvm.reduce_axis((0, shape[3]), "c0_max")
        res_max = tvm.compute(reduce_shape,
                              lambda c1, n1, n0, c0:
                              tvm.max(tensor_in_ub[i, n1, n0, j], axis=[i, j]),
                              name="res_max")
        op_list += [res_max]
        instruction_list += ['vector_reduce_max']

        # sub
        minus = tvm.const(-1, 'float16')
        res_minus = tvm.compute(reduce_shape, lambda *i: minus * res_max(*i), name="res_minus")
        op_list += [res_minus]
        instruction_list += ['vector_muls']

        res_sub = tvm.compute(shape,
                              lambda c1, n1, n0, c0:
                              tensor_in_ub[c1, n1, n0, c0] + res_minus[0, n1, n0, 0],
                              name="res_sub")
        op_list += [res_sub]
        instruction_list += ['vector_adds']

    if not tbe_platform.cce_conf.intrinsic_check_support("Intrinsic_exp",
                                                         "float32"):
        # exp
        res_exp = tvm.compute(shape, lambda *i: tvm.exp(res_sub(*i)), name="res_exp")
        op_list += [res_exp]
        instruction_list += ['vector_exp']

        # vcov fp16tofp32
        res_exp_fp32 = te.lang.cce.cast_to(res_exp, "float32")
        res_exp_fp32 = tvm.compute(shape,
                                   lambda *i: topi.cast(res_exp(*i), "float32"),
                                   name='res_vonv_fp32_exp')
        op_list += [res_exp_fp32]
        instruction_list += ['vector_conv']
    else:
        # vcov fp16tofp32
        res_sub_fp32 = te.lang.cce.cast_to(res_sub, "float32")
        res_sub_fp32 = tvm.compute(shape, lambda *i: topi.cast(res_sub(*i), "float32"),
                                   name='res_vonv_fp32_exp')
        op_list += [res_sub_fp32]
        instruction_list += ['vector_conv']

        # exp
        res_exp_fp32 = tvm.compute(shape, lambda *i: tvm.exp(res_sub_fp32(*i)), name="res_exp")
        op_list += [res_exp_fp32]
        instruction_list += ['vector_exp']


    # reduce sum
    ii = tvm.reduce_axis((0, shape[0]), "c1_sum")
    jj = tvm.reduce_axis((0, shape[3]), "c0_sum")
    res_sum = tvm.compute(reduce_shape,
                          lambda c1, n1, n0, c0:
                          tvm.sum(res_exp_fp32[ii, n1, n0, jj], axis=[ii, jj]),
                          name="res_sum")
    op_list += [res_sum]
    instruction_list += ['vector_reduce_sum']

    # rec
    res_rec = tvm.compute(reduce_shape, lambda *i: 1/(res_sum(*i)), name="res_rec")
    op_list += [res_rec]
    instruction_list += ['vector_rec']

    #loop 1  do newton iteration
    # vmlu
    res_mul_newton = tvm.compute(reduce_shape,
                                 lambda *i: res_rec(*i) * res_sum(*i), name="res_mul_newton")
    op_list += [res_mul_newton]
    instruction_list += ['vector_mul']

    # vmuls
    const1 = tvm.const(-1, 'float32')
    res_muls_newton = tvm.compute(reduce_shape,
                                  lambda *i: const1 * res_mul_newton(*i), name="res_muls_newton")
    op_list += [res_muls_newton]
    instruction_list += ['vector_muls']

    # vadds
    const2 = tvm.const(2, 'float32')
    res_adds_newton = tvm.compute(reduce_shape,
                                  lambda *i: const2 + res_muls_newton(*i), name="res_adds_newton")
    op_list += [res_adds_newton]
    instruction_list += ['vector_adds']

    # vmlu
    res_mul_newton = tvm.compute(reduce_shape,
                                 lambda *i: res_rec(*i) * res_adds_newton(*i),
                                 name="res_mul_newton")
    op_list += [res_mul_newton]
    instruction_list += ['vector_mul']

    # mul
    res_mul = tvm.compute(shape,
                          lambda c1, n1, n0, c0:
                          res_exp_fp32[c1, n1, n0, c0] * res_mul_newton[0, n1, n0, 0],
                          name="res_mul")
    op_list += [res_mul]
    instruction_list += ['vector_muls']

    if tensor_in.dtype == "float16":
        res_mul_fp16 = te.lang.cce.cast_to(res_mul, "float16")
        res_mul_fp16 = tvm.compute(shape,
                                   lambda *i: topi.cast(res_mul(*i), "float16"),
                                   name='res_vonv_fp16')
        op_list += [res_mul_fp16]
        instruction_list += ['vector_conv']

        # move data from ub to gm
        res = tvm.compute(shape, lambda *i: res_mul_fp16(*i), name="result_out")
        op_list += [res]
        instruction_list += ['dma_copy']
    else:
        # move data from ub to gm
        res = tvm.compute(shape, lambda *i: res_mul(*i), name="result_out")
        op_list += [res]
        instruction_list += ['dma_copy']

    # schedule
    schedule = tvm.create_schedule([res.op])
    return schedule, op_list, instruction_list


def compute_nopad(tensor_in, shape):
    """
    the compute produce, handling the scenes without padding
    """
    # preparing
    reduce_shape = (shape[0], 1, shape[2], shape[3], 1)
    op_list = []
    instruction_list = []

    # move data from gm to ub
    tensor_in_ub = \
        tvm.compute(shape, lambda *i: tensor_in(*i), name="tensor_in_ub")
    op_list += [tensor_in_ub]
    instruction_list += ['dma_copy']

    # reduce max
    i = tvm.reduce_axis((0, shape[1]), "c1_max")
    j = tvm.reduce_axis((0, shape[4]), "c0_max")
    res_max = tvm.compute(reduce_shape,
                          lambda n, c1, h, w, c0:
                          tvm.max(tensor_in_ub[n, i, h, w, j],
                                  axis=[i, j]), name="res_max")
    op_list += [res_max]
    instruction_list += ['vector_reduce_max']

    # sub
    minus = tvm.const(-1, 'float16')
    res_minus = tvm.compute(reduce_shape,
                            lambda *i:
                            minus * res_max(*i), name="res_minus")
    op_list += [res_minus]
    instruction_list += ['vector_muls']

    res_sub = tvm.compute(shape,
                          lambda n, c1, h, w, c0:
                          tensor_in_ub[n, c1, h, w, c0] +
                          res_minus[n, 0, h, w, 0], name="res_sub")
    op_list += [res_sub]
    instruction_list += ['vector_adds']

    # exp
    res_exp = \
        tvm.compute(shape, lambda *i: tvm.exp(res_sub(*i)), name="res_exp")
    op_list += [res_exp]
    instruction_list += ['vector_exp']

    # reduce sum
    ii = tvm.reduce_axis((0, shape[1]), "c1_sum")
    jj = tvm.reduce_axis((0, shape[4]), "c0_sum")
    res_sum = tvm.compute(reduce_shape,
                          lambda n, c1, h, w, c0:
                          tvm.sum(res_exp[n, ii, h, w, jj], axis=[ii, jj]),
                          name="res_sum")
    op_list += [res_sum]
    instruction_list += ['vector_reduce_sum']

    # rec
    res_rec = tvm.compute(reduce_shape,
                          lambda *i: 1/(res_sum(*i)), name="res_rec")
    op_list += [res_rec]
    instruction_list += ['vector_rec']

    # mul
    res_mul = tvm.compute(shape,
                          lambda n, c1, h, w, c0:
                          res_exp[n, c1, h, w, c0] * res_rec[n, 0, h, w, 0],
                          name="res_mul")
    op_list += [res_mul]
    instruction_list += ['vector_muls']

    # move data from ub to gm
    res = tvm.compute(shape, lambda *i: res_mul(*i), name="result_out")
    op_list += [res]
    instruction_list += ['dma_copy']

    # schedule
    schedule = tvm.create_schedule([res.op])
    return schedule, op_list, instruction_list


def compute_nz_nopad(tensor_in, shape):
    """
    the compute produce, handling the scenes without padding
    """
    # preparing
    reduce_shape = (1, shape[1], shape[2], 1)
    op_list = []
    instruction_list = []

    # move data from gm to ub
    tensor_in_ub = \
        tvm.compute(shape, lambda *i: tensor_in(*i), name="tensor_in_ub")
    op_list += [tensor_in_ub]
    instruction_list += ['dma_copy']

    # reduce max
    i = tvm.reduce_axis((0, shape[0]), "c1_max")
    j = tvm.reduce_axis((0, shape[3]), "c0_max")
    res_max = tvm.compute(reduce_shape,
                          lambda c1, n1, n0, c0:
                          tvm.max(tensor_in_ub[i, n1, n0, j],
                                  axis=[i, j]), name="res_max")
    op_list += [res_max]
    instruction_list += ['vector_reduce_max']

    # sub
    minus = tvm.const(-1, 'float16')
    res_minus = tvm.compute(reduce_shape,
                            lambda *i:
                            minus * res_max(*i), name="res_minus")
    op_list += [res_minus]
    instruction_list += ['vector_muls']

    res_sub = tvm.compute(shape,
                          lambda c1, n1, n0, c0:
                          tensor_in_ub[c1, n1, n0, c0] +
                          res_minus[0, n1, n0, 0], name="res_sub")
    op_list += [res_sub]
    instruction_list += ['vector_adds']

    # exp
    res_exp = \
        tvm.compute(shape, lambda *i: tvm.exp(res_sub(*i)), name="res_exp")
    op_list += [res_exp]
    instruction_list += ['vector_exp']

    # reduce sum
    ii = tvm.reduce_axis((0, shape[0]), "c1_sum")
    jj = tvm.reduce_axis((0, shape[3]), "c0_sum")
    res_sum = tvm.compute(reduce_shape,
                          lambda c1, n1, n0, c0:
                          tvm.sum(res_exp[ii, n1, n0, jj], axis=[ii, jj]),
                          name="res_sum")
    op_list += [res_sum]
    instruction_list += ['vector_reduce_sum']

    # rec
    res_rec = tvm.compute(reduce_shape,
                          lambda *i: 1/(res_sum(*i)), name="res_rec")
    op_list += [res_rec]
    instruction_list += ['vector_rec']

    # mul
    res_mul = tvm.compute(shape,
                          lambda c1, n1, n0, c0:
                          res_exp[c1, n1, n0, c0] * res_rec[0, n1, n0, 0],
                          name="res_mul")
    op_list += [res_mul]
    instruction_list += ['vector_muls']

    # move data from ub to gm
    res = tvm.compute(shape, lambda *i: res_mul(*i), name="result_out")
    op_list += [res]
    instruction_list += ['dma_copy']

    # schedule
    schedule = tvm.create_schedule([res.op])
    return schedule, op_list, instruction_list


def compute_padding_fp32(tensor_in, shape, pad_param, impl_mode):
    """
    the compute produce, handling the scenes with padding
    """
    # preparing
    reduce_shape = (shape[0], 1, shape[2], shape[3], 1)
    pad_c1 = pad_param[0]
    pad_c0 = pad_param[1]
    op_list = []
    instruction_list = []
    # move data from gm to ub
    tensor_in_ub = tvm.compute(shape, lambda *i: tensor_in(*i), name="tensor_in_ub")
    op_list += [tensor_in_ub]
    instruction_list += ['dma_copy']

    if tensor_in.dtype == "float16":
        if shape[1] == pad_c1:
            # reduce max
            i = tvm.reduce_axis((0, shape[1]), "c1_max_pad")
            j = tvm.reduce_axis((0, shape[4] - pad_c0), "c0_max_pad")
            res_max = tvm.compute(reduce_shape,
                                  lambda n, c1, h, w, c0:
                                  tvm.max(tensor_in_ub[n, i, h, w, j], axis=[i, j]),
                                  name="res_max")
            op_list += [res_max]
            if pad_c0 != 15:
                instruction_list += ['vector_reduce_max']
            else:
                instruction_list += ['vector_auto']
        else:
            # reduce max
            i = tvm.reduce_axis((0, shape[1] - pad_c1), "c1_max")
            j = tvm.reduce_axis((0, shape[4]), "c0_max")
            gmax1 = tvm.compute(reduce_shape,
                                lambda n, c1, h, w, c0:
                                tvm.max(tensor_in_ub[n, i, h, w, j], axis=[i, j]),
                                name="gmax1")
            op_list += [gmax1]
            instruction_list += ['vector_reduce_max']

            i = tvm.reduce_axis((shape[1] - pad_c1, shape[1]), "c1_max_pad")
            j = tvm.reduce_axis((0, shape[4] - pad_c0), "c0_max_pad")
            gmax2 = tvm.compute(reduce_shape,
                                lambda n, c1, h, w, c0:
                                tvm.max(tensor_in_ub[n, i, h, w, j], axis=[i, j]),
                                name="gmax2")
            op_list += [gmax2]
            if pad_c0 != 15:
                instruction_list += ['vector_reduce_max']
            else:
                instruction_list += ['vector_auto']

            res_max = tvm.compute(reduce_shape,
                                  lambda *i: tvm.max(gmax1(*i), gmax2(*i)), name="res_max")
            op_list += [res_max]
            instruction_list += ['vector_max']
    else:
        # vcov fp32tofp16
        tensor_in_ub_fp16 = te.lang.cce.cast_to(tensor_in_ub, "float16")
        tensor_in_ub_fp16 = tvm.compute(shape,
                                        lambda *i: topi.cast(tensor_in_ub(*i), "float16"),
                                        name='res_vonv_fp16_tensor')
        op_list += [tensor_in_ub_fp16]
        instruction_list += ['vector_conv']

        if shape[1] == pad_c1:
            # reduce max
            i = tvm.reduce_axis((0, shape[1]), "c1_max_pad")
            j = tvm.reduce_axis((0, shape[4] - pad_c0), "c0_max_pad")
            res_max = tvm.compute(reduce_shape,
                                  lambda n, c1, h, w, c0:
                                  tvm.max(tensor_in_ub_fp16[n, i, h, w, j], axis=[i, j]),
                                  name="res_max")
            op_list += [res_max]
            instruction_list += ['vector_reduce_max']
        else:
            # reduce max
            i = tvm.reduce_axis((0, shape[1] - pad_c1), "c1_max")
            j = tvm.reduce_axis((0, shape[4]), "c0_max")
            gmax1 = tvm.compute(reduce_shape,
                                lambda n, c1, h, w, c0:
                                tvm.max(tensor_in_ub_fp16[n, i, h, w, j], axis=[i, j]),
                                name="gmax1")
            op_list += [gmax1]
            instruction_list += ['vector_reduce_max']

            i = tvm.reduce_axis((shape[1] - pad_c1, shape[1]), "c1_max_pad")
            j = tvm.reduce_axis((0, shape[4] - pad_c0), "c0_max_pad")
            gmax2 = tvm.compute(reduce_shape,
                                lambda n, c1, h, w, c0:
                                tvm.max(tensor_in_ub_fp16[n, i, h, w, j], axis=[i, j]),
                                name="gmax2")
            op_list += [gmax2]
            instruction_list += ['vector_reduce_max']

            res_max = tvm.compute(reduce_shape,
                                  lambda *i: tvm.max(gmax1(*i), gmax2(*i)),
                                  name="res_max")
            op_list += [res_max]
            instruction_list += ['vector_max']

    res_max_broadcast = te.lang.cce.broadcast(res_max, shape)
    op_list += [res_max_broadcast]
    instruction_list += ['vector_broadcast']

    # sub
    if tensor_in.dtype == "float32":
        res_sub = tvm.compute(shape,
                              lambda *i:
                              tensor_in_ub_fp16(*i) - res_max_broadcast(*i),
                              name="res_sub")
        op_list += [res_sub]
        instruction_list += ['vector_sub']
    else:
        res_sub = tvm.compute(shape,
                              lambda *i:
                              tensor_in_ub(*i) - res_max_broadcast(*i),
                              name="res_sub")
        op_list += [res_sub]
        instruction_list += ['vector_sub']

    # exp
    res_exp = tvm.compute(shape, lambda *i: tvm.exp(res_sub(*i)), name="res_exp")
    op_list += [res_exp]
    instruction_list += ['vector_exp']

    # vcov fp16tofp32
    res_exp_fp32 = te.lang.cce.cast_to(res_exp, "float32")
    res_exp_fp32 = tvm.compute(shape,
                               lambda *i: topi.cast(res_exp(*i), "float32"),
                               name='res_vonv_exp')
    op_list += [res_exp_fp32]
    instruction_list += ['vector_conv']


    if shape[1] == pad_c1:
        if pad_c0 != 15:
            # reduce sum
            ii = tvm.reduce_axis((0, shape[1]), "c1_sum_pad")
            jj = tvm.reduce_axis((0, shape[4] - pad_c0), "c0_sum_pad")
            res_sum = tvm.compute(reduce_shape,
                                    lambda n, c1, h, w, c0:
                                    tvm.sum(res_exp_fp32[n, ii, h, w, jj], axis=[ii, jj]),
                                    name="res_sum")
            op_list += [res_sum]
            instruction_list += ['vector_reduce_sum']
        else:
            res_sum = tvm.compute(reduce_shape,
                            lambda n, c1, h, w, c0:
                            res_exp_fp32[n, shape[1] - 1, h, w, 0], name="res_sum")
            op_list += [res_sum]
            instruction_list += ['vector_auto']
    else:
        # reduce sum
        ii = tvm.reduce_axis((0, shape[1] - pad_c1), "c1_sum")
        jj = tvm.reduce_axis((0, shape[4]), "c0_sum")
        sum1 = tvm.compute(reduce_shape,
                           lambda n, c1, h, w, c0:
                           tvm.sum(res_exp_fp32[n, ii, h, w, jj], axis=[ii, jj]),
                           name="sum1")
        op_list += [sum1]
        instruction_list += ['vector_reduce_sum']

        if pad_c0 != 15:
            ii = tvm.reduce_axis((shape[1] - pad_c1, shape[1]), "c1_sum_pad")
            jj = tvm.reduce_axis((0, shape[4] - pad_c0), "c0_sum_pad")
            sum2 = tvm.compute(reduce_shape,
                            lambda n, c1, h, w, c0:
                            tvm.sum(res_exp_fp32[n, ii, h, w, jj],
                                    axis=[ii, jj]), name="sum2")
            op_list += [sum2]
            instruction_list += ['vector_reduce_sum']
        else:
            sum2 = tvm.compute(reduce_shape,
                            lambda n, c1, h, w, c0:
                            res_exp_fp32[n, shape[1] - 1, h, w, 0], name="sum2")
            op_list += [sum2]
            instruction_list += ['vector_auto']

        res_sum = tvm.compute(reduce_shape,
                              lambda *i: tvm.sum(sum1(*i), sum2(*i)), name="res_sum")
        op_list += [res_sum]
        instruction_list += ['vector_add']

    # judge the platform is mini or not
    if tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp", "float32"):
        res_mul_newton_broadcast = te.lang.cce.broadcast(res_sum, shape)
        op_list += [res_mul_newton_broadcast]
        instruction_list += ['vector_broadcast']

        res_mul = tvm.compute(shape,
                              lambda *i:
                              res_exp_fp32(*i) / res_mul_newton_broadcast(*i),
                              name="res_mul")
        op_list += [res_mul]
        instruction_list += ['vector_div']
    else:
        # rec
        res_rec = tvm.compute(reduce_shape, lambda *i: 1/(res_sum(*i)), name="res_rec")
        op_list += [res_rec]
        instruction_list += ['vector_rec']

        if impl_mode == "high_performance":
            res_mul_newton_broadcast = te.lang.cce.broadcast(res_rec, shape)
            op_list += [res_mul_newton_broadcast]
            instruction_list += ['vector_broadcast']
        else:
            #loop 1
            # vmlu
            res_mul_newton = tvm.compute(reduce_shape,
                                     lambda *i: res_rec(*i) * res_sum(*i), name="res_mul_newton")
            op_list += [res_mul_newton]
            instruction_list += ['vector_mul']

            res_const2 = tvm.compute(reduce_shape,
                                    lambda *i: 2.0,
                                    name="res_const2")
            op_list += [res_const2]
            instruction_list += ['vector_auto']

            # vsub
            res_sub_newton = tvm.compute(reduce_shape,
                                         lambda *i: res_const2(*i) - res_mul_newton(*i),
                                         name="res_sub_newton")
            op_list += [res_sub_newton]
            instruction_list += ['vector_sub']

            # vmlu
            res_mul_newton = tvm.compute(reduce_shape,
                                         lambda *i: res_sub_newton(*i) * res_rec(*i),
                                         name="res_mul_newton")
            op_list += [res_mul_newton]
            instruction_list += ['vector_mul']

            res_mul_newton_broadcast = te.lang.cce.broadcast(res_mul_newton, shape)
            op_list += [res_mul_newton_broadcast]
            instruction_list += ['vector_broadcast']
        # mul
        res_mul = tvm.compute(shape,
                            lambda *i:
                            res_exp_fp32(*i) * res_mul_newton_broadcast(*i),
                            name="res_mul")
        op_list += [res_mul]
        instruction_list += ['vector_mul']


    if tensor_in.dtype == "float16":
        res_mul_fp16 = te.lang.cce.cast_to(res_mul, "float16")
        res_mul_fp16 = tvm.compute(shape,
                                   lambda *i: topi.cast(res_mul(*i), "float16"),
                                   name='res_vonv_fp16')
        op_list += [res_mul_fp16]
        instruction_list += ['vector_conv']

        # move data from ub to gm
        res = tvm.compute(shape, lambda *i: res_mul_fp16(*i), name="result_out")
        op_list += [res]
        instruction_list += ['dma_copy']
    else:
        # move data from ub to gm
        res = tvm.compute(shape, lambda *i: res_mul(*i), name="result_out")
        op_list += [res]
        instruction_list += ['dma_copy']

    #schedule
    schedule = tvm.create_schedule([res.op])
    return schedule, op_list, instruction_list


def compute_nz_padding_fp32(tensor_in, shape, pad_param):
    """
    the compute produce, handling the scenes with padding
    """
    # preparing
    reduce_shape = (1, shape[1], shape[2], 1)
    pad_c1 = pad_param[0]
    pad_c0 = pad_param[1]
    op_list = []
    instruction_list = []
    # move data from gm to ub
    tensor_in_ub = tvm.compute(shape, lambda *i: tensor_in(*i), name="tensor_in_ub")
    op_list += [tensor_in_ub]
    instruction_list += ['dma_copy']

    if tensor_in.dtype == "float16":
        if shape[0] == pad_c1:
            # reduce max
            i = tvm.reduce_axis((0, shape[0]), "c1_max_pad")
            j = tvm.reduce_axis((0, shape[3] - pad_c0), "c0_max_pad")
            res_max = tvm.compute(reduce_shape,
                                  lambda c1, n1, n0, c0:
                                  tvm.max(tensor_in_ub[i, n1, n0, j], axis=[i, j]),
                                  name="res_max")
            op_list += [res_max]
            if pad_c0 != 15:
                instruction_list += ['vector_reduce_max']
            else:
                instruction_list += ['vector_auto']
        else:
            # reduce max
            i = tvm.reduce_axis((0, shape[0] - pad_c1), "c1_max")
            j = tvm.reduce_axis((0, shape[3]), "c0_max")
            gmax1 = tvm.compute(reduce_shape,
                                lambda c1, n1, n0, c0:
                                tvm.max(tensor_in_ub[i, n1, n0, j], axis=[i, j]),
                                name="gmax1")
            op_list += [gmax1]
            instruction_list += ['vector_reduce_max']

            i = tvm.reduce_axis((shape[0] - pad_c1, shape[0]), "c1_max_pad")
            j = tvm.reduce_axis((0, shape[3] - pad_c0), "c0_max_pad")
            gmax2 = tvm.compute(reduce_shape,
                                lambda c1, n1, n0, c0:
                                tvm.max(tensor_in_ub[i, n1, n0, j], axis=[i, j]),
                                name="gmax2")
            op_list += [gmax2]
            if pad_c0 != 15:
                instruction_list += ['vector_reduce_max']
            else:
                instruction_list += ['vector_auto']

            res_max = tvm.compute(reduce_shape,
                                  lambda *i: tvm.max(gmax1(*i), gmax2(*i)), name="res_max")
            op_list += [res_max]
            instruction_list += ['vector_max']
    else:
        # vcov fp32tofp16
        tensor_in_ub_fp16 = te.lang.cce.cast_to(tensor_in_ub, "float16")
        tensor_in_ub_fp16 = tvm.compute(shape,
                                        lambda *i: topi.cast(tensor_in_ub(*i), "float16"),
                                        name='res_vonv_fp16_tensor')
        op_list += [tensor_in_ub_fp16]
        instruction_list += ['vector_conv']

        if shape[1] == pad_c1:
            # reduce max
            i = tvm.reduce_axis((0, shape[0]), "c1_max_pad")
            j = tvm.reduce_axis((0, shape[3] - pad_c0), "c0_max_pad")
            res_max = tvm.compute(reduce_shape,
                                  lambda c1, n1, n0, c0:
                                  tvm.max(tensor_in_ub_fp16[i, n1, n0, j], axis=[i, j]),
                                  name="res_max")
            op_list += [res_max]
            instruction_list += ['vector_reduce_max']
        else:
            # reduce max
            i = tvm.reduce_axis((0, shape[0] - pad_c1), "c1_max")
            j = tvm.reduce_axis((0, shape[3]), "c0_max")
            gmax1 = tvm.compute(reduce_shape,
                                lambda c1, n1, n0, c0:
                                tvm.max(tensor_in_ub_fp16[i, n1, n0, j], axis=[i, j]),
                                name="gmax1")
            op_list += [gmax1]
            instruction_list += ['vector_reduce_max']

            i = tvm.reduce_axis((shape[0] - pad_c1, shape[0]), "c1_max_pad")
            j = tvm.reduce_axis((0, shape[3] - pad_c0), "c0_max_pad")
            gmax2 = tvm.compute(reduce_shape,
                                lambda c1, n1, n0, c0:
                                tvm.max(tensor_in_ub_fp16[i, n1, n0, j], axis=[i, j]),
                                name="gmax2")
            op_list += [gmax2]
            instruction_list += ['vector_reduce_max']

            res_max = tvm.compute(reduce_shape,
                                  lambda *i: tvm.max(gmax1(*i), gmax2(*i)),
                                  name="res_max")
            op_list += [res_max]
            instruction_list += ['vector_max']

    # sub
    minus = tvm.const(-1, 'float16')
    res_minus = tvm.compute(reduce_shape, lambda *i: minus * res_max(*i), name="res_minus")
    op_list += [res_minus]
    instruction_list += ['vector_muls']

    if tensor_in.dtype == "float32":
        res_sub = tvm.compute(shape,
                              lambda c1, n1, n0, c0:
                              tensor_in_ub_fp16[c1, n1, n0, c0] + res_minus[0, n1, n0, 0],
                              name="res_sub")
        op_list += [res_sub]
        instruction_list += ['vector_adds']
    else:
        res_sub = tvm.compute(shape,
                              lambda c1, n1, n0, c0:
                              tensor_in_ub[c1, n1, n0, c0] + res_minus[0, n1, n0, 0],
                              name="res_sub")
        op_list += [res_sub]
        instruction_list += ['vector_adds']

    # exp
    res_exp = tvm.compute(shape, lambda *i: tvm.exp(res_sub(*i)), name="res_exp")
    op_list += [res_exp]
    instruction_list += ['vector_exp']

    # vcov fp16tofp32
    res_exp_fp32 = te.lang.cce.cast_to(res_exp, "float32")
    res_exp_fp32 = tvm.compute(shape,
                               lambda *i: topi.cast(res_exp(*i), "float32"),
                               name='res_vonv_exp')
    op_list += [res_exp_fp32]
    instruction_list += ['vector_conv']


    if shape[0] == pad_c1:
        if pad_c0 != 15:
            # reduce sum
            ii = tvm.reduce_axis((0, shape[0]), "c1_sum_pad")
            jj = tvm.reduce_axis((0, shape[3] - pad_c0), "c0_sum_pad")
            res_sum = tvm.compute(reduce_shape,
                                  lambda c1, n1, n0, c0:
                                  tvm.sum(res_exp_fp32[ii, n1, n0, jj], axis=[ii, jj]),
                                  name="res_sum")
            op_list += [res_sum]
            instruction_list += ['vector_reduce_sum']
        else:
            res_sum = tvm.compute(reduce_shape,
                                  lambda c1, n1, n0, c0:
                                  res_exp_fp32[shape[0] - 1, n1, n0, 0], name="res_sum")
            op_list += [res_sum]
            instruction_list += ['vector_auto']
    else:
        # reduce sum
        ii = tvm.reduce_axis((0, shape[0] - pad_c1), "c1_sum")
        jj = tvm.reduce_axis((0, shape[3]), "c0_sum")
        sum1 = tvm.compute(reduce_shape,
                           lambda c1, n1, n0, c0:
                           tvm.sum(res_exp_fp32[ii, n1, n0, jj], axis=[ii, jj]),
                           name="sum1")
        op_list += [sum1]
        instruction_list += ['vector_reduce_sum']

        if pad_c0 != 15:
            ii = tvm.reduce_axis((shape[0] - pad_c1, shape[0]), "c1_sum_pad")
            jj = tvm.reduce_axis((0, shape[3] - pad_c0), "c0_sum_pad")
            sum2 = tvm.compute(reduce_shape,
                               lambda c1, n1, n0, c0:
                               tvm.sum(res_exp_fp32[ii, n1, n0, jj],
                                       axis=[ii, jj]), name="sum2")
            op_list += [sum2]
            instruction_list += ['vector_reduce_sum']
        else:
            sum2 = tvm.compute(reduce_shape,
                               lambda c1, n1, n0, c0:
                               res_exp_fp32[shape[0] - 1, n1, n0, 0], name="sum2")
            op_list += [sum2]
            instruction_list += ['vector_auto']

        res_sum = tvm.compute(reduce_shape,
                              lambda *i: tvm.sum(sum1(*i), sum2(*i)), name="res_sum")
        op_list += [res_sum]
        instruction_list += ['vector_add']

    # rec
    res_rec = tvm.compute(reduce_shape, lambda *i: 1/(res_sum(*i)), name="res_rec")
    op_list += [res_rec]
    instruction_list += ['vector_rec']

    #loop 1
    # vmlu
    res_mul_newton = tvm.compute(reduce_shape,
                                 lambda *i: res_rec(*i) * res_sum(*i), name="res_mul_newton")
    op_list += [res_mul_newton]
    instruction_list += ['vector_mul']

    # vmuls
    const1 = tvm.const(-1, 'float32')
    res_muls_newton = tvm.compute(reduce_shape,
                                  lambda *i: const1 * res_mul_newton(*i),
                                  name="res_muls_newton")
    op_list += [res_muls_newton]
    instruction_list += ['vector_muls']

    # vadds
    const2 = tvm.const(2, 'float32')
    res_adds_newton = tvm.compute(reduce_shape,
                                  lambda *i: const2 + res_muls_newton(*i),
                                  name="res_adds_newton")
    op_list += [res_adds_newton]
    instruction_list += ['vector_adds']

    # vmlu
    res_mul_newton = tvm.compute(reduce_shape,
                                 lambda *i: res_adds_newton(*i) * res_rec(*i),
                                 name="res_mul_newton")
    op_list += [res_mul_newton]
    instruction_list += ['vector_mul']


    # mul
    res_mul = tvm.compute(shape,
                          lambda c1, n1, n0, c0:
                          res_exp_fp32[c1, n1, n0, c0] * res_mul_newton[0, n1, n0, 0],
                          name="res_mul")
    op_list += [res_mul]
    instruction_list += ['vector_muls']


    if tensor_in.dtype == "float16":
        res_mul_fp16 = te.lang.cce.cast_to(res_mul, "float16")
        res_mul_fp16 = tvm.compute(shape,
                                   lambda *i: topi.cast(res_mul(*i), "float16"),
                                   name='res_vonv_fp16')
        op_list += [res_mul_fp16]
        instruction_list += ['vector_conv']

        # move data from ub to gm
        res = tvm.compute(shape, lambda *i: res_mul_fp16(*i), name="result_out")
        op_list += [res]
        instruction_list += ['dma_copy']
    else:
        # move data from ub to gm
        res = tvm.compute(shape, lambda *i: res_mul(*i), name="result_out")
        op_list += [res]
        instruction_list += ['dma_copy']

    #schedule
    schedule = tvm.create_schedule([res.op])
    return schedule, op_list, instruction_list


def compute_padding(tensor_in, shape, pad_param):
    """
    the compute produce, handling the scenes with padding
    """
    # preparing
    reduce_shape = (shape[0], 1, shape[2], shape[3], 1)
    pad_c1 = pad_param[0]
    pad_c0 = pad_param[1]
    op_list = []
    instruction_list = []

    # move data from gm to ub
    tensor_in_ub = tvm.compute(shape,
                               lambda *i: tensor_in(*i), name="tensor_in_ub")
    op_list += [tensor_in_ub]
    instruction_list += ['dma_copy']

    if shape[1] == pad_c1:
        # reduce max
        i = tvm.reduce_axis((0, shape[1]), "c1_max_pad")
        j = tvm.reduce_axis((0, shape[4] - pad_c0), "c0_max_pad")
        res_max = tvm.compute(reduce_shape,
                              lambda n, c1, h, w, c0:
                              tvm.max(tensor_in_ub[n, i, h, w, j],
                                      axis=[i, j]), name="res_max")
        op_list += [res_max]
        if pad_c0 != 15:
            instruction_list += ['vector_reduce_max']
        else:
            instruction_list += ['vector_auto']
    else:
        # reduce max
        i = tvm.reduce_axis((0, shape[1] - pad_c1), "c1_max")
        j = tvm.reduce_axis((0, shape[4]), "c0_max")
        gmax1 = tvm.compute(reduce_shape,
                            lambda n, c1, h, w, c0:
                            tvm.max(tensor_in_ub[n, i, h, w, j],
                                    axis=[i, j]), name="gmax1")
        op_list += [gmax1]
        instruction_list += ['vector_reduce_max']

        i = tvm.reduce_axis((shape[1] - pad_c1, shape[1]), "c1_max_pad")
        j = tvm.reduce_axis((0, shape[4] - pad_c0), "c0_max_pad")
        gmax2 = tvm.compute(reduce_shape,
                            lambda n, c1, h, w, c0:
                            tvm.max(tensor_in_ub[n, i, h, w, j],
                                    axis=[i, j]), name="gmax2")
        op_list += [gmax2]
        if pad_c0 != 15:
            instruction_list += ['vector_reduce_max']
        else:
            instruction_list += ['vector_auto']

        res_max = tvm.compute(reduce_shape,
                              lambda *i: tvm.max(gmax1(*i), gmax2(*i)),
                              name="res_max")
        op_list += [res_max]
        instruction_list += ['vector_max']

    # sub
    minus = tvm.const(-1, 'float16')
    res_minus = tvm.compute(reduce_shape,
                            lambda *i: minus * res_max(*i),
                            name="res_minus")
    op_list += [res_minus]
    instruction_list += ['vector_muls']

    res_sub = tvm.compute(shape,
                          lambda n, c1, h, w, c0:
                          tensor_in_ub[n, c1, h, w, c0] +
                          res_minus[n, 0, h, w, 0], name="res_sub")
    op_list += [res_sub]
    instruction_list += ['vector_adds']

    # exp
    res_exp = tvm.compute(shape,
                          lambda *i: tvm.exp(res_sub(*i)), name="res_exp")
    op_list += [res_exp]
    instruction_list += ['vector_exp']

    if shape[1] == pad_c1:
        if pad_c0 != 15:
            # reduce sum
            ii = tvm.reduce_axis((0, shape[1]), "c1_sum_pad")
            jj = tvm.reduce_axis((0, shape[4] - pad_c0), "c0_sum_pad")
            res_sum = tvm.compute(reduce_shape,
                                lambda n, c1, h, w, c0:
                                tvm.sum(res_exp[n, ii, h, w, jj],
                                        axis=[ii, jj]), name="res_sum")
            op_list += [res_sum]
            instruction_list += ['vector_reduce_sum']
        else:
            res_sum = tvm.compute(reduce_shape,
                            lambda n, c1, h, w, c0:
                            res_exp[n, shape[1] - 1, h, w, 0], name="res_sum")
            op_list += [res_sum]
            instruction_list += ['vector_auto']
    else:
        # reduce sum
        ii = tvm.reduce_axis((0, shape[1] - pad_c1), "c1_sum")
        jj = tvm.reduce_axis((0, shape[4]), "c0_sum")
        sum1 = tvm.compute(reduce_shape,
                           lambda n, c1, h, w, c0:
                           tvm.sum(res_exp[n, ii, h, w, jj],
                                   axis=[ii, jj]), name="sum1")
        op_list += [sum1]
        instruction_list += ['vector_reduce_sum']

        if pad_c0 != 15:
            ii = tvm.reduce_axis((shape[1] - pad_c1, shape[1]), "c1_sum_pad")
            jj = tvm.reduce_axis((0, shape[4] - pad_c0), "c0_sum_pad")
            sum2 = tvm.compute(reduce_shape,
                            lambda n, c1, h, w, c0:
                            tvm.sum(res_exp[n, ii, h, w, jj],
                                    axis=[ii, jj]), name="sum2")
            op_list += [sum2]
            instruction_list += ['vector_reduce_sum']
        else:
            sum2 = tvm.compute(reduce_shape,
                            lambda n, c1, h, w, c0:
                            res_exp[n, shape[1] - 1, h, w, 0], name="sum2")
            op_list += [sum2]
            instruction_list += ['vector_auto']

        res_sum = tvm.compute(reduce_shape,
                              lambda *i: tvm.sum(sum1(*i), sum2(*i)),
                              name="res_sum")
        op_list += [res_sum]
        instruction_list += ['vector_add']

    # rec
    res_rec = tvm.compute(reduce_shape,
                          lambda *i: 1/(res_sum(*i)), name="res_rec")
    op_list += [res_rec]
    instruction_list += ['vector_rec']

    # mul
    res_mul = tvm.compute(shape,
                          lambda n, c1, h, w, c0:
                          res_exp[n, c1, h, w, c0] * res_rec[n, 0, h, w, 0],
                          name="res_mul")
    op_list += [res_mul]
    instruction_list += ['vector_muls']

    # move data from ub to gm
    res = tvm.compute(shape, lambda *i: res_mul(*i), name="result_out")
    op_list += [res]
    instruction_list += ['dma_copy']

    schedule = tvm.create_schedule([res.op])
    return schedule, op_list, instruction_list


def compute_nz_padding(tensor_in, shape, pad_param):
    """
    the compute produce, handling the scenes with padding
    """
    # preparing
    reduce_shape = (1, shape[1], shape[2], 1)
    pad_c1 = pad_param[0]
    pad_c0 = pad_param[1]
    op_list = []
    instruction_list = []

    # move data from gm to ub
    tensor_in_ub = tvm.compute(shape,
                               lambda *i: tensor_in(*i), name="tensor_in_ub")
    op_list += [tensor_in_ub]
    instruction_list += ['dma_copy']

    if shape[0] == pad_c1:
        # reduce max
        i = tvm.reduce_axis((0, shape[0]), "c1_max_pad")
        j = tvm.reduce_axis((0, shape[3] - pad_c0), "c0_max_pad")
        res_max = tvm.compute(reduce_shape,
                              lambda c1, n1, n0, c0:
                              tvm.max(tensor_in_ub[i, n1, n0, j],
                                      axis=[i, j]), name="res_max")
        op_list += [res_max]
        if pad_c0 != 15:
            instruction_list += ['vector_reduce_max']
        else:
            instruction_list += ['vector_auto']
    else:
        # reduce max
        i = tvm.reduce_axis((0, shape[0] - pad_c1), "c1_max")
        j = tvm.reduce_axis((0, shape[3]), "c0_max")
        gmax1 = tvm.compute(reduce_shape,
                            lambda c1, n1, n0, c0:
                            tvm.max(tensor_in_ub[i, n1, n0, j],
                                    axis=[i, j]), name="gmax1")
        op_list += [gmax1]
        instruction_list += ['vector_reduce_max']

        i = tvm.reduce_axis((shape[0] - pad_c1, shape[0]), "c1_max_pad")
        j = tvm.reduce_axis((0, shape[3] - pad_c0), "c0_max_pad")
        gmax2 = tvm.compute(reduce_shape,
                            lambda c1, n1, n0, c0:
                            tvm.max(tensor_in_ub[i, n1, n0, j],
                                    axis=[i, j]), name="gmax2")
        op_list += [gmax2]
        if pad_c0 != 15:
            instruction_list += ['vector_reduce_max']
        else:
            instruction_list += ['vector_auto']

        res_max = tvm.compute(reduce_shape,
                              lambda *i: tvm.max(gmax1(*i), gmax2(*i)),
                              name="res_max")
        op_list += [res_max]
        instruction_list += ['vector_max']

    # sub
    minus = tvm.const(-1, 'float16')
    res_minus = tvm.compute(reduce_shape,
                            lambda *i: minus * res_max(*i),
                            name="res_minus")
    op_list += [res_minus]
    instruction_list += ['vector_muls']

    res_sub = tvm.compute(shape,
                          lambda c1, n1, n0, c0:
                          tensor_in_ub[c1, n1, n0, c0] +
                          res_minus[0, n1, n0, 0], name="res_sub")
    op_list += [res_sub]
    instruction_list += ['vector_adds']

    # exp
    res_exp = tvm.compute(shape,
                          lambda *i: tvm.exp(res_sub(*i)), name="res_exp")
    op_list += [res_exp]
    instruction_list += ['vector_exp']

    if shape[0] == pad_c1:
        if pad_c0 != 15:
            # reduce sum
            ii = tvm.reduce_axis((0, shape[0]), "c1_sum_pad")
            jj = tvm.reduce_axis((0, shape[3] - pad_c0), "c0_sum_pad")
            res_sum = tvm.compute(reduce_shape,
                                  lambda c1, n1, n0, c0:
                                  tvm.sum(res_exp[ii, n1, n0, jj],
                                          axis=[ii, jj]), name="res_sum")
            op_list += [res_sum]
            instruction_list += ['vector_reduce_sum']
        else:
            res_sum = tvm.compute(reduce_shape,
                                  lambda c1, n1, n0, c0:
                                  res_exp[shape[0] - 1, n1, n0, 0], name="res_sum")
            op_list += [res_sum]
            instruction_list += ['vector_auto']
    else:
        # reduce sum
        ii = tvm.reduce_axis((0, shape[0] - pad_c1), "c1_sum")
        jj = tvm.reduce_axis((0, shape[3]), "c0_sum")
        sum1 = tvm.compute(reduce_shape,
                           lambda c1, n1, n0, c0:
                           tvm.sum(res_exp[ii, n1, n0, jj],
                                   axis=[ii, jj]), name="sum1")
        op_list += [sum1]
        instruction_list += ['vector_reduce_sum']

        if pad_c0 != 15:
            ii = tvm.reduce_axis((shape[0] - pad_c1, shape[0]), "c1_sum_pad")
            jj = tvm.reduce_axis((0, shape[3] - pad_c0), "c0_sum_pad")
            sum2 = tvm.compute(reduce_shape,
                               lambda c1, n1, n0, c0:
                               tvm.sum(res_exp[ii, n1, n0, jj],
                                       axis=[ii, jj]), name="sum2")
            op_list += [sum2]
            instruction_list += ['vector_reduce_sum']
        else:
            sum2 = tvm.compute(reduce_shape,
                               lambda c1, n1, n0, c0:
                               res_exp[shape[0] - 1, n1, n0, 0], name="sum2")
            op_list += [sum2]
            instruction_list += ['vector_auto']

        res_sum = tvm.compute(reduce_shape,
                              lambda *i: tvm.sum(sum1(*i), sum2(*i)),
                              name="res_sum")
        op_list += [res_sum]
        instruction_list += ['vector_add']

    # rec
    res_rec = tvm.compute(reduce_shape,
                          lambda *i: 1/(res_sum(*i)), name="res_rec")
    op_list += [res_rec]
    instruction_list += ['vector_rec']

    # mul
    res_mul = tvm.compute(shape,
                          lambda c1, n1, n0, c0:
                          res_exp[c1, n1, n0, c0] * res_rec[0, n1, n0, 0],
                          name="res_mul")
    op_list += [res_mul]
    instruction_list += ['vector_muls']

    # move data from ub to gm
    res = tvm.compute(shape, lambda *i: res_mul(*i), name="result_out")
    op_list += [res]
    instruction_list += ['dma_copy']

    schedule = tvm.create_schedule([res.op])
    return schedule, op_list, instruction_list


def softmax_channel_calculate(shape, dtype, pad_flag, pad_param, kernel_name, impl_mode):
    """
    calculating data's softmax, produces an output tensor with shape
    the result equals the sum of the x power of e over the sum of
    the x power of e

    Parameters
    ----------
    shape : list
        shape of input tensor
    dtype: str
        the data dtype, support float16 and float32
    pad_flag : bool
        the flag using for indicating if there is padding
    kernel_name : str
        cce kernel name, default value is "softmax_cce"
    need_build : bool
        if need to build cce kernel, default value is False
    need_print : bool
        if need to print the ir, default value is False

    Returns
    -------
    None
    """

    # compute & instructions
    tensor_in = tvm.placeholder(shape, name='tensor_in', dtype=dtype)

    use_fp32 = check_isusefp32(shape, dtype)
    if not pad_flag:
        if use_fp32:
            sch, op_list, instruction_list = compute_nopad_fp32(tensor_in, shape)
        else:
            sch, op_list, instruction_list = compute_nopad(tensor_in, shape)
    else:
        if use_fp32:
            sch, op_list, instruction_list = compute_padding_fp32(tensor_in, shape, pad_param, impl_mode)
        else:
            sch, op_list, instruction_list = compute_padding(tensor_in, shape, pad_param)
    res = op_list[-1]

    # schedule
    # storage align
    align_factor = shape[4]
    align(sch, op_list, pad_param, align_factor, 0)

    npart_n, npart_h, npart_w, split_axis_0, split_size = multicore_factor_calculate(shape)

    xno, xni = sch[res].split(res.op.axis[0], nparts=npart_n)
    xho, xhi = sch[res].split(res.op.axis[2], nparts=npart_h)
    xwo, xwi = sch[res].split(res.op.axis[3], nparts=npart_w)

    sch[res].reorder(xno, xho, xwo, xni, xhi, xwi, res.op.axis[1], res.op.axis[4])
    block_axis = sch[res].fuse(xno, xho, xwo)
    sch[res].bind(block_axis, tvm.thread_axis("blockIdx.x"))

    # tiling strategy
    split_flag, split_axis, split_factor = \
        tiling_factor_calculate(shape, split_axis_0, split_size, use_fp32)


    # need splitting on  h or w
    if split_flag:
        if split_axis == 0:
            xo, xi = sch[res].split(xni, factor=split_factor)
        elif split_axis == 2:
            xo, xi = sch[res].split(xhi, factor=split_factor)
        elif split_axis == 3:
            xo, xi = sch[res].split(xwi, factor=split_factor)

        # schedule optimize
        ops_integrate(sch, op_list, xo)

        # buffer mapping
        buffer_mapping(sch, op_list[:-1])

        # double buffer
        double_buf(sch, op_list)

        # instructions replace
        axis_list = emit_axis_collect(op_list, pad_param, instruction_list, xi)
        axis_reorder(sch, op_list, instruction_list)
        instructions_replace(sch, op_list, axis_list, instruction_list)

    # no split
    else:

        # schedule optimize
        if split_axis_0 == 1:
            ops_integrate(sch, op_list, block_axis)
        elif split_axis_0 == 2:
            ops_integrate(sch, op_list, xni)
        elif split_axis_0 == 3 or split_axis_0 == 4:
            ops_integrate(sch, op_list, xhi)

        # buffer mapping
        buffer_mapping(sch, op_list[:-1])

        # double buffer
        double_buf(sch, op_list)

        # instructions replace
        if split_axis_0 == 1:
            axis_list = emit_axis_collect(op_list, pad_param, instruction_list, xni)
        elif split_axis_0 == 2:
            axis_list = emit_axis_collect(op_list, pad_param, instruction_list, xhi)
        elif split_axis_0 == 3 or split_axis_0 == 4:
            axis_list = emit_axis_collect(op_list, pad_param, instruction_list, xwi)

        axis_reorder(sch, op_list, instruction_list)
        instructions_replace(sch, op_list, axis_list, instruction_list)

    with build_config:
        tvm.build(sch, [tensor_in, res], "cce", name=kernel_name)


def softmax_nz_channel_calculate(shape, dtype, pad_flag, pad_param, kernel_name):
    """
    calculating data's softmax nz format, produces an output tensor with shape
    the result equals the sum of the x power of e over the sum of
    the x power of e

    Parameters
    ----------
    shape : list
        shape of input tensor
    dtype: str
        the data dtype, support float16 and float32
    pad_flag : bool
        the flag using for indicating if there is padding
    kernel_name : str
        cce kernel name, default value is "softmax_cce"
    need_build : bool
        if need to build cce kernel, default value is False
    need_print : bool
        if need to print the ir, default value is False

    Returns
    -------
    None
    """

    # compute & instructions
    tensor_in = tvm.placeholder(shape, name='tensor_in', dtype=dtype)
    use_fp32 = check_nz_isusefp32(shape, dtype)
    if not pad_flag:
        if use_fp32:
            sch, op_list, instruction_list = compute_nz_nopad_fp32(tensor_in, shape)
        else:
            sch, op_list, instruction_list = compute_nz_nopad(tensor_in, shape)
    else:
        if use_fp32:
            sch, op_list, instruction_list = compute_nz_padding_fp32(tensor_in, shape, pad_param)
        else:
            sch, op_list, instruction_list = compute_nz_padding(tensor_in, shape, pad_param)
    res = op_list[-1]

    # schedule
    # storage align
    align_factor = shape[3]
    align_nz(sch, op_list, pad_param, align_factor, 0)

    npart_n1, npart_n0, split_axis_0, split_size = multicore_factor_calculate_nz(shape)

    xn1o, xn1i = sch[res].split(res.op.axis[1], nparts=npart_n1)
    xn0o, xn0i = sch[res].split(res.op.axis[2], nparts=npart_n0)


    sch[res].reorder(xn1o, xn0o, xn1i, xn0i, res.op.axis[0], res.op.axis[3])
    block_axis = sch[res].fuse(xn1o, xn0o)
    sch[res].bind(block_axis, tvm.thread_axis("blockIdx.x"))

    # tiling strategy
    split_flag, split_axis, split_factor = \
        tiling_factor_calculate_nz(shape, split_axis_0, split_size, use_fp32)

    # need splitting on  h or w
    if split_flag:
        if split_axis == 1:
            xo, xi = sch[res].split(xn1i, factor=split_factor)
        elif split_axis == 2:
            xo, xi = sch[res].split(xn0i, factor=split_factor)

        # schedule optimize
        ops_integrate(sch, op_list, xo)

        # buffer mapping
        buffer_mapping(sch, op_list[:-1])

        # double buffer
        double_buf(sch, op_list)

        # instructions replace
        axis_list = emit_nz_axis_collect(op_list, pad_param, instruction_list, xi)
        axis_reorder_nz(sch, op_list, instruction_list)
        instructions_replace(sch, op_list, axis_list, instruction_list)

    # no split
    else:

        # schedule optimize
        if split_axis_0 == 1:
            ops_integrate(sch, op_list, block_axis)
        elif split_axis_0 == 2:
            ops_integrate(sch, op_list, xn1i)

        # buffer mapping
        buffer_mapping(sch, op_list[:-1])

        # double buffer
        double_buf(sch, op_list)

        # instructions replace
        if split_axis_0 == 1:
            axis_list = emit_nz_axis_collect(op_list, pad_param, instruction_list, xn1i)
        elif split_axis_0 == 2 or split_axis_0 == 3:
            axis_list = emit_nz_axis_collect(op_list, pad_param, instruction_list, xn0i)

        axis_reorder_nz(sch, op_list, instruction_list)
        instructions_replace(sch, op_list, axis_list, instruction_list)

    with build_config:
        tvm.build(sch, [tensor_in, res], "cce", name=kernel_name)


def softmax_param_check(in_tensor, output_tensor, axis, kernel_name):
    """
    checking the parameter of softmax, and calculating the intermediate
    data using for compute and schedule

    Parameters
    ----------
    in_tensor : dict
        shape and dtype of input tensor, shape and dtype of original tensor,
        input shape only support NC1HWC0, original shape support NCHW and NHWC,
        dtype  support float16 and float,
    output_tensor: dict
        shape and dtype of output tensor, should be same as input
    axis : listint
       the data's axis using for softmax,
    kernel_name : str
        cce kernel name, default value is "softmax_cce"
    need_build : bool
        if need to build cce kernel, default value is False
    need_print : bool
        if need to print the ir, default value is False

    Returns
    -------
    shape and stype of input tensor
    parameters of padding on dimension C0
    """

    #param calculate
    in_shape = in_tensor['shape']
    in_dtype = in_tensor['dtype']
    ori_shape = in_tensor['ori_shape']
    out_dtype = output_tensor['dtype']


    # shape check, check length,min,max,size
    check_shape(in_shape, min_rank=5, max_rank=5, param_name="x")

    if len(ori_shape) == 3:
        ori_shape = list(ori_shape)
        ori_shape.insert(0, 1)
    check_shape(ori_shape, min_rank=4, max_rank=4, param_name="x")

    # shape_matching check
    delta0 = in_shape[0] - ori_shape[0]
    delta2 = in_shape[2] - ori_shape[2]
    delta3 = in_shape[3] - ori_shape[3]

    # type check
    in_dtype = in_dtype.lower()
    check_dtype(in_dtype, ("float16", "float32"), param_name="x")
    out_dtype = out_dtype.lower()
    check_dtype(out_dtype, ("float16", "float32"), param_name="y")

    # shape check
    if in_dtype == "float16" and in_shape[1] * in_shape[4] * 2 > UB_SIZE_LIMIT:
        error_info = {}
        error_info['errCode'] = 'E80011'
        error_info['param_name'] = 'C'
        error_info['op_name'] = 'softmax_v2'
        error_info['max_value'] = UB_SIZE_LIMIT
        error_info['real_value'] = in_shape[1] * in_shape[4] * 2
        raise RuntimeError(error_info, "In op[%s], the shape size(product of all dimensions) of "
                                       "input[%s] should be less than [%s],but actually is [%s]."
                           % (error_info['op_name'], error_info['param_name'], \
                              error_info['max_value'], error_info['real_value']))

    if in_dtype == "float32" and in_shape[1] * in_shape[4] * 4 > UB_SIZE_LIMIT:
        error_info = {}
        error_info['errCode'] = 'E80011'
        error_info['param_name'] = 'C'
        error_info['op_name'] = 'softmax_v2'
        error_info['max_value'] = UB_SIZE_LIMIT
        error_info['real_value'] = in_shape[1] * in_shape[4] * 4
        raise RuntimeError(error_info, "In op[%s], the shape size(product of all dimensions) of "
                                       "input[%s] should be less than [%s],but actually is [%s]."
                           % (error_info['op_name'], error_info['param_name'], \
                              error_info['max_value'], error_info['real_value']))

    # calc padding parameters
    if in_tensor.get("ori_format") == "NCHW":
        padding = in_shape[1] * in_shape[4] - ori_shape[1]
    elif in_tensor.get("ori_format") == "NHWC":
        padding = in_shape[1] * in_shape[4] - ori_shape[3]
    else:
        check_format(in_tensor.get("ori_format"), ("NCHW", "NHWC"), param_name="x")

    pad_param = []
    if padding < 0:
        raise RuntimeError("the shapes of input tensor and original "
                           "tensor don't match")
    elif padding == 0:
        pad_flag = False
        pad_c1 = 0
        pad_c0 = 0
        pad_param = [pad_c1, pad_c0]
    else:
        pad_flag = True
        pad_c1 = (padding + 15) // 16
        pad_c0 = padding % 16
        pad_param = [pad_c1, pad_c0]

    return in_shape, in_dtype, pad_flag, pad_param


def softmax_nz_param_check(in_tensor, output_tensor, axis, kernel_name):
    """
    checking the parameter of softmax nz format, and calculating the
    intermediate data using for compute and schedule
    """

    #param calculate
    in_shape = in_tensor['shape']
    in_dtype = in_tensor['dtype']
    ori_shape = in_tensor['ori_shape']
    out_dtype = output_tensor['dtype']


    # shape check, check length,min,max,size
    check_shape(in_shape, min_rank=4, max_rank=4, param_name="x")

    # type check
    in_dtype = in_dtype.lower()
    check_dtype(in_dtype, ("float16"), param_name="x")
    out_dtype = out_dtype.lower()
    check_dtype(out_dtype, ("float16"), param_name="y")

    if not hasattr(axis, 'index'):
        if axis not in (-1, 1):
            raise RuntimeError("the nz format only support last axis.")
    else:
        if axis[0] not in (-1, 1):
            raise RuntimeError("the nz format only support last axis.")

    # shape check
    if in_dtype == "float16" and in_shape[0] * in_shape[3] * 2 > UB_SIZE_LIMIT:
        error_info = {}
        error_info['errCode'] = 'E80011'
        error_info['param_name'] = 'C'
        error_info['op_name'] = 'softmax_v2'
        error_info['max_value'] = UB_SIZE_LIMIT
        error_info['real_value'] = in_shape[1] * in_shape[4] * 2
        raise RuntimeError(error_info, "In op[%s], the shape size(product of all dimensions) of "
                                       "input[%s] should be less than [%s],but actually is [%s]."
                           % (error_info['op_name'], error_info['param_name'], \
                              error_info['max_value'], error_info['real_value']))

    # calc padding parameters
    padding = in_shape[0] * in_shape[3] - ori_shape[1]
    pad_param = []
    if padding < 0:
        raise RuntimeError("the shapes of input tensor and original "
                           "tensor don't match")
    elif padding == 0:
        pad_flag = False
        pad_c1 = 0
        pad_c0 = 0
        pad_param = [pad_c1, pad_c0]
    else:
        pad_flag = True
        pad_c1 = (padding + 15) // 16
        pad_c0 = padding % 16
        pad_param = [pad_c1, pad_c0]

    return in_shape, in_dtype, pad_flag, pad_param


def softmax_axis_check(origin_format, value):
    """
    checking the axis of softmax
    data using for compute and schedule

    Parameters
    ----------
    axis : listint
       the data's axis using for softmax

    Returns
    -------
    axic_is_c : bool
    if the data's axis is c default value is False
    """

    axic_is_c = False
    if origin_format == "NCHW":
        axic_range = [1, -3]
    elif origin_format == "NHWC":
        axic_range = [3, -1]

    if value in axic_range:
        axic_is_c = True
    return axic_is_c


def update_5hd_axis(origin_format, axis):
    """
    update the axis of 5hd format
    data using for compute and schedule

    Parameters
    ----------
    axis : listint
       the data's axis using for softmax

    Returns
    -------
    axis : listint
    update the axis of 5hd format
    """
    if not hasattr(axis, 'index'):
        if  origin_format == "NCHW" and axis < 0:
            axis = axis - 1
        elif origin_format == "NHWC":
            if axis > 0:
                axis = axis + 1
            if axis == -4:
                axis = -5
    else:
        if  origin_format == "NCHW" and axis[0] < 0:
            axis[0] = axis[0] - 1
        elif origin_format == "NHWC":
            if axis[0] > 0:
                axis[0] = axis[0] + 1
            if axis[0] == -4:
                axis[0] = -5
    return axis


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, (OPTION_ATTR_INT, OPTION_ATTR_LIST_INT), KERNEL_NAME, OPTION_ATTR_STR)
def softmax_v2(input_x, output_y, axis=-1, kernel_name="softmax_v2", impl_mode="high_performance"):
    """
    algorithm: softmax
    calculating data's softmax, produces an output tensor with shape
    the result equals the sum of the x power of e over the sum of
    the x power of e

    Parameters
    ----------
    input_x : dict
        format: FORMAT_ND , NC1HWC0
               dtype: only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    axis : int or list or tuple
        the data's axis.
        format: FORMAT_ND, NC1HWC0
                range == [-d, d-1]
    kernel_name : str
        cce kernel name, default value is softmax_v2
    impl_mode: str.
        high_precision or high_performance for inference, default value is "high_performance".
        no need to add into ops_info file.

    Returns
    -------
    None
    """
    # get input_x format
    input_format = input_x.get("format")
    axic_is_c = False

    if input_format == "NC1HWC0":
        if len(input_x.get("ori_shape")) == 2:
            ori_shape = input_x.get("ori_shape")
            new_ori_shape = [1, ori_shape[0], ori_shape[1], 1]
            input_x["ori_shape"] = new_ori_shape
            if not isinstance(axis, int):
                axis = list(axis)
            if not hasattr(axis, 'index'):
                if axis >= 0:
                    axis = axis + 1
                else:
                    axis = axis - 1
            else:
                if axis[0] >= 0:
                    axis[0] = axis[0] + 1
                else:
                    axis[0] = axis[0] - 1
        if not hasattr(axis, 'index'):
            axic_is_c = softmax_axis_check(input_x.get("ori_format"), axis)
        else:
            axic_is_c = softmax_axis_check(input_x.get("ori_format"), axis[0])
    if input_format == "FRACTAL_NZ" and len(input_x.get("ori_shape")) == 2 \
                                    and input_x['ori_shape'][1] % 16 != 0:
        in_shape, in_dtype, pad_flag, pad_param = \
            softmax_nz_param_check(input_x, output_y, axis, kernel_name)

        # compute & schedule & build
        softmax_nz_channel_calculate(in_shape, in_dtype, pad_flag,
                                     pad_param, kernel_name)
    elif input_format == "NC1HWC0" and axic_is_c:
        # 5D format, using TVM primitive, UB fusion is not supported.
        # parameters check
        in_shape, in_dtype, pad_flag, pad_param = \
            softmax_param_check(input_x, output_y, axis, kernel_name)

        # compute & schedule & build
        softmax_channel_calculate(in_shape, in_dtype, pad_flag,
                                  pad_param, kernel_name, impl_mode)
    else:
        # ND format, using DSL, UB fusion is not supported.
        # compute & schedule & build
        shape = input_x.get("shape")
        dtype = input_x.get("dtype").lower()

        if not isinstance(axis, int):
            axis = list(axis)

        if input_format == "NC1HWC0":
            axis = update_5hd_axis(input_x.get("ori_format"), axis)


        check_shape(shape, param_name="x")
        check_dtype(dtype, ("float16", "float32"), param_name="x")

        if fz.is_frac_z(input_x):
            axis = fz.to_frac_z_axis(input_x.get("ori_shape"), axis)
        axis = util.axis_check(len(shape), axis)

        shape, axis = util.shape_refine(list(shape), axis)
        shape, axis = util.simplify_axis_shape(shape, axis)

        data_input = tvm.placeholder(shape, dtype=dtype, name="data")
        output = softmax_v2_compute(data_input, output_y, axis, kernel_name)
        with tvm.target.cce():
            result = generic.auto_schedule(output)

        tensor_list = [data_input, output]

        config = {"print_ir": False,
                  "name": kernel_name,
                  "tensor_list": tensor_list}

        te.lang.cce.cce_build_code(result, config)
