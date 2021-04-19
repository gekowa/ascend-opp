#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

cast_cce
"""
import te.lang.dynamic
from te import platform as tbe_platform
from te import tvm
from te.platform.cce_build import build_config
from functools import reduce as reduceIns
from topi import generic
from te.platform.shape_classifier import classify
from te.platform.shape_classifier import Mode
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import REQUIRED_ATTR_INT
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_op_params
from te.utils.op_utils import variable_shape
from te.utils.op_utils import broadcast_shapes
from te.utils.error_manager import error_manager_vector

MAX_SUPPORT_SHAPE = 1 << 30  # Limit of all dims' product
SPECIAL_SHAPE_NUM = 10000000  # Limit of one dim


def _new_alloc(ir_builder, dtype, shape, name, scope):
    """
    alloc memory for decl new buffer
    """
    buf_var = ir_builder.allocate(dtype, shape, name=name, scope=scope)
    new_buffer = tvm.decl_buffer(shape,
                                 buf_var.dtype,
                                 name=name,
                                 scope=scope,
                                 data=buf_var)
    return new_buffer


def _kernel_ir(dst, src, dst_type, src_type):
    """
    convert a scale from src type to dst type
    NOTICE: SCALE ONLY
    """
    ir_builder = tvm.ir_builder.create()
    in_tensor = src[0]
    a_ub = _new_alloc(ir_builder,
                      src_type,
                      in_tensor.shape,
                      "a_ub",
                      scope=tbe_platform.scope_ubuf)
    out_tensor = dst[0]
    b_ub = _new_alloc(ir_builder,
                      dst_type,
                      in_tensor.shape,
                      "b_ub",
                      scope=tbe_platform.scope_ubuf)

    reg = ir_builder.allocate(dst_type, (1,), name='reg',
                              scope=tbe_platform.scope_reg)
    ir_builder.emit(
        tvm.call_extern(src_type, "copy_gm_to_ubuf", a_ub.access_ptr("w"),
                        in_tensor.access_ptr("r"), 0, 1, 1, 0, 0))
    ir_builder.emit(
        tvm.call_extern(src_type, "reg_mov",
                        tvm.call_extern(dst_type, "reg", reg[0]),
                        a_ub.access_ptr('r', offset=0)))
    ir_builder.emit(
        tvm.call_extern(dst_type, "reg_mov", b_ub.access_ptr('w', offset=0),
                        tvm.call_extern(dst_type, "reg", reg[0])))
    ir_builder.emit(
        tvm.call_extern(dst_type,
                        "copy_ubuf_to_gm", out_tensor.access_ptr('w'),
                        b_ub.access_ptr("r"), 0, 1, 1, 0, 0))

    return ir_builder.get()


def _int8_uint8_process(data, dst_type):
    """
    deal with src dtype=int8 and uint8 case
    """
    check_list_value = ("uint8", "int32", "float16", "float32")
    check_dtype(dst_type, check_list_value, param_name="from_int8_uint8_to_dsttype")

    if dst_type == "float16":
        return te.lang.dynamic.cast_to(data, "float16")

    if dst_type == "float32":
        data_fp16 = te.lang.dynamic.cast_to(data, "float16")
        return te.lang.dynamic.cast_to(data_fp16, "float32")

    if dst_type == "int32":
        data_fp16 = te.lang.dynamic.cast_to(data, "float16")
        return te.lang.dynamic.cast_to(data_fp16, "int32")

    if dst_type == "uint8":
        data_fp16 = te.lang.dynamic.cast_to(data, "float16")
        abs_fp16 = te.lang.dynamic.vabs(data_fp16)
        return te.lang.dynamic.cast_to(abs_fp16, "uint8")


def _int32_process(data, dst_type):
    """
    deal with src dtype=int32 case
    """
    check_list_value = ("bool", "int8", "uint8", "float16", "float32")
    check_dtype(dst_type, check_list_value, param_name="from_int32_to_dsttype")
    if dst_type == "bool":
        const_one = tvm.const(1.0, "float16")
        shape_data = te.lang.dynamic.shape_to_list(data.shape)
        const_broad = te.lang.dynamic.broadcast(const_one, shape_data)

        data = te.lang.dynamic.cast_to(data, "float16", True)
        x_abs = te.lang.dynamic.vabs(data)
        x_min = te.lang.dynamic.vmin(x_abs, const_broad)
        y_abs = te.lang.dynamic.vabs(x_min)
        return te.lang.dynamic.cast_to(y_abs, "int8", True)

    if dst_type == "int8":
        data_fp16 = te.lang.dynamic.cast_to(data, "float16")
        tensor_0 = te.lang.dynamic.vmuls(data_fp16, 0)
        tensor_256 = te.lang.dynamic.vadds(tensor_0, 256)
        result = te.lang.dynamic.vadds(data_fp16, 128)
        result = te.lang.dynamic.vmod(result, tensor_256)
        result = te.lang.dynamic.vadds(result, -128)
        result = te.lang.dynamic.cast_to(result, "float16")
        return te.lang.dynamic.cast_to(result, "int8", True)

    if dst_type == "uint8":
        data_fp16 = te.lang.dynamic.cast_to(data, "float16")
        tensor_0 = te.lang.dynamic.vmuls(data_fp16, 0)
        tensor_256 = te.lang.dynamic.vadds(tensor_0, 256)
        result = te.lang.dynamic.vmod(data_fp16, tensor_256)
        result = te.lang.dynamic.cast_to(result, "float16")
        return te.lang.dynamic.cast_to(result, "uint8", True)

    if dst_type == "float32":
        return te.lang.dynamic.cast_to(data, "float32")

    if dst_type == "float16":
        return te.lang.dynamic.cast_to(data, "float16")


def _float32_process(data, dst_type):
    """
    deal with src dtype=float32 case
    """
    check_list_value = ("int32", "float16")
    check_dtype(dst_type, check_list_value, param_name="from_fp32_to_dsttype")
    if dst_type == "int32":
        return te.lang.dynamic.cast_to(data, "int32")
    if dst_type == "float16":
        return te.lang.dynamic.cast_to(data, "float16")


def _float16_process(data, dst_type):
    """
    deal with src dtype=float16 case
    """
    check_list_value = ("uint8", "int32", "float32")
    check_dtype(dst_type, check_list_value, param_name="from_fp16_to_dsttype")
    if dst_type == "float32":
        return te.lang.dynamic.cast_to(data, "float32")

    if dst_type == "int32":
        return te.lang.dynamic.cast_to(data, "int32")

    if dst_type == "uint8":
        if not tbe_platform.cce_conf.api_check_support("te.lang.dynamic.cast_to", "s322f16") and \
            tbe_platform.cce_conf.api_check_support("te.lang.dynamic.vmod", "float16"):
            return te.lang.dynamic.cast_to(data, "uint8", True)
        data_int32 = te.lang.dynamic.cast_to(data, "int32")
        data_fp16 = te.lang.dynamic.cast_to(data_int32, "float16")
        tensor_0 = te.lang.dynamic.vmuls(data_fp16, 0)
        tensor_256 = te.lang.dynamic.vadds(tensor_0, 256)
        result = te.lang.dynamic.vmod(data_fp16, tensor_256)
        result = te.lang.dynamic.cast_to(result, "float16")
        return te.lang.dynamic.cast_to(result, "uint8", True)


def _cast_dsttype_conversion(dst_type):
    if dst_type == 0:
        dst_type = "float32"
    if dst_type == 1:
        dst_type = "float16"
    if dst_type == 2:
        dst_type = "int8"
    if dst_type == 3:
        dst_type = "int32"
    if dst_type == 4:
        dst_type = "uint8"
    if dst_type == 10:
        dst_type = "uint64"
    if dst_type == 12:
        dst_type = "bool"
    return dst_type


# pylint: disable=unused-argument
def check_supported(input_x, output_y, dst_type, kernel_name="cast"):
    """
    verify the types of cast supported by tbe
    """
    src_type = input_x.get("dtype").lower()
    check_result = False
    if src_type == "bool":
        src_type = "int8"

    dst_type = _cast_dsttype_conversion(dst_type)

    check_list = []
    if src_type == "float16":
        check_list = ["float32", "int32", "uint8"]
    elif src_type == "float32":
        check_list = ["float16", "int32"]
    elif src_type == "int8":
        check_list = ["float32", "float16", "int32", "uint8"]
    elif src_type == "uint8":
        check_list = ["float32", "float16", "int32"]
    elif src_type == "int32":
        check_list = ["bool", "uint8", "int8", "float32", "float16"]

    src_shape = input_x.get("shape")
    shape_size = reduceIns(lambda x, y: x * y, src_shape)
    if shape_size == 1 and src_type == "int64":
        check_list = ["int32", "float32"]

    if dst_type in check_list:
        check_result = True

    return check_result


# pylint: disable=locally-disabled,too-many-arguments,unused-argument

def cast_compute(data, output_y, dst_type, kernel_name="cast"):
    """
    core func of tensor casting. cast a tensor form src data type to dst data
    type. restrictions of input algorithms are as follow
    only types' groups blow are support tensor process:
        float16->float32
        float16->int32
        float32->float16
        float32->int32
        int8->float32
        uint8->float32
        int8->float16
        uint8->float16
        int8->int32
        uint8->int32
        int32->uint8 // number out of [0,255] can get unexpected result
        int32->int8 // number out of [-128,127] can get unexpected result
        int32->float32 // For tans with fp16, only guarantees
                        number in [-1023,1023] get correct result
        int32->float16 // only guarantees
                        number in [-1023,1023] get correct result
    Parameters
    ----------
    placeholders: list.
        the input tensor
    src_type: str
        the input data type.
    dst_type: str
        the output data type.

    Returns
    -------
        the compute result tensor with type dst_type
    """
    src_data_type = data.dtype
    check_dtype(src_data_type,
                ("float16", "float32", "int8", "uint8", "int32"),
                param_name="input_x")

    if src_data_type in ("int8", "uint8"):
        return _int8_uint8_process(data, dst_type)

    if src_data_type == "float32":
        return _float32_process(data, dst_type)

    if src_data_type == "float16":
        return _float16_process(data, dst_type)

    if src_data_type == "int32":
        return _int32_process(data, dst_type)


@te.op.register_operator("Cast")
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT,
                 REQUIRED_ATTR_INT, KERNEL_NAME)
def cast(input_x, output_y, dst_type, kernel_name="cast"):
    """
    cast a tensor/scaler with input shape form src data type to dst data
    type. restrictions of input algorithms are as follow
    only types' groups blow are support tensor process:
        float16->float32
        float16->int32
        float32->float16
        float32->int32
        int8->float32
        uint8->float32
        int8->float16
        uint8->float16
        int8->int32
        uint8->int32
        int32->uint8 // number out of [0,255] can get unexpected result
        int32->int8 // number out of [-128,127] can get unexpected result
        int32->float32 // For tans with fp16, only guarantees
                        number in [-1023,1023] get correct result
        int32->float16 // only guarantees
                        number in [-1023,1023] get correct result
    scale convert support:(means only support shape [1,])
        int64->int32
        int64->float32

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape as input,
        and the dtype is the dst dtype need to cast
    kernel_name : str
        cce kernel name, default value is cast

    Returns
    -------
    None
    """

    src_type = input_x.get("dtype").lower()

    if src_type == "bool":
        src_type = "int8"

    schedules, tensors = [], []
    ins = classify([input_x], Mode.ELEWISE)
    for (input_x,) in ins:
        with te.op.compute():
            x_shape = variable_shape([input_x])
            dst_type = _cast_dsttype_conversion(dst_type)
            fuseshape = [1]
            fuseshape[0] = reduceIns(lambda x, y: x * y, x_shape[0])
            data = tvm.placeholder(fuseshape, name="data", dtype=src_type)
            if src_type == "int64":
                check_dtype(dst_type, ("float32", "int32"),
                            param_name="dst_type")
                res = tvm.extern(
                    [fuseshape], [data],
                    lambda ins, outs: _kernel_ir(outs, ins, dst_type, "int64"),
                    name="res",
                    dtype=dst_type)
                tensor_list = [data, res]
                schedule = tvm.create_schedule(res.op)
                with build_config:
                    tvm.build(schedule, tensor_list, "cce", name=kernel_name)
            else:
                res = cast_compute(data, output_y, dst_type, kernel_name)
                tensors.append([data, res])
        if src_type != "int64":
            with tvm.target.cce():
                sch = generic.auto_schedule(res)
            schedules.append(sch)

    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensors
    }
    te.lang.dynamic.build(sch, config)
