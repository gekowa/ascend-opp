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
aipp_resize_padding
"""
# pylint: disable=invalid-name,too-many-locals,too-many-branches,too-many-statements
# pylint: disable=too-many-arguments,too-many-boolean-expressions,import-error
import math
from te import tvm
from te import platform as cce
from te import platform as tbe_platform
from te.platform.cce_util import get_const
from impl import aipp_comm

NoneType = type(None)


def process_padding(ib, input_data, output_buf, output_format="NC1HWC0"):
    """
    :param ib:
    :param input_data:
    :param output_buf:
    :return:
    """

    dtype = input_data[0]
    W = input_data[1]
    C0 = input_data[2]
    size = input_data[3]
    padding_size = input_data[4]
    offset = input_data[5]

    ub_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
    if output_format == "NC1HWC0_C04":
        C0 = 4
        ub_size = ub_size - 64
    buffer_upper_limit = ub_size//2//2//C0

    if buffer_upper_limit >= padding_size*W:
        with ib.new_scope():
            num = (((padding_size*W*C0*size + 31) // 32)*32) // size
            padding_ub = ib.allocate(dtype, (num,), "padding_ub",
                                     scope=cce.scope_ubuf)
            padding_ub_buf = tvm.decl_buffer((num,), dtype,
                                             "padding_ub", scope=cce.scope_ubuf,
                                             data=padding_ub)
            if dtype == "float16":
                aipp_comm.vector_dup(ib, "float16", padding_ub_buf, num)
            else:
                with ib.new_scope():
                    tmp_padding_ub = ib.allocate("float16",
                                                 (num,),
                                                 "tmp_padding_ub",
                                                 scope=cce.scope_ubuf)
                    tmp_padding_ub_buf = tvm.decl_buffer((num,),
                                                         "float16",
                                                         "tmp_padding_ub_buf",
                                                         scope=cce.scope_ubuf,
                                                         data=tmp_padding_ub)

                    aipp_comm.vector_dup(ib, "float16", tmp_padding_ub_buf, num)
                    aipp_comm.conv(ib, dtype, "float16", padding_ub_buf,
                                   tmp_padding_ub_buf, num)

            if output_format == "NC1HWC0_C04" and \
                    padding_size*W*C0*size % 32 != 0 and \
                    padding_size*W*C0*size > 32:

                tail_ub = ib.allocate(dtype, (32 // size,), "tail_ub", scope=cce.scope_ubuf)
                tail_ub_buf = tvm.decl_buffer((32 // size,), dtype, "tail_ub_buf",
                                              scope=cce.scope_ubuf, data=tail_ub)
                aipp_comm.copy_ubuf_to_gm_tail(
                    ib, dtype, output_buf, padding_ub_buf,
                    tail_ub_buf,
                    padding_size*W*C0, offset,
                    0)

            ib.emit(tvm.call_extern(
                dtype, 'copy_ubuf_to_gm',
                output_buf.access_ptr("w", ptr_type=dtype, offset=offset),
                padding_ub_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                0, 1, padding_size*W*C0*size//32, 0, 0))
    else:
        tiling_w, w_loop = aipp_comm.get_tiling_W(W, buffer_upper_limit, 1)
        tiling_h = buffer_upper_limit // tiling_w
        tail_w = W % tiling_w
        tail_h = padding_size % tiling_h
        if output_format == "NC1HWC0_C04" and tail_w > 0:
            if dtype == "float16":
                if tail_w < 4:
                    if 4 // w_loop > 0 and 4 % w_loop == 0:
                        tiling_w = tiling_w - (4 // w_loop)
                    elif 4 // w_loop > 0 and 4 % w_loop != 0:
                        tiling_w = tiling_w - (4 // w_loop) - 1
                    else:
                        tiling_w = tiling_w - 1
                    tiling_h = buffer_upper_limit // tiling_w
                    tail_w = W - w_loop * tiling_w
                    tail_h = padding_size % tiling_h
            else:
                if tail_w < 8:
                    if 8 // w_loop > 0 and 8 % w_loop == 0:
                        tiling_w = tiling_w - (8 // w_loop)
                    elif 8 // w_loop > 0 and 8 % w_loop != 0:
                        tiling_w = tiling_w - (8 // w_loop) - 1
                    else:
                        tiling_w = tiling_w - 1
                    tiling_h = buffer_upper_limit // tiling_w
                    tail_w = W - w_loop * tiling_w
                    tail_h = padding_size % tiling_h

        h_loop = padding_size // tiling_h
        zero_const = tvm.const(0, dtype="uint64")
        h_loop_const = tvm.const(h_loop, dtype="uint64")
        w_loop_const = tvm.const(w_loop, dtype="uint64")

        with ib.for_range(zero_const, h_loop_const, name="h1",
                          dtype="uint64") as h1:
            with ib.for_range(zero_const, w_loop_const,
                              name="w1", dtype="uint64") as w1:
                with ib.new_scope():
                    num = (((tiling_h*tiling_w*C0*size+31)//32)*32) // size
                    padding_ub = ib.allocate(dtype, (num,), "padding_ub",
                                             scope=cce.scope_ubuf)
                    padding_ub_buf = tvm.decl_buffer((num,), dtype,
                                                     "padding_ub",
                                                     scope=cce.scope_ubuf,
                                                     data=padding_ub)

                    if dtype == "float16":
                        aipp_comm.vector_dup(ib, "float16", padding_ub_buf, num)
                    else:
                        with ib.new_scope():
                            tmp_padding_ub = ib.allocate("float16",
                                                         (num,),
                                                         "tmp_padding_ub",
                                                         scope=cce.scope_ubuf)
                            tmp_padding_ub_buf = tvm.decl_buffer((num,),
                                                                 "float16",
                                                                 "tmp_padding_ub_buf",
                                                                 scope=cce.scope_ubuf,
                                                                 data=tmp_padding_ub)

                            aipp_comm.vector_dup(ib, "float16", tmp_padding_ub_buf, num)
                            aipp_comm.conv(ib, dtype, "float16", padding_ub_buf,
                                           tmp_padding_ub_buf, num)

                    if output_format == "NC1HWC0_C04" and \
                            tiling_h*tiling_w*C0*size % 32 != 0 and \
                            tiling_h*tiling_w*C0*size > 32:
                        tail_ub = ib.allocate(dtype, (32 // size,), "tail_ub", scope=cce.scope_ubuf)
                        tail_ub_buf = tvm.decl_buffer((32 // size,), dtype, "tail_ub_buf",
                                                      scope=cce.scope_ubuf, data=tail_ub)
                        aipp_comm.copy_ubuf_to_gm_tail(
                            ib, dtype, output_buf, padding_ub_buf,
                            tail_ub_buf,
                            tiling_h*tiling_w*C0,
                            offset + h1*tiling_h*W*C0 + w1*tiling_w*tiling_h*C0,
                            0)

                    ib.emit(tvm.call_extern(
                        dtype, 'copy_ubuf_to_gm',
                        output_buf.access_ptr("w", ptr_type=dtype,
                                              offset=offset + h1*tiling_h*W*C0 + \
                                                     w1*tiling_w*tiling_h*C0),
                        padding_ub_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                        0, 1, tiling_h*tiling_w*C0*size//32, 0, 0))

                if tail_w != 0:
                    with ib.new_scope():
                        num = (((tiling_h*tail_w*C0*size+31)//32)*32) // size
                        padding_ub = ib.allocate(dtype, (num,), "padding_ub",
                                                 scope=cce.scope_ubuf)
                        padding_ub_buf = tvm.decl_buffer((num,), dtype,
                                                         "padding_ub",
                                                         scope=cce.scope_ubuf,
                                                         data=padding_ub)

                        if dtype == "float16":
                            aipp_comm.vector_dup(ib, "float16", padding_ub_buf, num)
                        else:
                            with ib.new_scope():
                                tmp_padding_ub = ib.allocate("float16",
                                                             (num,),
                                                             "tmp_padding_ub",
                                                             scope=cce.scope_ubuf)
                                tmp_padding_ub_buf = tvm.decl_buffer((num,),
                                                                     "float16",
                                                                     "tmp_padding_ub_buf",
                                                                     scope=cce.scope_ubuf,
                                                                     data=tmp_padding_ub)

                                aipp_comm.vector_dup(ib, "float16", tmp_padding_ub_buf, num)
                                aipp_comm.conv(ib, dtype, "float16", padding_ub_buf,
                                               tmp_padding_ub_buf, num)

                        if output_format == "NC1HWC0_C04" and \
                                tiling_h*tail_w*C0*size % 32 != 0 and \
                                tiling_h*tail_w*C0*size > 32:

                            tail_ub = ib.allocate(dtype, (32 // size,), "tail_ub", scope=cce.scope_ubuf)
                            tail_ub_buf = tvm.decl_buffer((32 // size,), dtype, "tail_ub_buf",
                                                          scope=cce.scope_ubuf, data=tail_ub)
                            aipp_comm.copy_ubuf_to_gm_tail(
                                ib, dtype, output_buf, padding_ub_buf,
                                tail_ub_buf,
                                tiling_h*tail_w*C0,
                                offset + h1*tiling_h*tiling_w*C0 + w_loop*tiling_w*tiling_h*C0,
                                0)

                        ib.emit(tvm.call_extern(
                            dtype, 'copy_ubuf_to_gm',
                            output_buf.access_ptr("w", ptr_type=dtype,
                                                  offset=offset +
                                                  h1*tiling_h*tiling_w*C0 +
                                                  w_loop*tiling_w*tiling_h*C0),
                            padding_ub_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                            0, 1, tiling_h*tail_w*C0*size//32, 0, 0))

        if tail_h != 0:
            with ib.for_range(zero_const,
                              w_loop_const,
                              name="w1", dtype="uint64") as w1:
                with ib.new_scope():
                    num = (((tail_h*tiling_w*C0*size+31)//32)*32) // size
                    padding_ub = ib.allocate(dtype, (num,),
                                             "padding_ub", scope=cce.scope_ubuf)
                    padding_ub_buf = tvm.decl_buffer((num,), dtype,
                                                     "padding_ub",
                                                     scope=cce.scope_ubuf,
                                                     data=padding_ub)

                    if dtype == "float16":
                        aipp_comm.vector_dup(ib, "float16", padding_ub_buf, num)
                    else:
                        with ib.new_scope():
                            tmp_padding_ub = ib.allocate("float16", (num,),
                                                         "tmp_padding_ub",
                                                         scope=cce.scope_ubuf)
                            tmp_padding_ub_buf = tvm.decl_buffer((num,),
                                                                 "float16",
                                                                 "tmp_padding_ub_buf",
                                                                 scope=cce.scope_ubuf,
                                                                 data=tmp_padding_ub)

                            aipp_comm.vector_dup(ib, "float16", tmp_padding_ub_buf, num)
                            aipp_comm.conv(ib, dtype, "float16", padding_ub_buf,
                                           tmp_padding_ub_buf, num)

                    if output_format == "NC1HWC0_C04" and \
                            tail_h*tiling_w*C0*size % 32 != 0 and \
                            tail_h*tiling_w*C0*size > 32:

                        tail_ub = ib.allocate(dtype, (32 // size,), "tail_ub", scope=cce.scope_ubuf)
                        tail_ub_buf = tvm.decl_buffer((32 // size,), dtype, "tail_ub_buf",
                                                      scope=cce.scope_ubuf, data=tail_ub)
                        aipp_comm.copy_ubuf_to_gm_tail(
                            ib, dtype, output_buf, padding_ub_buf,
                            tail_ub_buf,
                            tail_h*tiling_w*C0,
                            offset + h_loop*tiling_h*W*C0 + w1*tiling_w*tail_h*C0,
                            0)

                    ib.emit(tvm.call_extern(
                        dtype, 'copy_ubuf_to_gm',
                        output_buf.access_ptr("w", ptr_type=dtype,
                                              offset=offset + h_loop*tiling_h*W*C0 + \
                                                     w1*tiling_w*tail_h*C0),
                        padding_ub_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                        0, 1, tail_h*tiling_w*C0*size//32, 0, 0))
            if tail_w != 0:
                with ib.new_scope():
                    num = (((tail_h*tail_w*C0*size+31)//32)*32) // size
                    padding_ub = ib.allocate(dtype, (num,), "padding_ub",
                                             scope=cce.scope_ubuf)
                    padding_ub_buf = tvm.decl_buffer((num,), dtype,
                                                     "padding_ub",
                                                     scope=cce.scope_ubuf,
                                                     data=padding_ub)

                    if dtype == "float16":
                        aipp_comm.vector_dup(ib, "float16", padding_ub_buf, num)
                    else:
                        with ib.new_scope():
                            tmp_padding_ub = ib.allocate("float16",
                                                         (((tail_h*tail_w*C0+31)//32)*32,),
                                                         "tmp_padding_ub",
                                                         scope=cce.scope_ubuf)
                            tmp_padding_ub_buf = tvm.decl_buffer((((tail_h*tail_w*C0+31)//32)*32,),
                                                                 "float16",
                                                                 "tmp_padding_ub_buf",
                                                                 scope=cce.scope_ubuf,
                                                                 data=tmp_padding_ub)

                            aipp_comm.vector_dup(ib, "float16", tmp_padding_ub_buf, num)
                            aipp_comm.conv(ib, dtype, "float16", padding_ub_buf,
                                           tmp_padding_ub_buf, tail_h*tail_w*C0)

                    if output_format == "NC1HWC0_C04" and \
                            tail_h*tail_w*C0*size % 32 != 0 and \
                            tail_h*tail_w*C0*size > 32:
                        with ib.new_scope():
                            tail_ub = ib.allocate(dtype, (32 // size,), "tail_ub", scope=cce.scope_ubuf)
                            tail_ub_buf = tvm.decl_buffer((32 // size,), dtype, "tail_ub_buf",
                                                          scope=cce.scope_ubuf, data=tail_ub)
                            aipp_comm.copy_ubuf_to_gm_tail(
                                ib, dtype, output_buf, padding_ub_buf,
                                tail_ub_buf,
                                tail_h*tail_w*C0,
                                offset + h_loop*tiling_h*W*C0 + w_loop*tiling_w*tail_h*C0,
                                0)

                    ib.emit(tvm.call_extern(
                        dtype, 'copy_ubuf_to_gm',
                        output_buf.access_ptr("w", ptr_type=dtype,
                                              offset=offset + h_loop*tiling_h*W*C0 + \
                                                     w_loop*tiling_w*tail_h*C0),
                        padding_ub_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                        0, 1, tail_h*tail_w*C0*size//32, 0, 0))


def move_data_from_l1_to_gm(ib, totol_num, dtype, output_cb_buf, output_buf, gm_output_offset,
                            output_format="NC1HWC0"):
    """
    :param ib:
    :param totol_num:
    :param dtype:
    :param output_cb_buf:
    :param output_buf:
    :param gm_output_offset:
    :return:
    """

    ub_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)

    if output_format == "NC1HWC0_C04":
        ub_size = ub_size - 64

    if dtype == "float16":
        size = 2
    else:
        size = 1

    if totol_num*size < (ub_size//2):
        with ib.new_scope():
            num = (((totol_num*size+31)//32)*32) // size
            output_ub = ib.allocate(dtype, (num,),
                                    "output_ub",
                                    scope=cce.scope_ubuf)
            output_ub_buf = tvm.decl_buffer(
                (num,), dtype, "output_ub_buf",
                scope=cce.scope_ubuf, data=output_ub)

            lenBurst, nBurst = \
                aipp_comm.get_lenBurst_and_nBurst(
                    (totol_num*size + 31)//32, 1)

            ib.emit(tvm.call_extern(
                dtype, 'copy_cbuf_to_ubuf',
                output_ub_buf.access_ptr("w",
                                         ptr_type=dtype,
                                         offset=0),
                output_cb_buf.access_ptr("rw",
                                         ptr_type=dtype,
                                         offset=0),
                0, nBurst, lenBurst, 0, 0))

            if output_format == "NC1HWC0_C04" and \
                    totol_num*size % 32 != 0 and \
                    totol_num*size > 32:
                lenBurst, nBurst = \
                    aipp_comm.get_lenBurst_and_nBurst(
                        (totol_num*size)//32, 1)

                tail_ub = ib.allocate(dtype, (32 // size,), "tail_ub", scope=cce.scope_ubuf)
                tail_ub_buf = tvm.decl_buffer((32 // size,), dtype, "tail_ub_buf",
                                              scope=cce.scope_ubuf, data=tail_ub)
                aipp_comm.copy_ubuf_to_gm_tail(
                    ib, dtype, output_buf, output_ub_buf,
                    tail_ub_buf,
                    totol_num, gm_output_offset,
                    0)

            ib.emit(tvm.call_extern(
                dtype, 'copy_ubuf_to_gm',
                output_buf.access_ptr(
                    "w", ptr_type=dtype,
                    offset=gm_output_offset),
                output_ub_buf.access_ptr("rw", ptr_type=dtype,
                                         offset=0),
                0, nBurst, lenBurst, 0, 0))
    else:
        cycle_num = (ub_size//2)//size
        move_num = totol_num // cycle_num
        move_tail = totol_num - move_num * cycle_num
        if output_format == "NC1HWC0_C04" and move_tail*size < 32:
            cycle_num = (cycle_num*size - 32) // size
            move_num = totol_num // cycle_num
            move_tail = totol_num - move_num*cycle_num

        with ib.for_range(tvm.const(0, dtype="uint64"),
                          tvm.const(move_num,
                                    dtype="uint64"),
                          name="move_index",
                          dtype="uint64") as move_index:
            with ib.new_scope():
                output_ub = ib.allocate(dtype, (ub_size//2,),
                                        "output_ub",
                                        scope=cce.scope_ubuf)
                output_ub_buf = tvm.decl_buffer(
                    (ub_size // 2,), dtype, "output_ub_buf",
                    scope=cce.scope_ubuf, data=output_ub)

                lenBurst, nBurst = \
                    aipp_comm.get_lenBurst_and_nBurst(
                        (cycle_num*size) // 32, 1)

                ib.emit(tvm.call_extern(
                    dtype, 'copy_cbuf_to_ubuf',
                    output_ub_buf.access_ptr("w", ptr_type=dtype,
                                             offset=0),
                    output_cb_buf.access_ptr("rw", ptr_type=dtype,
                                             offset=move_index*cycle_num),
                    0, nBurst, lenBurst, 0, 0))

                ib.emit(tvm.call_extern(
                    dtype, 'copy_ubuf_to_gm',
                    output_buf.access_ptr("w", ptr_type=dtype,
                                          offset=gm_output_offset +
                                          move_index*cycle_num),
                    output_ub_buf.access_ptr("rw", ptr_type=dtype,
                                             offset=0),
                    0, nBurst, lenBurst, 0, 0))
        if move_tail != 0:
            with ib.new_scope():
                output_ub = ib.allocate(dtype, (ub_size // 2,),
                                        "output_ub",
                                        scope=cce.scope_ubuf)
                output_ub_buf = tvm.decl_buffer(
                    (ub_size // 2,), dtype, "output_ub_buf",
                    scope=cce.scope_ubuf, data=output_ub)

                lenBurst, nBurst = \
                    aipp_comm.get_lenBurst_and_nBurst(
                        (move_tail*size + 31)//32, 1)

                ib.emit(tvm.call_extern(
                    dtype, 'copy_cbuf_to_ubuf',
                    output_ub_buf.access_ptr("w", ptr_type=dtype,
                                             offset=0),
                    output_cb_buf.access_ptr("rw", ptr_type=dtype,
                                             offset=move_num*cycle_num),
                    0, nBurst, lenBurst, 0, 0))

                if output_format == "NC1HWC0_C04" and \
                        (move_tail*size) % 32 != 0 and \
                        move_tail*size > 32:
                    lenBurst, nBurst = \
                        aipp_comm.get_lenBurst_and_nBurst(
                            (move_tail*size)//32, 1)

                    tail_ub = ib.allocate(dtype, (32 // size,), "tail_ub", scope=cce.scope_ubuf)
                    tail_ub_buf = tvm.decl_buffer((32 // size,), dtype, "tail_ub_buf",
                                                  scope=cce.scope_ubuf, data=tail_ub)
                    aipp_comm.copy_ubuf_to_gm_tail(
                        ib, dtype, output_buf, output_ub_buf,
                        tail_ub_buf,
                        move_tail,
                        gm_output_offset + move_num*cycle_num,
                        0)

                ib.emit(tvm.call_extern(
                    dtype, 'copy_ubuf_to_gm',
                    output_buf.access_ptr("w", ptr_type=dtype,
                                          offset=gm_output_offset +
                                          move_num*cycle_num),
                    output_ub_buf.access_ptr("rw", ptr_type=dtype,
                                             offset=0),
                    0, nBurst, lenBurst, 0, 0))


def aipp_compute(input_tensor, input_shape, input_format,
                 output_data, aipp_config):
    """
    :param input_tensor:
    :param param_tensor:
    :param input_shape:
    :param input_format:
    :param output_data:
    :param aipp_config:
    :return:
    """

    if input_format == "NHWC":
        N, H, W, C = input_shape
    else:
        N, C, H, W = input_shape

    output_shape = output_data.get('shape')
    N, C1, H, W, C0 = output_shape

    src_image_size_h = aipp_config.get("src_image_size_h")
    src_image_size_w = aipp_config.get("src_image_size_w")
    load_image_h = src_image_size_h
    load_image_w = src_image_size_w
    load_start_pos_h = 0
    load_start_pos_w = 0

    output_format = output_data.get('format')

    if "crop" in aipp_config and aipp_config.get("crop") == 1:
        src_image_size_h, src_image_size_w, load_start_pos_h, load_start_pos_w, \
        load_image_h, load_image_w = aipp_comm.get_crop_info(aipp_config)

    dtype = output_data.get('dtype')
    if dtype == "float16":
        size = 2
        C0 = 16
    else:
        size = 1
        C0 = 32

    if output_format == "NC1HWC0_C04":
        C0 = 4

    C1 = (C + C0 - 1) // C0

    actual_col_size = H * W
    if "crop" in aipp_config and aipp_config.get("crop") == 1 or \
       "resize" in aipp_config and aipp_config.get("resize") == 1 or \
       "padding" in aipp_config and aipp_config.get("padding") == 1:
        actual_col_size = aipp_comm.get_actual_col_size(aipp_config, H, W)

    l1_image_buf_max = \
        aipp_comm.get_l1_image_buf_max(actual_col_size, dtype, False, output_format)

    def aipp_ir(input_buf, output_buf):
        ib = tvm.ir_builder.create()

        cur_cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")

        if cur_cce_product not in ["Ascend310", "Ascend610", "Ascend710",
                                   "Ascend615", "Hi3796CV300ES", "Hi3796CV300CS"]:
            cause_dec = "Only support is Ascend310, Ascend610, Ascend710, Ascend615 " \
                        "Hi3796CV300ES and Hi3796CV300CS, " \
                        "cur_cce_product is %s" % cur_cce_product
            aipp_comm.raise_RuntimeError(cause_dec)

        device_core_num = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        batch_num = N
        batch_factor = 1

        if batch_num % device_core_num == 0:
            batch_factor = batch_num // device_core_num

        block_index = tvm.thread_axis("blockIdx.x")
        ib.scope_attr(block_index, "thread_extent", batch_num // batch_factor)

        if aipp_config.get('input_format') == "YUV420SP_U8":
            input_offset = batch_factor*((C*src_image_size_h*src_image_size_w)//2)
        elif aipp_config.get('input_format') in \
                ["XRGB8888_U8", "RGB888_U8", "ARGB8888_U8",
                 "AYUV444_U8", "YUV400_U8", "RGB16"]:
            input_offset = batch_factor*((C*src_image_size_h*src_image_size_w))
        elif aipp_config.get('input_format') in ["YUYV_U8", "YUV422SP_U8"]:
            input_offset = batch_factor*((2*src_image_size_h*src_image_size_w))

        offset = batch_factor*C1*H*W*C0

        def _aipp_intrin():
            # config SPR2~SPR9
            aipp_comm.set_spr2_spr9(ib, aipp_config, dtype, cur_cce_product, output_format)

            aippXt = (src_image_size_w - 1)

            left_padding_size, right_padding_size, \
            top_padding_size, bottom_padding_size =\
                aipp_comm.get_padding_size(aipp_config)

            with ib.for_range(tvm.const(0, dtype="uint64"),
                              tvm.const(batch_factor, dtype="uint64"),
                              name="n1", dtype="uint64") as n1:
                if aipp_config.get('input_format') == "YUV420SP_U8":
                    spr0 = get_const(
                        input_buf.access_ptr(
                            'r',
                            offset=block_index*input_offset +
                            n1*((C*src_image_size_h*src_image_size_w)//2)))
                elif aipp_config.get('input_format') in \
                        ["XRGB8888_U8", "RGB888_U8",
                         "ARGB8888_U8", "AYUV444_U8", "YUV400_U8", "RGB16"]:
                    spr0 = get_const(
                        input_buf.access_ptr(
                            'r',
                            offset=block_index*input_offset +
                            n1*((C*src_image_size_h*src_image_size_w))))
                elif aipp_config.get('input_format') in ["YUYV_U8", "YUV422SP_U8"]:
                    spr0 = get_const(
                        input_buf.access_ptr(
                            'r',
                            offset=block_index*input_offset +
                            n1*((2*src_image_size_h*src_image_size_w))))

                spr1 = tvm.const(0, dtype="uint64")
                if ('csc_switch' in aipp_config) and \
                        (aipp_config.get('csc_switch') == 1):
                    spr1 = tvm.const(1 << 63, dtype="uint64")
                else:
                    if cur_cce_product in ["Hi3796CV300ES", "Hi3796CV300CS"]:
                        if aipp_config.get('input_format') in \
                                ["YUV420SP_U8", "YUYV_U8",
                                 "YUV422SP_U8", "AYUV444_U8"]:
                            spr1 = tvm.const(1 << 63, dtype="uint64")

                if aipp_config.get('input_format') == "YUV420SP_U8":
                    spr1 = get_const(
                        input_buf.access_ptr(
                            'r',
                            offset=block_index*input_offset +
                            n1*((C*src_image_size_h*src_image_size_w)//2) +
                            src_image_size_h*src_image_size_w)) | \
                           spr1
                elif aipp_config.get('input_format') == "YUV422SP_U8":
                    spr1 = get_const(
                        input_buf.access_ptr(
                            'r',
                            offset=block_index*input_offset +
                            n1*((2*src_image_size_h*src_image_size_w)) +
                            src_image_size_h*src_image_size_w)) | \
                           spr1

                ib.emit(tvm.call_extern(dtype, "set_aipp_spr_0", spr0))
                ib.emit(tvm.call_extern(dtype, "set_aipp_spr_1", spr1))

                if "padding" in aipp_config and \
                        aipp_config.get("padding") == 1:
                    if "top_padding_size" in aipp_config and \
                            aipp_config.get("top_padding_size") > 0:
                        with ib.new_scope():
                            top_offset = block_index*offset + n1*C1*H*W*C0
                            process_padding(
                                ib,
                                (dtype, W, C0, size,
                                 top_padding_size, top_offset),
                                output_buf, output_format)

                    if "bottom_padding_size" in aipp_config and \
                            aipp_config.get("bottom_padding_size") > 0:
                        with ib.new_scope():
                            bottom_offset = block_index*offset +\
                                            n1*C1*H*W*C0 +\
                                            (H-bottom_padding_size)*W*C0
                            process_padding(
                                ib,
                                (dtype, W, C0, size,
                                 bottom_padding_size, bottom_offset),
                                output_buf, output_format)
                scfIncVscl = 0
                scfIncHscl = 0
                if cur_cce_product in ["Hi3796CV300ES", "Hi3796CV300CS"]:
                    spr13 = 0
                    spr16 = 0
                    if "resize" in aipp_config and \
                            aipp_config.get("resize") == 1:
                        resize_model = 0
                        if "resize_model" in aipp_config:
                            resize_model = aipp_config.get("resize_model")
                        if resize_model == 1:
                            spr13 = 1 << 8 | 1 << 9 | 1 << 10 | 1 << 11
                        else:
                            spr13 = 0

                        if aipp_config.get("resize_input_h") != \
                                aipp_config.get("resize_output_h"):
                            spr13 = spr13 | 1

                            scfIncVscl = \
                                math.floor(
                                    (aipp_config.get("resize_input_h") - 1) * 262144 /
                                    (aipp_config.get("resize_output_h") - 1)) & \
                                0xFFFFFC
                            spr16 = spr16 | scfIncVscl

                        if aipp_config.get("resize_input_w") != \
                                aipp_config.get("resize_output_w"):
                            spr13 = spr13 | 1 << 2

                            scfIncHscl = \
                                math.floor(
                                    (aipp_config.get("resize_input_w") - 1) * 262144 /
                                    (aipp_config.get("resize_output_w") - 1)) & \
                                0xFFFFFC
                            spr16 = spr16 | scfIncHscl << 32

                        if aipp_config.get("resize_output_w") > \
                                aipp_config.get("resize_input_w"):
                            spr13 = spr13 | 1 << 7

                if l1_image_buf_max >= actual_col_size:
                    with ib.new_scope():
                        output_cb_buf, output_ub_buf = \
                            aipp_comm.new_alloc(ib, dtype,
                                                C1*l1_image_buf_max*C0)

                        if cur_cce_product in ["Hi3796CV300ES", "Hi3796CV300CS"]:
                            spr12 = 0
                            spr15 = 0
                            if ("resize") not in aipp_config or \
                                    aipp_config.get("resize") == 0:
                                spr12 = ((load_image_h - 1)) | \
                                        ((load_image_w - 1) << 16)
                            else:
                                spr12 = \
                                    (aipp_config.get("resize_output_h") - 1) | \
                                    ((aipp_config.get("resize_output_w") - 1) << 16)
                            ib.emit(tvm.call_extern(dtype, "set_aipp_spr_12",
                                                    tvm.const(spr12,
                                                              dtype="uint64")))
                            ib.emit(tvm.call_extern(dtype, "set_aipp_spr_13",
                                                    tvm.const(spr13,
                                                              dtype="uint64")))
                            ib.emit(tvm.call_extern(dtype, "set_aipp_spr_15",
                                                    tvm.const(spr15,
                                                              dtype="uint64")))
                            ib.emit(tvm.call_extern(dtype, "set_aipp_spr_16",
                                                    tvm.const(spr16,
                                                              dtype="uint64")))

                        aippXt = aippXt | (left_padding_size & 0xff) << 32 | \
                                 (right_padding_size & 0xff) << 45

                        aippXs = tvm.const(
                            (load_image_w - 1) |
                            (load_image_h - 1) << 16 |
                            (load_start_pos_w) << 32 |
                            (load_start_pos_h) << 48, dtype="uint64")
                        ib.emit(tvm.call_extern(dtype, "load_image_to_cbuf",
                                                output_cb_buf.access_ptr(
                                                    "rw",
                                                    ptr_type=dtype,
                                                    offset=0),
                                                aippXs,
                                                tvm.const(aippXt,
                                                          dtype="uint64")))

                        output_offset = n1*C1*H*W*C0

                        ib.emit(tvm.call_extern(
                            dtype, 'copy_cbuf_to_ubuf',
                            output_ub_buf.access_ptr("w",
                                                     ptr_type=dtype,
                                                     offset=0),
                            output_cb_buf.access_ptr("rw",
                                                     ptr_type=dtype,
                                                     offset=0),
                            0, 1,
                            (C1*(H - top_padding_size - bottom_padding_size)*W*C0*size + 31) // 32,
                            0, 0))

                        num = C1*(H - top_padding_size - bottom_padding_size)*W*C0
                        if output_format == "NC1HWC0_C04" and \
                                num*size > 32 and num*size % 32 != 0:

                            tail_ub = ib.allocate(dtype, (32 // size,), "tail_ub", scope=cce.scope_ubuf)
                            tail_ub_buf = tvm.decl_buffer((32 // size,), dtype, "tail_ub_buf",
                                                          scope=cce.scope_ubuf, data=tail_ub)
                            aipp_comm.copy_ubuf_to_gm_tail(
                                ib, dtype, output_buf, output_ub_buf,
                                tail_ub_buf,
                                num,
                                block_index*offset +
                                top_padding_size*W*C0 +
                                output_offset,
                                0)
                        ib.emit(tvm.call_extern(
                            dtype, 'copy_ubuf_to_gm',
                            output_buf.access_ptr("w",
                                                  ptr_type=dtype,
                                                  offset=block_index*offset +
                                                  top_padding_size*W*C0 +
                                                  output_offset),
                            output_ub_buf.access_ptr("rw",
                                                     ptr_type=dtype, offset=0),
                            0, 1,
                            C1*(H - top_padding_size - bottom_padding_size)*W*C0*size // 32,
                            0, 0))

                else:
                    buffer_upper_limit = l1_image_buf_max
                    l1_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.L1_SIZE)
                    if output_format == "NC1HWC0_C04":
                        l1_size = l1_size - 64

                    if aipp_config.get('input_format') == "YUV420SP_U8":
                        if 2*W > l1_image_buf_max:
                            buffer_upper_limit = l1_size // size // C0
                    else:
                        if W > l1_image_buf_max:
                            buffer_upper_limit = l1_size // size // C0

                    tiling_h = buffer_upper_limit // W

                    if aipp_config.get('input_format') == "YUV420SP_U8":
                        #tiling_h must be even
                        if tiling_h % 2 != 0:
                            if tiling_h > 1:
                                tiling_h = tiling_h - 1

                    if "resize" in aipp_config and aipp_config.get("resize") == 1:
                        h_loop = aipp_config.get("resize_output_h")//tiling_h
                    else:
                        h_loop = load_image_h // tiling_h

                    h_start_pos = tvm.const(48, dtype="uint64")
                    w_start_pos = tvm.const(32, dtype="uint64")
                    load_h_pos = tvm.const(16, dtype="uint64")
                    tiling_h_const = tvm.const(tiling_h, dtype="uint64")
                    load_w = tvm.const(load_image_w - 1, dtype="uint64")
                    load_h = tvm.const(tiling_h - 1, dtype="uint64")
                    zero_const = tvm.const(0, dtype="uint64")
                    h_loop_const = tvm.const(h_loop, dtype="uint64")

                    with ib.for_range(zero_const, h_loop_const, name="h1",
                                      dtype="uint64") as h1:
                        with ib.new_scope():
                            num = (((C1*buffer_upper_limit*C0*size+31)//32)*32) // size
                            output_cb = ib.allocate(dtype,
                                                    (num,),
                                                    "output_cb",
                                                    scope=cce.scope_cbuf)
                            output_cb_buf = tvm.decl_buffer(
                                (num,), dtype,
                                "output_cb_buf", scope=cce.scope_cbuf,
                                data=output_cb)

                            # ib.scope_attr(output_cb_buf.data,
                            # "double_buffer_scope", 1)
                            # ib.scope_attr(output_ub_buf.data,
                            # "double_buffer_scope", 1)

                            aippXt = aippXt | \
                                     (left_padding_size & 0xff) << 32 | \
                                     (right_padding_size & 0xff) << 45

                            output_w = W
                            output_h = tiling_h
                            output_offset = n1*C1*H*W*C0 + \
                                            C1*(h1*tiling_h)*output_w*C0
                            resize_input_h_stat_pos = 0
                            if cur_cce_product in ["Hi3796CV300ES", "Hi3796CV300CS"]:
                                spr12 = 0
                                spr15 = 0
                                if "resize" not in aipp_config or \
                                        aipp_config.get("resize") == 0:
                                    spr12 = (tiling_h - 1) | \
                                            (load_image_w - 1) << 16
                                else:
                                    spr12 = \
                                        (tiling_h - 1) | \
                                        ((aipp_config.get("resize_output_w") - 1) << 16)

                                    if aipp_config.get("resize_input_h") != \
                                            aipp_config.get("resize_output_h"):
                                        resize_output_h_start_pos = h1*tiling_h
                                        resize_output_h_end_pos = \
                                            ((h1 + 1)*tiling_h - 1)

                                        resize_input_h_stat_pos = \
                                            (scfIncVscl*resize_output_h_start_pos) >> 18
                                        resize_input_h_end_pos = \
                                            ((scfIncVscl*resize_output_h_end_pos) +
                                             (1 << 18) - 1) >> 18
                                        if aipp_config.get("input_format") == "YUV420SP_U8":
                                            resize_input_h_stat_pos = \
                                                resize_input_h_stat_pos & \
                                                0xfffffffffffffffe
                                            resize_input_h_end_pos += \
                                                (resize_input_h_end_pos -
                                                 resize_input_h_stat_pos + 1) & \
                                                0x1

                                        acc_vscl = \
                                            (scfIncVscl*resize_output_h_start_pos) -\
                                            (resize_input_h_stat_pos << 18)
                                        spr15 = acc_vscl

                                        load_h = \
                                            get_const(
                                                (resize_input_h_end_pos -
                                                 resize_input_h_stat_pos))
                                ib.emit(
                                    tvm.call_extern(
                                        dtype, "set_aipp_spr_12",
                                        tvm.const(spr12, dtype="uint64")))
                                ib.emit(
                                    tvm.call_extern(
                                        dtype, "set_aipp_spr_13",
                                        tvm.const(spr13, dtype="uint64")))
                                ib.emit(
                                    tvm.call_extern(dtype, "set_aipp_spr_15",
                                                    get_const(spr15)))
                                ib.emit(
                                    tvm.call_extern(dtype, "set_aipp_spr_16",
                                                    tvm.const(spr16,
                                                              dtype="uint64")))

                            if "resize" in aipp_config and \
                                    aipp_config.get("resize") == 1 and \
                                    aipp_config.get("resize_input_h") != \
                                    aipp_config.get("resize_output_h"):
                                aippXs = get_const(
                                    load_w | load_h << load_h_pos |
                                    (tvm.const(load_start_pos_w, dtype="uint64")) << w_start_pos |
                                    (load_start_pos_h + resize_input_h_stat_pos) << h_start_pos)
                            else:
                                aippXs = get_const(
                                    load_w | load_h << load_h_pos |
                                    (tvm.const(load_start_pos_w, dtype="uint64")) << w_start_pos |
                                    (load_start_pos_h + h1*tiling_h_const) << h_start_pos)
                            ib.emit(tvm.call_extern(dtype, "load_image_to_cbuf",
                                                    output_cb_buf.access_ptr(
                                                        "rw", ptr_type=dtype,
                                                        offset=0),
                                                    aippXs,
                                                    tvm.const(aippXt,
                                                              dtype="uint64")))

                            move_data_from_l1_to_gm(ib, C1*output_h*W*C0, dtype,
                                                    output_cb_buf, output_buf,
                                                    block_index*offset+top_padding_size*W*C0+output_offset,
                                                    output_format)

                    if "resize" in aipp_config and \
                            aipp_config.get("resize") == 1:
                        tail_h = aipp_config.get("resize_output_h") % tiling_h
                    else:
                        tail_h = load_image_h % tiling_h
                    if tail_h != 0:
                        tail_h_postion = tvm.const(
                            load_start_pos_h + h_loop*tiling_h, dtype="uint64")
                        load_tail_h = tvm.const(tail_h - 1, dtype="uint64")

                        with ib.new_scope():
                            num = (((C1*buffer_upper_limit*C0*size + 31) // 32)*32) // size
                            output_cb = ib.allocate(dtype,
                                                    (num,),
                                                    "output_cb",
                                                    scope=cce.scope_cbuf)
                            output_cb_buf = tvm.decl_buffer(
                                (num,), dtype,
                                "output_cb_buf", scope=cce.scope_cbuf,
                                data=output_cb)

                            output_w = W
                            aippXt = aippXt | \
                                     (left_padding_size & 0xff) << 32 | \
                                     (right_padding_size & 0xff) << 45
                            output_h = tail_h
                            output_offset = \
                                n1*C1*H*W*C0 + C1*(h_loop_const*tiling_h)*output_w*C0

                            if cur_cce_product in ["Hi3796CV300ES", "Hi3796CV300CS"]:
                                spr12 = 0
                                spr15 = 0
                                if ("resize") not in aipp_config or \
                                        aipp_config.get("resize") == 0:
                                    spr12 = (tail_h - 1) | \
                                            (load_image_w - 1) << 16
                                else:
                                    spr12 = \
                                        (tail_h - 1) | \
                                        ((aipp_config.get("resize_output_w") - 1) << 16)


                                    if aipp_config.get("resize_input_h") != \
                                            aipp_config.get("resize_output_h"):
                                        resize_output_h_start_pos = h_loop*tiling_h
                                        resize_output_h_end_pos = \
                                            aipp_config.get("resize_output_h") - 1

                                        resize_input_h_stat_pos = \
                                            (scfIncVscl*resize_output_h_start_pos) >> 18
                                        resize_input_h_end_pos = \
                                            ((scfIncVscl*resize_output_h_end_pos) +
                                             (1 << 18) - 1) >> 18
                                        if aipp_config.get("input_format") == "YUV420SP_U8":
                                            resize_input_h_stat_pos = \
                                                resize_input_h_stat_pos & \
                                                0xfffffffffffffffe
                                            resize_input_h_end_pos += \
                                                (resize_input_h_end_pos -
                                                 resize_input_h_stat_pos + 1) & \
                                                0x1

                                        acc_vscl = \
                                            (scfIncVscl*resize_output_h_start_pos) - \
                                            (resize_input_h_stat_pos << 18)
                                        spr15 = acc_vscl

                                        load_tail_h = tvm.const(
                                            (resize_input_h_end_pos -
                                             resize_input_h_stat_pos),
                                            dtype="uint64")
                                        tail_h_postion = tvm.const(
                                            (load_start_pos_h +
                                             resize_input_h_stat_pos),
                                            dtype="uint64")
                                ib.emit(
                                    tvm.call_extern(
                                        dtype, "set_aipp_spr_12",
                                        tvm.const(spr12, dtype="uint64")))
                                ib.emit(
                                    tvm.call_extern(
                                        dtype, "set_aipp_spr_13",
                                        tvm.const(spr13, dtype="uint64")))
                                ib.emit(
                                    tvm.call_extern(
                                        dtype, "set_aipp_spr_15",
                                        tvm.const(spr15, dtype="uint64")))
                                ib.emit(
                                    tvm.call_extern(
                                        dtype, "set_aipp_spr_16",
                                        tvm.const(spr16, dtype="uint64")))

                            aippXs = get_const(
                                load_w | load_tail_h << load_h_pos |
                                tail_h_postion << h_start_pos |
                                (tvm.const(load_start_pos_w, dtype="uint64")) << w_start_pos)

                            ib.emit(tvm.call_extern(dtype, "load_image_to_cbuf",
                                                    output_cb_buf.access_ptr(
                                                        "rw", ptr_type=dtype,
                                                        offset=0),
                                                    tvm.const(aippXs,
                                                              dtype="uint64"),
                                                    tvm.const(aippXt,
                                                              dtype="uint64")))
                            move_data_from_l1_to_gm(ib, C1*output_h*W*C0, dtype,
                                                    output_cb_buf, output_buf,
                                                    block_index*offset+top_padding_size*W*C0+output_offset,
                                                    output_format)

        _aipp_intrin()
        return ib.get()

    return tvm.extern([(N, C1, H, W, C0)], [input_tensor],
                      lambda ins, outs: aipp_ir(ins[0], outs[0]),
                      dtype=[dtype], name="aipp")
