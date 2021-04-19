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
aipp
"""
# pylint: disable=too-many-branches
# pylint: disable=ungrouped-imports,E0401
import json
from te import tvm
from te import platform as cce
from te import platform as tbe_platform
from te.platform.cce_build import build_config
from topi.cce import util
from te.platform.cce_util import get_const
from impl import aipp_comm
from impl import aipp_resize_padding
from impl import aipp_dynamic
from te.utils.op_utils import *
from te.platform.fusion_manager import fusion_manager

NoneType = type(None)

# pylint: disable=invalid-name,unused-argument,too-many-statements
# pylint: disable=too-many-arguments,too-many-locals,


@fusion_manager.register("aipp")
def aipp_compute(input_data, input_dync_param, output_data,
                 aipp_config_json, kernel_name="aipp"):
    """

    :param input_data:
    :param input_dync_param:
    :param output_data:
    :param aipp_config_json:
    :param kernel_name:
    :return:
    """


    aipp_config = json.loads(aipp_config_json)

    input_shape = input_data.shape
    input_format = input_data.op.attrs['format']
    output_dtype = output_data.get('dtype')
    output_shape = output_data.get('shape')
    output_format = output_data.get('format')
    ori_format = output_data.get('ori_format')

    res_shape = output_shape

    c0_channel = output_shape[4]

    aipp_map = {}
    aipp_map["ori_format"] = ori_format

    load_start_pos_h = 0
    load_start_pos_w = 0
    crop_size_h = 0
    crop_size_w = 0

    cur_cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if output_format == "NC1HWC0_C04":
        if c0_channel != 4:
            cause_dec = "when output_format is NC1HWC0_C04, C0[%d] must be 4" % \
                        c0_channel
            aipp_comm.raise_RuntimeError(cause_dec)

    if aipp_config.get('aipp_mode') == "static":
        aipp_comm.set_aipp_default_params(aipp_config)
        aipp_map["input_format"] = aipp_config.get('input_format')

        if 'csc_switch' \
                in aipp_config and aipp_config.get('csc_switch') == 1:
            aipp_map["spr_1"] = tvm.const(1 << 63, dtype="uint64")
        else:
            aipp_map["spr_1"] = tvm.const(0 << 63, dtype="uint64")
            if cur_cce_product in ["Hi3796CV300ES", "Hi3796CV300CS"]:
                if aipp_config.get('input_format') in \
                        ["YUV420SP_U8", "YUYV_U8",
                         "YUV422SP_U8", "AYUV444_U8"]:
                    aipp_map["spr_1"] = tvm.const(1 << 63, dtype="uint64")

        aipp_comm.get_spr2_spr9(aipp_config, output_dtype, cur_cce_product,
                                output_format, aipp_map)

        if 'crop' in aipp_config and aipp_config.get('crop') == 1:
            if "load_start_pos_h" in aipp_config:
                load_start_pos_h = aipp_config.get('load_start_pos_h')
            if "load_start_pos_w" in aipp_config:
                load_start_pos_w = aipp_config.get('load_start_pos_w')

            if ("crop_size_h") in aipp_config:
                crop_size_h = aipp_config["crop_size_h"]
            else:
                crop_size_h = output_shape[2]

            if ("crop_size_w") in aipp_config:
                crop_size_w = aipp_config["crop_size_w"]
            else:
                crop_size_w = output_shape[3]

        if input_format == "NCHW":
            channel = input_shape[1]
            height = input_shape[2]
            width = input_shape[3]

        elif input_format == "NHWC":
            height = input_shape[1]
            width = input_shape[2]
            channel = input_shape[3]

        src_image_size_h = height
        src_image_size_w = width
        if "src_image_size_h" in aipp_config and \
                aipp_config.get('src_image_size_h') > 0:
            src_image_size_h = aipp_config.get('src_image_size_h')
        if "src_image_size_w" in aipp_config and \
                aipp_config.get('src_image_size_w') > 0:
            src_image_size_w = aipp_config.get('src_image_size_w')

        aipp_map["src_image_h"] = src_image_size_h
        aipp_map["src_image_w"] = src_image_size_w

    if 'crop' in aipp_config and aipp_config.get('crop') == 1:
        load_image_h = crop_size_h
        load_image_w = crop_size_w
    else:
        load_image_h = src_image_size_h
        load_image_w = src_image_size_w

    aipp_map["load_start_pos_h"] = load_start_pos_h
    aipp_map["load_start_pos_w"] = load_start_pos_w
    aipp_map["crop_size_h"] = load_image_h
    aipp_map["crop_size_w"] = load_image_w

    aipp_res = tvm.compute(
        res_shape,
        lambda n, c1, h, w, c0:
        tvm.select(tvm.all(c1*c0_channel + c0 < channel),
                   input_data[n,
                              c1*c0_channel+c0,
                              load_start_pos_h + h,
                              load_start_pos_w + w].astype(output_dtype),
                   tvm.const(0).astype(output_dtype)),
        name="aipp_res",
        tag="aipp_res_convolution",
        attrs=aipp_map)

    return aipp_res


def aipp_compute_single(input_tensor, input_shape, input_format,
                 output_data, aipp_config):
    """

    :param input_tensor:
    :param input_shape:
    :param input_format:
    :param output_data:
    :param aipp_config:
    :param is_dynamic:
    :return:
    """


    if input_format == "NHWC":
        N, H, W, C = input_shape
    elif input_format == "NCHW":
        N, C, H, W = input_shape
    else:
        N, C1, H, W, C0 = input_shape

    output_shape = output_data.get('shape')
    N, C1, H, W, C0 = output_shape

    src_image_size_h = H
    src_image_size_w = W
    load_start_pos_h = 0
    load_start_pos_w = 0
    load_image_h = H
    load_image_w = W

    output_format = output_data.get('format')

    if "crop" in aipp_config and aipp_config.get("crop") == 1:
        src_image_size_h, src_image_size_w, load_start_pos_h, load_start_pos_w, \
        load_image_h, load_image_w = aipp_comm.get_crop_info(aipp_config)

    dtype = output_data.get('dtype')
    if dtype == "float16":
        size = 2  # One pixel occupied bytes
        C0 = 16
    else:
        size = 1  # One pixel occupied bytes
        C0 = 32

    if output_format == "NC1HWC0_C04":
        C0 = 4

    if aipp_config.get('input_format') not in ["NC1HWC0DI_S8", "NC1HWC0DI_FP16"]:
        C1 = (C + C0 - 1) // C0

    actual_col_size = H * W
    if "crop" in aipp_config and aipp_config.get("crop") == 1:
        actual_col_size = aipp_comm.get_actual_col_size(aipp_config, H, W)

    l1_image_buf_max = aipp_comm.get_l1_image_buf_max(actual_col_size, dtype, False, output_format)

    def aipp_ir(input_buf, output_buf):
        ib = tvm.ir_builder.create()
        if l1_image_buf_max < actual_col_size:
            half_l1_image_buf_max = l1_image_buf_max

        cur_cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")

        if cur_cce_product not in ["Ascend310", "Ascend610", "Ascend710",
                                   "Ascend615", "Hi3796CV300ES", "Hi3796CV300CS"]:
            cause_desc = "Only support is Ascend310,Ascend610,Ascend710,Ascend615," \
                        "Hi3796CV300ES,Hi3796CV300CS. " \
                        "cur_cce_product is %s" % cur_cce_product
            aipp_comm.raise_RuntimeError(cause_desc)

        device_core_num = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        batch_num = N
        batch_factor = 1

        if batch_num % device_core_num == 0:
            batch_factor = batch_num // device_core_num

        block_index = tvm.thread_axis("blockIdx.x")
        ib.scope_attr(block_index, "thread_extent", batch_num // batch_factor)

        if aipp_config.get('input_format') == "YUV420SP_U8":
            input_offset = \
                batch_factor*((C*src_image_size_h*src_image_size_w) // 2)
        elif aipp_config.get('input_format') in ["XRGB8888_U8", "RGB888_U8",
                                                 "ARGB8888_U8", "AYUV444_U8",
                                                 "YUV400_U8", "RAW10", "RAW12",
                                                 "RAW16", "uint16",
                                                 "RGB16", "RGB20", "RGB24",
                                                 "RGB8_IR", "RGB16_IR",
                                                 "RGB24_IR"]:
            input_offset = \
                batch_factor*((C*src_image_size_h*src_image_size_w))
        elif aipp_config.get('input_format') in ["YUYV_U8", "YUV422SP_U8"]:
            input_offset = \
                batch_factor*((2*src_image_size_h*src_image_size_w))
        elif aipp_config.get('input_format') in \
                ["NC1HWC0DI_S8", "NC1HWC0DI_FP16"]:
            input_offset = \
                batch_factor*((C1*src_image_size_h*src_image_size_w*6))

        offset = batch_factor*C1*H*W*C0

        zero_const = tvm.const(0, dtype="uint64")

        def _aipp_intrin():
            # config SPR2~SPR9

            aipp_comm.set_spr2_spr9(ib, aipp_config, dtype, cur_cce_product, output_format)

            aippXt = (src_image_size_w - 1)

            with ib.for_range(zero_const,
                              tvm.const(batch_factor, dtype="uint64"),
                              name="n1", dtype="uint64") as n1:
                with ib.for_range(zero_const,
                                  tvm.const(C1, dtype="uint64"),
                                  name="c1_index",
                                  dtype="uint64") as c1_index:
                    if aipp_config.get('input_format') == "YUV420SP_U8":
                        spr0 = get_const(
                            input_buf.access_ptr('r',
                                                 offset=block_index*input_offset +
                                                        n1*((C*src_image_size_h*src_image_size_w)//2)))
                    elif aipp_config.get('input_format') in ["XRGB8888_U8",
                                                             "RGB888_U8",
                                                             "ARGB8888_U8",
                                                             "AYUV444_U8",
                                                             "YUV400_U8",
                                                             "RAW10", "RAW12",
                                                             "RAW16", "uint16",
                                                             "RGB16", "RGB8_IR",
                                                             "RGB16_IR"]:
                        spr0 = get_const(
                            input_buf.access_ptr('r',
                                                 offset=block_index*input_offset +
                                                        n1*((C*src_image_size_h*src_image_size_w))))
                    elif aipp_config.get('input_format') in ["YUYV_U8",
                                                             "YUV422SP_U8"]:
                        spr0 = get_const(
                            input_buf.access_ptr('r',
                                                 offset=block_index*input_offset +
                                                        n1*((2*src_image_size_h*src_image_size_w))))
                    elif aipp_config.get('input_format') in ["NC1HWC0DI_S8",
                                                             "NC1HWC0DI_FP16"]:
                        spr0 = get_const(
                            input_buf.access_ptr('r',
                                                 offset=block_index*input_offset +
                                                        n1*((C1*src_image_size_h*src_image_size_w*6)) +
                                                        c1_index*src_image_size_h*src_image_size_w*4))
                    elif aipp_config.get('input_format') in ["RGB20", "RGB24",
                                                             "RGB24_IR"]:
                        mean_chn_0 = 0
                        mean_chn_1 = 0
                        if 'mean_chn_0' in aipp_config:
                            mean_chn_0 = aipp_config.get('mean_chn_0')
                        if 'mean_chn_1' in aipp_config:
                            mean_chn_1 = aipp_config.get('mean_chn_1')
                        spr0 = get_const(
                            input_buf.access_ptr('r',
                                                 offset=block_index*input_offset +
                                                        n1*((C*src_image_size_h*src_image_size_w)))) | \
                               tvm.const(((mean_chn_0 >> 16) & 0xff) << 48,
                                         dtype="uint64") | \
                               tvm.const(((mean_chn_1 >> 16) & 0xff) << 56,
                                         dtype="uint64")

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
                                       src_image_size_h * src_image_size_w)) | \
                               spr1
                    elif aipp_config.get('input_format') == "YUV422SP_U8":
                        spr1 = get_const(
                            input_buf.access_ptr(
                                'r',
                                offset=block_index*input_offset +
                                       n1*((2*src_image_size_h*src_image_size_w)) +
                                       src_image_size_h*src_image_size_w)) | \
                               spr1
                    elif aipp_config.get('input_format') in \
                            ["NC1HWC0DI_S8", "NC1HWC0DI_FP16"]:
                        spr1 = get_const(
                            input_buf.access_ptr(
                                'r',
                                offset=block_index*input_offset +
                                       n1*(C1*src_image_size_h*src_image_size_w*6) +
                                       C1*src_image_size_h*src_image_size_w*4 +
                                       c1_index*src_image_size_h*src_image_size_w*2)) | \
                               spr1
                    elif aipp_config.get('input_format') in ["RGB8_IR",
                                                             "RGB16_IR",
                                                             "RGB24_IR"]:
                        spr1 = get_const(
                            input_buf.access_ptr(
                                'r',
                                offset=block_index*input_offset +
                                       n1*((C*src_image_size_h*src_image_size_w)) +
                                       3*src_image_size_h*src_image_size_w)) | \
                               spr1

                    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_0", spr0))
                    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_1", spr1))

                    if cur_cce_product in ["Hi3796CV300ES", "Hi3796CV300CS"]:
                        spr13 = 0
                        spr15 = 0
                        spr16 = 0
                        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_13",
                                                tvm.const(spr13, dtype="uint64")))
                        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_15",
                                                tvm.const(spr15, dtype="uint64")))
                        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_16",
                                                tvm.const(spr16, dtype="uint64")))
                    if l1_image_buf_max >= actual_col_size:
                        with ib.new_scope():
                            output_cb_buf, output_ub_buf = \
                                aipp_comm.new_alloc(ib, dtype,
                                                    l1_image_buf_max * C0)

                            if cur_cce_product in ["Hi3796CV300ES", "Hi3796CV300CS"]:
                                spr12 = 0
                                spr12 = \
                                    ((load_image_h - 1)) | \
                                    ((load_image_w - 1) << 16)
                                ib.emit(
                                    tvm.call_extern(dtype, "set_aipp_spr_12",
                                                    tvm.const(spr12,
                                                              dtype="uint64")))


                            aippXs = tvm.const((load_image_w - 1) |
                                               (load_image_h - 1) << 16 |
                                               (load_start_pos_w) << 32 |
                                               (load_start_pos_h) << 48,
                                               dtype="uint64")

                            ib.emit(tvm.call_extern(dtype, "load_image_to_cbuf",
                                                    output_cb_buf.access_ptr(
                                                        "rw", ptr_type=dtype,
                                                        offset=0),
                                                    aippXs,
                                                    tvm.const(aippXt,
                                                              dtype="uint64")))

                            lenBurst, nBurst = \
                                aipp_comm.get_lenBurst_and_nBurst(
                                    (H*W*C0*size + 31)//32, 1)
                            output_offset = n1*C1*H*W*C0 + c1_index*H*W*C0

                            ib.emit(tvm.call_extern(
                                dtype, 'copy_cbuf_to_ubuf',
                                output_ub_buf.access_ptr("w", ptr_type=dtype,
                                                         offset=0),
                                output_cb_buf.access_ptr("rw", ptr_type=dtype,
                                                         offset=0),
                                0, nBurst, lenBurst, 0, 0))

                            if output_format == "NC1HWC0_C04" and H*W*C0*size % 32 != 0:
                                lenBurst, nBurst = \
                                    aipp_comm.get_lenBurst_and_nBurst(
                                        (H*W*C0*size)//32, 1)

                                tail_ub = ib.allocate(dtype, (32 // size,), "tail_ub", scope=cce.scope_ubuf)
                                tail_ub_buf = tvm.decl_buffer((32 // size,), dtype, "tail_ub_buf",
                                                                scope=cce.scope_ubuf, data=tail_ub)
                                aipp_comm.copy_ubuf_to_gm_tail(
                                    ib, dtype, output_buf, output_ub_buf,
                                    tail_ub_buf,
                                    H*W*C0, block_index*offset + output_offset,
                                    0)

                            ib.emit(tvm.call_extern(
                                dtype, 'copy_ubuf_to_gm',
                                output_buf.access_ptr(
                                    "w", ptr_type=dtype,
                                    offset=block_index*offset + output_offset),
                                output_ub_buf.access_ptr("rw", ptr_type=dtype,
                                                         offset=0),
                                0, nBurst, lenBurst, 0, 0))

                    else:
                        tiling_w, w_loop = \
                            aipp_comm.get_tiling_W(W, half_l1_image_buf_max, 1)
                        tiling_h = half_l1_image_buf_max // tiling_w

                        if aipp_config.get('input_format') == "YUV420SP_U8":
                            #tiling_h of YUV420SP_U8 must be even
                            if tiling_h % 2 != 0:
                                if tiling_h > 1:
                                    tiling_h = tiling_h - 1
                                if tiling_h == 1:
                                    tiling_h = 2
                                    tiling_w = tiling_w // 2
                                    w_loop = w_loop * 2

                        if aipp_config.get('input_format') in \
                                ["YUV420SP_U8", "YUV422SP_U8", "YUYV_U8"]:
                            if tiling_w % 2 != 0:
                                tiling_w = tiling_w - 1

                        tail_w = W - w_loop * tiling_w

                        h_loop = load_image_h // tiling_h

                        if output_format == "NC1HWC0_C04" and tail_w > 0:
                            if dtype == "float16":
                                if tail_w < 4:
                                    if 4 // w_loop > 0 and 4 % w_loop == 0:
                                        tiling_w = tiling_w - (4 // w_loop)
                                    elif 4 // w_loop > 0 and 4 % w_loop != 0:
                                        tiling_w = tiling_w - (4 // w_loop) - 1
                                    else:
                                        tiling_w = tiling_w - 1
                                    tiling_h = half_l1_image_buf_max // tiling_w
                                    tail_w = W - w_loop * tiling_w
                                    h_loop = load_image_h // tiling_h
                            else:
                                if tail_w < 8:
                                    tiling_w = tiling_w - 1
                                    if 8 // w_loop > 0 and 8 % w_loop == 0:
                                        tiling_w = tiling_w - (8 // w_loop)
                                    elif 8 // w_loop > 0 and 8 % w_loop != 0:
                                        tiling_w = tiling_w - (8 // w_loop) - 1
                                    else:
                                        tiling_w = tiling_w - 1
                                    tiling_h = half_l1_image_buf_max // tiling_w
                                    tail_w = W - w_loop * tiling_w
                                    h_loop = load_image_h // tiling_h

                        h_start_pos = tvm.const(48, dtype="uint64")
                        w_start_pos = tvm.const(32, dtype="uint64")
                        load_h_pos = tvm.const(16, dtype="uint64")
                        tiling_h_const = tvm.const(tiling_h, dtype="uint64")
                        tiling_w_const = tvm.const(tiling_w, dtype="uint64")
                        load_w = tvm.const(tiling_w - 1, dtype="uint64")
                        load_h = tvm.const(tiling_h - 1, dtype="uint64")
                        load_tail_w = tvm.const(tail_w - 1, dtype="uint64")

                        h_loop_const = tvm.const(h_loop, dtype="uint64")
                        w_loop_const = tvm.const(w_loop, dtype="uint64")

                        with ib.for_range(zero_const, h_loop_const,
                                          name="h1", dtype="uint64") as h1:
                            with ib.for_range(zero_const, w_loop_const,
                                              name="w1", dtype="uint64") as w1:
                                with ib.new_scope():
                                    output_cb_buf, output_ub_buf = \
                                        aipp_comm.new_alloc(
                                            ib, dtype, half_l1_image_buf_max*C0)
                                    # ib.scope_attr(output_cb_buf.data,
                                    # "double_buffer_scope", 1)
                                    # ib.scope_attr(output_ub_buf.data,
                                    # "double_buffer_scope", 1)

                                    if cur_cce_product in ["Hi3796CV300ES", "Hi3796CV300CS"]:
                                        spr12 = 0
                                        spr12 = get_const((tiling_h - 1) |
                                                          (tiling_w - 1) << 16)
                                        ib.emit(tvm.call_extern(
                                            dtype, "set_aipp_spr_12",
                                            tvm.const(spr12, dtype="uint64")))

                                    aippXs = get_const(
                                        load_w | load_h << load_h_pos |
                                        (load_start_pos_w + w1*tiling_w_const) << w_start_pos |
                                        (load_start_pos_h + h1*tiling_h_const) << h_start_pos)
                                    ib.emit(tvm.call_extern(
                                        dtype, "load_image_to_cbuf",
                                        output_cb_buf.access_ptr("rw",
                                                                 ptr_type=dtype,
                                                                 offset=0),
                                        aippXs,
                                        tvm.const(aippXt, dtype="uint64")))

                                    lenBurst, nBurst = \
                                        aipp_comm.get_lenBurst_and_nBurst(
                                            (tiling_h*tiling_w*C0*size + 31)//32, 1)

                                    ib.emit(tvm.call_extern(
                                        dtype, 'copy_cbuf_to_ubuf',
                                        output_ub_buf.access_ptr("w",
                                                                 ptr_type=dtype,
                                                                 offset=0),
                                        output_cb_buf.access_ptr("rw",
                                                                 ptr_type=dtype,
                                                                 offset=0),
                                        0, nBurst, lenBurst, 0, 0))

                                    if w_loop == 1 and W % tiling_w == 0:
                                        output_offset = n1*C1*H*W*C0 + \
                                                        c1_index*H*W*C0 + \
                                                        h1*tiling_h*tiling_w*C0
                                        if output_format == "NC1HWC0_C04" and \
                                            tiling_h*tiling_w*C0*size % 32 != 0:
                                            lenBurst, nBurst = \
                                                aipp_comm.get_lenBurst_and_nBurst(
                                                    (tiling_h*tiling_w*C0*size)//32, 1)

                                            tail_ub = ib.allocate(dtype, (32 // size,), "tail_ub", scope=cce.scope_ubuf)
                                            tail_ub_buf = tvm.decl_buffer((32 // size,), dtype, "tail_ub_buf",
                                                                          scope=cce.scope_ubuf, data=tail_ub)
                                            aipp_comm.copy_ubuf_to_gm_tail(
                                                ib, dtype, output_buf, output_ub_buf,
                                                tail_ub_buf,
                                                tiling_h*tiling_w*C0, block_index*offset + output_offset,
                                                0)

                                        ib.emit(tvm.call_extern(
                                            dtype, 'copy_ubuf_to_gm',
                                            output_buf.access_ptr(
                                                "w", ptr_type=dtype,
                                                offset=block_index*offset + \
                                                       output_offset),
                                            output_ub_buf.access_ptr(
                                                "rw", ptr_type=dtype, offset=0),
                                            0, nBurst, lenBurst, 0, 0))
                                    else:
                                        with ib.for_range(
                                                zero_const,
                                                tvm.const(tiling_h,
                                                          dtype="uint64"),
                                                name="h2", dtype="uint64") as h2:
                                            output_offset = \
                                                n1*C1*H*W*C0 + \
                                                c1_index*H*W*C0 + \
                                                (h1*tiling_h)*W*C0 + \
                                                (w1*tiling_w) * C0 + h2*W*C0
                                            lenBurst, nBurst = \
                                                aipp_comm.get_lenBurst_and_nBurst(
                                                    (1*tiling_w*C0*size + 31)//32, 1)

                                            if output_format == "NC1HWC0_C04" and \
                                                    1*tiling_w*C0*size % 32 != 0:
                                                lenBurst, nBurst = \
                                                    aipp_comm.get_lenBurst_and_nBurst(
                                                        (1*tiling_w*C0*size)//32, 1)
                                                tail_ub = ib.allocate(dtype, (32 // size,), "tail_ub", scope=cce.scope_ubuf)
                                                tail_ub_buf = tvm.decl_buffer((32 // size,), dtype, "tail_ub_buf",
                                                                              scope=cce.scope_ubuf, data=tail_ub)
                                                aipp_comm.copy_ubuf_to_gm_tail(
                                                    ib, dtype, output_buf, output_ub_buf,
                                                    tail_ub_buf,
                                                    1*tiling_w*C0, block_index*offset + output_offset,
                                                    0)

                                            ib.emit(tvm.call_extern(
                                                dtype, 'copy_ubuf_to_gm',
                                                output_buf.access_ptr(
                                                    "w", ptr_type=dtype,
                                                    offset=block_index*offset +
                                                           output_offset),
                                                output_ub_buf.access_ptr(
                                                    "rw", ptr_type=dtype,
                                                    offset=h2*tiling_w*C0),
                                                0, nBurst, lenBurst, 0, 0))

                            #process W tail
                            if W % tiling_w != 0:
                                with ib.new_scope():
                                    output_cb_buf, output_ub_buf = \
                                        aipp_comm.new_alloc(
                                            ib, dtype, half_l1_image_buf_max*C0)
                                    # ib.scope_attr(output_cb_buf.data,
                                    # "double_buffer_scope", 1)
                                    # ib.scope_attr(output_ub_buf.data,
                                    # "double_buffer_scope", 1)

                                    if cur_cce_product in ["Hi3796CV300ES", "Hi3796CV300CS"]:
                                        spr12 = 0
                                        spr12 = \
                                            ((tiling_h - 1) | (tail_w - 1) << 16)
                                        ib.emit(
                                            tvm.call_extern(
                                                dtype, "set_aipp_spr_12",
                                                tvm.const(spr12, dtype="uint64")))

                                    aippXs = get_const(
                                        load_tail_w |
                                        load_h << load_h_pos |
                                        (load_start_pos_w +
                                         w_loop*tiling_w_const) << w_start_pos |
                                        (load_start_pos_h +
                                         h1*tiling_h_const) << h_start_pos)

                                    ib.emit(tvm.call_extern(
                                        dtype, "load_image_to_cbuf",
                                        output_cb_buf.access_ptr("rw",
                                                                 ptr_type=dtype,
                                                                 offset=0),
                                        aippXs,
                                        tvm.const(aippXt, dtype="uint64")))

                                    lenBurst, nBurst = \
                                        aipp_comm.get_lenBurst_and_nBurst(
                                            tiling_h*tail_w*C0*size//32, 1)

                                    ib.emit(tvm.call_extern(
                                        dtype, 'copy_cbuf_to_ubuf',
                                        output_ub_buf.access_ptr("w",
                                                                 ptr_type=dtype,
                                                                 offset=0),
                                        output_cb_buf.access_ptr("rw",
                                                                 ptr_type=dtype,
                                                                 offset=0),
                                        0, nBurst, lenBurst, 0, 0))

                                    with ib.for_range(zero_const,
                                                      tiling_h_const,
                                                      name="h2",
                                                      dtype="uint64") as h2:
                                        output_offset = \
                                            n1*C1*H*W*C0 + \
                                            c1_index*H*W*C0 + \
                                            (h1*tiling_h)*W*C0 + \
                                            (w_loop*tiling_w)*C0 + h2*W*C0
                                        lenBurst, nBurst = \
                                            aipp_comm.get_lenBurst_and_nBurst(
                                                (1*tail_w*C0*size + 31)//32, 1)

                                        if output_format == "NC1HWC0_C04" and 1*tail_w*C0*size % 32 != 0:
                                            lenBurst, nBurst = \
                                                aipp_comm.get_lenBurst_and_nBurst(
                                                    (1*tail_w*C0*size)//32, 1)

                                            tail_ub = ib.allocate(dtype, (32 // size,), "tail_ub", scope=cce.scope_ubuf)
                                            tail_ub_buf = tvm.decl_buffer((32 // size,), dtype, "tail_ub_buf",
                                                                          scope=cce.scope_ubuf, data=tail_ub)
                                            aipp_comm.copy_ubuf_to_gm_tail(
                                                ib, dtype, output_buf, output_ub_buf,
                                                tail_ub_buf,
                                                1*tail_w*C0, block_index*offset + output_offset,
                                                0)

                                        ib.emit(tvm.call_extern(
                                            dtype, 'copy_ubuf_to_gm',
                                            output_buf.access_ptr(
                                                "w", ptr_type=dtype,
                                                offset=block_index*offset + \
                                                       output_offset),
                                            output_ub_buf.access_ptr(
                                                "rw", ptr_type=dtype,
                                                offset=h2*tail_w*C0),
                                            0, nBurst, lenBurst, 0, 0))

                        if load_image_h % tiling_h != 0:
                            tail_h = load_image_h % tiling_h

                            tail_h_postion = tvm.const(
                                load_start_pos_h + h_loop*tiling_h,
                                dtype="uint64")

                            load_tail_h = tvm.const(tail_h - 1, dtype="uint64")

                            with ib.for_range(zero_const,
                                              w_loop_const,
                                              name="w1", dtype="uint64") as w1:
                                with ib.new_scope():
                                    output_cb_buf, output_ub_buf = \
                                        aipp_comm.new_alloc(
                                            ib, dtype, half_l1_image_buf_max*C0)
                                    # ib.scope_attr(output_cb_buf.data,
                                    # "double_buffer_scope", 1)
                                    # ib.scope_attr(output_ub_buf.data,
                                    # "double_buffer_scope", 1)

                                    if cur_cce_product in ["Hi3796CV300ES", "Hi3796CV300CS"]:
                                        spr12 = 0
                                        spr12 = \
                                            ((tail_h - 1) | (tiling_w - 1) << 16)
                                        ib.emit(
                                            tvm.call_extern(
                                                dtype, "set_aipp_spr_12",
                                                tvm.const(spr12,
                                                          dtype="uint64")))

                                    aippXs = get_const(
                                        load_w | load_tail_h << load_h_pos |
                                        tail_h_postion << h_start_pos |
                                        (load_start_pos_w +
                                         w1*tiling_w_const) << w_start_pos)

                                    ib.emit(
                                        tvm.call_extern(
                                            dtype, "load_image_to_cbuf",
                                            output_cb_buf.access_ptr(
                                                "rw", ptr_type=dtype, offset=0),
                                            aippXs,
                                            tvm.const(aippXt, dtype="uint64")))

                                    lenBurst, nBurst = \
                                        aipp_comm.get_lenBurst_and_nBurst(
                                            (tail_h*tiling_w*C0*size + 31)//32, 1)

                                    ib.emit(tvm.call_extern(
                                        dtype, 'copy_cbuf_to_ubuf',
                                        output_ub_buf.access_ptr("w",
                                                                 ptr_type=dtype,
                                                                 offset=0),
                                        output_cb_buf.access_ptr("rw",
                                                                 ptr_type=dtype,
                                                                 offset=0),
                                        0, nBurst, lenBurst, 0, 0))
                                    if w_loop == 1 and W % tiling_w == 0:
                                        output_offset = \
                                            n1*C1*H*W*C0 + \
                                            c1_index*H*W*C0 + \
                                            (tiling_h*h_loop)*tiling_w*C0

                                        if output_format == "NC1HWC0_C04" and tail_h*tiling_w*C0*size % 32 != 0:
                                            lenBurst, nBurst = \
                                                aipp_comm.get_lenBurst_and_nBurst(
                                                    (tail_h*tiling_w*C0*size)//32, 1)

                                            tail_ub = ib.allocate(dtype, (32 // size,), "tail_ub", scope=cce.scope_ubuf)
                                            tail_ub_buf = tvm.decl_buffer((32 // size,), dtype, "tail_ub_buf",
                                                                          scope=cce.scope_ubuf, data=tail_ub)
                                            aipp_comm.copy_ubuf_to_gm_tail(
                                                ib, dtype, output_buf, output_ub_buf,
                                                tail_ub_buf,
                                                tail_h*tiling_w*C0, block_index*offset + output_offset,
                                                0)

                                        ib.emit(tvm.call_extern(
                                            dtype, 'copy_ubuf_to_gm',
                                            output_buf.access_ptr(
                                                "w", ptr_type=dtype,
                                                offset=block_index*offset + output_offset),
                                            output_ub_buf.access_ptr(
                                                "rw", ptr_type=dtype, offset=0),
                                            0, nBurst, lenBurst, 0, 0))
                                    else:
                                        #if H has tail, W can not have tail
                                        with ib.for_range(0, tail_h,
                                                          name="h2",
                                                          dtype="uint64") as h2:
                                            output_offset = \
                                                n1*C1*H*W*C0 + \
                                                c1_index*H*W*C0 + \
                                                (tiling_h*h_loop)*W*C0 + \
                                                (w1*tiling_w)*C0 + h2*W*C0
                                            lenBurst, nBurst = \
                                                aipp_comm.get_lenBurst_and_nBurst(
                                                    (1*tiling_w*C0*size + 31)//32, 1)

                                            if output_format == "NC1HWC0_C04" and 1*tiling_w*C0*size % 32 != 0:
                                                lenBurst, nBurst = \
                                                    aipp_comm.get_lenBurst_and_nBurst(
                                                        (tail_h*tiling_w*C0*size)//32, 1)

                                                tail_ub = ib.allocate(dtype, (32 // size,), "tail_ub", scope=cce.scope_ubuf)
                                                tail_ub_buf = tvm.decl_buffer((32 // size,), dtype, "tail_ub_buf",
                                                                              scope=cce.scope_ubuf, data=tail_ub)
                                                aipp_comm.copy_ubuf_to_gm_tail(
                                                    ib, dtype, output_buf, output_ub_buf,
                                                    tail_ub_buf,
                                                    1*tiling_w*C0, block_index*offset + output_offset,
                                                    0)

                                            ib.emit(tvm.call_extern(
                                                dtype, 'copy_ubuf_to_gm',
                                                output_buf.access_ptr(
                                                    "w", ptr_type=dtype,
                                                    offset=block_index*offset +
                                                           output_offset),
                                                output_ub_buf.access_ptr(
                                                    "rw", ptr_type=dtype,
                                                    offset=h2*tiling_w*C0),
                                                0, nBurst, lenBurst, 0, 0))

        _aipp_intrin()
        return ib.get()

    return tvm.extern([(N, C1, H, W, C0)], [input_tensor],
                      lambda ins, outs: aipp_ir(ins[0], outs[0]),
                      dtype=[dtype], name="aipp")


@check_op_params(REQUIRED_INPUT, OPTION_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_STR, KERNEL_NAME)
def aipp(input_data, input_dync_param, output_data, aipp_config_json, kernel_name="aipp"):
    """Operation for aipp.
    Parameters
    ----------
    input_data: dict of input, include shape and dtype, dtype support uint8

    input_dync_param: dict of dynamic parameter,
    include shape and dtype, dtype support uint8

    aipp_config_json : json of aipp config

    kernel_name : cce kernel name, default value is "aipp"

    Returns
    -------
        None
    """

    if aipp_config_json == "{}" or aipp_config_json == "":
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_001
        error_info['param_name'] = 'aipp_config_json'
        error_info['op_name'] = 'aipp'
        raise RuntimeError(error_info, "In op[%s], the mandatory parameter[%s] is missed."
                           % (error_info['op_name'], error_info['param_name']))

    aipp_config = json.loads(aipp_config_json)

    if 'aipp_mode' not in aipp_config:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_001
        error_info['param_name'] = 'aipp_mode'
        error_info['op_name'] = 'aipp'
        raise RuntimeError(error_info, "In op[%s], the mandatory parameter[%s] is missed."
                           % (error_info['op_name'], error_info['param_name']))

    aipp_mode = aipp_config.get('aipp_mode')
    if aipp_mode not in ['static', 'dynamic']:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_000
        error_info['param_name'] = 'aipp_mode'
        error_info['op_name'] = 'aipp'
        error_info['expect_value'] = "static, dynamic"
        error_info['real_value'] = aipp_mode
        raise RuntimeError(error_info, "In op[%s], the parameter[%s] should be one of "
                                       "[%s], but actually is [%s]."
                           % (error_info['op_name'], error_info['param_name'], \
                              error_info['expect_value'], error_info['real_value']))

    input_shape = input_data.get('shape')
    input_dtype = input_data.get('dtype')
    input_format = input_data.get('format')
    output_dtype = output_data.get('dtype')
    output_shape = output_data.get('shape')
    output_format = output_data.get('format')

    check_shape(input_shape, param_name="input_data")

    check_list = ["float16", "uint8", "int8"]
    check_dtype(output_dtype.lower(), check_list, param_name="output")

    input_format_list = ["NCHW", "NHWC", "NC1HWC0_C04"]
    check_format(input_format, input_format_list, param_name="input")

    cur_cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")

    if output_format == "NC1HWC0_C04":
        if cur_cce_product not in ["Ascend610", "Ascend710", "Ascend615", "Hi3796CV300CS"]:
            cause_desc = "output_format is NC1HWC0_C04 only support Ascend610, Ascend710, " \
                         "Ascend615, Hi3796CV300CS!"
            aipp_comm.raise_RuntimeError(cause_desc)
        if output_shape[4] != 4:
            cause_dec = "when output_format is NC1HWC0_C04, C0[%d] must be 4" % \
                        output_shape[4]
            aipp_comm.raise_RuntimeError(cause_dec)

        if output_shape[2] < 8 or output_shape[3] < 8:
            cause_dec = "when output_format is NC1HWC0_C04, H[%d] and W[%d] of output " \
                        "must be greater or equal than 8!" % (output_shape[2], output_shape[3])
            aipp_comm.raise_RuntimeError(cause_dec)

    if aipp_mode == 'dynamic':
        input_dync_param_shape = input_dync_param.get('shape')
        input_dync_param_dtype = input_dync_param.get('dtype')
        check_shape(input_dync_param_shape, param_name="input_dync_param")

        check_dtype(input_dync_param_dtype, ["uint8"], param_name="input_dync_param_dtype")
        if cur_cce_product not in ["Ascend310", "Ascend610", "Ascend710", "Ascend615",
                                   "Hi3796CV300ES", "Hi3796CV300CS"]:
            cause_desc = "dynamic aipp only support " \
                         "Ascend310, Ascend610, Ascend710, Ascend615, " \
                         "Hi3796CV300ES and Hi3796CV300CS"
            aipp_comm.raise_RuntimeError(cause_desc)

        if output_format == "NC1HWC0_C04":
            cause_desc = "dynmic aipp single op not support " \
                         "NC1HWC0_C04 output_format"
            aipp_comm.raise_RuntimeError(cause_desc)

        # Compute
        data = tvm.placeholder(input_shape, name='input', dtype=input_dtype.lower())
        param = tvm.placeholder(input_dync_param_shape, name='param', dtype=input_dync_param_dtype)
        output = aipp_dynamic.aipp_compute(data, param, input_shape, input_format, output_data)

        # Schedule
        s = tvm.create_schedule([output.op])
        with build_config:
            tvm.build(s, [data, param, output], "cce", name=kernel_name)
    else:
        aipp_comm.set_aipp_default_params(aipp_config)
        aipp_comm.check_aipp_static_config(input_shape, input_format, output_data,
                                           aipp_config, cur_cce_product)

        if aipp_config.get("input_format") == "NC1HWC0DI_S8":
            if input_dtype != "int8":
                cause_desc = "when input_format is NC1HWC0DI_S8, the input dtype must be int8, " \
                             "but actually is %s" % input_dtype
                aipp_comm.raise_RuntimeError(cause_desc)

            if output_dtype != "int8":
                cause_desc = "when input_format is NC1HWC0DI_S8, the output dtype must be int8, " \
                             "but actually is %s" % output_dtype
                aipp_comm.raise_RuntimeError(cause_desc)
        elif aipp_config.get("input_format") == "NC1HWC0DI_FP16":
            if input_dtype != "float16":
                cause_desc = "when input_format is NC1HWC0DI_FP16, the input dtype must be float16, " \
                             "but actually is %s" % input_dtype
                aipp_comm.raise_RuntimeError(cause_desc)

            if output_dtype != "float16":
                cause_desc = "when input_format is NC1HWC0DI_FP16, the output dtype must be float16, " \
                             "but actually is %s" % output_dtype
                aipp_comm.raise_RuntimeError(cause_desc)
        elif aipp_config.get("input_format") in ["RAW10", "RAW12",
                                                 "RAW16", "uint16"]:
            if input_dtype != "uint16":
                cause_desc = "when input_format is %s, the input dtype must be uint16, " \
                             "but actually is %s" % (aipp_config.get("input_format"), input_dtype)
                aipp_comm.raise_RuntimeError(cause_desc)
        elif aipp_config.get("input_format") in ["RGB16", "RGB16_IR"]:
            if input_dtype != "uint16":
                cause_desc = "when input_format is %s, the input dtype must be uint16, " \
                             "but actually is %s" % (aipp_config.get("input_format"), input_dtype)
                aipp_comm.raise_RuntimeError(cause_desc)
            if output_dtype != "float16":
                cause_desc = "when input_format is %s, the output dtype must be float16, " \
                             "but actually is %s" % (aipp_config.get("input_format"), output_dtype)
                aipp_comm.raise_RuntimeError(cause_desc)
        elif aipp_config.get("input_format") in ["RGB20", "RGB24", "RGB24_IR"]:
            if input_dtype != "uint32":
                cause_desc = "when input_format is %s, the input dtype must be uint32, " \
                             "but actually is %s" % (aipp_config.get("input_format"), input_dtype)
                aipp_comm.raise_RuntimeError(cause_desc)
            if output_dtype != "float16":
                cause_desc = "when input_format is %s, the output dtype must be float16, " \
                             "but actually is %s" % (aipp_config.get("input_format"), output_dtype)
                aipp_comm.raise_RuntimeError(cause_desc)
        else:
            if input_dtype != "uint8":
                cause_desc = "when input_format is %s, the input dtype must be uint8, " \
                             "but actually is %s" % (aipp_config.get("input_format"), input_dtype)
                aipp_comm.raise_RuntimeError(cause_desc)

        # Compute
        data = tvm.placeholder(input_shape, name='input',
                               dtype=input_dtype.lower())
        if ("padding" in aipp_config and \
            aipp_config.get("padding") == 1) or \
                ("resize" in aipp_config and \
                 aipp_config.get("resize") == 1):
            output = aipp_resize_padding.aipp_compute(data, input_shape,
                                                      input_format, output_data,
                                                      aipp_config)
        else:
            output = aipp_compute_single(data, input_shape, input_format,
                                  output_data, aipp_config)

        # Schedule
        s = tvm.create_schedule([output.op])
        with build_config:
            tvm.build(s, [data, output], "cce", name=kernel_name)

        if ("resize" in aipp_config and aipp_config.get("resize") == 1) or \
           ("padding" in aipp_config and aipp_config.get("padding") == 1) or \
           aipp_config.get("input_format") in ["RGB16", "RGB20", "RGB24",
                                               "RGB8_IR", "RGB1_IR",
                                               "RGB24_IR"]:
            fusion_manager.set_current_op_pattern("Opaque")
        else:
            fusion_manager.set_current_op_pattern("aipp")
