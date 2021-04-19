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
max_pool_grad_with_argmax_cut_h_v2
"""
import math
from te import tik
from te.platform.cce_conf import CceProductParams
from impl import common_util
from impl import constant_util as constant

# size of vector calc one repeat
ONE_REPEAT = 256
# max repeat of vector calc
V_MAX_REPEAT = 255
# max num of fp16 in one repeat
FP16_MAX = 128
# max num of fp32 in one repeat
FP32_MAX = 64
# max num of fp16 mask handle one time
MASK_MAX = 8


# pylint: disable=locally-disabled,too-few-public-methods,too-many-instance-attributes,too-many-lines
class MaxpoolGradBase():
    """
    parameter for max_pool_grad_with_pool
    """

    # pylint: disable=locally-disabled,too-many-locals,too-many-arguments,too-many-statements
    def __init__(self, grad, argmax, input_x, ksize, strides, padding, dilation, ceil_mode):
        """
        init compare and bit pack base parameters
        Parameters
        ----------
        input_x: input of maxpool, useless for maxpool gard
        grad: input of maxpoolgard or output of maxpool
        argmax:output of maxpool mask or index
        strides: stride , minimum length is 4, just like [1, poolingStrideH, poolingStrideW, 1]
        padding: pad mode, just support "SANME" or "VALID"
        Returns
        -------
        None
        """
        self.blocknum = tik.Dprofile().get_aicore_num()
        self.ub_size = CceProductParams().getParams("Unified_Buffer")

        self.input_gard_shape = grad.get("shape")
        self.argmax_shape = argmax.get("shape")
        self.y_shape = input_x.get("shape")
        self.dtype = grad.get("dtype").lower()
        self.dtype_size = common_util.get_data_size(self.dtype)
        self.nc1 = 1
        self.block = self.input_gard_shape[0] * self.input_gard_shape[1]
        self.tik_instance = tik.Tik()
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

        pad_h, pad_w = self.padding[1:3]
        dxh, dxw = self.y_shape[2:4]
        strideh, stridew = self.strides[1:3]
        stridehw = strideh * stridew
        if strideh > dxh:
            strideh = dxh

        if stridew > dxw:
            stridew = dxw

        if stridehw == 1:
            self.ub_limit = self.ub_size // 8
        elif stridehw < 4:
            self.ub_limit = self.ub_size // 6
        else:
            self.ub_limit = self.ub_size // 4

        kernelh, kernelw = self.ksize[1:3]

        if self.ceil_mode == False:
            pad_top = pad_h
            pad_bottom = pad_h
            pad_left = pad_w
            pad_right = pad_w
        else:
            pad_top = pad_h
            pad_bottom = pad_h + strideh - 1
            pad_left = pad_w
            pad_right = pad_w + stridew - 1

        self.pad = (pad_top, pad_bottom, pad_left, pad_right)
        self.offset_w = dxw
        self.offset_h = dxh
        self.hoverlap = 0
        if kernelh > strideh:
            self.hoverlap = kernelh - strideh
        self.woverlap = 0
        if kernelw > stridew:
            self.woverlap = kernelw - stridew

    def change_blocknum(self, blocknum):
        """
        change blocknum
        Parameters
        ----------
        blocknum: blocknum
        Returns
        -------
        None
        """
        self.blocknum = blocknum

    def get_block_param(self, block):
        """
        get_block_param
        Parameters
        ----------
        block: block
        Returns
        -------
        None
        """
        if self.blocknum > block:
            real_block = block
            block_cycle = 1
            block_index = 0
        else:
            real_block = self.blocknum
            block_cycle = block // real_block
            block_index = block % real_block
        return real_block, block_cycle, block_index

    # pylint: disable=locally-disabled,too-many-arguments,too-many-locals,too-many-statements,too-many-branches
    def tik_instance_cut_nc1_cut_h(self, kernel_name):
        """
        function for max_pool_grad_with_pool calc for normal shape
        """
        batch, channel1, dyh, dyw, channel = self.input_gard_shape
        dxh, dxw = self.y_shape[2:4]
        strideh, stridew = self.strides[1:3]
        if strideh > dxh:
            strideh = dxh

        if stridew > dxw:
            stridew = dxw

        dtype = self.dtype
        dtype_size = self.dtype_size
        kernelh, kernelw = self.ksize[1:3]
        block = self.block
        nc1 = self.nc1
        pad_top = self.pad[0]
        pad_left = self.pad[2]
        hoverlap = self.hoverlap
        woverlap = self.woverlap
        dialate_h = self.dilation[1]
        dialate_w = self.dilation[2]
        wo_max = math.ceil(dyw / 16) * 16

        ho_max = 1 if hoverlap == 0 else (dialate_h * (kernelh - 1) + strideh) // strideh
        col2img_w = wo_max * stridew if woverlap == 0 else (wo_max - 1) * stridew + dialate_w * kernelw
        if col2img_w < dxw:
            col2img_w = dxw

        col2img_h = ho_max * strideh if hoverlap == 0 else (ho_max - 1) * strideh + dialate_h * kernelh
        flag = 0
        while col2img_w * col2img_h * channel * dtype_size < self.ub_limit and ho_max <= dyh:
            ho_max += 1
            col2img_h = ho_max * strideh if hoverlap == 0 else (ho_max - 1) * strideh + dialate_h * kernelh
            flag = 1
        if flag == 1:
            ho_max -= 1
            col2img_h = ho_max * strideh if hoverlap == 0 else (ho_max - 1) * strideh + dialate_h * kernelh

        if hoverlap == 0:
            h_cycle = math.ceil(dyh / ho_max)
        else:
            h_cycle = math.ceil((dyh - 1) / (ho_max - 1))

        if dyh == 1:
            ho_max = 1
            h_cycle = 1

        mask_one_window = ((dyh * dyw + 15) // 16 + 1) * 16
        # vector_repeat_time
        v_rep_time = ho_max * wo_max * channel * dtype_size // ONE_REPEAT
        v_rep_cycle_fp32 = 2 * v_rep_time // V_MAX_REPEAT
        # v_rep_last
        v_rep_last_fp32 = 2 * v_rep_time % V_MAX_REPEAT

        v_rep_time_col = (2 * (col2img_w * channel * col2img_h + 64) * dtype_size) // ONE_REPEAT
        v_rep_cycle_col = v_rep_time_col // V_MAX_REPEAT
        v_rep_last_col = v_rep_time_col % V_MAX_REPEAT

        data_input = self.tik_instance.Tensor(dtype, self.input_gard_shape, name="data_input",
                                              scope=tik.scope_gm)
        data_mask = self.tik_instance.Tensor("uint16", (batch * channel1 * kernelh * kernelw *
                                                        mask_one_window,), name="data_mask", scope=tik.scope_gm)
        data_output = self.tik_instance.Tensor(dtype, self.y_shape, name="data_output", scope=tik.scope_gm)

        data_input_origin = self.tik_instance.Tensor(dtype, self.y_shape, name="data_input_origin",
                                                     scope=tik.scope_gm)

        real_block, block_cycle, block_index = self.get_block_param(block)

        with self.tik_instance.for_range(0, real_block, block_num=real_block) as block_id:  # 1
            real_cycle = self.tik_instance.Scalar("int32")
            block_base = self.tik_instance.Scalar("int32")
            block_num = self.tik_instance.Scalar("int32")
            with self.tik_instance.if_scope(block_id < block_index):
                real_cycle.set_as(block_cycle + 1)
                block_base.set_as(block_id * real_cycle)
            with self.tik_instance.else_scope():
                real_cycle.set_as(block_cycle)
                block_base.set_as(block_index + block_id * block_cycle)
            with self.tik_instance.for_range(0, real_cycle) as cycle_id:  # 1
                block_num.set_as(block_base + cycle_id)
                data_vsel_ub_zero = self.tik_instance.Tensor(dtype, (128,), name="data_vsel_ub_zero",
                                                             scope=tik.scope_ubuf)

                # data move only 16 fp16 every time
                self.tik_instance.data_move(data_vsel_ub_zero[0], data_input_origin[0],
                                            constant.SID, constant.DEFAULT_NBURST,
                                            constant.DEFAULT_BURST_LEN, constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)

                # clear to zeros
                self.clean_fp16_one_repeat(data_vsel_ub_zero, dtype)

                with self.tik_instance.for_range(0, nc1) as loopc1:  # 1
                    # vector_dup ub every time
                    dxh_address_offset = self.tik_instance.Scalar("int32")
                    dxh_address_offset.set_as(0)
                    # 16x6x16
                    data_max_ub = self.tik_instance.Tensor(dtype, (ho_max * wo_max * channel,),
                                                           name="data_max_ub", scope=tik.scope_ubuf)
                    # 33x13x16+64
                    data_vmul_ub_col2img_fp32 = \
                        self.tik_instance.Tensor("float32", (col2img_w * channel * col2img_h + 64,),
                                                 name="data_vmul_ub_col2img_fp32", scope=tik.scope_ubuf)
                    # 33x13x16+128
                    data_vmul_ub_col2img_fp16 = \
                        self.tik_instance.Tensor(dtype, (col2img_w * channel * col2img_h + 128,),
                                                 name="data_vmul_ub_col2img_fp16", scope=tik.scope_ubuf)
                    # mask define 6x16
                    data_mask_ub = self.tik_instance.Tensor("uint16", (ho_max * wo_max,),
                                                            name="data_mask_ub", scope=tik.scope_ubuf)

                    new_looph = self.tik_instance.Scalar("int32")
                    new_looph.set_as(0)
                    in_nburst = self.tik_instance.Scalar("int32")
                    in_nburst.set_as(ho_max)
                    with self.tik_instance.for_range(0, h_cycle) as looph:  # 1
                        if hoverlap == 0:
                            new_looph.set_as(looph * ho_max)
                            with self.tik_instance.if_scope(looph == h_cycle - 1):
                                in_nburst.set_as(dyh - looph * ho_max)
                        else:
                            with self.tik_instance.if_scope(looph != 0):
                                new_looph.set_as(looph * (ho_max - 1))
                            with self.tik_instance.if_scope(looph == h_cycle - 1):
                                in_nburst.set_as(dyh - looph * (ho_max - 1))

                        # clearn to zeors
                        self.clean_fp32_multi_repeat(data_vmul_ub_col2img_fp32, dtype_size * 2)
                        self.clean_max_ub(data_max_ub, dtype)

                        # 一次搬入
                        self.tik_instance.data_move(data_max_ub,
                                                    data_input[(block_num * nc1 * dyh + loopc1 * dyh +
                                                                new_looph) * dyw * channel],
                                                    constant.SID, in_nburst, dyw, constant.STRIDE_ZERO, wo_max - dyw)

                        with self.tik_instance.for_range(0, kernelh * kernelw) as mask_id:  # 3 x 3
                            with self.tik_instance.for_range(0, in_nburst) as cycle:  # 6
                                # mask copy gm to ub 6 x 16
                                self.tik_instance.data_move(data_mask_ub[cycle * wo_max], data_mask[
                                    block_num * nc1 * mask_one_window * kernelw * kernelh +
                                    loopc1 * mask_one_window * kernelw * kernelh + (new_looph + cycle) * dyw +
                                    mask_one_window * mask_id], constant.SID, 1, wo_max // 16, 0, 0)

                            # 16 x 6 x 16
                            data_vsel_ub = self.tik_instance.Tensor(dtype, (ho_max * wo_max * channel,),
                                                                    name="data_vsel_ub", scope=tik.scope_ubuf)
                            data_vsel_ub_fp32 = self.tik_instance.Tensor("float32", (ho_max * wo_max * channel,),
                                                                         name="data_vsel_ub_fp32",
                                                                         scope=tik.scope_ubuf)
                            if v_rep_time > 0:  # 12
                                with self.tik_instance.for_range(0, v_rep_time, thread_num=1) as cycle:
                                    # 一次抽取8个mask，用来比较128个数
                                    cmpmask = self.tik_instance.mov_tensor_to_cmpmask(data_mask_ub[cycle * MASK_MAX])

                                    # 单次计算128个数， 16x6x16->12x128
                                    self.tik_instance.vsel(constant.MASK128, 0, data_vsel_ub[cycle * FP16_MAX],
                                                           cmpmask, data_max_ub[cycle * FP16_MAX],
                                                           data_vsel_ub_zero[0], constant.REPEAT_TIME_ONCE,
                                                           constant.STRIDE_ONE, constant.STRIDE_ONE,
                                                           constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                                                           constant.REPEAT_STRIDE_EIGHT,
                                                           constant.REPEAT_STRIDE_EIGHT)

                            # fp16 to fp32
                            if v_rep_cycle_fp32 > 0:  # 0
                                with self.tik_instance.for_range(0, v_rep_cycle_fp32, thread_num=1) as cycle:
                                    self.tik_instance.vconv(constant.MASK64, "",
                                                            data_vsel_ub_fp32[cycle * V_MAX_REPEAT * FP32_MAX],
                                                            data_vsel_ub[cycle * V_MAX_REPEAT * FP32_MAX],
                                                            V_MAX_REPEAT, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                                            constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_FOUR)

                            if v_rep_last_fp32 != 0:  # 24 x 64
                                self.tik_instance.vconv(constant.MASK64, "",
                                                        data_vsel_ub_fp32[v_rep_cycle_fp32 * V_MAX_REPEAT * FP32_MAX],
                                                        data_vsel_ub[v_rep_cycle_fp32 * V_MAX_REPEAT * FP32_MAX],
                                                        v_rep_last_fp32, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                                        constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_FOUR)

                            # col2img
                            fetch_filter_w = mask_id % kernelw
                            fetch_filter_h = mask_id // kernelw
                            left_top_w = 0
                            left_top_h = 0

                            self.tik_instance.col2img(data_vmul_ub_col2img_fp32[0],  # 33x13x16 + 64
                                                      data_vsel_ub_fp32[0],  # 16x6x16
                                                      (0, 0, 0, 0), col2img_h, col2img_w, fetch_filter_w,
                                                      fetch_filter_h, left_top_w, left_top_h,
                                                      stridew, strideh, kernelw, kernelh, self.dilation[1],
                                                      self.dilation[2], ho_max * wo_max // 16)

                        # convert fp32 to fp16
                        if v_rep_cycle_col > 0:  # 0
                            with self.tik_instance.for_range(0, v_rep_cycle_col) as cycle:
                                self.tik_instance.vconv(constant.MASK64, "",
                                                        data_vmul_ub_col2img_fp16[cycle * V_MAX_REPEAT * FP32_MAX],
                                                        data_vmul_ub_col2img_fp32[cycle * V_MAX_REPEAT * FP32_MAX],
                                                        V_MAX_REPEAT, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                                        constant.REPEAT_STRIDE_FOUR, constant.REPEAT_STRIDE_EIGHT)

                        if v_rep_last_col != 0:  # 108 x 64 = 6912
                            self.tik_instance.vconv(constant.MASK64, "", \
                                                    data_vmul_ub_col2img_fp16[
                                                        v_rep_cycle_col * V_MAX_REPEAT * FP32_MAX], \
                                                    data_vmul_ub_col2img_fp32[
                                                        v_rep_cycle_col * V_MAX_REPEAT * FP32_MAX], \
                                                    v_rep_last_col, constant.STRIDE_ONE, constant.STRIDE_ONE, \
                                                    constant.REPEAT_STRIDE_FOUR, constant.REPEAT_STRIDE_EIGHT)

                        # move ub to gm
                        output_cuthline = self.tik_instance.Scalar("int32")
                        output_cuthline.set_as(0)
                        src_address = self.tik_instance.Scalar("int32")
                        src_address.set_as(pad_left * channel)
                        if hoverlap == 0:
                            output_cuthline.set_as(col2img_h)
                            with self.tik_instance.if_scope(looph == 0):
                                src_address.set_as(src_address + pad_top * col2img_w * channel)
                                output_cuthline.set_as(output_cuthline - pad_top)
                                with self.tik_instance.if_scope(looph == h_cycle - 1):
                                    output_cuthline.set_as(dxh)
                            with self.tik_instance.else_scope():
                                with self.tik_instance.if_scope(looph == h_cycle - 1):
                                    output_cuthline.set_as(dxh - col2img_h * (h_cycle - 1) + pad_top)
                        else:
                            with self.tik_instance.if_scope(looph == 0):
                                output_cuthline.set_as(strideh * ho_max - pad_top)
                                src_address.set_as(src_address + pad_top * col2img_w * channel)
                                with self.tik_instance.if_scope(looph == h_cycle - 1):
                                    output_cuthline.set_as(dxh)
                            with self.tik_instance.else_scope():
                                output_cuthline.set_as(strideh * (ho_max - 1))
                                src_address.set_as(src_address + strideh * col2img_w * channel)
                                with self.tik_instance.if_scope(looph == h_cycle - 1):
                                    output_cuthline.set_as(dxh - strideh * ho_max - (h_cycle - 2) *
                                                           strideh * (ho_max - 1) + pad_top)

                        idx = block_num * nc1 * dxh * dxw * channel + loopc1 * dxh * dxw * \
                              channel + dxh_address_offset

                        self.tik_instance.data_move(data_output[idx],
                                                    data_vmul_ub_col2img_fp16[src_address],  # 544
                                                    constant.SID, output_cuthline, self.offset_w,  # 12, 12
                                                    col2img_w - self.offset_w,  # 21
                                                    dxw - self.offset_w)  # 0

                        dxh_address_offset.set_as(dxh_address_offset + output_cuthline * dxw * channel)

        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=(data_input_origin, data_input, data_mask),
                                   outputs=(data_output), enable_l2=False)
        return self.tik_instance

    # pylint: disable=locally-disabled,too-many-arguments,too-many-locals,too-many-statements,too-many-branches
    def tik_instance_cut_nc1h_cut_h(self, kernel_name):
        """
        function for max_pool_grad_with_pool calc for normal shape
        """
        batch, channel1, dyh, dyw, channel = self.input_gard_shape
        dxh, dxw = self.y_shape[2:4]
        strideh, stridew = self.strides[1:3]
        if strideh > dxh:
            strideh = dxh

        if stridew > dxw:
            stridew = dxw

        dtype = self.dtype
        dtype_size = self.dtype_size
        kernelh, kernelw = self.ksize[1:3]
        pad_top, pad_bottom, pad_left = self.pad[0:3]
        hoverlap = self.hoverlap
        woverlap = self.woverlap
        dialate_h = self.dilation[1]
        dialate_w = self.dilation[2]

        ho_count = math.ceil(self.blocknum // (batch * channel1))
        if hoverlap == 0:
            ho_every = dyh // ho_count  # 迭代次数
            ho_last = dyh - ho_every * (ho_count - 1)  # 剩余次数
        else:
            ho_every = (dyh + ho_count - 1) // ho_count
            if ho_every == 1:
                ho_count = ho_count // 2
                ho_every = (dyh + ho_count - 1) // ho_count
            ho_last = dyh + ho_count - 1 - ho_every * (ho_count - 1)
        all_blocknum = ho_count * batch * channel1

        wo_max = math.ceil(dyw / 16) * 16
        col2img_w = wo_max * stridew if woverlap == 0 else (wo_max - 1) * stridew + dialate_w * kernelw

        if col2img_w < dxw:
            col2img_w = dxw
        ho_max_every = 1 if hoverlap == 0 else 2
        ho_max_last = ho_max_every
        flag_every = 0
        col2img_h_every = ho_max_every * strideh if \
            hoverlap == 0 else (ho_max_every - 1) * strideh + dialate_h * kernelh
        while col2img_w * col2img_h_every * channel * dtype_size < self.ub_limit \
                and col2img_h_every <= ho_every:
            ho_max_every += 1
            col2img_h_every = ho_max_every * strideh if \
                hoverlap == 0 else (ho_max_every - 1) * strideh + dialate_h * kernelh
            flag_every = 1
        if flag_every == 1:
            ho_max_every -= 1
            col2img_h_every = ho_max_every * strideh if \
                hoverlap == 0 else (ho_max_every - 1) * strideh + dialate_h * kernelh

        flag_last = 0
        col2img_h_last = ho_max_last * strideh if \
            hoverlap == 0 else (ho_max_last - 1) * strideh + dialate_h * kernelh
        while col2img_w * col2img_h_last * channel * dtype_size < self.ub_limit \
                and ho_max_last <= ho_last:
            ho_max_last += 1
            col2img_h_last = ho_max_last * strideh if \
                hoverlap == 0 else (ho_max_last - 1) * strideh + dialate_h * kernelh
            flag_last = 1
        if flag_last == 1:
            ho_max_last -= 1
            col2img_h_last = ho_max_last * strideh if \
                hoverlap == 0 else (ho_max_last - 1) * strideh + dialate_h * kernelh

        if hoverlap == 0:
            h_cycle_every = math.ceil(ho_every / ho_max_every)
            h_cycle_last = math.ceil(ho_last / ho_max_last)
        else:
            h_cycle_every = (ho_every - 1 - ho_max_every) // (ho_max_every - 1) + 2
            h_cycle_last = (ho_last - 1 - ho_max_last) // (ho_max_last - 1) + 2

        mask_one_window = ((dyh * dyw + 15) // 16 + 1) * 16
        # vector_repeat_time
        v_rep_time_last = ho_max_last * wo_max * channel * dtype_size // ONE_REPEAT
        v_rep_cycle_fp32_last = 2 * v_rep_time_last // V_MAX_REPEAT
        v_rep_last_fp32_last = 2 * v_rep_time_last % V_MAX_REPEAT
        v_rep_time_col_last = (2 * (col2img_w * channel * col2img_h_last + 64) *
                               dtype_size) // ONE_REPEAT
        v_rep_cycle_col_last = v_rep_time_col_last // V_MAX_REPEAT
        v_rep_last_col_last = v_rep_time_col_last % V_MAX_REPEAT
        v_rep_time_every = ho_max_every * wo_max * channel * dtype_size // ONE_REPEAT
        v_rep_cycle_fp32_every = 2 * v_rep_time_every // V_MAX_REPEAT
        v_rep_last_fp32_every = 2 * v_rep_time_every % V_MAX_REPEAT
        v_rep_time_col_every = (2 * (col2img_w * channel * col2img_h_every + 64) *
                                dtype_size) // ONE_REPEAT
        v_rep_cycle_col_every = v_rep_time_col_every // V_MAX_REPEAT
        v_rep_last_col_every = v_rep_time_col_every % V_MAX_REPEAT

        data_input = self.tik_instance.Tensor(dtype, self.input_gard_shape, name="data_input",
                                              scope=tik.scope_gm)
        data_mask = self.tik_instance.Tensor("uint16", (batch * channel1 * kernelh * kernelw *
                                                        mask_one_window,),
                                             name="data_mask", scope=tik.scope_gm)

        data_output = self.tik_instance.Tensor(dtype, self.y_shape, name="data_output",
                                               scope=tik.scope_gm)
        data_input_origin = self.tik_instance.Tensor(dtype, self.y_shape, name="data_input_origin",
                                                     scope=tik.scope_gm)

        real_block, block_cycle, block_index = self.get_block_param(all_blocknum)
        with self.tik_instance.for_range(0, real_block, block_num=real_block) as block_id:  # 32
            real_cycle = self.tik_instance.Scalar("int32")
            block_base = self.tik_instance.Scalar("int32")
            block_num = self.tik_instance.Scalar("int32")
            with self.tik_instance.if_scope(block_id < block_index):
                real_cycle.set_as(block_cycle + 1)
                block_base.set_as(block_id * real_cycle)
            with self.tik_instance.else_scope():
                real_cycle.set_as(block_cycle)
                block_base.set_as(block_index + block_id * block_cycle)

            with self.tik_instance.for_range(0, real_cycle) as cycle_id:  # 1
                block_num.set_as(block_base + cycle_id)
                data_vsel_ub_zero = self.tik_instance.Tensor(dtype, (128,),
                                                             name="data_vsel_ub_zero",
                                                             scope=tik.scope_ubuf)
                self.tik_instance.data_move(data_vsel_ub_zero[0],
                                            data_input_origin[0],
                                            constant.SID,
                                            constant.DEFAULT_NBURST,
                                            constant.DEFAULT_BURST_LEN,
                                            constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)
                self.clean_fp16_one_repeat(data_vsel_ub_zero, dtype)

                block_batch = self.tik_instance.Scalar("int32")
                block_batch.set_as(block_num // ho_count)  # 0
                block_h = self.tik_instance.Scalar("int32")
                block_h.set_as(block_num % ho_count)  # 0
                with self.tik_instance.if_scope(block_h == ho_count - 1):
                    data_max_ub = self.tik_instance.Tensor(dtype, (ho_max_last * wo_max * channel,),
                                                           name="data_max_ub",
                                                           scope=tik.scope_ubuf)
                    data_vmul_ub_col2img_fp32 = \
                        self.tik_instance.Tensor("float32",
                                                 (col2img_w * channel * col2img_h_last + 64,),
                                                 name="data_vmul_ub_col2img_fp32",
                                                 scope=tik.scope_ubuf)
                    data_vmul_ub_col2img_fp16 = \
                        self.tik_instance.Tensor(dtype,
                                                 (col2img_w * channel * col2img_h_last + 128,),
                                                 name="data_vmul_ub_col2img_fp16",
                                                 scope=tik.scope_ubuf)
                    # mask define
                    data_mask_ub = self.tik_instance.Tensor("uint16", (ho_max_last * wo_max,),
                                                            name="data_mask_ub",
                                                            scope=tik.scope_ubuf)

                    new_looph = self.tik_instance.Scalar("int32")
                    new_looph.set_as(0)
                    in_nburst = self.tik_instance.Scalar("int32")
                    in_nburst.set_as(ho_max_last)
                    in_src_address = self.tik_instance.Scalar("int32")
                    mask_address = self.tik_instance.Scalar("int32")
                    with self.tik_instance.for_range(0, h_cycle_last) as looph:  # 4
                        in_src_address.set_as(block_batch * dyh * dyw * channel)
                        mask_address.set_as(block_batch * mask_one_window * kernelw * kernelh)  # 56 x 56 x 3 x 3
                        if hoverlap == 0:
                            new_looph.set_as(looph * ho_max_last)
                            with self.tik_instance.if_scope(looph == h_cycle_last - 1):
                                in_nburst.set_as(ho_last - looph * ho_max_last)
                            in_src_address.set_as(in_src_address +
                                                  (block_h * ho_every + new_looph) * dyw * channel)
                            mask_address.set_as(mask_address +
                                                (block_h * ho_every + new_looph) * dyw)
                        else:
                            with self.tik_instance.if_scope(looph != 0):
                                new_looph.set_as(looph * (ho_max_last - 1))
                            with self.tik_instance.if_scope(looph == h_cycle_last - 1):
                                in_nburst.set_as(ho_last - looph * (ho_max_last - 1))
                            in_src_address.set_as(in_src_address + (block_h * (ho_every - 1) +
                                                                    new_looph) * dyw * channel)
                            mask_address.set_as(mask_address + (block_h * (ho_every - 1) +
                                                                new_looph) * dyw)

                        self.clean_fp32_multi_repeat(data_vmul_ub_col2img_fp32, dtype_size * 2)
                        self.clean_max_ub(data_max_ub, dtype)
                        self.tik_instance.data_move(data_max_ub,
                                                    data_input[in_src_address],
                                                    constant.SID, in_nburst,
                                                    dyw, constant.STRIDE_ZERO,
                                                    wo_max - dyw)

                        with self.tik_instance.for_range(0, kernelh * kernelw) as mask_id:  # 3x3
                            with self.tik_instance.for_range(0, in_nburst) as cycle:  # 7
                                # mask copy gm to ub
                                self.tik_instance.data_move(data_mask_ub[cycle * wo_max],
                                                            data_mask[mask_address + cycle * dyw +
                                                                      mask_one_window * mask_id],
                                                            constant.SID, 1,
                                                            wo_max // 16, 0, 0)

                            data_vsel_ub = self.tik_instance.Tensor(dtype, (ho_max_last *
                                                                            wo_max * channel,),
                                                                    name="data_vsel_ub",
                                                                    scope=tik.scope_ubuf)
                            data_vsel_ub_fp32 = self.tik_instance.Tensor("float32",
                                                                         (ho_max_last * wo_max *
                                                                          channel,),
                                                                         name="data_vsel_ub_fp32",
                                                                         scope=tik.scope_ubuf)
                            if v_rep_time_last > 0:
                                with self.tik_instance.for_range(0, v_rep_time_last, thread_num=1) as cycle:  # 56
                                    cmpmask = self.tik_instance.mov_tensor_to_cmpmask(
                                        data_mask_ub[cycle * MASK_MAX])
                                    self.tik_instance.vsel(constant.MASK128, 0,
                                                           data_vsel_ub[cycle * FP16_MAX],
                                                           cmpmask,
                                                           data_max_ub[cycle * FP16_MAX],
                                                           data_vsel_ub_zero[0],
                                                           constant.REPEAT_TIME_ONCE,
                                                           constant.STRIDE_ONE,
                                                           constant.STRIDE_ONE,
                                                           constant.STRIDE_ONE,
                                                           constant.REPEAT_STRIDE_EIGHT,
                                                           constant.REPEAT_STRIDE_EIGHT,
                                                           constant.REPEAT_STRIDE_EIGHT)

                            # fp16 to fp32
                            if v_rep_cycle_fp32_last > 0:  # 0
                                with self.tik_instance.for_range(0, v_rep_cycle_fp32_last,
                                                                 thread_num=1) as cycle:
                                    self.tik_instance.vconv(constant.MASK64, "",
                                                            data_vsel_ub_fp32[cycle * V_MAX_REPEAT * FP32_MAX],
                                                            data_vsel_ub[cycle * V_MAX_REPEAT * FP32_MAX],
                                                            V_MAX_REPEAT, constant.STRIDE_ONE,
                                                            constant.STRIDE_ONE,
                                                            constant.REPEAT_STRIDE_EIGHT,
                                                            constant.REPEAT_STRIDE_FOUR)
                            if v_rep_last_fp32_last != 0:  # 112
                                self.tik_instance.vconv(constant.MASK64, "", data_vsel_ub_fp32[
                                    v_rep_cycle_fp32_last * V_MAX_REPEAT * FP32_MAX],
                                                        data_vsel_ub[v_rep_cycle_fp32_last * V_MAX_REPEAT * FP32_MAX],
                                                        v_rep_last_fp32_last, constant.STRIDE_ONE,
                                                        constant.STRIDE_ONE,
                                                        constant.REPEAT_STRIDE_EIGHT,
                                                        constant.REPEAT_STRIDE_FOUR)
                            # col2img
                            fetch_filter_w = mask_id % kernelw
                            fetch_filter_h = mask_id // kernelw
                            left_top_w = 0
                            left_top_h = 0

                            self.tik_instance.col2img(data_vmul_ub_col2img_fp32[0],
                                                      data_vsel_ub_fp32[0],
                                                      (0, 0, 0, 0),
                                                      col2img_h_last, col2img_w, fetch_filter_w,
                                                      fetch_filter_h, left_top_w, left_top_h,
                                                      stridew, strideh, kernelw, kernelh,
                                                      dialate_w, dialate_h,
                                                      ho_max_last * wo_max // 16)

                        if v_rep_cycle_col_last > 0:  # 0
                            with self.tik_instance.for_range(0, v_rep_cycle_col_last) as cycle:
                                self.tik_instance.vconv(constant.MASK64, "",
                                                        data_vmul_ub_col2img_fp16[
                                                            cycle * V_MAX_REPEAT * FP32_MAX],
                                                        data_vmul_ub_col2img_fp32[
                                                            cycle * V_MAX_REPEAT * FP32_MAX],
                                                        V_MAX_REPEAT, constant.STRIDE_ONE,
                                                        constant.STRIDE_ONE,
                                                        constant.REPEAT_STRIDE_FOUR,
                                                        constant.REPEAT_STRIDE_EIGHT)
                        if v_rep_last_col_last != 0:  # 162
                            self.tik_instance.vconv(constant.MASK64, "",
                                                    data_vmul_ub_col2img_fp16[
                                                        v_rep_cycle_col_last * V_MAX_REPEAT *
                                                        FP32_MAX],
                                                    data_vmul_ub_col2img_fp32[
                                                        v_rep_cycle_col_last * V_MAX_REPEAT *
                                                        FP32_MAX],
                                                    v_rep_last_col_last, constant.STRIDE_ONE,
                                                    constant.STRIDE_ONE,
                                                    constant.REPEAT_STRIDE_FOUR,
                                                    constant.REPEAT_STRIDE_EIGHT)
                        # move ub to gm
                        output_cuthline = self.tik_instance.Scalar("int32")
                        output_cuthline.set_as(0)
                        src_address = self.tik_instance.Scalar("int32")
                        dst_address = self.tik_instance.Scalar("int32")
                        dst_address.set_as(block_batch * dxh * dxw * channel)

                        if hoverlap == 0:
                            src_address.set_as(pad_left * channel)
                            output_cuthline.set_as(col2img_h_last)
                            dst_address.set_as(dst_address +
                                               (ho_count - 1) * ho_every * strideh * dxw *
                                               channel + looph * ho_max_last * strideh *
                                               dxw * channel - pad_top * dxw * channel)
                            with self.tik_instance.if_scope(looph == h_cycle_last - 1):
                                output_cuthline.set_as(dxh - ho_every *
                                                       (ho_count - 1) * strideh -
                                                       looph * ho_max_last * strideh +
                                                       pad_top)
                        else:  # 1
                            src_address.set_as(pad_left * channel + strideh * col2img_w * channel)
                            output_cuthline.set_as((ho_max_last - 1) * strideh)
                            with self.tik_instance.if_scope(looph == h_cycle_last - 1):
                                output_cuthline.set_as((ho_last - looph * (ho_max_last - 1) - 1) * strideh +
                                                       kernelh - strideh - pad_bottom + 1)
                            dst_address.set_as(dst_address +
                                               ((block_h * (ho_every - 1) +
                                                 looph * (ho_max_last - 1) + 1) * strideh -
                                                pad_top) * dxw * channel)
                        self.tik_instance.data_move(data_output[dst_address],
                                                    data_vmul_ub_col2img_fp16[src_address],
                                                    constant.SID, output_cuthline, self.offset_w,  # 12 x 112
                                                    col2img_w - self.offset_w,  # 17
                                                    dxw - self.offset_w)  # 0
                with self.tik_instance.else_scope():
                    data_max_ub = self.tik_instance.Tensor(dtype,
                                                           (ho_max_every * wo_max * channel,),
                                                           name="data_max_ub",
                                                           scope=tik.scope_ubuf)  # 2 x 64 x 16
                    data_vmul_ub_col2img_fp32 = \
                        self.tik_instance.Tensor("float32",
                                                 (col2img_w * channel * col2img_h_every + 64,),  # 129 x 16 x 5
                                                 name="data_vmul_ub_col2img_fp32",
                                                 scope=tik.scope_ubuf)
                    data_vmul_ub_col2img_fp16 = \
                        self.tik_instance.Tensor(dtype,
                                                 (col2img_w * channel * col2img_h_every + 128,),  # 129 x 16 x 5
                                                 name="data_vmul_ub_col2img_fp16",
                                                 scope=tik.scope_ubuf)
                    # mask define
                    data_mask_ub = self.tik_instance.Tensor("uint16", (ho_max_every * wo_max,),  # 2 x 64
                                                            name="data_mask_ub",
                                                            scope=tik.scope_ubuf)
                    in_nburst = self.tik_instance.Scalar("int32")
                    in_nburst.set_as(ho_max_every)
                    in_src_address = self.tik_instance.Scalar("int32")
                    mask_address = self.tik_instance.Scalar("int32")
                    with self.tik_instance.for_range(0, h_cycle_every) as looph:  # 1
                        in_src_address.set_as(block_batch * dyh * dyw * channel)
                        mask_address.set_as(block_batch * mask_one_window * kernelw * kernelh)  # 1 x 3 x 3 x 56 x 56
                        if hoverlap == 0:
                            in_src_address.set_as(in_src_address +
                                                  (block_h * ho_every +
                                                   looph * ho_max_every) * dyw * channel)
                            mask_address.set_as(mask_address + (block_h * ho_every +
                                                                looph * ho_max_every) * dyw)
                            with self.tik_instance.if_scope(looph == h_cycle_every - 1):
                                in_nburst.set_as(ho_every - looph * ho_max_every)
                        else:
                            in_src_address.set_as(in_src_address +
                                                  (block_h * (ho_every - 1) +
                                                   looph * (ho_max_every - 1)) * dyw * channel)
                            mask_address.set_as(mask_address +
                                                (block_h * (ho_every - 1) +
                                                 looph * (ho_max_every - 1)) * dyw)
                            with self.tik_instance.if_scope(looph == h_cycle_every - 1):
                                in_nburst.set_as(ho_every - looph * (ho_max_every - 1))

                        self.clean_fp32_multi_repeat(data_vmul_ub_col2img_fp32, dtype_size * 2)
                        self.clean_max_ub(data_max_ub, dtype)
                        self.tik_instance.data_move(data_max_ub,
                                                    data_input[in_src_address],
                                                    constant.SID, in_nburst,
                                                    dyw, constant.STRIDE_ZERO,
                                                    wo_max - dyw)

                        with self.tik_instance.for_range(0, kernelh * kernelw) as mask_id:  # 3 x 3
                            with self.tik_instance.for_range(0, in_nburst) as cycle:  # 2
                                # mask copy gm to ub
                                self.tik_instance.data_move(data_mask_ub[cycle * wo_max],
                                                            data_mask[mask_address + cycle * dyw +
                                                                      mask_one_window * mask_id],
                                                            constant.SID, 1,
                                                            wo_max // 16, 0, 0)
                            data_vsel_ub = self.tik_instance.Tensor(dtype, (ho_max_every *
                                                                            wo_max * channel,),
                                                                    name="data_vsel_ub",
                                                                    scope=tik.scope_ubuf)
                            data_vsel_ub_fp32 = self.tik_instance.Tensor("float32",
                                                                         (ho_max_every *
                                                                          wo_max * channel,),
                                                                         name="data_vsel_ub_fp32",
                                                                         scope=tik.scope_ubuf)
                            if v_rep_time_every > 0:  # 16
                                with self.tik_instance.for_range(0, v_rep_time_every, thread_num=1) as cycle:
                                    cmpmask = self.tik_instance.mov_tensor_to_cmpmask(
                                        data_mask_ub[cycle * MASK_MAX])
                                    self.tik_instance.vsel(constant.MASK128, 0,
                                                           data_vsel_ub[cycle * FP16_MAX],
                                                           cmpmask,
                                                           data_max_ub[cycle * FP16_MAX],
                                                           data_vsel_ub_zero[0],
                                                           constant.REPEAT_TIME_ONCE,
                                                           constant.STRIDE_ONE,
                                                           constant.STRIDE_ONE,
                                                           constant.STRIDE_ONE,
                                                           constant.REPEAT_STRIDE_EIGHT,
                                                           constant.REPEAT_STRIDE_EIGHT,
                                                           constant.REPEAT_STRIDE_EIGHT)
                            # fp16 to fp32
                            if v_rep_cycle_fp32_every > 0:  # 0
                                with self.tik_instance.for_range(0, v_rep_cycle_fp32_every,
                                                                 thread_num=1) as cycle:
                                    self.tik_instance.vconv(constant.MASK64, "",
                                                            data_vsel_ub_fp32[cycle * V_MAX_REPEAT * FP32_MAX],
                                                            data_vsel_ub[cycle * V_MAX_REPEAT * FP32_MAX],
                                                            V_MAX_REPEAT, constant.STRIDE_ONE,
                                                            constant.STRIDE_ONE,
                                                            constant.REPEAT_STRIDE_EIGHT,
                                                            constant.REPEAT_STRIDE_FOUR)
                            if v_rep_last_fp32_every != 0:  # 32
                                self.tik_instance.vconv(constant.MASK64, "",
                                                        data_vsel_ub_fp32[v_rep_cycle_fp32_every *
                                                                          V_MAX_REPEAT * FP32_MAX],
                                                        data_vsel_ub[v_rep_cycle_fp32_every * V_MAX_REPEAT * FP32_MAX],
                                                        v_rep_last_fp32_every, constant.STRIDE_ONE,
                                                        constant.STRIDE_ONE,
                                                        constant.REPEAT_STRIDE_EIGHT,
                                                        constant.REPEAT_STRIDE_FOUR)
                            # col2img
                            fetch_filter_w = mask_id % kernelw
                            fetch_filter_h = mask_id // kernelw
                            left_top_w = 0
                            left_top_h = 0
                            """"
                            self.tik_instance.col2img(data_vmul_ub_col2img_fp32[0],
                                                      data_vsel_ub_fp32[0],
                                                      (0, 0, 0, 0),
                                                      col2img_h_every, col2img_w, fetch_filter_w,
                                                      fetch_filter_h, left_top_w, left_top_h,
                                                      stridew, strideh,
                                                      kernelw, kernelh, 1, 1,
                                                      ho_max_every * wo_max // 16)
                            """
                            self.tik_instance.col2img(data_vmul_ub_col2img_fp32[0],
                                                      data_vsel_ub_fp32[0],
                                                      (0, 0, 0, 0),
                                                      col2img_h_every, col2img_w, fetch_filter_w,
                                                      fetch_filter_h, left_top_w, left_top_h,
                                                      stridew, strideh,
                                                      kernelw, kernelh, dialate_w, dialate_h,
                                                      ho_max_every * wo_max // 16)
                        if v_rep_cycle_col_every > 0:  # 0
                            with self.tik_instance.for_range(0, v_rep_cycle_col_every) as cycle:
                                self.tik_instance.vconv(constant.MASK64, "",
                                                        data_vmul_ub_col2img_fp16[
                                                            cycle * V_MAX_REPEAT * FP32_MAX],
                                                        data_vmul_ub_col2img_fp32[
                                                            cycle * V_MAX_REPEAT * FP32_MAX],
                                                        V_MAX_REPEAT, constant.STRIDE_ONE,
                                                        constant.STRIDE_ONE,
                                                        constant.REPEAT_STRIDE_FOUR,
                                                        constant.REPEAT_STRIDE_EIGHT)
                        if v_rep_last_col_every != 0:  # 162
                            self.tik_instance.vconv(constant.MASK64, "", data_vmul_ub_col2img_fp16[
                                v_rep_cycle_col_every * V_MAX_REPEAT * FP32_MAX],
                                                    data_vmul_ub_col2img_fp32[
                                                        v_rep_cycle_col_every *
                                                        V_MAX_REPEAT * FP32_MAX],
                                                    v_rep_last_col_every, constant.STRIDE_ONE,
                                                    constant.STRIDE_ONE,
                                                    constant.REPEAT_STRIDE_FOUR,
                                                    constant.REPEAT_STRIDE_EIGHT)
                        # move ub to gm
                        output_cuthline = self.tik_instance.Scalar("int32")
                        output_cuthline.set_as(0)
                        src_address = self.tik_instance.Scalar("int32")
                        src_address.set_as(pad_left * channel)
                        dst_address = self.tik_instance.Scalar("int32")
                        dst_address.set_as(block_batch * dxh * dxw * channel)

                        if hoverlap == 0:
                            output_cuthline.set_as(col2img_h_every)  # 5
                            dst_address.set_as(dst_address +
                                               ((block_h * ho_every +
                                                 looph * ho_max_every) *
                                                strideh - pad_top) * dxw * channel)
                            with self.tik_instance.if_scope(block_h == 0):
                                with self.tik_instance.if_scope(looph == 0):
                                    output_cuthline.set_as(output_cuthline - pad_top)
                                    src_address.set_as(pad_left * channel +
                                                       pad_top * col2img_w * channel)
                                    dst_address.set_as(block_batch * dxh * dxw * channel)
                            with self.tik_instance.if_scope(looph == h_cycle_every - 1):
                                output_cuthline.set_as(ho_every * strideh - col2img_h_every *
                                                       (h_cycle_every - 1))
                        else:
                            src_address.set_as(pad_left * channel + strideh * col2img_w * channel)
                            output_cuthline.set_as((ho_max_every - 1) * strideh)
                            with self.tik_instance.if_scope(looph == h_cycle_every - 1):
                                output_cuthline.set_as((ho_every - looph *
                                                        (ho_max_every - 1) - 1) * strideh)

                            dst_address.set_as(dst_address + ((block_h * (ho_every - 1) + looph * (ho_max_every - 1) -
                                                               pad_top + 1) * strideh + 1) * dxw * channel)

                            with self.tik_instance.if_scope(block_h == 0):  # 1
                                with self.tik_instance.if_scope(looph == 0):  # 0
                                    output_cuthline.set_as(ho_max_every * strideh - pad_top)
                                    src_address.set_as(pad_left * channel + pad_top * col2img_w * channel)
                                    dst_address.set_as(block_batch * dxh * dxw * channel)
                        self.tik_instance.data_move(data_output[dst_address],
                                                    data_vmul_ub_col2img_fp16[src_address],
                                                    constant.SID, output_cuthline, self.offset_w,
                                                    col2img_w - self.offset_w,
                                                    dxw - self.offset_w)  # 0
        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=(data_input_origin, data_input, data_mask),
                                   outputs=data_output, enable_l2=False)

        return self.tik_instance

    def clean_gm(self, dtype, num, start_index, data_output):
        """
        The fun just for clean gm
        """
        clear_cycle = num // 256
        last_cycle = num % 256 // 16
        if clear_cycle != 0:
            data_ub_zero_256 = self.tik_instance.Tensor(dtype, (256,),
                                                        name="data_ub_zero_256",
                                                        scope=tik.scope_ubuf)
            self.clean_fp16_one_repeat(data_ub_zero_256, dtype)
            with self.tik_instance.for_range(0, clear_cycle) as cycle:
                self.tik_instance.data_move(data_output[start_index + 256 * cycle],
                                            data_ub_zero_256[0],
                                            constant.SID, constant.DEFAULT_NBURST,
                                            16,
                                            constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)
        if last_cycle != 0:
            data_ub_zero = self.tik_instance.Tensor(dtype, (16,),
                                                    name="data_ub_zero",
                                                    scope=tik.scope_ubuf)
            self.clean_fp16_one_repeat(data_ub_zero, dtype)
            with self.tik_instance.for_range(0, last_cycle) as cycle:
                self.tik_instance.data_move(
                    data_output[start_index + 256 * clear_cycle + 16 * cycle],
                    data_ub_zero[0],
                    constant.SID, constant.DEFAULT_NBURST,
                    1,
                    constant.STRIDE_ZERO,
                    constant.STRIDE_ZERO)

    def clean_max_ub(self, data_max_ub, dtype):
        """
        The fun just for clean max ub
        """
        repeat_time = data_max_ub.shape[0] // 128
        data_vsel_scalar = self.tik_instance.Scalar(dtype)
        data_vsel_scalar.set_as(0)
        repeat_time_num = (repeat_time + 254) // 255
        if repeat_time_num > 255:
            with self.tik_instance.for_range(0, repeat_time_num) as repeat_index:
                with self.tik_instance.if_scope(repeat_index !=
                                                (repeat_time_num - 1)):
                    self.tik_instance.vector_dup(
                        constant.MASK128,
                        data_max_ub[repeat_index * 255 * 128],
                        data_vsel_scalar,
                        255,
                        constant.STRIDE_ONE,
                        constant.REPEAT_STRIDE_EIGHT)
                with self.tik_instance.else_scope():
                    self.tik_instance.vector_dup(
                        constant.MASK128,
                        data_max_ub[repeat_index * 255 * 128],
                        data_vsel_scalar,
                        (repeat_time - repeat_index * 255),
                        constant.STRIDE_ONE,
                        constant.REPEAT_STRIDE_EIGHT)
        else:
            self.tik_instance.vector_dup(constant.MASK128,
                                         data_max_ub,
                                         data_vsel_scalar,
                                         repeat_time,
                                         constant.STRIDE_ONE,
                                         constant.REPEAT_STRIDE_EIGHT)

    def clean_fp16_one_repeat(self, data_vsel_ub_zero, dtype):
        """
        The fun just for clean ub
        """
        data_vsel_scalar = self.tik_instance.Scalar(dtype)
        data_vsel_scalar.set_as(0)
        if data_vsel_ub_zero.shape[0] > 128:
            self.tik_instance.vector_dup(constant.MASK128,
                                         data_vsel_ub_zero[0],
                                         data_vsel_scalar,
                                         (data_vsel_ub_zero.shape[0] + 127) // 128,
                                         constant.STRIDE_ONE,
                                         constant.REPEAT_STRIDE_EIGHT)
        else:
            self.tik_instance.vector_dup(data_vsel_ub_zero.shape[0],
                                         data_vsel_ub_zero[0],
                                         data_vsel_scalar,
                                         constant.REPEAT_TIME_ONCE,
                                         constant.STRIDE_ONE,
                                         constant.REPEAT_STRIDE_EIGHT)

    def clean_fp32_multi_repeat(self, data_vmul_ub_col2img_fp32, dtype_size):
        """
        The fun just for clean ub
        """
        v_rep_clear_time = data_vmul_ub_col2img_fp32.shape[0] * dtype_size // ONE_REPEAT
        v_rep_clear_cycle = v_rep_clear_time // V_MAX_REPEAT
        v_rep_clear_last = v_rep_clear_time % V_MAX_REPEAT
        data_clean_scalar = self.tik_instance.Scalar("float32")
        data_clean_scalar.set_as(0)
        if v_rep_clear_cycle > 0:
            with self.tik_instance.for_range(0, v_rep_clear_cycle, thread_num=1) as cycle:
                self.tik_instance.vector_dup(constant.MASK64,
                                             data_vmul_ub_col2img_fp32[cycle * V_MAX_REPEAT
                                                                       * FP32_MAX],
                                             data_clean_scalar,
                                             V_MAX_REPEAT,
                                             constant.STRIDE_ONE,
                                             constant.REPEAT_STRIDE_EIGHT)
        if v_rep_clear_last != 0:
            self.tik_instance.vector_dup(constant.MASK64,
                                         data_vmul_ub_col2img_fp32[v_rep_clear_cycle *
                                                                   V_MAX_REPEAT * FP32_MAX],
                                         data_clean_scalar, v_rep_clear_last,
                                         constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT)
