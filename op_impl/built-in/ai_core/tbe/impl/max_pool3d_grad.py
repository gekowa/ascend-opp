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
max_pool3d_grad
"""
# pylint: disable=too-many-lines,import-error
from te import tik
from topi.cce import util
from te import platform as tbe_platform
import math

# available number of cores
MAX_CORE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
# vector_repeat
MAX_REPEAT = 255
# block_size
MINI_UNIT = 32
# mini value of fp16
MIN_VALUE_FP16 = -65535.0
# vector fp16 size
VECTOR_FP16_SIZE = 128
# vector fp32 size
VECTOR_FP32_SIZE = 64
# vconv mask
MASK64_VALUE = 64
# maximum dma_copy stride
MAX_STRIDE = 65535


def _ceil_div(value, block):
    """
    Integrate the input value by block.
    """
    return (value + block - 1) // block


def _prod(values):
    """
    Prod the input values by multiply.
    """
    res = 1
    for value in values:
        res *= value

    return res


def _cal_core(tik_instance, total_core_loop_num, num_core, core_number):
    """
    calculate the loop number on each core
    """
    if total_core_loop_num % core_number == 0:
        core_loop = total_core_loop_num // core_number
        sum_core = core_loop * num_core
    else:
        core_loop = tik_instance.Scalar("uint64")
        sum_core = tik_instance.Scalar("uint64")
        with tik_instance.if_scope(num_core < total_core_loop_num %
                                   MAX_CORE):
            core_loop.set_as((total_core_loop_num + core_number - 1) //
                             core_number)
            sum_core.set_as(core_loop * num_core)
        with tik_instance.else_scope():
            core_loop.set_as(total_core_loop_num // core_number)
            sum_core.set_as((core_loop + 1) * (total_core_loop_num % MAX_CORE) +
                            core_loop * (num_core - total_core_loop_num %
                                         MAX_CORE))
    return core_loop, sum_core


def init_coordinate(tik_instance, pad_x_top, xi_coordinate):
    # return actual xi_coord
    if pad_x_top != 0:
        xi_coord = tik_instance.Scalar(dtype='int64', name='xi_coord')
        with tik_instance.if_scope(xi_coordinate < 0):
            xi_coord.set_as(0)
        with tik_instance.else_scope():
            xi_coord.set_as(xi_coordinate)
    else:
        xi_coord = xi_coordinate

    return xi_coord


def calc_pad(tik_instance, pad_top, pad_bottom,
             xi_coord, xi_value, boundary):
    # return pad_value in different axis
    top = pad_top
    bottom = pad_bottom

    if pad_top != 0:
        top = tik_instance.Scalar(dtype='int64', name='top')
        with tik_instance.if_scope(xi_coord < 0):
            top.set_as(0 - xi_coord)
        with tik_instance.else_scope():
            top.set_as(0)

    if pad_bottom != 0:
        bottom = tik_instance.Scalar(dtype='int64', name='bottom')
        with tik_instance.if_scope(xi_coord + xi_value > boundary):
            bottom.set_as(xi_coord + xi_value - boundary)
        with tik_instance.else_scope():
            bottom.set_as(0)
    return top, bottom


def grad_model(pad_list):
    for i in pad_list:
        if i > 0:
            model = 'SAME'
            break
        model = 'VALID'

    return model


def _check_config(config):
    config = list(config)
    mark = True
    for i in config:
        if i > 255:
            mark = False
            break

    return mark



class Params:

    def __init__(self, ub_split, col_in_size,
                 forward_ou_size, mask_size, grad_size,
                 zero_size, grad_sel_fp16_size, grad_sel_fp32_size,
                 f_map_fp32_size, l1_in_size):
        self.ub_split = ub_split
        self.l1_in_size = l1_in_size
        self.col_in_size = col_in_size
        self.forward_ou_size = forward_ou_size
        self.mask_size = mask_size
        self.grad_size = grad_size
        self.zero_size = zero_size
        self.grad_sel_fp16_size = grad_sel_fp16_size
        self.grad_sel_fp32_size = grad_sel_fp32_size
        self.f_map_fp32_size = f_map_fp32_size


class MaxPool3DGradCompute(object):

    def __init__(self, shape_list, params):
        # forward_in_shape, forward_ou_shape, grad_shape, ou_shape
        # list(ksize), list(strides), pads, dtype
        self.forward_in_shape = shape_list[0]
        self.forward_ou_shape = shape_list[1]
        self.grad_shape = shape_list[2]
        self.ou_shape = shape_list[3]
        self.core_ou_shape = []
        self.core_in_shape = []

        self.n = self.forward_in_shape[0]
        self.d = self.forward_in_shape[1]
        self.c1 = self.forward_in_shape[2]
        self.h = self.forward_in_shape[3]
        self.w = self.forward_in_shape[4]
        self.c0 = self.forward_in_shape[5]

        self.ksize = params[0]
        self.strides = params[1]
        padding = params[2]
        self.pads = grad_model(padding)
        self.dtype = params[3]
        self.kernel_name = params[4]

        self.kd = self.ksize[1]
        self.kh = self.ksize[2]
        self.kw = self.ksize[3]
        self.sd = self.strides[1]
        self.sh = self.strides[2]
        self.sw = self.strides[3]

        self.do, self.ho, self.wo, self.pad = self._padding_mode()
        self.overlap_d = self._overlap_mode(self.sd, self.kd, self.do, self.d)
        self.overlap_h = self._overlap_mode(self.sh, self.kh, self.ho, self.h)
        self.overlap_w = self._overlap_mode(self.sw, self.kw, self.wo, self.w)
        self.di_invalid, \
        self.hi_invalid, self.wi_invalid = self._invalid_part()

        self.num_bit = 2
        self.num_bit_fp32 = 4
        self.mask_fp16 = 128
        self.mask_fp32 = 64
        self.ub_maxsize = tbe_platform.cce_conf.\
                              get_soc_spec(tbe_platform.
                                           cce_conf.UB_SIZE) // self.num_bit
        self.L1_maxsize = tbe_platform.cce_conf.\
                              get_soc_spec(tbe_platform.
                                           cce_conf.L1_SIZE) // self.num_bit

    def set_tik_instance(self):
        """
        set tik_instance
        """
        tik_instance = tik.Tik()
        self.set_src_dst_tensor(tik_instance)

        return tik_instance

    def set_src_dst_tensor(self, tik_instance):
        """
        set input and output tensor
        """
        self.orig_x_gm = tik_instance.Tensor(self.dtype,
                                             self.forward_in_shape,
                                             name="orig_x_gm",
                                             scope=tik.scope_gm)

        self.orig_y_gm = tik_instance.Tensor(self.dtype,
                                             self.forward_ou_shape,
                                             name="orig_y_gm",
                                             scope=tik.scope_gm)

        self.grads_gm = tik_instance.Tensor(self.dtype,
                                            self.grad_shape,
                                            name="grads_gm",
                                            scope=tik.scope_gm)

        self.ou_y_gm = tik_instance.Tensor("float32",
                                           self.ou_shape,
                                           name="ou_y_gm",
                                           scope=tik.scope_gm,
                                           is_atomic_add=True)

    def _padding_mode(self,):
        # NDC1HWC0
        _, map_d, _, map_h, map_w, _ = self.forward_in_shape
        _, kernel_d, kernel_h, kernel_w, _ = self.ksize
        if self.pads.upper() == 'VALID':
            do = int(math.ceil((map_d - kernel_d + 1) * 1.0 / self.sd))
            ho = int(math.ceil((map_h - kernel_h + 1) * 1.0 / self.sh))
            wo = int(math.ceil((map_w - kernel_w + 1) * 1.0 / self.sw))
            pad_d_top = pad_d_bottom = \
                pad_hw_top = pad_hw_left = pad_hw_bottom = pad_hw_right = 0

        else:
            do = (map_d + self.sd - 1) // self.sd
            ho = (map_h + self.sh - 1) // self.sh
            wo = (map_w + self.sw - 1) // self.sw

            pad_h = max((ho - 1) * self.sh + kernel_h - map_h, 0)
            pad_hw_top = pad_h // 2
            pad_hw_bottom = pad_h - pad_hw_top
            pad_w = max((wo - 1) * self.sw + kernel_w - map_w, 0)
            pad_hw_left = pad_w // 2
            pad_hw_right = pad_w - pad_hw_left

            pad_d = max((do - 1) * self.sd + kernel_d - map_d, 0)
            pad_d_top = pad_d // 2
            pad_d_bottom = pad_d - pad_d_top

        pad_model = [[pad_d_top, pad_d_bottom],
                     [pad_hw_top, pad_hw_bottom],
                     [pad_hw_left, pad_hw_right]]

        return do, ho, wo, pad_model

    def _infer_dim_return(self, do, ho, wo, model):

        if self.kd >= self.sd:
            di = self.kd + (do-1) * self.sd
        else:
            di = do * self.sd

        if self.kh >= self.sh:
            hi = self.kh + (ho-1) * self.sh
        else:
            hi = ho * self.sh

        if self.kw > self.sw:
            wi = self.kw + (wo-1) * self.sw
        else:
            wi = wo * self.sw

        # model: True, work for real split
        # model: False, calc used part for _invalid_part()
        # if not split do,ho,wo, all dim would
        # be return.
        if model:
            if self.do == do:
                # in "SAME", return the filled di
                di = self.d + self.pad[0][0] + self.pad[0][1]
            if self.ho == ho:
                hi = self.h
            if self.wo == wo:
                wi = self.w

        return di, hi, wi

    def _infer_map_return(self, do, ho, wo):
        # Only work in "SAME", return size of feature_map.
        # Because in "VALID", feature_map's size is as same as l1_in_buf.
        # But in "SAME", feature_map >= l1_in_buf

        if self.kd >= self.sd:
            di = self.kd + (do-1) * self.sd
        else:
            di = do * self.sd

        if self.kh >= self.sh:
            hi = self.kh + (ho-1) * self.sh
        else:
            hi = ho * self.sh

        if self.kw > self.sw:
            wi = self.kw + (wo-1) * self.sw
        else:
            wi = wo * self.sw

        if self.do == do:
            di = self.d + self.pad[0][0] + self.pad[0][1]
        if self.ho == ho:
            hi = self.h + self.pad[1][0] + self.pad[1][1]
        if self.wo == wo:
            wi = self.w + self.pad[2][0] + self.pad[2][1]

        return di, hi, wi

    def _invalid_part(self):
        # return area of kernel doesn't slides
        di, hi, wi = self._infer_dim_return(self.do, self.ho, self.wo, False)
        invalid_d = self.d - di
        invalid_h = self.h - hi
        invalid_w = self.w - wi

        return invalid_d, invalid_h, invalid_w

    def _overlap_mode(self, stride, size, xo, xi):
        # xo: direction of x can be slided by xo times
        # xi: the length of x
        if xo == 1:
            # If xo is 1,only xi >= stride, stride has work.
            # If xo is 1 and xi < stride, only kernel has work
            if xi >= stride:
                overlap = size - stride
            else:
                overlap = 0
        else:
            overlap = size - stride

        return overlap

    def _check_process_space(self, do, ho, wo):
        # If consider padding, L1_size must be less
        # than computation of _infer_dim_return.
        # So,actual of data move in L1 may less than space of malloc
        # If data of L1 col2img to UB, l1_in_data would be released.
        # Then, L1 will be used to save overlap which may include
        # overlap_d and overlap_h.
        # if l1_split = True, UB also can't process do ho wo.

        # due to valid, self.pads is [[0,0],[0,0],[0,0]]
        # l1_in_shape is most
        infer_di, infer_hi, infer_wi = self._infer_dim_return(do, ho, wo, True)
        l1_in_shape = [infer_di, infer_hi, infer_wi, self.c0]
        l1_in_size = _prod(l1_in_shape)
        '''
        =====================================
        UB_space: compute virtual space in UB
        =====================================
        col_in_shape:ho wo c0 (512B) ---> for load3d
        forward_ou_shape: do ho wo c0(last do have 256B) ---> for vcmp_eq
        mask_shape: ho wo (uint16,)
        mask_or_shape: same
        mask_not_shape: same
        grad_shape: do ho wo c0(as same as forward_ou_shape)
        zero_shape: 256B
        grad_vsel_fp16_shape: ho wo c0(256B)
        grad_vsel_fp32_shape: ho wo c0(256B)
        f_map_fp32_shape: di hi wi c0
        '''
        col_in_shape = [ho, wo, self.c0]
        col_in_size = _ceil_div(_prod(col_in_shape), 256) * 256

        # forward_ou_shape = [do, ho, wo, self.c0]
        forward_ou_shape_last_do = [1, ho, wo, self.c0]
        forward_ou_shape_except_last = [do-1, ho, wo, self.c0]
        forward_ou_size = _ceil_div(_prod(forward_ou_shape_last_do), 128) * 128
        forward_ou_size += _prod(forward_ou_shape_except_last)

        mask_shape = [ho, wo]
        mask_size = _ceil_div(_prod(mask_shape), 128) * 128
        grad_size = forward_ou_size
        zero_size = 128

        grad_sel_fp16_shape = [ho, wo, self.c0]
        grad_sel_fp16_size = _ceil_div(_prod(grad_sel_fp16_shape), 128) * 128
        grad_sel_fp32_size = grad_sel_fp16_size

        if self.pads.upper() == "VALID":
            f_map_fp32_shape = [infer_di, infer_hi, infer_wi, self.c0]
        else:
            map_di, map_hi, map_wi = self._infer_map_return(do, ho, wo)
            f_map_fp32_shape = [map_di, map_hi, map_wi, self.c0]
        f_map_fp32_size = _prod(f_map_fp32_shape)

        used_ub_byte = (col_in_size + forward_ou_size + mask_size * 3 +
                        grad_size + zero_size +
                        grad_sel_fp16_size) * self.num_bit + \
                       (grad_sel_fp32_size + f_map_fp32_size) * 4

        if used_ub_byte > self.ub_maxsize * self.num_bit:
            ub_split = True
        else:
            ub_split = False

        param = Params(ub_split, col_in_size, forward_ou_size,
                       mask_size, grad_size, zero_size, grad_sel_fp16_size,
                       grad_sel_fp32_size, f_map_fp32_size, l1_in_size)
        return param

    def _check_cut_model(self, cut_model, split_model, all_do, core_branch):
        # "not_tiling": 0
        # "tiling_do": 1
        # "tiling_do_ho": 2
        # "tiling_do_ho_wo": 3
        branch_list = ["not_tiling",  "tiling_do",
                       "tiling_do_ho", "tiling_do_ho_wo"]

        if cut_model == [True, False, False]:
            if split_model[0] == all_do:
                model = 0
            else:
                model = 1
        elif cut_model == [True, True, False]:
            model = 2
        else:
            model = 3

        model = max(model, core_branch)
        return branch_list[model]

    def _pattern(self, core_ou_shape, core_branch):
        # valid
        # D H W C0 -> Do Ho Wo C0
        all_wo = core_ou_shape[-2]
        all_ho = core_ou_shape[-3]
        all_do = core_ou_shape[-4]

        wo = all_wo
        ho = all_ho
        do = all_do

        split_do = False
        split_ho = False
        split_wo = False

        for k in range(all_do):
            do = all_do - k
            param = self._check_process_space(do, ho, wo)
            if not param.ub_split:
                split_do = True
                break

        if not split_do:
            do = 1
            for k in range(all_ho):
                ho = all_ho - k
                param = self._check_process_space(do, ho, wo)
                if not param.ub_split:
                    split_do = True
                    split_ho = True
                    break

        if not split_do and not split_ho:
            do = ho = 1
            for k in range(all_wo):
                wo = all_wo - k
                param = self._check_process_space(do, ho, wo)
                if not param.ub_split:
                    split_do = True
                    split_ho = True
                    split_wo = True
                    break

        cut_model = [split_do, split_ho, split_wo]
        split_model = [do, ho, wo]
        if cut_model == [False, False, False]:
            raise RuntimeError("kernel is too larger")

        # avoid hardware bugs that load3dv1 can't
        # support wo=1 and ho != 1 in cloud_v100
        if split_model[-1] == 1 and split_model[-2] != 1:
            param = self._check_process_space(1, 1, 1)
            cut_model = [True, True, True]
            split_model = [1, 1, 1]

        branch = self._check_cut_model(cut_model, split_model,
                                       all_do, core_branch)

        return branch, split_model, param

    def _ultimate_data_move(self, tik_instance, src_buf, dst_buf, in_list, num_bit):
        src_idx, dst_idx = in_list[-2], in_list[-1]
        n_burst, burst_len = in_list[0], in_list[1]
        src_stride, dst_stride = in_list[2], in_list[3]

        with tik_instance.for_range(0, n_burst) as i:
            src_idx += i * (src_stride + burst_len) * MINI_UNIT // num_bit
            dst_idx += i * (dst_stride + burst_len) * MINI_UNIT // num_bit

            tik_instance.data_move(dst_buf[dst_idx],
                                   src_buf[src_idx],
                                   0, 1, burst_len, 0, 0)

    def norm_data_move(self, tik_instance, src_buf, dst_buf, in_list):
        src_idx, dst_idx = in_list[-2], in_list[-1]
        n_burst, burst_len = in_list[0], in_list[1]
        src_stride, dst_stride = in_list[2], in_list[3]

        tik_instance.data_move(dst_buf[dst_idx],
                               src_buf[src_idx],
                               0,
                               n_burst,
                               burst_len,
                               src_stride,
                               dst_stride)

    def _copy_gm_to_l1(self, tik_instance, l1_buf, src_idx, dst_idx, in_shape):
        n_burst = in_shape[0]
        burst_len = _prod(in_shape[1:]) * self.num_bit // MINI_UNIT
        src_stride = (_prod(self.forward_in_shape[3:]) * (self.c1-1) +
                      _prod(self.forward_in_shape[4:]) *
                      (self.h-in_shape[1])) * self.num_bit // MINI_UNIT
        dst_stride = 0

        in_list = [n_burst, burst_len, src_stride, dst_stride, src_idx, dst_idx]
        check = isinstance(src_stride, int)
        with tik_instance.if_scope(src_stride > MAX_STRIDE):
            self._ultimate_data_move(tik_instance, self.orig_x_gm,
                                     l1_buf, in_list, self.num_bit)

        with tik_instance.else_scope():
            if check:
                if src_stride <= MAX_STRIDE:
                    self.norm_data_move(tik_instance, self.orig_x_gm,
                                        l1_buf, in_list)
            else:
                self.norm_data_move(tik_instance, self.orig_x_gm,
                                    l1_buf, in_list)

    def _gm2l1_tiling_do_ho(self, tik_instance, l1_buf, src_idx, dst_idx,
                            in_shape, hi_batch):
        n_burst = in_shape[0]
        burst_len = _prod(in_shape[1:]) * self.num_bit // MINI_UNIT
        src_stride = (_prod(self.forward_in_shape[3:]) * (self.c1-1) +
                      _prod(self.forward_in_shape[4:]) *
                      (self.h-in_shape[1])) * self.num_bit // MINI_UNIT
        dst_stride = (hi_batch - in_shape[1]) * self.w * self.c0 * \
                     self.num_bit // MINI_UNIT

        in_list = [n_burst, burst_len, src_stride,
                   dst_stride, src_idx, dst_idx]
        # dst_stride and src_stride must be same type
        check = isinstance(src_stride, int)

        with tik_instance.if_scope(
                tik.any(src_stride > MAX_STRIDE,
                        dst_stride > MAX_STRIDE)):
            self._ultimate_data_move(tik_instance, self.orig_x_gm,
                                     l1_buf, in_list, self.num_bit)
        with tik_instance.else_scope():
            if check:
                if src_stride <= MAX_STRIDE and dst_stride <= MAX_STRIDE:
                    self.norm_data_move(tik_instance, self.orig_x_gm,
                                        l1_buf, in_list)
            else:
                self.norm_data_move(tik_instance, self.orig_x_gm,
                                    l1_buf, in_list)

    def _gm2l1_tiling_do_ho_wo(self, tik_instance,
                               l1_buf, src_idx, dst_idx,
                               input0, input1):

        di_val, hi_val, wi_val = input0[0], input0[1], input0[2]
        di_batch, hi_batch, wi_batch = input1[0], input1[1], input1[2]

        # ==================================
        # copy gm to l1
        # ==================================
        c1 = self.c1
        c0 = self.c0
        in_shape = [hi_val, wi_val, c0]
        n_burst = in_shape[0]
        burst_len = _prod(in_shape[1:]) * self.num_bit // MINI_UNIT
        src_stride = (self.w - wi_val) * c0 * self.num_bit // MINI_UNIT
        dst_stride = 0

        with tik_instance.for_range(0, di_val) as idx:
            src_idx_new = src_idx + _prod(self.forward_in_shape[3:]) * c1 * idx
            dst_idx_new = dst_idx + hi_batch * wi_batch * c0 * idx

            in_list = [n_burst, burst_len, src_stride,
                       dst_stride, src_idx_new, dst_idx_new]
            check = isinstance(src_stride, int)

            with tik_instance.if_scope(src_stride > MAX_STRIDE):
                self._ultimate_data_move(tik_instance, self.orig_x_gm,
                                         l1_buf, in_list, self.num_bit)

            with tik_instance.else_scope():
                if check:
                    if src_stride <= MAX_STRIDE:
                        self.norm_data_move(tik_instance, self.orig_x_gm,
                                            l1_buf, in_list)
                else:
                    self.norm_data_move(tik_instance, self.orig_x_gm,
                                        l1_buf, in_list)

    def _copy_ub_to_gm(self, tik_instance, src_buf, src_idx,
                       dst_buf, dst_idx, in_shape):
        # "float32"
        n_burst = in_shape[0]
        burst_len = _prod(in_shape[1:]) * self.num_bit_fp32 // MINI_UNIT
        src_stride = 0
        dst_stride = (_prod(self.forward_in_shape[3:]) * (self.c1-1) +
                      _prod(self.forward_in_shape[4:]) *
                      (self.h-in_shape[1])) * self.num_bit_fp32 // MINI_UNIT

        in_list = [n_burst, burst_len, src_stride,
                   dst_stride, src_idx, dst_idx]
        check = isinstance(dst_stride, int)

        with tik_instance.if_scope(dst_stride > MAX_STRIDE):
            self._ultimate_data_move(tik_instance, src_buf,
                                     dst_buf, in_list, self.num_bit_fp32)
        with tik_instance.else_scope():
            if check:
                if dst_stride <= MAX_STRIDE:
                    self.norm_data_move(tik_instance, src_buf, dst_buf, in_list)
            else:
                self.norm_data_move(tik_instance, src_buf, dst_buf, in_list)

    def _ub2gm_split_do_ho_2(self, tik_instance, src,
                             src_idx, dst, dst_idx, in_shape, hi_batch):
        n_burst = in_shape[0]
        burst_len = _prod(in_shape[1:]) * \
                    self.num_bit_fp32 // MINI_UNIT
        src_stride = (hi_batch - in_shape[1]) * self.w * self.c0 * \
                     self.num_bit_fp32 // MINI_UNIT
        dst_stride = (_prod(self.forward_in_shape[3:]) * (self.c1-1) +
                      _prod(self.forward_in_shape[4:]) *
                      (self.h-in_shape[1])) * \
                     self.num_bit_fp32 // MINI_UNIT

        in_list = [n_burst, burst_len, src_stride,
                   dst_stride, src_idx, dst_idx]
        # src_stride and dst_stride must be same type
        check = isinstance(src_stride, int)

        with tik_instance.if_scope(
                tik.any(src_stride > MAX_STRIDE,
                        dst_stride > MAX_STRIDE)):
            self._ultimate_data_move(tik_instance, src, dst,
                                     in_list, self.num_bit_fp32)

        with tik_instance.else_scope():
            if check:
                if src_stride <= MAX_STRIDE and dst_stride <= MAX_STRIDE:
                    self.norm_data_move(tik_instance, src, dst, in_list)
            else:
                self.norm_data_move(tik_instance, src, dst, in_list)

    def _ub2gm_split_do_ho(self, tik_instance, src_buf, src_idx,
                           dst_buf, dst_idx, in_shape):
        n_burst = in_shape[0]
        burst_len = _prod(in_shape[1:]) * \
                    self.num_bit_fp32 // MINI_UNIT
        src_stride = self.overlap_h * self.w * self.c0 * \
                     self.num_bit_fp32 // MINI_UNIT
        dst_stride = (_prod(self.forward_in_shape[3:]) * (self.c1-1) +
                      _prod(self.forward_in_shape[4:]) *
                      (self.h-in_shape[1])) * \
                     self.num_bit_fp32 // MINI_UNIT

        in_list = [n_burst, burst_len, src_stride,
                   dst_stride, src_idx, dst_idx]
        # src_stride is int,
        check = isinstance(dst_stride, int)

        with tik_instance.if_scope(
                tik.any(src_stride > MAX_STRIDE,
                        dst_stride > MAX_STRIDE)):
            self._ultimate_data_move(tik_instance, src_buf,
                                     dst_buf, in_list, self.num_bit_fp32)

        with tik_instance.else_scope():
            if check:
                if src_stride <= MAX_STRIDE and dst_stride <= MAX_STRIDE:
                    self.norm_data_move(tik_instance, src_buf, dst_buf, in_list)
            else:
                if src_stride <= MAX_STRIDE:
                    self.norm_data_move(tik_instance, src_buf, dst_buf, in_list)

    def _ub2gm_split_do_ho_wo(self, tik_instance, src, src_idx, dst, dst_idx,
                              in_shape, hi_batch, wi_batch):
        # ac_data: [di, hi, wi, c0] = in_shape
        # vir_data:[di_batch, hi_batch, wi_batch, c0]
        # relationship: xi_batch >= xi
        # [hi, wi, c0] matrix per process

        c0 = in_shape[-1]
        c1 = self.c1
        num_bit = self.num_bit_fp32

        n_burst = in_shape[1]
        burst_len = _prod(in_shape[2:]) * num_bit // MINI_UNIT
        src_stride = (wi_batch - in_shape[2]) * c0 * num_bit // MINI_UNIT
        dst_stride = (self.w - in_shape[2]) * c0 * num_bit // MINI_UNIT

        for idx in range(in_shape[0]):
            dst_idx_new = dst_idx + _prod(self.forward_in_shape[3:]) * c1 * idx
            src_idx_new = src_idx + hi_batch * wi_batch * c0 * idx

            in_list = [n_burst, burst_len, src_stride,
                       dst_stride, src_idx_new, dst_idx_new]
            # type of src_stride is as same as dst_stride
            check = isinstance(src_stride, int)

            with tik_instance.if_scope(
                    tik.any(src_stride > MAX_STRIDE,
                            dst_stride > MAX_STRIDE)):
                self._ultimate_data_move(tik_instance, src,
                                         dst, in_list, num_bit)
            with tik_instance.else_scope():
                if check:
                    if src_stride <= MAX_STRIDE and dst_stride <= MAX_STRIDE:
                        self.norm_data_move(tik_instance, src, dst, in_list)
                else:
                    self.norm_data_move(tik_instance, src, dst, in_list)

    def _mov_init(self, tik_instance, ubuf,
                  num_overlap, num_init_zero):
        # mov  float32
        all_num = num_overlap + num_init_zero
        n_burst = 1
        burst_len = num_overlap * 4 // MINI_UNIT
        src_stride = 0
        dst_stride = 0
        if num_overlap > 0:
            tik_instance.data_move(ubuf[0],
                                   ubuf[all_num-num_overlap],
                                   0,
                                   n_burst,
                                   burst_len,
                                   src_stride,
                                   dst_stride)
        # vec_dup
        self.set_vector_dup(tik_instance, num_init_zero,
                            ubuf, num_overlap, 0, "float32")

    def _copy_gm_to_ub(self, tik_instance, dst_buf, src_buf, src_idx, in_shape):
        # in_shape is [do, ho, wo, c0]
        # Only split do, self.ho is equal to in_shape[1], self.wo is equal
        # to in_shape[2].
        # Only split do and ho, self.wo is equal to in_shape[2], and do is 1.
        # Only split do, ho, wo, do and ho is 1.
        n_burst = in_shape[0]
        burst_len = _prod(in_shape[1:]) * self.num_bit // MINI_UNIT
        src_stride = (_prod(self.forward_ou_shape[3:]) * (self.c1-1) +
                      _prod(self.forward_ou_shape[4:]) * (self.ho-in_shape[1]) +
                      self.c0 * (self.wo-in_shape[2])) * \
                     self.num_bit // MINI_UNIT
        dst_stride = 0

        if src_stride > MAX_STRIDE or dst_stride > MAX_STRIDE:
            in_list = [n_burst, burst_len, src_stride,
                       dst_stride, src_idx, 0]
            self._ultimate_data_move(tik_instance, src_buf,
                                     dst_buf, in_list, self.num_bit)
        else:
            tik_instance.data_move(dst_buf[0],
                                   src_buf[src_idx],
                                   0,
                                   n_burst,
                                   burst_len,
                                   src_stride,
                                   dst_stride)

    def set_vector_dup(self, tik_instance, psm, dst, idx, number, dtype):
        # idx is begin_index in dst,
        # must be 32B align
        if dtype == "float16":
            mask = 128
        else:
            mask = 64

        dup_psm = MAX_REPEAT * mask
        dup_repeat_merchant = psm // dup_psm
        dup_repeat_remainder = psm % dup_psm
        dst_blk_stride = 1
        dst_rep_stride = 8

        with tik_instance.for_range(0, dup_repeat_merchant) as i:
            tik_instance.vector_dup(mask,
                                    dst[idx + i * dup_psm],
                                    number,
                                    MAX_REPEAT,
                                    dst_blk_stride,
                                    dst_rep_stride)

        if dup_repeat_remainder != 0:
            repeats = dup_repeat_remainder // mask
            dup_remainder = dup_repeat_remainder % mask
            if repeats != 0:
                tik_instance.vector_dup(mask,
                                        dst[idx + dup_repeat_merchant * dup_psm],
                                        number,
                                        repeats,
                                        dst_blk_stride,
                                        dst_rep_stride)
            if dup_remainder != 0:
                tik_instance.vector_dup(dup_remainder,
                                        dst[idx + dup_repeat_merchant * dup_psm +
                                            repeats * mask],
                                        number,
                                        1,
                                        dst_blk_stride,
                                        dst_rep_stride)

    def _vconv(self, tik_instance, src, src_start, dst,
               dst_start, ele_num, src_dtype):
        total_repeat_time = ele_num // VECTOR_FP32_SIZE
        remain_ele = ele_num % VECTOR_FP32_SIZE
        mask_value = VECTOR_FP32_SIZE

        repeat_max_time = total_repeat_time // MAX_REPEAT
        remain_repeat_time = total_repeat_time % MAX_REPEAT

        if src_dtype == 'float16':
            src_stride, dst_stride = 4, 8
            if repeat_max_time > 0:
                with tik_instance.for_range(0, repeat_max_time) as loop1:
                    tik_instance.vconv(
                        MASK64_VALUE, "",
                        dst[
                            dst_start + loop1 * MAX_REPEAT * mask_value],
                        src[
                            src_start + loop1 * MAX_REPEAT * mask_value],
                        MAX_REPEAT, 1, 1, dst_stride, src_stride)
            if remain_repeat_time > 0:
                tik_instance.vconv(
                    MASK64_VALUE, "",
                    dst[
                        dst_start + repeat_max_time * MAX_REPEAT * mask_value],
                    src[
                        src_start + repeat_max_time * MAX_REPEAT * mask_value],
                    remain_repeat_time, 1, 1, dst_stride, src_stride)
            if remain_ele > 0:
                tik_instance.vconv(
                    remain_ele, "",
                    dst[dst_start + repeat_max_time * MAX_REPEAT *
                        mask_value + remain_repeat_time * mask_value],
                    src[src_start + repeat_max_time * MAX_REPEAT *
                        mask_value + remain_repeat_time * mask_value],
                    1, 1, 1, dst_stride, src_stride)

        else:
            src_stride, dst_stride = 8, 4
            if repeat_max_time > 0:
                with tik_instance.for_range(0, repeat_max_time) as loop1:
                    tik_instance.vconv(
                        MASK64_VALUE, "",
                        dst[
                            dst_start + loop1 * MAX_REPEAT * mask_value],
                        src[
                            src_start + loop1 * MAX_REPEAT * mask_value],
                        MAX_REPEAT, 1, 1, dst_stride, src_stride)
            if remain_repeat_time > 0:
                tik_instance.vconv(
                    MASK64_VALUE, "",
                    dst[
                        dst_start + repeat_max_time * MAX_REPEAT * mask_value],
                    src[
                        src_start + repeat_max_time * MAX_REPEAT * mask_value],
                    remain_repeat_time, 1, 1, dst_stride, src_stride)
            if remain_ele > 0:
                tik_instance.vconv(
                    remain_ele, "",
                    dst[dst_start + repeat_max_time * MAX_REPEAT *
                        mask_value + remain_repeat_time * mask_value],
                    src[src_start + repeat_max_time * MAX_REPEAT *
                        mask_value + remain_repeat_time * mask_value],
                    1, 1, 1, dst_stride, src_stride)

    def _vector_op(self, tik_instance, operator,
                   src1, src2, dst, dtype, ele_num,
                   stride_config=None):
        stride_config = list(stride_config)
        if dtype == "float16":
            repeat_times = ele_num // VECTOR_FP16_SIZE
            remain_ele = ele_num % VECTOR_FP16_SIZE
            mask = VECTOR_FP16_SIZE
        else:
            repeat_times = ele_num // VECTOR_FP32_SIZE
            remain_ele = ele_num % VECTOR_FP32_SIZE
            mask = VECTOR_FP32_SIZE

        repeat_max_loop = repeat_times // MAX_REPEAT
        remain_max_loop = repeat_times % MAX_REPEAT

        if operator == "vadd":
            if stride_config is None:
                stride_config = 1, 1, 1, 8, 8, 8
            dst_offset = 0
            src1_offset = 0
            src2_offset = 0
            if repeat_max_loop > 0:
                tik_instance.vadd(mask, dst[dst_offset], src1[src1_offset],
                                  src2[src2_offset],
                                  MAX_REPEAT,
                                  stride_config[0], stride_config[1],
                                  stride_config[2], stride_config[3],
                                  stride_config[4], stride_config[5])
                dst_offset += MINI_UNIT // (tbe_platform.cce_intrin.get_bit_len(
                    dst.dtype.lower()) // 8) * stride_config[3] * 255
                src1_offset += MINI_UNIT // (tbe_platform.cce_intrin.get_bit_len(
                    src1.dtype.lower()) // 8) * stride_config[4] * 255
                src2_offset += MINI_UNIT // (tbe_platform.cce_intrin.get_bit_len(
                    src2.dtype.lower()) // 8) * stride_config[5] * 255
            if remain_max_loop > 0:
                # rep_stride maybe more than 255 while repeat_times=1.
                if remain_max_loop == 1:
                    tik_instance.vadd(mask, dst[dst_offset], src1[src1_offset],
                                      src2[src2_offset],
                                      remain_max_loop,
                                      stride_config[0], stride_config[1],
                                      stride_config[2], 0, 0, 0)
                else:
                    tik_instance.vadd(mask, dst[dst_offset], src1[src1_offset],
                                      src2[src2_offset],
                                      remain_max_loop,
                                      stride_config[0], stride_config[1],
                                      stride_config[2], stride_config[3],
                                      stride_config[4], stride_config[5])
                dst_offset += MINI_UNIT // (tbe_platform.cce_intrin.get_bit_len(
                    dst.dtype.lower()) // 8) * stride_config[3] * remain_max_loop
                src1_offset += MINI_UNIT // (tbe_platform.cce_intrin.get_bit_len(
                    src1.dtype.lower()) // 8) * stride_config[4] * remain_max_loop
                src2_offset += MINI_UNIT // (tbe_platform.cce_intrin.get_bit_len(
                    src2.dtype.lower()) // 8) * stride_config[5] * remain_max_loop
            if remain_ele > 0:
                stride_config[3] = stride_config[4] = stride_config[5] = 0
                tik_instance.vadd(remain_ele, dst[dst_offset], src1[src1_offset],
                                  src2[src2_offset], 1,
                                  stride_config[0], stride_config[1],
                                  stride_config[2], stride_config[3],
                                  stride_config[4], stride_config[5])

    def _rewrite_fmap(self, tik_instance, operator,
                      src1, src2, dst, dtype, once_elem,
                      repeat_times, shape_map, shape_grad, config=None):
        # once_elem: amount of data processed at a time in the Wo direction.
        # shape_map: container size of src1[1:].
        # shape_grad: valid data size of src2.

        h, w, c0 = shape_map[0], shape_map[1], shape_map[2]
        ho, wo = shape_grad[0], shape_grad[1]
        config = list(config)
        if dtype == "float16":
            max_mask = 128
            num_block = 8
            block_size = 16
        else:
            max_mask = 64
            num_block = 8
            block_size = 8

        # num_instr_loop_w: num of instructions on direct W
        # num_instr_loop_h: num of instructions on direct H
        num_instr_loop_w = math.ceil(once_elem/max_mask)
        remain_mask = once_elem % max_mask
        if remain_mask == 0 and once_elem != 0:
            remain_mask = max_mask
        num_instr_loop_h = math.ceil(repeat_times/MAX_REPEAT)
        remain_repeat = repeat_times % MAX_REPEAT
        if remain_repeat == 0 and repeat_times != 0:
            remain_repeat = MAX_REPEAT

        dst_offset = src1_offset = src2_offset = 0
        if operator == "vadd":
            for idx_h, _ in enumerate(range(num_instr_loop_h)):
                for idx_w, _ in enumerate(range(num_instr_loop_w)):
                    src1_offset = idx_w * num_block * config[1] * block_size + \
                                   idx_h * MAX_REPEAT * w * c0
                    src2_offset = idx_w * num_block * config[2] * block_size + \
                                   idx_h * MAX_REPEAT * wo * c0
                    dst_offset = idx_w * num_block * config[0] * block_size + \
                                  idx_h * MAX_REPEAT * w * c0

                    if idx_w < num_instr_loop_w - 1:
                        mask = max_mask
                    else:
                        mask = remain_mask
                    if idx_h < num_instr_loop_h - 1:
                        rep = MAX_REPEAT
                    else:
                        rep = remain_repeat
                    tik_instance.vadd(mask, dst[dst_offset],
                                      src1[src1_offset], src2[src2_offset],
                                      rep,
                                      config[0], config[1],
                                      config[2], config[3],
                                      config[4], config[5])

    def _set_buf_tensor(self, tik_instance, param):

        l1_in_buf = tik_instance.Tensor(self.dtype,
                                        [param.l1_in_size, ],
                                        name="l1_in_buf",
                                        scope=tik.scope_cbuf)
        forward_ou_buf = tik_instance.Tensor(self.dtype,
                                             [param.forward_ou_size, ],
                                             name="forward_ou_buf",
                                             scope=tik.scope_ubuf)
        grad_buf = tik_instance.Tensor(self.dtype,
                                       [param.grad_size, ],
                                       name="grad_buf",
                                       scope=tik.scope_ubuf)
        col_in_buf = tik_instance.Tensor(self.dtype,
                                         [param.col_in_size, ],
                                         name="col_in_buf",
                                         scope=tik.scope_ubuf)
        mask_buf = tik_instance.Tensor("uint16",
                                       [param.mask_size, ],
                                       name='mask_buf',
                                       scope=tik.scope_ubuf)
        mask_or_buf = tik_instance.Tensor("uint16",
                                          [param.mask_size, ],
                                          name='mask_or_buf',
                                          scope=tik.scope_ubuf)
        mask_not_buf = tik_instance.Tensor("uint16",
                                           [param.mask_size, ],
                                           name='mask_not_buf',
                                           scope=tik.scope_ubuf)
        zero_buf = tik_instance.Tensor(self.dtype,
                                       [param.zero_size, ],
                                       name='zero_buf',
                                       scope=tik.scope_ubuf)

        grad_sel_fp16_buf = tik_instance.Tensor(self.dtype,
                                                [param.grad_sel_fp16_size, ],
                                                name='grad_sel_fp16_buf',
                                                scope=tik.scope_ubuf)
        grad_sel_fp32_buf = tik_instance.Tensor("float32",
                                                [param.grad_sel_fp32_size, ],
                                                name='grad_sel_fp32_buf',
                                                scope=tik.scope_ubuf)
        f_map_fp32_buf = tik_instance.Tensor("float32",
                                             [param.f_map_fp32_size, ],
                                             name='f_map_fp32_buf',
                                             scope=tik.scope_ubuf)

        buf_list = [l1_in_buf, forward_ou_buf, grad_buf, col_in_buf,
                    mask_buf, mask_or_buf, mask_not_buf, zero_buf,
                    grad_sel_fp16_buf, grad_sel_fp32_buf,
                    f_map_fp32_buf]

        return buf_list

    def _calc_mask(self, tik_instance, buf_list, param,
                   idx_list, const_list):
        # ---calculate mask---
        forward_ou_buf = buf_list[1]
        col_in_buf = buf_list[3]
        mask_buf = buf_list[4]
        mask_or_buf = buf_list[5]
        mask_not_buf = buf_list[6]

        idx_do = idx_list[0]
        idx_d = idx_list[1]
        idx_h = idx_list[2]
        idx_w = idx_list[3]
        ho, wo, c0 = const_list

        with tik_instance.if_scope(tik.all(idx_d == 0, idx_h == 0, idx_w == 0)):
            tik_instance.vcmpv_eq(mask_buf[0],
                                  forward_ou_buf[idx_do*ho*wo*c0],
                                  col_in_buf[0],
                                  math.ceil(ho*wo*c0/VECTOR_FP16_SIZE),
                                  1, 1, 8, 8)

            tik_instance.data_move(mask_or_buf[0],
                                   mask_buf[0], 0, 1,
                                   param.mask_size//16, 0, 0)

            tik_instance.vnot(self.mask_fp16, mask_not_buf, mask_or_buf,
                              param.mask_size // VECTOR_FP16_SIZE,
                              1, 1, 8, 8)

        with tik_instance.else_scope():
            tik_instance.vcmpv_eq(mask_buf[0],
                                  forward_ou_buf[idx_do*ho*wo*c0],
                                  col_in_buf[0],
                                  math.ceil(ho*wo*c0/VECTOR_FP16_SIZE),
                                  1, 1, 8, 8)

            tik_instance.vand(self.mask_fp16, mask_buf, mask_not_buf, mask_buf,
                              param.mask_size // VECTOR_FP16_SIZE,
                              1, 1, 1, 8, 8, 8)

            tik_instance.vor(self.mask_fp16, mask_or_buf, mask_or_buf, mask_buf,
                             param.mask_size // VECTOR_FP16_SIZE,
                             1, 1, 1, 8, 8, 8)

            tik_instance.vnot(self.mask_fp16, mask_not_buf, mask_or_buf,
                              param.mask_size // VECTOR_FP16_SIZE,
                              1, 1, 8, 8)

    def _sel(self, tik_instance, buf_list, idx_list, const_list):
        mask_buf = buf_list[4]
        zero_buf = buf_list[7]
        grad_buf = buf_list[2]
        grad_sel_fp16_buf = buf_list[8]

        ho, wo, c0 = const_list
        idx_do = idx_list[0]

        repeat_times_sel = math.ceil(ho*wo*c0/VECTOR_FP16_SIZE)
        with tik_instance.for_range(0, repeat_times_sel) as serial:
            grad_sel_offset = serial * 128
            grad_offset = serial * 128 + idx_do*ho*wo*c0
            mask_offset = serial * 8
            cmp_mask = tik_instance.mov_tensor_to_cmpmask(mask_buf[mask_offset])
            tik_instance.vsel(self.mask_fp16, 0,
                              grad_sel_fp16_buf[grad_sel_offset],
                              cmp_mask,
                              grad_buf[grad_offset],
                              zero_buf,
                              1, 1, 1, 1, 8, 8, 0)

    def not_tiling_main(self, tik_instance, core_loop, sum_core,
                        model, param):

        do = model[0]
        ho = model[1]
        wo = model[2]
        di, hi, wi = self._infer_dim_return(do, ho, wo, True)
        c0 = self.c0
        c1 = self.c1

        buf_list = self._set_buf_tensor(tik_instance, param)
        l1_in_buf = buf_list[0]
        forward_ou_buf = buf_list[1]
        grad_buf = buf_list[2]
        col_in_buf = buf_list[3]
        mask_buf = buf_list[4]
        mask_or_buf = buf_list[5]
        mask_not_buf = buf_list[6]
        zero_buf = buf_list[7]
        grad_sel_fp16_buf = buf_list[8]
        grad_sel_fp32_buf = buf_list[9]
        f_map_fp32_buf = buf_list[10]
        self.set_vector_dup(tik_instance, param.zero_size,
                            zero_buf, 0, 0, self.dtype)

        with tik_instance.for_range(0, core_loop) as num_core_loop:
            # init
            self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                                f_map_fp32_buf, 0, 0, "float32")
            merchant = (sum_core + num_core_loop) // c1
            remainder = (sum_core + num_core_loop) % c1

            # ----COPY_GM_2_L1_BUF----
            src_orig_x_gm = merchant * _prod(self.forward_in_shape[1:]) + \
                            remainder * hi * wi * c0
            gm2l1_shape = [di, hi, wi, c0]
            self._copy_gm_to_l1(tik_instance, l1_in_buf,
                                src_orig_x_gm, 0, gm2l1_shape)

            # ----COPY_ORI_OUTPUT_2_FORWARD_OU_BUF----
            # ----COPY_GRAD_2_GRAD_BUF----
            src_orig_y_gm = merchant * _prod(self.forward_ou_shape[1:]) + \
                            remainder * ho * wo * c0
            gm2ub_data_shape = [do, ho, wo, c0]
            self._copy_gm_to_ub(tik_instance, forward_ou_buf,
                                self.orig_y_gm, src_orig_y_gm, gm2ub_data_shape)

            src_grad_gm = src_orig_y_gm
            self._copy_gm_to_ub(tik_instance, grad_buf, self.grads_gm,
                                src_grad_gm, gm2ub_data_shape)

            # ---load3d l1 to col_in_buffer---
            repeat_times = _ceil_div(ho*wo, 16)
            # window
            with tik_instance.for_range(0, do, thread_num=1) as idx_do:
                # number of hwc0 in window
                with tik_instance.for_range(0, self.kd, thread_num=1) as idx_d:
                    with tik_instance.for_range(0, self.kh, thread_num=1) as idx_h:
                        with tik_instance.for_range(0, self.kw, thread_num=1) as idx_w:
                            src_l1 = (idx_do * self.sd + idx_d) * hi * wi * c0
                            tik_instance.load3dv1(col_in_buf[0],
                                                  l1_in_buf[src_l1],
                                                  [0, 0, 0, 0],
                                                  hi, wi, 0,
                                                  idx_w, idx_h,
                                                  0, 0,
                                                  self.sw, self.sh,
                                                  self.kw, self.kh,
                                                  1, 1, 1, 1,
                                                  repeat_times, 0,
                                                  MIN_VALUE_FP16
                                                  )

                            # ---calculate mask---
                            with tik_instance.if_scope(tik.all(idx_d == 0,
                                                               idx_h == 0,
                                                               idx_w == 0)):
                                tik_instance.vcmpv_eq(mask_buf[0],
                                                      forward_ou_buf[idx_do*ho*wo*c0],
                                                      col_in_buf[0],
                                                      math.ceil(ho*wo*c0/VECTOR_FP16_SIZE),
                                                      1, 1, 8, 8)

                                tik_instance.data_move(mask_or_buf[0],
                                                       mask_buf[0], 0, 1,
                                                       param.mask_size//16, 0, 0)

                                tik_instance.vnot(self.mask_fp16,
                                                  mask_not_buf, mask_or_buf,
                                                  param.mask_size // VECTOR_FP16_SIZE,
                                                  1, 1, 8, 8)

                            with tik_instance.else_scope():
                                tik_instance.vcmpv_eq(mask_buf[0],
                                                      forward_ou_buf[idx_do*ho*wo*c0],
                                                      col_in_buf[0],
                                                      math.ceil(ho*wo*c0/VECTOR_FP16_SIZE),
                                                      1, 1, 8, 8)

                                tik_instance.vand(self.mask_fp16, mask_buf,
                                                  mask_not_buf, mask_buf,
                                                  param.mask_size // VECTOR_FP16_SIZE,
                                                  1, 1, 1, 8, 8, 8)

                                tik_instance.vor(self.mask_fp16, mask_or_buf,
                                                 mask_or_buf, mask_buf,
                                                 param.mask_size // VECTOR_FP16_SIZE,
                                                 1, 1, 1, 8, 8, 8)

                                tik_instance.vnot(self.mask_fp16, mask_not_buf,
                                                  mask_or_buf,
                                                  param.mask_size // VECTOR_FP16_SIZE,
                                                  1, 1, 8, 8)

                            # ---vsel(grad,zero,mask)---
                            repeat_times_sel = math.ceil(ho*wo*c0/VECTOR_FP16_SIZE)
                            with tik_instance.for_range(0, repeat_times_sel) as serial:
                                grad_sel_offset = serial * 128
                                grad_offset = serial * 128 + idx_do*ho*wo*c0
                                mask_offset = serial * 8
                                cmp_mask = tik_instance.mov_tensor_to_cmpmask(mask_buf[mask_offset])
                                tik_instance.vsel(self.mask_fp16, 0,
                                                  grad_sel_fp16_buf[grad_sel_offset],
                                                  cmp_mask,
                                                  grad_buf[grad_offset],
                                                  zero_buf,
                                                  1, 1, 1, 1, 8, 8, 0)

                            # ---vconv grad_sel_fp16 to fp32---
                            self._vconv(tik_instance, grad_sel_fp16_buf, 0,
                                        grad_sel_fp32_buf, 0,
                                        param.grad_sel_fp16_size, "float16")

                            # ---rewrite grad_sel_fp32 to f_map_fp32
                            config = (self.sw*2, self.sw*2, 2,
                                      self.sh*wi*2, self.sh*wi*2, wo*2)
                            if _check_config(config):
                                with tik_instance.for_range(0, 1) as ho_idx:
                                    map_index = (idx_do*self.sd+idx_d)*hi*wi*c0 + \
                                                (idx_h*wi*c0+idx_w*c0)
                                    mask_index = wo * ho_idx * c0
                                    shape_map_hw = [hi, wi, c0]
                                    shape_grad = [ho, wo, c0]

                                    self._rewrite_fmap(tik_instance, "vadd",
                                                       f_map_fp32_buf[map_index],
                                                       grad_sel_fp32_buf[mask_index],
                                                       f_map_fp32_buf[map_index],
                                                       "float32", wo*c0//2, ho,
                                                       shape_map_hw, shape_grad,
                                                       config=config)

                                    self._rewrite_fmap(tik_instance, "vadd",
                                                       f_map_fp32_buf[map_index+8],
                                                       grad_sel_fp32_buf[mask_index+8],
                                                       f_map_fp32_buf[map_index+8],
                                                       "float32", wo*c0//2, ho,
                                                       shape_map_hw, shape_grad,
                                                       config=config)

                            else:
                                # map_index has three part: which hwc0 in
                                # which window, begin_index of kernel,
                                # begin_index of child kernel
                                with tik_instance.for_range(0, ho) as ho_idx:
                                    map_index = (idx_do*self.sd+idx_d)*hi*wi*c0 + \
                                                (ho_idx*self.sh*wi*c0) + \
                                                (idx_h*wi*c0+idx_w*c0)
                                    mask_index = wo * ho_idx * c0

                                    self._vector_op(tik_instance, "vadd",
                                                    f_map_fp32_buf[map_index],
                                                    grad_sel_fp32_buf[mask_index],
                                                    f_map_fp32_buf[map_index],
                                                    "float32", wo*c0//2,
                                                    stride_config=(self.sw*2,
                                                                   self.sw*2, 2,
                                                                   self.sw*16,
                                                                   self.sw*16, 16))
                                    self._vector_op(tik_instance, "vadd",
                                                    f_map_fp32_buf[map_index+8],
                                                    grad_sel_fp32_buf[mask_index+8],
                                                    f_map_fp32_buf[map_index + 8],
                                                    "float32", wo * c0 // 2,
                                                    stride_config=(self.sw*2,
                                                                   self.sw*2, 2,
                                                                   self.sw*16,
                                                                   self.sw*16, 16))

            # ---mov_out---
            dst_ou_gm = merchant * _prod(self.forward_in_shape[1:]) + \
                        remainder * hi * wi * c0
            ub2gm_shape = [di, hi, wi, c0]
            self._copy_ub_to_gm(tik_instance, f_map_fp32_buf, 0,
                                self.ou_y_gm, dst_ou_gm,
                                ub2gm_shape)

    def tiling_do_main(self, tik_instance, core_loop,
                       sum_core, model, param):
        '''
        =========================
        Just only split do
        =========================
        '''
        do_batch = model[0]
        ho = model[1]
        wo = model[2]

        # batch + tail
        loop_do = self.do // do_batch
        di_batch, hi, wi = self._infer_dim_return(do_batch, ho, wo, True)
        c0 = self.c0
        c1 = self.c1
        if loop_do <= 0:
            raise RuntimeError("loop_do must >= 1")
        do_tail = self.do % do_batch
        di_tail, hi_tail, wi_tail = self._infer_dim_return(do_tail, ho, wo, True)
        if do_tail == 0:
            di_tail = 0

        buf_list = self._set_buf_tensor(tik_instance, param)
        l1_in_buf = buf_list[0]
        forward_ou_buf = buf_list[1]
        grad_buf = buf_list[2]
        col_in_buf = buf_list[3]
        zero_buf = buf_list[7]
        grad_sel_fp16_buf = buf_list[8]
        grad_sel_fp32_buf = buf_list[9]
        f_map_fp32_buf = buf_list[10]
        self.set_vector_dup(tik_instance, param.zero_size,
                            zero_buf, 0, 0, self.dtype)
        self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                            f_map_fp32_buf, 0, 0, "float32")

        # do is batch_do
        # can't fused loop_do in core_loop
        # due to overlap will init next loop_do,
        # if tail existed, fused loop will result in fail
        with tik_instance.for_range(0, core_loop) as num_core_loop:
            merchant = (sum_core + num_core_loop) // c1
            remainder = (sum_core + num_core_loop) % c1

            def _main(loop_idx, di, do):
                # ----Init_Begin_Idx----
                if self.kd >= self.sd:
                    di_coordinate = loop_idx * \
                                    (di_batch-self.overlap_d)
                else:
                    di_coordinate = loop_idx * di_batch
                do_coordinate = loop_idx * do_batch
                src_orig_x_gm = merchant * _prod(self.forward_in_shape[1:]) + \
                                remainder * hi * wi * c0 + \
                                di_coordinate * c1 * hi * wi * c0
                src_orig_y_gm = merchant * _prod(self.forward_ou_shape[1:]) + \
                                remainder * ho * wo * c0 + \
                                do_coordinate * c1 * ho * wo * c0
                src_grad_gm = src_orig_y_gm

                # ----COPY_GM_2_L1_BUF----
                # Prevent reading gm out of bounds
                # which only happened in kd<sd
                with tik_instance.if_scope(di_coordinate + di <= self.d):
                    self._copy_gm_to_l1(tik_instance, l1_in_buf,
                                        src_orig_x_gm, 0, [di, hi, wi, c0])
                with tik_instance.else_scope():
                    self._copy_gm_to_l1(tik_instance, l1_in_buf,
                                        src_orig_x_gm, 0,
                                        [di+self.overlap_d, hi, wi, c0])

                # ----COPY_ORI_OUTPUT_2_FORWARD_OU_BUF----
                self._copy_gm_to_ub(tik_instance, forward_ou_buf,
                                    self.orig_y_gm, src_orig_y_gm,
                                    [do, ho, wo, c0])

                # ----COPY_GRAD_2_GRAD_BUF----
                self._copy_gm_to_ub(tik_instance, grad_buf, self.grads_gm,
                                    src_grad_gm, [do, ho, wo, c0])

                # ---load3d l1 to col_in_buffer---
                repeat_times = _ceil_div(ho*wo, 16)
                # which window
                with tik_instance.for_range(0, do, thread_num=1) as idx_do:
                    # which hwc0
                    with tik_instance.for_range(0, self.kd, thread_num=1) as idx_d:
                        with tik_instance.for_range(0, self.kh, thread_num=1) as idx_h:
                            with tik_instance.for_range(0, self.kw, thread_num=1) as idx_w:
                                src_l1 = (idx_do * self.sd + idx_d) * hi * wi * c0
                                tik_instance.load3dv1(col_in_buf[0],
                                                      l1_in_buf[src_l1],
                                                      [0, 0, 0, 0],
                                                      hi, wi, 0,
                                                      idx_w, idx_h,
                                                      0, 0,
                                                      self.sw, self.sh,
                                                      self.kw, self.kh,
                                                      1, 1, 1, 1,
                                                      repeat_times, 0,
                                                      MIN_VALUE_FP16
                                                      )

                                # ---calculate mask---
                                idx_list = [idx_do, idx_d, idx_h, idx_w]
                                const_list = [ho, wo, c0]
                                self._calc_mask(tik_instance, buf_list, param,
                                                idx_list, const_list)

                                # ---sel(grad,zero,mask)---
                                self._sel(tik_instance, buf_list,
                                          idx_list, const_list)

                                # ---vconv grad_sel_fp16 to fp32---
                                self._vconv(tik_instance, grad_sel_fp16_buf, 0,
                                            grad_sel_fp32_buf, 0,
                                            param.grad_sel_fp16_size, "float16")

                                # ---rewrite grad_sel_fp32 to f_map_fp32---
                                config = (self.sw*2, self.sw*2, 2,
                                          self.sh*wi*2, self.sh*wi*2, wo*2)
                                if _check_config(config):
                                    with tik_instance.for_range(0, 1) as ho_idx:
                                        map_index = (idx_do*self.sd+idx_d)*hi*wi*c0 + \
                                                    (idx_h*wi*c0+idx_w*c0)
                                        mask_index = wo * ho_idx * c0
                                        shape_map_hw = [hi, wi, c0]
                                        shape_grad = [ho, wo, c0]

                                        self._rewrite_fmap(tik_instance, "vadd",
                                                           f_map_fp32_buf[map_index],
                                                           grad_sel_fp32_buf[mask_index],
                                                           f_map_fp32_buf[map_index],
                                                           "float32", wo*c0//2, ho,
                                                           shape_map_hw, shape_grad,
                                                           config=config)

                                        self._rewrite_fmap(tik_instance, "vadd",
                                                           f_map_fp32_buf[map_index+8],
                                                           grad_sel_fp32_buf[mask_index+8],
                                                           f_map_fp32_buf[map_index+8],
                                                           "float32", wo*c0//2, ho,
                                                           shape_map_hw, shape_grad,
                                                           config=config)
                                else:
                                    # map_index has three part: which hwc0 in
                                    # which window, begin_index of kernel,
                                    # begin_index of child kernel
                                    with tik_instance.for_range(0, ho) as ho_idx:
                                        map_index = (idx_do*self.sd+idx_d)*hi*wi*c0 + \
                                                    (ho_idx*self.sh*wi*c0) + \
                                                    (idx_h*wi*c0+idx_w*c0)
                                        mask_index = wo * ho_idx * c0

                                        self._vector_op(tik_instance, "vadd",
                                                        f_map_fp32_buf[map_index],
                                                        grad_sel_fp32_buf[mask_index],
                                                        f_map_fp32_buf[map_index],
                                                        "float32", wo*c0//2,
                                                        stride_config=(self.sw*2,
                                                                       self.sw*2, 2,
                                                                       self.sw*16,
                                                                       self.sw*16, 16))
                                        self._vector_op(tik_instance, "vadd",
                                                        f_map_fp32_buf[map_index+8],
                                                        grad_sel_fp32_buf[mask_index+8],
                                                        f_map_fp32_buf[map_index + 8],
                                                        "float32", wo * c0 // 2,
                                                        stride_config=(self.sw*2,
                                                                       self.sw*2, 2,
                                                                       self.sw*16,
                                                                       self.sw*16, 16))

                # ---mov_out---
                dst_ou_gm = merchant * _prod(self.forward_in_shape[1:]) + \
                            remainder * hi * wi * c0 + \
                            di_coordinate * c1 * hi * wi * c0

                # effective boundary of d
                boundary_d = self.d - max(0, self.di_invalid)
                # di_coordinate + di < boundary_d means:
                # last effective kernel need SPECIAL TREATMENT
                with tik_instance.if_scope(di_coordinate + di < boundary_d):
                    if self.kd >= self.sd:
                        # move accumulated data to gm
                        ub2gm_shape = [di-self.overlap_d, hi, wi, c0]
                        self._copy_ub_to_gm(tik_instance, f_map_fp32_buf, 0,
                                            self.ou_y_gm, dst_ou_gm,
                                            ub2gm_shape)
                        # move overlap data to ub and vec_dup
                        ub2ub_shape = [di, hi, wi, c0]
                        num_overlap = _prod(ub2ub_shape) // di * self.overlap_d
                        num_init_zero = _prod(ub2ub_shape) - num_overlap
                        self._mov_init(tik_instance, f_map_fp32_buf, num_overlap,
                                       num_init_zero)
                    else:
                        # in case of sd > kd,
                        # di contains stride
                        ub2gm_shape = [di, hi, wi, c0]
                        self._copy_ub_to_gm(tik_instance, f_map_fp32_buf, 0,
                                            self.ou_y_gm, dst_ou_gm,
                                            ub2gm_shape)
                        self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                                            f_map_fp32_buf, 0, 0, "float32")

                with tik_instance.else_scope():
                    if self.kd >= self.sd:
                        # the last kernel
                        ub2gm_shape = [di, hi, wi, c0]
                        self._copy_ub_to_gm(tik_instance, f_map_fp32_buf, 0,
                                            self.ou_y_gm, dst_ou_gm,
                                            ub2gm_shape)
                        self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                                            f_map_fp32_buf, 0, 0, "float32")
                        if self.di_invalid != 0:
                            dst_ou_gm_new = dst_ou_gm + di * c1 * hi * wi * c0
                            ub2gm_shape = [self.di_invalid, hi, wi, c0]
                            self._copy_ub_to_gm(tik_instance, f_map_fp32_buf, 0,
                                                self.ou_y_gm, dst_ou_gm_new,
                                                ub2gm_shape)
                    else:
                        # useful data
                        if self.di_invalid <= 0:
                            # overlap_d make di exceed self.d
                            ub2gm_shape = [di+self.di_invalid, hi, wi, c0]
                            self._copy_ub_to_gm(tik_instance, f_map_fp32_buf, 0,
                                                self.ou_y_gm, dst_ou_gm,
                                                ub2gm_shape)
                            self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                                                f_map_fp32_buf, 0, 0, "float32")
                        else:
                            ub2gm_shape = [di, hi, wi, c0]
                            self._copy_ub_to_gm(tik_instance, f_map_fp32_buf, 0,
                                                self.ou_y_gm, dst_ou_gm,
                                                ub2gm_shape)
                            dst_ou_gm_new = dst_ou_gm + di * c1 * hi * wi * c0
                            ub2gm_shape = [self.di_invalid, hi, wi, c0]
                            self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                                                f_map_fp32_buf, 0, 0, "float32")
                            self._copy_ub_to_gm(tik_instance, f_map_fp32_buf, 0,
                                                self.ou_y_gm, dst_ou_gm_new,
                                                ub2gm_shape)

            with tik_instance.for_range(0, loop_do) as idx:
                # idx+1 represent kernel_d filter next position,
                # if self.overlap_d > 0, result of idx would be
                # used init idx+1(include tail)
                _main(idx, di_batch, do_batch)

            if do_tail != 0:
                _main(loop_do, di_tail, do_tail)

    def tiling_do_ho_main(self, tik_instance, core_loop,
                          sum_core, model, param):
        '''
        ============================
        Just only split do, ho
        ============================
        '''
        do_batch = model[0]
        ho_batch = model[1]
        wo = model[2]
        c0 = self.c0
        c1 = self.c1

        # batch + tail
        loop_do = self.do // do_batch
        loop_ho = self.ho // ho_batch
        ho_tail = self.ho % ho_batch

        di_batch, hi_batch, wi = self._infer_dim_return(do_batch, ho_batch,
                                                        wo, True)
        di_tail, hi_tail, wi_tail = self._infer_dim_return(do_batch, ho_tail,
                                                           wo, True)
        if ho_tail == 0:
            hi_tail = 0

        buf_list = self._set_buf_tensor(tik_instance, param)
        l1_in_buf = buf_list[0]
        forward_ou_buf = buf_list[1]
        grad_buf = buf_list[2]
        col_in_buf = buf_list[3]
        zero_buf = buf_list[7]
        grad_sel_fp16_buf = buf_list[8]
        grad_sel_fp32_buf = buf_list[9]
        f_map_fp32_buf = buf_list[10]
        self.set_vector_dup(tik_instance, param.zero_size,
                            zero_buf, 0, 0, self.dtype)
        self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                            f_map_fp32_buf, 0, 0, "float32")

        with tik_instance.for_range(0, core_loop) as num_core_loop:
            merchant = (sum_core + num_core_loop) // c1
            remainder = (sum_core + num_core_loop) % c1

            def _main(loop_do_idx, loop_ho_idx, di, do, hi, ho):
                # ==========================
                # Init_Begin_Idx
                # ==========================
                if self.kd >= self.sd:
                    di_coordinate = loop_do_idx * \
                                    (di_batch-self.overlap_d)
                else:
                    di_coordinate = loop_do_idx * di_batch

                if self.kh >= self.sh:
                    hi_coordinate = loop_ho_idx * \
                                    (hi_batch-self.overlap_h)
                else:
                    hi_coordinate = loop_ho_idx * hi_batch

                do_coordinate = loop_do_idx * do_batch
                ho_coordinate = loop_ho_idx * ho_batch

                src_orig_x_gm = merchant * _prod(self.forward_in_shape[1:]) + \
                                remainder * self.h * wi * c0 + \
                                di_coordinate * c1 * self.h * wi * c0 + \
                                hi_coordinate * wi * c0
                src_orig_y_gm = merchant * _prod(self.forward_ou_shape[1:]) + \
                                remainder * self.ho * wo * c0 + \
                                do_coordinate * c1 * self.ho * wo * c0 + \
                                ho_coordinate * wo * c0
                src_grad_gm = src_orig_y_gm
                # ================================
                # COPY_GM_2_L1_BUF
                # Prevent reading gm out of bounds
                # ================================
                with tik_instance.if_scope(di_coordinate + di <= self.d):
                    with tik_instance.if_scope(hi_coordinate + hi <= self.h):
                        self._gm2l1_tiling_do_ho(tik_instance, l1_in_buf,
                                                 src_orig_x_gm, 0,
                                                 [di, hi, wi, c0],
                                                 hi_batch)
                    with tik_instance.else_scope():
                        if self.overlap_h < 0:
                            self._gm2l1_tiling_do_ho(tik_instance, l1_in_buf,
                                                     src_orig_x_gm, 0,
                                                     [di, hi+self.overlap_h,
                                                      wi, c0],
                                                     hi_batch)

                with tik_instance.else_scope():
                    if self.overlap_d < 0:
                        with tik_instance.if_scope(hi_coordinate+hi <= self.h):
                            self._gm2l1_tiling_do_ho(tik_instance, l1_in_buf,
                                                     src_orig_x_gm, 0,
                                                     [di+self.overlap_d,
                                                      hi, wi, c0],
                                                     hi_batch)
                        with tik_instance.else_scope():
                            if self.overlap_h < 0:
                                self._gm2l1_tiling_do_ho(tik_instance,
                                                         l1_in_buf,
                                                         src_orig_x_gm, 0,
                                                         [di+self.overlap_d,
                                                          hi+self.overlap_h,
                                                          wi, c0],
                                                         hi_batch)

                # ================================
                # COPY_ORI_OUTPUT_2_FORWARD_OU_BUF
                # COPY_GRAD_2_GRAD_BUF
                # ================================
                self._copy_gm_to_ub(tik_instance, forward_ou_buf,
                                    self.orig_y_gm, src_orig_y_gm,
                                    [do, ho, wo, c0])

                self._copy_gm_to_ub(tik_instance, grad_buf, self.grads_gm,
                                    src_grad_gm, [do, ho, wo, c0])

                # ================================
                # load3d l1 to col_in_buffer
                # ================================
                repeat_times = _ceil_div(ho*wo, 16)
                with tik_instance.for_range(0, do, thread_num=1) as idx_do:
                    with tik_instance.for_range(0, self.kd, thread_num=1) as idx_d:
                        with tik_instance.for_range(0, self.kh, thread_num=1) as idx_h:
                            with tik_instance.for_range(0, self.kw, thread_num=1) as idx_w:
                                src_l1 = (idx_do * self.sd + idx_d) * hi_batch * wi * c0
                                tik_instance.load3dv1(col_in_buf[0],
                                                      l1_in_buf[src_l1],
                                                      [0, 0, 0, 0],
                                                      hi, wi, 0,
                                                      idx_w, idx_h,
                                                      0, 0,
                                                      self.sw, self.sh,
                                                      self.kw, self.kh,
                                                      1, 1, 1, 1,
                                                      repeat_times, 0,
                                                      MIN_VALUE_FP16
                                                      )

                                # ---calculate mask---
                                idx_list = [idx_do, idx_d, idx_h, idx_w]
                                const_list = [ho, wo, c0]
                                self._calc_mask(tik_instance, buf_list, param,
                                                idx_list, const_list)

                                # ---sel(grad,zero,mask)---
                                self._sel(tik_instance, buf_list,
                                          idx_list, const_list)

                                # ---vconv grad_sel_fp16 to fp32---
                                self._vconv(tik_instance, grad_sel_fp16_buf, 0,
                                            grad_sel_fp32_buf, 0,
                                            param.grad_sel_fp16_size, "float16")

                                # ---rewrite grad_sel_fp32 to f_map_fp32
                                config = (self.sw*2, self.sw*2, 2,
                                          self.sh*wi*2, self.sh*wi*2, wo*2)
                                if _check_config(config):
                                    with tik_instance.for_range(0, 1) as ho_idx:
                                        map_index = (idx_do*self.sd+idx_d)*hi_batch*wi*c0 + \
                                                    (idx_h*wi*c0+idx_w*c0)
                                        mask_index = wo * ho_idx * c0
                                        shape_map_hw = [hi_batch, wi, c0]
                                        shape_grad = [ho, wo, c0]

                                        self._rewrite_fmap(tik_instance, "vadd",
                                                           f_map_fp32_buf[map_index],
                                                           grad_sel_fp32_buf[mask_index],
                                                           f_map_fp32_buf[map_index],
                                                           "float32", wo*c0//2, ho,
                                                           shape_map_hw, shape_grad,
                                                           config=config)

                                        self._rewrite_fmap(tik_instance, "vadd",
                                                           f_map_fp32_buf[map_index+8],
                                                           grad_sel_fp32_buf[mask_index+8],
                                                           f_map_fp32_buf[map_index+8],
                                                           "float32", wo*c0//2, ho,
                                                           shape_map_hw, shape_grad,
                                                           config=config)
                                else:
                                    # map_index has three part: which hwc0 in
                                    # which window, begin_index of kernel,
                                    # begin_index of child kernel
                                    with tik_instance.for_range(0, ho) as ho_idx:
                                        map_index = (idx_do*self.sd+idx_d)*hi_batch*wi*c0 + \
                                                    (ho_idx*self.sh*wi*c0) + \
                                                    (idx_h*wi*c0+idx_w*c0)
                                        mask_index = wo * ho_idx * c0

                                        self._vector_op(tik_instance, "vadd",
                                                        f_map_fp32_buf[map_index],
                                                        grad_sel_fp32_buf[mask_index],
                                                        f_map_fp32_buf[map_index],
                                                        "float32", wo*c0//2,
                                                        stride_config=(self.sw*2,
                                                                       self.sw*2, 2,
                                                                       self.sw*16,
                                                                       self.sw*16, 16))
                                        self._vector_op(tik_instance, "vadd",
                                                        f_map_fp32_buf[map_index+8],
                                                        grad_sel_fp32_buf[mask_index+8],
                                                        f_map_fp32_buf[map_index + 8],
                                                        "float32", wo * c0 // 2,
                                                        stride_config=(self.sw*2,
                                                                       self.sw*2, 2,
                                                                       self.sw*16,
                                                                       self.sw*16, 16))

                # mov_out
                dst_ou_gm = merchant * _prod(self.forward_in_shape[1:]) + \
                            remainder * self.h * wi * c0 + \
                            di_coordinate * c1 * self.h * wi * c0 + \
                            hi_coordinate * wi * c0

                def mov_atomic(num_d, dst, dst_idx, src_idx):
                    boundary_h = self.h - max(0, self.hi_invalid)
                    if self.kh > self.sh:
                        # ================================
                        # Split kernels as n-1 and the nth
                        # ================================
                        with tik_instance.if_scope(hi_coordinate+hi < boundary_h):
                            # ==============================
                            # move accumulated data to gm
                            # ==============================
                            in_shape = [num_d, hi-self.overlap_h, wi, c0]
                            self._ub2gm_split_do_ho(tik_instance,
                                                    f_map_fp32_buf,
                                                    src_idx, dst,
                                                    dst_idx, in_shape)

                            # ==============================
                            # mov to init and vec_dup
                            # ==============================
                            in_shape = [num_d, hi, wi, c0]
                            overlap = [num_d, self.overlap_h, wi, c0]
                            non_overlap = [num_d, hi-self.overlap_h, wi, c0]

                            n_burst = in_shape[0]
                            burst_len = _prod(overlap[1:]) * \
                                        self.num_bit_fp32 // MINI_UNIT
                            src_stride = _prod(non_overlap[1:]) * \
                                         self.num_bit_fp32 // MINI_UNIT
                            dst_stride = _prod(non_overlap[1:]) * \
                                         self.num_bit_fp32 // MINI_UNIT
                            tik_instance.data_move(
                                f_map_fp32_buf[src_idx],
                                f_map_fp32_buf[src_idx + _prod(non_overlap[1:])],
                                0,
                                n_burst,
                                burst_len,
                                src_stride,
                                dst_stride)

                            # vec_dup for next ho_idx
                            num_zero = _prod(non_overlap[1:])
                            for i in range(in_shape[0]):
                                dst_vec_idx = src_idx + _prod(overlap[1:]) + i *\
                                              _prod(in_shape[1:])
                                self.set_vector_dup(tik_instance, num_zero,
                                                    f_map_fp32_buf, dst_vec_idx,
                                                    0, "float32")

                        with tik_instance.else_scope():
                            in_shape = [num_d, hi, wi, c0]
                            # if tail_h existed, ub2gm has different model
                            self._ub2gm_split_do_ho_2(tik_instance,
                                                      f_map_fp32_buf,
                                                      src_idx, dst,
                                                      dst_idx, in_shape,
                                                      hi_batch)

                            self.set_vector_dup(tik_instance,
                                                param.f_map_fp32_size,
                                                f_map_fp32_buf, 0, 0, "float32")

                    elif self.kh == self.sh:
                        in_shape = [num_d, hi, wi, c0]
                        self._ub2gm_split_do_ho_2(tik_instance,
                                                  f_map_fp32_buf,
                                                  src_idx, dst,
                                                  dst_idx, in_shape, hi_batch)

                        self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                                            f_map_fp32_buf, 0, 0, "float32")

                    else:
                        if self.hi_invalid >= 0:
                            in_shape = [num_d, hi, wi, c0]
                            self._ub2gm_split_do_ho_2(tik_instance,
                                                      f_map_fp32_buf,
                                                      src_idx, dst,
                                                      dst_idx, in_shape,
                                                      hi_batch)

                            self.set_vector_dup(tik_instance,
                                                param.f_map_fp32_size,
                                                f_map_fp32_buf, 0, 0, "float32")
                        else:
                            with tik_instance.if_scope(hi_coordinate+hi < boundary_h):
                                in_shape = [num_d, hi, wi, c0]
                                self._ub2gm_split_do_ho_2(tik_instance,
                                                          f_map_fp32_buf,
                                                          src_idx, dst,
                                                          dst_idx, in_shape,
                                                          hi_batch)

                                self.set_vector_dup(tik_instance,
                                                    param.f_map_fp32_size,
                                                    f_map_fp32_buf, 0, 0,
                                                    "float32")
                            with tik_instance.else_scope():
                                in_shape = [num_d, hi+self.hi_invalid, wi, c0]
                                self._ub2gm_split_do_ho_2(tik_instance,
                                                          f_map_fp32_buf,
                                                          src_idx, dst,
                                                          dst_idx, in_shape,
                                                          hi_batch)
                                self.set_vector_dup(tik_instance,
                                                    param.f_map_fp32_size,
                                                    f_map_fp32_buf, 0, 0,
                                                    "float32")

                if self.kd >= self.sd:
                    tik_instance.set_atomic_add(1)
                    mov_atomic(di, self.ou_y_gm, dst_ou_gm, 0)
                    tik_instance.set_atomic_add(0)
                else:
                    # di_invalid can less than 0
                    tik_instance.set_atomic_add(1)
                    if self.di_invalid >= 0:
                        mov_atomic(di, self.ou_y_gm, dst_ou_gm, 0)
                    else:
                        with tik_instance.if_scope(di_coordinate+di <= self.d):
                            mov_atomic(di, self.ou_y_gm, dst_ou_gm, 0)
                        with tik_instance.else_scope():
                            mov_atomic(di+self.di_invalid,
                                       self.ou_y_gm, dst_ou_gm, 0)
                    tik_instance.set_atomic_add(0)

            if ho_tail != 0:
                with tik_instance.for_range(0, loop_do) as do_idx:
                    with tik_instance.for_range(0, loop_ho) as ho_idx:
                        _main(do_idx, ho_idx, di_batch, do_batch,
                              hi_batch, ho_batch)
                    _main(do_idx, loop_ho, di_tail, do_batch, hi_tail, ho_tail)
            else:
                with tik_instance.for_range(0, loop_do) as do_idx:
                    with tik_instance.for_range(0, loop_ho) as ho_idx:
                        _main(do_idx, ho_idx, di_batch, do_batch,
                              hi_batch, ho_batch)

    def tiling_do_ho_wo_main(self, tik_instance, core_loop,
                             sum_core, model, param):
        '''
        ============================
        Just split do, ho, wo
        ============================
        '''
        do_batch = model[0]
        ho_batch = model[1]
        wo_batch = model[2]
        c0 = self.c0
        c1 = self.c1

        if do_batch != ho_batch != 1:
            raise RuntimeError("In the branch, do_batch and "
                               "ho_batch should be 1.")

        loop_do = self.do // do_batch
        loop_ho = self.ho // ho_batch
        loop_wo = self.wo // wo_batch
        wo_tail = self.wo % wo_batch
        di_batch, hi_batch, wi_batch = self._infer_dim_return(do_batch,
                                                              ho_batch,
                                                              wo_batch, True)
        di_tail, hi_tail, wi_tail = self._infer_dim_return(do_batch,
                                                           ho_batch,
                                                           wo_tail, True)

        buf_list = self._set_buf_tensor(tik_instance, param)
        l1_in_buf = buf_list[0]
        forward_ou_buf = buf_list[1]
        grad_buf = buf_list[2]
        col_in_buf = buf_list[3]
        zero_buf = buf_list[7]
        grad_sel_fp16_buf = buf_list[8]
        grad_sel_fp32_buf = buf_list[9]
        f_map_fp32_buf = buf_list[10]
        self.set_vector_dup(tik_instance, param.zero_size,
                            zero_buf, 0, 0, self.dtype)
        self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                            f_map_fp32_buf, 0, 0, "float32")

        with tik_instance.for_range(0, core_loop) as num_core_loop:
            merchant = (sum_core + num_core_loop) // c1
            remainder = (sum_core + num_core_loop) % c1

            def _main(loop_do_idx, loop_ho_idx, loop_wo_idx,
                      di, do, hi, ho, wi, wo):
                # ==========================
                # Init_Begin_Idx
                # ==========================
                if self.kd >= self.sd:
                    di_coordinate = loop_do_idx * \
                                    (di_batch-self.overlap_d)
                else:
                    di_coordinate = loop_do_idx * di_batch

                if self.kh >= self.sh:
                    hi_coordinate = loop_ho_idx * \
                                    (hi_batch-self.overlap_h)
                else:
                    hi_coordinate = loop_ho_idx * hi_batch

                if self.kw >= self.sw:
                    wi_coordinate = loop_wo_idx * \
                                    (wi_batch-self.overlap_w)
                else:
                    wi_coordinate = loop_wo_idx * wi_batch

                do_coordinate = loop_do_idx * do_batch
                ho_coordinate = loop_ho_idx * ho_batch
                wo_coordinate = loop_wo_idx * wo_batch

                src_orig_x_gm = merchant * _prod(self.forward_in_shape[1:]) + \
                                remainder * self.h * self.w * c0 + \
                                di_coordinate * c1 * self.h * self.w * c0 + \
                                hi_coordinate * self.w * c0 + \
                                wi_coordinate * c0
                src_orig_y_gm = merchant * _prod(self.forward_ou_shape[1:]) + \
                                remainder * self.ho * self.wo * c0 + \
                                do_coordinate * c1 * self.ho * self.wo * c0 + \
                                ho_coordinate * self.wo * c0 + \
                                wo_coordinate * c0
                src_grad_gm = src_orig_y_gm

                # ================================
                # COPY_GM_2_L1_BUF
                # Prevent reading gm out of bounds
                # ================================
                di_val = min(0, self.overlap_d) + di
                hi_val = min(0, self.overlap_h) + hi
                wi_val = min(0, self.overlap_w) + wi
                input0 = [di_val, hi_val, wi_val]
                input1 = [di_batch, hi_batch, wi_batch]
                self._gm2l1_tiling_do_ho_wo(tik_instance,
                                            l1_in_buf, src_orig_x_gm, 0,
                                            input0, input1)

                # ================================
                # COPY_ORI_OUTPUT_2_FORWARD_OU_BUF
                # COPY_GRAD_2_GRAD_BUF
                # in the branch, do and ho are 1.
                # ================================
                self._copy_gm_to_ub(tik_instance, forward_ou_buf,
                                    self.orig_y_gm, src_orig_y_gm,
                                    [do, ho, wo, c0])

                self._copy_gm_to_ub(tik_instance, grad_buf, self.grads_gm,
                                    src_grad_gm, [do, ho, wo, c0])

                # ================================
                # load3d l1 to col_in_buffer
                # ================================
                repeat_times = _ceil_div(ho*wo, 16)
                with tik_instance.for_range(0, do, thread_num=1) as idx_do:
                    with tik_instance.for_range(0, self.kd, thread_num=1) as idx_d:
                        with tik_instance.for_range(0, self.kh, thread_num=1) as idx_h:
                            with tik_instance.for_range(0, self.kw, thread_num=1) as idx_w:
                                src_l1 = (idx_do * self.sd + idx_d) * hi_batch * wi_batch * c0
                                tik_instance.load3dv1(col_in_buf[0],
                                                      l1_in_buf[src_l1],
                                                      [0, 0, 0, 0],
                                                      hi_val, wi_val, 0,
                                                      idx_w, idx_h,
                                                      0, 0,
                                                      self.sw, self.sh,
                                                      self.kw, self.kh,
                                                      1, 1, 1, 1,
                                                      repeat_times, 0,
                                                      MIN_VALUE_FP16
                                                      )

                                # ---calculate mask---
                                idx_list = [idx_do, idx_d, idx_h, idx_w]
                                const_list = [ho, wo, c0]
                                self._calc_mask(tik_instance, buf_list, param,
                                                idx_list, const_list)

                                # ---sel(grad,zero,mask)---
                                self._sel(tik_instance, buf_list,
                                          idx_list, const_list)

                                # ---vconv grad_sel_fp16 to fp32---
                                self._vconv(tik_instance, grad_sel_fp16_buf, 0,
                                            grad_sel_fp32_buf, 0,
                                            param.grad_sel_fp16_size, "float16")

                                # ---rewrite grad_sel_fp32 to f_map_fp32
                                # do = 1, ho = 1
                                # map_index has two part: begin_index of kernel,
                                # begin_index of child kernel
                                # must use tik variable as index of grad_sel_fp32_buf,
                                # python variable is not work in grad_sel_fp32_buf[mask_index],
                                # while x = grad_sel_fp32_buf[mask_index], y = x[n].
                                with tik_instance.for_range(0, 1) as index_mask:
                                    map_index = idx_d * hi_batch * wi_batch * c0 + \
                                                idx_h * wi_batch * c0 + idx_w * c0
                                    mask_index = index_mask

                                    self._vector_op(tik_instance, "vadd",
                                                    f_map_fp32_buf[map_index],
                                                    grad_sel_fp32_buf[mask_index],
                                                    f_map_fp32_buf[map_index],
                                                    "float32", wo*c0//2,
                                                    stride_config=(self.sw*2,
                                                                   self.sw*2, 2,
                                                                   self.sw*16,
                                                                   self.sw*16, 16))
                                    self._vector_op(tik_instance, "vadd",
                                                    f_map_fp32_buf[map_index+8],
                                                    grad_sel_fp32_buf[mask_index+8],
                                                    f_map_fp32_buf[map_index + 8],
                                                    "float32", wo*c0//2,
                                                    stride_config=(self.sw*2,
                                                                   self.sw*2, 2,
                                                                   self.sw*16,
                                                                   self.sw*16, 16))

                # mov_out
                dst_ou_gm = merchant * _prod(self.forward_in_shape[1:]) + \
                            remainder * self.h * self.w * c0 + \
                            di_coordinate * c1 * self.h * self.w * c0 + \
                            hi_coordinate * self.w * c0 + \
                            wi_coordinate * c0

                def mov_atomic(num_d, dst, dst_idx, src_idx):
                    num_h = hi + min(0, self.overlap_h)
                    num_w = wi + min(0, self.overlap_w)
                    in_shape = [num_d, num_h, num_w, c0]
                    self._ub2gm_split_do_ho_wo(tik_instance, f_map_fp32_buf,
                                               src_idx, dst, dst_idx,
                                               in_shape, hi_batch, wi_batch)

                    self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                                        f_map_fp32_buf, 0, 0, "float32")

                tik_instance.set_atomic_add(1)
                mov_atomic(di + min(0, self.overlap_d),
                           self.ou_y_gm, dst_ou_gm, 0)
                tik_instance.set_atomic_add(0)

            if wo_tail != 0:
                with tik_instance.for_range(0, loop_do) as do_idx:
                    with tik_instance.for_range(0, loop_ho) as ho_idx:
                        with tik_instance.for_range(0, loop_wo) as wo_idx:
                            _main(do_idx, ho_idx, wo_idx,
                                  di_batch, do_batch,
                                  hi_batch, ho_batch,
                                  wi_batch, wo_batch)
                        _main(do_idx, ho_idx, loop_wo,
                              di_tail, do_batch,
                              hi_tail, ho_batch,
                              wi_tail, wo_tail)
            else:
                with tik_instance.for_range(0, loop_do) as do_idx:
                    with tik_instance.for_range(0, loop_ho) as ho_idx:
                        with tik_instance.for_range(0, loop_wo) as wo_idx:
                            _main(do_idx, ho_idx, wo_idx,
                                  di_batch, do_batch,
                                  hi_batch, ho_batch,
                                  wi_batch, wo_batch)

    def pure_atomic_tiling_do(self, tik_instance, core_loop,
                              sum_core, model, param):
        '''
        ==================================================
        In the case, do must be split as part of core_axis.
        Solution:
        0: split do as core, not_tiling
        1: split do as core, tiling_do
        ==================================================
        '''
        do_batch = model[0]
        ho_batch = model[1]
        wo_batch = model[2]
        c0 = self.c0
        c1 = self.c1

        # batch + tail
        loop_do = self.core_ou_shape[0] // do_batch
        di_batch, hi_batch, wi_batch = self._infer_dim_return(do_batch,
                                                              ho_batch,
                                                              wo_batch, True)
        do_tail = self.core_ou_shape[0] % do_batch
        di_tail, hi_tail, wi_tail = self._infer_dim_return(do_tail,
                                                           ho_batch,
                                                           wo_batch, True)

        buf_list = self._set_buf_tensor(tik_instance, param)
        l1_in_buf = buf_list[0]
        forward_ou_buf = buf_list[1]
        grad_buf = buf_list[2]
        col_in_buf = buf_list[3]
        zero_buf = buf_list[7]
        grad_sel_fp16_buf = buf_list[8]
        grad_sel_fp32_buf = buf_list[9]
        f_map_fp32_buf = buf_list[10]
        self.set_vector_dup(tik_instance, param.zero_size,
                            zero_buf, 0, 0, self.dtype)
        self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                            f_map_fp32_buf, 0, 0, "float32")

        with tik_instance.for_range(0, core_loop) as num_core_loop:
            # ======================
            # Assume  ori_output_shape is [n,do,c1,ho,wo,c0],
            # split [n,do,c1] as core_num.
            # merchant: index of n-axis
            # merchant_c1: index of c1-axis
            # remainder_c1: index of do-axis
            # ======================
            core_do = self.core_ou_shape[0]
            core_do_times = self.do // core_do
            core_di = self.core_in_shape[0]

            merchant = (sum_core + num_core_loop) // (c1 * core_do_times)
            remainder = (sum_core + num_core_loop) % (c1 * core_do_times)
            merchant_c1 = remainder // core_do_times
            remainder_c1 = remainder % core_do_times

            def _main(loop_idx, di, do):
                # ----Init_Begin_Idx----
                if self.kd >= self.sd:
                    di_coordinate = loop_idx * (di_batch-self.overlap_d) + \
                                    remainder_c1 * (core_di-self.overlap_d)
                else:
                    di_coordinate = loop_idx * di_batch + \
                                    remainder_c1 * core_di
                do_coordinate = loop_idx * do_batch + remainder_c1 * core_do

                src_orig_x_gm = merchant * _prod(self.forward_in_shape[1:]) + \
                                merchant_c1 * self.h * self.w * c0 + \
                                di_coordinate * c1 * self.h * self.w * c0
                src_orig_y_gm = merchant * _prod(self.forward_ou_shape[1:]) + \
                                merchant_c1 * self.ho * self.wo * c0 + \
                                do_coordinate * c1 * self.ho * self.wo * c0
                src_grad_gm = src_orig_y_gm

                # ----COPY_GM_2_L1_BUF----
                # Prevent reading gm out of bounds
                # which only happened in kd<sd
                with tik_instance.if_scope(di_coordinate + di <= self.d):
                    self._copy_gm_to_l1(tik_instance, l1_in_buf,
                                        src_orig_x_gm, 0,
                                        [di, hi_batch, wi_batch, c0])
                with tik_instance.else_scope():
                    self._copy_gm_to_l1(tik_instance, l1_in_buf,
                                        src_orig_x_gm, 0,
                                        [di+self.overlap_d, hi_batch,
                                         wi_batch, c0])

                # ----COPY_ORI_OUTPUT_2_FORWARD_OU_BUF----
                self._copy_gm_to_ub(tik_instance, forward_ou_buf,
                                    self.orig_y_gm, src_orig_y_gm,
                                    [do, ho_batch, wo_batch, c0])

                # ----COPY_GRAD_2_GRAD_BUF----
                self._copy_gm_to_ub(tik_instance, grad_buf, self.grads_gm,
                                    src_grad_gm, [do, ho_batch, wo_batch, c0])

                # ---load3d l1 to col_in_buffer---
                repeat_times = _ceil_div(ho_batch*wo_batch, 16)
                # which window
                with tik_instance.for_range(0, do, thread_num=1) as idx_do:
                    # which hwc0
                    with tik_instance.for_range(0, self.kd, thread_num=1) as idx_d:
                        with tik_instance.for_range(0, self.kh, thread_num=1) as idx_h:
                            with tik_instance.for_range(0, self.kw, thread_num=1) as idx_w:
                                src_l1 = (idx_do * self.sd + idx_d) * hi_batch * wi_batch * c0
                                tik_instance.load3dv1(col_in_buf[0],
                                                      l1_in_buf[src_l1],
                                                      [0, 0, 0, 0],
                                                      hi_batch, wi_batch, 0,
                                                      idx_w, idx_h,
                                                      0, 0,
                                                      self.sw, self.sh,
                                                      self.kw, self.kh,
                                                      1, 1, 1, 1,
                                                      repeat_times, 0,
                                                      MIN_VALUE_FP16
                                                      )

                                # ---calculate mask---
                                idx_list = [idx_do, idx_d, idx_h, idx_w]
                                const_list = [ho_batch, wo_batch, c0]
                                self._calc_mask(tik_instance, buf_list, param,
                                                idx_list, const_list)

                                # ---sel(grad,zero,mask)---
                                # v100 only support repeat_times_sel = 1
                                self._sel(tik_instance, buf_list,
                                          idx_list, const_list)

                                # ---vconv grad_sel_fp16 to fp32---
                                self._vconv(tik_instance, grad_sel_fp16_buf, 0,
                                            grad_sel_fp32_buf, 0,
                                            param.grad_sel_fp16_size, "float16")

                                # ---rewrite grad_sel_fp32 to f_map_fp32
                                config = (self.sw*2, self.sw*2, 2,
                                          self.sh*wi_batch*2, self.sh*wi_batch*2, wo_batch*2)
                                if _check_config(config):
                                    with tik_instance.for_range(0, 1) as ho_idx:
                                        map_index = (idx_do*self.sd+idx_d)*hi_batch*wi_batch*c0 + \
                                                    (idx_h*wi_batch*c0+idx_w*c0)
                                        mask_index = wo_batch * ho_idx * c0
                                        shape_map_hw = [hi_batch, wi_batch, c0]
                                        shape_grad = [ho_batch, wo_batch, c0]

                                        self._rewrite_fmap(tik_instance, "vadd",
                                                           f_map_fp32_buf[map_index],
                                                           grad_sel_fp32_buf[mask_index],
                                                           f_map_fp32_buf[map_index],
                                                           "float32", wo_batch*c0//2, ho_batch,
                                                           shape_map_hw, shape_grad,
                                                           config=config)

                                        self._rewrite_fmap(tik_instance, "vadd",
                                                           f_map_fp32_buf[map_index+8],
                                                           grad_sel_fp32_buf[mask_index+8],
                                                           f_map_fp32_buf[map_index+8],
                                                           "float32", wo_batch*c0//2, ho_batch,
                                                           shape_map_hw, shape_grad,
                                                           config=config)
                                else:
                                    # map_index has three part: which hwc0 in
                                    # which window, begin_index of kernel,
                                    # begin_index of child kernel
                                    with tik_instance.for_range(0, ho_batch) as ho_idx:
                                        map_index = (idx_do*self.sd+idx_d)*hi_batch*wi_batch*c0 + \
                                                    (ho_idx*self.sh*wi_batch*c0) + \
                                                    (idx_h*wi_batch*c0+idx_w*c0)
                                        mask_index = wo_batch * ho_idx * c0

                                        self._vector_op(tik_instance, "vadd",
                                                        f_map_fp32_buf[map_index],
                                                        grad_sel_fp32_buf[mask_index],
                                                        f_map_fp32_buf[map_index],
                                                        "float32", wo_batch*c0//2,
                                                        stride_config=(self.sw*2,
                                                                       self.sw*2, 2,
                                                                       self.sw*16,
                                                                       self.sw*16, 16))
                                        self._vector_op(tik_instance, "vadd",
                                                        f_map_fp32_buf[map_index+8],
                                                        grad_sel_fp32_buf[mask_index+8],
                                                        f_map_fp32_buf[map_index + 8],
                                                        "float32", wo_batch*c0//2,
                                                        stride_config=(self.sw*2,
                                                                       self.sw*2, 2,
                                                                       self.sw*16,
                                                                       self.sw*16, 16))

                # ---mov_out---
                dst_ou_gm = merchant * _prod(self.forward_in_shape[1:]) + \
                            merchant_c1 * self.h * self.w * c0 + \
                            di_coordinate * c1 * self.h * self.w * c0

                def mov_atomic(num_d, dst, dst_idx, src_idx):
                    ub2gm_shape = [num_d, hi_batch, wi_batch, c0]
                    self._copy_ub_to_gm(tik_instance, f_map_fp32_buf, src_idx,
                                        dst, dst_idx, ub2gm_shape)
                    self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                                        f_map_fp32_buf, 0, 0, "float32")

                tik_instance.set_atomic_add(1)
                mov_atomic((di + min(0, self.overlap_d)),
                           self.ou_y_gm, dst_ou_gm, 0)
                tik_instance.set_atomic_add(0)

            with tik_instance.for_range(0, loop_do) as idx:
                # idx+1 represent kernel_d filter next position,
                # if self.overlap_d > 0, result of idx would be
                # used init idx+1(include tail)
                _main(idx, di_batch, do_batch)

            if do_tail != 0:
                _main(loop_do, di_tail, do_tail)

    def pure_atomic_tiling_do_ho(self, tik_instance, core_loop,
                                 sum_core, model, param):
        '''
        ===================================================
        In the case, do must be split as part of core_axis,
        ho may be split as part of core_axis.
        Solution:
        0: split do as core, tiling_do_ho: do_batch is 1,
        1: split do_ho as core, not_tiling: do_batch is 1,
        2: split do_ho as core, tiling_do: do_batch is 1,
        3: split do_ho as core, tiling_do_ho: do_batch is 1,
        result:
        Only have ho_tail, do_tail is not existed.
        ===================================================
        '''
        do_batch = model[0]
        ho_batch = model[1]
        wo_batch = model[2]
        c0 = self.c0
        c1 = self.c1

        # batch + tail
        loop_do = self.core_ou_shape[0] // do_batch
        loop_ho = self.core_ou_shape[1] // ho_batch
        ho_tail = self.core_ou_shape[1] % ho_batch

        di_batch, hi_batch, wi_batch = self._infer_dim_return(do_batch,
                                                              ho_batch,
                                                              wo_batch, True)
        di_tail, hi_tail, wi_tail = self._infer_dim_return(do_batch,
                                                           ho_tail,
                                                           wo_batch, True)

        buf_list = self._set_buf_tensor(tik_instance, param)
        l1_in_buf = buf_list[0]
        forward_ou_buf = buf_list[1]
        grad_buf = buf_list[2]
        col_in_buf = buf_list[3]
        zero_buf = buf_list[7]
        grad_sel_fp16_buf = buf_list[8]
        grad_sel_fp32_buf = buf_list[9]
        f_map_fp32_buf = buf_list[10]
        self.set_vector_dup(tik_instance, param.zero_size,
                            zero_buf, 0, 0, self.dtype)
        self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                            f_map_fp32_buf, 0, 0, "float32")

        with tik_instance.for_range(0, core_loop) as num_core_loop:
            # ==============================================
            # Assume  ori_output_shape is [n,do,c1,ho,wo,c0],
            # split [n,do,c1,ho] as core_num.
            # merchant: index of n-axis
            # merchant_c1: index of c1-axis
            # merchant_d: index of do-axis
            # remainder_d: index of ho-axis
            # ==============================================
            core_do = self.core_ou_shape[0]
            core_do_times = self.do // core_do
            core_di = self.core_in_shape[0]

            core_ho = self.core_ou_shape[1]
            core_ho_times = self.ho // core_ho
            core_hi = self.core_in_shape[1]

            merchant = (sum_core+num_core_loop) // (c1*core_do_times*core_ho_times)
            remainder = (sum_core+num_core_loop) % (c1*core_do_times*core_ho_times)

            merchant_c1 = remainder // (core_do_times*core_ho_times)
            remainder_c1 = remainder % (core_do_times*core_ho_times)

            merchant_d = remainder_c1 // core_ho_times
            remainder_d = remainder_c1 % core_ho_times

            def _main(loop_do_idx, loop_ho_idx, di, do, hi, ho):
                # ==========================
                # Init_Begin_Idx
                # ==========================
                if self.kd >= self.sd:
                    di_coordinate = loop_do_idx * (di_batch-self.overlap_d) + \
                                    merchant_d * (core_di-self.overlap_d)
                else:
                    di_coordinate = loop_do_idx * di_batch + \
                                    merchant_d * core_di

                if self.kh >= self.sh:
                    hi_coordinate = loop_ho_idx * (hi_batch-self.overlap_h) + \
                                    remainder_d * (core_hi-self.overlap_h)
                else:
                    hi_coordinate = loop_ho_idx * hi_batch + \
                                    remainder_d * core_hi

                do_coordinate = loop_do_idx * do_batch + merchant_d * core_do
                ho_coordinate = loop_ho_idx * ho_batch + remainder_d * core_ho

                src_orig_x_gm = merchant * _prod(self.forward_in_shape[1:]) + \
                                merchant_c1 * self.h * self.w * c0 + \
                                di_coordinate * c1 * self.h * self.w * c0 + \
                                hi_coordinate * self.w * c0
                src_orig_y_gm = merchant * _prod(self.forward_ou_shape[1:]) + \
                                merchant_c1 * self.ho * self.wo * c0 + \
                                do_coordinate * c1 * self.ho * self.wo * c0 + \
                                ho_coordinate * self.wo * c0
                src_grad_gm = src_orig_y_gm

                # ================================
                # COPY_GM_2_L1_BUF
                # Prevent reading gm out of bounds
                # ================================
                di_val = min(0, self.overlap_d) + di
                hi_val = min(0, self.overlap_h) + hi
                in_shape = [di_val, hi_val, wi_batch, c0]
                self._gm2l1_tiling_do_ho(tik_instance, l1_in_buf,
                                         src_orig_x_gm, 0,
                                         in_shape,
                                         hi_batch)

                # ================================
                # COPY_ORI_OUTPUT_2_FORWARD_OU_BUF
                # COPY_GRAD_2_GRAD_BUF
                # ================================
                self._copy_gm_to_ub(tik_instance, forward_ou_buf,
                                    self.orig_y_gm, src_orig_y_gm,
                                    [do, ho, wo_batch, c0])

                self._copy_gm_to_ub(tik_instance, grad_buf, self.grads_gm,
                                    src_grad_gm, [do, ho, wo_batch, c0])

                # ================================
                # load3d l1 to col_in_buffer
                # ================================
                repeat_times = _ceil_div(ho*wo_batch, 16)
                with tik_instance.for_range(0, do, thread_num=1) as idx_do:
                    with tik_instance.for_range(0, self.kd, thread_num=1) as idx_d:
                        with tik_instance.for_range(0, self.kh, thread_num=1) as idx_h:
                            with tik_instance.for_range(0, self.kw, thread_num=1) as idx_w:
                                src_l1 = (idx_do * self.sd + idx_d) * hi_batch * wi_batch * c0
                                tik_instance.load3dv1(col_in_buf[0],
                                                      l1_in_buf[src_l1],
                                                      [0, 0, 0, 0],
                                                      hi, wi_batch, 0,
                                                      idx_w, idx_h,
                                                      0, 0,
                                                      self.sw, self.sh,
                                                      self.kw, self.kh,
                                                      1, 1, 1, 1,
                                                      repeat_times, 0,
                                                      MIN_VALUE_FP16
                                                      )

                                # ---calculate mask---
                                idx_list = [idx_do, idx_d, idx_h, idx_w]
                                const_list = [ho, wo_batch, c0]
                                self._calc_mask(tik_instance, buf_list, param,
                                                idx_list, const_list)

                                # ---sel(grad,zero,mask)---
                                self._sel(tik_instance, buf_list,
                                          idx_list, const_list)

                                # ---vconv grad_sel_fp16 to fp32---
                                self._vconv(tik_instance, grad_sel_fp16_buf, 0,
                                            grad_sel_fp32_buf, 0,
                                            param.grad_sel_fp16_size, "float16")

                                # ---rewrite grad_sel_fp32 to f_map_fp32
                                config = (self.sw*2, self.sw*2, 2,
                                          self.sh*wi_batch*2, self.sh*wi_batch*2, wo_batch*2)
                                if _check_config(config):
                                    with tik_instance.for_range(0, 1) as ho_idx:
                                        map_index = (idx_do*self.sd+idx_d)*hi_batch*wi_batch*c0 + \
                                                    (idx_h*wi_batch*c0+idx_w*c0)
                                        mask_index = wo_batch * ho_idx * c0
                                        shape_map_hw = [hi_batch, wi_batch, c0]
                                        shape_grad = [ho, wo_batch, c0]

                                        self._rewrite_fmap(tik_instance, "vadd",
                                                           f_map_fp32_buf[map_index],
                                                           grad_sel_fp32_buf[mask_index],
                                                           f_map_fp32_buf[map_index],
                                                           "float32", wo_batch*c0//2, ho,
                                                           shape_map_hw, shape_grad,
                                                           config=config)

                                        self._rewrite_fmap(tik_instance, "vadd",
                                                           f_map_fp32_buf[map_index+8],
                                                           grad_sel_fp32_buf[mask_index+8],
                                                           f_map_fp32_buf[map_index+8],
                                                           "float32", wo_batch*c0//2, ho,
                                                           shape_map_hw, shape_grad,
                                                           config=config)
                                else:
                                    # map_index has three part: which hwc0 in
                                    # which window, begin_index of kernel,
                                    # begin_index of child kernel
                                    with tik_instance.for_range(0, ho) as ho_idx:
                                        map_index = (idx_do*self.sd+idx_d)*hi_batch*wi_batch*c0 + \
                                                    (ho_idx*self.sh*wi_batch*c0) + \
                                                    (idx_h*wi_batch*c0+idx_w*c0)
                                        mask_index = wo_batch * ho_idx * c0

                                        self._vector_op(tik_instance, "vadd",
                                                        f_map_fp32_buf[map_index],
                                                        grad_sel_fp32_buf[mask_index],
                                                        f_map_fp32_buf[map_index],
                                                        "float32", wo_batch*c0//2,
                                                        stride_config=(self.sw*2,
                                                                       self.sw*2, 2,
                                                                       self.sw*16,
                                                                       self.sw*16, 16))
                                        self._vector_op(tik_instance, "vadd",
                                                        f_map_fp32_buf[map_index+8],
                                                        grad_sel_fp32_buf[mask_index+8],
                                                        f_map_fp32_buf[map_index + 8],
                                                        "float32", wo_batch*c0//2,
                                                        stride_config=(self.sw*2,
                                                                       self.sw*2, 2,
                                                                       self.sw*16,
                                                                       self.sw*16, 16))

                # mov_out
                dst_ou_gm = merchant * _prod(self.forward_in_shape[1:]) + \
                            merchant_c1 * self.h * self.w * c0 + \
                            di_coordinate * c1 * self.h * self.w * c0 + \
                            hi_coordinate * self.w * c0

                def mov_atomic(num_d, dst, dst_idx, src_idx):
                    num_h = hi + min(0, self.overlap_h)
                    ub2gm_shape = [num_d, num_h, wi_batch, c0]

                    # mov_out
                    self._ub2gm_split_do_ho_2(tik_instance,
                                              f_map_fp32_buf,
                                              src_idx, dst,
                                              dst_idx, ub2gm_shape, hi_batch)

                    # vec_dup
                    self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                                        f_map_fp32_buf, 0, 0, "float32")

                tik_instance.set_atomic_add(1)
                mov_atomic(di + min(0, self.overlap_d),
                           self.ou_y_gm, dst_ou_gm, 0)
                tik_instance.set_atomic_add(0)

            if ho_tail != 0:
                with tik_instance.for_range(0, loop_do) as do_idx:
                    with tik_instance.for_range(0, loop_ho) as ho_idx:
                        _main(do_idx, ho_idx, di_batch, do_batch,
                              hi_batch, ho_batch)
                    _main(do_idx, loop_ho, di_tail, do_batch, hi_tail, ho_tail)
            else:
                with tik_instance.for_range(0, loop_do) as do_idx:
                    with tik_instance.for_range(0, loop_ho) as ho_idx:
                        _main(do_idx, ho_idx, di_batch, do_batch,
                              hi_batch, ho_batch)

    def pure_atomic_tiling_do_ho_wo(self, tik_instance, core_loop,
                                    sum_core, model, param):
        '''
        ===================================================
        In the case, do must be split as part of core_axis,
        ho, wo may be split as part of core_axis.
        Solution:
        0: split do as core, tiling_do_ho_wo: do_batch, ho_batch is 1
        1: split do_ho as core, tiling_do_ho_wo: do_batch, ho_batch is 1
        2: split do_ho_wo as core, not_tiling: do_batch, ho_batch is 1
        3: split do_ho_wo as core, tiling_do: do_batch, ho_batch is 1
        4: split do_ho_wo as core, tiling_do_ho: do_batch, ho_batch, is 1
        5: split do_ho_wo as core, tiling_do_ho_wo: do_batch, ho_batch, is 1
        result:
        Only have wo_tail, do_tail ho_tail are not existed.
        ===================================================
        '''
        do_batch = model[0]
        ho_batch = model[1]
        wo_batch = model[2]
        c0 = self.c0
        c1 = self.c1

        if do_batch != ho_batch != 1:
            raise RuntimeError("In the branch, do_batch and "
                               "ho_batch should be 1.")

        loop_do = self.core_ou_shape[0] // do_batch
        loop_ho = self.core_ou_shape[1] // ho_batch
        loop_wo = self.core_ou_shape[2] // wo_batch
        wo_tail = self.core_ou_shape[2] % wo_batch
        di_batch, hi_batch, wi_batch = self._infer_dim_return(do_batch,
                                                              ho_batch,
                                                              wo_batch, True)
        di_tail, hi_tail, wi_tail = self._infer_dim_return(do_batch,
                                                           ho_batch,
                                                           wo_tail, True)

        buf_list = self._set_buf_tensor(tik_instance, param)
        l1_in_buf = buf_list[0]
        forward_ou_buf = buf_list[1]
        grad_buf = buf_list[2]
        col_in_buf = buf_list[3]
        zero_buf = buf_list[7]
        grad_sel_fp16_buf = buf_list[8]
        grad_sel_fp32_buf = buf_list[9]
        f_map_fp32_buf = buf_list[10]
        self.set_vector_dup(tik_instance, param.zero_size,
                            zero_buf, 0, 0, self.dtype)
        self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                            f_map_fp32_buf, 0, 0, "float32")

        with tik_instance.for_range(0, core_loop) as num_core_loop:
            # ==============================================
            # Assume  ori_output_shape is [n,do,c1,ho,wo,c0],
            # split [n,do,c1,ho] as core_num.
            # merchant: index of n-axis
            # merchant_c1: index of c1-axis
            # merchant_d: index of do-axis
            # merchant_h: index of ho-axis
            # remainder_h: index of wo-axis
            # ==============================================
            core_do = self.core_ou_shape[0]
            core_do_times = self.do // core_do
            core_di = self.core_in_shape[0]

            core_ho = self.core_ou_shape[1]
            core_ho_times = self.ho // core_ho
            core_hi = self.core_in_shape[1]

            core_wo = self.core_ou_shape[2]
            core_wo_times = self.wo // core_wo
            core_wi = self.core_in_shape[2]

            merchant = (sum_core+num_core_loop) // \
                       (c1*core_do_times*core_ho_times*core_wo_times)
            remainder = (sum_core+num_core_loop) % \
                        (c1*core_do_times*core_ho_times*core_wo_times)

            merchant_c1 = remainder // (core_do_times*core_ho_times*core_wo_times)
            remainder_c1 = remainder % (core_do_times*core_ho_times*core_wo_times)

            merchant_d = remainder_c1 // (core_ho_times*core_wo_times)
            remainder_d = remainder_c1 % (core_ho_times*core_wo_times)

            merchant_h = remainder_d // core_wo_times
            remainder_h = remainder_d % core_wo_times

            def _main(loop_do_idx, loop_ho_idx, loop_wo_idx,
                      di, do, hi, ho, wi, wo):
                # ==========================
                # Init_Begin_Idx
                # ==========================
                if self.kd >= self.sd:
                    di_coordinate = loop_do_idx * (di_batch-self.overlap_d) + \
                                    merchant_d * (core_di-self.overlap_d)
                else:
                    di_coordinate = loop_do_idx * di_batch + \
                                    merchant_d * core_di

                if self.kh >= self.sh:
                    hi_coordinate = loop_ho_idx * (hi_batch-self.overlap_h) + \
                                    merchant_h * (core_hi-self.overlap_h)
                else:
                    hi_coordinate = loop_ho_idx * hi_batch + \
                                    merchant_h * core_hi

                if self.kw >= self.sw:
                    wi_coordinate = loop_wo_idx * (wi_batch-self.overlap_w) + \
                                    remainder_h * (core_wi-self.overlap_w)
                else:
                    wi_coordinate = loop_wo_idx * wi_batch + \
                                    remainder_h * core_wi

                do_coordinate = loop_do_idx * do_batch + merchant_d * core_do
                ho_coordinate = loop_ho_idx * ho_batch + merchant_h * core_ho
                wo_coordinate = loop_wo_idx * wo_batch + remainder_h * core_wo

                src_orig_x_gm = merchant * _prod(self.forward_in_shape[1:]) + \
                                merchant_c1 * self.h * self.w * c0 + \
                                di_coordinate * c1 * self.h * self.w * c0 + \
                                hi_coordinate * self.w * c0 + \
                                wi_coordinate * c0
                src_orig_y_gm = merchant * _prod(self.forward_ou_shape[1:]) + \
                                merchant_c1 * self.ho * self.wo * c0 + \
                                do_coordinate * c1 * self.ho * self.wo * c0 + \
                                ho_coordinate * self.wo * c0 + \
                                wo_coordinate * c0
                src_grad_gm = src_orig_y_gm

                # ================================
                # COPY_GM_2_L1_BUF
                # Prevent reading gm out of bounds
                # ================================
                di_val = min(0, self.overlap_d) + di
                hi_val = min(0, self.overlap_h) + hi
                wi_val = min(0, self.overlap_w) + wi
                input0 = [di_val, hi_val, wi_val]
                input1 = [di_batch, hi_batch, wi_batch]
                self._gm2l1_tiling_do_ho_wo(tik_instance,
                                            l1_in_buf, src_orig_x_gm, 0,
                                            input0, input1)

                # ================================
                # COPY_ORI_OUTPUT_2_FORWARD_OU_BUF
                # COPY_GRAD_2_GRAD_BUF
                # in the branch, do and ho are 1.
                # ================================
                self._copy_gm_to_ub(tik_instance, forward_ou_buf,
                                    self.orig_y_gm, src_orig_y_gm,
                                    [do, ho, wo, c0])

                self._copy_gm_to_ub(tik_instance, grad_buf, self.grads_gm,
                                    src_grad_gm, [do, ho, wo, c0])

                # ================================
                # load3d l1 to col_in_buffer
                # ================================
                repeat_times = _ceil_div(ho*wo, 16)
                with tik_instance.for_range(0, do, thread_num=1) as idx_do:
                    with tik_instance.for_range(0, self.kd, thread_num=1) as idx_d:
                        with tik_instance.for_range(0, self.kh, thread_num=1) as idx_h:
                            with tik_instance.for_range(0, self.kw, thread_num=1) as idx_w:
                                src_l1 = (idx_do * self.sd + idx_d) * hi_batch * wi_batch * c0
                                tik_instance.load3dv1(col_in_buf[0],
                                                      l1_in_buf[src_l1],
                                                      [0, 0, 0, 0],
                                                      hi_val, wi_val, 0,
                                                      idx_w, idx_h,
                                                      0, 0,
                                                      self.sw, self.sh,
                                                      self.kw, self.kh,
                                                      1, 1, 1, 1,
                                                      repeat_times, 0,
                                                      MIN_VALUE_FP16
                                                      )

                                # ---calculate mask---
                                idx_list = [idx_do, idx_d, idx_h, idx_w]
                                const_list = [ho, wo, c0]
                                self._calc_mask(tik_instance, buf_list, param,
                                                idx_list, const_list)

                                # ---sel(grad,zero,mask)---
                                self._sel(tik_instance, buf_list,
                                          idx_list, const_list)

                                # ---vconv grad_sel_fp16 to fp32---
                                self._vconv(tik_instance, grad_sel_fp16_buf, 0,
                                            grad_sel_fp32_buf, 0,
                                            param.grad_sel_fp16_size, "float16")

                                # ---rewrite grad_sel_fp32 to f_map_fp32
                                # do = 1, ho = 1
                                # map_index has two part: begin_index of kernel,
                                # begin_index of child kernel
                                # must use tik variable as index of grad_sel_fp32_buf,
                                # python variable is not work in grad_sel_fp32_buf[mask_index],
                                # while x = grad_sel_fp32_buf[mask_index], y = x[n].
                                with tik_instance.for_range(0, 1) as index_mask:
                                    map_index = idx_d * hi_batch * wi_batch * c0 + \
                                                idx_h * wi_batch * c0 + idx_w * c0
                                    mask_index = index_mask

                                    self._vector_op(tik_instance, "vadd",
                                                    f_map_fp32_buf[map_index],
                                                    grad_sel_fp32_buf[mask_index],
                                                    f_map_fp32_buf[map_index],
                                                    "float32", wo*c0//2,
                                                    stride_config=(self.sw*2,
                                                                   self.sw*2, 2,
                                                                   self.sw*16,
                                                                   self.sw*16, 16))
                                    self._vector_op(tik_instance, "vadd",
                                                    f_map_fp32_buf[map_index+8],
                                                    grad_sel_fp32_buf[mask_index+8],
                                                    f_map_fp32_buf[map_index + 8],
                                                    "float32", wo*c0//2,
                                                    stride_config=(self.sw*2,
                                                                   self.sw*2, 2,
                                                                   self.sw*16,
                                                                   self.sw*16, 16))

                # mov_out
                dst_ou_gm = merchant * _prod(self.forward_in_shape[1:]) + \
                            merchant_c1 * self.h * self.w * c0 + \
                            di_coordinate * c1 * self.h * self.w * c0 + \
                            hi_coordinate * self.w * c0 + \
                            wi_coordinate * c0

                def mov_atomic(num_d, dst, dst_idx, src_idx):
                    num_h = hi + min(0, self.overlap_h)
                    num_w = wi + min(0, self.overlap_w)
                    in_shape = [num_d, num_h, num_w, c0]

                    self._ub2gm_split_do_ho_wo(tik_instance, f_map_fp32_buf,
                                               src_idx, dst, dst_idx,
                                               in_shape, hi_batch, wi_batch)

                    self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                                        f_map_fp32_buf, 0, 0, "float32")

                tik_instance.set_atomic_add(1)
                mov_atomic(di + min(0, self.overlap_d),
                           self.ou_y_gm, dst_ou_gm, 0)
                tik_instance.set_atomic_add(0)

            if wo_tail != 0:
                with tik_instance.for_range(0, loop_do) as do_idx:
                    with tik_instance.for_range(0, loop_ho) as ho_idx:
                        with tik_instance.for_range(0, loop_wo) as wo_idx:
                            _main(do_idx, ho_idx, wo_idx,
                                  di_batch, do_batch,
                                  hi_batch, ho_batch,
                                  wi_batch, wo_batch)
                        _main(do_idx, ho_idx, loop_wo,
                              di_tail, do_batch,
                              hi_tail, ho_batch,
                              wi_tail, wo_tail)
            else:
                with tik_instance.for_range(0, loop_do) as do_idx:
                    with tik_instance.for_range(0, loop_ho) as ho_idx:
                        with tik_instance.for_range(0, loop_wo) as wo_idx:
                            _main(do_idx, ho_idx, wo_idx,
                                  di_batch, do_batch,
                                  hi_batch, ho_batch,
                                  wi_batch, wo_batch)

    def same_pure_atomic_tiling_do(self, tik_instance, core_loop,
                                   sum_core, model, param):
        '''
        ==============================================================
        In the case, [do,ho,wo] will be infer return
        [di_batch,hi_batch,wi_batch] and [map_di, map_hi, map_wi].
        xi_batch: size of input_data which restored in l1_in_buf.
        map_xi: size of feature_map which restored in f_map_fp32_buf.
        ==============================================================
        '''
        do_batch = model[0]
        ho_batch = model[1]
        wo_batch = model[2]
        c0 = self.c0
        c1 = self.c1

        # size of input_data
        loop_do = self.core_ou_shape[0] // do_batch
        di_batch, hi_batch, wi_batch = self._infer_dim_return(do_batch,
                                                              ho_batch,
                                                              wo_batch, True)
        do_tail = self.core_ou_shape[0] % do_batch
        di_tail, hi_tail, wi_tail = self._infer_dim_return(do_tail,
                                                           ho_batch,
                                                           wo_batch, True)
        # feature_map's size
        map_di, map_hi, map_wi = self._infer_map_return(do_batch, ho_batch,
                                                        wo_batch)

        pad_d_top, pad_d_bottom = self.pad[0][0], self.pad[0][1]
        pad_hw_top, pad_hw_bottom = self.pad[1][0], self.pad[1][1]
        pad_hw_left, pad_hw_right = self.pad[2][0], self.pad[2][1]
        pad_hw_list = [pad_hw_left, pad_hw_right, pad_hw_top, pad_hw_bottom]

        buf_list = self._set_buf_tensor(tik_instance, param)
        l1_in_buf = buf_list[0]
        forward_ou_buf = buf_list[1]
        grad_buf = buf_list[2]
        col_in_buf = buf_list[3]
        zero_buf = buf_list[7]
        grad_sel_fp16_buf = buf_list[8]
        grad_sel_fp32_buf = buf_list[9]
        f_map_fp32_buf = buf_list[10]
        self.set_vector_dup(tik_instance, param.zero_size,
                            zero_buf, 0, 0, self.dtype)
        self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                            f_map_fp32_buf, 0, 0, "float32")

        with tik_instance.for_range(0, core_loop) as num_core_loop:
            # ======================
            # Assume  ori_output_shape is [n,do,c1,ho,wo,c0],
            # split [n,do,c1] as core_num.
            # merchant: index of n-axis
            # merchant_c1: index of c1-axis
            # remainder_c1: index of do-axis
            # ======================
            core_do = self.core_ou_shape[0]
            core_do_times = self.do // core_do
            core_di = self.core_in_shape[0]

            merchant = (sum_core + num_core_loop) // (c1 * core_do_times)
            remainder = (sum_core + num_core_loop) % (c1 * core_do_times)
            merchant_c1 = remainder // core_do_times
            remainder_c1 = remainder % core_do_times

            def _main(loop_idx, di, do):
                # ============================================================
                # ----Init_Begin_Idx----
                # If pad_d_top exist, actual begin_idx of d_axis is -pad_d_top.
                # Meanwhile, don't move pad_d_x to l1_in_buf, but leave space
                # enough in l1_in_buf.
                # ============================================================
                if self.kd >= self.sd:
                    di_coordinate = loop_idx * (di_batch-self.overlap_d) + \
                                    remainder_c1 * (core_di-self.overlap_d) - \
                                    pad_d_top
                else:
                    di_coordinate = loop_idx * di_batch + \
                                    remainder_c1 * core_di - \
                                    pad_d_top

                do_coordinate = loop_idx * do_batch + remainder_c1 * core_do

                # if pad_d_top exist, the begin_index would be less than 0
                di_coord = init_coordinate(tik_instance, pad_d_top,
                                           di_coordinate)

                src_orig_x_gm = merchant * _prod(self.forward_in_shape[1:]) + \
                                merchant_c1 * self.h * self.w * c0 + \
                                di_coord * c1 * self.h * self.w * c0
                src_orig_y_gm = merchant * _prod(self.forward_ou_shape[1:]) + \
                                merchant_c1 * self.ho * self.wo * c0 + \
                                do_coordinate * c1 * self.ho * self.wo * c0
                src_grad_gm = src_orig_y_gm

                # ----COPY_GM_2_L1_BUF----
                # Prevent reading gm out of bounds.
                # Judge value of di_val according to do_coordinate.
                # di_val contains pad_d_top and pad_d_bottom.
                di_value = min(0, self.overlap_d) + di
                di_val = di_value
                l1_idx = 0
                # pad: used in gm2l1 and load3dv1
                d_top, d_bottom = calc_pad(tik_instance, pad_d_top, pad_d_bottom,
                                           di_coordinate, di_value, self.d)

                # gm2l1: filled regions don't move except d
                if pad_d_top != 0:
                    di_val -= d_top
                    l1_idx = d_top
                if pad_d_bottom != 0:
                    di_val -= d_bottom

                in_shape = [di_val, hi_batch, wi_batch, c0]
                self._copy_gm_to_l1(tik_instance, l1_in_buf,
                                    src_orig_x_gm, l1_idx*hi_batch*wi_batch*c0,
                                    in_shape)

                # ----COPY_ORI_OUTPUT_2_FORWARD_OU_BUF----
                self._copy_gm_to_ub(tik_instance, forward_ou_buf,
                                    self.orig_y_gm, src_orig_y_gm,
                                    [do, ho_batch, wo_batch, c0])

                # ----COPY_GRAD_2_GRAD_BUF----
                self._copy_gm_to_ub(tik_instance, grad_buf, self.grads_gm,
                                    src_grad_gm, [do, ho_batch, wo_batch, c0])

                # ---load3d l1 to col_in_buffer---
                load3d_mark = tik_instance.Scalar(dtype='int64', name='load3d_mark')
                repeat_times = _ceil_div(ho_batch*wo_batch, 16)
                with tik_instance.for_range(0, do, thread_num=1) as idx_do:
                    with tik_instance.for_range(0, self.kd, thread_num=1) as idx_d:
                        # =====================================================
                        # if window in position of pad, not load3d, but vec_dup.
                        # =====================================================
                        self.filled_vec_dup(tik_instance, load3d_mark, di_value,
                                            pad_d_top, pad_d_bottom,
                                            idx_do, idx_d, d_top, d_bottom,
                                            param, col_in_buf)

                        with tik_instance.for_range(0, self.kh, thread_num=1) as idx_h:
                            with tik_instance.for_range(0, self.kw, thread_num=1) as idx_w:
                                src_l1 = (idx_do * self.sd + idx_d) * hi_batch * wi_batch * c0

                                with tik_instance.if_scope(load3d_mark != 1):
                                    tik_instance.load3dv1(col_in_buf[0],
                                                          l1_in_buf[src_l1],
                                                          pad_hw_list,
                                                          hi_batch, wi_batch, 0,
                                                          idx_w, idx_h,
                                                          -pad_hw_left,
                                                          -pad_hw_top,
                                                          self.sw, self.sh,
                                                          self.kw, self.kh,
                                                          1, 1, 1, 1,
                                                          repeat_times, 0,
                                                          MIN_VALUE_FP16
                                                          )

                                # ---calculate mask---
                                idx_list = [idx_do, idx_d, idx_h, idx_w]
                                const_list = [ho_batch, wo_batch, c0]
                                self._calc_mask(tik_instance, buf_list, param,
                                                idx_list, const_list)

                                # ---sel(grad,zero,mask)---
                                self._sel(tik_instance, buf_list,
                                          idx_list, const_list)

                                # ---vconv grad_sel_fp16 to fp32---
                                self._vconv(tik_instance, grad_sel_fp16_buf, 0,
                                            grad_sel_fp32_buf, 0,
                                            param.grad_sel_fp16_size, "float16")

                                # ---rewrite grad_sel_fp32 to f_map_fp32
                                config = (self.sw*2, self.sw*2, 2,
                                          self.sh*map_wi*2, self.sh*map_wi*2, wo_batch*2)
                                if _check_config(config):
                                    with tik_instance.for_range(0, 1) as ho_idx:
                                        map_index = (idx_do*self.sd+idx_d)*map_hi*map_wi*c0 + \
                                                    (idx_h*map_wi*c0+idx_w*c0)
                                        mask_index = wo_batch * ho_idx * c0
                                        shape_map_hw = [map_hi, map_wi, c0]
                                        shape_grad = [ho_batch, wo_batch, c0]

                                        self._rewrite_fmap(tik_instance, "vadd",
                                                           f_map_fp32_buf[map_index],
                                                           grad_sel_fp32_buf[mask_index],
                                                           f_map_fp32_buf[map_index],
                                                           "float32", wo_batch*c0//2, ho_batch,
                                                           shape_map_hw, shape_grad,
                                                           config=config)

                                        self._rewrite_fmap(tik_instance, "vadd",
                                                           f_map_fp32_buf[map_index+8],
                                                           grad_sel_fp32_buf[mask_index+8],
                                                           f_map_fp32_buf[map_index+8],
                                                           "float32", wo_batch*c0//2, ho_batch,
                                                           shape_map_hw, shape_grad,
                                                           config=config)
                                else:
                                    # map_index has three part: which hwc0 in
                                    # which window, begin_index of kernel,
                                    # begin_index of child kernel
                                    with tik_instance.for_range(0, ho_batch) as ho_idx:
                                        map_index = (idx_do*self.sd+idx_d)*map_hi*map_wi*c0 + \
                                                    (ho_idx*self.sh*map_wi*c0) + \
                                                    (idx_h*map_wi*c0+idx_w*c0)
                                        mask_index = wo_batch * ho_idx * c0

                                        self._vector_op(tik_instance, "vadd",
                                                        f_map_fp32_buf[map_index],
                                                        grad_sel_fp32_buf[mask_index],
                                                        f_map_fp32_buf[map_index],
                                                        "float32", wo_batch*c0//2,
                                                        stride_config=(self.sw*2,
                                                                       self.sw*2, 2,
                                                                       self.sw*16,
                                                                       self.sw*16, 16))
                                        self._vector_op(tik_instance, "vadd",
                                                        f_map_fp32_buf[map_index+8],
                                                        grad_sel_fp32_buf[mask_index+8],
                                                        f_map_fp32_buf[map_index + 8],
                                                        "float32", wo_batch*c0//2,
                                                        stride_config=(self.sw*2,
                                                                       self.sw*2, 2,
                                                                       self.sw*16,
                                                                       self.sw*16, 16))

                # ---mov_out---
                dst_ou_gm = merchant * _prod(self.forward_in_shape[1:]) + \
                            merchant_c1 * self.h * self.w * c0 + \
                            di_coord * c1 * self.h * self.w * c0

                def mov_atomic(num_d, dst, dst_idx, src_idx):
                    ub2gm_shape = [num_d, hi_batch, wi_batch, c0]
                    src_idx += (pad_hw_top * map_wi + pad_hw_left) * c0

                    num_bit = self.num_bit_fp32
                    n_burst = ub2gm_shape[1]
                    burst_len = _prod(ub2gm_shape[2:]) * num_bit // MINI_UNIT
                    # c0 * num_bit // MINI_UNIT is 2
                    src_stride = (pad_hw_left + pad_hw_right) * 2
                    dst_stride = 0

                    with tik_instance.for_range(0, ub2gm_shape[0]) as idx:
                        src_idx_new = src_idx + idx * map_hi*map_wi*c0
                        dst_idx_new = dst_idx + _prod(self.forward_in_shape[3:]) * c1 * idx

                        in_list = [n_burst, burst_len, src_stride,
                                   dst_stride, src_idx_new, dst_idx_new]
                        if src_stride > MAX_STRIDE:
                            self._ultimate_data_move(tik_instance, f_map_fp32_buf,
                                                     dst, in_list, num_bit)
                        else:
                            self.norm_data_move(tik_instance, f_map_fp32_buf,
                                                dst, in_list)

                    self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                                        f_map_fp32_buf, 0, 0, "float32")

                tik_instance.set_atomic_add(1)
                mov_atomic(di_val, self.ou_y_gm, dst_ou_gm,
                           l1_idx*map_hi*map_wi*c0)
                tik_instance.set_atomic_add(0)

            with tik_instance.for_range(0, loop_do) as idx:
                _main(idx, di_batch, do_batch)

            if do_tail != 0:
                _main(loop_do, di_tail, do_tail)

    def same_pure_atomic_tiling_do_ho(self, tik_instance, core_loop,
                                      sum_core, model, param):
        '''
        ===================================================
        In the case, hi will be split/tiling.Due to load3d
        has the ability to fill h*w, l1_in_buf will save
        factual data(h*w).
        ===================================================
        '''
        do_batch = model[0]
        ho_batch = model[1]
        wo_batch = model[2]
        c0 = self.c0
        c1 = self.c1

        # size of input_data
        loop_do = self.core_ou_shape[0] // do_batch
        loop_ho = self.core_ou_shape[1] // ho_batch
        ho_tail = self.core_ou_shape[1] % ho_batch

        di_batch, hi_batch, wi_batch = self._infer_dim_return(do_batch,
                                                              ho_batch,
                                                              wo_batch, True)
        di_tail, hi_tail, wi_tail = self._infer_dim_return(do_batch,
                                                           ho_tail,
                                                           wo_batch, True)

        # size of feature map
        map_di, map_hi, map_wi = self._infer_map_return(do_batch, ho_batch,
                                                        wo_batch)

        pad_d_top, pad_d_bottom = self.pad[0][0], self.pad[0][1]
        pad_hw_top, pad_hw_bottom = self.pad[1][0], self.pad[1][1]
        pad_hw_left, pad_hw_right = self.pad[2][0], self.pad[2][1]
        pad_hw_list = [pad_hw_left, pad_hw_right, pad_hw_top, pad_hw_bottom]

        buf_list = self._set_buf_tensor(tik_instance, param)
        l1_in_buf = buf_list[0]
        forward_ou_buf = buf_list[1]
        grad_buf = buf_list[2]
        col_in_buf = buf_list[3]
        zero_buf = buf_list[7]
        grad_sel_fp16_buf = buf_list[8]
        grad_sel_fp32_buf = buf_list[9]
        f_map_fp32_buf = buf_list[10]
        self.set_vector_dup(tik_instance, param.zero_size,
                            zero_buf, 0, 0, self.dtype)
        self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                            f_map_fp32_buf, 0, 0, "float32")

        with tik_instance.for_range(0, core_loop) as num_core_loop:
            # ==============================================
            # Assume  ori_output_shape is [n,do,c1,ho,wo,c0],
            # split [n,do,c1,ho] as core_num.
            # merchant: index of n-axis
            # merchant_c1: index of c1-axis
            # merchant_d: index of do-axis
            # remainder_d: index of ho-axis
            # ==============================================
            core_do = self.core_ou_shape[0]
            core_do_times = self.do // core_do
            core_di = self.core_in_shape[0]

            core_ho = self.core_ou_shape[1]
            core_ho_times = self.ho // core_ho
            core_hi = self.core_in_shape[1]

            merchant = (sum_core+num_core_loop) // (c1*core_do_times*core_ho_times)
            remainder = (sum_core+num_core_loop) % (c1*core_do_times*core_ho_times)

            merchant_c1 = remainder // (core_do_times*core_ho_times)
            remainder_c1 = remainder % (core_do_times*core_ho_times)

            merchant_d = remainder_c1 // core_ho_times
            remainder_d = remainder_c1 % core_ho_times

            def _main(loop_do_idx, loop_ho_idx, di, do, hi, ho):
                # ==========================
                # Init_Begin_Idx
                # ==========================
                if self.kd >= self.sd:
                    di_coordinate = loop_do_idx * (di_batch-self.overlap_d) + \
                                    merchant_d * (core_di-self.overlap_d) - \
                                    pad_d_top
                else:
                    di_coordinate = loop_do_idx * di_batch + \
                                    merchant_d * core_di - \
                                    pad_d_top

                if self.kh >= self.sh:
                    hi_coordinate = loop_ho_idx * (hi_batch-self.overlap_h) + \
                                    remainder_d * (core_hi-self.overlap_h) - \
                                    pad_hw_top
                else:
                    hi_coordinate = loop_ho_idx * hi_batch + \
                                    remainder_d * core_hi - \
                                    pad_hw_top

                do_coordinate = loop_do_idx * do_batch + merchant_d * core_do
                ho_coordinate = loop_ho_idx * ho_batch + remainder_d * core_ho

                # init begin coordinate of di,hi.
                di_coord = init_coordinate(tik_instance, pad_d_top,
                                           di_coordinate)
                hi_coord = init_coordinate(tik_instance, pad_hw_top,
                                           hi_coordinate)

                src_orig_x_gm = merchant * _prod(self.forward_in_shape[1:]) + \
                                merchant_c1 * self.h * self.w * c0 + \
                                di_coord * c1 * self.h * self.w * c0 + \
                                hi_coord * self.w * c0
                src_orig_y_gm = merchant * _prod(self.forward_ou_shape[1:]) + \
                                merchant_c1 * self.ho * self.wo * c0 + \
                                do_coordinate * c1 * self.ho * self.wo * c0 + \
                                ho_coordinate * self.wo * c0
                src_grad_gm = src_orig_y_gm

                # ================================
                # COPY_GM_2_L1_BUF
                # Prevent reading gm out of bounds.
                # ================================
                # use immediate number
                di_value = min(0, self.overlap_d) + di
                hi_value = min(0, self.overlap_h) + hi
                di_val = di_value
                hi_val = hi_value
                l1_idx = 0

                # pad: used in gm2l1 and load3dv1
                d_top, d_bottom = calc_pad(tik_instance, pad_d_top, pad_d_bottom,
                                           di_coordinate, di_value, self.d)
                h_top, h_bottom = calc_pad(tik_instance, pad_hw_top, pad_hw_bottom,
                                           hi_coordinate, hi_value, self.h)
                pad_hw_list[-1] = h_bottom
                pad_hw_list[-2] = h_top

                # gm2l1: filled regions don't move except d
                if pad_d_top != 0:
                    di_val -= d_top
                    l1_idx = d_top
                if pad_d_bottom != 0:
                    di_val -= d_bottom

                if pad_hw_top != 0:
                    hi_val -= h_top
                if pad_hw_bottom != 0:
                    hi_val -= h_bottom

                in_shape = [di_val, hi_val, wi_batch, c0]
                self._gm2l1_tiling_do_ho(tik_instance, l1_in_buf,
                                         src_orig_x_gm,
                                         l1_idx*hi_batch*wi_batch*c0,
                                         in_shape,
                                         hi_batch)

                # ================================
                # COPY_ORI_OUTPUT_2_FORWARD_OU_BUF
                # COPY_GRAD_2_GRAD_BUF
                # ================================
                self._copy_gm_to_ub(tik_instance, forward_ou_buf,
                                    self.orig_y_gm, src_orig_y_gm,
                                    [do, ho, wo_batch, c0])

                self._copy_gm_to_ub(tik_instance, grad_buf, self.grads_gm,
                                    src_grad_gm, [do, ho, wo_batch, c0])

                # ================================
                # load3d l1 to col_in_buffer
                # ================================
                load3d_mark = tik_instance.Scalar(dtype='int64', name='load3d_mark')
                repeat_times = _ceil_div(ho*wo_batch, 16)
                with tik_instance.for_range(0, do, thread_num=1) as idx_do:
                    with tik_instance.for_range(0, self.kd, thread_num=1) as idx_d:
                        # =====================================================
                        # if window in position of pad, not load3d, but vec_dup.
                        # =====================================================
                        self.filled_vec_dup(tik_instance, load3d_mark, di_value,
                                            pad_d_top, pad_d_bottom,
                                            idx_do, idx_d, d_top, d_bottom,
                                            param, col_in_buf)

                        with tik_instance.for_range(0, self.kh, thread_num=1) as idx_h:
                            with tik_instance.for_range(0, self.kw, thread_num=1) as idx_w:
                                src_l1 = (idx_do * self.sd + idx_d) * hi_batch * wi_batch * c0

                                with tik_instance.if_scope(load3d_mark != 1):
                                    # in the case, l1_h must be hi_val to assure
                                    # correctness of result after filled.
                                    tik_instance.load3dv1(col_in_buf[0],
                                                          l1_in_buf[src_l1],
                                                          pad_hw_list,
                                                          hi_val, wi_batch, 0,
                                                          idx_w, idx_h,
                                                          -pad_hw_left,
                                                          -h_top,
                                                          self.sw, self.sh,
                                                          self.kw, self.kh,
                                                          1, 1, 1, 1,
                                                          repeat_times, 0,
                                                          MIN_VALUE_FP16
                                                          )

                                # ---calculate mask---
                                idx_list = [idx_do, idx_d, idx_h, idx_w]
                                const_list = [ho, wo_batch, c0]
                                self._calc_mask(tik_instance, buf_list, param,
                                                idx_list, const_list)

                                # ---sel(grad,zero,mask)---
                                self._sel(tik_instance, buf_list,
                                          idx_list, const_list)

                                # ---vconv grad_sel_fp16 to fp32---
                                self._vconv(tik_instance, grad_sel_fp16_buf, 0,
                                            grad_sel_fp32_buf, 0,
                                            param.grad_sel_fp16_size, "float16")

                                # ---rewrite grad_sel_fp32 to f_map_fp32
                                config = (self.sw*2, self.sw*2, 2,
                                          self.sh*map_wi*2, self.sh*map_wi*2, wo_batch*2)
                                if _check_config(config):
                                    with tik_instance.for_range(0, 1) as ho_idx:
                                        map_index = (idx_do*self.sd+idx_d)*map_hi*map_wi*c0 + \
                                                    (idx_h*map_wi*c0+idx_w*c0)
                                        mask_index = wo_batch * ho_idx * c0
                                        shape_map_hw = [map_hi, map_wi, c0]
                                        shape_grad = [ho, wo_batch, c0]

                                        self._rewrite_fmap(tik_instance, "vadd",
                                                           f_map_fp32_buf[map_index],
                                                           grad_sel_fp32_buf[mask_index],
                                                           f_map_fp32_buf[map_index],
                                                           "float32", wo_batch*c0//2, ho,
                                                           shape_map_hw, shape_grad,
                                                           config=config)

                                        self._rewrite_fmap(tik_instance, "vadd",
                                                           f_map_fp32_buf[map_index+8],
                                                           grad_sel_fp32_buf[mask_index+8],
                                                           f_map_fp32_buf[map_index+8],
                                                           "float32", wo_batch*c0//2, ho,
                                                           shape_map_hw, shape_grad,
                                                           config=config)
                                else:
                                    # map_index has three part: which hwc0 in
                                    # which window, begin_index of kernel,
                                    # begin_index of child kernel
                                    with tik_instance.for_range(0, ho) as ho_idx:
                                        map_index = (idx_do*self.sd+idx_d)*map_hi*map_wi*c0 + \
                                                    (ho_idx*self.sh*map_wi*c0) + \
                                                    (idx_h*map_wi*c0+idx_w*c0)
                                        mask_index = wo_batch * ho_idx * c0

                                        self._vector_op(tik_instance, "vadd",
                                                        f_map_fp32_buf[map_index],
                                                        grad_sel_fp32_buf[mask_index],
                                                        f_map_fp32_buf[map_index],
                                                        "float32", wo_batch*c0//2,
                                                        stride_config=(self.sw*2,
                                                                       self.sw*2, 2,
                                                                       self.sw*16,
                                                                       self.sw*16, 16))
                                        self._vector_op(tik_instance, "vadd",
                                                        f_map_fp32_buf[map_index+8],
                                                        grad_sel_fp32_buf[mask_index+8],
                                                        f_map_fp32_buf[map_index + 8],
                                                        "float32", wo_batch*c0//2,
                                                        stride_config=(self.sw*2,
                                                                       self.sw*2, 2,
                                                                       self.sw*16,
                                                                       self.sw*16, 16))

                # mov_out
                dst_ou_gm = merchant * _prod(self.forward_in_shape[1:]) + \
                            merchant_c1 * self.h * self.w * c0 + \
                            di_coord * c1 * self.h * self.w * c0 + \
                            hi_coord * self.w * c0

                def mov_atomic(num_d, num_h, dst, dst_idx, src_idx):
                    ub2gm_shape = [num_d, num_h, wi_batch, c0]
                    src_idx += (h_top * map_wi + pad_hw_left) * c0

                    num_bit = self.num_bit_fp32
                    n_burst = ub2gm_shape[1]
                    burst_len = _prod(ub2gm_shape[2:]) * num_bit // MINI_UNIT
                    # c0 * num_bit // MINI_UNIT is 2
                    src_stride = (pad_hw_left + pad_hw_right) * 2
                    dst_stride = 0

                    with tik_instance.for_range(0, ub2gm_shape[0]) as idx:
                        src_idx_new = src_idx + idx * map_hi*map_wi*c0
                        dst_idx_new = dst_idx + _prod(self.forward_in_shape[3:]) * c1 * idx

                        in_list = [n_burst, burst_len, src_stride,
                                   dst_stride, src_idx_new, dst_idx_new]
                        if src_stride > MAX_STRIDE:
                            self._ultimate_data_move(tik_instance, f_map_fp32_buf,
                                                     dst, in_list, num_bit)
                        else:
                            self.norm_data_move(tik_instance, f_map_fp32_buf,
                                                dst, in_list)

                    # vec_dup
                    self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                                        f_map_fp32_buf, 0, 0, "float32")

                tik_instance.set_atomic_add(1)
                mov_atomic(di_val, hi_val, self.ou_y_gm, dst_ou_gm,
                           l1_idx*map_hi*map_wi*c0)
                tik_instance.set_atomic_add(0)

            if ho_tail != 0:
                with tik_instance.for_range(0, loop_do) as do_idx:
                    with tik_instance.for_range(0, loop_ho) as ho_idx:
                        _main(do_idx, ho_idx, di_batch, do_batch,
                              hi_batch, ho_batch)
                    _main(do_idx, loop_ho, di_tail, do_batch, hi_tail, ho_tail)
            else:
                with tik_instance.for_range(0, loop_do) as do_idx:
                    with tik_instance.for_range(0, loop_ho) as ho_idx:
                        _main(do_idx, ho_idx, di_batch, do_batch,
                              hi_batch, ho_batch)

    def same_pure_atomic_tiling_do_ho_wo(self, tik_instance, core_loop,
                                         sum_core, model, param):
        '''
        ===================================================
        In the case, do,ho,wo will be split/tiling.So,need
        to assure pad_value of different axis.
        ===================================================
        '''
        do_batch = model[0]
        ho_batch = model[1]
        wo_batch = model[2]
        c0 = self.c0
        c1 = self.c1

        if do_batch != ho_batch != 1:
            raise RuntimeError("In the branch of 'tiling_do_ho', do_batch and "
                               "ho_batch should be 1.")

        # size of input_data
        loop_do = self.core_ou_shape[0] // do_batch
        loop_ho = self.core_ou_shape[1] // ho_batch
        loop_wo = self.core_ou_shape[2] // wo_batch
        wo_tail = self.core_ou_shape[2] % wo_batch
        di_batch, hi_batch, wi_batch = self._infer_dim_return(do_batch,
                                                              ho_batch,
                                                              wo_batch, True)
        di_tail, hi_tail, wi_tail = self._infer_dim_return(do_batch,
                                                           ho_batch,
                                                           wo_tail, True)

        # size of feature_map
        map_di, map_hi, map_wi = self._infer_map_return(do_batch, ho_batch,
                                                        wo_batch)
        pad_d_top, pad_d_bottom = self.pad[0][0], self.pad[0][1]
        pad_hw_top, pad_hw_bottom = self.pad[1][0], self.pad[1][1]
        pad_hw_left, pad_hw_right = self.pad[2][0], self.pad[2][1]
        pad_hw_list = [pad_hw_left, pad_hw_right, pad_hw_top, pad_hw_bottom]

        buf_list = self._set_buf_tensor(tik_instance, param)
        l1_in_buf = buf_list[0]
        forward_ou_buf = buf_list[1]
        grad_buf = buf_list[2]
        col_in_buf = buf_list[3]
        zero_buf = buf_list[7]
        grad_sel_fp16_buf = buf_list[8]
        grad_sel_fp32_buf = buf_list[9]
        f_map_fp32_buf = buf_list[10]
        self.set_vector_dup(tik_instance, param.zero_size,
                            zero_buf, 0, 0, self.dtype)
        self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                            f_map_fp32_buf, 0, 0, "float32")

        with tik_instance.for_range(0, core_loop) as num_core_loop:
            # ==============================================
            # Assume  ori_output_shape is [n,do,c1,ho,wo,c0],
            # split [n,do,c1,ho] as core_num.
            # merchant: index of n-axis
            # merchant_c1: index of c1-axis
            # merchant_d: index of do-axis
            # merchant_h: index of ho-axis
            # remainder_h: index of wo-axis
            # ==============================================
            core_do = self.core_ou_shape[0]
            core_do_times = self.do // core_do
            core_di = self.core_in_shape[0]

            core_ho = self.core_ou_shape[1]
            core_ho_times = self.ho // core_ho
            core_hi = self.core_in_shape[1]

            core_wo = self.core_ou_shape[2]
            core_wo_times = self.wo // core_wo
            core_wi = self.core_in_shape[2]

            merchant = (sum_core+num_core_loop) // \
                       (c1*core_do_times*core_ho_times*core_wo_times)
            remainder = (sum_core+num_core_loop) % \
                        (c1*core_do_times*core_ho_times*core_wo_times)

            merchant_c1 = remainder // (core_do_times*core_ho_times*core_wo_times)
            remainder_c1 = remainder % (core_do_times*core_ho_times*core_wo_times)

            merchant_d = remainder_c1 // (core_ho_times*core_wo_times)
            remainder_d = remainder_c1 % (core_ho_times*core_wo_times)

            merchant_h = remainder_d // core_wo_times
            remainder_h = remainder_d % core_wo_times

            def _main(loop_do_idx, loop_ho_idx, loop_wo_idx,
                      di, do, hi, ho, wi, wo):
                # ==========================
                # Init_Begin_Idx
                # ==========================
                if self.kd >= self.sd:
                    di_coordinate = loop_do_idx * (di_batch-self.overlap_d) + \
                                    merchant_d * (core_di-self.overlap_d) - \
                                    pad_d_top
                else:
                    di_coordinate = loop_do_idx * di_batch + \
                                    merchant_d * core_di - \
                                    pad_d_top

                if self.kh >= self.sh:
                    hi_coordinate = loop_ho_idx * (hi_batch-self.overlap_h) + \
                                    merchant_h * (core_hi-self.overlap_h) - \
                                    pad_hw_top
                else:
                    hi_coordinate = loop_ho_idx * hi_batch + \
                                    merchant_h * core_hi - \
                                    pad_hw_top

                if self.kw >= self.sw:
                    wi_coordinate = loop_wo_idx * (wi_batch-self.overlap_w) + \
                                    remainder_h * (core_wi-self.overlap_w) - \
                                    pad_hw_left
                else:
                    wi_coordinate = loop_wo_idx * wi_batch + \
                                    remainder_h * core_wi - \
                                    pad_hw_left

                do_coordinate = loop_do_idx * do_batch + merchant_d * core_do
                ho_coordinate = loop_ho_idx * ho_batch + merchant_h * core_ho
                wo_coordinate = loop_wo_idx * wo_batch + remainder_h * core_wo

                # init begin coordinate of di,hi,wi
                di_coord = init_coordinate(tik_instance, pad_d_top,
                                           di_coordinate)
                hi_coord = init_coordinate(tik_instance, pad_hw_top,
                                           hi_coordinate)
                wi_coord = init_coordinate(tik_instance, pad_hw_left,
                                           wi_coordinate)

                src_orig_x_gm = merchant * _prod(self.forward_in_shape[1:]) + \
                                merchant_c1 * self.h * self.w * c0 + \
                                di_coord * c1 * self.h * self.w * c0 + \
                                hi_coord * self.w * c0 + \
                                wi_coord * c0
                src_orig_y_gm = merchant * _prod(self.forward_ou_shape[1:]) + \
                                merchant_c1 * self.ho * self.wo * c0 + \
                                do_coordinate * c1 * self.ho * self.wo * c0 + \
                                ho_coordinate * self.wo * c0 + \
                                wo_coordinate * c0
                src_grad_gm = src_orig_y_gm

                # ================================
                # COPY_GM_2_L1_BUF
                # Prevent reading gm out of bounds
                # ================================
                di_value = min(0, self.overlap_d) + di
                hi_value = min(0, self.overlap_h) + hi
                wi_value = min(0, self.overlap_w) + wi
                di_val = di_value
                hi_val = hi_value
                wi_val = wi_value

                # pad: used in gm2l1 and load3dv1
                d_top, d_bottom = calc_pad(tik_instance, pad_d_top, pad_d_bottom,
                                           di_coordinate, di_value, self.d)
                h_top, h_bottom = calc_pad(tik_instance, pad_hw_top, pad_hw_bottom,
                                           hi_coordinate, hi_value, self.h)
                w_top, w_bottom = calc_pad(tik_instance, pad_hw_left, pad_hw_right,
                                           wi_coordinate, wi_value, self.w)
                pad_hw_list[-1], pad_hw_list[-2] = h_bottom, h_top
                pad_hw_list[-3], pad_hw_list[-4] = w_bottom, w_top

                # gm2l1: filled regions don't move except d
                di_val = di_val - d_top - d_bottom
                hi_val = hi_val - h_top - h_bottom
                wi_val = wi_val - w_top - w_bottom
                l1_idx = d_top

                input0 = [di_val, hi_val, wi_val]
                input1 = [di_batch, hi_batch, wi_batch]
                self._gm2l1_tiling_do_ho_wo(tik_instance,
                                            l1_in_buf, src_orig_x_gm,
                                            l1_idx*hi_batch*wi_batch*c0,
                                            input0, input1)

                # ================================
                # COPY_ORI_OUTPUT_2_FORWARD_OU_BUF
                # COPY_GRAD_2_GRAD_BUF
                # in the branch, do and ho are 1.
                # ================================
                self._copy_gm_to_ub(tik_instance, forward_ou_buf,
                                    self.orig_y_gm, src_orig_y_gm,
                                    [do, ho, wo, c0])

                self._copy_gm_to_ub(tik_instance, grad_buf, self.grads_gm,
                                    src_grad_gm, [do, ho, wo, c0])

                # ================================
                # load3d l1 to col_in_buffer
                # ================================
                load3d_mark = tik_instance.Scalar(dtype='int64', name='load3d_mark')
                repeat_times = _ceil_div(ho*wo, 16)
                with tik_instance.for_range(0, do, thread_num=1) as idx_do:
                    with tik_instance.for_range(0, self.kd, thread_num=1) as idx_d:
                        # =====================================================
                        # if window in position of pad, not load3d, but vec_dup.
                        # =====================================================
                        self.filled_vec_dup(tik_instance, load3d_mark, di_value,
                                            pad_d_top, pad_d_bottom,
                                            idx_do, idx_d, d_top, d_bottom,
                                            param, col_in_buf)

                        with tik_instance.for_range(0, self.kh, thread_num=1) as idx_h:
                            with tik_instance.for_range(0, self.kw, thread_num=1) as idx_w:
                                src_l1 = (idx_do * self.sd + idx_d) * hi_batch * wi_batch * c0

                                with tik_instance.if_scope(load3d_mark != 1):
                                    tik_instance.load3dv1(col_in_buf[0],
                                                          l1_in_buf[src_l1],
                                                          pad_hw_list,
                                                          hi_val, wi_val, 0,
                                                          idx_w, idx_h,
                                                          -w_top, -h_top,
                                                          self.sw, self.sh,
                                                          self.kw, self.kh,
                                                          1, 1, 1, 1,
                                                          repeat_times, 0,
                                                          MIN_VALUE_FP16
                                                          )

                                # ---calculate mask---
                                idx_list = [idx_do, idx_d, idx_h, idx_w]
                                const_list = [ho, wo, c0]
                                self._calc_mask(tik_instance, buf_list, param,
                                                idx_list, const_list)

                                # ---sel(grad,zero,mask)---
                                self._sel(tik_instance, buf_list,
                                          idx_list, const_list)

                                # ---vconv grad_sel_fp16 to fp32---
                                self._vconv(tik_instance, grad_sel_fp16_buf, 0,
                                            grad_sel_fp32_buf, 0,
                                            param.grad_sel_fp16_size, "float16")

                                # ---rewrite grad_sel_fp32 to f_map_fp32
                                # do = 1, ho = 1
                                # map_index has two part: begin_index of kernel,
                                # begin_index of child kernel
                                # must use tik variable as index of grad_sel_fp32_buf,
                                # python variable is not work in grad_sel_fp32_buf[mask_index],
                                # while x = grad_sel_fp32_buf[mask_index], y = x[n].
                                with tik_instance.for_range(0, 1) as index_mask:
                                    map_index = idx_d * map_hi * map_wi * c0 + \
                                                idx_h * map_wi * c0 + idx_w * c0
                                    mask_index = index_mask

                                    self._vector_op(tik_instance, "vadd",
                                                    f_map_fp32_buf[map_index],
                                                    grad_sel_fp32_buf[mask_index],
                                                    f_map_fp32_buf[map_index],
                                                    "float32", wo*c0//2,
                                                    stride_config=(self.sw*2,
                                                                   self.sw*2, 2,
                                                                   self.sw*16,
                                                                   self.sw*16, 16))
                                    self._vector_op(tik_instance, "vadd",
                                                    f_map_fp32_buf[map_index+8],
                                                    grad_sel_fp32_buf[mask_index+8],
                                                    f_map_fp32_buf[map_index + 8],
                                                    "float32", wo*c0//2,
                                                    stride_config=(self.sw*2,
                                                                   self.sw*2, 2,
                                                                   self.sw*16,
                                                                   self.sw*16, 16))

                # mov_out
                dst_ou_gm = merchant * _prod(self.forward_in_shape[1:]) + \
                            merchant_c1 * self.h * self.w * c0 + \
                            di_coord * c1 * self.h * self.w * c0 + \
                            hi_coord * self.w * c0 + \
                            wi_coord * c0

                def mov_atomic(num_d, num_h, num_w, dst, dst_idx, src_idx):
                    ub2gm_shape = [num_d, num_h, num_w, c0]
                    src_idx += (h_top * map_wi + w_top) * c0

                    num_bit = self.num_bit_fp32
                    n_burst = ub2gm_shape[1]
                    burst_len = _prod(ub2gm_shape[2:]) * num_bit // MINI_UNIT
                    # c0 * num_bit // MINI_UNIT is 2
                    src_stride = (map_wi - num_w) * 2
                    dst_stride = (self.w - num_w) * 2

                    with tik_instance.for_range(0, ub2gm_shape[0]) as idx:
                        src_idx_new = src_idx + idx * map_hi*map_wi*c0
                        dst_idx_new = dst_idx + _prod(self.forward_in_shape[3:]) * c1 * idx

                        in_list = [n_burst, burst_len, src_stride,
                                   dst_stride, src_idx_new, dst_idx_new]
                        check = isinstance(src_stride, int)

                        with tik_instance.if_scope(
                                tik.any(src_stride > MAX_STRIDE,
                                        dst_stride > MAX_STRIDE)):
                            self._ultimate_data_move(tik_instance, f_map_fp32_buf,
                                                     dst, in_list, num_bit)

                        with tik_instance.else_scope():
                            if check:
                                if src_stride <= MAX_STRIDE and dst_stride <= MAX_STRIDE:
                                    self.norm_data_move(tik_instance, f_map_fp32_buf,
                                                        dst, in_list)
                            else:
                                self.norm_data_move(tik_instance, f_map_fp32_buf,
                                                    dst, in_list)

                    self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                                        f_map_fp32_buf, 0, 0, "float32")

                tik_instance.set_atomic_add(1)
                mov_atomic(di_val, hi_val, wi_val,
                           self.ou_y_gm, dst_ou_gm,
                           l1_idx*map_hi*map_wi*c0)
                tik_instance.set_atomic_add(0)

            if wo_tail != 0:
                with tik_instance.for_range(0, loop_do) as do_idx:
                    with tik_instance.for_range(0, loop_ho) as ho_idx:
                        with tik_instance.for_range(0, loop_wo) as wo_idx:
                            _main(do_idx, ho_idx, wo_idx,
                                  di_batch, do_batch,
                                  hi_batch, ho_batch,
                                  wi_batch, wo_batch)
                        _main(do_idx, ho_idx, loop_wo,
                              di_tail, do_batch,
                              hi_tail, ho_batch,
                              wi_tail, wo_tail)
            else:
                with tik_instance.for_range(0, loop_do) as do_idx:
                    with tik_instance.for_range(0, loop_ho) as ho_idx:
                        with tik_instance.for_range(0, loop_wo) as wo_idx:
                            _main(do_idx, ho_idx, wo_idx,
                                  di_batch, do_batch,
                                  hi_batch, ho_batch,
                                  wi_batch, wo_batch)

    def filled_vec_dup(self, tik_instance, mark, di_value, pad_d_top,
                       pad_d_bottom, idx_do, idx_d, d_top, d_bottom,
                       param, dst_buf):
        # make filled region in l1_buf, not move to
        # col_in_buf by load3d, but vec_dup in col_in_buf.
        mark.set_as(0)
        win_idx = idx_do * self.sd + idx_d
        if pad_d_top != 0:
            with tik_instance.if_scope(win_idx < d_top):
                self.set_vector_dup(tik_instance, param.col_in_size,
                                    dst_buf, 0, MIN_VALUE_FP16, "float16")
                mark.set_as(1)

        if pad_d_bottom != 0:
            with tik_instance.if_scope(win_idx > di_value-d_bottom-1):
                self.set_vector_dup(tik_instance, param.col_in_size,
                                    dst_buf, 0, MIN_VALUE_FP16, "float16")
                mark.set_as(1)

    def _division_nearest(self, number, base_num):
        # split number as n0 and n1,
        # return n1, base_num*n0 as new_number and core_num
        n1 = number
        new_base_num = base_num
        for n0 in range(1, number + 1):
            if number % n0 == 0:
                new_base_num = base_num * n0
                n1 = int(number / n0)
                if new_base_num >= MAX_CORE:
                    break
        return n1, new_base_num

    def _split_core(self):
        # ============================
        # in: [N,D,C1,H,W,C0]
        # ou: [N,Do,C1,Ho,Wo,C0]
        # SPLIT Do,Ho,Wo for core_num
        # core_branch:
        # 0: "not_split"
        # 1: "split_do"
        # 2: "split_do_ho"
        # 3: "split_do_ho_wo"
        # =============================
        n, do, c1, ho, wo, c0 = self.n, self.do, self.c1, self.ho, self.wo, \
                                self.c0
        core_ou_shape = [do, ho, wo, c0]
        base_num = n * c1

        if base_num >= MAX_CORE:
            total_num = base_num
            core_num = MAX_CORE
            core_branch = 0

        elif base_num * do >= MAX_CORE:
            new_do, total_num = self._division_nearest(do, base_num)
            core_num = MAX_CORE
            core_ou_shape[0] = new_do
            core_branch = 1

        elif base_num * do * ho >= MAX_CORE:
            base_num, new_do = base_num * do, 1
            new_ho, total_num = self._division_nearest(ho, base_num)
            core_num = MAX_CORE
            core_ou_shape[0] = new_do
            core_ou_shape[1] = new_ho
            core_branch = 2

        else:
            # base_num * do * ho * wo
            base_num, new_do, new_ho = base_num * do * ho, 1, 1
            new_wo, total_num = self._division_nearest(wo, base_num)
            core_ou_shape[0] = new_do
            core_ou_shape[1] = new_ho
            core_ou_shape[2] = new_wo
            core_num = total_num
            if total_num >= MAX_CORE:
                core_num = MAX_CORE
            core_branch = 3

        do, ho, wo = core_ou_shape[0], core_ou_shape[1], core_ou_shape[2]
        di, hi, wi = self._infer_dim_return(do, ho, wo, True)
        core_in_shape = [di, hi, wi, c0]

        return total_num, core_num, core_ou_shape, core_in_shape, core_branch

    def grad(self, tik_instance, split_model,
             param, total_num, core_num, func):
        # just tiling do ho
        # support valid
        core_loop = tik_instance.Scalar("int64")
        sum_core = tik_instance.Scalar("int64")
        with tik_instance.for_range(0, core_num,
                                    block_num=core_num) as blk_idx:

            core_loop_uint64, sum_core_uint64 = _cal_core(tik_instance, total_num,
                                                          blk_idx, core_num)
            core_loop.set_as(core_loop_uint64)
            sum_core.set_as(sum_core_uint64)

            func(tik_instance, core_loop, sum_core, split_model, param)

    def _compute(self):
        """
        the overall data move process
        """
        tik_instance = self.set_tik_instance()
        total_num, core_num, core_ou_shape, \
        core_in_shape, core_branch = self._split_core()
        self.core_ou_shape = core_ou_shape
        self.core_in_shape = core_in_shape
        branch, split_model, param = self._pattern(core_ou_shape, core_branch)

        if self.pads.upper() == 'VALID':
            if core_branch == 0:
                # =====================
                # case0: n*c1 as core
                # =====================
                if branch == "not_tiling":
                    self.grad(tik_instance, split_model, param,
                              total_num, core_num,
                              self.not_tiling_main)

                elif branch == "tiling_do":
                    self.grad(tik_instance, split_model, param,
                              total_num, core_num,
                              self.tiling_do_main)

                elif branch == "tiling_do_ho":
                    self.grad(tik_instance, split_model, param,
                              total_num, core_num,
                              self.tiling_do_ho_main)

                else:
                    # "tiling_do_ho_wo"
                    self.grad(tik_instance, split_model, param,
                              total_num, core_num,
                              self.tiling_do_ho_wo_main)
            else:
                # =====================
                # case1: split do,ho,wo as core
                # use pure atomic
                # =====================
                if branch == "tiling_do":
                    self.grad(tik_instance, split_model, param,
                              total_num, core_num,
                              self.pure_atomic_tiling_do)

                elif branch == "tiling_do_ho":
                    self.grad(tik_instance, split_model, param,
                              total_num, core_num,
                              self.pure_atomic_tiling_do_ho)
                else:
                    # "tiling_do_ho_wo"
                    self.grad(tik_instance, split_model, param,
                              total_num, core_num,
                              self.pure_atomic_tiling_do_ho_wo)

        else:
            if branch in ["tiling_do", "not_tiling"]:
                self.grad(tik_instance, split_model, param,
                          total_num, core_num,
                          self.same_pure_atomic_tiling_do)

            elif branch == "tiling_do_ho":
                self.grad(tik_instance, split_model, param,
                          total_num, core_num,
                          self.same_pure_atomic_tiling_do_ho)
            else:
                # "tiling_do_ho_wo"
                self.grad(tik_instance, split_model, param,
                          total_num, core_num,
                          self.same_pure_atomic_tiling_do_ho_wo)

        return tik_instance

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        tik_instance = self._compute()
        tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=[self.orig_x_gm,
                                      self.orig_y_gm,
                                      self.grads_gm],
                              outputs=[self.ou_y_gm])

        return tik_instance


# pylint: disable = too-many-arguments
def check_param(ori_input, ori_output, grad, ksize, strides,
                kernel_name):
    """
    check parameters, if one is invalid, then raise error

    Parameters
    ----------
    ori_input: dict
        shape and data type of ori_input
    ori_output: dict
        shape and data type of ori_output
    grad: dict
        shape and data type of grad
    ksize: list or tuple
        the size of the window
    strides: list or tuple
        the stride of the sliding window
    kernel_name: str

    Returns
    -------
    None
    """
    ori_input_shape = ori_input.get("shape")
    ori_input_dtype = ori_input.get("dtype").lower()
    ori_output_shape = ori_output.get("shape")
    grad_shape = grad.get("shape")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(ori_input_shape)
    util.check_tensor_shape_size(ori_input_shape)
    util.check_dtype_rule(ori_input_dtype, ("float16",))
    # the format of input_x must be NDC1HWC0
    if len(ori_input_shape) != 6:
        raise RuntimeError("invalid shape params, input feature map must be "
                           "6D format in kernel.")
    if len(ori_output_shape) != 6:
        raise RuntimeError("invalid shape params, forward output must be "
                           "6D format in kernel.")
    if len(grad_shape) != 6:
        raise RuntimeError("invalid shape params, update grad must be "
                           "6D format in kernel.")

    if grad_shape != ori_output_shape:
        raise RuntimeError(
            "invalid shape params, update grad must be same shape as forward output")

    if len(ksize) != 5 or len(strides) != 5:
        raise RuntimeError("Invalid ksize or strides params,"
                           " ksize dim must be 5.")

    if ksize[0] != 1 or ksize[4] != 1:
        raise RuntimeError("MaxPoolGRAD only supports pooling "
                           "across width/height, and other ksize "
                           "dimension should be one")

    if strides[0] != 1 or strides[4] != 1:
        raise RuntimeError("MaxPoolGRAD only supports pooling across "
                           "width/height, and other strides dimension "
                           "should be one")


# pylint: disable=invalid-name,unused-argument
@util.check_input_type(dict, dict, dict, dict,
                       (tuple, list), (tuple, list), (tuple, list), str, str)
def max_pool3d_grad(orig_x, orig_y, grads, y,
                    ksize, strides, pads=(0, 0, 0, 0, 0, 0),
                    data_format="NDHWC",
                    kernel_name="max_pool3d_grad"):
    """
    main function of max_pool3d_grad

    Parameters
    ----------
    orig_x: dict
        shape and data type of max_pool3d's forward_input
    orig_y: dict
        shape and data type of max_pool3d's forward_output
    grads: dict
        shape and data type of grads
    y: dict
        shape and data type of y
    ksize: list or tuple
        the size of the window
    strides: list or tuple
        the stride of the sliding window
    pads: list or tuple
        the fill value of input
    data_format: str
        value from `NDHWC`, `NCDHW`
    kernel_name: str

    Returns
    -------
    return the tik api function
    """
    forward_in_shape = list(orig_x.get("shape"))
    forward_ou_shape = list(orig_y.get("shape"))
    grad_shape = list(grads.get("shape"))
    ou_shape = list(y.get("shape"))
    dtype = orig_x.get("dtype")
    ksize = list(ksize)
    strides = list(strides)
    check_param(orig_x, orig_y, grads, ksize, strides, kernel_name)
    shape_list = [forward_in_shape, forward_ou_shape, grad_shape, ou_shape]

    if data_format == "NCDHW":
        ksize = [ksize[0], ksize[2], ksize[3], ksize[4], ksize[1]]
        strides = [strides[0], strides[2], strides[3], strides[4], strides[1]]
    if data_format not in ["NCDHW", "NDHWC"]:
        raise RuntimeError("data_format should be NDHWC or NCDHW")

    params = [ksize, strides, list(pads), dtype, kernel_name]
    result = MaxPool3DGradCompute(shape_list, params)
    return result.get_tik_instance()
