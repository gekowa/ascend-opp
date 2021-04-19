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
concat_last_dim
"""
from functools import reduce as functools_reduce
from te import tik
from te import platform as tbe_platform

# vnchwconv can deal 16*16
TRANSPOSE_SIZE = 256
# one block can save the size of fp16
ONE_BLOCK_FP16_SIZE = 16


def get_ceil_int(int1, int2):
    """get cel for input1 and input2
    """
    if int1 == 0:
        return 1
    _result = int1 // int2
    if int1 % int2 == 0:
        return _result

    return _result + 1


# pylint: disable=too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-locals,too-many-statements,too-many-branches
class ConcatWithVnchw:
    """Function: use to finish ConcatLastDim main functions
    """
    def __init__(self, input_data, output_data, kernel_name="concat_last_dim"):
        """init concat base parameters
        """
        self.input_shapes = []
        self.data_dtype = input_data[0].get("dtype").lower()
        self.gm_in = []
        self.last_dim = input_data[0].get("shape")[-1]
        self.input_num = len(input_data)
        for index, input_dict in enumerate(input_data):
            shape_input = input_dict.get("shape")
            self.input_shapes.append(shape_input)

        self.output_shape = output_data.get("shape")

        self.kernel_name = kernel_name

        if self.data_dtype == "float32":
            self.last_dim = self.last_dim * 2

        self.tik_instance = tik.Tik()
        self.core_num = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)

        self.src_size = int(self.get_tensor_size_in_fp16(self.input_shapes[0]))
        self.dst_size = int(self.get_tensor_size_in_fp16(self.output_shape))

        self.use_last_dim = False
        self.use_last_dim_odd = False

        for index, _ in enumerate(self.input_shapes):
            self.gm_in.append(
                self.tik_instance.Tensor(
                    "float16", (self.src_size,),
                    scope=tik.scope_gm,
                    name="data_gm_in_{}".format(index)))

        self.gm_out = self.tik_instance.Tensor(
            "float16", (self.dst_size,), scope=tik.scope_gm, name="data_gm_out")

    def whether_last_dim_same(self):
        """check whether all the last dim of inputs are the same
        """
        last_dim_the_same = True
        input_last_dim = self.input_shapes[0][-1]
        for i, _ in enumerate(self.input_shapes):
            if input_last_dim != self.input_shapes[i][-1]:
                last_dim_the_same = False
                break
        return last_dim_the_same

    def check_vnchw_supported(self):
        """
        check if vnchw schedule support this shape

        Returns
        -------
        if_supported: bool
            if vnchw schedule support this shape
        """
        last_dim_the_same = self.whether_last_dim_same()
        if not last_dim_the_same \
                or len(self.output_shape) == 1 \
                or self.input_num == 1:
            return False

        input_last_dim = self.input_shapes[0][-1]
        output_last_dim = self.output_shape[-1]

        if output_last_dim != self.input_num * input_last_dim:
            return False

        sup_shape = [1, 2, 4, 8]
        sup_count = [2, 4, 8, 16]
        factor = 1
        if self.data_dtype == "float32":
            factor = 2

        if self.data_dtype in ["float32", "float16"] \
                and input_last_dim in sup_shape \
                and self.input_num in sup_count \
                and output_last_dim * factor <= 16 and self.src_size >= 256:
            self.use_last_dim = True

        if self.data_dtype in ["float16"] \
                and input_last_dim == 1 and self.src_size >= TRANSPOSE_SIZE:
            self.use_last_dim = True

        if self.data_dtype == "float16" \
                and input_last_dim == 1 \
                and self.input_num == 3 \
                and self.src_size >= TRANSPOSE_SIZE * 8:
            self.use_last_dim_odd = True

        return self.use_last_dim or self.use_last_dim_odd

    def get_tensor_size_in_fp16(self, data_shape):
        """get_tensor_size_in_fp16
        """
        data_size = functools_reduce(lambda x, y: x * y, data_shape)
        fp16_size = data_size
        if self.data_dtype == "float32":
            fp16_size = fp16_size * 2
        return fp16_size

    def concat_last_dim_one_core(self, src_core_offset, des_core_offset,
                                 core_pro, mov_tail, max_mov_num):
        """concat_last_dim_one_core
        """
        # per core scedule
        if mov_tail != 0:
            core_pro = core_pro - 1

        core_pro = max(core_pro, 0)
        core_loop = core_pro // max_mov_num
        core_tail = core_pro % max_mov_num

        input_ub_0 = \
            self.tik_instance.Tensor("float16", (256*max_mov_num,),
                                     tik.scope_ubuf, "input_ub_0")
        vnchw_ub_0 = \
            self.tik_instance.Tensor("float16",
                                     (256*self.input_num*max_mov_num,),
                                     tik.scope_ubuf, "vnchw_ub_0")
        out_ub_0 = \
            self.tik_instance.Tensor("float16",
                                     (256*self.input_num*max_mov_num,),
                                     tik.scope_ubuf, "out_ub_0")
        input_ub_1 = self.tik_instance.Tensor("float16", (256*max_mov_num,),
                                              tik.scope_ubuf, "input_ub_1")
        vnchw_ub_1 = \
            self.tik_instance.Tensor("float16",
                                     (256*self.input_num*max_mov_num,),
                                     tik.scope_ubuf, "vnchw_ub_1")
        out_ub_1 = self.tik_instance.Tensor("float16",
                                            (256*self.input_num*max_mov_num,),
                                            tik.scope_ubuf, "out_ub_1")

        tiling_ub_list_0 = [input_ub_0, vnchw_ub_0, out_ub_0]
        tiling_ub_list_1 = [input_ub_1, vnchw_ub_1, out_ub_1]

        def _run_copy_input_and_vnchw(input_idx, ub_list, gm_input_offset,
                                      run_mov_num, copy_tail):
            copy_len = run_mov_num * TRANSPOSE_SIZE - copy_tail
            nburst = get_ceil_int(copy_len,
                                  ONE_BLOCK_FP16_SIZE)
            ub_copy, ub_vnchw, _, gm_input = ub_list
            # copy gm to ub
            self.tik_instance.data_move(ub_copy,
                                        gm_input[gm_input_offset],
                                        0, 1, nburst, 0, 0)
            # vnchwconv to ub_vnchw
            _src_addrs = [
                ub_copy[ONE_BLOCK_FP16_SIZE * x]
                for x in range(ONE_BLOCK_FP16_SIZE)
            ]
            dst_offset_ub = self.last_dim * input_idx
            dst_loop = ONE_BLOCK_FP16_SIZE // self.last_dim
            inner_offset_ub = self.last_dim * self.input_num
            _dst_addrs = []
            for dloop in range(dst_loop):
                for in_loop in range(self.last_dim):
                    _dst_addrs.append(
                        ub_vnchw[
                            (dst_offset_ub + dloop * inner_offset_ub + in_loop)
                            * ONE_BLOCK_FP16_SIZE])
            _dst_rep_stride = 16 * self.input_num
            _src_rep_stride = 16
            if run_mov_num == 1:
                _dst_rep_stride = 0
                _src_rep_stride = 0
            self.tik_instance.vnchwconv(False, False, _dst_addrs,
                                        _src_addrs, run_mov_num,
                                        _dst_rep_stride, _src_rep_stride)

        def _run_vnchw_all_to_out(ub_list, gm_output_offset,
                                  run_mov_num, copy_tail):
            _, ub_vnchw, ub_out, gm_output = ub_list
            _dst_rep_stride = (256 // 16) * self.input_num
            _src_rep_stride = (256 // 16) * self.input_num
            if run_mov_num == 1:
                _dst_rep_stride = 0
                _src_rep_stride = 0
            for i_idx in range(self.input_num):
                _src_addrs = [
                    ub_vnchw[i_idx * TRANSPOSE_SIZE + ONE_BLOCK_FP16_SIZE * x]
                    for x in range(ONE_BLOCK_FP16_SIZE)
                ]
                _dst_addrs = [
                    ub_out[(i_idx + x * self.input_num) *
                           ONE_BLOCK_FP16_SIZE]
                    for x in range(ONE_BLOCK_FP16_SIZE)
                ]
                self.tik_instance.vnchwconv(False, False, _dst_addrs,
                                            _src_addrs, run_mov_num,
                                            _dst_rep_stride, _src_rep_stride)
            copy_len = \
                (run_mov_num * TRANSPOSE_SIZE - copy_tail)*self.input_num
            nburst = get_ceil_int(copy_len,
                                  ONE_BLOCK_FP16_SIZE)
            # copy ub to gm
            self.tik_instance.data_move(gm_output[gm_output_offset], ub_out, 0,
                                        1, nburst, 0, 0)

        def _run_one_loop(tiling_ub_list, _loop_offset,
                          run_mov_num, copy_tail=0):
            src_offset = \
                src_core_offset \
                + _loop_offset
            dst_offset = \
                des_core_offset \
                + _loop_offset * self.input_num
            # copy input one by one and vnchwconv input to vnchw_ub
            for i_idx in range(self.input_num):
                _run_copy_input_and_vnchw(
                    i_idx,
                    tiling_ub_list + [self.gm_in[i_idx]],
                    src_offset, run_mov_num, copy_tail
                )

            # vnchwconv vnchw_ub to res_ub and copy un to gm out
            _run_vnchw_all_to_out(
                tiling_ub_list + [self.gm_out],
                dst_offset, run_mov_num, copy_tail
            )

        with self.tik_instance.for_range(
                0, core_loop // 2) as loop_idx:
            _idx = loop_idx*2
            _run_one_loop(tiling_ub_list_0,
                          _idx * TRANSPOSE_SIZE * max_mov_num,
                          max_mov_num)
            _idx = loop_idx*2 + 1
            _run_one_loop(tiling_ub_list_1,
                          _idx * TRANSPOSE_SIZE * max_mov_num,
                          max_mov_num)

        if core_loop % 2 == 1:
            _idx = core_loop - 1
            _run_one_loop(tiling_ub_list_0,
                          _idx * TRANSPOSE_SIZE * max_mov_num,
                          max_mov_num)

        if core_tail != 0:
            _idx = core_loop
            _run_one_loop(tiling_ub_list_1,
                          _idx * TRANSPOSE_SIZE * max_mov_num,
                          core_tail)

        if mov_tail != 0:
            _offset = core_loop * TRANSPOSE_SIZE * max_mov_num \
                      + core_tail * TRANSPOSE_SIZE
            _run_one_loop(tiling_ub_list_0, _offset, 1, mov_tail)

    def concat_last_dim_compute(self):
        """concat_last_dim_compute
        """
        ub_half_size = \
            int(tbe_platform.CceProductParams().getParams("Unified_Buffer")
                // 2 // 2 - 16)
        max_mov_num = \
            ub_half_size // TRANSPOSE_SIZE // (self.input_num*2 + 1)

        # core scedule
        mov_num = (self.src_size + TRANSPOSE_SIZE - 1) // TRANSPOSE_SIZE
        mov_tail = mov_num*TRANSPOSE_SIZE - self.src_size

        move_num_per_core = get_ceil_int(mov_num, self.core_num)

        core_used = mov_num // move_num_per_core
        if mov_num % move_num_per_core != 0:
            core_used = core_used + 1
        move_num_core_tail = \
            mov_num - (core_used - 1)*move_num_per_core

        # define concat fuction
        concat_fuc = self.concat_last_dim_one_core

        with self.tik_instance.for_range(
                0, core_used, block_num=core_used) as core_idx:
            src_offset_core = \
                core_idx * move_num_per_core * TRANSPOSE_SIZE
            dst_offset_core = src_offset_core * self.input_num
            if mov_tail == 0 and move_num_core_tail == move_num_per_core:
                concat_fuc(
                    src_offset_core, dst_offset_core,
                    move_num_per_core, 0, max_mov_num)
            else:
                with self.tik_instance.if_scope(
                        core_idx < (core_used - 1)):
                    concat_fuc(
                        src_offset_core, dst_offset_core,
                        move_num_per_core, 0, max_mov_num)

                with self.tik_instance.else_scope():
                    concat_fuc(
                        src_offset_core, dst_offset_core,
                        move_num_core_tail, mov_tail, max_mov_num)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=self.gm_in,
            outputs=[self.gm_out],
            enable_l2=False)

        return self.tik_instance

    def do_concat_vnchw(self):
        """do_concat_vnchw for last dims
        """
        self.concat_last_dim_compute()


class ConcatWith5HD:
    """Function: use to finish ConcatWith5HD main functions
    """

    def __init__(self, input_data, output_data,
                 axis, kernel_name="concat_last_dim"):
        """init concat base parameters
        """
        self.data_dtype = input_data[0].get("dtype").lower()
        self.format = input_data[0].get("format")
        self.ori_format = input_data[0].get("ori_format")
        self.input_shapes = []
        self.input_ori_shapes = []
        self.src_size_list = []
        self.axis = axis
        for _, input_dict in enumerate(input_data):
            shape_input = input_dict.get("shape")
            self.input_shapes.append(shape_input)
            shape_input = input_dict.get("ori_shape")
            self.input_ori_shapes.append(shape_input)
            src_size = int(self.get_tensor_size_in_fp16(shape_input))
            self.src_size_list.append(src_size)

        self.last_ori_dim = 0
        self.last_ori_dim_list = []
        self.shape_5hd_c0 = 16
        self.input_num = len(input_data)
        self.output_shape = output_data.get("shape")
        self.output_ori_shape = output_data.get("ori_shape")
        self.kernel_name = kernel_name
        self.dst_size = int(self.get_tensor_size_in_fp16(self.output_shape))
        self.gm_out = None
        self.gm_in = []
        self.tik_instance = None
        self.core_num = 0
        self.is_the_same_c = False

    def check_shape_support(self):
        """check_shape_support
        """
        if self.ori_format in ("NCHW",):
            shape_c_dim = 1
        elif self.ori_format in ("NHWC",):
            shape_c_dim = 3
        else:
            return False, -1

        if len(self.input_ori_shapes[0]) != 4:
            return False, shape_c_dim

        input_shape = self.input_ori_shapes[0]
        self.is_the_same_c = True

        for i, _ in enumerate(self.input_ori_shapes):
            if input_shape != self.input_ori_shapes[i]:
                self.is_the_same_c = False

        output_support = False
        output_shape_c = self.output_ori_shape[shape_c_dim]
        if output_shape_c <= 16:
            output_support = True

        input_shape_c = input_shape[shape_c_dim]

        # shape equal and c in (2, 4, 8)
        if self.is_the_same_c and input_shape_c in (2, 4, 8):
            return True, shape_c_dim

        # shape not equal and output c < one block
        if output_support and not self.is_the_same_c:
            return True, shape_c_dim

        return False, shape_c_dim

    def check_op_select(self):
        """function for op_select_format in concat
        """
        is_shape_support, c_dim_num = self.check_shape_support()
        if not is_shape_support:
            return False

        if self.axis % 4 != c_dim_num:
            return False

        return True

    def check_5hd_vnchw(self):
        """check_5hd_vnchw
        """
        is_shape_support, c_dim_num = self.check_shape_support()

        if not is_shape_support:
            return False
        if self.format not in ("NC1HWC0",):
            return False

        if self.axis % 5 != 1 \
                or self.data_dtype not in ("float16", "int16", "uint16"):
            return False

        for i, _ in enumerate(self.input_ori_shapes):
            input_c = self.input_ori_shapes[i][c_dim_num]
            self.last_ori_dim_list.append(input_c)
        self.last_ori_dim = max(self.last_ori_dim_list)

        return True

    def init_gm_tensor(self):
        """init_gm_tensor
        """
        self.tik_instance = tik.Tik()
        self.core_num = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)

        for index, _ in enumerate(self.input_shapes):
            self.gm_in.append(
                self.tik_instance.Tensor(
                    "float16", (self.src_size_list[index],),
                    scope=tik.scope_gm,
                    name="data_gm_in_{}".format(index)))

        self.gm_out = self.tik_instance.Tensor(
            "float16", (self.dst_size,),
            scope=tik.scope_gm, name="data_gm_out")

    def get_tensor_size_in_fp16(self, data_shape):
        """get_tensor_size_in_fp16
        """
        data_size = functools_reduce(lambda x, y: x * y, data_shape)
        fp16_size = data_size
        if self.data_dtype == "float32":
            fp16_size = fp16_size * 2
        return fp16_size

    def do_5hd_concat_cut_by_batch(self):
        """do_5hd_concat_cut_by_batch
        """
        if self.data_dtype == "float32":
            self.last_ori_dim = self.last_ori_dim*2
            self.shape_5hd_c0 = self.shape_5hd_c0*2

        self.init_gm_tensor()
        shape_n = self.output_shape[0]
        shape_c1 = self.output_shape[1]
        shape_h = self.output_shape[2]
        shape_w = self.output_shape[3]
        batch_offset = shape_h*shape_w*self.shape_5hd_c0
        batch_per_core = get_ceil_int(shape_n, self.core_num)
        core_used = get_ceil_int(shape_n, batch_per_core)
        core_tail = \
            shape_n - (core_used - 1)*batch_per_core

        # define concat fuction
        concat_fuc = self.do_5hd_concat_scedule

        with self.tik_instance.for_range(
                0, core_used, block_num=core_used) as core_idx:
            src_offset_core = \
                core_idx * batch_per_core * batch_offset
            dst_offset_core = src_offset_core * shape_c1
            if core_tail == batch_per_core:
                concat_fuc(
                    src_offset_core, dst_offset_core,
                    batch_per_core)
            else:
                with self.tik_instance.if_scope(
                        core_idx < (core_used - 1)):
                    concat_fuc(
                        src_offset_core, dst_offset_core,
                        batch_per_core)

                with self.tik_instance.else_scope():
                    concat_fuc(
                        src_offset_core, dst_offset_core,
                        core_tail)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=self.gm_in,
            outputs=[self.gm_out],
            enable_l2=False)

        return self.tik_instance

    def do_5hd_concat_scedule(self, input_offset, output_offset,
                              process_batch):
        """do_5hd_concat_cut_by_batch
        """
        # get ub size, max fp16 num in half ub
        ub_half_size = \
            int(tbe_platform.CceProductParams().getParams("Unified_Buffer")
                // 2 // 2 - 16)

        if self.is_the_same_c:
            inner_sigment = self.shape_5hd_c0 // self.last_ori_dim
        else:
            inner_sigment = self.input_num
            if sum(self.last_ori_dim_list) != 16:
                inner_sigment = inner_sigment + 1

        max_transpose_sigment = \
            ub_half_size // TRANSPOSE_SIZE // 3

        copy_size = \
            self.output_shape[2]*self.output_shape[3]*self.shape_5hd_c0
        copy_loop = copy_size // (max_transpose_sigment*TRANSPOSE_SIZE)
        copy_loop_tail = copy_size % (max_transpose_sigment*TRANSPOSE_SIZE)
        tail_transpose_sigment = copy_loop_tail // TRANSPOSE_SIZE
        copy_nchw_tail = \
            get_ceil_int(copy_loop_tail, TRANSPOSE_SIZE)*TRANSPOSE_SIZE \
            - copy_loop_tail

        input_size = TRANSPOSE_SIZE*max_transpose_sigment
        input_ub_0 = \
            self.tik_instance.Tensor("float16", (input_size,),
                                     tik.scope_ubuf, "input_ub_0")
        vnchw_ub_0 = \
            self.tik_instance.Tensor("float16",
                                     (input_size,),
                                     tik.scope_ubuf, "vnchw_ub_0")
        out_ub_0 = \
            self.tik_instance.Tensor("float16",
                                     (input_size,),
                                     tik.scope_ubuf, "out_ub_0")
        input_ub_1 = self.tik_instance.Tensor("float16", (input_size,),
                                              tik.scope_ubuf, "input_ub_1")
        vnchw_ub_1 = \
            self.tik_instance.Tensor("float16",
                                     (input_size,),
                                     tik.scope_ubuf, "vnchw_ub_1")
        out_ub_1 = self.tik_instance.Tensor("float16",
                                            (input_size,),
                                            tik.scope_ubuf, "out_ub_1")

        tiling_ub_list_0 = [input_ub_0, vnchw_ub_0, out_ub_0]
        tiling_ub_list_1 = [input_ub_1, vnchw_ub_1, out_ub_0]
        out_ub_list = [out_ub_0, out_ub_1, out_ub_0]

        def _run_copy_input_and_vnchw(input_idx, ub_list, gm_input_offset,
                                      run_mov_num, copy_tail):
            copy_len = run_mov_num * TRANSPOSE_SIZE - copy_tail
            nburst = get_ceil_int(copy_len,
                                  ONE_BLOCK_FP16_SIZE)
            ub_copy, ub_vnchw, ub_out, gm_input = ub_list
            # copy gm to ub
            self.tik_instance.data_move(ub_copy,
                                        gm_input[gm_input_offset],
                                        0, 1, nburst, 0, 0)
            # vnchwconv to ub_vnchw
            _src_addrs = [
                ub_copy[ONE_BLOCK_FP16_SIZE * x]
                for x in range(ONE_BLOCK_FP16_SIZE)
            ]
            _dst_addrs = [
                ub_vnchw[ONE_BLOCK_FP16_SIZE * x]
                for x in range(ONE_BLOCK_FP16_SIZE)
            ]
            _dst_rep_stride = 16
            _src_rep_stride = 16
            if run_mov_num == 1:
                _dst_rep_stride = 0
                _src_rep_stride = 0
            self.tik_instance.vnchwconv(False, False, _dst_addrs,
                                        _src_addrs, run_mov_num,
                                        _dst_rep_stride, _src_rep_stride)

            # copy vnchw_ub_0 to out_ub_0
            nburst = self.last_ori_dim_list[input_idx]
            des_c_dim = \
                sum(self.last_ori_dim_list[:input_idx]) % ONE_BLOCK_FP16_SIZE
            des_offset = des_c_dim * ONE_BLOCK_FP16_SIZE
            src_repeat_block = \
                self.shape_5hd_c0 - self.last_ori_dim_list[input_idx]
            des_repeat_block = \
                self.shape_5hd_c0 - self.last_ori_dim_list[input_idx]
            self.tik_instance.data_move(ub_out[des_offset],
                                        ub_vnchw,
                                        0, run_mov_num,
                                        nburst,
                                        src_repeat_block,
                                        des_repeat_block)

        def _run_vector_dump(ub_list, run_mov_num):
            _, _, ub_out, _ = ub_list
            # copy gm to ub
            vector_size = run_mov_num * TRANSPOSE_SIZE
            repeat = get_ceil_int(vector_size, ONE_BLOCK_FP16_SIZE*8)
            self.tik_instance.vector_dup(ONE_BLOCK_FP16_SIZE*8,
                                         ub_out, 0.0,
                                         repeat, 1, 8)

        def _run_vnchw_all_to_out(ub_list, gm_output_offset,
                                  run_mov_num, copy_tail):
            _, ub_vnchw, ub_out, gm_output = ub_list
            _dst_rep_stride = 16
            _src_rep_stride = 16
            if run_mov_num == 1:
                _dst_rep_stride = 0
                _src_rep_stride = 0

            _src_addrs = [
                ub_out[ONE_BLOCK_FP16_SIZE * x]
                for x in range(ONE_BLOCK_FP16_SIZE)
            ]
            _dst_addrs = [
                ub_vnchw[ONE_BLOCK_FP16_SIZE * x]
                for x in range(ONE_BLOCK_FP16_SIZE)
            ]
            self.tik_instance.vnchwconv(False, False, _dst_addrs,
                                        _src_addrs, run_mov_num,
                                        _dst_rep_stride, _src_rep_stride)

            copy_len = \
                (run_mov_num * TRANSPOSE_SIZE - copy_tail)
            nburst = get_ceil_int(copy_len,
                                  ONE_BLOCK_FP16_SIZE)
            # copy ub to gm
            self.tik_instance.data_move(gm_output[gm_output_offset], ub_vnchw,
                                        0, 1, nburst, 0, 0)

        def _run_one_loop(tiling_ub_list, _offset_list, _loop_offset,
                          run_mov_num, copy_tail=0):
            batch_input_offset, batch_output_offset = _offset_list
            src_offset = \
                batch_input_offset \
                + _loop_offset
            dst_offset = \
                batch_output_offset \
                + _loop_offset
            # copy input one by one and vnchwconv input to vnchw_ub
            c1_loop = get_ceil_int(self.input_num, inner_sigment)
            c1_tail = self.input_num % inner_sigment
            for c1_idx in range(c1_loop):
                if c1_tail != 0 and c1_idx == c1_loop - 1:
                    _run_vector_dump(tiling_ub_list + [self.gm_out],
                                     run_mov_num)
                for i_idx in range(inner_sigment):
                    input_idx = c1_idx*inner_sigment + i_idx
                    if input_idx % 2 == 0:
                        tiling_ub = tiling_ub_list_0
                    else:
                        tiling_ub = tiling_ub_list_1
                    if input_idx < self.input_num:
                        _run_copy_input_and_vnchw(
                            input_idx,
                            tiling_ub + [self.gm_in[input_idx]],
                            src_offset, run_mov_num, copy_tail
                        )

                # vnchwconv vnchw_ub to res_ub and copy ub to gm out
                _run_vnchw_all_to_out(
                    out_ub_list + [self.gm_out],
                    dst_offset
                    + c1_idx*self.output_shape[2]
                    * self.output_shape[3]*self.shape_5hd_c0,
                    run_mov_num, copy_tail
                )

        with self.tik_instance.for_range(0, process_batch) as _batch_idx:
            _batch_input_offset = \
                input_offset + _batch_idx*copy_size
            _batch_output_offset = \
                output_offset + _batch_idx*copy_size*self.output_shape[1]
            batch_offset_list = [_batch_input_offset, _batch_output_offset]
            # copy input to ub
            if copy_loop > 0:
                with self.tik_instance.for_range(
                        0, copy_loop) as loop_idx:
                    _idx = loop_idx
                    _run_one_loop(tiling_ub_list_0, batch_offset_list,
                                  _idx * TRANSPOSE_SIZE * max_transpose_sigment,
                                  max_transpose_sigment)

            if tail_transpose_sigment != 0:
                _idx = copy_loop
                _run_one_loop(tiling_ub_list_1, batch_offset_list,
                              _idx * TRANSPOSE_SIZE * max_transpose_sigment,
                              tail_transpose_sigment)

            if copy_nchw_tail != 0:
                offset = \
                    copy_loop*max_transpose_sigment + tail_transpose_sigment
                _run_one_loop(tiling_ub_list_1, batch_offset_list,
                              offset * TRANSPOSE_SIZE,
                              1, copy_nchw_tail)

