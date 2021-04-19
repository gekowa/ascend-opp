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
concat_l1fusion
"""
from functools import reduce as functools_reduce
from te import tik
from te import platform as tbe_platform
import numpy as np


def get_offset_and_mask(dim, shape_list, output_shape,
                        align_len):
    """
    get offset and mask
    for exp:
       input:
         align_len = 8 (fp32)
         input_1 = [n, 16]
         input_2 = [n, 12]
         output = [n, 28]
       output:
         min_align = 2
         start_offset_all = [[0, 28], [16, 44]]
         mask_value_all = [[[0, 2, 0], [4, 1, 4]], [[0, 1, 4], [4, 1, 0]]]

    Parameters
    ----------
    dim: int
        concat axis
    shape_list: list
        input shape list for concat
    output_shape: list
        output shape for concat
    align_len: int
        ele number in one block

    Returns
    -------
    loop_list : list
         min align loop num per input
    start_offset_all : list
        the offset in the align loop num for inputs
    mask_value_all: list
        the mask in the align loop num for inputs
        [first_mask, repeat, last_mask]
        repeat = align_len // align_len
    """
    offset = 0
    start_offset_all = []
    mask_value_all = []
    loop_list = []
    for _, input_index in enumerate(range(len(shape_list))):
        # calcu repeat loop with input and output
        mask_value = []
        start_offset = []
        concat_size = shape_list[input_index][dim]
        loop_idx = 0
        input_offset_list = []

        for idx, _ in enumerate(range(align_len)):
            # calcu input and output align loop for each input
            input_offset = (concat_size*idx) % align_len
            output_offset = (output_shape[dim]*idx) % align_len
            input_offset_list.append(input_offset)
            if len(input_offset_list) == 1:
                pass
            elif input_offset == output_offset \
                    and input_offset == input_offset_list[0]:
                loop_idx = idx
                break
            else:
                loop_idx = align_len

            mask_1 = \
                align_len \
                - ((offset + output_shape[dim]*idx) % align_len)
            mask_1 = mask_1 % align_len
            mask_1 = mask_1 % align_len
            mask_2 = (concat_size - mask_1) // align_len
            mask_3 = (concat_size - mask_1) % align_len

            mask_value.append([mask_1, mask_2, mask_3])
            start_offset.append(offset + output_shape[dim]*idx)

        loop_list.append(loop_idx)
        start_offset_all.append(start_offset)
        mask_value_all.append(mask_value)
        offset = offset + concat_size

    return loop_list, start_offset_all, mask_value_all


def get_ceil_int(int1, int2):
    """get cel for input1 and input2
    """
    if int1 == 0:
        return 1
    _result = int1 // int2
    if int1 % int2 == 0:
        return _result

    return _result + 1


def cal_loop(ele_num, ub_size, align_len):
    """
    calcute loop

    Parameters
    ----------
    ele_num: int
        total number
    ub_size: int
        ele number in one block

    Returns
    -------
    loop: int
        loop number
    ele_each_loop: int
        ele number in one loop
    """
    loop = ele_num // ub_size
    tail = ele_num % ub_size
    ele_each_loop = (ub_size // align_len)*align_len
    if tail <= (ub_size * 0.8):
        loop = loop * 2
        ele_each_loop = ((ele_num // loop) // align_len)*align_len
        if ele_num % loop != 0:
            loop = ele_num // ub_size
            ele_each_loop = (ub_size // align_len)*align_len

    return loop, ele_each_loop


class ConcatL1Fusion:
    def __init__(self, input_values, output_data, axis, kernel_name):
        self.tik_instance = tik.Tik()
        self.aicore_num = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)

        self.dtype = input_values[0].get("dtype").lower()
        self.output_shape = output_data.get("shape")

        self.input_addr_type = input_values[0]["addr_type"]
        self.output_addr_type = output_data["addr_type"]

        self.input_slice_offset = input_values[0]["slice_offset"]
        self.output_slice_offset = output_data["slice_offset"]
        if len(self.output_slice_offset) == 0:
            self.jump_write = False
        else:
            self.jump_write = True
        if len(self.input_slice_offset) == 0:
            self.jump_read = False
        else:
            self.jump_read = True

        self.input_shapes, self.concat_axis, self.output_shape, self.input_slice_offset, self.output_slice_offset, self.input_valid_shapes = \
            self.reshape_simple(input_values, output_data, axis)

        self.input_tensors, self.output_tensor = self.init_tensor(
            self.input_shapes, self.output_shape, self.dtype, self.input_addr_type, self.output_addr_type)

        dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(self.dtype) // 8
        self.ele_each_block = 32 // dtype_bytes_size

        self.ub_half_size = \
            (tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
             // dtype_bytes_size // 2 - self.ele_each_block)
        self.max_input_dim_size = 0
        self.is_data_move_first_dim = False
        self.kernel_name = kernel_name

    def check_support_l1_fusion(self):
        max_dims, _, _ = self.get_max_dims_remainder_half()
        if self.max_input_dim_size < self.ub_half_size \
                and len(self.output_shape) == 2 \
                and self.aicore_num <= self.output_shape[0]:
            self.is_data_move_first_dim = True
        return True

    def get_max_dims_remainder_half(self):
        loop_num_list, _, _ = \
            get_offset_and_mask(self.concat_axis, self.input_shapes,
                                self.output_shape, self.ele_each_block)
        loop_num = max(loop_num_list)
        thread_num = 2
        # get max input size
        max_input_dim_size = 0
        for _, shape in enumerate(self.input_shapes):
            max_input_dim_size = \
                max(max_input_dim_size, shape[self.concat_axis])

        # get ub size for one dim, 1 out + 2 max size of input
        ub_need_one_dim = \
            max_input_dim_size*2 + self.output_shape[self.concat_axis] \
            + 2*self.ele_each_block

        max_dims = self.ub_half_size // ub_need_one_dim

        max_dims = (max_dims // (loop_num*8))*loop_num*8
        if max_dims <= 0:
            max_dims = \
                (self.ub_half_size*2 - max_input_dim_size) // ub_need_one_dim
            max_dims = (max_dims // (loop_num*8))*loop_num*8
            thread_num = 1

        max_dims = max(max_dims, 0)

        self.max_input_dim_size = max_input_dim_size

        if len(self.output_shape) == 1 \
                or self.output_shape[0] <= self.ele_each_block:
            return 0, 0, 0

        return max_dims, thread_num, max_input_dim_size

    def do_concat_l1fusion(self):
        if self.is_data_move_first_dim:
            print(1.1)
            self.data_move_cut_by_fisrt_dim()
        else:
            # tensor move branch
            print(1.4)
            with self.tik_instance.for_range(
                    0, self.aicore_num, block_num=self.aicore_num) as index:
                self.concat_compute_each_core(index)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=self.input_tensors,
            outputs=(self.output_tensor,),
            )

        return self.tik_instance

    def proc_data_scedule_align(self, _core_index,
                                core_dims_offset, core_process):
        """proc_data_scedule
        """
        output_gm_offset = []
        for idx, input_shape in enumerate(self.input_shapes):
            if idx == 0:
                output_gm_offset.append(0)
                continue
            pre_offset = output_gm_offset[idx - 1]
            pre_len = self.input_shapes[idx - 1][-1]
            cu_offset = pre_offset + pre_len
            output_gm_offset.append(cu_offset)

        def _copy_one_dim(src_gm, des_gm, _dims_idx,
                          _input_idx, dim_size, _ub_tensor):
            """
            add attr

            """
            if self.jump_read:
                input_offest = self.input_slice_offset[_input_idx][2]

                input_w = self.input_tensors[_input_idx]["shape"][3]
                input_co = self.input_tensors[_input_idx]["shape"][4]
                src_offset = _dims_idx*dim_size
                src_offset = src_offset + _dims_idx*input_offest*input_w*input_co
                slice_dim_size = self.input_valid_shapes[_input_idx][1]
                burst_len = get_ceil_int(slice_dim_size,
                                         self.ele_each_block)
            else:
                src_offset = _dims_idx*dim_size
                burst_len = get_ceil_int(dim_size,
                                         self.ele_each_block)
            if self.jump_write:
                output_offest = self.output_slice_offset[2]

                output_w = self.output_slice_offset[3]
                output_co = self.output_slice_offset[4]
                out_offest = output_offest * output_w * output_co
                des_offset = \
                    _dims_idx*self.output_shape[1] + output_gm_offset[_input_idx]+out_offest
            else:
                des_offset = \
                    _dims_idx*self.output_shape[1] + output_gm_offset[_input_idx]


            self.tik_instance.data_move(
                _ub_tensor,
                src_gm[src_offset],
                0, 1, burst_len, 1, 1)
            self.tik_instance.data_move(
                des_gm[des_offset],
                _ub_tensor,
                0, 1, burst_len, 1, 1)

        input_num = len(self.input_shapes)

        ub_tensor = self.tik_instance.Tensor(
            self.dtype,
            (self.ub_half_size,),
            name="ub_tensor", scope=tik.scope_ubuf)
        ub_tensor_1 = self.tik_instance.Tensor(
            self.dtype,
            (self.ub_half_size,),
            name="ub_tensor_1", scope=tik.scope_ubuf)
        for _, _index in enumerate(range(input_num)):
            input_index = _index
            copy_ub_0, copy_ub_1 = [ub_tensor, ub_tensor_1]
            if core_process % 2 == 1 and input_index % 2 == 1:
                copy_ub_0, copy_ub_1 = [ub_tensor_1, ub_tensor]
            with self.tik_instance.for_range(
                    0, core_process // 2) as _dims_loop:
                _copy_one_dim(self.input_tensors[input_index],
                              self.output_tensor,
                              _dims_loop*2 + core_dims_offset,
                              input_index,
                              self.input_shapes[input_index][1],
                              copy_ub_0)
                _copy_one_dim(self.input_tensors[input_index],
                              self.output_tensor,
                              _dims_loop*2 + 1 + core_dims_offset,
                              input_index,
                              self.input_shapes[input_index][1],
                              copy_ub_1)
            if (core_process % 2) != 0:
                _copy_one_dim(self.input_tensors[input_index],
                              self.output_tensor,
                              core_process - 1 + core_dims_offset,
                              input_index,
                              self.input_shapes[input_index][1],
                              copy_ub_0)

    def data_move_cut_by_fisrt_dim(self):

        concat_fuc = self.proc_data_scedule_align
        inner_loop = 1
        data_size_first_dim = self.input_shapes[0][0]
        core_len = get_ceil_int(data_size_first_dim, inner_loop)
        core_len = get_ceil_int(core_len, self.aicore_num)
        if core_len == 0:
            core_len = 1

        dims_per_core = core_len * inner_loop
        core_used = get_ceil_int(data_size_first_dim, dims_per_core)
        tail_dims_core = \
            data_size_first_dim - (core_used - 1)*dims_per_core

        # for core loop
        with self.tik_instance.for_range(
                0, core_used, block_num=core_used) as _core_index:
            core_dims_offset = _core_index * dims_per_core
            if tail_dims_core != dims_per_core:
                # for copy segment loop
                with self.tik_instance.if_scope(
                        _core_index < (core_used - 1)):
                    concat_fuc(_core_index, core_dims_offset, dims_per_core)

                with self.tik_instance.else_scope():
                    concat_fuc(_core_index, core_dims_offset, tail_dims_core)
            else:
                concat_fuc(_core_index, core_dims_offset, dims_per_core)

    def concat_compute_for_each_tensor(self, tensor_list, input_tensor_info,
                                       output_tensor_info, ele_num):
        """
        concat each tensor

        Parameters
        ----------
        tensor_list: list
            ub tensor for data move in
        input_tensor_info: list
            input gm tensor, offset
        output_tensor_info: list
            input gm tensor, offset
        ele_num: int
            element number

        Returns
        -------
        None
        """
        input_tensor, move_in_index = input_tensor_info
        output_tensor, move_out_index = output_tensor_info
        loop_burst_len = 0
        if ele_num < self.ub_half_size:
            loop_num = 0
            last_ele = ele_num
            ele_each_loop = 0
        else:
            if ele_num % self.ub_half_size < self.ele_each_block:
                ub_size = self.ub_half_size - self.ele_each_block
            else:
                ub_size = self.ub_half_size
            ub_size = (ub_size // self.ele_each_block)*self.ele_each_block
            last_ele = ele_num % ub_size
            if last_ele == 0:
                loop_num = ele_num // ub_size
                ele_each_loop = ub_size
                loop_burst_len = ub_size // self.ele_each_block
            else:
                loop_num, ele_each_loop = cal_loop(ele_num, ub_size,
                                                   self.ele_each_block)
                loop_burst_len = ele_each_loop // self.ele_each_block
                if ele_each_loop % self.ele_each_block != 0:
                    loop_burst_len = loop_burst_len + 1
                last_ele = ele_num % ele_each_loop

        ping_pang_flag = 0
        loop_burst_len = int(loop_burst_len)
        if loop_num > 0:
            with self.tik_instance.for_range(
                    0, loop_num // 2) as inner_loop:
                ub_tensor = tensor_list[0]
                offset = inner_loop * 2 * ele_each_loop
                self.tik_instance.data_move(ub_tensor,
                                            input_tensor[move_in_index
                                                         + offset],
                                            0, 1, loop_burst_len, 0, 0)
                self.tik_instance.data_move(
                    output_tensor[move_out_index + offset], ub_tensor,
                    0, 1, loop_burst_len, 0, 0)
                ub_tensor = tensor_list[1]
                offset = (inner_loop * 2 + 1) * ele_each_loop
                self.tik_instance.data_move(ub_tensor,
                                            input_tensor[move_in_index
                                                         + offset],
                                            0, 1, loop_burst_len, 0, 0)
                self.tik_instance.data_move(
                    output_tensor[move_out_index + offset], ub_tensor,
                    0, 1, loop_burst_len, 0, 0)
            if loop_num % 2 == 1:
                offset = (loop_num - 1) * ele_each_loop
                ub_tensor = tensor_list[ping_pang_flag]
                self.tik_instance.data_move(ub_tensor,
                                            input_tensor[move_in_index
                                                         + offset],
                                            0, 1, loop_burst_len, 0, 0)
                self.tik_instance.data_move(
                    output_tensor[move_out_index + offset], ub_tensor,
                    0, 1, loop_burst_len, 0, 0)
                ping_pang_flag = (ping_pang_flag + 1) % 2

        if last_ele > 0:
            offset = loop_num * ele_each_loop
            loop_burst_len = last_ele // self.ele_each_block
            ub_tensor = tensor_list[ping_pang_flag]
            if loop_burst_len > 0:
                self.tik_instance.data_move(ub_tensor,
                                            input_tensor[move_in_index
                                                         + offset],
                                            0, 1, loop_burst_len, 0, 0)
                self.tik_instance.data_move(
                    output_tensor[move_out_index + offset], ub_tensor,
                    0, 1, loop_burst_len, 0, 0)
                ping_pang_flag = (ping_pang_flag + 1) % 2

            if last_ele % self.ele_each_block != 0:
                ub_tensor = tensor_list[ping_pang_flag]
                offset = ele_num - self.ele_each_block
                self.tik_instance.data_move(ub_tensor,
                                            input_tensor[move_in_index
                                                         + offset],
                                            0, 1, 1, 0, 0)
                self.tik_instance.data_move(
                    output_tensor[move_out_index + offset], ub_tensor,
                    0, 1, 1, 0, 0)

    def get_loop_para(self, input_shape, concat_axis):
        """
        Get concat loop parameters

        Parameters
        ----------
        input_shape: list
            input shapes
        concat_axis: int
            axis for concat

        Returns
        -------
        loop_num: int
            number of loop
        ele_num_each_loop: int
            element number for each loop
        ele_last: int
            element number of last loop
        """
        total_ele_num = input_shape[concat_axis]
        ele_num_each_loop = get_ceil_int(total_ele_num, self.aicore_num)
        ele_num_each_loop = \
            get_ceil_int(ele_num_each_loop, self.ele_each_block) \
            * self.ele_each_block
        ele_last = total_ele_num % ele_num_each_loop
        use_core_num = get_ceil_int(total_ele_num, ele_num_each_loop)

        return use_core_num, ele_num_each_loop, ele_last

    def concat_compute_each_core(self, core_index):
        """
        concat input tensor on each core

        Parameters
        ----------
        core_index: int
            aicore index

        Returns
        -------
        None
        """
        if len(self.input_shapes[0]) > 1:
            out_loop = self.input_shapes[0][0]
        else:
            out_loop = 1

        # init ub for double buff
        ub_tensor = self.tik_instance.Tensor(
            self.dtype, (self.ub_half_size,),
            name="ub_tensor",
            scope=tik.scope_ubuf)
        ub_tensor_1 = self.tik_instance.Tensor(
            self.dtype, (self.ub_half_size,),
            name="ub_tensor_1",
            scope=tik.scope_ubuf)

        ub_list = [ub_tensor, ub_tensor_1]

        # define fuc for one ele_num
        def _run_one_core(out_loop_idx, _out_offset,
                          ele_num_each_core, ele_num_process,
                          input_idx):
            if self.jump_read:
                input_idx_shape = self.input_shapes[input_idx]
                slice_offect = self.input_slice_offset[input_idx][2]
                slice_w = self.input_slice_offset[input_idx][3]
                slice_c0 = self.input_slice_offset[input_idx][4]
                tensor_slice_offest = slice_offect*slice_c0*slice_w
                move_in_idx = \
                    (ele_num_each_core * core_index + tensor_slice_offest +
                     + out_loop_idx * (input_idx_shape[self.concat_axis]))
                move_out_idx = \
                    (_out_offset + ele_num_each_core * core_index
                     + out_loop_idx * self.output_shape[self.concat_axis])
                self.concat_compute_for_each_tensor(
                    ub_list,
                    [self.input_tensors[input_idx], move_in_idx],
                    [self.output_tensor, move_out_idx], ele_num_process)
            else:
                input_idx_shape = self.input_shapes[input_idx]
                move_in_idx = \
                    (ele_num_each_core * core_index
                     + out_loop_idx * input_idx_shape[self.concat_axis])
                move_out_idx = \
                    (_out_offset + ele_num_each_core * core_index
                     + out_loop_idx * self.output_shape[self.concat_axis])
                self.concat_compute_for_each_tensor(
                    ub_list,
                    [self.input_tensors[input_idx], move_in_idx],
                    [self.output_tensor, move_out_idx], ele_num_process)

        one_core_flag = False
        tensor_index = 0
        ele_num_each_loop = 0
        # do concat the input to output one by one
        out_offset = 0
        for tensor_index, input_shape in enumerate(self.input_shapes):
            # get ele number of one core for one input
            if self.jump_read:
                use_core_num, ele_num_each_loop, ele_last = \
                    self.get_loop_para(self.input_valid_shapes[tensor_index], self.concat_axis)
            else:
                use_core_num, ele_num_each_loop, ele_last = \
                    self.get_loop_para(input_shape, self.concat_axis)
            # when ele_num in core is less than ele_each_block
            # all core process the same data
            one_core_flag = False
            if ele_num_each_loop < self.ele_each_block or \
                    0 < ele_last < self.ele_each_block or \
                    use_core_num != self.aicore_num:
                one_core_flag = True
                ele_last = 0
                ele_num_each_loop = input_shape[self.concat_axis]

            def _run_one_input_double_buff(ele_num):
                with self.tik_instance.for_range(
                        0, out_loop // 2) as loop_index:
                    one_core_idx = loop_index*2
                    if one_core_flag:
                        _run_one_core(one_core_idx, out_offset,
                                      0, ele_num,
                                      tensor_index)
                    else:
                        _run_one_core(one_core_idx, out_offset,
                                      ele_num_each_loop, ele_num,
                                      tensor_index)
                    one_core_idx = loop_index*2 + 1
                    ub_list.reverse()
                    if one_core_flag:
                        _run_one_core(one_core_idx, out_offset,
                                      0, ele_num,
                                      tensor_index)
                    else:
                        _run_one_core(one_core_idx, out_offset,
                                      ele_num_each_loop, ele_num,
                                      tensor_index)
                    ub_list.reverse()
                if out_loop % 2 == 1:
                    one_core_idx = out_loop - 1
                    if one_core_flag:
                        _run_one_core(one_core_idx, out_offset,
                                      0, ele_num,
                                      tensor_index)
                    else:
                        _run_one_core(one_core_idx, out_offset,
                                      ele_num_each_loop, ele_num,
                                      tensor_index)
                    ub_list.reverse()

            if ele_last == 0:
                _run_one_input_double_buff(ele_num_each_loop)
            else:
                with self.tik_instance.if_scope(
                        core_index < self.aicore_num - 1):
                    _run_one_input_double_buff(ele_num_each_loop)
                with self.tik_instance.else_scope():
                    _run_one_input_double_buff(ele_last)

            out_offset += input_shape[self.concat_axis]

    def reshape_simple(self, shape_list, output_dic, concat_axis):
        """
        Init concat base parameters

        Parameters
        ----------
        shape_list: list
            input shapes
        concat_axis: int
            axis for concat

        Returns
        -------
        input_shapes: list
            input shapes
        concat_axis: int
            axis for concat
        """
        input_shapes = []
        input_slice_offset = []
        input_valid_shapes = []
        for _, input_dict in enumerate(shape_list):

            shape_input = input_dict.get("shape")
            valid_shape_input = input_dict.get("valid_shape")
            input_offset = input_dict.get("slice_offset")
            if len(input_offset) != 0:
                input_slice_offset.append(input_offset)
            out_dim = int(np.prod(shape_input[0:concat_axis]))
            if out_dim == 1:
                shape_input = (int(np.prod(shape_input)),)
            else:
                inner_dim = int(np.prod(shape_input[concat_axis:]))
                shape_input = (out_dim, inner_dim)
            if self.jump_read:
                out_dim = int(np.prod(valid_shape_input[0:concat_axis]))
                if out_dim == 1:
                    shape_input = (int(np.prod(valid_shape_input)),)
                else:
                    inner_dim = int(np.prod(valid_shape_input[concat_axis:]))
                    shape_input = (out_dim, inner_dim)
                input_valid_shapes.append(shape_input)

        if len(input_shapes[0]) == 2:
            out_dim = int(np.prod(self.output_shape[0:concat_axis]))
            inner_dim = int(np.prod(self.output_shape[concat_axis:]))
            self.output_shape = (out_dim, inner_dim)
            concat_axis = 1
        else:
            self.output_shape = (int(np.prod(self.output_shape)),)
            concat_axis = 0

        # calcu the output shape again
        out_shape = list(input_shapes[0]).copy()
        out_shape[concat_axis] = 0
        for _, input_shape in enumerate(input_shapes):
            out_shape[concat_axis] = \
                out_shape[concat_axis] + input_shape[concat_axis]

        output_offset = output_dic.get("slice_offset")
        return input_shapes, concat_axis, out_shape, input_slice_offset, output_offset, input_valid_shapes

    def init_tensor(self, input_shapes, output_shape, dtype, input_addr_type, output_addr_type):
        """
        init gm tensor

        Parameters
        ----------
        input_shapes: list
            shape of input tensors
        output_shape: list
            shape of output tensor
        dtype: str
            data type

        Returns
        -------
        input_tensors: tik tensor
            input gm tensor
        output_tensor: tik tensor
            output gm tensor
        """
        input_tensors = []

        if input_addr_type == 1:
            tensor_name_per = "l1_input"
            input_tensor_scope = tik.scope_cbuf
        else:
            tensor_name_per = "gm_input"
            input_tensor_scope = tik.scope_gm
        if output_addr_type == 1:
            output_tensor_name_per = "l1_output"
            output_tensor_scope = tik.scope_cbuf
        else:
            output_tensor_name_per = "gm_output"
            output_tensor_scope = tik.scope_gm

        for index, tensor_shape in enumerate(input_shapes):

            tensor_name = tensor_name_per + str(index)
            tensor = self.tik_instance.Tensor(
                dtype, tensor_shape, name=tensor_name, scope=input_tensor_scope)
            input_tensors.append(tensor)
        if len(self.output_slice_offset) == 0:
            output_tensor = self.tik_instance.Tensor(
                dtype, output_shape, name=output_tensor_name_per, scope=output_tensor_scope)
        else:
            output_tensor = self.tik_instance.Tensor(
                dtype, self.output_valid_shape, name=output_tensor_name_per, scope=output_tensor_scope)

        return input_tensors, output_tensor

