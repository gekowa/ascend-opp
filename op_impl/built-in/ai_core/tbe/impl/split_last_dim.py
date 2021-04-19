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
split_last_dim
"""
from te import tik
from te import platform as tbe_platform
from topi.cce import util

# 1111000011110000111100001111000011110000111100001111000011110000
MASK_FOR_11110000 = 17361641481138401520
# 0000111100001111000011110000111100001111000011110000111100001111
MASK_FOR_00001111 = 1085102592571150095
# vnchwconv can deal 16*16 one time
TRANSPOSE_SIZE = 256
# one block can save the size of fp16
ONE_BLOCK_FP16_SIZE = 16


def _apply_mem(tik_instance, dtype, shape, name, scope=tik.scope_ubuf):
    """apply mem fuc

    Parameters
    ----------
    tik_instance: tik_instance
        tik_instance
    dtype: str
        ub dtype
    shape: list
        ub shape
    name: str
        ub name
    scope: scope
        scope_ubuf or scope_gm

    Returns
    -------
    Tensor: Tensor
    """
    return tik_instance.Tensor(dtype, shape, name=name, scope=scope)


def _get_ceil_int(int1, int2):
    """get cel for input1 and input2
    """
    if int1 == 0:
        return 1
    _result = int1 // int2
    if int1 % int2 == 0:
        return _result

    return _result + 1


# pylint: disable=locally-disabled,too-many-instance-attributes
# pylint: disable=too-many-arguments,unused-argument,too-many-locals
# pylint: disable=too-many-statements
class SplitLastDim():
    """Function: use to finish SplitLastDim main functions
    """
    def __init__(self, shape, dtype, split_dim, num_split, size_splits):
        """init SplitLastDim parameters
        """
        self.src_shape = shape
        self.src_dtype = dtype
        self.data_size = util.check_tensor_shape_size(list(self.src_shape))
        self.split_dim = split_dim
        self.num_split = num_split
        self.split_dim_size = self.src_shape[self.split_dim]

        self.data_size_first_dim = self.data_size // self.split_dim_size
        self.split_output_dim_size = \
            self.src_shape[self.split_dim] // self.num_split
        self.output_size = \
            self.split_output_dim_size * self.data_size_first_dim
        # get dtype size, float16 size = 2 byte   / float32 size = 4 byte
        self.dtype_size = \
            tbe_platform.cce_intrin.get_bit_len(self.src_dtype) // 8
        # get one block data size, block align len
        # the len in one block = 16 fp16 and = 8 fp32
        self.data_len_one_block = 32 // self.dtype_size
        self.data_len_one_vector = self.data_len_one_block * 8

        self.ub_availble = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.UB_SIZE) - 8 * 1024
        self.ub_max_data = self.ub_availble // self.dtype_size
        self.tik_instance = tik.Tik()
        self.core_num = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)
        self.max_dims = 1
        self.segment_len = 1
        self.out_ub = None
        self.out_ub_1 = None
        self.index_reg = None
        self.index_reg_1 = None

        # input and output tensor in gm
        self.src_gm = self.tik_instance.Tensor(
            self.src_dtype, [self.data_size_first_dim, self.split_dim_size],
            name="src_gm",
            scope=tik.scope_gm)
        self.dst_gm_list = []

        for _, i in enumerate(range(num_split)):
            dst_gm = self.tik_instance.Tensor(
                self.src_dtype,
                [self.data_size_first_dim, self.split_output_dim_size],
                name="dst_gm_" + str(i),
                scope=tik.scope_gm)
            self.dst_gm_list.append(dst_gm)

    def split_last_dim_less_block(self):
        """copy all data from src to des
        """
        # core scedule
        self.max_dims = 256 // 2
        inner_loop = self.data_len_one_block
        core_len = _get_ceil_int(self.data_size_first_dim, inner_loop)
        core_len = _get_ceil_int(core_len, self.core_num)
        if core_len == 0:
            core_len = 1

        dims_per_core = core_len * inner_loop
        core_used = self.data_size_first_dim // dims_per_core
        if self.data_size_first_dim % dims_per_core != 0:
            core_used = core_used + 1
        tail_dims_core = \
            self.data_size_first_dim - (core_used - 1)*dims_per_core

        self.segment_len = self.max_dims * self.split_dim_size
        if self.split_output_dim_size == 4 and self.data_len_one_block == 8 \
                and self.num_split % 2 == 0:
            split_fuc = self.proc_4_with_fp32
        else:
            split_fuc = self.proc_default
        # for core loop
        with self.tik_instance.for_range(
                0, core_used, block_num=core_used) as _core_index:
            core_dims_offset = _core_index * dims_per_core
            if tail_dims_core != dims_per_core:
                # for copy segment loop
                with self.tik_instance.if_scope(_core_index < (core_used - 1)):
                    split_fuc(dims_per_core, core_dims_offset)

                with self.tik_instance.else_scope():
                    split_fuc(tail_dims_core, core_dims_offset)
            else:
                split_fuc(dims_per_core, core_dims_offset)

    def proc_4_with_fp32(self, work_dims, core_offset):
        """when output size = 4 run this do_spilt_four_fp32
        0 1 2  3  4  5  6  7
        8 9 10 11 12 13 14 15

        step1:
            copy gm(+0) --> ub1
            copy gm(+4) --> ub2
        step2:
            set_mask(00001111)
            output[0:4] = vadds(ub1(0:4),0)
            set_mask(11110000)
            output[4:8] = vadds(ub2(4:8),0)
        step3:
            copy output to gm
        """
        segment_loop = work_dims * self.split_dim_size // self.segment_len
        segment_tail_len = \
            work_dims * self.split_dim_size - segment_loop*self.segment_len

        def _run_one_segment(_segment_len, _segment_index):
            """_run_one_segment
            """
            # get gm offset
            offset = \
                core_offset*self.split_dim_size + \
                _segment_index*self.segment_len
            out_offset = \
                core_offset*self.split_output_dim_size \
                + _segment_index*self.max_dims*self.split_output_dim_size

            # apply ub for data
            data_ub = _apply_mem(self.tik_instance, self.src_dtype,
                                 [self.segment_len], "data_ub")
            # apply ub for data_1
            data_ub_1 = _apply_mem(self.tik_instance, self.src_dtype,
                                   [self.segment_len], "data_ub_1")
            # calcu len for copy
            burst_len = _get_ceil_int(_segment_len, self.data_len_one_block)
            # copy data from gm to ub1
            self.tik_instance.data_move(data_ub, self.src_gm[offset], 0, 1,
                                        burst_len, 0, 0)
            burst_len = _get_ceil_int(_segment_len - 4, self.data_len_one_block)
            # copy data from gm to ub2
            self.tik_instance.data_move(
                data_ub_1, self.src_gm[offset + self.split_output_dim_size], 0,
                1, burst_len, 0, 0)

            # apply ub to save output
            self.out_ub = \
                _apply_mem(self.tik_instance, self.src_dtype,
                           [self.max_dims*self.split_output_dim_size],
                           "out_ub")
            self.out_ub_1 = \
                _apply_mem(self.tik_instance, self.src_dtype,
                           [self.max_dims*self.split_output_dim_size],
                           "out_ub_1")
            # do split_d use adds_4_to_ub
            self.adds_4_to_ub(_segment_len, [data_ub, data_ub_1], out_offset)

        with self.tik_instance.for_range(0, segment_loop) as _segment_loop:
            _run_one_segment(self.segment_len, _segment_loop)

        if segment_tail_len != 0:
            # process tail data
            _run_one_segment(segment_tail_len, segment_loop)

    # pylint: disable=locally-disabled,too-many-locals
    def adds_4_to_ub(self, segment_len, data_ub_list, out_offset):
        """used adds 0 to move data from input ub to output ub
        """
        data_ub, data_ub_1 = data_ub_list
        dst_m0 = 1
        src_m0 = self.split_dim_size // self.data_len_one_block * 2
        dst_m1 = 8
        src_m1 = self.split_dim_size * 8 * 2 // self.data_len_one_block
        mask1_scalar = self.tik_instance.Scalar(dtype="uint64")
        mask2_scalar = self.tik_instance.Scalar(dtype="uint64")
        mask1_scalar.set_as(MASK_FOR_00001111)
        mask2_scalar.set_as(MASK_FOR_11110000)
        work_dim = segment_len // self.split_dim_size
        repeat_time = _get_ceil_int(work_dim // 2, 8)
        nbust = _get_ceil_int(work_dim * self.split_output_dim_size,
                              self.data_len_one_block)

        def process_one_output(_index, _out_ub):
            if _index % 2 == 0:
                # the output index is odd number
                first_ub = data_ub
                second_ub = data_ub_1
                first_offset = (_index // 2) * self.data_len_one_block
                second_offset = first_offset + self.split_dim_size - 8
            else:
                # the output index is even number
                first_ub = data_ub_1
                second_ub = data_ub
                first_offset = (_index // 2) * self.data_len_one_block
                second_offset = first_offset + self.split_dim_size
            data_ub_first = first_ub[first_offset]
            data_ub_second = second_ub[second_offset]
            # the max value of src_m1 is 255,
            # when src_m1 > 255, connot use repeat for vadds
            if src_m1 <= 255:
                # conditons: src_m1 <= 255 vadds use repeat
                self.tik_instance.vadds([mask1_scalar, mask1_scalar], _out_ub,
                                        data_ub_first, 0, repeat_time, dst_m0,
                                        src_m0, dst_m1, src_m1)
                self.tik_instance.vadds([mask2_scalar, mask2_scalar], _out_ub,
                                        data_ub_second, 0, repeat_time, dst_m0,
                                        src_m0, dst_m1, src_m1)
                self.tik_instance.data_move(
                    self.dst_gm_list[_index][out_offset], _out_ub, 0, 1, nbust,
                    0, 0)
            elif repeat_time == 1:
                # conditons: src_m1 > 255 and repeat_time is equal to 0
                # vector cmd "vadds" ignore src_m1
                self.tik_instance.vadds([mask1_scalar, mask1_scalar], _out_ub,
                                        data_ub_first, 0, repeat_time, dst_m0,
                                        src_m0, dst_m1, 8)
                self.tik_instance.vadds([mask2_scalar, mask2_scalar], _out_ub,
                                        data_ub_second, 0, repeat_time, dst_m0,
                                        src_m0, dst_m1, 8)
            else:
                # vadds 0 one by one
                for _, i in enumerate(range(repeat_time)):
                    data_ub_first = \
                        first_ub[i*src_m1*self.data_len_one_block
                                 + first_offset]
                    data_ub_second = \
                        second_ub[second_offset
                                  + i*src_m1*self.data_len_one_block]
                    self.tik_instance.vadds(
                        [mask1_scalar, mask1_scalar],
                        _out_ub[i * dst_m1 * self.data_len_one_block],
                        data_ub_first, 0, 1, dst_m0, src_m0, dst_m1, 8)
                    self.tik_instance.vadds(
                        [mask2_scalar, mask2_scalar],
                        _out_ub[i * dst_m1 * self.data_len_one_block],
                        data_ub_second, 0, 1, dst_m0, src_m0, dst_m1, 8)

            self.tik_instance.data_move(self.dst_gm_list[_index][out_offset],
                                        _out_ub, 0, 1, nbust, 0, 0)

        for _, output_index in enumerate(range(self.num_split // 2)):
            process_one_output(output_index * 2, self.out_ub)
            process_one_output(output_index * 2 + 1, self.out_ub_1)
        if self.num_split % 2 == 1:
            process_one_output(self.num_split - 1, self.out_ub)

    def proc_default(self, work_dims, core_offset):
        """run this do_spilt use scalar
        """
        segment_loop = work_dims * self.split_dim_size // self.segment_len
        segment_tail_len = \
            work_dims * self.split_dim_size - segment_loop * self.segment_len

        def _run_one_segment(_segment_len, _segment_index):
            # calcu gm offset
            offset = core_offset*self.split_dim_size + \
                     _segment_index*self.segment_len
            out_offset = \
                core_offset*self.split_output_dim_size \
                + _segment_index * self.max_dims * self.split_output_dim_size
            # copy from gm to ub
            data_ub = _apply_mem(self.tik_instance, self.src_dtype,
                                 [self.segment_len], "data_ub")
            nbust = _get_ceil_int(_segment_len, self.data_len_one_block)
            self.tik_instance.data_move(data_ub, self.src_gm[offset], 0, 1,
                                        nbust, 0, 0)
            self.out_ub = \
                _apply_mem(self.tik_instance, self.src_dtype,
                           [self.max_dims*self.split_output_dim_size],
                           "out_ub")
            self.out_ub_1 = \
                _apply_mem(self.tik_instance, self.src_dtype,
                           [self.max_dims*self.split_output_dim_size],
                           "out_ub_1")
            self.index_reg = [
                self.tik_instance.Scalar(dtype=self.src_dtype)
                for _, _ in enumerate(range(8))
            ]
            self.index_reg_1 = [
                self.tik_instance.Scalar(dtype=self.src_dtype)
                for _, _ in enumerate(range(8))
            ]
            self.scalar_to_ub(_segment_len, data_ub, out_offset)

        with self.tik_instance.for_range(0, segment_loop) as _segment_loop:
            _run_one_segment(self.segment_len, _segment_loop)

        if segment_tail_len != 0:
            _run_one_segment(segment_tail_len, segment_loop)

    def scalar_to_ub(self, segment_len, data_ub, out_offset):
        """used scalar to move data from input ub to output ub
        """
        first_loop = segment_len // self.split_dim_size
        if segment_len % self.split_dim_size != 0:
            first_loop = first_loop + 8

        def _run_one_output(_index, _out_ub):
            with self.tik_instance.for_range(0, _get_ceil_int(first_loop,
                                                              8)) as i:
                with self.tik_instance.for_range(
                        0, self.split_output_dim_size) as j:
                    with self.tik_instance.for_range(0, 8) as k:
                        _input_index = \
                            (i*8 + k)*self.split_dim_size \
                            + _index*self.split_output_dim_size + j
                        _out_index = (i * 8 +
                                      k) * self.split_output_dim_size + j
                        _out_ub[_out_index].set_as(data_ub[_input_index])
            nbust = _get_ceil_int(first_loop * self.split_output_dim_size,
                                  self.data_len_one_block)
            self.tik_instance.data_move(self.dst_gm_list[_index][out_offset],
                                        _out_ub, 0, 1, nbust, 0, 0)

        for _, output_index in enumerate(range(self.num_split // 2)):
            _run_one_output(output_index * 2, self.out_ub)
            _run_one_output(output_index * 2 + 1, self.out_ub_1)
        if self.num_split % 2 == 1:
            _run_one_output(self.num_split - 1, self.out_ub)

    def split_last_dim_with_blocks(self):
        """copy all data from src to des
        """
        # core scedule
        many_copy_num = self.ub_max_data // 2
        self.max_dims = many_copy_num // self.split_output_dim_size
        dims_per_core = _get_ceil_int(self.data_size_first_dim, self.core_num)
        if dims_per_core == 0:
            dims_per_core = 1

        core_used = self.data_size_first_dim // dims_per_core
        if self.data_size_first_dim % dims_per_core != 0:
            core_used = core_used + 1
        tail_dims_core = \
            self.data_size_first_dim - (core_used - 1)*dims_per_core

        self.segment_len = self.max_dims * self.split_output_dim_size
        # for core loop
        with self.tik_instance.for_range(
                0, core_used, block_num=core_used) as _core_index:
            # for copy segment loop
            core_dims_offset = _core_index * dims_per_core
            if tail_dims_core != dims_per_core:
                with self.tik_instance.if_scope(_core_index < (core_used - 1)):
                    self.data_move_with_blocks(dims_per_core, core_dims_offset)

                with self.tik_instance.else_scope():
                    self.data_move_with_blocks(tail_dims_core, core_dims_offset)
            else:
                self.data_move_with_blocks(dims_per_core, core_dims_offset)

    def data_move_with_blocks(self, work_dims, core_offset):
        """copy all data from src to des the last size is 32B align
        """
        segment_loop = \
            work_dims * self.split_output_dim_size // self.segment_len
        segment_tail_len = \
            work_dims * self.split_output_dim_size \
            - segment_loop * self.segment_len
        # copy from gm to ub
        data_ub = _apply_mem(self.tik_instance, self.src_dtype,
                             [self.segment_len], "data_ub")
        data_ub_1 = _apply_mem(self.tik_instance, self.src_dtype,
                               [self.segment_len], "data_ub_1")

        def _run_one_segment(_segment_index, _segment_len):
            offset = core_offset*self.split_dim_size + \
                     _segment_index*self.max_dims*self.split_dim_size
            out_offset = \
                core_offset*self.split_output_dim_size \
                + _segment_index*self.max_dims*self.split_output_dim_size
            len_burst = self.split_output_dim_size // self.data_len_one_block
            n_burst = _get_ceil_int(_segment_len, self.data_len_one_block)
            n_burst = _get_ceil_int(n_burst, len_burst)
            src_stride = _get_ceil_int(
                (self.split_dim_size - self.split_output_dim_size),
                self.data_len_one_block)
            out_n_burst = _get_ceil_int(_segment_len, self.data_len_one_block)

            def _run_one_output(_index):
                if _index % 2 == 0:
                    src_ub = data_ub
                else:
                    src_ub = data_ub_1
                src_offset = offset + _index * self.split_output_dim_size
                self.tik_instance.data_move(src_ub, self.src_gm[src_offset], 0,
                                            n_burst, len_burst, src_stride, 0)
                self.tik_instance.data_move(
                    self.dst_gm_list[_index][out_offset], src_ub, 0, 1,
                    out_n_burst, 0, 0)

            for _, i in enumerate(range(self.num_split // 2)):
                _run_one_output(i * 2)
                _run_one_output(i * 2 + 1)
            if self.num_split % 2 == 1:
                _run_one_output(self.num_split - 1)

        with self.tik_instance.for_range(0, segment_loop) as _segment_loop:
            _run_one_segment(_segment_loop, self.segment_len)

        if segment_tail_len != 0:
            _run_one_segment(segment_loop, segment_tail_len)

    def run_tik(self, kernel_name):
        """cal tik_instance according
        """
        if self.split_output_dim_size % self.data_len_one_block == 0:
            self.split_last_dim_with_blocks()
        else:
            self.split_last_dim_less_block()

        self.tik_instance.BuildCCE(
            kernel_name=kernel_name,
            inputs=[self.src_gm],
            outputs=self.dst_gm_list)
        return self.tik_instance


def check_whether_lastdim(shape, split_dim):
    """check whether the shape and axis is split_d by last dim
    """

    if len(shape) == 1 or split_dim != len(shape) - 1:
        return False

    return True


def check_whether_equal_split(size_splits):
    """check whether split_v == split_d
    """
    size_set = list(set(size_splits))
    if len(size_set) == 1:
        return True

    return False


def check_use_last_dim_branch(shape,
                              dtype,
                              split_dim,
                              num_split,
                              size_splits=None):
    """check whether use new tik branch for last dim tp split_d
    """
    # check whether split_d by last dim
    is_last_dim = check_whether_lastdim(shape, split_dim)

    # check whether in support_dtype
    support_dtype = ("float16", "float32")
    is_dtype_support = dtype in support_dtype

    # check whether the value in size_splits must be equal
    is_split = check_whether_equal_split(size_splits)

    # check the size in new branch condition
    split_l = SplitLastDim(shape, dtype, split_dim, num_split, size_splits)
    half_ub = split_l.ub_max_data // 2
    out_split_size = shape[split_dim] // num_split
    if dtype in ("float16",):
        data_len_one_block = 16
    else:
        data_len_one_block = 8
    is_shape_support = ((out_split_size % data_len_one_block == 0 and
                         out_split_size < half_ub and shape[split_dim] //
                         data_len_one_block < 65535) or
                        out_split_size < data_len_one_block)

    return is_shape_support and is_dtype_support and is_split and is_last_dim


# pylint: disable=locally-disabled,unused-argument,too-many-arguments
def split_last_dim(shape, dtype, split_dim, num_split, size_splits,
                   kernel_name):
    """Split a tensor into len(size_splits) tensors along last dimension.

    Parameters
    ----------
    shape: list or tuple
        the shape of input tensor.
    dtype: str
        the dtype of input tensor.
    split_dim: int
        the dimension along which to split_d.
    num_split: int
        used to specify the number of outputs.
    size_splits: list or tuple
        a Python list containing the sizes of each output tensor
        along `split_dim`.
    kernel_name: str
        cce kernel name.

    Returns
    -------
    None.
    """
    res = SplitLastDim(shape, dtype, split_dim, num_split, size_splits)

    return res.run_tik(kernel_name)


class SplitWith5HD:
    """Function: use to finish ConcatLastDim main functions
    """
    def __init__(self, input_value, output_data,
                 split_dim, num_split, kernel_name):
        """init SplitWith5HD parameters
        """
        self.data_dtype = input_value.get("dtype").lower()
        self.src_shape = input_value.get("shape")
        self.src_ori_shape = input_value.get("ori_shape")
        self.format = input_value.get("format")
        self.ori_format = input_value.get("ori_format")
        self.output_data = output_data
        self.src_size = util.check_tensor_shape_size(list(self.src_shape))
        self.dst_size = self.src_size // num_split

        self.split_dim = split_dim
        self.num_split = num_split
        self.kernel_name = kernel_name
        self.split_dim_size = self.src_shape[self.split_dim]
        self.split_output_dim_size = \
            self.src_shape[self.split_dim] // self.num_split

        self.data_size_first_dim = self.src_size // self.split_dim_size
        self.output_size = \
            self.split_output_dim_size * self.data_size_first_dim
        # get dtype size, float16 size = 2 byte   / float32 size = 4 byte
        self.dtype_size = \
            tbe_platform.cce_intrin.get_bit_len(self.data_dtype) // 8
        # get one block data size, block align len
        # the len in one block = 16 fp16 and = 8 fp32
        self.data_len_one_block = 32 // self.dtype_size
        self.data_len_one_vector = self.data_len_one_block * 8

        self.gm_out = []
        self.gm_in = None
        self.tik_instance = None
        self.core_num = 0
        self.input_c0 = 16
        self.input_n = 0
        self.input_c = 0
        self.input_h = 0
        self.input_w = 0
        self.src_c1 = 0
        self.des_c1 = 0
        self.last_ori_dim = 0
        self.split_out_dim = 0

    def check_shape_support(self):
        """function for check input can use SplitWith5HD
        """
        if self.ori_format not in ("NCHW", "NHWC"):
            return False, -1

        if len(self.src_ori_shape) != 4:
            return False, -1

        if self.ori_format == "NCHW":
            c_dim_num = 1
        else:
            c_dim_num = 3

        split_c = self.src_ori_shape[c_dim_num]
        split_out_c = split_c // self.num_split
        if split_out_c not in (1, 2, 4, 8):
            return False, -1

        return True, c_dim_num

    def check_op_select(self):
        """function for op_select_format in split
        """
        is_shape_support, c_dim_num = self.check_shape_support()
        if not is_shape_support:
            return False

        if self.split_dim % 4 != c_dim_num:
            return False

        return True

    def check_5hd_vnchw(self):
        """function check_5hd_vnchw to check whether can use SplitWith5HD
        """
        is_shape_support, c_dim_num = self.check_shape_support()
        if not is_shape_support:
            return False

        if self.format not in ("NC1HWC0",):
            return False

        self.input_n, self.src_c1, self.input_h, self.input_w, _ \
            = self.src_shape

        if self.split_dim % 5 != 1 \
                or self.data_dtype not in ("float16", "int16", "uint16"):
            return False

        self.last_ori_dim = self.src_ori_shape[c_dim_num]
        self.split_out_dim = self.last_ori_dim // self.num_split

        return True

    def init_gm_tensor(self):
        """init_gm_tensor and tik_instance
        """
        self.tik_instance = tik.Tik()
        self.core_num = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        for index, _ in enumerate(self.output_data):
            self.gm_out.append(
                self.tik_instance.Tensor(
                    "float16", (self.dst_size,),
                    scope=tik.scope_gm,
                    name="data_gm_out_{}".format(index)))

        self.gm_in = self.tik_instance.Tensor(
            "float16", (self.src_size,),
            scope=tik.scope_gm, name="data_gm_in")

    def do_5hd_split_cut_by_batch(self):
        """function do_5hd_split_cut_by_batch
        """
        self.init_gm_tensor()
        batch_offset = self.input_h*self.input_w*self.input_c0
        batch_per_core = _get_ceil_int(self.input_n, self.core_num)
        core_used = _get_ceil_int(self.input_n, batch_per_core)
        core_tail = \
            self.input_n - (core_used - 1)*batch_per_core

        # define split fuction
        split_fuc = self.do_5hd_split_scedule

        with self.tik_instance.for_range(
                0, core_used, block_num=core_used) as core_idx:
            dst_offset_core = \
                core_idx * batch_per_core * batch_offset
            src_offset_core = dst_offset_core * self.src_c1
            if core_tail == batch_per_core:
                split_fuc(
                    src_offset_core, dst_offset_core,
                    batch_per_core)
            else:
                with self.tik_instance.if_scope(
                        core_idx < (core_used - 1)):
                    split_fuc(
                        src_offset_core, dst_offset_core,
                        batch_per_core)

                with self.tik_instance.else_scope():
                    split_fuc(
                        src_offset_core, dst_offset_core,
                        core_tail)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.gm_in],
            outputs=self.gm_out,
            enable_l2=False)

        return self.tik_instance

    def do_5hd_split_scedule(self, input_offset, output_offset,
                             process_batch):
        """do_5hd_split_scedule
        """
        # get ub size, max fp16 num in half ub
        ub_half_size = \
            int(tbe_platform.CceProductParams().getParams("Unified_Buffer")
                // 2 // 2 - 16) * 2

        inner_sigment = self.input_c0 // self.split_out_dim
        max_transpose_sigment = ub_half_size // TRANSPOSE_SIZE
        max_transpose_sigment = int(max_transpose_sigment // (inner_sigment*3))

        copy_size = \
            self.input_h*self.input_w*self.input_c0
        copy_loop = copy_size // (max_transpose_sigment*TRANSPOSE_SIZE)
        copy_loop_tail = copy_size % (max_transpose_sigment*TRANSPOSE_SIZE)
        tail_transpose_sigment = copy_loop_tail // TRANSPOSE_SIZE
        copy_nchw_tail = \
            _get_ceil_int(copy_loop_tail, TRANSPOSE_SIZE)*TRANSPOSE_SIZE \
            - copy_loop_tail

        input_size = TRANSPOSE_SIZE*max_transpose_sigment
        input_ub_ping = \
            self.tik_instance.Tensor("float16", (input_size*inner_sigment,),
                                     tik.scope_ubuf, "input_ub_ping")
        input_ub_pang = \
            self.tik_instance.Tensor("float16", (input_size*inner_sigment,),
                                     tik.scope_ubuf, "input_ub_pang")
        vnchw_ub_0 = \
            self.tik_instance.Tensor("float16",
                                     (input_size*inner_sigment,),
                                     tik.scope_ubuf, "vnchw_ub_0")

        tiling_ub_list_0 = [input_ub_ping, input_ub_pang, vnchw_ub_0]
        self.vector_dup_init(vnchw_ub_0, 0, input_size*inner_sigment)

        # define ub list for vnchwconv to ub_vnchw
        _src_addrs_all = [
            input_ub_ping[ONE_BLOCK_FP16_SIZE * x]
            for x in range(ONE_BLOCK_FP16_SIZE)
        ]

        _dst_addrs_all = []
        _tmp_list = [x*1 for x in range(self.split_out_dim)]
        block_input = \
            TRANSPOSE_SIZE*max_transpose_sigment // self.data_len_one_block
        for i in range(inner_sigment - 1):
            _tmp_list = \
                _tmp_list \
                + [x + (i + 1)*block_input for x in range(self.split_out_dim)]
        _dst_addrs_all = [
            vnchw_ub_0[x * ONE_BLOCK_FP16_SIZE]
            for x in _tmp_list
        ]

        def _run_copy_input_and_vnchw(ub_list, gm_input_offset,
                                      run_mov_num, copy_tail):
            copy_len = run_mov_num * TRANSPOSE_SIZE - copy_tail
            nburst = _get_ceil_int(copy_len,
                                   self.data_len_one_block)
            ub_copy_ping, _, _, gm_input = ub_list
            # copy gm to ub
            self.tik_instance.data_move(ub_copy_ping,
                                        gm_input[gm_input_offset],
                                        0, 1, nburst, 0, 0)

            _dst_rep_stride = 16
            _src_rep_stride = 16
            if run_mov_num == 1:
                _dst_rep_stride = 0
                _src_rep_stride = 0

            self.tik_instance.vnchwconv(False, False, _dst_addrs_all,
                                        _src_addrs_all, run_mov_num,
                                        _dst_rep_stride, _src_rep_stride)

        def _run_vnchw_all_to_out(input_index, ub_list, gm_output_offset,
                                  run_mov_num, copy_tail):
            _, ub_copy_pang, ub_vnchw, gm_output = ub_list
            ub_out_offset = \
                (input_index % inner_sigment) * TRANSPOSE_SIZE \
                * max_transpose_sigment

            _dst_rep_stride = 16
            _src_rep_stride = 16
            if run_mov_num == 1:
                _dst_rep_stride = 0
                _src_rep_stride = 0

            _src_addrs = [
                ub_vnchw[ub_out_offset + ONE_BLOCK_FP16_SIZE * x]
                for x in range(ONE_BLOCK_FP16_SIZE)
            ]
            _dst_addrs = [
                ub_copy_pang[ub_out_offset + ONE_BLOCK_FP16_SIZE * x]
                for x in range(ONE_BLOCK_FP16_SIZE)
            ]

            self.tik_instance.vnchwconv(False, False, _dst_addrs,
                                        _src_addrs, run_mov_num,
                                        _dst_rep_stride, _src_rep_stride)

            copy_len = \
                (run_mov_num * TRANSPOSE_SIZE - copy_tail)
            nburst = _get_ceil_int(copy_len,
                                   ONE_BLOCK_FP16_SIZE)

            # copy ub to gm
            self.tik_instance.data_move(gm_output[gm_output_offset],
                                        ub_copy_pang[ub_out_offset],
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
            c1_loop = _get_ceil_int(int(self.num_split), int(inner_sigment))
            for c1_idx in range(c1_loop):
                inner_loop = inner_sigment
                _run_copy_input_and_vnchw(
                    tiling_ub_list + [self.gm_in],
                    src_offset
                    + c1_idx*self.input_h*self.input_w*self.input_c0,
                    run_mov_num, copy_tail)

                for i_idx in range(inner_loop):
                    input_idx = c1_idx*inner_sigment + i_idx
                    if input_idx >= self.num_split:
                        break
                    _run_vnchw_all_to_out(
                        input_idx,
                        tiling_ub_list + [self.gm_out[input_idx]],
                        dst_offset,
                        run_mov_num, copy_tail
                    )

        with self.tik_instance.for_range(0, process_batch) as _batch_idx:
            _batch_input_offset = \
                input_offset + _batch_idx*copy_size*self.src_c1
            _batch_output_offset = \
                output_offset + _batch_idx*copy_size
            batch_offset_list = [_batch_input_offset, _batch_output_offset]
            # copy input to
            if copy_loop != 0:
                with self.tik_instance.for_range(
                        0, copy_loop) as loop_idx:
                    _idx = loop_idx
                    _run_one_loop(tiling_ub_list_0, batch_offset_list,
                                  _idx * TRANSPOSE_SIZE * max_transpose_sigment,
                                  max_transpose_sigment)

            if tail_transpose_sigment != 0:
                _idx = copy_loop
                _run_one_loop(tiling_ub_list_0, batch_offset_list,
                              _idx * TRANSPOSE_SIZE * max_transpose_sigment,
                              tail_transpose_sigment)

            if copy_nchw_tail != 0:
                offset = \
                    copy_loop*max_transpose_sigment + tail_transpose_sigment
                _run_one_loop(tiling_ub_list_0, batch_offset_list,
                              offset * TRANSPOSE_SIZE,
                              1, copy_nchw_tail)

    def vector_dup_init(self, vector_dup_ub, ub_offset, dup_len):
        """vector_dup_init vector_dup ub to 0
        """
        dup_block = _get_ceil_int(dup_len, self.data_len_one_block)
        dup_repeat = dup_block // 8
        dup_tail = dup_block % 8
        max_repeat_num = 255
        repeat_loop = dup_repeat // max_repeat_num
        repeat_tail = dup_repeat % max_repeat_num

        def _vector_dup(_offset, repeat, tail_block=0):
            if tail_block == 0:
                self.tik_instance.vector_dup(ONE_BLOCK_FP16_SIZE*8,
                                             vector_dup_ub[_offset], 0.0,
                                             repeat,
                                             1, 8)
            else:
                self.tik_instance.vector_dup(ONE_BLOCK_FP16_SIZE*tail_block,
                                             vector_dup_ub[_offset], 0.0,
                                             1,
                                             1, 8)

        for idx, _ in enumerate(range(repeat_loop)):
            vec_offset = ub_offset + idx*max_repeat_num*8*ONE_BLOCK_FP16_SIZE
            _vector_dup(vec_offset, max_repeat_num)

        if repeat_tail != 0:
            vec_offset = \
                ub_offset + repeat_loop*max_repeat_num*8*ONE_BLOCK_FP16_SIZE
            _vector_dup(vec_offset, repeat_tail)

        if dup_tail != 0:
            vec_offset = \
                ub_offset \
                + (repeat_loop*max_repeat_num + dup_tail)*8*ONE_BLOCK_FP16_SIZE
            _vector_dup(vec_offset, 1, dup_tail)

