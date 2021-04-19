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
strided_slice_for_last_dim_mte
"""
from functools import reduce as funct_reduce
from te import tik
from te import platform as tbe_platform


# pylint: disable=too-many-arguments, too-many-locals, too-many-statements
def strided_slice_last_dim_mte(input_shape, dtype, output_shape,
                               begin, kernel_name):
    """
    the main process of only last dim changing
    """

    tik_instance = tik.Tik()
    core_num = \
        tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
    ub_max_size = \
        tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)

    input_size = funct_reduce(lambda x, y: x * y, input_shape[:])
    output_size = funct_reduce(lambda x, y: x * y, output_shape[:])
    # can't change
    start_num = begin[len(begin) - 1]
    if len(output_shape) == 1:
        return False

    # gm_size
    output_data = tik_instance.Tensor(dtype,
                                      (output_size,),
                                      name="output_data",
                                      scope=tik.scope_gm)
    input_data = tik_instance.Tensor(dtype,
                                     (input_size,),
                                     name="input_data",
                                     scope=tik.scope_gm)
    # element count in a block
    if dtype.lower() in ("float32", "int32"):
        block_data_cnt = 8
        byte_cnt = 4
    elif dtype.lower() == "float16":
        block_data_cnt = 16
        byte_cnt = 2
    else:
        block_data_cnt = 32
        byte_cnt = 1

    output_dim_size_except_last = \
        funct_reduce(lambda x, y: x * y, output_shape[:-1])
    input_dim_size_except_last = \
        funct_reduce(lambda x, y: x * y, input_shape[:-1])
    output_core_dim_num = output_dim_size_except_last // core_num
    input_core_dim_num = input_dim_size_except_last // core_num
    input_core_dim_num_left = input_dim_size_except_last % core_num

    input_ub_size = input_core_dim_num * input_shape[-1]

    output_last_dim_part_block_1 = output_shape[-1] // block_data_cnt
    output_last_dim_part_size_1 = \
        output_last_dim_part_block_1 * block_data_cnt
    allowed_repeat_cnt = input_ub_size // \
        (output_last_dim_part_size_1 * block_data_cnt)

    if input_ub_size == 0 or allowed_repeat_cnt == 0:
        return False
    if input_ub_size > ub_max_size // byte_cnt:
        input_ub_size = \
            ub_max_size // byte_cnt // block_data_cnt * block_data_cnt

    # ub_size change
    input_data_ub = tik_instance.Tensor(dtype,
                                        (input_ub_size,),
                                        name="input_data_ub",
                                        scope=tik.scope_ubuf)

    with tik_instance.for_range(0, core_num,
                                block_num=core_num) as core_index:

        def _inner_do_strides(repeat_num, in_offset, out_offset):
            """
            do strides operation
            """

            output_last_dim_part_block_2 = 1
            output_last_dim_part_size_2 = \
                output_last_dim_part_block_2 * block_data_cnt
            max_repeat = 255

            repeat_loop = repeat_num // max_repeat
            repeat_left = repeat_num % max_repeat

            def _inner_process(repeat_loop_index, repeat_cnt):
                """
                strides operation
                """

                def _inner_repeat_process(buf_loop_index, allow_repeat_cnt):
                    """
                    repeat process for ub is not enough
                    """

                    # for part1
                    with tik_instance.for_range(0, block_data_cnt) as lp_index:
                        tik_instance.data_move(
                            input_data_ub[lp_index *
                                          output_last_dim_part_size_1],
                            input_data[(core_index * input_core_dim_num +
                                       lp_index + in_offset +
                                       (repeat_loop_index * max_repeat +
                                       buf_loop_index * allowed_repeat_cnt) *
                                       block_data_cnt) *
                                       input_shape[-1] + start_num],
                            0, allow_repeat_cnt,
                            output_last_dim_part_block_1,
                            input_shape[-1] - output_last_dim_part_block_1,
                            output_last_dim_part_size_1 -
                            output_last_dim_part_block_1)
                    with tik_instance.for_range(0, block_data_cnt) as lp_idx:
                        tik_instance.data_move(
                            output_data[(core_index * output_core_dim_num +
                                         lp_idx + out_offset +
                                         (repeat_loop_index * max_repeat +
                                          buf_loop_index * allowed_repeat_cnt) *
                                         block_data_cnt) * output_shape[-1]],
                            input_data_ub[lp_idx * output_last_dim_part_size_1],
                            0, allow_repeat_cnt,
                            output_last_dim_part_block_1,
                            output_last_dim_part_size_1 -
                            output_last_dim_part_block_1,
                            output_shape[-1] - output_last_dim_part_block_1)

                    # for part2
                    with tik_instance.for_range(0, block_data_cnt)as lp_index1:
                        tik_instance.data_move(
                            input_data_ub[lp_index1 * block_data_cnt],
                            input_data[(core_index * input_core_dim_num +
                                        lp_index1 + in_offset +
                                        (repeat_loop_index * max_repeat +
                                         buf_loop_index * allowed_repeat_cnt) *
                                        block_data_cnt) * input_shape[-1] +
                                       (output_shape[-1] - block_data_cnt) + start_num],
                            0, allow_repeat_cnt,
                            output_last_dim_part_block_2,
                            input_shape[-1] - output_last_dim_part_block_2,
                            output_last_dim_part_size_2 -
                            output_last_dim_part_block_2)
                    with tik_instance.for_range(0, block_data_cnt) as lp_idx1:
                        tik_instance.data_move(
                            output_data[(core_index * output_core_dim_num +
                                         lp_idx1 + out_offset +
                                         (repeat_loop_index * max_repeat +
                                          buf_loop_index * allowed_repeat_cnt)
                                         * block_data_cnt) * output_shape[-1] +
                                        (output_shape[-1] - block_data_cnt)],
                            input_data_ub[lp_idx1 * block_data_cnt],
                            0, allow_repeat_cnt,
                            output_last_dim_part_block_2,
                            output_last_dim_part_size_2 -
                            output_last_dim_part_block_2,
                            output_shape[-1] - output_last_dim_part_block_2)

                allowed_repeat_cnt = \
                    input_ub_size // \
                    (output_last_dim_part_size_1 * block_data_cnt)

                buf_loop = repeat_cnt // allowed_repeat_cnt
                buf_left = repeat_cnt % allowed_repeat_cnt

                if buf_loop:
                    with tik_instance.for_range(0, buf_loop) as buf_lp_idx:
                        _inner_repeat_process(buf_lp_idx, allowed_repeat_cnt)
                if buf_left:
                    _inner_repeat_process(buf_loop, buf_left)

            if repeat_loop:
                with tik_instance.for_range(0, repeat_loop) as repeat_idx:
                    _inner_process(repeat_idx, max_repeat)
            if repeat_left:
                _inner_process(repeat_loop, repeat_left)

        def _inner_do_tail_strides(left_num, in_offset, out_offset):
            """
            do strides operation for left num
            """
            last_dim_block_cnt = \
                (output_shape[-1] + block_data_cnt - 1) // block_data_cnt
            with tik_instance.for_range(0, left_num) as dim_index:
                with tik_instance.if_scope(dim_index < left_num - 1):
                    tik_instance.data_move(
                        input_data_ub,
                        input_data[(core_index * input_core_dim_num +
                                    dim_index + in_offset) * input_shape[-1] +
                                   start_num],
                        0, 1, last_dim_block_cnt, 0, 0)
                    tik_instance.data_move(
                        output_data[(core_index * output_core_dim_num +
                                     dim_index + out_offset) *
                                    output_shape[-1]],
                        input_data_ub,
                        0, 1, last_dim_block_cnt, 0, 0)
                with tik_instance.else_scope():
                    # for last line of last dim
                    tik_instance.data_move(
                        input_data_ub,
                        input_data[(core_index * input_core_dim_num +
                                    dim_index + in_offset) * input_shape[-1] +
                                   start_num],
                        0, 1, output_shape[-1] // block_data_cnt, 0, 0)
                    tik_instance.data_move(
                        output_data[(core_index * output_core_dim_num +
                                     dim_index + out_offset) *
                                    output_shape[-1]],
                        input_data_ub,
                        0, 1, output_shape[-1] // block_data_cnt, 0, 0)

                    # for last block of last line
                    tik_instance.data_move(
                        input_data_ub,
                        input_data[(core_index * input_core_dim_num +
                                    dim_index + in_offset) * input_shape[-1] +
                                   (output_shape[-1] - block_data_cnt) + start_num],
                        0, 1, 1, 0, 0)
                    tik_instance.data_move(
                        output_data[(core_index * output_core_dim_num +
                                     dim_index + out_offset) *
                                    output_shape[-1] +
                                    (output_shape[-1] - block_data_cnt)],
                        input_data_ub,
                        0, 1, 1, 0, 0)

        if input_core_dim_num:
            last_dim_repeat_num = input_core_dim_num // block_data_cnt
            last_dim_left_num = input_core_dim_num % block_data_cnt

            if last_dim_repeat_num:
                _inner_do_strides(last_dim_repeat_num, 0, 0)
            if last_dim_left_num:
                _inner_do_tail_strides(last_dim_left_num,
                                       last_dim_repeat_num * block_data_cnt,
                                       last_dim_repeat_num * block_data_cnt)

        if input_core_dim_num_left:
            with tik_instance.if_scope(core_index < 1):
                last_dim_repeat_num = input_core_dim_num_left // block_data_cnt
                last_dim_left_num = input_core_dim_num_left % block_data_cnt

                if last_dim_repeat_num:
                    _inner_do_strides(last_dim_repeat_num,
                                      input_core_dim_num * core_num,
                                      output_core_dim_num * core_num)
                if last_dim_left_num:
                    _inner_do_tail_strides(
                        last_dim_left_num,
                        input_core_dim_num * core_num +
                        last_dim_repeat_num * block_data_cnt,
                        output_core_dim_num * core_num +
                        last_dim_repeat_num * block_data_cnt)

    tik_instance.BuildCCE(kernel_name,
                          inputs=[input_data], outputs=[output_data])

    return tik_instance
