#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

ifmr
"""


from math import ceil
import functools
from te import tik
from te import platform as cce
from te import tvm
from te.tvm import make as _make
from te.tvm import expr as _expr
from te.tvm import stmt as _stmt
from te.platform import cce_params
from te.platform.cce_runtime import PIPELINES
from te.utils.op_utils import check_op_params
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import REQUIRED_ATTR_FLOAT
from te.utils.op_utils import REQUIRED_ATTR_LIST_FLOAT
from te.utils.op_utils import REQUIRED_ATTR_BOOL
from te.utils.op_utils import KERNEL_NAME


SCALAR_MAX_FP16 = (2 ** 16 - 1)
SHAPE_SIZE_LIMIT = 2 ** 31
MAX_BINS = 8192
MAX_STEPS = 4096
ESP = 1.192092896e-07


class Barrier:
    """this class should be part of tik."""
    def emit(self, tik_inst, stmt):
        """Emit a statement to the end of current scope.

        Parameters
        ----------
        stmt : Stmt or callable.
        The statement to be emitted or callable that build stmt given
        body.
        """
        if isinstance(stmt, _expr.Call):
            stmt = _make.Evaluate(stmt)
        assert isinstance(stmt, _stmt.Stmt) or callable(stmt)
        tik_inst._seq_stack[-1].append(stmt)

    def __init__(self, tik_instance, workspace, block_num, block_id):
        """soft synchronization initialize"""
        self.block_id = block_id
        self.block_num = block_num
        self.int32_byte_size = 4
        self.tik_instance = tik_instance
        self.gm_workspace = workspace
        self.seq = self.tik_instance.Scalar('int64', init_value=1)
        self.sync_ub = tik_instance.Tensor(
            'int64', (self.int32_byte_size * self.block_num,), tik.scope_ubuf, 'barrier_ub')
        zero = self.tik_instance.Scalar('int64')
        zero.set_as(0)
        self.sync_ub[self.block_id * self.int32_byte_size].set_as(zero)
        self.sync_ub[self.block_id * self.int32_byte_size + 1].set_as(zero)
        self.int64_max = 0x7FFFFFFFFFFFFFFF
        self.loop_bound = self.tik_instance.Scalar('int64')
        self.sync_ub[self.block_id * self.int32_byte_size].set_as(zero)
        self.sync_ub[self.block_id * self.int32_byte_size + 1].set_as(zero)
        self.tik_instance.tensor_mov(
            self.gm_workspace[self.block_id * self.int32_byte_size],
            self.sync_ub[self.block_id * self.int32_byte_size], '', 1, 1, 0, 0)
        self.emit(
            self.tik_instance, tvm.call_extern(
                'int32', 'pipe_barrier', tvm.call_pure_intrin('int32', 'tvm_cce_string_print', 'PIPE_MTE3')))

    def sync(self):
        """ barrier sync func"""
        # add pipe_barrier MTE3 here manually
        self.emit(
            self.tik_instance, tvm.call_extern(
                'int32', 'pipe_barrier', tvm.call_pure_intrin('int32', 'tvm_cce_string_print', 'PIPE_MTE3')))
        self.sync_ub[self.block_id * self.int32_byte_size + self.seq % 2].set_as(self.seq)
        self.tik_instance.tensor_mov(
            self.gm_workspace[self.block_id * self.int32_byte_size],
            self.sync_ub[self.block_id * self.int32_byte_size], '', 1, 1, 0, 0)
        self.loop_bound.set_as(self.int64_max)
        pipe_line_dict = dict(zip(PIPELINES.values(), PIPELINES.keys()))
        with self.tik_instance.new_scope():
            self.tik_instance.scope_attr(cce_params.CCE_AXIS, 'group_coproc_scope', pipe_line_dict['PIPE_ALL'])
            with self.tik_instance.for_range(0, self.loop_bound, dtype='int64'):
                self.tik_instance.tensor_mov(self.sync_ub, self.gm_workspace, '', 1, self.block_num, 0, 0)
                # insert set_flag wait by manual
                self.emit(self.tik_instance, tvm.call_intrin(
                    'int32', 'tvm_cce_string_print', 'set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0)'))
                synced = self.tik_instance.Scalar('int64', init_value=1)
                # insert wait_flag by manual. IR above all is four.
                self.emit(self.tik_instance, tvm.call_intrin(
                    'int32', 'tvm_cce_string_print', 'wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0)'))

                with self.tik_instance.for_range(0, self.block_num, dtype='int64') as core_id:
                    with self.tik_instance.if_scope(
                        self.sync_ub[core_id * self.int32_byte_size + self.seq % 2] != self.seq):
                        synced.set_as(0)

                with self.tik_instance.if_scope(synced == 1):
                    self.loop_bound.set_as(0)

        self.seq.set_as(self.seq + 1)


class Reconstruction():
    """IFMR: input feature map reconstruction"""
    def __init__(
        self,
        input_data,
        input_min,
        input_max,
        input_cumsum,
        output_scale,
        output_offset,
        min_percentile,
        max_percentile,
        search_range,
        search_step,
        with_offset,
        kernel_name):

        # The name with suffix "num" represent the true value of
        # variable. The name with suffix "size" represent the memory
        # space of variable. The name with suffix "repeat" represent the
        # number of repeat time when processing this variable.
        self.tik_instance = tik.Tik(tik.Dprofile('v100', 'cloud'))
        self.aicore_num = 30
        self.unified_buffer_size = tik.Dprofile('v100', 'cloud').get_unified_buffer_size()

        self.data_dtype = input_data.get('dtype')
        data_shape = input_data.get('shape')
        self.data_num = functools.reduce(lambda x, y: x * y, data_shape)
        if self.data_num > SHAPE_SIZE_LIMIT:
            raise ValueError('Excessive amount of "input_data"(more than 2^31)!')

        if len(input_min.get('shape')) != 1 or input_min.get('shape')[0] != 1:
            raise ValueError('The shape of "input_min" must be "[1]"!')
        if len(input_max.get('shape')) != 1 or input_max.get('shape')[0] != 1:
            raise ValueError('The shape of "input_max" must be "[1]"!')

        cumsum_shape = input_cumsum.get('shape')
        if len(cumsum_shape) != 1:
            raise ValueError('The shape of "input_cumsum" must be "(x,)"!')
        self.cumsum_num = cumsum_shape[0]
        if cumsum_shape[0] > MAX_BINS:
            raise ValueError('Excessive amount of "input_cumsum"(more than 8192)!')

        # input&output global memory
        self.input_data = self.tik_instance.Tensor(self.data_dtype, data_shape, tik.scope_gm, 'input_data')
        self.input_max = self.tik_instance.Tensor(self.data_dtype, (1,), tik.scope_gm, 'input_max')
        self.input_min = self.tik_instance.Tensor(self.data_dtype, (1,), tik.scope_gm, 'input_min')
        self.input_cumsum = self.tik_instance.Tensor('int32', cumsum_shape, tik.scope_gm, 'input_cumsum')
        self.output_scale = self.tik_instance.Tensor('float32', (1,), tik.scope_gm, 'output_scale')
        self.output_offset = self.tik_instance.Tensor('float32', (1,), tik.scope_gm, 'output_offset')

        # IFMR parameter initialize
        self.max_percentile = max_percentile
        self.min_percentile = 1 - min_percentile
        if search_range[1] <= search_range[0]:
            raise ValueError('The "search_range" must be monotonically increasing!')
        self.search_range = search_range
        if search_step <= 0:
            raise ValueError('The "search_step" must be positive!')
        self.search_step = search_step
        self.with_offset = with_offset

        self.kernel_name = kernel_name

        # uniform use float32 in processing
        self.data_byte_size = 4
        self.block_byte_size = 32
        self.data_each_block = self.block_byte_size // self.data_byte_size
        self.vector_mask_max = 8 * self.data_each_block

        # search space
        self.steps_num = int((search_range[1] - search_range[0]) // search_step + 1)
        if self.steps_num > 4096:
            raise ValueError('step size should be equal or less than 4096')

        self.steps_size = self.steps_num // self.vector_mask_max * self.vector_mask_max
        if self.steps_size < self.steps_num:
            self.steps_size = self.steps_size + self.vector_mask_max
        self.step_repeat = self.steps_size // self.vector_mask_max

        self.barrier_workspace = self.tik_instance.Tensor(
            'int64', (self.data_byte_size * self.aicore_num,), tik.scope_gm, 'barrier_workspace', is_workspace=True,
            is_atomic_add=True)

        self.input_data_bytes_size = cce.cce_intrin.get_bit_len(self.data_dtype) // 8
        self.input_data_each_block = self.block_byte_size // self.input_data_bytes_size
        self.data_num_each_core = self.data_num // self.data_each_block // self.aicore_num * self.data_each_block
        self.data_num_last_core = self.data_num % self.data_each_block + \
            self.data_num // self.data_each_block % self.aicore_num * self.data_each_block
        self.ub_tensor_size = (
                (self.unified_buffer_size - 2 * self.steps_size * self.data_byte_size) //
                self.data_byte_size // 4 // self.data_each_block * self.data_each_block)

        self.loss_each_core = (
            self.steps_num + self.data_each_block - 1) // self.data_each_block * self.data_each_block
        self.loss_workspace = self.tik_instance.Tensor(
            'float32', (self.loss_each_core * self.aicore_num,), tik.scope_gm, 'loss_workspace',  is_workspace=True,
            is_atomic_add=True)

    def _compute_scale_offset(self):
        """Calculate the candidate scale and offset."""
        cumsum_size = self.cumsum_num // self.vector_mask_max * self.vector_mask_max
        if cumsum_size < self.cumsum_num:
            cumsum_size = cumsum_size + self.vector_mask_max
        cumsum_repeat = cumsum_size // self.vector_mask_max
        cumsum_redundance = cumsum_size - self.cumsum_num

        move_burst = self.cumsum_num // self.data_each_block
        if move_burst * self.data_each_block < self.cumsum_num:
            move_burst = move_burst + 1

        with self.tik_instance.new_stmt_scope():
            cumsum_int = self.tik_instance.Tensor('int32', (cumsum_size,), tik.scope_ubuf, 'cumsum_int')
            self.tik_instance.vec_dup(self.vector_mask_max, cumsum_int, 0, cumsum_repeat, 8)
            self.tik_instance.data_move(cumsum_int, self.input_cumsum, 0, 1, move_burst, 0, 0)

            cumsum_float = self.tik_instance.Tensor('float32', (cumsum_size,), tik.scope_ubuf, 'cumsum_float')
            self.tik_instance.vec_conv(self.vector_mask_max, 'none', cumsum_float, cumsum_int, cumsum_repeat, 8, 8)

            cumsum_max = self.tik_instance.Tensor('float32', (self.vector_mask_max,), tik.scope_ubuf, 'cumsum_max')
            self.tik_instance.vec_dup(self.vector_mask_max, cumsum_max, self.data_num, 1, 8)

            cdf = self.tik_instance.Tensor('float32', (cumsum_size,), tik.scope_ubuf, 'cdf')
            self.tik_instance.vdiv(
                self.vector_mask_max, cdf, cumsum_float, cumsum_max, cumsum_repeat, 1, 1, 1, 8, 8, 0)

            max_tensor = self.tik_instance.Tensor('float32', (self.vector_mask_max,), tik.scope_ubuf, 'max_tensor')
            min_tensor = self.tik_instance.Tensor('float32', (self.vector_mask_max,), tik.scope_ubuf, 'min_tensor')
            self.tik_instance.vec_dup(self.vector_mask_max, max_tensor, self.max_percentile, 1, 8)
            self.tik_instance.vec_dup(self.vector_mask_max, min_tensor, self.min_percentile, 1, 8)

            # 0&1 vector for comparing, just need length of "vector_mask_max"
            zeros = self.tik_instance.Tensor('float32', (self.vector_mask_max,), tik.scope_ubuf, 'zeros')
            ones = self.tik_instance.Tensor('float32', (self.vector_mask_max,), tik.scope_ubuf, 'ones')
            self.tik_instance.vec_dup(self.vector_mask_max, zeros, 0, 1, 8)
            self.tik_instance.vec_dup(self.vector_mask_max, ones, 1, 1, 8)

            # For convenience,  the length of the comparison results is
            # selected from only four values, respectively 2048, 4096,
            # 6144 and 8192
            compare_repeat = (
                self.cumsum_num + self.vector_mask_max * self.data_byte_size * 8 - 1) // (
                    self.vector_mask_max * self.data_byte_size * 8)
            compare_size = compare_repeat * self.vector_mask_max
            result_max = self.tik_instance.Tensor(
                'uint32', (compare_size + self.data_each_block,), tik.scope_ubuf, 'result_max')
            result_min = self.tik_instance.Tensor(
                'uint32', (compare_size + self.data_each_block,), tik.scope_ubuf, 'result_min')
            self.tik_instance.vec_dup(self.vector_mask_max, result_max, 0, compare_repeat, 8)
            self.tik_instance.vec_dup(self.vector_mask_max, result_min, 0, compare_repeat, 8)

            self.tik_instance.vec_cmpv_gt(result_max, max_tensor, cdf, cumsum_repeat, 0, 8)
            self.tik_instance.vec_cmpv_gt(result_min, min_tensor, cdf, cumsum_repeat, 0, 8)

            stat_max = self.tik_instance.Tensor('float32', (cumsum_size,), tik.scope_ubuf, 'stat_max')
            stat_min = self.tik_instance.Tensor('float32', (cumsum_size,), tik.scope_ubuf, 'stat_min')
            self.tik_instance.vec_dup(self.vector_mask_max, stat_max, 0, cumsum_repeat, 8)
            self.tik_instance.vec_dup(self.vector_mask_max, stat_min, 0, cumsum_repeat, 8)

            select = self.tik_instance.Tensor('uint32', (self.data_each_block,), tik.scope_ubuf, 'select')
            with self.tik_instance.for_range(0, cumsum_repeat) as i:
                select[0].set_as(result_max[i * 2])
                select[1].set_as(result_max[i * 2 + 1])
                self.tik_instance.vec_sel(
                    self.vector_mask_max, 0, stat_max[i * self.vector_mask_max], select, ones, zeros, 1)
                select[0].set_as(result_min[i * 2])
                select[1].set_as(result_min[i * 2 + 1])
                self.tik_instance.vec_sel(
                    self.vector_mask_max, 0, stat_min[i * self.vector_mask_max], select, ones, zeros, 1)

            max_index = self.tik_instance.Tensor('float32', (self.data_each_block,), tik.scope_ubuf, 'max_index')
            min_index = self.tik_instance.Tensor('float32', (self.data_each_block,), tik.scope_ubuf, 'min_index')
            work_tensor = self.tik_instance.Tensor('float32', (cumsum_repeat,), tik.scope_ubuf, 'work_tensor')
            self.tik_instance.vec_reduce_add(self.vector_mask_max, max_index, stat_max, work_tensor, cumsum_repeat, 8)
            self.tik_instance.vec_reduce_add(self.vector_mask_max, min_index, stat_min, work_tensor, cumsum_repeat, 8)

            cumsum_redundance_tensor = self.tik_instance.Tensor(
                'float32', (self.data_each_block,), tik.scope_ubuf, 'cumsum_redundance_tensor')
            self.tik_instance.vec_dup(1, cumsum_redundance_tensor, cumsum_redundance, 1, 8)
            self.tik_instance.vec_sub(1, max_index, max_index, cumsum_redundance_tensor, 1, 8, 8, 8)
            self.tik_instance.vec_sub(1, min_index, min_index, cumsum_redundance_tensor, 1, 8, 8, 8)

            data_max = self.tik_instance.Tensor('float32', (self.data_each_block,), tik.scope_ubuf, 'data_max')
            data_min = self.tik_instance.Tensor('float32', (self.data_each_block,), tik.scope_ubuf, 'data_min')
            if self.data_dtype == 'float32':
                self.tik_instance.data_move(data_max, self.input_max, 0, 1, 1, 0, 0)
                self.tik_instance.data_move(data_min, self.input_min, 0, 1, 1, 0, 0)
            else:
                input_data_max = self.tik_instance.Tensor(
                    self.data_dtype, (self.input_data_each_block,), tik.scope_ubuf, 'input_data_max')
                input_data_min = self.tik_instance.Tensor(
                    self.data_dtype, (self.input_data_each_block,), tik.scope_ubuf, 'input_data_min')
                self.tik_instance.data_move(input_data_max, self.input_max, 0, 1, 1, 0, 0)
                self.tik_instance.data_move(input_data_min, self.input_min, 0, 1, 1, 0, 0)
                self.tik_instance.vec_conv(1, 'none', data_max, input_data_max, 1, 8, 8)
                self.tik_instance.vec_conv(1, 'none', data_min, input_data_min, 1, 8, 8)

            max_init = self.tik_instance.Tensor('float32', (self.data_each_block,), tik.scope_ubuf, 'max_init')
            min_init = self.tik_instance.Tensor('float32', (self.data_each_block,), tik.scope_ubuf, 'min_init')
            cumsum_num_tensor = self.tik_instance.Tensor(
                'float32', (self.data_each_block,), tik.scope_ubuf, 'cumsum_num')
            data_range = self.tik_instance.Tensor('float32', (self.data_each_block,), tik.scope_ubuf, 'data_range')
            self.tik_instance.vec_dup(1, cumsum_num_tensor, self.cumsum_num, 1, 8)
            self.tik_instance.vdiv(1, max_init, max_index, cumsum_num_tensor, 1, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vdiv(1, min_init, min_index, cumsum_num_tensor, 1, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vec_sub(1, data_range, data_max, data_min, 1, 8, 8, 8)
            self.tik_instance.vec_mul(1, max_init, max_init, data_range, 1, 8, 8, 8)
            self.tik_instance.vec_mul(1, min_init, min_init, data_range, 1, 8, 8, 8)
            self.tik_instance.vec_add(1, max_init, max_init, data_min, 1, 8, 8, 8)
            self.tik_instance.vec_add(1, min_init, min_init, data_min, 1, 8, 8, 8)

            if self.with_offset:
                self.tik_instance.vec_max(1, max_init, max_init, zeros, 1, 8, 8, 8)
                self.tik_instance.vec_min(1, min_init, min_init, zeros, 1, 8, 8, 8)
                min_value = self.tik_instance.Scalar('float32', 'min_value', min_init[0])
                min_list = self.tik_instance.Tensor('float32', (self.steps_size,), tik.scope_ubuf, 'min_list')
                self.tik_instance.vec_dup(self.vector_mask_max, min_list, min_value, self.step_repeat, 8)
            else:
                self.tik_instance.vec_abs(1, max_init, max_init, 1, 8, 8)
                self.tik_instance.vec_abs(1, min_init, min_init, 1, 8, 8)
                self.tik_instance.vec_max(1, max_init, max_init, min_init, 1, 8, 8, 8)

            max_list = self.tik_instance.Tensor('float32', (self.steps_size,), tik.scope_ubuf, 'max_list')
            self.tik_instance.vec_dup(self.vector_mask_max, max_list, 0, self.step_repeat, 8)
            search_step = self.tik_instance.Tensor('float32', (self.data_each_block,), tik.scope_ubuf, 'search_step')
            self.tik_instance.vec_dup(1, search_step, self.search_step, 1, 8)
            search_min = self.tik_instance.Tensor('float32', (self.data_each_block,), tik.scope_ubuf, 'search_min')
            self.tik_instance.vec_dup(1, search_min, self.search_range[0], 1, 8)
            step_length = self.tik_instance.Tensor('float32', (self.data_each_block,), tik.scope_ubuf, 'step_length')
            with self.tik_instance.for_range(0, self.steps_num) as i:
                self.tik_instance.vec_dup(1, step_length, i, 1, 8)
                self.tik_instance.vec_mul(1, step_length, step_length, search_step, 1, 8, 8, 8)
                self.tik_instance.vec_add(1, step_length, step_length, search_min, 1, 8, 8, 8)
                self.tik_instance.vec_mul(1, step_length, step_length, max_init, 1, 8, 8, 8)
                max_list[i].set_as(step_length[0])

            if self.with_offset:
                quant_step = self.tik_instance.Tensor('float32', (self.vector_mask_max,), tik.scope_ubuf, 'quant_step')
                self.tik_instance.vec_dup(self.vector_mask_max, quant_step, 255, 1, 8)
                self.tik_instance.vec_sub(
                    self.vector_mask_max, self.scale, max_list, min_list, self.step_repeat, 8, 8, 8)
                self.tik_instance.vdiv(
                    self.vector_mask_max, self.scale, self.scale, quant_step, self.step_repeat, 1, 1, 1, 8, 8, 0)
                round_offset = self.tik_instance.Tensor('int32', (self.steps_size,), tik.scope_ubuf, 'round_offset')
                self.tik_instance.vdiv(
                    self.vector_mask_max, self.offset, min_list, self.scale, self.step_repeat, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vec_conv(
                    self.vector_mask_max, 'round', round_offset, self.offset, self.step_repeat, 8, 8)
                self.tik_instance.vec_conv(self.vector_mask_max, '', self.offset, round_offset, self.step_repeat, 8, 8)
                half_quant_step = self.tik_instance.Tensor(
                    'float32', (self.vector_mask_max,), tik.scope_ubuf, 'half_quant_step')
                self.tik_instance.vec_dup(self.vector_mask_max, half_quant_step, 128, 1, 8)
                self.tik_instance.vec_add(
                    self.vector_mask_max, self.offset, self.offset, half_quant_step, self.step_repeat, 8, 8, 0)
                negative = self.tik_instance.Tensor('float32', (self.vector_mask_max,), tik.scope_ubuf, 'negative')
                self.tik_instance.vec_dup(self.vector_mask_max, negative, -1, 1, 8)
                self.tik_instance.vec_mul(
                    self.vector_mask_max, self.offset, self.offset, negative, self.step_repeat, 8, 8, 0)
            else:
                quant_step = self.tik_instance.Tensor('float32', (self.vector_mask_max,), tik.scope_ubuf, 'quant_step')
                self.tik_instance.vec_dup(self.vector_mask_max, quant_step, 127, 1, 8)
                self.tik_instance.vdiv(
                    self.vector_mask_max, self.scale, max_list, quant_step, self.step_repeat, 1, 1, 1, 8, 8, 0)

    def _compute_mse_loss(self, index):
        with self.tik_instance.new_stmt_scope():
            loss_repeat = ceil(self.steps_size / self.vector_mask_max)
            self.loss_ub = self.tik_instance.Tensor('float32', (self.steps_size,), tik.scope_ubuf, 'loss_ub')
            self.tik_instance.vec_dup(self.vector_mask_max, self.loss_ub, 0, loss_repeat, 8)
            self.input_fm_ub = self.tik_instance.Tensor('float32', (self.ub_tensor_size, ), tik.scope_ubuf, 'fm_ub')
            self.input_fm_quant_ub = self.tik_instance.Tensor(
                'float32', (self.ub_tensor_size, ), tik.scope_ubuf, 'fm_quant_ub')
            self.one_loss_ub = self.tik_instance.Tensor(
                'float32', (self.data_each_block,), tik.scope_ubuf, 'one_loss_ub')
            with self.tik_instance.for_range(0, self.steps_num) as search_index:
                self.tik_instance.vec_dup(self.data_each_block, self.one_loss_ub, 0, 1, 8)
                self._compute_mse(index, search_index)
            self._move_loss_out(index)

    def _compute_mse(self, index, search_index):
        with self.tik_instance.if_scope(index == (self.aicore_num - 1)):
            input_data_each_core = self.data_num_last_core + self.data_num_each_core
            move_offset = index * self.data_num_each_core
            loop_time = input_data_each_core // self.ub_tensor_size
            if loop_time > 0:
                with self.tik_instance.for_range(0, loop_time) as loop_index:
                    move_offset += loop_index * self.ub_tensor_size
                    # do vector mse
                    self._mse_compute_each_loop(move_offset, self.ub_tensor_size, search_index)
            move_offset = index * self.data_num_each_core + loop_time * self.ub_tensor_size
            last_num = input_data_each_core % self.ub_tensor_size
            if last_num > 0:
                # do last num mse
                self._mse_compute_each_loop(move_offset, last_num, search_index)

        with self.tik_instance.else_scope():
            move_offset = index * self.data_num_each_core
            loop_time = self.data_num_each_core // self.ub_tensor_size
            if loop_time > 0:
                with self.tik_instance.for_range(0, loop_time) as loop_index:
                    move_offset += loop_index * self.ub_tensor_size
                    # do vector mse
                    self._mse_compute_each_loop(move_offset, self.ub_tensor_size, search_index)
            move_offset = index * self.data_num_each_core + loop_time * self.ub_tensor_size
            last_num = self.data_num_each_core % self.ub_tensor_size
            if last_num > 0:
                # do last num mse
                self._mse_compute_each_loop(move_offset, last_num, search_index)

    def _mse_compute_each_loop(self, move_offset, move_num, search_index):
        # cal each loop move burst
        if self.data_dtype == 'float16':
            # conv fp16 to fp32
            with self.tik_instance.new_stmt_scope():
                self.input_fm_fp16_ub = self.tik_instance.Tensor(
                    'float16', (self.ub_tensor_size,), tik.scope_ubuf, 'input_fm_fp16_ub')
                self._convert_each_loop(move_offset, move_num)
        else:
            burst_len = ceil(move_num / self.data_each_block)
            self.tik_instance.data_move(self.input_fm_ub, self.input_data[move_offset], 0, 1, burst_len, 0, 0)
        mse_loop = move_num // (self.vector_mask_max * 255)
        mse_offset = 0
        if mse_loop > 0:
            with self.tik_instance.for_range(0, mse_loop) as mse_index:
                mse_offset = mse_index * self.vector_mask_max * 255
                # do mse
                repeat_times = 255
                self._ifmr_mse(self.vector_mask_max, repeat_times, mse_offset, search_index)

        repeat_time = (move_num % (self.vector_mask_max * 255) // self.vector_mask_max)
        if repeat_time > 0:
            # do mse
            mse_offset = mse_loop * self.vector_mask_max * 255
            repeat_times = repeat_time
            self._ifmr_mse(self.vector_mask_max, repeat_times, mse_offset, search_index)
        last_num = move_num % self.vector_mask_max

        if last_num > 0:
            # do mse
            mse_offset = mse_loop * self.vector_mask_max * 255 + repeat_time * self.vector_mask_max
            repeat_times = 1
            self._ifmr_mse(last_num, repeat_times, mse_offset, search_index)

    def _convert_each_loop(self, move_offset, move_num):
        fp16_burst_len = ceil(move_num / self.input_data_each_block)
        self.tik_instance.data_move(self.input_fm_fp16_ub, self.input_data[move_offset], 0, 1, fp16_burst_len, 0, 0)
        convert_loop = move_num // (self.vector_mask_max * 255)
        convert_offset = 0
        if convert_loop > 0:
            with self.tik_instance.for_range(0, convert_loop) as convert_index:
                convert_offset = convert_index * self.vector_mask_max * 255
                repeat_times = 255
                self.tik_instance.vec_conv(
                    self.vector_mask_max, '', self.input_fm_ub[convert_offset], self.input_fm_fp16_ub[convert_offset],
                    repeat_times, 8, 4)

        repeat_time = (move_num % (self.vector_mask_max * 255) // self.vector_mask_max)
        if repeat_time > 0:
            convert_offset = convert_loop * self.vector_mask_max * 255
            repeat_times = repeat_time
            self.tik_instance.vec_conv(
                self.vector_mask_max, '', self.input_fm_ub[convert_offset], self.input_fm_fp16_ub[convert_offset],
                repeat_times, 8, 4)
        last_num = move_num % self.vector_mask_max

        if last_num > 0:
            convert_offset = convert_loop * self.vector_mask_max * 255 + repeat_time * self.vector_mask_max
            repeat_times = 1
            self.tik_instance.vec_conv(
                last_num, '', self.input_fm_ub[convert_offset], self.input_fm_fp16_ub[convert_offset], repeat_times, 8,
                4)


    def _ifmr_mse(self, mask_num, repeat_time, mse_offset, search_index):
        with self.tik_instance.new_stmt_scope():
            scale_scalar = self.tik_instance.Scalar('float32')
            scale_scalar.set_as(self.scale[search_index])
            new_scale_scalar = self.tik_instance.Scalar('float32')
            new_scale_scalar.set_as(1.0 / scale_scalar)
            with self.tik_instance.new_stmt_scope():
                self.fm_tmp_ub = self.tik_instance.Tensor('int32', (self.ub_tensor_size,), tik.scope_ubuf, 'fm_tmp_ub')
                self.tik_instance.vec_muls(
                    mask_num, self.input_fm_quant_ub[mse_offset], self.input_fm_ub[mse_offset], new_scale_scalar,
                    repeat_time, 8, 8)
                self.tik_instance.vec_conv(
                    mask_num, 'round', self.fm_tmp_ub, self.input_fm_quant_ub[mse_offset], repeat_time, 8, 8)
                self.tik_instance.vec_conv(
                    mask_num, '', self.input_fm_quant_ub[mse_offset], self.fm_tmp_ub, repeat_time, 8, 8)
            if self.with_offset:
                offset_scalar = self.tik_instance.Scalar('float32')
                offset_scalar.set_as(self.offset[search_index])
                self.tik_instance.vec_adds(
                    mask_num, self.input_fm_quant_ub[mse_offset], self.input_fm_quant_ub[mse_offset], offset_scalar,
                    repeat_time, 8, 8)
            # clip fm_quant max min
            with self.tik_instance.new_stmt_scope():
                self.clip_max_ub = self.tik_instance.Tensor('float32', (mask_num,), tik.scope_ubuf, 'clip_max_ub')
                self.tik_instance.vec_dup(mask_num, self.clip_max_ub, 127, 1, 8)
                self.tik_instance.vec_min(
                    mask_num, self.input_fm_quant_ub[mse_offset], self.input_fm_quant_ub[mse_offset], self.clip_max_ub,
                    repeat_time, 8, 8, 0)
            with self.tik_instance.new_stmt_scope():
                self.clip_min_ub = self.tik_instance.Tensor('float32', (mask_num,), tik.scope_ubuf, 'clip_min_ub')
                self.tik_instance.vec_dup(mask_num, self.clip_min_ub, -128, 1, 8)
                self.tik_instance.vec_max(
                    mask_num, self.input_fm_quant_ub[mse_offset], self.input_fm_quant_ub[mse_offset], self.clip_min_ub,
                    repeat_time, 8, 8, 0)
            if self.with_offset:
                new_offset_scalar = self.tik_instance.Scalar('float32')
                new_offset_scalar.set_as(-1.0 * offset_scalar)
                self.tik_instance.vec_adds(
                    mask_num, self.input_fm_quant_ub[mse_offset], self.input_fm_quant_ub[mse_offset],
                    new_offset_scalar, repeat_time, 8, 8)
            self.tik_instance.vec_muls(
                mask_num, self.input_fm_quant_ub[mse_offset], self.input_fm_quant_ub[mse_offset], scale_scalar,
                repeat_time, 8, 8)
            self.tik_instance.vec_sub(
                mask_num, self.input_fm_quant_ub[mse_offset], self.input_fm_quant_ub[mse_offset],
                self.input_fm_ub[mse_offset], repeat_time, 8, 8, 8)
            self.tik_instance.vec_mul(
                mask_num, self.input_fm_quant_ub[mse_offset], self.input_fm_quant_ub[mse_offset],
                self.input_fm_quant_ub[mse_offset], repeat_time, 8, 8, 8)

        # reduce each loss and move to loss_ub
        with self.tik_instance.new_stmt_scope():
            self.work_ub = self.tik_instance.Tensor('float32', (repeat_time,), tik.scope_ubuf, 'work_ub')
            self.tmp_loss_ub = self.tik_instance.Tensor(
                'float32', (self.data_each_block,), tik.scope_ubuf, 'one_loss_ub')
            self.tik_instance.vec_dup(self.data_each_block, self.tmp_loss_ub, 0, 1, 8)
            self.tik_instance.vec_reduce_add(
                mask_num, self.tmp_loss_ub, self.input_fm_quant_ub[mse_offset], self.work_ub, repeat_time, 8)
            self.tik_instance.vec_add(1, self.one_loss_ub, self.one_loss_ub, self.tmp_loss_ub, 1, 8, 8, 8)
            self.loss_ub[search_index].set_as(self.one_loss_ub[0])

    def _move_loss_out(self, index):
        burst_len = self.loss_each_core // self.data_each_block
        self.tik_instance.tensor_mov(
            self.loss_workspace[index * self.loss_each_core], self.loss_ub, '', 1, burst_len, 0, 0)

    def _reduce_and_output(self):
        # sum of all loss
        total_loss = self.tik_instance.Tensor('float32', (self.loss_each_core,), tik.scope_ubuf, 'total_loss')

        # zero out every item
        repeats = self.steps_num // self.vector_mask_max
        if repeats > 0:
            self.tik_instance.vec_dup(self.vector_mask_max, total_loss, 0, repeats, 8)
        mask = self.steps_num % self.vector_mask_max
        if mask > 0:
            self.tik_instance.vec_dup(mask, total_loss[self.vector_mask_max * repeats], 0, 1, 8)

        # loss of one core
        loss = self.tik_instance.Tensor('float32', (self.loss_each_core,), tik.scope_ubuf, 'loss_sum')
        burst_len = self.loss_each_core // self.data_each_block
        for i in range(0, self.aicore_num):
            self.tik_instance.data_move(loss, self.loss_workspace[self.loss_each_core * i], 0, 1, burst_len, 0, 0)
            if repeats > 0:
                self.tik_instance.vec_add(self.vector_mask_max, total_loss, loss, total_loss, repeats, 8, 8, 8)
            if mask > 0:
                self.tik_instance.vec_add(
                    mask, total_loss[self.vector_mask_max * repeats], loss[self.vector_mask_max * repeats],
                    total_loss[self.vector_mask_max * repeats], 1, 8, 8, 8)

        # do 8-block align
        fp16_each_block = self.block_byte_size // 2
        fp16_8_block = fp16_each_block * 8
        fp16_loss_each_core = (self.loss_each_core + fp16_8_block - 1) // fp16_8_block * fp16_8_block
        total_loss_fp16 = self.tik_instance.Tensor('float16', (fp16_loss_each_core,), tik.scope_ubuf, 'loss_sum')

        fp16_repeat_time = fp16_loss_each_core // fp16_8_block
        if fp16_repeat_time > 0:
            self.tik_instance.vec_dup(fp16_8_block, total_loss_fp16, SCALAR_MAX_FP16, fp16_repeat_time, 8)

        # conv fp32 to fp16
        data_num_scalar = self.tik_instance.Scalar('float32')
        data_num_scalar.set_as(self.data_num)
        data_num = self.tik_instance.Tensor('float32', (self.data_each_block,), tik.scope_ubuf, 'data_num')
        self.tik_instance.vec_dup(self.data_each_block, data_num, data_num_scalar, 1, 8)
        if repeats > 0:
            self.tik_instance.vdiv(self.vector_mask_max, loss, total_loss, data_num, repeats, 1, 1, 0, 8, 8, 0)
            self.tik_instance.vec_conv(self.vector_mask_max, 'none', total_loss_fp16, loss, repeats, 4, 8)
        if mask > 0:
            self.tik_instance.vdiv(
                mask, loss[self.vector_mask_max * repeats], total_loss[self.vector_mask_max * repeats], data_num, 1, 1,
                1, 0, 8, 8, 0)
            self.tik_instance.vec_conv(
                mask, 'none', total_loss_fp16[self.vector_mask_max * repeats],
                loss[self.vector_mask_max * repeats], 1, 4, 8)

        it1_output_count = 2 * fp16_repeat_time
        it2_align_start = ceil(it1_output_count / fp16_each_block) * fp16_each_block
        it2_output_count = ceil(it1_output_count / fp16_8_block) * 2
        it3_align_start = ceil(it2_output_count / fp16_each_block) * fp16_each_block
        it3_output_count = ceil(it2_output_count / fp16_8_block) * 2
        it4_align_start = ceil(it3_output_count / fp16_each_block) * fp16_each_block
        it4_output_count = ceil(it3_output_count / fp16_8_block) * 2
        final_work_tensor_need_size = it2_align_start + it3_align_start + it4_align_start + it4_output_count
        work_tensor_ub = self.tik_instance.Tensor(
            'float16', (final_work_tensor_need_size,), tik.scope_ubuf, 'work_tensor_ub')

        result = self.tik_instance.Tensor('float16', (32,), tik.scope_ubuf, 'result')
        self.tik_instance.vec_reduce_min(
            128, result, total_loss_fp16, work_tensor_ub, fp16_repeat_time, 8, cal_index=True)

        index_scalar = self.tik_instance.Scalar('uint16')
        index_scalar.set_as(result[1])

        minimal_scalar = self.tik_instance.Scalar('float32')
        minimal_scalar.set_as(total_loss[index_scalar])
        minimal = self.tik_instance.Tensor('float32', (self.data_each_block,), tik.scope_ubuf, 'minial_loss')
        self.tik_instance.vec_dup(self.data_each_block, minimal, minimal_scalar, 1, 8)
        if repeats > 0:
            self.tik_instance.vsub(self.vector_mask_max, loss, total_loss, minimal, repeats, 1, 1, 0, 8, 8, 0)
            self.tik_instance.vec_conv(self.vector_mask_max, 'none', total_loss_fp16, loss, repeats, 4, 8)
        if mask > 0:
            self.tik_instance.vsub(
                mask, loss[self.vector_mask_max * repeats], total_loss[self.vector_mask_max * repeats], minimal, 1, 1,
                1, 0, 8, 8, 0)
            self.tik_instance.vec_conv(
                mask, 'none', total_loss_fp16[self.vector_mask_max * repeats],
                loss[self.vector_mask_max * repeats], 1, 4, 8)

        self.tik_instance.vec_reduce_min(
            128, result, total_loss_fp16, work_tensor_ub, fp16_repeat_time, 8, cal_index=True)

        index_scalar.set_as(result[1])

        optimal_scale = self.tik_instance.Scalar('float32')
        optimal_scale.set_as(self.scale[index_scalar])

        scale_tensor = self.tik_instance.Tensor('float32', (8,), tik.scope_ubuf, 'optimal_scale')
        self.tik_instance.vec_dup(1, scale_tensor, optimal_scale, 1, 8)
        # set scale to one if too small
        eps_scalar = self.tik_instance.Scalar('float32')
        eps_scalar.set_as(ESP)
        eps_tensor = self.tik_instance.Tensor('float32', (8,), tik.scope_ubuf, 'esp_scale')
        self.tik_instance.vec_dup(1, eps_tensor, eps_scalar, 1, 8)
        cmpmask = self.tik_instance.vcmp_le(8, scale_tensor, eps_tensor, 1, 1)
        eps_scalar.set_as(1)
        self.tik_instance.vec_dup(1, eps_tensor, eps_scalar, 1, 8)
        self.tik_instance.vsel(8, 0, scale_tensor, cmpmask, eps_tensor, scale_tensor, 1, 1, 1, 1, 8, 8, 8)

        self.tik_instance.tensor_mov(self.output_scale, scale_tensor, '', 1, 1, 0, 0)
        optimal_offset = self.tik_instance.Scalar('float32')
        if self.with_offset:
            optimal_offset.set_as(self.offset[index_scalar])
        else:
            optimal_offset.set_as(0.0)
        offset_tensor = self.tik_instance.Tensor('float32', (8,), tik.scope_ubuf, 'optimal_offset')
        self.tik_instance.vec_dup(1, offset_tensor, optimal_offset, 1, 8)
        if self.with_offset:
            eps_scalar.set_as(-128)
            self.tik_instance.vec_dup(1, eps_tensor, eps_scalar, 1, 8)
            self.tik_instance.vsel(8, 0, offset_tensor, cmpmask, eps_tensor, offset_tensor, 1, 1, 1, 1, 8, 8, 8)
        self.tik_instance.tensor_mov(self.output_offset, offset_tensor, '', 1, 1, 0, 0)

    def ifmr_compute(self):
        with self.tik_instance.for_range(0, self.aicore_num, block_num=self.aicore_num) as index:
            self.barrier = Barrier(self.tik_instance, self.barrier_workspace, self.aicore_num, index)

            self.scale = self.tik_instance.Tensor('float32', (self.steps_size,), tik.scope_ubuf, 'scale')
            if self.with_offset:
                self.offset = self.tik_instance.Tensor('float32', (self.steps_size,), tik.scope_ubuf, 'offset')

            self._compute_scale_offset()
            self._compute_mse_loss(index)

            # define loss ub tensor here
            self.barrier.sync()
            # core 0 do reduce
            with self.tik_instance.if_scope(index == 0):
                self._reduce_and_output()

        self.tik_instance.BuildCCE(
            self.kernel_name, [self.input_data, self.input_min, self.input_max, self.input_cumsum],
            [self.output_scale, self.output_offset])
        return self.tik_instance


@check_op_params(
    REQUIRED_INPUT,
    REQUIRED_INPUT,
    REQUIRED_INPUT,
    REQUIRED_INPUT,
    REQUIRED_OUTPUT,
    REQUIRED_OUTPUT,
    REQUIRED_ATTR_FLOAT,
    REQUIRED_ATTR_FLOAT,
    REQUIRED_ATTR_LIST_FLOAT,
    REQUIRED_ATTR_FLOAT,
    REQUIRED_ATTR_BOOL,
    KERNEL_NAME)
def ifmr(
    data,
    data_min,
    data_max,
    cumsum,
    scale,
    offset,
    min_percentile,
    max_percentile,
    search_range,
    search_step,
    with_offset,
    kernel_name='ifmr'):

    ifmr = Reconstruction(
        data,
        data_min,
        data_max,
        cumsum,
        scale,
        offset,
        min_percentile,
        max_percentile,
        search_range,
        search_step,
        with_offset,
        kernel_name)

    return ifmr.ifmr_compute()
