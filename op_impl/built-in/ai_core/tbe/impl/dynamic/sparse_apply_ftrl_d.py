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
dynamic sparse_apply_ftrl_d
"""
from te import tvm
import te.lang.dynamic
from topi.cce import util
from te import tik
from te import platform as tbe_platform
from te.utils.op_utils import check_op_params
from te.utils.op_utils import check_dtype
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import REQUIRED_ATTR_FLOAT
from te.utils.op_utils import REQUIRED_ATTR_INT
from te.utils.op_utils import OPTION_ATTR_BOOL
from te.utils.op_utils import KERNEL_NAME
from te.utils.error_manager import error_manager_vector


# data type of int32
INT32 = "int32"
# data type of float32
FP32 = "float32"
# one block size takes up 32b
BLOCK_SIZE = 32
# digit 256
DIGIT_256 = 256

UB_1K_SIZE = 1024
# The 4KB space of UB is used to store indices data
UB_INDICES_SIZE = 4*1024
UB_2K_SIZE = 2*1024

# paramsRow is 32B aligned, params is in gm
TILING_MODE_1 = 1
# paramsRow is smaller than 32B
TILING_MODE_2 = 2

TILING_ARG_NUM = 24

# the max size of SHAPE is 2^31 - 1
MAX_SHAPE_SIZE = 2**31 - 1

TYPE_LEN_DICT = {"float32": 4, "int32": 4, "int64": 8}


def ceil_value(value, factor):
    """
    if not divide exactly then plus 1

    Parameters
    ----------
    value:  input number
    factor: factor

    Returns
    -------
    ceil value
    """
    return (value + factor - 1) // factor


# pylint: disable=invalid-name, too-many-locals, too-many-arguments, disable=too-many-statements
# pylint: unused-argument, too-many-instance-attributes, too-many-boolean-expressions
class SparseApplyFtrl():
    """
    Function: class that execute sparse_apply_ftrl
    """
    def input_params_check(self, input_dicts, output_dicts):
        """
        check if the inputs are valid

        Parameters
        ----------
        input_dicts: contains var_dict, accum_dict, linear_dict, grad_dict, indices_dict
        output_dicts: contains var_out_dict, accum_out_dict, linear_out_dict

        Returns
        -------
        None
        """
        var_dtype = input_dicts[0].get("dtype").lower()
        accum_dtype = input_dicts[1].get("dtype").lower()
        linear_dtype = input_dicts[2].get("dtype").lower()
        grad_dtype = input_dicts[3].get("dtype").lower()
        indices_dtype = input_dicts[4].get("dtype").lower()
        var_out_dtype = output_dicts[0].get("dtype").lower()
        var_support_dtype_list = ("float32",)
        indices_support_dtype_list = ("int32", "int64")
        check_dtype(var_dtype, var_support_dtype_list, param_name="var")
        check_dtype(accum_dtype, var_support_dtype_list, param_name="accum")
        check_dtype(linear_dtype, var_support_dtype_list, param_name="linear")
        check_dtype(grad_dtype, var_support_dtype_list, param_name="grad")
        check_dtype(var_out_dtype, var_support_dtype_list, param_name="var_out")
        check_dtype(indices_dtype, indices_support_dtype_list, param_name="indices")

        var_shape = input_dicts[0].get("shape")
        accum_shape = input_dicts[1].get("shape")
        linear_shape = input_dicts[2].get("shape")
        grad_shape = input_dicts[3].get("shape")
        indices_shape = input_dicts[4].get("shape")
        var_out_shape = output_dicts[0].get("shape")

        # check shape
        if len(var_shape) != len(accum_shape):
            error_detail = "the shape of var and accum must be equal"
            error_manager_vector.raise_err_two_input_shpae_invalid(self.kernel_name, "var", "accum", error_detail)
        if len(var_shape) != len(linear_shape):
            error_detail = "the shape of var and accum must be equal"
            error_manager_vector.raise_err_two_input_shpae_invalid(self.kernel_name, "var", "linear", error_detail)
        if len(var_shape) != len(grad_shape):
            error_detail = "the shape of var and linear must be equal"
            error_manager_vector.raise_err_two_input_shpae_invalid(self.kernel_name, "var", "grad", error_detail)
        if len(var_shape) != len(var_out_shape):
            error_detail = "the shape of var and var_out must be equal"
            error_manager_vector.raise_err_two_input_shpae_invalid(self.kernel_name, "var", "var_out", error_detail)
        if len(indices_shape) != 1:
            error_detail = "the shape of indices must be 1"
            error_manager_vector.raise_err_input_shpae_invalid(self.kernel_name, "indices", error_detail)
        if len(var_shape) < 2:
            error_detail = "the dim of var can not be smaller than 2"
            error_manager_vector.raise_err_input_shpae_invalid(self.kernel_name, "var", error_detail)

    def __init__(self, input_dicts, input_attrs, output_dicts, kernel_name):
        """
        constructor of SparseApplyFtrl

        Parameters
        ----------
        input_dicts: contains var_dict, accum_dict, linear_dict, grad_dict, indices_dict
        output_dicts: contains var_out_dict, accum_out_dict, linear_out_dict
        input_attrs: contains attr lr, l1, l2, lr_power
        kernel_name: kernel name, default value is "sparse_apply_ftrl"

        Returns
        -------
        None
        """
        self.kernel_name = kernel_name
        self.input_params_check(input_dicts, output_dicts)

        self.var_dtype = input_dicts[0].get("dtype").lower()
        self.indices_dtype = input_dicts[4].get("dtype").lower()
        self.tiling_dtype = INT32

        profile = tik.Dprofile()
        self.tik_instance = tik.Tik(profile, disable_debug=False)
        self.ub_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
        self.core_num = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)

        self.var_dsize = TYPE_LEN_DICT.get(self.var_dtype)
        self.block_elem = BLOCK_SIZE // self.var_dsize
        self.vector_elem = DIGIT_256 // self.var_dsize
        self.indices_dsize = TYPE_LEN_DICT.get(self.indices_dtype)
        self.block_indices = BLOCK_SIZE // self.indices_dsize
        self.indices_nums_once = UB_INDICES_SIZE // self.indices_dsize
        self.remain_size = self.ub_size - UB_2K_SIZE - UB_INDICES_SIZE
        # The remaining UB space is divided into six parts
        self.one_part_size = self.remain_size // 6
        self.one_part_elem = self.one_part_size // self.var_dsize
        self.one_part_elem = self.one_part_elem // self.vector_elem * self.vector_elem

        self.var_shape = (MAX_SHAPE_SIZE,)
        self.grad_shape = (MAX_SHAPE_SIZE,)
        self.indices_shape = (MAX_SHAPE_SIZE,)
        self.tiling_shape = (TILING_ARG_NUM, )
        self.block_shape = (self.block_elem,)

        self.var = None
        self.accum = None
        self.linear = None
        self.grad = None
        self.indices = None
        self.var_out = None
        self.accum_out = None
        self.linear_out = None

        self.lr_attr = input_attrs[0]
        self.l1_attr = input_attrs[1]
        self.l2_attr = input_attrs[2]
        self.lr_power_attr = input_attrs[3]
        if self.lr_attr <= 0:
            expected_value = "greater than 0"
            real_value = "smaller than or equal to 0"
            error_manager_vector.raise_err_input_value_invalid(self.kernel_name, "lr", expected_value, real_value)
        if self.l1_attr < 0:
            expected_value = "greater than or equal to 0"
            real_value = "smaller than 0"
            error_manager_vector.raise_err_input_value_invalid(self.kernel_name, "l1", expected_value, real_value)
        if self.l2_attr < 0:
            expected_value = "greater than or equal to 0"
            real_value = "smaller than 0"
            error_manager_vector.raise_err_input_value_invalid(self.kernel_name, "l2", expected_value, real_value)
        if self.lr_power_attr > 0:
            expected_value = "smaller than or equal to 0"
            real_value = "greater than 0"
            error_manager_vector.raise_err_input_value_invalid(self.kernel_name, "lr_power",
                                                               expected_value, real_value)

        self.lr_attr_rec = 1.0 / self.lr_attr

        self.tiling_gm = None
        self.tiling_ub = None

        # tiling data
        self.tiling_mode = None
        self.need_core_num = None
        self.tail_process_core = None
        self.indices_num_each_core = None
        self.indices_num_remaining = None
        self.indices_loop_num = None
        self.indices_nums_last = None
        self.var_row_elem = None

        self.var_rows = None
        self.indices_step = None
        self.num_multi_rows = None

        self.partial_factor = None
        self.elems_per_core = None
        self.elems_last_core = None

        self.var_cur_row = None
        self.var_row_repeat = None
        self.core_rows_start_index = None
        self.core_rows_end_index = None
        self.cached_rows_start_index = None

        self.grad_cur_row = None

    def get_tiling_args(self):
        """
        get runtime params from tiling

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.tiling_mode = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="tiling_mode")
        self.tiling_mode.set_as(self.tiling_ub[0])
        self.need_core_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="need_core_num")
        self.need_core_num.set_as(self.tiling_ub[1])
        self.tail_process_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="tail_process_core")
        self.tail_process_core.set_as(self.tiling_ub[2])
        self.indices_num_each_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_num_each_core")
        self.indices_num_each_core.set_as(self.tiling_ub[3])
        self.indices_num_remaining = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_num_remaining")
        self.indices_num_remaining.set_as(self.tiling_ub[4])
        self.indices_loop_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_loop_num")
        self.indices_loop_num.set_as(self.tiling_ub[5])
        self.indices_nums_last = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_nums_last")
        self.indices_nums_last.set_as(self.tiling_ub[6])
        self.var_row_elem = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="var_row_elem")
        self.var_row_elem.set_as(self.tiling_ub[7])

        self.var_rows = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="var_rows")
        self.var_rows.set_as(self.tiling_ub[8])
        self.indices_step = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_step")
        self.indices_step.set_as(self.tiling_ub[9])
        self.num_multi_rows = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="num_multi_rows")
        self.num_multi_rows.set_as(self.tiling_ub[10])

        self.partial_factor = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="partial_factor")
        self.partial_factor.set_as(self.tiling_ub[11])
        self.elems_per_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="elems_per_core")
        self.elems_per_core.set_as(self.tiling_ub[12])
        self.elems_last_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="elems_last_core")
        self.elems_last_core.set_as(self.tiling_ub[13])
        self.elems_core_loop = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="elems_core_loop")
        self.elems_core_loop.set_as(self.tiling_ub[14])
        self.elems_core_remain = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="elems_core_remain")
        self.elems_core_remain.set_as(self.tiling_ub[15])
        self.elems_last_core_loop = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="elems_last_core_loop")
        self.elems_last_core_loop.set_as(self.tiling_ub[16])
        self.elems_last_core_remain = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="elems_last_core_remain")
        self.elems_last_core_remain.set_as(self.tiling_ub[17])

    def compute_mode_2(self, block_id):
        """
        compute for tiling mode 2: smaller than 32B of var row elements

        Parameters
        ----------
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        indices_ub = tik_instance.Tensor(self.indices_dtype, (self.indices_nums_once,),
                                         name="indices_ub", scope=tik.scope_ubuf)
        var_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,),
                                     name="var_ub", scope=tik.scope_ubuf)
        accum_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,),
                                       name="accum_ub", scope=tik.scope_ubuf)
        linear_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,),
                                        name="linear_ub", scope=tik.scope_ubuf)
        grad_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,),
                                      name="grad_ub", scope=tik.scope_ubuf)
        tmp_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,),
                                     name="tmp_ub", scope=tik.scope_ubuf)
        tmp2_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,),
                                      name="tmp2_ub", scope=tik.scope_ubuf)
        ub_tuples = (var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub)

        var_ub_block = tik_instance.Tensor(self.var_dtype, self.block_shape,
                                           name="var_ub_block", scope=tik.scope_ubuf)
        accum_ub_block = tik_instance.Tensor(self.var_dtype, self.block_shape,
                                           name="accum_ub_block", scope=tik.scope_ubuf)
        linear_ub_block = tik_instance.Tensor(self.var_dtype, self.block_shape,
                                           name="linear_ub_block", scope=tik.scope_ubuf)
        grad_ub_block = tik_instance.Tensor(self.var_dtype, self.block_shape,
                                              name="grad_ub_block", scope=tik.scope_ubuf)
        ub_block_tuples = (var_ub_block, accum_ub_block, linear_ub_block, grad_ub_block)

        self.var_cur_row = tik_instance.Scalar(dtype=self.tiling_dtype, name="var_cur_row")
        self.cached_rows_start_index = tik_instance.Scalar(dtype=self.tiling_dtype, name="cached_rows_start_index")
        self.cached_rows_start_index.set_as(self.var_rows)

        self.core_rows_start_index = self.indices_step * block_id
        self.core_rows_end_index = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="core_rows_end_index")
        with self.tik_instance.if_scope(block_id < self.need_core_num - 1):
            self.core_rows_end_index.set_as(self.indices_step * (block_id + 1))
        with self.tik_instance.else_scope():
            self.core_rows_end_index.set_as(self.var_rows)

        # process indices_num_each_core: indices_nums_once * indices_loop_num + indices_nums_last
        burst_len_indices = ceil_value(self.indices_nums_once, self.block_indices)
        burst_len_grad = ceil_value(self.indices_nums_once * self.var_row_elem, self.block_elem)
        burst_len_multi_row = ceil_value(self.num_multi_rows * self.var_row_elem, self.block_elem)
        with tik_instance.for_range(0, self.indices_loop_num) as indices_loop_i:
            indices_num_offset = indices_loop_i * self.indices_nums_once
            # move indices and grad data to ub from gm
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                   burst_len_indices, 0, 0)
            tik_instance.data_move(grad_ub, self.grad[indices_num_offset * self.var_row_elem], 0, 1,
                                   burst_len_grad, 0, 0)

            self.calc_multi_indices(indices_ub, self.indices_nums_once, burst_len_multi_row,
                                    ub_tuples, ub_block_tuples)

        with tik_instance.if_scope(self.indices_nums_last > 0):
            indices_num_offset = self.indices_loop_num * self.indices_nums_once
            burst_len_indices = ceil_value(self.indices_nums_last, self.block_indices)
            burst_len_grad = ceil_value(self.indices_nums_last * self.var_row_elem, self.block_elem)
            # move indices and grad data to ub from gm
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                   burst_len_indices, 0, 0)
            tik_instance.data_move(grad_ub, self.grad[indices_num_offset * self.var_row_elem], 0, 1,
                                   burst_len_grad, 0, 0)

            self.calc_multi_indices(indices_ub, self.indices_nums_last, burst_len_multi_row,
                                    ub_tuples, ub_block_tuples)

    def calc_multi_indices(self, indices_ub, indices_num, burst_len_multi_row, ub_tuples, ub_block_tuples):
        """
        calculate multi rows, multi rows will read at one to avoid loading
        little data from gm to ubuf at a high frequency

        Parameters
        ----------
        indices_ub: indices_ub
        indices_num: how many indices to calculate
        burst_len_multi_row: burst length of multi row
        ub_tuples: contains var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub
        ub_block_tuples: contains var_ub_block, accum_ub_block, linear_ub_block, grad_ub_block

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        with tik_instance.for_range(0, indices_num) as indices_i:
            self.var_cur_row.set_as(indices_ub[indices_i])

            # check whether current indices is within the processing range of the core
            with tik_instance.if_scope(tik.all(self.var_cur_row >= self.core_rows_start_index,
                                               self.var_cur_row < self.core_rows_end_index)):
                # check whether the var, accum, linear corresponding to current indices is cached in the UB
                with tik_instance.if_scope(tik.all(self.var_cur_row >= self.cached_rows_start_index,
                                                   self.var_cur_row < self.cached_rows_start_index +
                                                   self.num_multi_rows)):
                    self.calc_a_small_row(indices_i, ub_tuples, ub_block_tuples)
                with tik_instance.else_scope():
                    with tik_instance.if_scope(self.cached_rows_start_index < self.var_rows):
                        self.save_multi_rows(ub_tuples, burst_len_multi_row)
                    self.load_multi_rows(ub_tuples, burst_len_multi_row)
                    self.calc_a_small_row(indices_i, ub_tuples, ub_block_tuples)
        with tik_instance.if_scope(self.cached_rows_start_index < self.var_rows):
            self.save_multi_rows(ub_tuples, burst_len_multi_row)

    def calc_a_small_row(self, grad_idx, ub_tuples, ub_block_tuples):
        """
        calc a small whole row

        Parameters
        ----------
        grad_idx: row index of grad_ub
        ub_tuples: contains var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub
        ub_block_tuples: contains var_ub_block, accum_ub_block, linear_ub_block, grad_ub_block

        Returns
        -------
        None
        """
        var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub = ub_tuples
        var_ub_block, accum_ub_block, linear_ub_block, grad_ub_block = ub_block_tuples
        offset = self.var_cur_row - self.cached_rows_start_index

        with self.tik_instance.for_range(0, self.var_row_elem) as i:
            var_ub_block[i].set_as(var_ub[offset * self.var_row_elem + i])
            accum_ub_block[i].set_as(accum_ub[offset * self.var_row_elem + i])
            linear_ub_block[i].set_as(linear_ub[offset * self.var_row_elem + i])
            grad_ub_block[i].set_as(grad_ub[grad_idx * self.var_row_elem + i])

        calc_tuples = (var_ub_block, accum_ub_block, linear_ub_block, grad_ub_block, tmp_ub, tmp2_ub)
        self.sparse_calc(calc_tuples, 0, self.var_row_elem, 1)

        with self.tik_instance.for_range(0, self.var_row_elem) as i:
            var_ub[offset * self.var_row_elem + i].set_as(var_ub_block[i])
            accum_ub[offset * self.var_row_elem + i].set_as(accum_ub_block[i])
            linear_ub[offset * self.var_row_elem + i].set_as(linear_ub_block[i])

    def load_multi_rows(self, ub_tuples, burst_len_multi_row):
        """
        load multi rows of var, accum and linear from gm to ub

        Parameters
        ----------
        ub_tuples: contains var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub
        burst_len_multi_row: burst length of multi row

        Returns
        -------
        None
        """
        var_ub, accum_ub, linear_ub = ub_tuples[:3]
        with self.tik_instance.if_scope(self.var_cur_row + self.num_multi_rows <= self.core_rows_end_index):
            self.cached_rows_start_index.set_as(self.var_cur_row)
        with self.tik_instance.else_scope():
            self.cached_rows_start_index.set_as(self.core_rows_end_index - self.num_multi_rows)

        self.tik_instance.data_move(var_ub, self.var[self.cached_rows_start_index * self.var_row_elem],
                                    0, 1, burst_len_multi_row, 0, 0)
        self.tik_instance.data_move(accum_ub, self.accum[self.cached_rows_start_index * self.var_row_elem],
                                    0, 1, burst_len_multi_row, 0, 0)
        self.tik_instance.data_move(linear_ub, self.linear[self.cached_rows_start_index * self.var_row_elem],
                                    0, 1, burst_len_multi_row, 0, 0)

    def save_multi_rows(self, ub_tuples, burst_len_multi_row):
        """
        save multi rows var, accum and linear to gm

        Parameters
        ----------
        ub_tuples: contains var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub
        burst_len_multi_row: burst length of multi row

        Returns
        -------
        None
        """
        var_ub, accum_ub, linear_ub = ub_tuples[:3]
        self.tik_instance.data_move(self.var[self.cached_rows_start_index * self.var_row_elem], var_ub,
                                    0, 1, burst_len_multi_row, 0, 0)
        self.tik_instance.data_move(self.accum[self.cached_rows_start_index * self.var_row_elem], accum_ub,
                                    0, 1, burst_len_multi_row, 0, 0)
        self.tik_instance.data_move(self.linear[self.cached_rows_start_index * self.var_row_elem], linear_ub,
                                    0, 1, burst_len_multi_row, 0, 0)
        self.cached_rows_start_index.set_as(self.var_rows)

    def compute_mode_3(self, block_id):
        """
        compute for tiling mode 3

        Parameters
        ----------
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        indices_ub = tik_instance.Tensor(self.indices_dtype, (self.indices_nums_once,),
                                         name="indices_ub", scope=tik.scope_ubuf)
        var_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,),
                                     name="var_ub", scope=tik.scope_ubuf)
        accum_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,),
                                       name="accum_ub", scope=tik.scope_ubuf)
        linear_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,),
                                        name="linear_ub", scope=tik.scope_ubuf)
        grad_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,),
                                      name="grad_ub", scope=tik.scope_ubuf)
        tmp_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,),
                                     name="tmp_ub", scope=tik.scope_ubuf)
        tmp2_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,),
                                      name="tmp2_ub", scope=tik.scope_ubuf)
        ub_tuples = (var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub)

        self.grad_cur_row = tik_instance.Scalar(dtype=self.tiling_dtype, name="grad_cur_row")
        self.var_cur_row = tik_instance.Scalar(dtype=self.tiling_dtype, name="var_cur_row")

        burst_len_indices = ceil_value(self.indices_nums_once, self.block_indices)
        # move indices data to ub from gm
        tik_instance.data_move(indices_ub, self.indices, 0, 1, burst_len_indices, 0, 0)

        self.grad_cur_row.set_as(block_id / self.partial_factor)
        self.var_cur_row.set_as(indices_ub[self.grad_cur_row])
        self.calc_core_partial(self.var_cur_row, self.grad_cur_row, block_id, ub_tuples)

    def calc_core_partial(self, var_id, grad_id, block_id, ub_tuples):
        """
        calculate partial of a row by cores

        Parameters
        ----------
        var_id: row index
        grad_id: grad index
        block_id: core index
        ub_tuples: contains var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        core_start_offset = (block_id - grad_id * self.partial_factor) * self.elems_per_core
        var_ub_block = tik_instance.Tensor(self.var_dtype, self.block_shape,
                                           name="var_ub_block", scope=tik.scope_ubuf)
        accum_ub_block = tik_instance.Tensor(self.var_dtype, self.block_shape,
                                             name="accum_ub_block", scope=tik.scope_ubuf)
        linear_ub_block = tik_instance.Tensor(self.var_dtype, self.block_shape,
                                              name="linear_ub_block", scope=tik.scope_ubuf)
        block_ubs = (var_ub_block, accum_ub_block, linear_ub_block)

        with tik_instance.if_scope(block_id == (grad_id + 1) * self.partial_factor - 1):
            with tik_instance.if_scope(self.elems_last_core_loop > 0):
                self.process_part_loop(ub_tuples, var_id, grad_id, core_start_offset, self.elems_last_core_loop)

            with tik_instance.if_scope(self.elems_last_core_remain > 0):
                self.process_part_tail(ub_tuples, block_ubs, var_id, grad_id, core_start_offset,
                                       self.elems_last_core_loop, self.elems_last_core_remain)

        with tik_instance.else_scope():
            with tik_instance.if_scope(self.elems_core_loop > 0):
                self.process_part_loop(ub_tuples, var_id, grad_id, core_start_offset, self.elems_core_loop)

            with tik_instance.if_scope(self.elems_core_remain > 0):
                self.process_part_tail(ub_tuples, block_ubs, var_id, grad_id, core_start_offset,
                                       self.elems_core_loop, self.elems_core_remain)

    def process_part_tail(self, ub_tuples, block_ubs, var_id, grad_id, core_start_offset,
                          loops, part_elems_cnt):
        """
        process tail part

        Parameters
        ----------
        ub_tuples: contains var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub
        block_ubs: contains var_ub_block, accum_ub_block, linear_ub_block
        var_id: row index of var
        grad_id: row index of grad
        core_start_offset: offset of this part in this core
        loops: loop time
        part_elems_cnt: elements of tail part

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub = ub_tuples
        var_ub_block, accum_ub_block, linear_ub_block = block_ubs

        burst_len = ceil_value(part_elems_cnt, self.block_elem)
        self.var_row_repeat = ceil_value(part_elems_cnt, self.vector_elem)
        offset = self.one_part_elem * loops + core_start_offset
        var_offset = var_id * self.var_row_elem + offset
        # move grad to ub
        tik_instance.data_move(grad_ub, self.grad[grad_id * self.var_row_elem + offset],
                               0, 1, burst_len, 0, 0)
        # move var, accum, linear to ub
        tik_instance.data_move(var_ub, self.var[var_offset], 0, 1,
                               burst_len, 0, 0)
        tik_instance.data_move(accum_ub, self.accum[var_offset], 0, 1,
                               burst_len, 0, 0)
        tik_instance.data_move(linear_ub, self.linear[var_offset], 0, 1,
                               burst_len, 0, 0)

        # calculate
        self.sparse_calc(ub_tuples, 0, self.vector_elem, self.var_row_repeat)

        # move result to gm
        with tik_instance.if_scope(part_elems_cnt % self.block_elem == 0):
            tik_instance.data_move(self.var_out[var_offset], var_ub, 0, 1,
                                   burst_len, 0, 0)
            tik_instance.data_move(self.accum_out[var_offset], accum_ub, 0, 1,
                                   burst_len, 0, 0)
            tik_instance.data_move(self.linear_out[var_offset], linear_ub, 0, 1,
                                   burst_len, 0, 0)
        with tik_instance.else_scope():
            with self.tik_instance.for_range(0, self.block_elem) as i:
                var_ub_block[i].set_as(var_ub[part_elems_cnt - self.block_elem + i])
                accum_ub_block[i].set_as(accum_ub[part_elems_cnt - self.block_elem + i])
                linear_ub_block[i].set_as(linear_ub[part_elems_cnt - self.block_elem + i])

            tik_instance.data_move(self.var_out[var_offset], var_ub, 0, 1,
                                   burst_len - 1, 0, 0)
            tik_instance.data_move(self.accum_out[var_offset], accum_ub, 0, 1,
                                   burst_len - 1, 0, 0)
            tik_instance.data_move(self.linear_out[var_offset], linear_ub, 0, 1,
                                   burst_len - 1, 0, 0)

            tik_instance.data_move(self.var_out[var_offset + part_elems_cnt - self.block_elem],
                                   var_ub_block, 0, 1, 1, 0, 0)
            tik_instance.data_move(self.accum_out[var_offset + part_elems_cnt - self.block_elem],
                                   accum_ub_block, 0, 1, 1, 0, 0)
            tik_instance.data_move(self.linear_out[var_offset + part_elems_cnt - self.block_elem],
                                   linear_ub_block, 0, 1, 1, 0, 0)

    def process_part_loop(self, ub_tuples, var_id, grad_id, core_start_offset, loops):
        """
        process loops time of one part

        Parameters
        ----------
        ub_tuples: contains var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub
        var_id: row index of var
        grad_id: row index of grad
        core_start_offset: offset of this part in this core
        loops: loop time

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub = ub_tuples

        with tik_instance.for_range(0, loops) as loop_i:
            burst_len = ceil_value(self.one_part_elem, self.block_elem)
            self.var_row_repeat = self.one_part_elem // self.vector_elem
            offset = self.one_part_elem * loop_i + core_start_offset
            var_offset = var_id * self.var_row_elem + offset
            # move grad to ub
            tik_instance.data_move(grad_ub, self.grad[grad_id * self.var_row_elem + offset],
                                   0, 1, burst_len, 0, 0)
            # move var, accum, linear to ub
            tik_instance.data_move(var_ub, self.var[var_offset], 0, 1,
                                   burst_len, 0, 0)
            tik_instance.data_move(accum_ub, self.accum[var_offset], 0, 1,
                                   burst_len, 0, 0)
            tik_instance.data_move(linear_ub, self.linear[var_offset], 0, 1,
                                   burst_len, 0, 0)

            # calculate
            self.sparse_calc(ub_tuples, 0, self.vector_elem, self.var_row_repeat)

            # move result to gm
            tik_instance.data_move(self.var_out[var_offset], var_ub, 0, 1,
                                   burst_len, 0, 0)
            tik_instance.data_move(self.accum_out[var_offset], accum_ub, 0, 1,
                                   burst_len, 0, 0)
            tik_instance.data_move(self.linear_out[var_offset], linear_ub, 0, 1,
                                   burst_len, 0, 0)

    def compute_mode_1(self, block_id):
        """
        compute for tiling mode 1 of 32B aligned for var row

        Parameters
        ----------
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        indices_ub = tik_instance.Tensor(self.indices_dtype, (self.indices_nums_once,),
                                         name="indices_ub", scope=tik.scope_ubuf)
        var_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,),
                                     name="var_ub", scope=tik.scope_ubuf)
        accum_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,),
                                       name="accum_ub", scope=tik.scope_ubuf)
        linear_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,),
                                        name="linear_ub", scope=tik.scope_ubuf)
        grad_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,),
                                      name="grad_ub", scope=tik.scope_ubuf)
        tmp_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,),
                                     name="tmp_ub", scope=tik.scope_ubuf)
        tmp2_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,),
                                      name="tmp2_ub", scope=tik.scope_ubuf)
        ub_tuples = (var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub)

        self.var_cur_row = tik_instance.Scalar(dtype=self.tiling_dtype, name="var_cur_row")
        self.var_row_repeat = ceil_value(self.var_row_elem, self.vector_elem)

        # process indices_num_each_core: indices_nums_once * indices_loop_num + indices_nums_last
        with tik_instance.for_range(0, self.indices_loop_num) as indices_loop_i:
            indices_num_offset = block_id * self.indices_num_each_core + indices_loop_i * self.indices_nums_once
            self.process_num_indices(ub_tuples, indices_ub, self.indices_nums_once, indices_num_offset)

        with tik_instance.if_scope(self.indices_nums_last > 0):
            indices_num_offset = block_id * self.indices_num_each_core + \
                                 self.indices_loop_num * self.indices_nums_once
            self.process_num_indices(ub_tuples, indices_ub, self.indices_nums_last, indices_num_offset)

        with tik_instance.if_scope(tik.all(self.indices_num_remaining > 0, block_id < self.indices_num_remaining)):
            indices_num_offset = self.indices_num_each_core * self.need_core_num + block_id
            self.process_num_indices(ub_tuples, indices_ub, 1, indices_num_offset)

    def process_num_indices(self, ub_tuples, indices_ub, indices_num, indices_num_offset):
        """
        process num indices

        Parameters
        ----------
        ub_tuples: contains var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub
        indices_ub: indices ub
        indices_num: the number of indices
        indices_num_offset: the offset of indices in gm

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        var_ub, accum_ub, linear_ub, grad_ub = ub_tuples[0:4]

        burst_len = self.var_row_elem // self.block_elem

        # move indices data to ub from gm
        tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                               ceil_value(indices_num, self.block_indices), 0, 0)

        with tik_instance.for_range(0, indices_num) as indices_i:
            self.var_cur_row.set_as(indices_ub[indices_i])

            # move grad to ub
            tik_instance.data_move(grad_ub, self.grad[(indices_num_offset + indices_i) * self.var_row_elem],
                                   0, 1, burst_len, 0, 0)
            # move var, accum, linear to ub
            var_offset = self.var_cur_row * self.var_row_elem
            tik_instance.data_move(var_ub, self.var[var_offset], 0, 1,
                                   burst_len, 0, 0)
            tik_instance.data_move(accum_ub, self.accum[var_offset], 0, 1,
                                   burst_len, 0, 0)
            tik_instance.data_move(linear_ub, self.linear[var_offset], 0, 1,
                                   burst_len, 0, 0)

            # calculate
            self.sparse_calc(ub_tuples, 0, self.vector_elem, self.var_row_repeat)

            # move result to gm
            tik_instance.data_move(self.var_out[var_offset], var_ub, 0, 1,
                                   burst_len, 0, 0)
            tik_instance.data_move(self.accum_out[var_offset], accum_ub, 0, 1,
                                   burst_len, 0, 0)
            tik_instance.data_move(self.linear_out[var_offset], linear_ub, 0, 1,
                                   burst_len, 0, 0)

    def sparse_calc(self, ub_tuples, offset, mask, repeat):
        """
        calculate data according to the Ftrl-proximal scheme

        Parameters
        ----------
        ub_tuples: contains var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub
        offset: offset of var_row_elem
        mask: effective operation on element
        repeat: repeated iterations times

        Returns
        -------
        None
        """
        var_ub = ub_tuples[0][offset]
        accum_ub = ub_tuples[1][offset]
        linear_ub = ub_tuples[2][offset]
        grad_ub = ub_tuples[3][offset]
        tmp_ub = ub_tuples[4][offset]
        tmp2_ub = ub_tuples[5][offset]

        # tmp: grad*grad
        self.tik_instance.vmul(mask, tmp_ub, grad_ub, grad_ub,
                               repeat, 1, 1, 1, 8, 8, 8)
        # linear += grad, grad will not used after this operation
        self.tik_instance.vadd(mask, linear_ub, grad_ub, linear_ub,
                               repeat, 1, 1, 1, 8, 8, 8)
        # grad: ln(accum)
        self.tik_instance.vln(mask, grad_ub, accum_ub,
                              repeat, 1, 1, 8, 8)

        self.tik_instance.vmuls(mask, grad_ub, grad_ub, -self.lr_power_attr,
                                repeat, 1, 1, 8, 8)

        self.tik_instance.vexp(mask, grad_ub, grad_ub,
                               repeat, 1, 1, 8, 8)

        # accum_new = accum + grad*grad
        self.tik_instance.vadd(mask, accum_ub, accum_ub, tmp_ub,
                               repeat, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vln(mask, tmp_ub, accum_ub,
                              repeat, 1, 1, 8, 8)

        self.tik_instance.vmuls(mask, tmp_ub, tmp_ub, -self.lr_power_attr,
                                repeat, 1, 1, 8, 8)

        self.tik_instance.vexp(mask, tmp_ub, tmp_ub,
                               repeat, 1, 1, 8, 8)

        self.tik_instance.vmuls(mask, tmp2_ub, tmp_ub, self.lr_attr_rec,
                                repeat, 1, 1, 8, 8)

        # tmp: accum^(-lr_power)- accum_new^(-lr_power)
        self.tik_instance.vsub(mask, tmp_ub, grad_ub, tmp_ub,
                               repeat, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vmuls(mask, tmp_ub, tmp_ub, self.lr_attr_rec,
                                repeat, 1, 1, 8, 8)

        self.tik_instance.vmul(mask, tmp_ub, tmp_ub, var_ub,
                               repeat, 1, 1, 1, 8, 8, 8)

        # linear out
        self.tik_instance.vadd(mask, linear_ub, tmp_ub, linear_ub,
                               repeat, 1, 1, 1, 8, 8, 8)

        # x_res = max(min(linear, l1), -l1) - linear
        self.tik_instance.vector_dup(mask, tmp_ub, self.l1_attr,
                                     repeat, 1, 8)
        self.tik_instance.vmin(mask, grad_ub, linear_ub, tmp_ub,
                               repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vector_dup(mask, tmp_ub, -self.l1_attr,
                                     repeat, 1, 8)
        self.tik_instance.vmax(mask, tmp_ub, grad_ub, tmp_ub,
                               repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(mask, tmp_ub, tmp_ub, linear_ub,
                               repeat, 1, 1, 1, 8, 8, 8)

        # y_res = accum_new^(-lr_power)/lr + 2*l2
        self.tik_instance.vadds(mask, tmp2_ub, tmp2_ub, 2 * self.l2_attr,
                                repeat, 1, 1, 8, 8)

        # var = x_res/y_res
        self.tik_instance.vdiv(mask, var_ub, tmp_ub, tmp2_ub,
                               repeat, 1, 1, 1, 8, 8, 8)

    def sparse_apply_ftrl_compute_tiling(self):
        """
        Main process of sparse_apply_ftrl

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        with tik_instance.for_range(0, self.core_num, block_num=self.core_num) as block_id:
            # get run tiling data
            self.tiling_ub = tik_instance.Tensor(self.tiling_dtype, self.tiling_shape,
                                                 name="tiling_ub", scope=tik.scope_ubuf)
            tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0,
                                   1, ceil_value(TILING_ARG_NUM * TYPE_LEN_DICT.get(self.tiling_dtype), BLOCK_SIZE),
                                   0, 0)
            self.get_tiling_args()

            with tik_instance.if_scope(block_id < self.need_core_num):
                # TILING_MODE_1: var_row_elem is 32B aligned
                with tik_instance.if_scope(self.tiling_mode == TILING_MODE_1):
                    with tik_instance.new_stmt_scope():
                        self.compute_mode_1(block_id)
                # TILING_MODE_2: var_row_elem is smaller than 32B
                with tik_instance.else_scope():
                    with tik_instance.if_scope(self.tiling_mode == TILING_MODE_2):
                        with tik_instance.new_stmt_scope():
                            self.compute_mode_2(block_id)
                    with tik_instance.else_scope():
                        with tik_instance.new_stmt_scope():
                            self.compute_mode_3(block_id)

    def sparse_apply_ftrl_compute(self):
        """
        compute of sparse_apply_ftrl

        Parameters
        ----------
        None

        Returns
        -------
        compile info
        """
        self.var = self.tik_instance.Tensor(self.var_dtype, self.var_shape,
                                            name="var", scope=tik.scope_gm)
        self.accum = self.tik_instance.Tensor(self.var_dtype, self.var_shape,
                                              name="accum", scope=tik.scope_gm)
        self.linear = self.tik_instance.Tensor(self.var_dtype, self.var_shape,
                                               name="linear", scope=tik.scope_gm)
        self.grad = self.tik_instance.Tensor(self.var_dtype, self.grad_shape,
                                             name="grad", scope=tik.scope_gm)
        self.indices = self.tik_instance.Tensor(self.indices_dtype, self.indices_shape,
                                                name="indices", scope=tik.scope_gm)

        self.var_out = self.tik_instance.Tensor(self.var_dtype, shape=self.var_shape,
                                                name="var_out", scope=tik.scope_gm)
        self.accum_out = self.tik_instance.Tensor(self.var_dtype, shape=self.var_shape,
                                                  name="accum_out", scope=tik.scope_gm)
        self.linear_out = self.tik_instance.Tensor(self.var_dtype, shape=self.var_shape,
                                                   name="linear_out", scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, self.tiling_shape,
                                                  name="ddr_arg", scope=tik.scope_gm)

        self.sparse_apply_ftrl_compute_tiling()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.var, self.accum, self.linear, self.grad, self.indices),
                                   outputs=(self.var_out, self.accum_out, self.linear_out),
                                   flowtable=(self.tiling_gm,), enable_l2=True)

        # add compile info
        te.op.add_compile_info("vars", {"core_num": self.core_num,
                                        "ub_size": self.ub_size,
                                        "indices_dsize": self.indices_dsize
                                        })


@te.op.register_operator("SparseApplyFtrlD")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT,
                 REQUIRED_INPUT,
                 REQUIRED_OUTPUT, REQUIRED_OUTPUT, REQUIRED_OUTPUT,
                 (REQUIRED_ATTR_FLOAT, REQUIRED_ATTR_INT),
                 (REQUIRED_ATTR_FLOAT, REQUIRED_ATTR_INT),
                 (REQUIRED_ATTR_FLOAT, REQUIRED_ATTR_INT),
                 (REQUIRED_ATTR_FLOAT, REQUIRED_ATTR_INT), OPTION_ATTR_BOOL,
                 KERNEL_NAME)
def sparse_apply_ftrl_d(var_dict, accum_dict, linear_dict, grad_dict, indices_dict,
                        var_out_dict, accum_out_dict, linear_out_dict,
                        lr, l1, l2, lr_power, use_locking=False,
                        kernel_name="sparse_apply_ftrl"):
    """
    sparse_apply_ftrl_d interface, update the variable referenced by resource.

    Parameters
    ----------
    var_dict: data of input var, only support float32
    accum_dict: data of input accum, only support float32
    linear_dict: data of input linear, only support float32
    grad_dict: data of input grad, only support float32
    indices_dict: data of input indices, only support int32
    var_out_dict: data of input var, only support float32
    accum_out_dict: data of input accum, only support float32
    linear_out_dict: data of input linear, only support float32
    lr: attr, only supports support float32
    l1: attr, only supports support float32
    l2: attr, only supports support float32
    lr_power: attr, only support float32
    use_locking: bool, not used
    kernel_name: str, kernel name, default value is sparse_apply_ftrl

    Returns
    -------
    compile info
    """
    input_dicts = (var_dict, accum_dict, linear_dict, grad_dict, indices_dict)
    input_attrs = (lr, l1, l2, lr_power)
    output_dicts = (var_out_dict, accum_out_dict, linear_out_dict)
    obj = SparseApplyFtrl(input_dicts, input_attrs, output_dicts, kernel_name)

    return obj.sparse_apply_ftrl_compute()
