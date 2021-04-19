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
inplace_update
"""
from te import tik
from te import platform as tbe_platform
from te.utils.op_utils import *


# pylint: disable=unused-argument,invalid-name
def check_supported(x, indices, v, y, kernel_name="inplace_update"):
    """
    To check whether the AICORE operator can support the shape of indices or not
    """
    shape_indices = indices.get("shape")
    shape_v = v.get("shape")
    dtype_v = v.get("dtype").lower()
    reg_v_len = 1
    for i in range(1, len(shape_v)):
        reg_v_len = reg_v_len * shape_v[i]

    if dtype_v in ("float32", "int32"):
        dtype_size = 4
    else:
        dtype_size = 2
    reg_v_size = reg_v_len * dtype_size

    try:
        if len(shape_indices) != 1 or (reg_v_size % 32 != 0):
            return False

    except RuntimeError:
        return False

    return True


# pylint: disable=invalid-name,too-many-instance-attributes
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT,
                 REQUIRED_OUTPUT, KERNEL_NAME)
def inplace_update(x, indices, v, y, kernel_name="inplace_update"):
    """
    Updates specified rows with values in v

    Parameters
    ----------
    x : dict
        shape and dtype of input tensor x, only support float16, float32, int32
    indices: dict
        Indices into the left-most dimension of x
    v : dict
        shape and dtype of input tensor v,
         should be same shape and type as input
    y : dict
        shape and dtype of output tensor should be same shape and type as input
    kernel_name : str
        kernel name, default value is "inplace_update"

    Returns
    -------
    None
    """
    output_reslut = InplaceUpdate(x, indices, v)
    return output_reslut.tik_instance_function(kernel_name)


class InplaceUpdate():
    """
       Function: use to finish InplaceUpdate main functions
       Modify : 2020-7-14
    """

    def __init__(self, x, indices, v):
        self.shape_x = x.get("shape")
        self.shape_indices = indices.get("shape")
        self.shape_v = v.get("shape")
        self.dtype_x = x.get("dtype").lower()
        self.dtype_indices = indices.get("dtype").lower()
        self.dtype_v = v.get("dtype").lower()
        self.tik_instance = tik.Tik()
        self.core_num = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        self.dtype_size = \
            tbe_platform.cce_intrin.get_bit_len(self.dtype_x) // 8
        self.one_block_ele = 32 // self.dtype_size
        self.ub_size = \
            (tbe_platform.cce_conf.get_soc_spec(
                tbe_platform.cce_conf.UB_SIZE) - 1024) // self.dtype_size
        # input and output
        self.input_x_gm = self.tik_instance.Tensor(self.dtype_x,
                                                   self.shape_x,
                                                   name="input_x_gm",
                                                   scope=tik.scope_gm)
        self.input_indices_gm = self.tik_instance.Tensor(
            self.dtype_indices, (8,),
            name="input_indices_gm",
            scope=tik.scope_gm)
        self.input_v_gm = self.tik_instance.Tensor(self.dtype_v,
                                                   self.shape_v,
                                                   name="input_v_gm",
                                                   scope=tik.scope_gm)
        self.output_y_gm = self.tik_instance.Tensor(self.dtype_x,
                                                    self.shape_x,
                                                    name="output_y_gm",
                                                    scope=tik.scope_gm)

    # pylint: disable=too-many-locals, too-many-function-args
    def tik_instance_function(self, kernel_name):
        """
        tik_instance_function

        Parameters
        ----------
        kernel_name: str
            kernel_name

        Returns
        -------
        tik_instance
        """
        self.check_param()
        if self.check_special():
            self.fun_special(kernel_name)
            return

        # caculate block number
        value_n = self.shape_x[0]
        n_size = value_n // self.core_num + \
                 (1 if value_n % self.core_num > 0 else 0)
        if (value_n % self.core_num == 0) or (value_n % n_size == 0):
            is_same_core = 0
        else:
            is_same_core = 1

        block_dim = value_n // n_size + (0 if value_n //
                                         self.core_num == 0 else is_same_core)
        # caculate lenth of input data
        reg_x_len = 1
        for i in range(1, len(self.shape_x)):
            reg_x_len = reg_x_len * self.shape_x[i]

        if self.dtype_x in ("float32", "int32"):
            input_size = reg_x_len * 4
        else:
            input_size = reg_x_len * 2

        if input_size < self.ub_size * self.dtype_size:
            # ub_size is enough
            with self.tik_instance.for_range(0, block_dim, block_num=block_dim) \
                    as block_index:
                with self.tik_instance.if_scope(block_index != block_dim - 1):
                    self.fun_no_cut(reg_x_len, n_size, block_index, n_size)
                with self.tik_instance.else_scope():
                    self.fun_no_cut(reg_x_len, n_size, block_index,
                                    self.shape_x[0] - block_index * n_size)
        else:
            # ub_size is not enough
            with self.tik_instance.for_range(0, block_dim, block_num=block_dim) \
                    as block_index:
                with self.tik_instance.if_scope(block_index != block_dim - 1):
                    self.fun_ned_cut(reg_x_len, n_size, block_index, n_size)
                with self.tik_instance.else_scope():
                    self.fun_ned_cut(reg_x_len, n_size, block_index,
                                     self.shape_x[0] - block_index * n_size)
        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=(self.input_x_gm,
                                           self.input_indices_gm,
                                           self.input_v_gm),
                                   outputs=(self.output_y_gm))
        return self.tik_instance

    def fun_special(self, kernel_name):
        """special function"""
        dim_0 = self.shape_x[0] * self.shape_x[1]
        dim_1 = self.shape_x[2]
        dim_2 = self.shape_x[3]

        core_num = self.core_num
        dim_0_div = dim_0 // core_num
        if dim_0 % core_num != 0:
            core_num = 1
            dim_0_div = dim_0

        num = self.ub_size // dim_2
        num_x = min(dim_0_div * dim_1, num)
        loop_x = dim_0_div * dim_1 // num_x
        last_x = dim_0_div * dim_1 % num_x
        num_v = min(dim_0_div, num)
        loop_v = dim_0_div // num_v
        last_v = dim_0_div % num_v
        v_ub = self.tik_instance.Tensor(self.dtype_x, (num, dim_2),
                                        name="v_ub",
                                        scope=tik.scope_ubuf)
        indices_ub = self.tik_instance.Tensor(self.dtype_indices, (8,),
                                              name="indices_ub",
                                              scope=tik.scope_ubuf)
        self.tik_instance.data_move(indices_ub, self.input_indices_gm, 0, 1, 1,
                                    0, 0)
        reg_idx = self.tik_instance.Scalar(dtype="int32")
        reg_idx.set_as(indices_ub[0])
        with self.tik_instance.for_range(0, core_num,
                                         block_num=core_num) as core_idx:
            offset_core = core_idx * dim_0_div * dim_1 * dim_2
            with self.tik_instance.for_range(0, loop_x) as loop_idx:
                offset_loop = offset_core + loop_idx * num_x * dim_2
                self.tik_instance.data_move(v_ub, self.input_x_gm[offset_loop],
                                            0, num_x,
                                            dim_2 // self.one_block_ele, 0, 0)
                self.tik_instance.data_move(self.output_y_gm[offset_loop], v_ub,
                                            0, num_x,
                                            dim_2 // self.one_block_ele, 0, 0)
            if last_x != 0:
                offset_last = offset_core + loop_x * num_x * dim_2
                self.tik_instance.data_move(v_ub, self.input_x_gm[offset_last],
                                            0, last_x,
                                            dim_2 // self.one_block_ele, 0, 0)
                self.tik_instance.data_move(self.output_y_gm[offset_last], v_ub,
                                            0, last_x,
                                            dim_2 // self.one_block_ele, 0, 0)
            src_offset_core = core_idx * dim_0_div * dim_2
            dst_offset_core = core_idx * dim_0_div * dim_1 * dim_2
            with self.tik_instance.for_range(0, loop_v) as loop_idx:
                src_offset_loop = src_offset_core + loop_idx * num_v * dim_2
                dst_offset_loop = dst_offset_core + reg_idx * dim_2 + loop_idx \
                                  * num_v * dim_1 * dim_2
                self.tik_instance.data_move(v_ub,
                                            self.input_v_gm[src_offset_loop], 0,
                                            num_v, dim_2 // self.one_block_ele,
                                            0, 0)
                self.tik_instance.data_move(
                    self.output_y_gm[dst_offset_loop], v_ub, 0, num_v,
                    dim_2 // self.one_block_ele, 0,
                    (dim_1 - 1) * dim_2 // self.one_block_ele)
            if last_v != 0:
                src_offset_last = src_offset_core + loop_v * num_v * dim_2
                dst_offset_last = dst_offset_core + reg_idx * dim_2 + loop_v \
                                  * num_v * dim_1 * dim_2
                self.tik_instance.data_move(v_ub,
                                            self.input_v_gm[src_offset_last], 0,
                                            last_v, dim_2 // self.one_block_ele,
                                            0, 0)
                self.tik_instance.data_move(
                    self.output_y_gm[dst_offset_last], v_ub, 0, last_v,
                    dim_2 // self.one_block_ele, 0,
                    (dim_1 - 1) * dim_2 // self.one_block_ele)

        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=(self.input_x_gm,
                                           self.input_indices_gm,
                                           self.input_v_gm),
                                   outputs=(self.output_y_gm))
        return self.tik_instance

    def fun_no_cut(self, reg_x_len, n_size, block_index, n_loop):
        """
        funtion no need cut

        Parameters
        ----------
        reg_x_len: lenth of x
        nc1_size: size of n
        block_index: index of block
        n_loop: number of loop

        Returns
        -------
        none
        """
        data_input_ub = self.tik_instance.Tensor(self.dtype_x,
                                                 self.shape_v,
                                                 name="data_input_ub",
                                                 scope=tik.scope_ubuf)
        input_indices_ub = self.tik_instance.Tensor(self.dtype_indices, (8,),
                                                    name="input_indices_ub",
                                                    scope=tik.scope_ubuf)
        self.tik_instance.data_move(input_indices_ub[0],
                                    self.input_indices_gm[0], 0, 1, 1, 0, 0)
        reg_start = self.tik_instance.Scalar(dtype="int32")
        reg_start.set_as(input_indices_ub[0])
        reg_burst = self.tik_instance.Scalar(dtype="int32")
        if self.dtype_x in ("float32", "int32"):
            reg_burst.set_as(reg_x_len // 8)
        else:
            reg_burst.set_as(reg_x_len // 16)

        with self.tik_instance.for_range(0, n_loop) as n_index:
            with self.tik_instance.if_scope(
                    block_index * n_size + n_index != reg_start):
                self.tik_instance.data_move(
                    data_input_ub[0],
                    self.input_x_gm[(block_index * n_size + n_index) *
                                    reg_x_len], 0, 1, reg_burst, 0, 0)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(data_input_ub[0],
                                            self.input_v_gm[0], 0, 1, reg_burst,
                                            0, 0)
            self.tik_instance.data_move(
                self.output_y_gm[(block_index * n_size + n_index) * reg_x_len],
                data_input_ub[0], 0, 1, reg_burst, 0, 0)

    def fun_ned_cut(self, reg_x_len, n_size, block_index, n_loop):
        """
        funtion no need cut

        Parameters
        ----------
        reg_x_len: lenth of x
        nc1_size: size of n
        block_index: index of block
        n_loop: number of loop

        Returns
        -------
        none
        """
        input_indices_ub = self.tik_instance.Tensor(self.dtype_indices, (8,),
                                                    name="input_indices_ub",
                                                    scope=tik.scope_ubuf)
        self.tik_instance.data_move(input_indices_ub[0],
                                    self.input_indices_gm[0], 0, 1, 1, 0, 0)
        reg_start = self.tik_instance.Scalar(dtype="int32")
        reg_start.set_as(input_indices_ub[0])
        reg_burst = 3200
        if self.dtype_x in ("float32", "int32"):
            reg_dtype_size = 8
            if reg_x_len % 25600 == 0:
                num_move = reg_x_len // 25600
                data_tail = 25600
            else:
                num_move = (reg_x_len // 25600) + 1
                data_tail = reg_x_len - (reg_x_len // 25600) * 25600
            tail_burst = data_tail // 8
        else:
            reg_dtype_size = 16
            if reg_x_len % 51200 == 0:
                num_move = reg_x_len // 51200
                data_tail = 51200
            else:
                num_move = (reg_x_len // 51200) + 1
                data_tail = reg_x_len - (reg_x_len // 51200) * 51200
            tail_burst = data_tail // 16

        data_input_ub = self.tik_instance.Tensor(self.dtype_x,
                                                 (reg_burst, reg_dtype_size),
                                                 name="data_input_ub",
                                                 scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, n_loop) as n_index:
            with self.tik_instance.if_scope(
                    block_index * n_size + n_index != reg_start):
                with self.tik_instance.for_range(0, num_move) as move_index:
                    with self.tik_instance.if_scope(move_index != num_move - 1):
                        self.tik_instance.data_move(
                            data_input_ub[0], self.input_x_gm[
                                (block_index * n_size + n_index) * reg_x_len +
                                move_index * reg_burst * reg_dtype_size], 0, 1,
                            reg_burst, 0, 0)
                        self.tik_instance.data_move(
                            self.output_y_gm[
                                (block_index * n_size + n_index) * reg_x_len +
                                move_index * reg_burst * reg_dtype_size],
                            data_input_ub[0], 0, 1, reg_burst, 0, 0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.data_move(
                            data_input_ub[0], self.input_x_gm[
                                (block_index * n_size + n_index) * reg_x_len +
                                move_index * reg_burst * reg_dtype_size], 0, 1,
                            tail_burst, 0, 0)
                        self.tik_instance.data_move(
                            self.output_y_gm[
                                (block_index * n_size + n_index) * reg_x_len +
                                move_index * reg_burst * reg_dtype_size],
                            data_input_ub[0], 0, 1, tail_burst, 0, 0)
            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, num_move) as move_index:
                    with self.tik_instance.if_scope(move_index != num_move - 1):
                        self.tik_instance.data_move(
                            data_input_ub[0],
                            self.input_v_gm[move_index * reg_burst *
                                            reg_dtype_size], 0, 1, reg_burst, 0,
                            0)
                        self.tik_instance.data_move(
                            self.output_y_gm[
                                (block_index * n_size + n_index) * reg_x_len +
                                move_index * reg_burst * reg_dtype_size],
                            data_input_ub[0], 0, 1, reg_burst, 0, 0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.data_move(
                            data_input_ub[0],
                            self.input_v_gm[move_index * reg_burst *
                                            reg_dtype_size], 0, 1, tail_burst,
                            0, 0)
                        self.tik_instance.data_move(
                            self.output_y_gm[
                                (block_index * n_size + n_index) * reg_x_len +
                                move_index * reg_burst * reg_dtype_size],
                            data_input_ub[0], 0, 1, tail_burst, 0, 0)

    def check_param(self):
        """
        check parameters, if one is invalid, then raise error

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        check_tuple = ("float16", "float32", "int32")
        check_shape(self.shape_x, param_name="x")
        check_shape(self.shape_indices, param_name="indices")
        check_shape(self.shape_v, param_name="v")
        check_dtype(self.dtype_x, check_tuple, param_name="x")
        check_dtype(self.dtype_indices, ("int32",), param_name="indices")
        check_dtype(self.dtype_v, check_tuple, param_name="v")
        if len(self.shape_x) != len(self.shape_v):
            raise RuntimeError("The number of dimension x must"
                               " be same as dimension v")

        if self.shape_v[0] != self.shape_indices[0]:
            raise RuntimeError("The length of rank 0 of tensor v must"
                               " be the same as length of indices")

        if len(self.shape_indices) != 1:
            raise RuntimeError("The length of indices only support 1")
        for i in range(1, len(self.shape_v)):
            if self.shape_x[i] != self.shape_v[i]:
                if not self.check_special():
                    raise RuntimeError("The length of each rank of tensor x"
                                       " must be the same as length of"
                                       " each or next rank of tensor v")

    def check_special(self):
        """check special"""
        if len(self.shape_x) == 4:
            if (self.shape_x[0] != self.shape_v[1]) or \
                    (self.shape_x[1] != self.shape_v[2]) or \
                    (self.shape_x[3] != self.shape_v[3]) or \
                    self.shape_x[3] > 256:
                return False
            else:
                return True
        else:
            return False
