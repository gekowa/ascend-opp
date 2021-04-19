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
scatter_update
"""
from te import tik
from te import platform as tbe_platform
import te.lang.dynamic
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_op_params
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import OPTION_ATTR_BOOL
from te.utils.error_manager import error_manager_vector

# max int32
MAX_INT32 = 2**31 - 1
# tiling param num
TILING_ARG_NUM = 16
# reserved ub size
RESERVED_UB_SIZE = 8 * 1024
# apply ub for indices
INDICES_MAX_UB_NUM = 57064
# apply ub for updates
UPDATES_MAX_UB_NUM = 80


# pylint: disable=too-many-arguments,too-many-instance-attributes
class ScatterUpdate():
    """
       Function: use to store scatter_update base parameters
       Modify : 2020-07-26
    """
    def __init__(self, var, indices, updates, var_out, use_locking, kernel_name):
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.var_dtype = var.get("dtype").lower()
        self.indices_dtype = indices.get("dtype").lower()
        self.updates_dtype = updates.get("dtype").lower()
        self.out_dtype = var_out.get("dtype").lower()
        indices_support_dtype_list = ("int32", )
        var_support_dtype_list = ("float32", )
        check_dtype(self.indices_dtype, indices_support_dtype_list, param_name="indices")
        check_dtype(self.var_dtype, var_support_dtype_list, param_name="var")
        if self.var_dtype != self.updates_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "updates", "var",
                                                                  self.updates_dtype, self.var_dtype)
        if self.var_dtype != self.out_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "out", "var",
                                                                  self.out_dtype, self.var_dtype)
        self.kernel_name = kernel_name

        self.ai_core_num = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)
        self.ub_size_bytes = (tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.UB_SIZE) - RESERVED_UB_SIZE)
        self.var_dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(
            self.var_dtype) // 8
        self.indices_dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(
            self.indices_dtype) // 8
        self.var_data_each_block = 32 // self.var_dtype_bytes_size
        self.indices_data_each_block = 32 // self.indices_dtype_bytes_size

        self.tiling_gm = self.tik_instance.Tensor("int32", (TILING_ARG_NUM, ), name="tiling_gm", scope=tik.scope_gm)
        self.var_gm = self.tik_instance.Tensor(self.var_dtype, (MAX_INT32, ), name="var_gm", scope=tik.scope_gm)
        self.indices_gm = self.tik_instance.Tensor(self.indices_dtype, (MAX_INT32, ),
                                                   name="indices_gm", scope=tik.scope_gm)
        self.updates_gm = self.tik_instance.Tensor(self.updates_dtype, (MAX_INT32, ),
                                                   name="updates_gm", scope=tik.scope_gm)
        self.out_gm = self.tik_instance.Tensor(self.var_dtype, (MAX_INT32, ), name="out_gm", scope=tik.scope_gm)

        self.updates_ub = None
        self.indices_ub = None
        self.var_read_index = None
        self.updates_read_index = None
        self.indices_loop_index = None

    def tiling_args(self):
        """
        get runtime params from tiling

        Parameters
        ----------
        tiling_ub: tensor, runtime params from scatter_update tiling

        Returns
        -------
        None
        """
        self.tiling_mode = self.tik_instance.Scalar("int32", name="tiling_mode")
        self.indices_ub_number = self.tik_instance.Scalar("int32", name="indices_ub_number")
        self.updates_ub_number = self.tik_instance.Scalar("int32", name="updates_ub_number")
        self.indice_step = self.tik_instance.Scalar("int32", name="indice_step")
        self.core_num = self.tik_instance.Scalar("int32", name="core_num")
        self.updates_data_num = self.tik_instance.Scalar("int32", name="updates_data_num")
        self.indices_loop_num = self.tik_instance.Scalar("int32", name="indices_loop_num")
        self.indices_last_num = self.tik_instance.Scalar("int32", name="indices_last_num")
        self.indices_num = self.tik_instance.Scalar("int32", name="indices_num")

        self.tiling_mode.set_as(self.tiling_ub[0])
        self.indices_ub_number.set_as(self.tiling_ub[1])
        self.updates_ub_number.set_as(self.tiling_ub[2])
        self.indice_step.set_as(self.tiling_ub[3])
        self.core_num.set_as(self.tiling_ub[4])
        self.updates_data_num.set_as(self.tiling_ub[5])
        self.indices_loop_num.set_as(self.tiling_ub[6])
        self.indices_last_num.set_as(self.tiling_ub[7])
        self.indices_num.set_as(self.tiling_ub[8])

    def init_ub_tensor(self):
        """
        Compute the ub size of tensors

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.updates_ub = self.tik_instance.Tensor(self.updates_dtype, (UPDATES_MAX_UB_NUM,),
                                                   name="updates_ub", scope=tik.scope_ubuf)
        self.indices_ub = self.tik_instance.Tensor(self.indices_dtype, (INDICES_MAX_UB_NUM,),
                                                   name="indices_ub", scope=tik.scope_ubuf)

        self.var_read_index = self.tik_instance.Scalar("int32", name="var_read_index")
        self.var_read_index.set_as(0)
        self.updates_read_index = self.tik_instance.Scalar("int32", name="updates_read_index")
        self.updates_read_index.set_as(0)
        self.indices_loop_index = self.tik_instance.Scalar("int32", name="indices_loop_index")
        self.indices_loop_index.set_as(0)

    def updates_the_var(self, indices_in_index, indice_num):
        """
        Update the update fragment corresponding to the index

        Parameters
        ----------
        indices_in_index: int32
            Indices index on GM
        indice_num: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(indice_num % self.indices_data_each_block == 0):
            indices_burst_len = indice_num // self.indices_data_each_block
        with self.tik_instance.else_scope():
            indices_burst_len = (indice_num // self.indices_data_each_block) + 1
        updates_burst_len = self.updates_data_num // self.indices_data_each_block
        with self.tik_instance.if_scope(self.indices_num == 1):
            self.tik_instance.data_move(self.indices_ub, self.indices_gm, 0, 1,
                                        indices_burst_len, 0, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(self.indices_ub,
                                        self.indices_gm[indices_in_index], 0, 1,
                                        indices_burst_len, 0, 0)

        with self.tik_instance.for_range(0, indice_num) as indices_ub_index:
            self.var_read_index.set_as(self.indices_ub[indices_ub_index])
            with self.tik_instance.if_scope(self.core_num > 1):
                with self.tik_instance.if_scope(self.indices_loop_index * self.indice_step <= self.var_read_index):
                    with self.tik_instance.if_scope(
                            (self.indices_loop_index + 1) * self.indice_step > self.var_read_index):
                        self.tik_instance.data_move(self.updates_ub,
                                                    self.updates_gm[
                                                        (indices_ub_index + indices_in_index) * self.updates_data_num],
                                                    0, 1, updates_burst_len, 0, 0)
                        self.var_read_index.set_as(self.var_read_index * self.updates_data_num)
                        self.tik_instance.data_move(self.var_gm[self.var_read_index],
                                                    self.updates_ub, 0, 1, updates_burst_len, 0, 0)
            with self.tik_instance.else_scope():
                self.var_read_index.set_as(self.var_read_index * self.updates_data_num)

    def traversing_indices(self):
        """
        Traversing the index in the indices

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(self.indices_loop_num > 0):
            with self.tik_instance.for_range(
                    0, self.indices_loop_num) as indices_loop_index:
                self.updates_the_var(indices_loop_index * self.indices_ub_number, self.indices_ub_number)

        with self.tik_instance.if_scope(self.indices_last_num > 0):
            self.updates_the_var(self.indices_loop_num * self.indices_ub_number, self.indices_last_num)

    def scatter_update_compute_tiling(self):
        """
        Main process of scatter_update

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.ai_core_num, block_num=self.ai_core_num) as indices_loop_index:
            self.tiling_ub = self.tik_instance.Tensor("int32", (TILING_ARG_NUM,), name="tiling_ub",
                                                      scope=tik.scope_ubuf)
            self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 2, 0, 0)
            self.tiling_args()
            with self.tik_instance.if_scope(self.core_num > 1):
                with self.tik_instance.if_scope(indices_loop_index < self.core_num):
                    self.init_ub_tensor()
                    self.indices_loop_index.set_as(indices_loop_index)
                    self.traversing_indices()
            with self.tik_instance.else_scope():
                self.init_ub_tensor()
                self.traversing_indices()

    def scatter_update_operator(self):
        """
        scatter_update operation

        Parameters
        ----------
        None

        Returns:
        ----------
        compile info
        """
        self.scatter_update_compute_tiling()
        opt_config = {"out_of_bound_sync_check": True}
        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=(self.var_gm, self.indices_gm, self.updates_gm),
            outputs=(self.out_gm), flowtable=[self.tiling_gm], config=opt_config)

        te.op.add_compile_info("vars", {"ub_size": self.ub_size_bytes, "core_num": self.ai_core_num})

# pylint: disable=unused-argument
@te.op.register_operator("ScatterUpdate")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT,
                 OPTION_ATTR_BOOL, KERNEL_NAME)
def scatter_update(var, indices, updates, var_out, use_locking=False,
                   kernel_name="scatter_update"):
    """
    scatter_update interface

    Parameters
    ----------
    var_dict: input var shape, dtype and range
    indices_dict: input indices shape, dtype and range
    updates_dict: input updates shape, dtype and range
    var_out_dict: output shape, dtype and range
    kernel_name: kernel name of scatter_add op

    Returns
    -------
    compile info
    """
    obj = ScatterUpdate(var, indices, updates, var_out, False, kernel_name)
    return obj.scatter_update_operator()
