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
n_p_u_alloc_float_status
"""
from te import platform as tbe_platform
from topi.cce import util
from te import tik
from te.utils.op_utils import *

#constant 8
NUM_EIGHT = 8

# pylint: disable=invalid-name, too-many-locals, unused-argument
@check_op_params(REQUIRED_OUTPUT, KERNEL_NAME)
def n_p_u_alloc_float_status(data, kernel_name="n_p_u_alloc_float_status"):
    """
    the main function of n_p_u_alloc_float_status

    Parameters
    ----------
    data: dict,shape and datatype,datatype supports float32
    kernel_name: cce kernel name, default value is "n_p_u_alloc_float_status"

    Returns
    -------
    tik_instance: tik_instance
    """
    tik_instance = tik.Tik()
    data_output = tik_instance.Tensor("float32", (NUM_EIGHT,),
                                      name="data_output", scope=tik.scope_gm)
    data_ub = tik_instance.Tensor("float32", (NUM_EIGHT,), name="data_ub",
                                  scope=tik.scope_ubuf)
    tik_instance.vector_dup(NUM_EIGHT, data_ub, 0, 1, 1, 1)
    tik_instance.data_move(data_output, data_ub, 0, 1, 1, 0, 0)
    tik_instance.BuildCCE(kernel_name, inputs=[], outputs=[data_output])
    return tik_instance
