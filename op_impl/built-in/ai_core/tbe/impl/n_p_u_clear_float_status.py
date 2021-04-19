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
n_p_u_clear_float_status
"""
from te import platform as tbe_platform
from topi.cce import util
from te import tik
from te.utils.op_utils import *

#constant 8
NUM_EIGHT = 8

# pylint:disable=invalid-name,too-many-locals,unused-argument,unused-variable
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def n_p_u_clear_float_status(addr, data, kernel_name="n_p_u_clear_float_status"):
    """
    the main function of npu_clear_float_status

    Parameters
    ----------
    addr: dict,shape and datatype,datatype supports float32
    data: dict,shape and datatype,datatype supports float32
    kernel_name: cce kernel name, default value is "n_p_u_clear_float_status"

    Returns
    -------
    tik_instance: tik_instance
    """
    tik_instance = tik.Tik()
    aicore_num = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)

    output_data = tik_instance.Tensor("float32", (NUM_EIGHT,),
                                      name="output_data", scope=tik.scope_gm)
    input_data = tik_instance.Tensor("float32", (NUM_EIGHT,), name="input_data",
                                     scope=tik.scope_gm)
    input_data_ub = tik_instance.Tensor("float32", (NUM_EIGHT,),
                                        name="input_data_ub",
                                        scope=tik.scope_ubuf)
    tik_instance.data_move(input_data_ub, input_data, 0, 1, 1, 0, 0)
    with tik_instance.for_range(0, aicore_num, block_num=aicore_num) as cycle:
        tik_instance.set_overflow_status(0)
        data_ub = tik_instance.Tensor("float32", (NUM_EIGHT,), name="data_ub",
                                      scope=tik.scope_ubuf)
        data_ub_input = tik_instance.Tensor("float16", (38400,),
                                            name="data_ub_input",
                                            scope=tik.scope_ubuf)
        tik_instance.vector_dup(128, data_ub_input, 3, 150, 8, 1)
        tik_instance.vector_dup(128, data_ub_input, 4, 150, 8, 1)
        tik_instance.vector_dup(128, data_ub_input, 5, 150, 8, 1)
        tik_instance.vector_dup(128, data_ub_input, 6, 150, 8, 1)
        tik_instance.vector_dup(128, data_ub_input, 7, 150, 8, 1)
        tik_instance.vector_dup(128, data_ub_input, 8, 150, 8, 1)
        tik_instance.vector_dup(NUM_EIGHT, data_ub, 0, 1, 0, 0)

        tik_instance.data_move(output_data, data_ub, 0, 1, 1, 8, 8)

    tik_instance.BuildCCE(kernel_name, inputs=[input_data],
                          outputs=[output_data])
    return tik_instance
