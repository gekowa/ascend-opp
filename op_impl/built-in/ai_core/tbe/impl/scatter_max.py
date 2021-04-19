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
scatter_max
"""
# pylint: disable=import-error
from topi.scatter import Scatter
from topi.cce import util
from te.utils.op_utils import *

# pylint: disable=too-many-arguments,unused-argument,invalid-name
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT,
                 OPTION_ATTR_BOOL, KERNEL_NAME)
def scatter_max(var,
                indices,
                updates,
                var_out,
                use_locking=False,
                kernel_name="scatter_max"):
    """
    Subtracts sparse updates to a variable reference.

    Parameters
    ----------
    var: dict
        data of input.
        source data type, support "int8", "uint8", "int32", "float16", "float32"
    indices: dict
         A tensor of indices into var, support "int32"
    updates: dict
        data of updates
        source data type should ne same as var
    var_out: dict
        data of output.
    use_locking: bool
        not used in this compute
    kernel_name: str
        kernel name, default value is "scatter_max"

    Returns:
    None
    """
    scatter_nd = Scatter(var, indices, updates, var_out, False, kernel_name,
                         "vmax")

    scatter_nd.scatter_operator()
