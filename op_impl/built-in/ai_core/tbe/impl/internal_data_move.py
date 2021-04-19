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
internal data move
"""
from __future__ import absolute_import
from impl import load_to_l1
from impl import store_to_gm
from te.utils.op_utils import *


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_STR, REQUIRED_ATTR_STR, KERNEL_NAME)
def internal_data_move(x, y, src_buf, dst_buf,
                       kernel_name='internal_data_move'):
    """
    algorithm: internal_data_move
    Assist in handling internal data

    Parameters
    ----------
    src : dict
        shape and dtype of input
    dst: dict
        shape and dtype of output, should be same shape and type as input
    src_buf: str
        source data in buf, can be L1, Gm.
    dst_buf: str
        target data in buf, can be Gm, L1.
    kernel_name: str
        kernel name, default value is "internal_data_move"

    Returns
    -------
    None
    """
    if (src_buf.upper() == "GM" and dst_buf.upper() == "L1"):
        load_to_l1(x, y, kernel_name)
    elif (src_buf.upper() == "L1" and dst_buf.upper() == "GM"):
        store_to_gm(x, y, kernel_name)
    else:
        raise RuntimeError("not support this kind of data move !")

