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
common function
"""
from te.tvm import api as tvm
from te.lang.cce.te_compute.elewise_compute import vmuls
from te.lang.cce.te_compute.elewise_compute import vabs
from te.lang.cce.te_compute.elewise_compute import vadds
from te.lang.cce.te_compute.elewise_compute import vdiv


def sign(input_data):
    """
    Algrithm:
        sign(x) = 2**(15)/(2**(-15) + 2**(15) *|x|)
    ----------
    Parameters
        input_data: the placeholder of data input
    ----------
    Returns
        A tensor of sign(x)
    -------
    """
    dtype = input_data.dtype

    if dtype == "float16":
        fp_max = tvm.const(2**15, dtype)
        fp_min = tvm.const(2**(-15), dtype)
    elif dtype == "float32":
        fp_max = tvm.const(2**62, dtype)
        fp_min = tvm.const(2**(-62), dtype)
    else:
        raise RuntimeError(
            "The type must be float16 or float32.")
    new_data = vmuls(input_data, fp_max)
    abs_data = vabs(new_data)
    denominator = vadds(abs_data, fp_min)
    res = vdiv(new_data, denominator)

    return res
