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
load_to_l1
"""
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.platform.cce_build import build_config
from te import platform as cce

PARA_LIST_LEN = 5


# pylint: disable=locally-disabled,unused-argument,unnecessary-lambda
@fusion_manager.register("load_to_l1")
def load_to_l1_compute(input_tensor, output_x, kernel_name="load_to_l1"):
    """
    calculating data

    Parameters
    ----------
    input_tensor : TVM tensor
        the input tensor
    output_x : dict
        dict of output_x, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "load_to_l1"

    Returns
    -------
    res
    """
    input_shape = input_tensor.shape
    res = tvm.compute(input_shape, lambda *indice: input_tensor(*indice), name="res")

    return res


def load_to_l1(input_x, output_x, kernel_name="load_to_l1"):
    """
    copy data from ddr to l1

    Parameters
    ----------
    input_x : TVM tensor
        the input tensor
    output_x : dict
        dict of output_x, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "load_to_l1"

    Returns
    -------
    None
    """
    input_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype")

    input_tensor = tvm.placeholder(input_shape,
                                   name="input_tensor",
                                   dtype=input_dtype)

    res = load_to_l1_compute(input_tensor, output_x, kernel_name=kernel_name)
    sch = tvm.create_schedule([res.op])

    sch[res].set_scope(cce.scope_cbuf_fusion)
    sch[res].emit_insn(res.op.axis[0], 'dma_copy')

    tensor_list = [input_tensor, res]

    with build_config:
        tvm.build(sch, tensor_list, "cce", name=kernel_name)



