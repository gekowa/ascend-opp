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
segment_min
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util


SHAPE_SIZE_LIMIT = 2147483648


@fusion_manager.register("segment_min")
def segment_min_compute(input_x, input_y, output_y, kernel_name="segment_min"):
    """
    calculating data

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    input_y : list int
        the ids of input_tensor
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "segment_min"

    Returns
    -------
    res : result of the segmentmin
    """

    res = te.lang.cce.unsorted_segment_min(input_x, input_y, input_y[-1] + 1)
    return res


def ids_is_1d_and_sorted(ids):
    """
    judgement of ids

    Parameters
    ----------
    ids : listint
        the segment_ids

    Returns
    -------
    None
    """
    n = len(ids)
    if ids != sorted(ids):
        raise RuntimeError("ids must be sorted!")
    for i in ids:
        if not isinstance(i, int):
            raise RuntimeError("ids must be 1D and all int!")
        if i < 0:
            raise RuntimeError("ids should all be greater than zero!")


@util.check_input_type(dict, list, dict, str)
def segment_min(input_tensor, segment_ids, output_y, kernel_name="segment_min"):
    """
    calculating data

    Parameters
    ----------
    input_tensor : dict
        shape and dtype of input
    segment_ids : list int
        the list of segment_ids
    output_y : dict
        shape and dtype of output,
    kernel_name : str
        kernel name, default value is "segment_min"

    Returns
    -------
    None
    """


    shape_tensor = input_tensor.get("shape")
    dtype_tensor = input_tensor.get("dtype")
    input_tensor_dtype = dtype_tensor.lower()

    # judgement of ids
    length_ids = len(segment_ids)
    if length_ids != shape_tensor[0]:
        raise RuntimeError("length of ids must equal to shape[0] of input_tensor!")
    ids_is_1d_and_sorted(segment_ids)

    check_tuple_tensor = ("float16", "float32", "int32", "int8", "uint8")
    util.check_dtype_rule(dtype_tensor, check_tuple_tensor)
    util.check_shape_size(shape_tensor, SHAPE_SIZE_LIMIT)
    util.check_shape_rule(shape_tensor) # 校验轴

    if dtype_tensor == "int8":
        data_input = tvm.placeholder(shape_tensor, name="data_input", dtype=input_tensor_dtype)
        data_input1 = te.lang.cce.cast_to(data_input, "float16")
        res1 = segment_min_compute(data_input1, segment_ids, output_y, kernel_name)
        res = te.lang.cce.cast_to(res1, "int8")
    else:
        data_input = tvm.placeholder(shape_tensor, name="data_input", dtype=input_tensor_dtype)
        res = segment_min_compute(data_input, segment_ids, output_y, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}

    te.lang.cce.cce_build_code(schedule, config)

