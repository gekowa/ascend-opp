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
segment_prod
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util


@fusion_manager.register("segment_prod")
def segment_prod_compute(input_x, input_y, output_y, kernel_name="segment_prod"):
    """
    calculating data

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    input_y : list
        segment_ids
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "segment_prod"

    Returns
    -------
    res : output of the data`s segment_prod, shape is (input_y[-1]+1, XXX),
          XXX is the same as input_x shape from the second dimension to the end dimension
    """

    res = te.lang.cce.unsorted_segment_prod(input_x, input_y, input_y[-1]+1, init_value=1)
    return res


@util.check_input_type(dict, list, dict, str)
def segment_prod(input_data, segment_ids, output_data, kernel_name="segment_prod"):
    """
    algorithm: segment_prod
    calculating data

    Parameters
    ----------
    input_data : dict
        shape and dtype of first input, only support float16, float32
    segment_ids : list
        shape and dtype of second input, only support int32
        size is equal to the size of input_data first dimension
        values should be sorted and can be repeated
    output_data : dict
        shape and dtype of output,
    kernel_name : str
        kernel name, default value is "segment_prod"

    Returns
    -------
    None
    """

    shape_data = input_data.get("shape")
    shape_segment = (len(segment_ids),)
    dtype_data = input_data.get("dtype").lower()

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_data)
    util.check_shape_rule(shape_segment)
    util.check_tensor_shape_size(shape_data)
    util.check_tensor_shape_size(shape_segment)

    check_tuple_data = ("float16", "float32", "int8", "int32")
    util.check_dtype_rule(dtype_data, check_tuple_data)

    if shape_segment[0] != shape_data[0]:
        raise RuntimeError("the size od input_segment should equal to the size of data's first dimension")

    for i in range(shape_segment[0]):
        if not isinstance(segment_ids[i], int):
            raise RuntimeError("input_segment should be 1-D and all int!")
        if segment_ids[i] < 0:
            raise RuntimeError("input_segment should not be less than 0！")
        if i > 0 and segment_ids[i] < segment_ids[i - 1]:
            raise RuntimeError("input_segment not be sorted")

    if dtype_data == "int8":
        data_input = tvm.placeholder(shape_data, name="data_input", dtype="float16")
        res = segment_prod_compute(data_input, segment_ids, output_data, kernel_name)
        res = te.lang.cce.cast_to(res, "int8")
    else:
        data_input = tvm.placeholder(shape_data, name="data_input", dtype=dtype_data)
        res = segment_prod_compute(data_input, segment_ids, output_data, kernel_name)


    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}

    te.lang.cce.cce_build_code(schedule, config)
