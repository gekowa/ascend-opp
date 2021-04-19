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
unsorted_segment_prod_d
"""
import te.lang.cce
from te import platform as tbe_platform
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils import op_utils
from topi import generic

#block length in number
BLOCK_LENGTH = 32
#max ub size
UB_SIZE_MAX = 293952


# pylint: disable=unused-argument,invalid-name
def check_supported(x,
                    segment_ids,
                    y,
                    num_segments,
                    kernel_name="unsorted_segment_prod_d"):
    """
    fusion pass test if num_segments is int32
    """
    shape = x.get("shape")
    dtype = x.get("dtype").lower()
    segment_ids_shape = segment_ids.get("shape")
    segment_ids_dtype = segment_ids.get("dtype").lower()
    check_list = ("float16", "float32", "int32", "int16")
    op_utils.check_dtype(dtype, check_list, param_name="x")
    check_list_ids = ("int32")
    op_utils.check_dtype(segment_ids_dtype,
                         check_list_ids,
                         param_name="segment_ids")
    if num_segments <= 0:
        return False
    first_shape = int(shape[0])
    ids_length = int(segment_ids_shape[0])
    if first_shape != ids_length:
        return False
    total_ub_size = (num_segments + first_shape) * BLOCK_LENGTH + (
        (BLOCK_LENGTH // 2 - first_shape %
         (BLOCK_LENGTH // 4)) + first_shape) * (BLOCK_LENGTH // 8)
    if total_ub_size > UB_SIZE_MAX // 2:
        return False
    return True


# pylint: disable=unused-argument,invalid-name,no-member
@fusion_manager.register("unsorted_segment_prod_d")
def unsorted_segment_prod_d_compute(x,
                                    segment_ids,
                                    y,
                                    num_segments,
                                    kernel_name="unsorted_segment_prod_d"):
    """
    compute for unsorted_segment_prod_d_compute
    """
    res = te.lang.cce.unsorted_segment_prod(x, segment_ids, num_segments)
    return res


# pylint: disable =too-many-locals
@op_utils.check_op_params(op_utils.REQUIRED_INPUT, op_utils.REQUIRED_INPUT,
                          op_utils.REQUIRED_OUTPUT, op_utils.REQUIRED_ATTR_INT,
                          op_utils.KERNEL_NAME)
def unsorted_segment_prod_d(x,
                            segment_ids,
                            y,
                            num_segments,
                            kernel_name="unsorted_segment_prod_d"):
    """
    Operation and Schedule for unsorted_segment_prod_d.

    Parameters
    ----------
    x: dict
        shape and dtype of input.
        dtype only support float16, float32, int32

    segment_ids : dict
        should be the size of the first dimension
        need not cover all values in the full range of valid values
        dtype only support int32

    y: dict
        shape and dtype of output.

    num_segments : the dimension of the first axis of
                   the output tensor(>= max(segment_ids) + 1)

    kernel_name : cce kernel name,
                  default value is "unsorted_segment_prod_d"

    Returns
    -------
        None
    """
    shape = x.get("shape")
    dtype = x.get("dtype").lower()
    segment_ids_shape = segment_ids.get("shape")
    segment_ids_dtype = segment_ids.get("dtype").lower()
    op_utils.check_shape(shape, param_name="x")
    op_utils.check_shape(segment_ids_shape, param_name="segment_ids")

    check_list = ("float16", "float32", "int32", "int16")
    op_utils.check_dtype(dtype, check_list, param_name="x")
    check_list_ids = ("int32", )
    op_utils.check_dtype(segment_ids_dtype,
                         check_list_ids,
                         param_name="segment_ids")
    prod_support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.vmul", "float32")
    if dtype == "float32" and not prod_support:
        raise RuntimeError(
            "Input dtype only support float16 while input dtype is float32")
    if num_segments <= 0:
        raise RuntimeError("unsorted_segment_prod_d only supports num_segments"
                           " greater than 0, while num_segments is %d" %
                           (num_segments))

    first_shape = int(shape[0])
    ids_length = int(segment_ids_shape[0])
    if first_shape != ids_length:
        raise RuntimeError(
            "unsorted_segment_prod_d only support inputs[0] equal "
            "to segment_ids_shape[0], while inputs[0] is %d, "
            "segment_ids_shape[0] is %d" % (first_shape, ids_length))
    total_ub_size = (num_segments + first_shape) * BLOCK_LENGTH + (
        (BLOCK_LENGTH // 2 - first_shape %
         (BLOCK_LENGTH // 4)) + first_shape) * (BLOCK_LENGTH // 8)
    if total_ub_size > UB_SIZE_MAX // 2:
        raise RuntimeError(
            "unsorted_segment_prod_d num_segments=%d, shape[0]=%d, "
            "greater than UB_SIZE_MAX" % (num_segments, shape[0]))

    dtype = dtype.lower()
    data_inputs = tvm.placeholder(shape, name="data_inputs", dtype=dtype)
    data_segments_id = tvm.placeholder(segment_ids_shape,
                                       name="data_segments_id",
                                       dtype=segment_ids_dtype)
    with tvm.target.cce():
        res = unsorted_segment_prod_d_compute(data_inputs, data_segments_id, y,
                                              num_segments, kernel_name)

        sch = generic.auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": [data_inputs, data_segments_id, res]
    }
    te.lang.cce.cce_build_code(sch, config)
