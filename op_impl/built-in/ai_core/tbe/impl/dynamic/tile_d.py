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
dynamic tile_d
"""
import te.lang.dynamic
from te import tvm
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import OPTION_ATTR_LIST_INT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_op_params
from te.utils.op_utils import variable_shape
from te.utils.op_utils import OP_ERROR_CODE_005
from te.utils.op_utils import OP_ERROR_CODE_009
from te.utils.op_utils import OP_ERROR_CODE_012
from topi import generic


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
def tile_d_compute(data, output_x, multiples, kernel_name="tile_d"):
    """TVM calculation process, used for fusion operation.

    Parameters
    ----------
    data: list of placeholders.
        Input data.
    output_x: dict.
        dict of output.
    multiples : list or tuple.
        Number of the axis replicates.
    kernel_name : str.
        Cce kernel name, default value is "tile_d".

    Returns
    -------
    res
    """
    shape = te.lang.dynamic.shape_to_list(data.shape)
    out_shape = []
    for shape_i, multiples_i in zip(shape, multiples):
        out_shape_i = shape_i * multiples_i
        out_shape.append(out_shape_i)
    res = te.lang.dynamic.broadcast(data, out_shape)

    return res


# pylint: disable=too-many-locals
@te.op.register_operator("TileD")
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, OPTION_ATTR_LIST_INT, KERNEL_NAME)
def tile_d(input_x, output_x, multiples, kernel_name="tile_d"):
    """algorithm: tile.
    The tile in tensorflow can multiple the shape of the given tensor.
    For example, tiling [a b c d] by [2] produces [a b c d a b c d].
    The tile op in TBE is different from tf.tile, tile of TBE use broadcast
    api, and only support that at least an axis in shape is 1.The '1' axis
    is to be multipled.
    For example, if shape = [51, 1] and multiples = [1, 77], after computation,
    the output shape will be [51, 77].
    Abnormal condition:
    1. The length of shape must be equal to or less than the shape of multiples.
    2. The type of kernel_name is not string.
    3. The shape is neither list nor tuple.
    4. The dtype is not float32, float16, or int32.
    5. All of the axises of the multiples is 1.

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_x: dict
        dict of output.
    multiples : list or tuple.
        Number of the axis replicates.
    kernel_name : str.
        kernel name, default value is "tile_d".

    Returns
    -------
    None
    """

    dtype = input_x.get("dtype").lower()
    check_list = ("float16", "float32", "int32")
    check_dtype(dtype, check_list, param_name="input_x")
    unkown_shape = []
    shape = input_x.get("shape")
    for i in range(0, len(shape)):
        if shape[i] == -1:
            unkown_shape.append(i)

    with te.op.compute():
        shape = te.lang.dynamic.shape_to_list(variable_shape([input_x])[0])
        multiples = te.lang.dynamic.shape_to_list(multiples)
        origin_multiples = multiples

        input_format = input_x.get("format")
        output_format = output_x.get("format")
        if input_format in ("NCHW", "NHWC") and output_format in ("NC1HWC0",):
            # branch: 4D tile to 5HD ((N, 1, 1, 1) to (N, C1, H, W, C0))
            # and output C is 16 align
            # change input shape from (N, 1, 1, 1) to (N, 1, 1, 1, 1)
            shape = shape + [1]
            if input_format == "NCHW":
                # change multiples from (1, C, H, W) to (1, C1, H, W, C0)
                multiples = [multiples[0], multiples[1] // 16, multiples[2], multiples[3], 16]
            else:
                # change multiples from (1, H, W, C) to (1, C1, H, W, C0)
                multiples = [multiples[0], multiples[3] // 16, multiples[1], multiples[2], 16]

        if len(shape) > len(multiples):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_012
            error_info['op_name'] = 'tile_d'
            error_info['param_name'] = 'shape'
            error_info['max_value'] = str(len(multiples))
            error_info['min_value'] = '1'
            error_info['real_value'] = str(len(shape))
            raise RuntimeError(error_info, "In op[%s], the num of dimensions of input[%s] should be in the range of "
                                           "[%s, %s], but actually is [%s]." % (error_info['op_name'],
                                                                                error_info['param_name'],
                                                                                error_info['min_value'],
                                                                                error_info['max_value'],
                                                                                error_info['real_value']))
        if len(shape) < len(multiples):
            len_error = len(multiples) - len(shape)
            shape = [1] * len_error + shape

        shape_adapt = []
        multiples_adapt = []
        for i, shape_i in enumerate(shape):
            multiples_i = multiples[i]
            if multiples_i != 1 and shape_i != 1:
                shape_adapt.append(1)
                multiples_adapt.append(multiples_i)
                multiples_i = 1
            shape_adapt.append(shape_i)
            multiples_adapt.append(multiples_i)

        shape = shape_adapt
        multiples = multiples_adapt

        for shape_i, multiples_i in zip(shape, multiples):
            if not (shape_i == 1 or multiples_i == 1):
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_009
                error_info['op_name'] = 'tile_d'
                error_info['rule_desc'] = "Any axis of either shape or multiples have to be 1"
                error_info['param_name1'] = 'shape_i'
                error_info['param_name2'] = 'multiples_i'
                error_info['param1_value'] = str(shape_i)
                error_info['param2_value'] = str(multiples_i)
                raise RuntimeError(error_info, "Op[%s] has rule: %s, but [%s] is [%s], [%s] is [%s]." % (
                    error_info['op_name'], error_info['rule_desc'], error_info['param_name1'],
                    error_info['param1_value'], error_info['param_name2'], error_info['param2_value']))

        axis_not_multiple = 0
        for multiples_i in multiples:
            if multiples_i == 1:
                axis_not_multiple += 1
        if axis_not_multiple == len(multiples):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_005
            error_info['op_name'] = 'tile_d'
            error_info['param_name'] = 'axis_not_multiple'
            error_info['min_len'] = '1'
            error_info['max_len'] = str(len(multiples) - 1)
            error_info['length'] = str(axis_not_multiple)
            raise RuntimeError(error_info, "In op[%s], the length of parameter[%s] be in the range of [%s, %s], but "
                                           "actually is [%s]." % (error_info['op_name'], error_info['param_name'],
                                                                  error_info['min_len'], error_info['max_len'],
                                                                  error_info['length']))

        data = tvm.placeholder(shape, name="data", dtype=dtype)

        res = tile_d_compute(data, output_x, multiples, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": [data, res]}

    te.lang.dynamic.build(sch, config)
    te.op.add_compile_info("_unknown_shape", unkown_shape)
    te.op.add_compile_info("_origin_multiples", origin_multiples)
    te.op.add_compile_info("_multiples_adapt", multiples_adapt)
