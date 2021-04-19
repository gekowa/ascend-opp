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
normalize_scale
"""
import te.lang.cce
from te import tvm
from te import platform as tbe_platform
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import *
import te.utils.op_utils as op_utils


# pylint: disable=locally-disabled,unused-argument,invalid-name,too-many-locals
# pylint: disable=locally-disabled,too-many-arguments,protected-access
# pylint: disable=locally-disabled,too-many-branches
@fusion_manager.register("normalize_scale")
def normalize_scale_compute(x1, x2, x3, y,
                            across_spatial=True, eps=1e-10,
                            kernel_name="normalize_scale"):
    """
    calculating data

    Parameters
    ----------
    x1 : TVM tensor
        the placeholder of x1
    x2 : TVM tensor
        the placeholder of x2
    x3 : TVM tensor
        the placeholder of x3
    y : dict
        dict of y, include keys(shape and dtype, format)
    across_spatial: bool
        indicates whether standardization should cross spatial locations.
        Default(True)
    eps: float
        prevent dividing by 0.
        Default(1e-10)
    kernel_name : str
        kernel name, default value is "normalize_scale"

    Returns
    -------
    output tensor
    """

    # set intermediate dtype
    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS"):
        # hisi es, cs
        intermediate_dtype = "float16"
        dtype_cast_mapping = {"int8": "float16"}
        dtype_reverse_cast_mapping = {"float16": "int8"}
    else:
        # mini, cloud
        intermediate_dtype = "float32"
        dtype_cast_mapping = {"int8": "float16", "float16": "float32"}
        dtype_reverse_cast_mapping = {"float16": "int8",
                                      "float32": "float16"}

    x1_shape = te.lang.cce.util.shape_to_list(x1.shape)

    x1_cast = x1
    while x1_cast.dtype in dtype_cast_mapping:
        x1_cast = te.lang.cce.cast_to(x1_cast,
                                      dtype_cast_mapping[x1_cast.dtype])
    x2_cast = x2
    while x2_cast.dtype in dtype_cast_mapping:
        x2_cast = te.lang.cce.cast_to(x2_cast,
                                      dtype_cast_mapping[x2_cast.dtype])

    x3_cast = x3
    while x3_cast.dtype in dtype_cast_mapping:
        x3_cast = te.lang.cce.cast_to(x3_cast,
                                      dtype_cast_mapping[x3_cast.dtype])

    x1_sqr_sum = te.lang.cce.vadds(x3_cast,
                                   tvm.const(eps, dtype=intermediate_dtype))

    x2_cast_broadcast = te.lang.cce.broadcast(x2_cast, x1_shape)

    x1_scaled = te.lang.cce.vmul(x1_cast, x2_cast_broadcast)

    if cce_product in ("Ascend910", "Hi3796CV300ES", "Hi3796CV300CS", \
                       "Ascend610", "Ascend710"):
        x1_sqr_sum_sqrt = te.lang.cce.vsqrt(x1_sqr_sum)
        x1_sqr_sum_sqrt_broadcast = te.lang.cce.broadcast(x1_sqr_sum_sqrt,
                                                          x1_shape)
        x1_normalized = te.lang.cce.vdiv(x1_scaled, x1_sqr_sum_sqrt_broadcast)
    elif cce_product in ("Ascend310",):
        # customized for mini, using newton
        x1_sqr_sum_sqrt = te.lang.cce.vsqrt(x1_sqr_sum)

        for _ in range(1):
            res = te.lang.cce.vdiv(x1_sqr_sum, x1_sqr_sum_sqrt)
            res = te.lang.cce.vadd(res, x1_sqr_sum_sqrt)
            res = te.lang.cce.vmuls(res, tvm.const(0.5, intermediate_dtype))
            x1_sqr_sum_sqrt = res
        x1_sqr_sum_rsqrt = te.lang.cce.vrec(x1_sqr_sum_sqrt)
        x1_sqr_sum_rsqrt_broadcast = te.lang.cce.broadcast(x1_sqr_sum_rsqrt,
                                                           x1_shape)
        x1_normalized = te.lang.cce.vmul(x1_scaled, x1_sqr_sum_rsqrt_broadcast)
    else:
        # for mini and hisi-es
        x1_sqr_sum_rsqrt = te.lang.cce.vrsqrt(x1_sqr_sum)
        x1_sqr_sum_rsqrt_broadcast = te.lang.cce.broadcast(x1_sqr_sum_rsqrt,
                                                           x1_shape)
        x1_normalized = te.lang.cce.vmul(x1_scaled, x1_sqr_sum_rsqrt_broadcast)

    x1_normalized_cast = x1_normalized
    while x1_normalized_cast.dtype != x1.dtype and \
            x1_normalized_cast.dtype in dtype_reverse_cast_mapping:
        x1_normalized_cast = te.lang.cce.cast_to(x1_normalized_cast,
                                                 dtype_reverse_cast_mapping[
                                                     x1_normalized_cast.dtype])

    return x1_normalized_cast


def check_format(data_format, data_format_3):
    """
    check the format for x1 and x3

    Parameters
    ----------
    data_format : str
        the format for x1
    data_format_3 : str
        the format for x3

    Returns
    -------
    None
    """

    if data_format != data_format_3:
        error_info = {}
        error_info['errCode'] = 'E80019'
        error_info['param_name1'] = 'data_format'
        error_info['param_name2'] = 'data_format_3'
        error_info['op_name'] = 'normalize_scale'
        raise RuntimeError(error_info, "In op[%s], the parameter[%s][%s] is not "
                                       "equal to the parameter[%s][%s]in format."
                           % (error_info['op_name'], error_info['param_name1'],
                              data_format, error_info['param_name2'], data_format_3))

    op_utils.check_format(data_format, ("NCHW", "NHWC"), param_name="x1")


def check_dtype(dtype_1, dtype_3):
    """
    check the dtype for x1, x3

    Parameters
    ----------
    dtype_1 : str
        dtype for x1
    dtype_3 : str
        dtype for x3

    Returns
    -------
    None
    """

    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS"):
        # hisi es, cs
        op_utils.check_dtype(dtype_1, ("int8", "float16",), param_name="x1")
        op_utils.check_dtype(dtype_3, ("int8", "float16",), param_name="x3")
    else:
        op_utils.check_dtype(dtype_1, ("int8", "float16", "float32",), param_name="x1")
        op_utils.check_dtype(dtype_3, ("int8", "float16", "float32",), param_name="x3")


def check_shape_1(shape_1):
    """
    check the shape for x1

    Parameters
    ----------
    shape_1 : list or tuple
        shape for x1

    Returns
    -------
    None
    """

    op_utils.check_shape(shape_1, param_name="x1")
    op_utils.check_shape(shape_1, min_rank=4, max_rank=4, param_name="x1")


def check_shape_2(shape_1, data_format, channel_shared):
    """
    check the shape for x2

    Parameters
    ----------
    shape_1 : list or tuple
        shape for x1
    data_format : str
        format for x1
    channel_shared: bool
        used to control whether x2 are shared by multiple channels.
        Default(True)

    Returns
    -------
    the expand shape for x2, used for placeholder
    """

    if channel_shared:
        shape_2 = [1, 1, 1, 1]
    elif data_format == "NCHW":
        shape_2 = [1, shape_1[1], 1, 1]
    elif data_format == "NHWC":
        shape_2 = [1, 1, 1, shape_1[3]]

    return shape_2


def check_shape_3(shape_1, shape_3, data_format, across_spatial):
    """
    check the shape for x3

    Parameters
    ----------
    shape_1 : list or tuple
        shape for x1
    shape_3 : list or tuple
        shape for x3
    data_format : str
        format for x1 and x3
    across_spatial: bool
        indicates whether standardization should cross spatial locations.
        Default(True)

    Returns
    -------
    None
    """
    op_utils.check_shape(shape_3, param_name="x3")
    op_utils.check_shape(shape_3, min_rank=4, max_rank=4, param_name="x3")

    if across_spatial:
        if not (shape_3[0] == shape_1[0] and shape_3[1] == 1 and
                shape_3[2] == 1 and shape_3[3] == 1):
            error_info = {}
            error_info['errCode'] = 'E80017'
            error_info['param_name1'] = 'x3.shape'
            error_info['param_name2'] = 'x1.shape'
            error_info['op_name'] = 'normalize_scale'
            error_info['param1_shape1'] = shape_3
            error_info['param1_shape2'] = shape_1
            error_info['expect_shape'] = (shape_1[0], 1, 1, 1)
            raise RuntimeError(error_info, "In op[%s], the parameter[%s][%s] is not "
                                           "match with the parameter[%s][%s],it should be [%s]."
                               % (error_info['op_name'], error_info['param_name1'],
                                  error_info['param1_shape1'], error_info['param_name2'],
                                  error_info['param1_shape2'], error_info['expect_shape']))

    elif data_format == "NCHW":
        if not (shape_3[0] == shape_1[0] and shape_3[1] == 1 and
                shape_3[2] == shape_1[2] and shape_3[3] == shape_1[3]):
            error_info = {}
            error_info['errCode'] = 'E80017'
            error_info['param_name1'] = 'x3.shape'
            error_info['param_name2'] = 'x1.shape'
            error_info['op_name'] = 'normalize_scale'
            error_info['param1_shape1'] = shape_3
            error_info['param1_shape2'] = shape_1
            error_info['expect_shape'] = (shape_1[0], 1, shape_1[2], shape_1[3])
            raise RuntimeError(error_info, "In op[%s], the parameter[%s][%s] is not "
                                           "match with the parameter[%s][%s],it should be [%s]."
                               % (error_info['op_name'], error_info['param_name1'],
                                  error_info['param1_shape1'], error_info['param_name2'],
                                  error_info['param1_shape2'], error_info['expect_shape']))

    elif data_format == "NHWC":
        if not (shape_3[0] == shape_1[0] and shape_3[1] == shape_1[1] and
                shape_3[2] == shape_1[2] and shape_3[3] == 1):
            error_info = {}
            error_info['errCode'] = 'E80017'
            error_info['param_name1'] = 'x3.shape'
            error_info['param_name2'] = 'x1.shape'
            error_info['op_name'] = 'normalize_scale'
            error_info['param1_shape1'] = shape_3
            error_info['param1_shape2'] = shape_1
            error_info['expect_shape'] = (shape_1[0], shape_1[1], shape_1[2], 1)
            raise RuntimeError(error_info, "In op[%s], the parameter[%s][%s] is not "
                                           "match with the parameter[%s][%s],it should be [%s]."
                               % (error_info['op_name'], error_info['param_name1'],
                                  error_info['param1_shape1'], error_info['param_name2'],\
                                  error_info['param1_shape2'], error_info['expect_shape']))

# pylint: disable=locally-disabled,invalid-name,too-many-arguments
# pylint: disable=locally-disabled,too-many-locals
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT,
                 OPTION_ATTR_BOOL, OPTION_ATTR_BOOL, OPTION_ATTR_FLOAT, KERNEL_NAME)
def normalize_scale(x1, x2, x3, y, across_spatial=True,
                    channel_shared=True, eps=1e-10,
                    kernel_name="normalize_scale"):
    """
    calculating data

    Parameters
    ----------
    x1 : dict
        shape and dtype, format of input 1
    x2 : dict
        shape and dtype, format of input 2
    x3 : dict
        shape and dtype, format of input 3
    y : dict
        shape and dtype, format of output,
        should be same shape and type, format as input 1
    across_spatial: bool
        indicates whether standardization should cross spatial locations.
        Default(True)
    channel_shared: bool
        used to control whether x2 are shared by multiple channels.
        Default(True)
    eps: float
        prevent dividing by 0.
        Default(1e-10)
    kernel_name : str
        kernel name, default value is "normalize_scale"

    Returns
    -------
    None
    """

    shape_1 = x1.get("shape")
    dtype_1 = x1.get("dtype").lower()
    data_format = x1.get("format")

    shape_3 = x3.get("shape")
    dtype_3 = x3.get("dtype").lower()
    data_format_3 = x3.get("format")

    check_format(data_format, data_format_3)
    check_dtype(dtype_1, dtype_3)

    if len(list(shape_1)) == 2:
        if data_format == "NCHW":
            shape_1 = [shape_1[0], shape_1[1], 1, 1]
        elif data_format == "NHWC":
            shape_1 = [shape_1[0], 1, 1, shape_1[1]]

    check_shape_1(shape_1)
    check_shape_3(shape_1, shape_3, data_format, across_spatial)

    # the expand shape for x2, used for placeholder
    shape_2 = check_shape_2(shape_1, data_format, channel_shared)
    dtype_2 = dtype_1

    data_x1 = tvm.placeholder(shape_1, name="data_1", dtype=dtype_1)
    data_x2 = tvm.placeholder(shape_2, name="data_2", dtype=dtype_2)
    data_x3 = tvm.placeholder(shape_3, name="data_3", dtype=dtype_3)
    res = normalize_scale_compute(data_x1, data_x2, data_x3, y,
                                  across_spatial, eps, kernel_name)

    # pylint: disable=no-member
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "need_build": True,
              "name": kernel_name,
              "tensor_list": [data_x1, data_x2, data_x3, res]}

    te.lang.cce.cce_build_code(sch, config)
