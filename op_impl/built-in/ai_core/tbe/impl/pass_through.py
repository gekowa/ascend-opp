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
pass_through
"""
# pylint: disable=too-many-arguments,unused-argument
from topi.cce import util
from te import platform as tbe_platform
from impl import pass_through_forward
from impl import pass_through_backward
from te.utils.op_utils import *
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json

MINI_STRIDE = 1


def op_select_format(in_dic, filter_dic, out_dic,
                     stride, reverse, kernel_name="pass_through"):
    """
    select format dynamically
    """
    product_version = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if len(filter_dic['shape']) != 0:
        if product_version in ("Hi3796CV300ES", "Hi3796CV300CS"):
            dtype0 = "float16, int8, uint8, int16, uint16, int32, uint32, int64, uint64"
            dtype1 = "float16, float16, float16, float16, float16, float16, float16, float16, float16"
            dformat0 = "NC1HWC0, NHWC, NHWC, NHWC, NHWC, NHWC, NHWC, NHWC, NHWC"
            dformat1 = "FRACTAL_Z, FRACTAL_Z, FRACTAL_Z, FRACTAL_Z, FRACTAL_Z, \
                        FRACTAL_Z, FRACTAL_Z, FRACTAL_Z, FRACTAL_Z"
        else:
            dtype0 = "float16, float, int8, uint8, int16, uint16, int32, uint32, int64, uint64"
            dtype1 = "float16, float16, float16, float16, float16, float16, float16, float16, float16, float16"
            dformat0 = "NC1HWC0, NHWC, NHWC, NHWC, NHWC, NHWC, NHWC, NHWC, NHWC, NHWC"
            dformat1 = "FRACTAL_Z, FRACTAL_Z, FRACTAL_Z, FRACTAL_Z, FRACTAL_Z, \
                        FRACTAL_Z, FRACTAL_Z, FRACTAL_Z, FRACTAL_Z, FRACTAL_Z"
    else:
        if product_version in ("Hi3796CV300ES", "Hi3796CV300CS"):
            dtype0 = "float16, int8, uint8, int16, uint16, int32, uint32, int64, uint64"
            dtype1 = "float16, float16, float16, float16, float16, float16, float16, float16, float16"
            dformat0 = "NHWC, NHWC, NHWC, NHWC, NHWC, NHWC, NHWC, NHWC, NHWC"
            dformat1 = "FRACTAL_Z, FRACTAL_Z, FRACTAL_Z, FRACTAL_Z, FRACTAL_Z, \
                        FRACTAL_Z, FRACTAL_Z, FRACTAL_Z, FRACTAL_Z"
        else:
            dtype0 = "float16, float, int8, uint8, int16, uint16, int32, uint32, int64, uint64"
            dtype1 = "float16, float16, float16, float16, float16, float16, float16, float16, float16, float16"
            dformat0 = "NHWC, NHWC, NHWC, NHWC, NHWC, NHWC, NHWC, NHWC, NHWC, NHWC"
            dformat1 = "FRACTAL_Z, FRACTAL_Z, FRACTAL_Z, FRACTAL_Z, FRACTAL_Z, \
                        FRACTAL_Z, FRACTAL_Z, FRACTAL_Z, FRACTAL_Z, FRACTAL_Z"

    input0 = gen_param(classify="input0", name="x",
                        datatype=dtype0, format=dformat0)
    input1 = gen_param(classify="input1", name="filter",
                        datatype=dtype1, format=dformat1)
    output0 = gen_param(classify="output0", name="y",
                        datatype=dtype0, format=dformat0)

    param_list = [input0, input1, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def check_param(in_dic, out_dic, stride, reverse, kernel_name):
    """
    check validation of input param

    Parameters
    ----------
    in_dic : dict
        shape/dtype/fmt of input
    out_dic : dict
        shape/dtype/fmt of output, should be same shape and type as input
    stride : int32
        c/h/w stride
    reverse : bool
        forward or reverse flag
    kernel_name : str
        kernel name, default value is "pass_through"

    Returns
    -------
    None
    """
    shape_in = in_dic.get("shape")
    dtype_in = in_dic.get("dtype")
    fmt_in = in_dic.get("format")
    shape_out = out_dic.get("shape")
    dtype_out = out_dic.get("dtype")
    fmt_out = out_dic.get("format")

    check_shape(shape_in, param_name="in_dic")
    check_shape(shape_out, param_name="out_dic")
    check_dtype(dtype_in.lower(), ["float16", "float32",
                                   "int8", "uint8",
                                   "int16", "uint16",
                                   "int32", "uint32",
                                   "int64", "uint64"], param_name="in_dic")
    check_dtype(dtype_out.lower(), ["float16", "float32",
                                    "int8", "uint8",
                                    "int16", "uint16",
                                    "int32", "uint32",
                                    "int64", "uint64"], param_name="out_dic")

    if fmt_in.lower() != "nhwc" or fmt_out.lower() != "nhwc":
        error_info = {}
        error_info['errCode'] = 'E80015'
        error_info['op_name'] = 'pass_through'
        error_info['param_name1'] = 'fmt_in'
        error_info['param_name2'] = 'fmt_out'
        error_info['expect_value'] = 'NHWC'
        error_info['real_value1'] = fmt_in.lower()
        error_info['real_value2'] = fmt_out.lower()
        raise ValueError(error_info, "In op[%s], the format of [%s]/[%s] must be [%s], "
                         "but actually is [%s]/[%s]." % (error_info['op_name'], \
                         error_info['param_name1'], error_info['param_name2'],
                         error_info['expect_value'], error_info['real_value1'], error_info['real_value2']))

    if stride < MINI_STRIDE:
        error_info = {}
        error_info['errCode'] = 'E81007'
        error_info['param_name'] = 'stride'
        error_info['op_name'] = 'pass_through'
        error_info['real_value'] = stride
        raise ValueError(error_info, "In op[%s], the parameter [%s] must be greater than 0, "
                         "but actually is [%s]." % (error_info['op_name'], error_info['param_name'], \
                         error_info['real_value']))
    if reverse is True:
        if (shape_in[3] % (stride * stride)) != 0:
            error_info = {}
            error_info['errCode'] = 'E81008'
            error_info['param_name'] = 'C'
            error_info['op_name'] = 'pass_through'
            error_info['real_value'] = shape_in[3]
            error_info['expect_value'] = stride * stride
            raise ValueError(error_info, "In op[%s], the parameter [%s] must be "
                             "times of stride**2[%s], but actually is [%s]."
                             % (error_info['op_name'], error_info['param_name'], \
                                error_info['expect_value'], error_info['real_value']))

    else:
        if (shape_in[1] % stride != 0) or (shape_in[2] % stride != 0):
            error_info = {}
            error_info['errCode'] = 'E81008'
            error_info['op_name'] = 'pass_through'
            error_info['param_value1'] = shape_in[1]
            error_info['param_value2'] = shape_in[2]
            raise ValueError(error_info, "In op[%s], the parameter w/h must be "
                             "times of stride[%s], but actually is [%s]/[%s]."
                             % (error_info['op_name'], stride, error_info['param_value2'], \
                                error_info['param_value1']))


@check_op_params(REQUIRED_INPUT, OPTION_INPUT, REQUIRED_OUTPUT,
                 OPTION_ATTR_INT,
                 OPTION_ATTR_BOOL, KERNEL_NAME)
def pass_through(in_dic, filter_dic, out_dic, stride,
                 reverse, kernel_name="pass_through"):
    """
    pass_through ops interface

    Parameters
    ----------
    in_dic : dict
        shape/dtype/fmt of input
    filter_dic : dict
        shape/dtype/fmt of optional filter input
    out_dic : dict
        shape/dtype/fmt of output, should be same shape and type as input
    stride : int32
        c/h/w stride
    reverse : bool
        forward or reverse flag
    kernel_name : str
        kernel name, default value is "pass_through"

    Returns
    -------
    tik instance
    """

    check_param(in_dic, out_dic, stride, reverse, kernel_name)

    if reverse is False:
        tik_instance, input_gm, output_gm = \
            pass_through_backward.pass_through_backward_func(in_dic, stride)
    else:
        tik_instance, input_gm, output_gm = \
            pass_through_forward.pass_through_forward_func(in_dic, stride)

    tik_instance.BuildCCE(kernel_name=kernel_name,
                          inputs=input_gm,
                          outputs=output_gm)
    return tik_instance
