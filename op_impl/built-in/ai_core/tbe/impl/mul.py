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
mul
"""
import te.lang.cce
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from te import platform as tbe_platform
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils import op_utils
from topi import generic
from topi.cce import util

SIZE_SIXTEEN = 16


def _can_division_sixteen(shape):

    if len(shape) < 2:
        if shape[-1] == 0:
            raise RuntimeError("value of shape is illegal")
        return False

    if shape[-1] == 0 or shape[-2] == 0:
        raise RuntimeError("value of shape is illegal")

    if shape[-1] % SIZE_SIXTEEN == 0 and shape[-2] % SIZE_SIXTEEN == 0:
        return True

    return False


def _broadcast_zn_rule(shape0, shape1, format0, format1):

    if format1 != format0:
        raise RuntimeError("format should be same")

    if len(shape0) != len(shape1) != 4:
        raise RuntimeError("length of shapes should be 4")

    x_cdim = shape0[format0.index("C")]
    x_wdim = shape0[format0.index("W")]
    x_hdim = shape0[format0.index("H")]
    x_ndim = shape0[format0.index("N")]
    y_cdim = shape1[format1.index("C")]
    y_wdim = shape1[format1.index("W")]
    y_hdim = shape1[format1.index("H")]
    y_ndim = shape1[format1.index("N")]

    x_c0 = 16
    x_n0 = 16
    x_c1 = x_cdim // 16
    x_n1 = x_ndim // 16
    shape0_zn = [x_hdim*x_wdim*x_c1, x_n1, x_n0, x_c0]

    y_c0 = 16
    y_n0 = 16
    y_c1 = y_cdim // 16
    y_n1 = y_ndim // 16
    shape1_zn = [y_hdim*y_wdim*y_c1, y_n1, y_n0, y_c0]

    if len(shape0_zn) < len(shape1_zn):
        shape0_zn, shape1_zn = shape1_zn, shape0_zn

    output_shape_len = len(shape0_zn)
    dec = output_shape_len - len(shape1_zn)
    for i in range(dec):
        shape1_zn = [1] + shape1_zn

    for i in range(output_shape_len):
        if (shape0_zn[i] != shape1_zn[i]) and (shape0_zn[i] != 1) and (shape1_zn[i] != 1):
            return False

    return True


# pylint: disable=unused-argument,invalid-name,too-many-boolean-expressions
# pylint: disable=chained-comparison,too-many-locals
# pylint: disable=locally-disabled,too-many-arguments
def op_select_format(x, y, output, kernel_name="mul"):
    """
    select format dynamically
    """
    shape_x = x.get("ori_shape")
    shape_y = y.get("ori_shape")

    shape_x = util.scalar2tensor_one(shape_x)
    shape_y = util.scalar2tensor_one(shape_y)

    format_4d_list = ["NCHW", "NHWC", "HWCN"]
    dtype_list = ["float16", "float", "int32", "int16"]
    vmul_support_s16 = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.vmul", "int16")
    vmul_support_fp32 = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.vmul", "float32")
    if not vmul_support_s16:
        dtype_list.remove("int16")
    if not vmul_support_fp32:
        dtype_list.remove("float")

    format_x = x.get("ori_format")
    format_y = y.get("ori_format")

    dtype_total = []
    format_nd = ["ND"]
    format_list = ["ND"]
    format_nz = ["FRACTAL_NZ"]
    len_format_list = len(dtype_list)

    if len(shape_x) == 4 and len(shape_y) == 4 and \
            format_x in format_4d_list and format_y in format_4d_list:
        x_cdim = shape_x[format_x.index("C")]
        x_wdim = shape_x[format_x.index("W")]
        x_hdim = shape_x[format_x.index("H")]
        x_ndim = shape_x[format_x.index("N")]
        y_cdim = shape_y[format_y.index("C")]
        y_wdim = shape_y[format_y.index("W")]
        y_hdim = shape_y[format_y.index("H")]
        y_ndim = shape_y[format_y.index("N")]
    if (len(shape_y) == 1 and len(shape_x) == 4) and \
            format_x in format_4d_list:
        x_cdim = shape_x[format_x.index("C")]
        x_ndim = shape_x[format_x.index("N")]
    if (len(shape_x) == 1 and len(shape_y) == 4) and \
            format_y in format_4d_list:
        y_cdim = shape_y[format_y.index("C")]
        y_ndim = shape_y[format_y.index("N")]

    # ND+ND NZ+NZ 5HD+5HD FZ+FZ
    if len(shape_x) >= 2 and len(shape_y) >= 2 and shape_x[-2:] == shape_y[-2:]:
        format_list.append("FRACTAL_NZ")
        if len(shape_x) == 4 and len(shape_y) == 4 and \
                format_x in format_4d_list and format_y in format_4d_list:
            if x_cdim % 16 == 0 and y_cdim % 16 == 0:
                if format_x == format_y == "NCHW" and \
                        (x_cdim == y_cdim or x_cdim // 16 == 1 or y_cdim // 16 == 1) and \
                        (x_ndim == y_ndim or x_ndim == 1 or y_ndim == 1):
                    format_list.append("NC1HWC0")
                if format_x == format_y == "HWCN":
                    if x_hdim == y_hdim and (x_wdim == 1 or y_wdim == 1):
                        format_list.append("NC1HWC0")
                    if x_wdim == y_wdim and (x_hdim == 1 or y_hdim == 1):
                        format_list.append("NC1HWC0")
                    if x_wdim == y_wdim and x_hdim == y_hdim:
                        format_list.append("NC1HWC0")
                    if (x_wdim == x_hdim == 1) or (y_hdim == y_wdim == 1):
                        format_list.append("NC1HWC0")
                    if (x_hdim == y_wdim == 1) or (x_wdim == y_hdim == 1):
                        format_list.append("NC1HWC0")
                if format_x == format_y == "NHWC":
                    if x_hdim == y_hdim and (x_ndim == 1 or y_ndim == 1):
                        format_list.append("NC1HWC0")
                    if x_ndim == y_ndim and (x_hdim == 1 or y_hdim == 1):
                        format_list.append("NC1HWC0")
                    if x_ndim == y_ndim and x_hdim == y_hdim:
                        format_list.append("NC1HWC0")
                    if (x_ndim == x_hdim == 1) or (y_ndim == y_hdim == 1):
                        format_list.append("NC1HWC0")
                    if (x_ndim == 1 and y_hdim == 1) or (x_hdim == 1 and y_ndim == 1):
                        format_list.append("NC1HWC0")
            if x_cdim % 16 == 0 and y_cdim % 16 == 0 and \
                    y_ndim % 16 == 0 and x_ndim % 16 == 0:
                if format_x == format_y and \
                        _broadcast_zn_rule(shape_x, shape_y,
                                           format_x, format_y):
                    format_list.append("FRACTAL_Z")
            if list(shape_x) == list(shape_y) and -1 not in shape_x:
                format_list.append("NC1HWC0")

        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype]*len(format_list)
        format_list = format_list*len_format_list
        unknownshape_format_list = ["ND"]*len(format_list)
        if -1 in shape_x or -1 in shape_y:
            input0 = gen_param(
                classify="input0", name="x1",
                datatype=",".join(dtype_total),
                format=",".join(format_list),
                unknownshape_format=",".join(unknownshape_format_list))
            input1 = gen_param(
                classify="input1", name="x2",
                datatype=",".join(dtype_total),
                format=",".join(format_list),
                unknownshape_format=",".join(unknownshape_format_list))
            output0 = gen_param(
                classify="output0", name="y",
                datatype=",".join(dtype_total),
                format=",".join(format_list),
                unknownshape_format=",".join(unknownshape_format_list))
        else:
            input0 = gen_param(classify="input0", name="x1",
                               datatype=",".join(dtype_total),
                               format=",".join(format_list))
            input1 = gen_param(classify="input1", name="x2",
                               datatype=",".join(dtype_total),
                               format=",".join(format_list))
            output0 = gen_param(classify="output0", name="y",
                                datatype=",".join(dtype_total),
                                format=",".join(format_list))

    # NZ+ND,ND+ND,5HD+5HD,FZ+FZ,ND+NZ
    elif len(shape_x) >= 2 and len(shape_y) >= 2 and \
            ((_can_division_sixteen(shape_x) and
              not _can_division_sixteen(shape_y)) or
             (not _can_division_sixteen(shape_x) and
              _can_division_sixteen(shape_y))):
        if len(shape_x) == 4 and len(shape_y) == 4 and \
                format_x in format_4d_list and format_y in format_4d_list:
            if x_cdim % 16 == 0 and y_cdim % 16 == 0:
                if x_cdim == y_cdim or x_cdim // 16 == 1 or y_cdim // 16 == 1:
                    format_list.append("NC1HWC0")
            if x_cdim % 16 == 0 and x_ndim % 16 == 0 and \
                    y_cdim % 16 == 0 and y_ndim % 16 == 0:
                if format_x == format_y and \
                        _broadcast_zn_rule(shape_x, shape_y,
                                           format_x, format_y):
                    format_list.append("FRACTAL_Z")

        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype]*len(format_list)
        format_list = format_list*len_format_list
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype]*1
        format_list0 = format_list + format_nz*len_format_list
        format_list1 = format_list + format_nd*len_format_list
        unknownshape_format_list = ["ND"]*len(dtype_total)
        if _can_division_sixteen(shape_x) and not _can_division_sixteen(shape_y):
            if -1 in shape_x or -1 in shape_y:
                input0 = gen_param(
                    classify="input0", name="x1",
                    datatype=",".join(dtype_total),
                    format=",".join(format_list0),
                    unknownshape_format=",".join(unknownshape_format_list))
                input1 = gen_param(
                    classify="input1", name="x2",
                    datatype=",".join(dtype_total),
                    format=",".join(format_list1),
                    unknownshape_format=",".join(unknownshape_format_list))
                output0 = gen_param(
                    classify="output0", name="y",
                    datatype=",".join(dtype_total),
                    format=",".join(format_list0),
                    unknownshape_format=",".join(unknownshape_format_list))
            else:
                input0 = gen_param(classify="input0", name="x1",
                                   datatype=",".join(dtype_total),
                                   format=",".join(format_list0))
                input1 = gen_param(classify="input1", name="x2",
                                   datatype=",".join(dtype_total),
                                   format=",".join(format_list1))
                output0 = gen_param(classify="output0", name="y",
                                    datatype=",".join(dtype_total),
                                    format=",".join(format_list0))
        else:
            if -1 in shape_x or -1 in shape_y:
                input0 = gen_param(
                    classify="input0", name="x1",
                    datatype=",".join(dtype_total),
                    format=",".join(format_list1),
                    unknownshape_format=",".join(unknownshape_format_list))
                input1 = gen_param(
                    classify="input1", name="x2",
                    datatype=",".join(dtype_total),
                    format=",".join(format_list0),
                    unknownshape_format=",".join(unknownshape_format_list))
                output0 = gen_param(
                    classify="output0", name="y",
                    datatype=",".join(dtype_total),
                    format=",".join(format_list1),
                    unknownshape_format=",".join(unknownshape_format_list))
            else:
                input0 = gen_param(classify="input0", name="x1",
                                   datatype=",".join(dtype_total),
                                   format=",".join(format_list1))
                input1 = gen_param(classify="input1", name="x2",
                                   datatype=",".join(dtype_total),
                                   format=",".join(format_list0))
                output0 = gen_param(classify="output0", name="y",
                                    datatype=",".join(dtype_total),
                                    format=",".join(format_list1))

    # 5HD+scalar,ND+ND,FZ+scalar,6D+scalar
    elif len(shape_x) >= 2 and len(shape_y) == 1 and shape_y[0] == 1:
        if len(shape_x) == 4 and len(shape_y) == 1 and format_x in format_4d_list:
            format_list.append("C1HWNCoC0")
            format_list.append("NC1HWC0")
            if x_cdim % 16 == 0 and x_ndim % 16 == 0:
                format_list.append("FRACTAL_Z")
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype]*len(format_list)
        format_list = format_list*len_format_list
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype]*1
        format_list0 = format_list + format_nd*len_format_list
        format_list1 = format_nd*len(format_list) + format_nd*len_format_list
        unknownshape_format_list = ["ND"]*len(dtype_total)
        if -1 in shape_x or -1 in shape_y:
            input0 = gen_param(
                classify="input0", name="x1",
                datatype=",".join(dtype_total),
                format=",".join(format_list0),
                unknownshape_format=",".join(unknownshape_format_list))
            input1 = gen_param(
                classify="input1", name="x2",
                datatype=",".join(dtype_total),
                format=",".join(format_list1),
                unknownshape_format=",".join(unknownshape_format_list))
            output0 = gen_param(
                classify="output0", name="y",
                datatype=",".join(dtype_total),
                format=",".join(format_list0),
                unknownshape_format=",".join(unknownshape_format_list))
        else:
            input0 = gen_param(classify="input0", name="x1",
                               datatype=",".join(dtype_total),
                               format=",".join(format_list0))
            input1 = gen_param(classify="input1", name="x2",
                               datatype=",".join(dtype_total),
                               format=",".join(format_list1))
            output0 = gen_param(classify="output0", name="y",
                                datatype=",".join(dtype_total),
                                format=",".join(format_list0))

    # ND+ND,scalar+5HD,scalar+FZ,scalar+6D
    elif len(shape_y) >= 2 and len(shape_x) == 1 and shape_x[0] == 1:
        if len(shape_x) == 1 and len(shape_y) == 4 and format_y in format_4d_list:
            format_list.append("C1HWNCoC0")
            format_list.append("NC1HWC0")
            if y_cdim % 16 == 0 and y_ndim % 16 == 0:
                format_list.append("FRACTAL_Z")
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype]*len(format_list)
        format_list = format_list*len_format_list
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype]*1
        format_list0 = format_list + format_nd*len_format_list
        format_list1 = format_nd*len(format_list) + format_nd*len_format_list
        unknownshape_format_list = ["ND"]*len(dtype_total)
        if -1 in shape_x or -1 in shape_y:
            input0 = gen_param(
                classify="input0", name="x1",
                datatype=",".join(dtype_total),
                format=",".join(format_list1),
                unknownshape_format=",".join(unknownshape_format_list))
            input1 = gen_param(
                classify="input1", name="x2",
                datatype=",".join(dtype_total),
                format=",".join(format_list0),
                unknownshape_format=",".join(unknownshape_format_list))
            output0 = gen_param(
                classify="output0", name="y",
                datatype=",".join(dtype_total),
                format=",".join(format_list0),
                unknownshape_format=",".join(unknownshape_format_list))
        else:
            input0 = gen_param(classify="input0", name="x1",
                               datatype=",".join(dtype_total),
                               format=",".join(format_list1))
            input1 = gen_param(classify="input1", name="x2",
                               datatype=",".join(dtype_total),
                               format=",".join(format_list0))
            output0 = gen_param(classify="output0", name="y",
                                datatype=",".join(dtype_total),
                                format=",".join(format_list0))
    # ND+ND,5HD+5HD
    else:
        if len(shape_x) == 1 and len(shape_y) == 1 and \
                shape_x[0] % 16 == 0 and shape_y[0] % 16 == 0:
            format_list.append("NC1HWC0")
        if len(shape_x) == 1 and len(shape_y) == 4:
            if format_x == format_y and format_y in ("NHWC", ):
                format_list.append("NC1HWC0")
            if format_x == format_y and format_y in ("NCHW", "HWCN"):
                if y_cdim == shape_x[0] or y_cdim == 1 or shape_x[0] == 1 \
                    or y_cdim // 16 == 1 or shape_x[0] // 16 == 1:
                    format_list.append("NC1HWC0")
        if len(shape_y) == 1 and len(shape_x) == 4:
            if format_x == format_y and format_x in ("NHWC", ):
                format_list.append("NC1HWC0")
            if format_x == format_y and format_x in ("NCHW", "HWCN"):
                if x_cdim == shape_y[0] or x_cdim == 1 or shape_y[0] == 1 \
                    or x_cdim // 16 == 1 or shape_y[0] // 16 == 1:
                    format_list.append("NC1HWC0")
        if len(shape_x) == 4 and len(shape_y) == 4 and format_x in format_4d_list \
            and format_y in format_4d_list:
            if format_x == format_y == "NCHW" or format_x == format_y == "HWCN" \
                    or format_x == format_y == "NHWC":
                if x_cdim % 16 == 0 and y_cdim % 16 == 0:
                    if (x_cdim // 16 == 1 or y_cdim // 16 == 1) or (x_cdim == y_cdim):
                        if x_ndim == y_ndim:
                            if x_hdim == y_hdim and (x_wdim == 1 or y_wdim == 1):
                                format_list.append("NC1HWC0")
                            if x_wdim == y_wdim and (x_hdim == 1 or y_hdim == 1):
                                format_list.append("NC1HWC0")
                            if x_hdim == y_hdim and x_wdim == y_wdim:
                                format_list.append("NC1HWC0")
                            if (x_wdim == x_hdim == 1) or (y_wdim == y_hdim == 1):
                                format_list.append("NC1HWC0")
                            if (x_hdim == 1 and y_wdim == 1) or (x_wdim == 1 and y_hdim == 1):
                                format_list.append("NC1HWC0")
                        if x_hdim == y_hdim:
                            if x_ndim == y_ndim and (x_wdim == 1 or y_wdim == 1):
                                format_list.append("NC1HWC0")
                            if x_wdim == y_wdim and (x_ndim == 1 or y_ndim == 1):
                                format_list.append("NC1HWC0")
                            if x_ndim == y_ndim and x_wdim == y_wdim:
                                format_list.append("NC1HWC0")
                            if (x_ndim == x_wdim == 1) or (y_ndim == y_wdim == 1):
                                format_list.append("NC1HWC0")
                            if (x_ndim == 1 and y_wdim == 1) or (x_wdim == 1 and y_ndim == 1):
                                format_list.append("NC1HWC0")
                        if x_wdim == y_wdim:
                            if x_ndim == y_ndim and (x_hdim == 1 or y_hdim == 1):
                                format_list.append("NC1HWC0")
                            if x_hdim == y_hdim and (x_ndim == 1 or y_ndim == 1):
                                format_list.append("NC1HWC0")
                            if x_ndim == y_ndim and x_hdim == y_hdim:
                                format_list.append("NC1HWC0")
                            if (x_ndim == x_hdim == 1) or (y_ndim == y_hdim == 1):
                                format_list.append("NC1HWC0")
                            if (x_ndim == 1 and y_hdim == 1) or (x_hdim == 1 and y_ndim == 1):
                                format_list.append("NC1HWC0")
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype]*len(format_list)
        len_format_list = len(dtype_list)
        format_list = format_list*len_format_list
        unknownshape_format_list = ["ND"]*len(dtype_total)
        if -1 in shape_x or -1 in shape_y:
            input0 = gen_param(
                classify="input0", name="x1",
                datatype=",".join(dtype_total),
                format=",".join(format_list),
                unknownshape_format=",".join(unknownshape_format_list))
            input1 = gen_param(
                classify="input1", name="x2",
                datatype=",".join(dtype_total),
                format=",".join(format_list),
                unknownshape_format=",".join(unknownshape_format_list))
            output0 = gen_param(
                classify="output0", name="y",
                datatype=",".join(dtype_total),
                format=",".join(format_list),
                unknownshape_format=",".join(unknownshape_format_list))
        else:
            input0 = gen_param(classify="input0", name="x1",
                               datatype=",".join(dtype_total),
                               format=",".join(format_list))
            input1 = gen_param(classify="input1", name="x2",
                               datatype=",".join(dtype_total),
                               format=",".join(format_list))
            output0 = gen_param(classify="output0", name="y",
                                datatype=",".join(dtype_total),
                                format=",".join(format_list))

    param_list = [input0, input1, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def _mul_check_format(x, y):
    format_pattern = 0
    shape1 = x.get("shape")
    shape2 = y.get("shape")
    list_format = [x.get("format"), y.get("format")]
    shape1 = util.scalar2tensor_one(shape1)
    shape2 = util.scalar2tensor_one(shape2)
    check_list = [["FRACTAL_NZ", "ND"],
                  ["ND", "FRACTAL_NZ"],
                  ["FRACTAL_NZ", "NHWC"],
                  ["NHWC", "FRACTAL_NZ"],
                  ["FRACTAL_NZ", "NCHW"],
                  ["NCHW", "FRACTAL_NZ"]]
    if list_format == check_list[0] \
            and (len(shape2) != 1 or (len(shape2) == 1 and shape2[0] != 1)):
        format_pattern = 1
    elif list_format == check_list[1] \
            and (len(shape1) != 1 or (len(shape1) == 1 and shape1[0] != 1)):
        format_pattern = 2
    elif list_format == check_list[2] \
            and (len(shape2) != 1 or (len(shape2) == 1 and shape2[0] != 1)):
        format_pattern = 1
    elif list_format == check_list[3] \
            and (len(shape1) != 1 or (len(shape1) == 1 and shape1[0] != 1)):
        format_pattern = 2
    elif list_format == check_list[4] \
            and (len(shape2) != 1 or (len(shape2) == 1 and shape2[0] != 1)):
        format_pattern = 1
    elif list_format == check_list[5] \
            and (len(shape1) != 1 or (len(shape1) == 1 and shape1[0] != 1)):
        format_pattern = 2

    return format_pattern


# pylint: disable=unused-variable
def _infer_shape(format_pattern, x, y):
    shape_x = x.get("shape")
    shape_y = y.get("shape")
    ori_shape_x = x.get("ori_shape")
    ori_shape_y = y.get("ori_shape")
    shape_x = util.scalar2tensor_one(shape_x)
    shape_y = util.scalar2tensor_one(shape_y)
    if format_pattern == 1:
        ori_shape_x, shape_y, shape_max = op_utils.broadcast_shapes(
            ori_shape_x, shape_y, param_name_input1="x", param_name_input2="y")
        if shape_y[-2] == ori_shape_x[-2] and shape_y[-1] == ori_shape_x[-1]:
            raise RuntimeError("the inputshape of y is illegal")

        if shape_y[-2] == 1 and shape_y[-1] == ori_shape_x[-1]:
            shape_y.append(1)
            shape_y.append(1)
            shape_y[-3] = 1
            shape_y[-1] = shape_x[-1]
            shape_y[-4] = shape_x[-4]

        elif shape_y[-2] == ori_shape_x[-2] and shape_y[-1] == 1:
            shape_y.append(1)
            shape_y.append(1)
            shape_y[-4] = 1
            shape_y[-2] = shape_x[-2]
            shape_y[-3] = shape_x[-3]

        elif shape_y[-2] == shape_y[-1] == 1:
            shape_y.append(1)
            shape_y.append(1)

    elif format_pattern == 2:
        shape_x, ori_shape_y, shape_max = op_utils.broadcast_shapes(
            shape_x, ori_shape_y, param_name_input1="x", param_name_input2="y")
        if shape_x[-2] == ori_shape_y[-2] and shape_x[-1] == ori_shape_y[-1]:
            raise RuntimeError("the inputshape of x is illegal")

        if shape_x[-2] == 1 and shape_x[-1] == ori_shape_y[-1]:
            shape_x.append(1)
            shape_x.append(1)
            shape_x[-3] = 1
            shape_x[-1] = shape_y[-1]
            shape_x[-4] = shape_y[-4]

        elif shape_x[-2] == ori_shape_y[-2] and shape_x[-1] == 1:
            shape_x.append(1)
            shape_x.append(1)
            shape_x[-4] = 1
            shape_x[-2] = shape_y[-2]
            shape_x[-3] = shape_y[-3]

        elif shape_x[-2] == shape_x[-1] == 1:
            shape_x.append(1)
            shape_x.append(1)

    return shape_x, shape_y


@fusion_manager.register("mul")
def _mul_compute(input_x, input_y, output_data, kernel_name="mul"):
    """
    calculating element-wise mul

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of first input data
    input_y: TVM tensor
        the placeholder of second input data
    output_data: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is "mul"

    Returns
    -------
    output of the element-wise mul
    """
    shape_x = te.lang.cce.util.shape_to_list(input_x.shape)
    shape_y = te.lang.cce.util.shape_to_list(input_y.shape)

    shape_x, shape_y, shape_max = op_utils.broadcast_shapes(
        shape_x, shape_y, param_name_input1="x", param_name_input2="y")
    if shape_x != shape_y and len(shape_x) == 2 and len(shape_y) == 2:
        res = _mul_compute_ex(input_x, input_y, shape_x, shape_y, shape_max)
        if res is not None:
            return res
    input_x = te.lang.cce.broadcast(input_x, shape_max)
    input_y = te.lang.cce.broadcast(input_y, shape_max)
    res = te.lang.cce.vmul(input_x, input_y)

    return res


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
def _mul_compute_ex(input_x, input_y, shape_x, shape_y, shape_max):
    if shape_x == shape_max:
        small_input = input_y
        large_input = input_x
    elif shape_y == shape_max:
        small_input = input_x
        large_input = input_y
    else:
        return None

    small_index = []
    small_shape = 1
    for i in range(len(small_input.shape)):
        if int(small_input.shape[i]) < int(shape_max[i]):
            small_index.append(i)
            small_shape *= shape_max[i]
        elif int(small_input.shape[i]) == int(shape_max[i]):
            pass
        else:
            return None

    if small_shape < 10000:
        return None

    if int(small_input.shape[-1]) != 1:
        return None

    def get_tensor_slice(inp, small_index, is_large, *shapes):
        def get_index(inp_tensor, index):
            return inp_tensor[index]

        if is_large:
            for axis in shapes:
                inp = get_index(inp, axis)
        else:
            for ind, _ in enumerate(shapes):
                if ind in small_index:
                    inp = get_index(inp, 0)
                else:
                    inp = get_index(inp, shapes[ind])

        return inp

    with tvm.tag_scope("elewise_binary_mul"):
        res = tvm.compute(shape_max,
                          lambda *indices: get_tensor_slice(large_input,
                                                            small_index,
                                                            True,
                                                            *indices)
                          * get_tensor_slice(small_input,
                                             small_index,
                                             False,
                                             *indices),
                          name="manual_mul_without_broadcast_"
                          + str(te.lang.cce.te_compute.elewise_compute.NAME_INDEX[0]))
    te.lang.cce.te_compute.elewise_compute.NAME_INDEX[0] += 1

    return res

# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=too-many-locals
@op_utils.check_op_params(op_utils.REQUIRED_INPUT, op_utils.REQUIRED_INPUT,
                          op_utils.REQUIRED_OUTPUT, op_utils.KERNEL_NAME)
def mul(x, y, output, kernel_name="mul"):
    """
    do element-wise mul operation between two input tensors

    Parameters:
    ----------
    x : dict.
        shape, dtype of input x
    y : dict.
        shape, dtype of input y
    output : dict.
        shape, dtype of ouput
    kernel_name : str.
        cce kernel name, default value is "mul"

    Returns
    -------
    None
    """
    # format_pattern = 1  Nz and vector
    # format_pattern = 2  vector and Nz
    # format_pattern = 0  Nz scalar  Nz Nz  ND ND
    format_pattern = _mul_check_format(x, y)
    shape_x, shape_y = _infer_shape(format_pattern, x, y)

    shape_x = util.scalar2tensor_one(shape_x)
    dtype_x = x.get("dtype").lower()
    shape_y = util.scalar2tensor_one(shape_y)
    dtype_y = y.get("dtype").lower()

    op_utils.check_shape(shape_x, param_name="x")
    op_utils.check_shape(shape_y, param_name="y")

    if dtype_x != dtype_y:
        raise RuntimeError("dtype of inputs should be consistent")
    dtype = dtype_x
    check_list = ("int32", "float16", "float32", "int16")
    op_utils.check_dtype(dtype, check_list, param_name="x")

    vmul_support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.vmul", "float32")
    if dtype_x == "float32" and not vmul_support:
        raise RuntimeError(
            "Input dtype is float32, but do not support on the platform")

    shape_x, shape_y, shape_max = op_utils.broadcast_shapes(
        shape_x, shape_y, param_name_input1="x", param_name_input2="y")

    shape_x, shape_y = op_utils.refine_shapes_for_broadcast(shape_x, shape_y)
    input_x = tvm.placeholder(shape_x, dtype=dtype, name="x")
    input_y = tvm.placeholder(shape_y, dtype=dtype, name="y")

    res = _mul_compute(input_x, input_y, output, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": (input_x, input_y, res)}
    te.lang.cce.cce_build_code(sch, config)
