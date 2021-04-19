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
eltwise
"""
from functools import reduce as reduceIns

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import *
from topi import generic
from topi.cce import util
from te.platform.cce_policy import get_L1_info

SHAPE_SIZE_LIMIT = 2147483648


def get_fusion_params(x_tensor, y, x_tensor_num):
    """
    Get L1 fusion_params
    Parameters
    ----------
    x_tensor : tensor of input data
    y : dict of output data
    x_tensor_num: input tensor num
    Returns
    -------
    fusion_params
    """
    # 0: L1 depth fusion, 1: L1 width fusion, -1: no L1 fusion
    in_l1_flag_list = []
    in_valid_shape_list = []
    in_slice_offset_list = []
    in_select_read_flag_list = []
    is_l1_depth_fusion = False

    for i in range(0, x_tensor_num):
        l1_fusion_type = x_tensor[i].op.attrs["L1_fusion_type"].value \
            if "L1_fusion_type" in x_tensor[i].op.attrs else -1
        if l1_fusion_type == 1:
            raise RuntimeError("eltwise does not support l1 width fusion")
        is_l1_depth_fusion = (l1_fusion_type == 0) or is_l1_depth_fusion
        in_l1_flag = x_tensor[i].op.attrs["addr_type"].value == 1 \
            if "addr_type" in x_tensor[i].op.attrs else False
        in_l1_flag_list.append(in_l1_flag)
        in_valid_shape = x_tensor[i].op.attrs["valid_shape"] \
            if "valid_shape" in x_tensor[i].op.attrs else []
        in_valid_shape_list.append(in_valid_shape)
        in_slice_offset = x_tensor[i].op.attrs["slice_offset"] \
            if "slice_offset" in x_tensor[i].op.attrs else []
        in_slice_offset_list.append(in_slice_offset)
        in_select_read_flag = x_tensor[i].op.tag == "read_select_5d"
        in_select_read_flag_list.append(in_select_read_flag)

    l1_fusion_type = 0 if is_l1_depth_fusion is True else -1
    out_l1_flag = y.get("addr_type", 0) == 1
    out_valid_shape = y.get("valid_shape", [])
    out_slice_offset = y.get("slice_offset", [])
    out_select_write_flag = bool(out_valid_shape)

    fusion_params = {"is_l1fusion": is_l1_depth_fusion,
                     "l1_fusion_type": l1_fusion_type,
                     "in_l1_flag": in_l1_flag_list,
                     "in_select_read_flag": in_select_read_flag_list,
                     "in_valid_shape": in_valid_shape_list,
                     "in_slice_offset": in_slice_offset_list,
                     "out_l1_flag": out_l1_flag,
                     "out_select_write_flag": out_select_write_flag,
                     "out_valid_shape": out_valid_shape,
                     "out_slice_offset": out_slice_offset}
    return fusion_params


# pylint: disable=unidiomatic-typecheck,too-many-branches,too-many-locals
# pylint: disable=no-member,dangerous-default-value,invalid-name
@fusion_manager.register("eltwise")
def eltwise_compute(x, y, mode=1, coeff=[], kernel_name="eltwise"):
    '''
    Compute elementwise operation
    '''
    tensor_num = len(x)
    inp_dtype = x[0].dtype
    data0_tmp = x[0]

    tmp_y = {}
    tmp_y["addr_type"] = 0
    tmp_y["valid_shape"] = []
    tmp_y["slice_offset"] = []
    fuse_y = tmp_y if y is None else y
    fusion_params = get_fusion_params(x, fuse_y, tensor_num)

    if mode == 1:
        if len(coeff) != 0 and len(coeff) != tensor_num:

            errorInfo = {}
            errorInfo['errCode'] = "E81002"
            errorInfo['op_name'] = 'eltwise'
            errorInfo['coeff_length'] = str(len(coeff))
            errorInfo['input_num'] = str(tensor_num)
            raise RuntimeError(errorInfo, "In op[%s], the parameter[coeff]'s length[%s] should be "
                                          "equal to inputs'num[%s]." %
                               (errorInfo['op_name'], errorInfo['coeff_length'],
                                errorInfo['input_num']))
        if len(coeff) == tensor_num:
            if type(coeff[0]) != int and type(coeff[0]) != float:
                raise RuntimeError("ele of coeff must be a number.")
            if coeff[0] != 1:
                coeff1 = tvm.const(coeff[0], dtype=inp_dtype)
                data0_tmp = te.lang.cce.vmuls(data0_tmp, coeff1)

    res = None
    if tensor_num == 1:
        const_val_0 = tvm.const(0, dtype=inp_dtype)
        data0_tmp = te.lang.cce.vadds(data0_tmp, const_val_0)
        res = data0_tmp
    elif tensor_num > 1:
        for i in range(1, tensor_num):
            datan_tmp = x[i]
            if mode == 0:
                data0_tmp = te.lang.cce.vmul(data0_tmp, datan_tmp)
            elif mode == 2:
                data0_tmp = te.lang.cce.vmax(data0_tmp, datan_tmp)
            else:
                if len(coeff) == 0:
                    data0_tmp = te.lang.cce.vadd(data0_tmp, datan_tmp)
                elif coeff[i] == 1:
                    data0_tmp = te.lang.cce.vadd(data0_tmp, datan_tmp)
                else:
                    coeff2 = tvm.const(coeff[i], dtype=inp_dtype)
                    datan_tmp = te.lang.cce.vmuls(datan_tmp, coeff2)
                    data0_tmp = te.lang.cce.vadd(data0_tmp, datan_tmp)
        res = data0_tmp

    res.op.attrs["ele_fusion_params"] = fusion_params
    return res


def _eltwise_check_para(x, y, mode=1, coeff=[],
                        kernel_name="eltwise"):

    shape = x[0].get("shape")
    dtype = x[0].get("dtype").lower()
    check_shape(shape, param_name="x")

    dtype_check_list = ["float16", "float32"]
    check_dtype(dtype, dtype_check_list, param_name="x")

    tensor_num = len(x)
    if tensor_num < 1 or tensor_num > 32:
        errorInfo = {}
        errorInfo['errCode'] = OP_ERROR_CODE_002
        errorInfo['op_name'] = 'eltwise'
        errorInfo['param_name'] = 'tensor_num'
        errorInfo['min_value'] = '1'
        errorInfo['max_value'] = '32'
        errorInfo['real_value'] = tensor_num
        raise RuntimeError(errorInfo,
                           "In op[%s], the parameter[%s] should be in the range "
                           "of [%s, %s], but actually is [%s]." %
                           (errorInfo['op_name'], errorInfo['param_name'],
                            errorInfo['min_value'], errorInfo['max_value'],
                            errorInfo['real_value']))

    # all input data should be same shape and dtype
    if tensor_num > 1:
        for i in range(1, tensor_num):

            shape_tmp = x[i].get("shape")
            dtype_tmp = x[i].get("dtype").lower()

            if shape_tmp != shape:
                errorInfo = {}
                errorInfo['errCode'] = 'E80017'
                errorInfo['op_name'] = 'eltwise'
                errorInfo['param_name1'] = 'shape'
                errorInfo['param_name2'] = 'shape_tmp'
                errorInfo['param1_shape'] = ','.join(str(i) for i in shape)
                errorInfo['param2_shape'] = ','.join(str(i) for i in shape_tmp)
                raise RuntimeError(errorInfo,
                                   "In op[%s], the parameter[%s][%s] are not equal"
                                   " in shape with shapes[%s][%s]." %
                                   (errorInfo['op_name'], errorInfo['param_name1'],
                                    errorInfo['param_name2'], errorInfo['param1_shape'],
                                    errorInfo['param2_shape']))

            if dtype_tmp != dtype:
                errorInfo = {}
                errorInfo['errCode'] = 'E80018'
                errorInfo['op_name'] = 'eltwise'
                errorInfo['param_name1'] = 'dtype_tmp'
                errorInfo['param_name2'] = 'dtype'
                errorInfo['param1_shape'] = str(dtype_tmp)
                errorInfo['param2_shape'] = str(dtype)
                raise RuntimeError(errorInfo,
                                   "In op[%s], the parameter[%s][%s] are not "
                                   "equal in dtype with dtype[%s][%s]." %
                                   (errorInfo['op_name'], errorInfo['param_name1'],
                                    errorInfo['param_name2'], errorInfo['param1_shape'],
                                    errorInfo['param2_shape']))


    shape_output = y.get("shape")
    check_shape(shape_output, param_name="y")
    if shape_output != shape:
        errorInfo = {}
        errorInfo['errCode'] = 'E80017'
        errorInfo['op_name'] = 'eltwise'
        errorInfo['param_name1'] = 'shape_output'
        errorInfo['param_name2'] = 'shape'
        errorInfo['param1_shape'] = ','.join(str(i) for i in shape_output)
        errorInfo['param2_shape'] = ','.join(str(i) for i in shape)
        raise RuntimeError(errorInfo,
                           "In op[%s], the parameter[%s][%s] are not equal in"
                           " shape with shapes[%s][%s]." %
                           (errorInfo['op_name'], errorInfo['param_name1'],
                            errorInfo['param_name2'], errorInfo['param1_shape'],
                            errorInfo['param2_shape']))

    dtype_output = y.get("dtype").lower()
    if dtype_output != dtype:
        errorInfo = {}
        errorInfo['errCode'] = 'E80018'
        errorInfo['op_name'] = 'eltwise'
        errorInfo['param_name1'] = 'dtype_output'
        errorInfo['param_name2'] = 'dtype'
        errorInfo['param1_shape'] = str(dtype_output)
        errorInfo['param2_shape'] = str(dtype)
        raise RuntimeError(errorInfo,
                           "In op[%s], the parameter[%s][%s] are not equal in"
                           " dtype with dtype[%s][%s]." %
                           (errorInfo['op_name'], errorInfo['param_name1'],
                            errorInfo['param_name2'], errorInfo['param1_shape'],
                            errorInfo['param2_shape']))

    #mode type must be 0, 1 or 2
    op_list = (0, 1, 2)
    if mode not in op_list:
        errorInfo = {}
        errorInfo['errCode'] = OP_ERROR_CODE_000
        errorInfo['op_name'] = 'eltwise'
        errorInfo['param_name'] = "mode"
        errorInfo['expected_value'] = ",".join(str(i) for i in op_list)
        errorInfo['real_value'] = mode
        raise RuntimeError("In op[%s], the parameter[%s] should be [%s],"
                           " but actually is [%s]." %
                           (errorInfo['op_name'], errorInfo['param_name'],
                            errorInfo['expected_value'], errorInfo['real_value']))

@check_op_params(DYNAMIC_INPUT, REQUIRED_OUTPUT, OPTION_ATTR_INT,
                 OPTION_ATTR_LIST_FLOAT, KERNEL_NAME)
def eltwise(x, y, mode=1, coeff=[], kernel_name="eltwise"):
    """
    Compute elementwise modes, such as 0:PRODUCT, 1:SUM and 2:MAX

    Parameters
    ----------
    x : the list of input data, it's element is dict:{"shape":[], "dtype":""}

    y : the dict of output

    mode : 0:product,1:sum,2:max;default is 1:sum.

    coeff : input_num should be equal with coeff size.

    kernel_name : cce kernel name, default value is "eltwise"

    Returns
    -------
    None

    """
    tensor_num = len(x)
    shapes = [item.get("shape") for item in x]
    shape0 = shapes[0]
    for i in range(1, tensor_num):
        if shapes[i] != shape0:
            errorInfo = {}
            errorInfo['errCode'] = "E81003"
            errorInfo['op_name'] = 'eltwise'
            errorInfo['shapes_list'] = str(shapes)
            raise RuntimeError(errorInfo, "In op[%s], the shapes[%s] of inputs should"
                                          " be the same." %
                               (errorInfo['op_name'], errorInfo['shapes_list']))
    _eltwise_check_para(x, y, mode=mode,
                        coeff=coeff, kernel_name=kernel_name)
    shape = x[0].get("shape")
    dtype = x[0].get("dtype").lower()

    shape = util.shape_refine(shape)
    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x*y, shape)

    tlist = []
    is_l1_depth_fusion = False
    with tvm.target.cce():
        for i in range(0, tensor_num):
            datan_name = 'data%d' % i
            l1_fusion_type = x[i].get("L1_fusion_type", -1)
            if l1_fusion_type == 1:
                raise RuntimeError("eltwise does not support l1 width fusion")
            is_l1_depth_fusion = (l1_fusion_type == 0) or is_l1_depth_fusion
            addr_type = x[i].get("addr_type", 0)
            valid_shape = x[i].get("valid_shape", [])
            slice_offset = x[i].get("slice_offset", [])
            attr_x = {"addr_type": addr_type,
                      "valid_shape": valid_shape,
                      "slice_offset": slice_offset,
                      "L1_fusion_type": l1_fusion_type}
            datan_tmp = tvm.placeholder(fuseshape, name=datan_name,
                                        dtype=dtype, attrs=attr_x)
            tlist.append(datan_tmp)

        res = eltwise_compute(tlist, y, mode, coeff, kernel_name)
        sch = generic.auto_schedule(res)
    tlist.append(res)

    config = {"print_ir": False,
              "need_build": False,
              "name": kernel_name,
              "tensor_list": tlist,
              "l1_fusion_option": is_l1_depth_fusion}
    te.lang.cce.cce_build_code(sch, config)
