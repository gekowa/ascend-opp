# Copyright 2018 Huawei Technologies Co., Ltd
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
leaky_relu
"""
from functools import reduce as reduceIns
from te.utils.op_utils import *

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic


# pylint: disable=locally-disabled,unused-argument,invalid-name
@fusion_manager.register("leaky_relu")
def leaky_relu_compute(x, y, negative_slope=0, kernel_name="leaky_relu"):
    """
    compute for caffe_relu_layer_cce
    """
    res = te.lang.cce.vlrelu(x, negative_slope)
    if x.op.attrs:
        if 'format' in x.op.attrs:
            res.op.attrs['format'] = x.op.attrs['format']
    res.op.attrs["negative_slope"] = negative_slope
    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT,
                 OPTION_ATTR_FLOAT, KERNEL_NAME)
def leaky_relu(x, y, negative_slope=0, kernel_name="leaky_relu"):
    """leaky_relu op for input tensor

       f(x)= x(x>=0) or negative_slope*x(x<0) equal to
       f(x)=negative_slope*x

    Parameters
    ----------
    x : TVM tensor
        input tensor has shape and dtype attributes
    y : dict
        dict with keys(shape and dtype) of output

    negative_slope : float or int
        allow non-zero slope for negative inputs to speed up optimization

    kernel_name : str
        cce kernel name, default value is "leaky_relu"

    Returns
    ------
    None
    """

    # check input tensor shape
    shape = x.get("shape")
    dtype = x.get("dtype")
    check_shape(shape, param_name="x")

    # check input tensor data_type
    check_list = ["float16", "float32", "int32", "int8"]
    check_dtype(dtype.lower(), check_list, param_name="x")
    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x*y, shape)
    inp_dtype = dtype.lower()
    input_data_x = tvm.placeholder(fuseshape, name="input_data_x",
                                   dtype=inp_dtype)

    with tvm.target.cce():

        res = leaky_relu_compute(input_data_x, y, negative_slope, kernel_name)
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_data_x, res]}
    te.lang.cce.cce_build_code(sch, config)
