"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

prior_box_d_v2
"""

from te import platform as tbe_platform
from topi.cce import util
from te import tik
import math
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_shape
from te.utils.op_utils import check_op_params
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import REQUIRED_ATTR_LIST_FLOAT
from te.utils.op_utils import OPTION_ATTR_INT
from te.utils.op_utils import OPTION_ATTR_FLOAT
from te.utils.op_utils import OPTION_ATTR_BOOL
from te.utils.op_utils import OPTION_ATTR_LIST_FLOAT
from te.utils.op_utils import KERNEL_NAME


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT,
                 REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_LIST_FLOAT,
                 REQUIRED_ATTR_LIST_FLOAT, OPTION_ATTR_INT, OPTION_ATTR_INT,
                 OPTION_ATTR_FLOAT, OPTION_ATTR_FLOAT, OPTION_ATTR_BOOL, OPTION_ATTR_BOOL,
                 OPTION_ATTR_FLOAT, OPTION_ATTR_LIST_FLOAT, KERNEL_NAME)
# pylint: disable=locally-disabled,too-many-arguments,too-many-locals,invalid-name
def prior_box_d_v2(feature, img, boxes, y, min_size, max_size, img_h=0,
                   img_w=0, step_h=0.0, step_w=0.0, flip=True, clip=False,
                   offset=0.5, variance=[0.1], kernel_name="prior_box"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "prior_box_d_v2"

    Returns
    -------
    None
    """

    tik_instance = tik.Tik()

    feature_shape = feature.get("shape")
    feature_type = feature.get("dtype").lower()
    img_shape = img.get("shape")
    img_type = img.get("dtype").lower()
    boxes_shape = boxes.get("shape")
    boxes_type = boxes.get("dtype").lower()

    check_list = ("float16", "float32")
    check_dtype(feature_type, check_list, param_name="feature")
    check_list = ("float16", "float32")
    check_dtype(img_type, check_list, param_name="img")
    check_list = ("float16", "float32")
    check_dtype(boxes_type, check_list, param_name="boxes")

    check_shape(feature_shape)
    check_shape(img_shape)
    check_shape(boxes_shape)

    feature_input = tik_instance.Tensor(feature_type, feature_shape, name="feature_input",
                                        scope=tik.scope_gm)
    img_input = tik_instance.Tensor(img_type, img_shape, name="img_input",
                                    scope=tik.scope_gm)
    boxes_input = tik_instance.Tensor(boxes_type, boxes_shape, name="boxes_input",
                                      scope=tik.scope_gm)

    UB_SIZE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)

    block_bite_size = 32
    dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(boxes_type) // 8
    # move number per block
    data_each_block = block_bite_size // dtype_bytes_size

    # compute number of looping
    move_num = boxes_shape[0] * boxes_shape[1] * boxes_shape[2] * boxes_shape[3]

    burse_len = math.ceil(move_num / data_each_block)
    result_ub_size = UB_SIZE / 4

    loop_num = math.ceil((burse_len * block_bite_size) / result_ub_size)

    output_data_ub = tik_instance.Tensor(boxes_type, (result_ub_size,), name="output_data_ub",
                                         scope=tik.scope_ubuf)

    # Need move some data to ub, from feature and image.
    tik_instance.data_move(output_data_ub, feature_input, 0, 1, 1, 0, 0)
    tik_instance.data_move(output_data_ub, img_input, 0, 1, 1, 0, 0)
    # y is output, take value of boxes
    y_shape = y.get("shape")
    y_type = y.get("dtype").lower()
    y_data = tik_instance.Tensor(y_type, y_shape,
                                 name="y_data",
                                 scope=tik.scope_gm)

    # Move value from boxes to ub, then move it to y
    offset = int(result_ub_size / block_bite_size)
    last_offset = burse_len - offset * (loop_num - 1)

    with tik_instance.for_range(0, loop_num) as i0:
        with tik_instance.if_scope(i0 < loop_num - 1):
            tik_instance.data_move(output_data_ub, boxes_input[i0 * data_each_block * offset], 0, 1, offset, 0, 0)
            tik_instance.data_move(y_data[i0 * data_each_block * offset], output_data_ub, 0, 1, offset, 0, 0)
        with tik_instance.else_scope():
            tik_instance.data_move(output_data_ub, boxes_input[i0 * data_each_block * offset], 0, 1, last_offset, 0, 0)
            tik_instance.data_move(y_data[i0 * data_each_block * offset], output_data_ub, 0, 1, last_offset, 0, 0)

    tik_instance.BuildCCE(kernel_name, inputs=[feature_input, img_input, boxes_input],
                          outputs=[y_data])
    return tik_instance

