import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from te.utils import op_utils
from topi.cce import util


@fusion_manager.register("masked_fill")
def masked_fill_compute(x, mask, value, y, kernel_name="masked_fill"):
    '''
    calculating masked_fill
    :param x: TVM tensor
                   the output of previous layer
    :param mask: TVM tensor
                    mask dtype is bool
    :param value: scalar or TVM tensor
                    the value to fill in with
    :param kernel_name: str
                    kernel name, default value is "masked_fill"
    :return:y
            TVM tensor
    '''

    ori_dtype = x.dtype
    if x.dtype in ('int8', 'int32'):
        x = te.lang.cce.cast_to(x, 'float16')

    x_shape = te.lang.cce.util.shape_to_list(x.shape)
    mask_shape = te.lang.cce.util.shape_to_list(mask.shape)
    # computer output shape
    x_shape, mask_shpae, target_shape = op_utils.broadcast_shapes(x_shape, mask_shape)
    target_dtype = x.dtype
    mask = te.lang.cce.cast_to(mask, x.dtype)
    value = te.lang.cce.cast_to(value, x.dtype)

    mask = te.lang.cce.broadcast(mask, target_shape)
    tensor_ones = te.lang.cce.broadcast(tvm.const(1, target_dtype), target_shape)
    value = te.lang.cce.broadcast(value, target_shape)
    x = te.lang.cce.broadcast(x, target_shape)
    y = te.lang.cce.vcmpsel(mask, tensor_ones, 'eq', value, x)

    if y.dtype != ori_dtype:
        y = te.lang.cce.cast_to(y, ori_dtype)

    return y


@op_utils.check_op_params(dict, dict, dict, dict, str)
def masked_fill(x, mask, value, y, kernel_name="masked_fill"):
    '''
    :param x: dict
                    shape and dtype of tensor x input
    :param mask: dict
                    shape and dtype of tensor mask,
                    can be boardcast as shape as x
    :param value: dict
                    shape and dtype of value
    :param y: dict
                    the output of masked _fill
    :param kernel_name: str
                      kernel name, default value is "masked _fill"
    :return: none
    '''

    x_shape = x.get("shape")
    x_dtype = x.get("dtype")
    x_dtype_lower = x_dtype.lower()

    mask_shape = mask.get("shape")
    mask_dtype = mask.get("dtype")

    value_shape = value.get("shape")
    value_dtype = value.get("dtype")
    value_dtype_lower = value_dtype.lower()

    # check dtype
    x_dtype_list = ("float16", "float32", "int8", "int32")
    op_utils.check_dtype(x_dtype, x_dtype_list)

    mask_dtype_list = ("bool", "int8")
    op_utils.check_dtype(mask_dtype, mask_dtype_list)

    if mask_dtype == "bool":
        mask_dtype = "int8"

    value_dtype_list = ("float16", "float32", "int8", "int32")
    op_utils.check_dtype(value_dtype, value_dtype_list)

    # check shape
    op_utils.check_shape(x_shape)
    op_utils.check_shape(mask_shape)
    op_utils.check_shape(value_shape)

    # check boardcast shape
    x_shape, mask_shape, out_shape = op_utils.broadcast_shapes(x_shape, mask_shape)
    op_utils.check_shape(out_shape)

    # check kernel_name
    util.check_kernel_name(kernel_name)

    pos_mask_shape = tuple([1] * (len(x_shape) - len(mask_shape))) + tuple(mask_shape)
    data_x = tvm.placeholder(x_shape, dtype=x_dtype_lower, name="data_x")

    data_mask = tvm.placeholder(pos_mask_shape, dtype=mask_dtype, name="data_mask")

    data_value = tvm.placeholder(pos_mask_shape, dtype=value_dtype_lower, name="data_value")

    y = masked_fill_compute(data_x, data_mask, data_value, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(y)

    config = {"name": kernel_name,
              "tensor_list": [data_x, data_mask, data_value, y],
              }
    te.lang.cce.cce_build_code(schedule, config)
