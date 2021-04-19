import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from te import tvm
from functools import reduce as reduceIns
from topi import generic
from te.utils.op_utils import *


@fusion_manager.register("mish")
def mish_compute(input_x, output_y, kernel_name="mish"):
    """
    algorithm: mish
    calculating data's mish,y= x*(1 - 2/(1+(1+exp(x))^2))

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is mish

    Returns
    -------
    res : tvm.tensor
        the result of mish
    """
    dtype = input_x.dtype
    exp_val = te.lang.cce.vexp(input_x)
    add_exp_val = te.lang.cce.vadds(exp_val, tvm.const(1, dtype))
    pow_var = te.lang.cce.vmul(add_exp_val, add_exp_val)
    add_val = te.lang.cce.vadds(pow_var, tvm.const(1, dtype))
    rec_val = te.lang.cce.vrec(add_val)
    mul_val = te.lang.cce.vmuls(rec_val, tvm.const(-2, dtype=dtype))
    add_val2 = te.lang.cce.vadds(mul_val, tvm.const(1, dtype=dtype))
    res = te.lang.cce.vmul(input_x, add_val2)

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def mish(input_x, output_y, kernel_name="mish"):
    """
    algorithm: mish
    calculating data's mish,y= x*(1 - 2/(1+(1+exp(x))^2))

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is mish

    Returns
    -------
    None
    """

    input_shape = input_x.get("shape")
    input_format = input_x.get("format")
    input_dtype = input_x.get("dtype").lower()
    check_shape(input_shape, param_name="input_x")
    check_list = ("float16", "float32")
    check_dtype(input_dtype, check_list, param_name="input_x")
    check_format(input_format)

    # fuse single axis
    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x * y, input_shape)

    data_x = tvm.placeholder(fuseshape, dtype=input_dtype, name="data_x")
    res = mish_compute(data_x, output_y, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_x, res]}
    te.lang.cce.cce_build_code(schedule, config)