import te.lang.cce
from te import tvm
from topi.cce import util
from te.platform.fusion_manager import fusion_manager
from topi import generic


SHAPE_SIZE_LIMIT = 2147483648


# Analysis parameter dim
def reduce_std_check_dim(axis_dim, shape_x, dim):
    dims = len(shape_x)
    if isinstance(dim, int):
        axis_dim.append(dim)
    elif ((dim is None) or (len(dim) == 0)):
        for i in range(dims):
            axis_dim.append(i)
    else:
        for i in dim:
            axis_dim.append(i)
    return axis_dim


@fusion_manager.register("reduce_std")
def reduce_std_compute(x, dim, unbiased, keepdim,
                       kernel_name="reduce_std"):

    # Analysis parameter dim
    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    ori_dtype = x.dtype
    dtype = "float32"
    if ori_dtype == "float16":
        x = te.lang.cce.cast_to(x, dtype)

    axis_dim = []
    axis_dim = reduce_std_check_dim(axis_dim, shape_x, dim)

    # got total number of tensor
    reduce_ele = 1.0
    for i in axis_dim:
        reduce_ele *= shape_x[i]
    cof = reduce_ele**(-1)

    # calculate the mu_muls
    mu_muls = te.lang.cce.vmuls(x, cof)

    # calulate mu
    mu = te.lang.cce.sum(mu_muls, axis=axis_dim, keepdims=True)

    # broadcast
    mu_broadcast = te.lang.cce.broadcast(mu, shape_x)

    # calculate x-mubroadcast
    x_mu_sub = te.lang.cce.vsub(x, mu_broadcast)

    # calculate x_mu_sub^2
    var_mul = te.lang.cce.vmul(x_mu_sub, x_mu_sub)

    # Divided by N or (N-1)
    if unbiased:
        var_muls = te.lang.cce.vmuls(var_mul, (reduce_ele-1.0))
    else:
        var_muls = te.lang.cce.vmuls(var_mul, reduce_ele)

    # sum
    var = te.lang.cce.sum(var_muls, axis=axis_dim, keepdims=keepdim)

    # calculate the square root
    y = te.lang.cce.vsqrt(var)

    # calculate mu_res and return
    mu_res = te.lang.cce.sum(mu_muls, axis=axis_dim, keepdims=keepdim)

    # judge the ori_dtype
    if ori_dtype == "float16":
        y = te.lang.cce.cast_to(y, "float16")
        mu_res = te.lang.cce.cast_to(mu_res, "float16")

    # form a list and return
    return [y, mu_res]


def reduce_std(x, y1, y2, dim=None, unbiased=True, keepdim=False,
               kernel_name="reduce_std"):

    # calculating data parameters
    check_list = ("float16", "float32")

    shape_x = x.get("shape")
    dtype_x = x.get("dtype").lower()
    util.check_dtype_rule(dtype_x, check_list)
    util.check_shape_rule(shape_x)

    data_x = tvm.placeholder(x.get("shape"), dtype=x.get("dtype"), name="data_x")

    res = reduce_std_compute(data_x, dim, unbiased, keepdim, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x] + list(res)}
    te.lang.cce.cce_build_code(schedule, config)
