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

gru
"""
import operator

import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tik
from te import tvm
from te.utils import check_para as para_check
from te.utils.error_manager import error_manager_vector


def _sigmoid_compute(input_x):
    """
    calculating sigmoid
    """
    data_input = input_x
    dtype = input_x.dtype
    exp_support = tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp", "float32")
    mul_support = tbe_platform.cce_conf.api_check_support("te.lang.cce.vmuls", "float32")
    if dtype == "float32" and not mul_support:
        error_manager_vector.raise_err_check_params_rules("DynamicGRU",
                                                          'vmuls should support float32',
                                                          'mul_support', str(mul_support))

    const_num_neg_one = tvm.const(-1, dtype=dtype)
    const_num_one = tvm.const(1, dtype=dtype)
    tmp_negative = tbe.vmuls(data_input, const_num_neg_one)
    if dtype == "float32" and not exp_support:
        tmp_negative = tbe.cast_to(tmp_negative, "float16")
    tmp_exp = tbe.vexp(tmp_negative)
    if dtype == "float32" and not exp_support:
        tmp_exp = tbe.cast_to(tmp_exp, "float32")
    tmp_sum = tbe.vadds(tmp_exp, const_num_one)
    if dtype == "float32":
        inp_shape = tmp_sum.shape
        tensor_one = tbe.broadcast(tvm.const(1, dtype), inp_shape)
        res = tbe.vdiv(tensor_one, tmp_sum)
    else:
        res = tbe.vrec(tmp_sum)

    return res


def _tanh_compute(input_x):
    """
    algorithm: tanh
    calculating data's tanh,y= (e^(2x)-1)/(e^(2x)+1)
    """
    input_dtype = input_x.dtype
    # positive min float32 value
    min_fp_data = 2 ** (-126)
    const_dtype = input_dtype
    # positive min float16 value
    if input_dtype == "float16":
        min_fp_data = 2 ** (-14)

    has_improve_precision = False

    if input_dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("vexp", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        has_improve_precision = True
        const_dtype = "float32"

    input_abs = tbe.vabs(input_x)
    power_val = tbe.vmuls(input_abs, tvm.const(-2, const_dtype))
    exp_val = tbe.vexp(power_val)

    up_val_tmp = tbe.vmul(exp_val, input_x)
    up_val = tbe.vsub(input_x, up_val_tmp)

    input_x_tmp = tbe.vadds(input_abs, min_fp_data)
    down_val_tmp = tbe.vadds(exp_val, tvm.const(1, const_dtype))
    down_val = tbe.vmul(down_val_tmp, input_x_tmp)

    res = tbe.vdiv(up_val, down_val)

    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


def _get_emit_insn_map(tensor):
    """
    get tensor's emit_insn key
    """
    insn_map = {"elewise_single_cast": "vector_conv",
                "elewise_single_VS_max": "vector_maxs",
                "elewise_single_VS_min": "vector_mins",
                "elewise_single_log": "vector_ln",
                "elewise_single_exp": "vector_exp",
                "elewise_single_rec": "vector_rec",
                "elewise_single_relu": "vector_relu",
                "elewise_single_abs": "vector_abs",
                "elewise_single_not": "vector_not",
                "elewise_single_sqrt": "vector_sqrt",
                "elewise_single_rsqrt": "vector_rsqrt",
                "elewise_binary_mul": "vector_mul",
                "elewise_single_VS_mul": "vector_muls",
                "elewise_binary_div": "vector_div",
                "elewise_binary_add": "vector_add",
                "elewise_single_VS_add": "vector_adds",
                "elewise_binary_min": "vector_min",
                "elewise_binary_max": "vector_max",
                "elewise_binary_vcmpv_gt": "vector_gt",
                "elewise_binary_vcmpv_ge": "vector_ge",
                "elewise_binary_vcmpv_lt": "vector_lt",
                "elewise_binary_vcmpv_le": "vector_le",
                "elewise_binary_vcmpv_eq": "vector_eq",
                "elewise_binary_vcmpv_ne": "vector_ne",
                "elewise_binary_or": "vector_or",
                "elewise_binary_and": "vector_and",
                "elewise_multiple_mla": "vector_multiple",
                "elewise_multiple_madd": "vector_multiple",
                "elewise_multiple_maddrelu": "vector_multiple",
                "broadcast_for_tensor": "broadcast_for_tensor",
                "elewise_binary_sub": "vector_sub",
                "broadcast": "broadcast"}
    if tensor.op.tag.find("|") != -1:
        str_list = tensor.op.tag.split("|")
        insn = insn_map.get(str_list[0])
    else:
        insn = insn_map.get(tensor.op.tag)
    return insn


def _get_tiling(m_size, k_size, hidden_size):
    """
    get tiling
    :return:
    """
    if k_size * hidden_size <= 128:
        return (1, hidden_size, k_size, 1, hidden_size, k_size)

    n = 256 // hidden_size if hidden_size <= 256 else 1
    k = 128 // n if n <= 128 else 1

    return (1, n, k, 1, n, k)


# pylint: disable=too-many-arguments,too-many-branches,too-many-locals
def _check_dtype(input_x, weight, bias, cw, cb, init_h, y, output_h, rg, ig, ng):
    """
    check parameters dtype
    :return:
    """
    para_check.check_dtype(input_x["dtype"], ["float16", ], "x")
    para_check.check_dtype(weight["dtype"], ["float16", ], "w")
    para_check.check_dtype(cw["dtype"], ["float16", ], "cw")

    bias_dtype = bias["dtype"]
    para_check.check_dtype(bias_dtype, ["float16", "float32"], "b")


    def _check_equal_bias_dtype(p, name):
        if p["dtype"] != bias_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal("DynamicGRU", 'b', name, bias_dtype, p["dtype"])
    _check_equal_bias_dtype(cb, "cb")
    _check_equal_bias_dtype(y, "y")
    _check_equal_bias_dtype(output_h, "output_h")
    if init_h is not None:
        _check_equal_bias_dtype(init_h, "init_h")
    if rg is not None:
        _check_equal_bias_dtype(rg, "r")
    if ig is not None:
        _check_equal_bias_dtype(ig, "i")
    if ng is not None:
        _check_equal_bias_dtype(ng, "n")


# pylint: disable=too-many-arguments,too-many-branches,too-many-locals
def _check_param(input_x, weight, bias, cw, cb, seq_length, init_h, y, output_h, rg, ig, ng):
    """
    check parameters
    :return:
    """
    # t size
    if input_x["shape"][0] != output_h["shape"][0]:
        error_manager_vector.raise_err_check_params_rules("DynamicGRU", 'x.shape[0] == output_h.shape[0]',
                                                          'output_h.shape[0]', output_h["shape"][0])

    # batch_size
    if input_x["shape"][2] != output_h["shape"][2]:
        error_manager_vector.raise_err_check_params_rules("DynamicGRU", 'x.shape[2] == output_h.shape[2]',
                                                          'output_h.shape[2]', output_h["shape"][2])

    # k_size = input + hidden
    if weight["shape"][0] != input_x["shape"][1] + output_h["shape"][1]:
        error_manager_vector.raise_err_check_params_rules("DynamicGRU", 'w.shape[0] == x.shape[1] + output_h.shape[1]',
                                                          'w.shape[0]', weight["shape"][0])

    # hidden_size
    if weight["shape"][1] != 2 * output_h["shape"][1]:
        error_manager_vector.raise_err_check_params_rules("DynamicGRU", 'w.shape[1] == 2 * output_h.shape[1]',
                                                          'w.shape[1]', weight["shape"][1])

    if cw["shape"][1] != output_h["shape"][1]:
        error_manager_vector.raise_err_check_params_rules("DynamicGRU", 'cw.shape[1] == output_h.shape[1]',
                                                          'cw.shape[1]', cw["shape"][1])

    if (bias["shape"][0] + 15) // 16 != weight["shape"][1]:
        error_manager_vector.raise_err_check_params_rules("DynamicGRU", '(b.shape[0] + 15) // 16 == w.shape[1]',
                                                          'b.shape[0]', bias["shape"][0])

    if (cb["shape"][0] + 15) // 16 != cw["shape"][1]:
        error_manager_vector.raise_err_check_params_rules("DynamicGRU", '(cb.shape[0] + 15) // 16 == cw.shape[1]',
                                                          'cb.shape[0]', cb["shape"][0])

    # check output
    if not operator.eq(output_h["shape"], y["shape"]):
        error_manager_vector.raise_err_check_params_rules("DynamicGRU", 'y.shape == output_h.shape',
                                                          'y.shape', str(y["shape"]))

    if rg is not None and not operator.eq(output_h["shape"], rg["shape"]):
        error_manager_vector.raise_err_check_params_rules("DynamicGRU", 'r.shape == output_h.shape',
                                                          'r.shape', str(rg["shape"]))

    if ig is not None and not operator.eq(output_h["shape"], ig["shape"]):
        error_manager_vector.raise_err_check_params_rules("DynamicGRU", 'i.shape == output_h.shape',
                                                          'i.shape', str(ig["shape"]))

    if ng is not None and not operator.eq(output_h["shape"], ng["shape"]):
        error_manager_vector.raise_err_check_params_rules("DynamicGRU", 'n.shape == output_h.shape',
                                                          'n.shape', str(ng["shape"]))

    # check unsupport pramas
    if seq_length is not None:
        error_manager_vector.raise_err_check_params_rules("DynamicGRU", 'seq_length == None',
                                                          'seq_length', str(seq_length))


# pylint: disable=too-many-arguments,too-many-branches,too-many-locals
def _check_attr(direction, cell_depth, keep_prob, cell_clip, num_proj, time_major,
                activation, is_training):
    if direction not in ["UNIDIRECTIONAL", "BIDIRECTIONAL"]:
        error_manager_vector.raise_err_check_params_rules("DynamicGRU",
                                                          'direction in ["UNIDIRECTIONAL", "BIDIRECTIONAL"]',
                                                          'direction', str(direction))

    if cell_depth != 1:
        error_manager_vector.raise_err_check_params_rules("DynamicGRU", 'cell_depth == 1',
                                                          'cell_depth', str(cell_depth))

    if keep_prob != 1.0:
        error_manager_vector.raise_err_check_params_rules("DynamicGRU", 'keep_prob == 1.0',
                                                          'keep_prob', str(keep_prob))

    if cell_clip != -1.0:
        error_manager_vector.raise_err_check_params_rules("DynamicGRU", 'cell_clip == -1.0',
                                                          'cell_clip', str(cell_clip))

    if num_proj != 0:
        error_manager_vector.raise_err_check_params_rules("DynamicGRU", 'num_proj == 0',
                                                          'num_proj', str(num_proj))

    if time_major is not True:
        error_manager_vector.raise_err_check_params_rules("DynamicGRU", 'time_major is True',
                                                          'time_major', str(time_major))

    if activation != "tanh":
        error_manager_vector.raise_err_check_params_rules("DynamicGRU", 'activation is tanh',
                                                          'activation', str(activation))


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_OUTPUT, para_check.OPTION_OUTPUT, para_check.OPTION_OUTPUT,
                            para_check.OPTION_OUTPUT, para_check.OPTION_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
# pylint: disable=too-many-arguments,too-many-locals,invalid-name
# pylint: disable=too-many-function-args,too-many-statements
def dynamic_gru(input_x, weight, bias, cw, cb,
                seq_length, init_h,
                y, output_h,
                rg, ig, ng,
                direction="UNIDIRECTIONAL", cell_depth=1, keep_prob=1.0,
                cell_clip=-1.0, num_proj=0, time_major=True,
                activation="tanh", is_training=True,
                kernel_name="dynamic_gru"):

    _check_dtype(input_x, weight, bias, cw, cb,
                init_h, y, output_h, rg, ig, ng)
    _check_param(input_x, weight, bias, cw, cb,
                seq_length, init_h, y, output_h, rg, ig, ng)
    _check_attr(direction, cell_depth, keep_prob,
               cell_clip, num_proj, time_major,
               activation, is_training)

    shape_x_input = input_x.get("shape")
    shape_w_input = weight.get("shape")
    input_dtype = input_x.get("dtype").lower()
    bias_dtype = bias.get("dtype").lower()

    t_size = shape_x_input[0]
    m_size = shape_x_input[2]
    k_size = shape_w_input[0]
    n_size = shape_w_input[1]
    hidden_size = n_size // 2
    in_x = k_size - hidden_size

    shape_x = (t_size, in_x, m_size, 16, 16)
    shape_w_1 = (1, k_size, 2, hidden_size, 16, 16)
    shape_w_2 = (1, k_size, 1, hidden_size, 16, 16)
    shape_h = (t_size, hidden_size, m_size, 16, 16)
    shape_bias_1 = (1, 2, hidden_size, 1, 1, 16)
    shape_bias_2 = (1, hidden_size, 1, 1, 16)
    shape_h_init = (1, hidden_size, m_size, 16, 16)

    is_global_init = init_h is not None
    is_gate_output = ig is not None

    tik_instance = tik.Tik(tik.Dprofile())
    input_x = tik_instance.Tensor(shape=shape_x, dtype=input_dtype, scope=tik.scope_gm, name='input_x')
    weight1 = tik_instance.Tensor(shape=shape_w_1, dtype=input_dtype, scope=tik.scope_gm, name='weight1')
    bias1 = tik_instance.Tensor(shape=shape_bias_1, dtype=bias_dtype, scope=tik.scope_gm, name='bias1')
    weight2 = tik_instance.Tensor(shape=shape_w_2, dtype=input_dtype, scope=tik.scope_gm, name='weight2')
    bias2 = tik_instance.Tensor(shape=shape_bias_2, dtype=bias_dtype, scope=tik.scope_gm, name='bias2')
    if is_global_init:
        s_init_h_gm = tik_instance.Tensor(shape=shape_h_init, dtype=bias_dtype, scope=tik.scope_gm, name='s_init_h_gm')
    update_h_gm = tik_instance.Tensor(shape=shape_h, dtype=bias_dtype, scope=tik.scope_gm, name='update_h_gm')
    update_y_gm = tik_instance.Tensor(shape=shape_h, dtype=bias_dtype, scope=tik.scope_gm, name='update_y_gm')
    if is_gate_output:
        r_t_gm = tik_instance.Tensor(shape=shape_h, dtype=bias_dtype, scope=tik.scope_gm, name='r_t_gm')
        i_t_gm = tik_instance.Tensor(shape=shape_h, dtype=bias_dtype, scope=tik.scope_gm, name='i_t_gm')
        n_t_gm = tik_instance.Tensor(shape=shape_h, dtype=bias_dtype, scope=tik.scope_gm, name='n_t_gm')

    last = 1
    sub_t = 1
    sub_m = 1
    loop_t = t_size // sub_t
    loop_m = m_size // sub_m
    with tik_instance.for_range(0, loop_t) as i:
        with tik_instance.for_range(0, loop_m, block_num=loop_m) as j:
            input_x_var = input_x[i * sub_t: i * sub_t + sub_t, :, j * sub_m: j * sub_m + sub_m, :, :]
            if is_global_init:
                s_init_h_gm_var = s_init_h_gm[:, :, j * sub_m: j * sub_m + sub_m, :, :]
            else:
                s_init_h_gm_var = None
            last_h = update_h_gm[i * sub_t - last: i * sub_t + sub_t - last:, :, j * sub_m: j * sub_m + sub_m, :, :]
            update_h_gm_var = update_h_gm[i * sub_t: i * sub_t + sub_t, :, j * sub_m: j * sub_m + sub_m, :, :]
            update_y_gm_var = update_y_gm[i * sub_t: i * sub_t + sub_t, :, j * sub_m: j * sub_m + sub_m, :, :]
            if is_gate_output:
                r_t_gm_var = r_t_gm[i * sub_t: i * sub_t + sub_t, :, j * sub_m: j * sub_m + sub_m, :, :]
                i_t_gm_var = i_t_gm[i * sub_t: i * sub_t + sub_t, :, j * sub_m: j * sub_m + sub_m, :, :]
                n_t_gm_var = n_t_gm[i * sub_t: i * sub_t + sub_t, :, j * sub_m: j * sub_m + sub_m, :, :]
            else:
                r_t_gm_var = None
                i_t_gm_var = None
                n_t_gm_var = None

            input_list = [input_x_var, weight1, weight2, bias1, bias2, s_init_h_gm_var, last_h]
            if is_gate_output:
                output_list = [update_y_gm_var, update_h_gm_var, r_t_gm_var, i_t_gm_var, n_t_gm_var]
            else:
                output_list = [update_y_gm_var, update_h_gm_var]

            with tik_instance.if_scope(i == 0):
                is_first_round = True
                tik_instance.call_module(
                    _dynamic_gru_inner,
                    input_list,
                    output_list,
                    [is_gate_output, is_first_round, is_global_init])

            with tik_instance.if_scope(i > 0):
                is_first_round = False
                tik_instance.call_module(
                    _dynamic_gru_inner,
                    input_list,
                    output_list,
                    [is_gate_output, is_first_round, is_global_init])

    config_map = {
        "dump_cce_code": False,
    }

    build_input_list = [input_x, weight1, bias1, weight2, bias2]
    if is_global_init:
        build_input_list.append(s_init_h_gm)

    build_output_list = [update_y_gm, update_h_gm]
    if is_gate_output:
        build_output_list.extend([r_t_gm, i_t_gm, n_t_gm])

    tik_instance.BuildCCE(kernel_name,
                          build_input_list,
                          build_output_list,
                          config=config_map)


def _dynamic_gru_inner(input_list, custom_list):
    input_x = input_list[0]
    weight1 = input_list[1]
    weight2 = input_list[2]
    bias1 = input_list[3]
    bias2 = input_list[4]
    s_init_h_gm = input_list[5]
    s_state_h_gm_last = input_list[6]

    is_gate_output = custom_list[0]
    is_first_round = custom_list[1]
    is_global_init = custom_list[2]

    input_dtype = 'float16'
    bias_dtype = bias1.dtype
    fp16_input_output = bias_dtype == 'float16'

    shape_x_input = input_x.shape
    shape_w1_input = weight1.shape
    w1_size = 2
    w2_size = 1
    t_size = shape_x_input[0].value
    m_size = shape_x_input[2].value
    k_size = shape_w1_input[1].value
    hidden_size = shape_w1_input[3].value
    in_x = k_size - hidden_size

    shape_b_1 = (1, k_size, w1_size, hidden_size, 16, 16)
    shape_b_2 = (1, k_size, w2_size, hidden_size, 16, 16)
    shape_c_1 = (1, w1_size, hidden_size, m_size, 16, 16)
    shape_c_2 = (1, w2_size, hidden_size, m_size, 16, 16)
    shape_bias_1 = (1, w1_size, hidden_size, 1, 1, 16)
    shape_bias_2 = (1, hidden_size, 1, 1, 16)
    shape_i = (1, hidden_size, m_size, 16, 16)
    shape_i_t = (t_size, hidden_size, m_size, 16, 16)
    k0_size = 16

    if is_first_round and not is_global_init:
        s_state_h = tvm.compute(shape_i,
                                lambda *indices: tvm.const(0.0, dtype='float32'),
                                name='s_state_h')
        s_state_h_fp16 = tvm.compute(shape_i,
                                     lambda *indices: s_state_h(*indices).astype('float16'),
                                     name="s_state_h_fp16")
    else:
        last_h = s_init_h_gm if is_first_round else s_state_h_gm_last
        if fp16_input_output:
            s_state_h_fp16 = tvm.compute(shape_i,
                                         lambda *indices: last_h(*indices),
                                         name='s_state_h_fp16')
            s_state_h = tvm.compute(shape_i,
                                    lambda *indices: s_state_h_fp16(*indices).astype('float32'),
                                    name="s_state_h")
        else:
            s_state_h = tvm.compute(shape_i,
                                    lambda *indices: last_h(*indices),
                                    name='s_state_h')
            s_state_h_fp16 = tvm.compute(shape_i,
                                         lambda *indices: s_state_h(*indices).astype('float16'),
                                         name="s_state_h_fp16")

    # compute
    # input and s_state_h need first to ub and cast to float16
    shape_a_z_bigz = (1, m_size, k_size, 16, 16)

    # input and s_start_h is Nz, need trans to zZ
    # so change axis 1 and 2
    a_l1_1 = tvm.compute(shape_a_z_bigz,
                         lambda *indice:
                         tvm.select(indice[2] < in_x,
                                    input_x[indice[0],
                                            indice[2],
                                            indice[1],
                                            indice[3],
                                            indice[4]],
                                    s_state_h_fp16[0,
                                                   indice[2] - in_x,
                                                   indice[1],
                                                   indice[3],
                                                   indice[4]]
                                    ),
                         name="a_l1_1", tag="concat")
    b_l1_1 = tvm.compute(shape_b_1,
                         lambda *indices: weight1(*indices),
                         name='b_l1_1')
    a_l0a_1 = tvm.compute(shape_a_z_bigz, lambda *indices: a_l1_1(*indices), name="a_l0a_1")
    b_l0b_1 = tvm.compute(shape_b_1, lambda *indices: b_l1_1(*indices), name="b_l0b_1")
    k1_1 = tvm.reduce_axis((0, k_size), name='k1_1')
    k0_1 = tvm.reduce_axis((0, k0_size), name='k0_1')
    c_l0c_1 = tvm.compute(shape_c_1,
                          lambda t, nb_0, nb_1, mb, mp, np:
                          tvm.sum((a_l0a_1[t, mb, k1_1, mp, k0_1] * \
                                   b_l0b_1[t, k1_1, nb_0, nb_1, np, k0_1]) \
                                  .astype('float32'),
                                  axis=[k1_1, k0_1]),
                          name='c_l0c_1')
    c_ub_1 = tvm.compute(shape_c_1, lambda *indices: c_l0c_1(*indices), name="c_ub_1")
    bias_ub_1 = tvm.compute(shape_bias_1,
                            lambda *indices: bias1(*indices),
                            name='bias_ub_1')
    bias_ub_1_fp32 = bias_ub_1
    if fp16_input_output:
        bias_ub_1_fp32 = tvm.compute(shape_bias_1,
                                     lambda *indices: bias_ub_1(*indices).astype('float32'),
                                     name="bias_ub_1_fp32")
    bias_bc_ub_1 = tbe.broadcast(bias_ub_1_fp32, shape_c_1)
    c_ub_bias_1 = tbe.vadd(c_ub_1, bias_bc_ub_1)

    # split matmul res
    r_t_index = 0
    i_t_index = 1
    r_t = tvm.compute(shape_i,
                      lambda t, i, j, k, l: c_ub_bias_1(t, r_t_index, i, j, k, l),
                      name="r_t")
    i_t = tvm.compute(shape_i,
                      lambda t, i, j, k, l: c_ub_bias_1(t, i_t_index, i, j, k, l),
                      name="i_t")
    r_t_sigmoid = _sigmoid_compute(r_t)
    i_t_sigmoid = _sigmoid_compute(i_t)
    r_t_mid = r_t_sigmoid
    i_t_mid = i_t_sigmoid
    if is_gate_output:
        if fp16_input_output:
            r_t_sigmoid_fp16 = tvm.compute(shape_i,
                                           lambda *indices: r_t_sigmoid(*indices).astype('float16'),
                                           name="r_t_sigmoid_fp16")
            i_t_sigmoid_fp16 = tvm.compute(shape_i,
                                           lambda *indices: i_t_sigmoid(*indices).astype('float16'),
                                           name="i_t_sigmoid_fp16")

            r_t_gm = tvm.compute(shape_i,
                                 lambda *indices: r_t_sigmoid_fp16(*indices),
                                 name="r_t_gm")
            i_t_gm = tvm.compute(shape_i,
                                 lambda *indices: i_t_sigmoid_fp16(*indices),
                                 name="i_t_gm")

            r_t_gm_back = tvm.compute(shape_i,
                                      lambda *indices: r_t_gm(*indices),
                                      name="r_t_gm_back")
            i_t_gm_back = tvm.compute(shape_i,
                                      lambda *indices: i_t_gm(*indices),
                                      name="i_t_gm_back")

            r_t_gm_back_fp32 = tvm.compute(shape_i,
                                           lambda *indices: r_t_gm_back(*indices).astype('float32'),
                                           name="r_t_gm_back_fp32")
            i_t_gm_back_fp32 = tvm.compute(shape_i,
                                           lambda *indices: i_t_gm_back(*indices).astype('float32'),
                                           name="i_t_gm_back_fp32")

            r_t_mid = r_t_gm_back_fp32
            i_t_mid = i_t_gm_back_fp32
        else:
            r_t_gm = tvm.compute(shape_i,
                                 lambda *indices: r_t_sigmoid(*indices),
                                 name="r_t_gm")
            i_t_gm = tvm.compute(shape_i,
                                 lambda *indices: i_t_sigmoid(*indices),
                                 name="i_t_gm")

            r_t_gm_back = tvm.compute(shape_i,
                                      lambda *indices: r_t_gm(*indices),
                                      name="r_t_gm_back")
            i_t_gm_back = tvm.compute(shape_i,
                                      lambda *indices: i_t_gm(*indices),
                                      name="i_t_gm_back")

            r_t_mid = r_t_gm_back
            i_t_mid = i_t_gm_back
    r_t_h = tbe.vmul(r_t_mid, s_state_h)
    r_t_h_fp16 = \
        tvm.compute(shape_i,
                    lambda *indices: r_t_h(*indices).astype(input_dtype),
                    name="r_t_h_fp16")

    # second matmul
    a_l1_2 = tvm.compute(shape_a_z_bigz,
                         lambda *indice:
                         tvm.select(indice[2] < in_x,
                                    input_x[indice[0],
                                            indice[2],
                                            indice[1],
                                            indice[3],
                                            indice[4]],
                                    r_t_h_fp16[0,
                                               indice[2] - in_x,
                                               indice[1],
                                               indice[3],
                                               indice[4]]
                                    ),
                         name="a_l1_2", tag="concat")

    b_l1_2 = tvm.compute(shape_b_2,
                         lambda *indices: weight2(*indices),
                         name='b_l1_2')
    a_l0a_2 = tvm.compute(shape_a_z_bigz, lambda *indices: a_l1_2(*indices), name="a_l0a_2")
    b_l0b_2 = tvm.compute(shape_b_2, lambda *indices: b_l1_2(*indices), name="b_l0b_2")
    k1_2 = tvm.reduce_axis((0, k_size), name='k1_2')
    k0_2 = tvm.reduce_axis((0, k0_size), name='k0_2')
    c_l0c_2 = tvm.compute(shape_c_2,
                          lambda t, nb_0, nb_1, mb, mp, np:
                          tvm.sum((a_l0a_2[t, mb, k1_2, mp, k0_2] * \
                                   b_l0b_2[t, k1_2, nb_0, nb_1, np, k0_2]) \
                                  .astype('float32'),
                                  axis=[k1_2, k0_2]),
                          name='c_l0c_2')
    c_ub_2 = tvm.compute(shape_i, lambda t, h, m, i, j: c_l0c_2(t, 0, h, m, i, j), name="c_ub_2")
    bias_ub_2 = tvm.compute(shape_bias_2,
                            lambda t, h, m, i, j: bias2(t, h, m, i, j),
                            name='bias_ub_2')
    bias_ub_2_fp32 = bias_ub_2
    if fp16_input_output:
        bias_ub_2_fp32 = tvm.compute(shape_bias_2,
                                     lambda *indices: bias_ub_2(*indices).astype('float32'),
                                     name="bias_ub_2_fp32")
    bias_bc_ub_2 = tbe.broadcast(bias_ub_2_fp32, shape_i)
    c_ub_bias_2 = tbe.vadd(c_ub_2, bias_bc_ub_2)

    h_t_tanh = _tanh_compute(c_ub_bias_2)
    h_t_tanh_mid = h_t_tanh
    if is_gate_output:
        if fp16_input_output:
            h_t_tanh_fp16 = tvm.compute(shape_i,
                                        lambda *indices: h_t_tanh(*indices).astype('float16'),
                                        name="h_t_tanh_fp16")
            n_t_gm = tvm.compute(shape_i,
                                 lambda *indices: h_t_tanh_fp16(*indices),
                                 name="n_t_gm")
            n_t_gm_back = tvm.compute(shape_i,
                                      lambda *indices: n_t_gm(*indices),
                                      name="n_t_gm_back")
            n_t_gm_back_fp32 = tvm.compute(shape_i,
                                           lambda *indices: n_t_gm_back(*indices).astype('float32'),
                                           name="n_t_gm_back_fp32")
            h_t_tanh_mid = n_t_gm_back_fp32
        else:
            n_t_gm = tvm.compute(shape_i,
                                 lambda *indices: h_t_tanh(*indices),
                                 name="n_t_gm")
            n_t_gm_back = tvm.compute(shape_i,
                                      lambda *indices: n_t_gm(*indices),
                                      name="n_t_gm_back")
            h_t_tanh_mid = n_t_gm_back

    c_t_tmp1 = tbe.vsub(s_state_h, h_t_tanh_mid)
    c_t_tmp2 = tbe.vmul(c_t_tmp1, i_t_mid)
    update_h = tbe.vadd(c_t_tmp2, h_t_tanh_mid)
    update_h_ub = update_h
    if fp16_input_output:
        update_h_fp16 = tvm.compute(shape_i_t,
                                    lambda *indices: update_h(*indices).astype('float16'),
                                    name="update_h_fp16")
        update_h_ub = update_h_fp16
    update_y_gm = tvm.compute(shape_i_t,
                              lambda t, i, j, k, l: update_h_ub(0, i, j, k, l),
                              name="update_y_gm")
    update_y_gm_back = tvm.compute(shape_i_t,
                                   lambda t, i, j, k, l: update_y_gm(0, i, j, k, l),
                                   name="update_y_gm_back")
    update_h_gm = tvm.compute(shape_i_t,
                              lambda t, i, j, k, l: update_y_gm_back(0, i, j, k, l),
                              name="update_h_gm")
    # end compute

    # schedule
    s = tvm.schedule.create_schedule([update_h_gm.op])

    def gen_reversed_subgraph_list(out_tensor, tensor_list):
        """
        traverse tensors by Depth-First-Search
        """
        if out_tensor is None:
            return
        stack = [out_tensor]
        visited_list = []
        while stack:
            cur_tensor = stack.pop()
            visited_list.append(cur_tensor)
            for in_tensor in cur_tensor.op.input_tensors:
                if in_tensor not in visited_list:
                    stack.append(in_tensor)
                    if "elewise" in in_tensor.op.tag or \
                            "broadcast" == in_tensor.op.tag:
                        if in_tensor not in tensor_list:
                            tensor_list.append(in_tensor)

    elewise_tensors_r_t_h_fp16 = []
    gen_reversed_subgraph_list(r_t_h_fp16, elewise_tensors_r_t_h_fp16)

    elewise_tensors = []
    tmp_tensors = []
    gen_reversed_subgraph_list(update_h_gm, tmp_tensors)
    for i in tmp_tensors:
        if i not in elewise_tensors_r_t_h_fp16:
            elewise_tensors.append(i)

    # set scope
    s[s_state_h].set_scope(tbe_platform.scope_ubuf)
    s[s_state_h_fp16].set_scope(tbe_platform.scope_ubuf)
    s[a_l1_1].set_scope(tbe_platform.scope_cbuf)
    s[b_l1_1].set_scope(tbe_platform.scope_cbuf)
    s[a_l0a_1].set_scope(tbe_platform.scope_ca)
    s[b_l0b_1].set_scope(tbe_platform.scope_cb)
    s[c_l0c_1].set_scope(tbe_platform.scope_cc)
    s[c_ub_1].set_scope(tbe_platform.scope_ubuf)
    s[bias_ub_1].set_scope(tbe_platform.scope_ubuf)
    s[bias_bc_ub_1].set_scope(tbe_platform.scope_ubuf)
    s[r_t_h_fp16].set_scope(tbe_platform.scope_ubuf)
    s[a_l1_2].set_scope(tbe_platform.scope_cbuf)
    s[b_l1_2].set_scope(tbe_platform.scope_cbuf)
    s[a_l0a_2].set_scope(tbe_platform.scope_ca)
    s[b_l0b_2].set_scope(tbe_platform.scope_cb)
    s[c_l0c_2].set_scope(tbe_platform.scope_cc)
    s[c_ub_2].set_scope(tbe_platform.scope_ubuf)
    s[bias_ub_2].set_scope(tbe_platform.scope_ubuf)
    s[bias_bc_ub_2].set_scope(tbe_platform.scope_ubuf)
    s[update_y_gm_back].set_scope(tbe_platform.scope_ubuf)
    if is_gate_output:
        s[r_t_gm_back].set_scope(tbe_platform.scope_ubuf)
        s[i_t_gm_back].set_scope(tbe_platform.scope_ubuf)
        s[n_t_gm_back].set_scope(tbe_platform.scope_ubuf)
        if fp16_input_output:
            s[r_t_sigmoid_fp16].set_scope(tbe_platform.scope_ubuf)
            s[i_t_sigmoid_fp16].set_scope(tbe_platform.scope_ubuf)
            s[h_t_tanh_fp16].set_scope(tbe_platform.scope_ubuf)
            s[r_t_gm_back_fp32].set_scope(tbe_platform.scope_ubuf)
            s[i_t_gm_back_fp32].set_scope(tbe_platform.scope_ubuf)
            s[n_t_gm_back_fp32].set_scope(tbe_platform.scope_ubuf)
    if fp16_input_output:
        s[bias_ub_1_fp32].set_scope(tbe_platform.scope_ubuf)
        s[bias_ub_2_fp32].set_scope(tbe_platform.scope_ubuf)
        s[update_h_fp16].set_scope(tbe_platform.scope_ubuf)

    # compute inline
    compute_inline_tensors = [i_t, r_t]
    for tensor in compute_inline_tensors:
        s[tensor].compute_inline()

    # matmul tiling
    factor_l1_m, factor_l1_n, factor_l1_k, factor_l0_m, factor_l0_n, factor_l0_k = \
        _get_tiling(m_size, k_size, hidden_size)

    l1_n_outer_1, l1_n_inner_1 = s[c_l0c_1].split(c_l0c_1.op.axis[2], factor=factor_l1_n)
    l1_m_outer_1, l1_m_inner_1 = s[c_l0c_1].split(c_l0c_1.op.axis[3], factor=factor_l1_m)
    l1_k_outer_1, l1_k_inner_1 = s[c_l0c_1].split(c_l0c_1.op.reduce_axis[0], factor=factor_l1_k)
    l0_n_outer_1, l0_n_inner_1 = s[c_l0c_1].split(l1_n_inner_1, factor=factor_l0_n)
    l0_m_outer_1, l0_m_inner_1 = s[c_l0c_1].split(l1_m_inner_1, factor=factor_l0_m)
    l0_k_outer_1, l0_k_inner_1 = s[c_l0c_1].split(l1_k_inner_1, factor=factor_l0_k)
    s[c_l0c_1].reorder(c_l0c_1.op.axis[0],
                       l1_n_outer_1,
                       l1_k_outer_1,
                       c_l0c_1.op.axis[1],
                       l1_m_outer_1,
                       l0_n_outer_1,
                       l0_m_outer_1,
                       l0_k_outer_1,
                       l0_n_inner_1,
                       l0_m_inner_1,
                       c_l0c_1.op.axis[4],
                       c_l0c_1.op.axis[5],
                       l0_k_inner_1,
                       c_l0c_1.op.reduce_axis[1])
    s[a_l1_1].double_buffer()
    s[b_l1_1].double_buffer()
    s[a_l0a_1].double_buffer()
    s[b_l0b_1].double_buffer()
    s[c_l0c_1].double_buffer()
    s[c_ub_1].double_buffer()
    s[a_l1_1].compute_at(s[c_l0c_1], l1_k_outer_1)
    s[b_l1_1].compute_at(s[c_l0c_1], c_l0c_1.op.axis[1])
    s[a_l0a_1].compute_at(s[c_l0c_1], l1_k_outer_1)
    s[b_l0b_1].compute_at(s[c_l0c_1], l0_k_outer_1)

    c_ub_bias_1_outer, c_ub_bias_1_inner = s[c_ub_bias_1].split(c_ub_bias_1.op.axis[2], factor=factor_l1_n)
    s[c_ub_bias_1].reorder(c_ub_bias_1.op.axis[0],
                           c_ub_bias_1_outer,
                           c_ub_bias_1.op.axis[1],
                           c_ub_bias_1_inner,
                           c_ub_bias_1.op.axis[3],
                           c_ub_bias_1.op.axis[4],
                           c_ub_bias_1.op.axis[5])
    s[c_l0c_1].compute_at(s[c_ub_bias_1], c_ub_bias_1_outer)
    s[c_ub_1].compute_at(s[c_ub_bias_1], c_ub_bias_1_outer)
    s[bias_ub_1].compute_at(s[c_ub_bias_1], c_ub_bias_1_outer)
    s[bias_bc_ub_1].compute_at(s[c_ub_bias_1], c_ub_bias_1_outer)
    if fp16_input_output:
        s[bias_ub_1_fp32].compute_at(s[c_ub_bias_1], c_ub_bias_1_outer)
    s[c_ub_bias_1].emit_insn(c_ub_bias_1.op.axis[1], 'vector_add')

    r_t_h_fp16_outer, r_t_h_fp16_inner = s[r_t_h_fp16].split(r_t_h_fp16.op.axis[1], factor=factor_l1_n)
    for tensor in elewise_tensors_r_t_h_fp16:
        s[tensor].set_scope(tbe_platform.scope_ubuf)
        if tensor == c_ub_bias_1:
            continue
        s[tensor].compute_at(s[r_t_h_fp16], r_t_h_fp16_outer)
        insn = _get_emit_insn_map(tensor)
        s[tensor].emit_insn(tensor.op.axis[0], insn)
    if is_gate_output:
        s[r_t_gm].compute_at(s[r_t_h_fp16], r_t_h_fp16_outer)
        s[r_t_gm_back].compute_at(s[r_t_h_fp16], r_t_h_fp16_outer)
        if fp16_input_output:
            s[r_t_sigmoid_fp16].compute_at(s[r_t_h_fp16], r_t_h_fp16_outer)
            s[r_t_gm_back_fp32].compute_at(s[r_t_h_fp16], r_t_h_fp16_outer)
    s[r_t_h_fp16].emit_insn(r_t_h_fp16_inner, 'vector_conv')

    l1_n_outer_2, l1_n_inner_2 = s[c_l0c_2].split(c_l0c_2.op.axis[2], factor=factor_l1_n)
    l1_m_outer_2, l1_m_inner_2 = s[c_l0c_2].split(c_l0c_2.op.axis[3], factor=factor_l1_m)
    l1_k_outer_2, l1_k_inner_2 = s[c_l0c_2].split(c_l0c_2.op.reduce_axis[0], factor=factor_l1_k)
    l0_n_outer_2, l0_n_inner_2 = s[c_l0c_2].split(l1_n_inner_2, factor=factor_l0_n)
    l0_m_outer_2, l0_m_inner_2 = s[c_l0c_2].split(l1_m_inner_2, factor=factor_l0_m)
    l0_k_outer_2, l0_k_inner_2 = s[c_l0c_2].split(l1_k_inner_2, factor=factor_l0_k)
    s[c_l0c_2].reorder(c_l0c_2.op.axis[0],
                       l1_n_outer_2,
                       l1_k_outer_2,
                       c_l0c_2.op.axis[1],
                       l1_m_outer_2,
                       l0_n_outer_2,
                       l0_m_outer_2,
                       l0_k_outer_2,
                       l0_n_inner_2,
                       l0_m_inner_2,
                       c_l0c_2.op.axis[4],
                       c_l0c_2.op.axis[5],
                       l0_k_inner_2,
                       c_l0c_2.op.reduce_axis[1])
    s[a_l1_2].double_buffer()
    s[b_l1_2].double_buffer()
    s[a_l0a_2].double_buffer()
    s[b_l0b_2].double_buffer()
    s[c_l0c_2].double_buffer()
    s[c_ub_2].double_buffer()
    s[a_l1_2].compute_at(s[c_l0c_2], l1_k_outer_2)
    s[b_l1_2].compute_at(s[c_l0c_2], c_l0c_2.op.axis[1])
    s[a_l0a_2].compute_at(s[c_l0c_2], l1_k_outer_2)
    s[b_l0b_2].compute_at(s[c_l0c_2], l0_k_outer_2)

    update_h_gm_outer, update_h_gm_inner = s[update_h_gm].split(update_h_gm.op.axis[1], factor=factor_l1_n)
    s[c_l0c_2].compute_at(s[update_h_gm], update_h_gm_outer)
    s[c_ub_2].compute_at(s[update_h_gm], update_h_gm_outer)
    s[bias_ub_2].compute_at(s[update_h_gm], update_h_gm_outer)
    s[bias_bc_ub_2].compute_at(s[update_h_gm], update_h_gm_outer)
    s[c_ub_bias_2].compute_at(s[update_h_gm], update_h_gm_outer)
    s[update_y_gm].compute_at(s[update_h_gm], update_h_gm_outer)
    s[update_y_gm_back].compute_at(s[update_h_gm], update_h_gm_outer)
    if fp16_input_output:
        s[bias_ub_2_fp32].compute_at(s[update_h_gm], update_h_gm_outer)
        s[update_h_fp16].compute_at(s[update_h_gm], update_h_gm_outer)
    if is_gate_output:
        s[i_t_gm].compute_at(s[update_h_gm], update_h_gm_outer)
        s[i_t_gm_back].compute_at(s[update_h_gm], update_h_gm_outer)
        s[n_t_gm].compute_at(s[update_h_gm], update_h_gm_outer)
        s[n_t_gm_back].compute_at(s[update_h_gm], update_h_gm_outer)
        if fp16_input_output:
            s[i_t_sigmoid_fp16].compute_at(s[update_h_gm], update_h_gm_outer)
            s[i_t_gm_back_fp32].compute_at(s[update_h_gm], update_h_gm_outer)
            s[h_t_tanh_fp16].compute_at(s[update_h_gm], update_h_gm_outer)
            s[n_t_gm_back_fp32].compute_at(s[update_h_gm], update_h_gm_outer)

    for tensor in elewise_tensors:
        s[tensor].set_scope(tbe_platform.scope_ubuf)
        s[tensor].compute_at(s[update_h_gm], update_h_gm_outer)
        insn = _get_emit_insn_map(tensor)
        s[tensor].emit_insn(tensor.op.axis[0], insn)

    # emit insn
    if is_first_round and not is_global_init:
        s[s_state_h].emit_insn(s_state_h.op.axis[0], 'broadcast')
        s[s_state_h_fp16].emit_insn(s_state_h_fp16.op.axis[0], 'vector_conv')
    else:
        if fp16_input_output:
            s[s_state_h_fp16].emit_insn(s_state_h_fp16.op.axis[0], 'dma_copy')
            s[s_state_h].emit_insn(s_state_h.op.axis[0], 'vector_conv')
        else:
            s[s_state_h].emit_insn(s_state_h.op.axis[0], 'dma_copy')
            s[s_state_h_fp16].emit_insn(s_state_h_fp16.op.axis[0], 'vector_conv')

    s[a_l1_1].emit_insn(a_l1_1.op.axis[0], 'dma_copy')
    s[b_l1_1].emit_insn(b_l1_1.op.axis[0], 'dma_copy')
    s[a_l0a_1].emit_insn(a_l0a_1.op.axis[0], 'dma_copy')
    s[b_l0b_1].emit_insn(b_l0b_1.op.axis[0], 'dma_copy')
    mad_dict = {"mad_pattern": 0, "k_outer": [l1_k_outer_1, l0_k_outer_1]}
    s[c_l0c_1].emit_insn(l0_n_inner_1, 'mad', mad_dict)
    s[c_ub_1].emit_insn(c_ub_1.op.axis[0], 'dma_copy')
    s[bias_ub_1].emit_insn(bias_ub_1.op.axis[0], 'dma_copy')
    if fp16_input_output:
        s[bias_ub_1_fp32].emit_insn(bias_ub_1_fp32.op.axis[0], 'vector_conv')
        s[bias_ub_2_fp32].emit_insn(bias_ub_2_fp32.op.axis[0], 'vector_conv')
        s[update_h_fp16].emit_insn(update_h_fp16.op.axis[0], 'vector_conv')
    s[bias_bc_ub_1].emit_insn(bias_bc_ub_1.op.axis[0], 'unified_broadcast')
    s[a_l1_2].emit_insn(a_l1_2.op.axis[0], 'dma_copy')
    s[b_l1_2].emit_insn(b_l1_2.op.axis[0], 'dma_copy')
    s[a_l0a_2].emit_insn(a_l0a_2.op.axis[0], 'dma_copy')
    s[b_l0b_2].emit_insn(b_l0b_2.op.axis[0], 'dma_copy')
    mad_dict = {"mad_pattern": 0, "k_outer": [l1_k_outer_2, l0_k_outer_2]}
    s[c_l0c_2].emit_insn(l0_n_inner_2, 'mad', mad_dict)
    s[c_ub_2].emit_insn(c_ub_2.op.axis[0], 'dma_copy')
    s[bias_ub_2].emit_insn(bias_ub_2.op.axis[0], 'dma_copy')
    s[bias_bc_ub_2].emit_insn(bias_bc_ub_2.op.axis[0], 'unified_broadcast')
    s[update_y_gm].emit_insn(update_y_gm.op.axis[0], 'dma_copy')
    s[update_y_gm_back].emit_insn(update_y_gm_back.op.axis[0], 'phony_insn')
    s[update_y_gm_back].reused_by(update_h_ub)
    if is_gate_output:
        s[r_t_gm].emit_insn(r_t_gm.op.axis[0], 'dma_copy')
        s[i_t_gm].emit_insn(i_t_gm.op.axis[0], 'dma_copy')
        s[n_t_gm].emit_insn(n_t_gm.op.axis[0], 'dma_copy')
        s[r_t_gm_back].emit_insn(r_t_gm_back.op.axis[0], 'phony_insn')
        s[i_t_gm_back].emit_insn(i_t_gm_back.op.axis[0], 'phony_insn')
        s[n_t_gm_back].emit_insn(n_t_gm_back.op.axis[0], 'phony_insn')
        if fp16_input_output:
            s[r_t_sigmoid_fp16].emit_insn(r_t_sigmoid_fp16.op.axis[0], 'vector_conv')
            s[i_t_sigmoid_fp16].emit_insn(i_t_sigmoid_fp16.op.axis[0], 'vector_conv')
            s[h_t_tanh_fp16].emit_insn(h_t_tanh_fp16.op.axis[0], 'vector_conv')
            s[r_t_gm_back_fp32].emit_insn(r_t_gm_back_fp32.op.axis[0], 'phony_insn')
            s[i_t_gm_back_fp32].emit_insn(i_t_gm_back_fp32.op.axis[0], 'phony_insn')
            s[n_t_gm_back_fp32].emit_insn(n_t_gm_back_fp32.op.axis[0], 'phony_insn')
            s[r_t_gm_back_fp32].reused_by(r_t_sigmoid)
            s[i_t_gm_back_fp32].reused_by(i_t_sigmoid)
            s[n_t_gm_back_fp32].reused_by(h_t_tanh)
            s[r_t_gm_back].reused_by(r_t_sigmoid_fp16)
            s[i_t_gm_back].reused_by(i_t_sigmoid_fp16)
            s[n_t_gm_back].reused_by(h_t_tanh_fp16)
        else:
            s[r_t_gm_back].reused_by(r_t_sigmoid)
            s[i_t_gm_back].reused_by(i_t_sigmoid)
            s[n_t_gm_back].reused_by(h_t_tanh)
    s[update_h_gm].emit_insn(update_h_gm_inner, 'dma_copy')

    output_list = [update_y_gm, update_h_gm]
    if is_gate_output:
        output_list.append(r_t_gm)
        output_list.append(i_t_gm)
        output_list.append(n_t_gm)
    return output_list, s
