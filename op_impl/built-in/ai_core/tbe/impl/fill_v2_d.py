import te.lang.cce
import te.platform.cce_params as cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils import op_utils
from topi.cce import util
from te.platform import cce_util
import types


# Alignment requirement
ALIGNMENT_BYTES = 32


def _do_buffer_tile(self):
    self._elewise_binary_phony_as_output = False


def _get_emit_insn_map(self):
    self._insn_map = {"elewise_single_cast": "vector_conv",
                      "elewise_single_round_d": "vector_conv_round",
                      "elewise_single_trunc": "vector_conv_trunc",
                      "elewise_single_VS_max": "vector_maxs",
                      "elewise_single_VS_min": "vector_mins",
                      "elewise_single_log": "vector_ln",
                      "elewise_single_exp": "vector_exp",
                      "elewise_single_relu": "vector_relu",
                      "elewise_single_lrelu": "vector_lrelu",
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
                      "elewise_binary_cmpsel_gt": "vector_select_gt",
                      "elewise_binary_cmpsel_ge": "vector_select_ge",
                      "elewise_binary_cmpsel_lt": "vector_select_lt",
                      "elewise_binary_cmpsel_le": "vector_select_le",
                      "elewise_binary_cmpsel_eq": "vector_select_eq",
                      "elewise_binary_cmpsel_ne": "vector_select_ne",
                      "elewise_binary_or": "vector_or",
                      "elewise_binary_and": "vector_and",
                      "elewise_binary_addrelu": "vector_addrelu",
                      "elewise_binary_subrelu": "vector_subrelu",
                      "elewise_multiple_mla": "vector_multiple",
                      "elewise_multiple_madd": "vector_multiple",
                      "elewise_multiple_maddrelu": "vector_multiple",
                      "elewise_multiple_sel": "vector_select_bool",
                      "elewise_binary_sub": "vector_sub",
                      "elewise_binary_phony": "elewise_binary_phony_ex"}


@tvm.register_func("tvm.intrin.cce.elewise_binary_phony_ex")
def elewise_binary_phony_ex(stmt_op):
    """
    elewise_binary_phony_ex which will eliminate its second input tensor completely
    """
    ins, outs, _, _ = cce_util.get_dma_buffer(stmt_op)
    ir_builder = tvm.ir_builder.create()

    def new_alloc(ir_builder, dtype, shape, name, scope):
        """
        new_alloc
        """
        buf_var = ir_builder.allocate(dtype, shape, name=name, scope=scope)
        new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name, scope=scope, data=buf_var)

        return new_buffer

    # Move first input to out
    dtype = ins[0].dtype
    total_element = 0
    for dim in ins[0].shape:
        if total_element == 0:
            total_element = dim
        else:
            total_element *= dim
    _block_unit_size = ALIGNMENT_BYTES // cce_util.get_align_factor(dtype)[1]
    total_block = int(total_element) // int(_block_unit_size)
    remain = int(total_element % _block_unit_size)

    if total_block > 0:
        ir_builder.emit(tvm.call_extern(
            ins[0].dtype, "copy_ubuf_to_gm",
            outs[0].access_ptr("rw"),
            ins[0].access_ptr("r"),
            0, 1, total_block, 0, 0))

    if remain > 0 and total_block > 0:
        # Roll back for remaining data
        roll_back_size = _block_unit_size - remain

        # Allocate reg buffer needed for holding src data
        reg = new_alloc(ir_builder,
                        ins[0].dtype,
                        (_block_unit_size,),
                        "copy_part",
                        scope=cce.scope_ubuf)

        # reg_mov src data
        with ir_builder.for_range(0, _block_unit_size, name="reg_idx") as reg_idx:
            ir_builder.emit(tvm.call_extern(
                ins[0].dtype, "reg_mov",
                reg.access_ptr("rw", offset=reg_idx),
                ins[0].access_ptr("r", offset=total_block*_block_unit_size-roll_back_size+reg_idx)))
        ir_builder.emit(tvm.call_extern(
            ins[0].dtype, "copy_ubuf_to_gm",
            outs[0].access_ptr("rw", offset=total_block*_block_unit_size-roll_back_size),
            reg.access_ptr("r"),
            0, 1, 1, 0, 0))

    if remain > 0 and total_block == 0:
        ir_builder.emit(tvm.call_extern(
            ins[0].dtype, "copy_ubuf_to_gm",
            outs[0].access_ptr("rw", offset=0),
            ins[0].access_ptr("r", offset=0),
            0, 1, 1, 0, 0))
    return ir_builder.get()


@fusion_manager.register("fill_v2_d")
def fill_v2_compute(data_x, x1, x2, y, kernel_name="fill_v2_d"):
    # broadcast
    res = te.lang.cce.broadcast(tvm.const(x1), x2)

    with tvm.tag_scope("elewise_binary_phony"):
        res = te.tvm.compute(res.shape,
                             lambda *indices: res[indices] + data_x[indices],
                             name="elewise_binary_phony_output")

    return res


@op_utils.check_op_params(dict, float, (list, tuple), str)
def fill_v2_d(y, value, shape, kernel_name="fill_v2_d"):
    """
    interface of fill_v2_d
    :param y: output
    :param value: value to fill the shape, float32
    :param shape: list int, output shape
    :param kernel_name: fill_v2_d
    :return:
    """
    # check kernel name
    util.check_kernel_name(kernel_name)
    # shape to list
    shape = te.lang.cce.util.shape_to_list(shape)
    util.check_shape_rule(shape)

    # pseudo input, won't be used.
    data_x = tvm.placeholder(shape, dtype="float32", name="data_x")

    # do compute
    res = fill_v2_compute(data_x, value, shape, y, kernel_name)

    # new schedule
    schedule = [tvm.create_schedule(res.op)]
    elewise_sch = te.lang.cce.te_schedule.cce_schedule.ElewiseSchedule()
    elewise_sch._get_emit_insn_map = types.MethodType(_get_emit_insn_map, elewise_sch)
    elewise_sch._do_buffer_tile = types.MethodType(_do_buffer_tile, elewise_sch)
    elewise_sch.do_schedule([res], schedule, [])
    schedule = schedule[0]
    schedule.cce_special = {"tensor_list": (), "orign_out_tensor": [res], "real_out_tensor": [res]}

    # build operater
    config = {"name": kernel_name,
              "tensor_list": (data_x, res)}
    te.lang.cce.cce_build_code(schedule, config)
