#!/usr/bin/python
# -*- coding: utf-8 -*-

from te.platform.fusion_manager import fusion_manager
from te import tik
from topi.cce import util
from functools import reduce as functools_reduce

PROPOSAL_NUM = 8
FP16_BYTE = 2
MAX_NUM = 7040


@fusion_manager.register("sort")
def cheak(x, y1, y2, axis, kernel_name):
    """
    Function: Check parameters (eg: shape dtype etc).
    Modify : 2020-08-03
    """
    util.check_kernel_name(kernel_name)

    shape = y1.get("shape")
    dtype = y1.get("dtype").lower()
    util.check_dtype_rule(dtype, ("float16"))
    util.check_shape_rule(shape)

    shape = y2.get("shape")
    dtype = y2.get("dtype").lower()
    util.check_dtype_rule(dtype, ("int32"))
    util.check_shape_rule(shape)

    shape = x.get("shape")
    dtype = x.get("dtype").lower()
    util.check_dtype_rule(dtype, ("float16"))
    util.check_shape_rule(shape)

    if axis == -1:
        axis = len(shape) - 1

    if axis != len(shape) - 1:
        raise RuntimeError("Dim should take the last one.")

    allnum = functools_reduce(lambda x, y: x * y, shape)

    num = shape[axis]

    if num > MAX_NUM:
        raise RuntimeError("Num in dim is too big (>7040).")

    return shape, dtype, allnum, num


def vbs16(tik_instance, num, total, input_ub, descending):
    """
    Function: Sort every 16 numsi in UB.
    Modify : 2020-08-03

    Init base parameters
    Parameters
    ----------
    num: The number of effective object.
    total: The number of all object (16 alignment).
    input_ub: UB
    ----------
    """
    Max = tik_instance.Scalar('float16', init_value=65504)
    Min = tik_instance.Scalar('float16', init_value=-65504)
    # Add ineffective object for 16 alignment
    if descending:
        with tik_instance.for_range(0, total - num) as i:
            input_ub[(num + i) * PROPOSAL_NUM + 4].set_as(Min)
    else:
        with tik_instance.for_range(0, total - num) as i:
            input_ub[(num + i) * PROPOSAL_NUM + 4].set_as(Max)

    # dest position in UB
    dest_pos_ub = total * PROPOSAL_NUM
    n_repeat_total = total // 16

    if int(n_repeat_total) > 255:
        tik_instance.vrpsort16(dst=input_ub[dest_pos_ub], src=input_ub[0], repeat_times=255)
        tik_instance.vrpsort16(dst=input_ub[dest_pos_ub + 255 * 16 * PROPOSAL_NUM],
                               src=input_ub[255 * 16 * PROPOSAL_NUM], repeat_times=n_repeat_total - 255)
    else:
        tik_instance.vrpsort16(dst=input_ub[dest_pos_ub], src=input_ub[0], repeat_times=n_repeat_total)

    return input_ub, dest_pos_ub


def merge4(tik_instance, num_list, input_ub, offset, src_pos_ub, index, dest_pos_ub):
    """
    Function: Merge 4 lists in UB.
    Modify : 2020-08-03

    Init base parameters
    Parameters
    ----------
    num_list: record the lists info
    offset: used space
    src_pos_ub, dest_pos_ub: position info
    input_ub: UB
    ----------
    """
    src_list = [input_ub[src_pos_ub + offset * PROPOSAL_NUM],
                input_ub[src_pos_ub + (offset + num_list[index]) * PROPOSAL_NUM],
                input_ub[src_pos_ub + (offset + num_list[index] + num_list[index + 1]) * PROPOSAL_NUM],
                input_ub[
                    src_pos_ub + (offset + num_list[index] + num_list[index + 1] + num_list[index + 2]) * PROPOSAL_NUM]]

    src_list_lengths = [num_list[index], num_list[index + 1], num_list[index + 2], num_list[index + 3]]
    # merge 4 lists
    tik_instance.vmrgsort4(input_ub[dest_pos_ub + offset * PROPOSAL_NUM], src_list, src_list_lengths,
                           if_exhausted_suspension=False, valid_bit="1111", repeat_times=1)
    # update the lists info : Merge the four element values and record them in a(num_list)
    num_list[index] = sum(num_list[index:index + 4])
    a = num_list[:index + 1:]
    b = num_list[index + 4::]
    a.extend(b)
    offset += a[index]

    return a, input_ub, offset


def merge3(tik_instance, num_list, input_ub, offset, src_pos_ub, index, dest_pos_ub):
    """
    Function: Merge 3 lists in UB.
    Modify : 2020-08-03

    Init base parameters
    Parameters
    ----------
    num_list: record the lists info
    offset: used space
    src_pos_ub, dest_pos_ub: position info
    input_ub: UB
    ----------
    """
    src_list = [input_ub[src_pos_ub + offset * PROPOSAL_NUM],
                input_ub[src_pos_ub + (offset + num_list[index]) * PROPOSAL_NUM],
                input_ub[src_pos_ub + (offset + num_list[index] + num_list[index + 1]) * PROPOSAL_NUM], input_ub[0]]
    src_list_lengths = [num_list[index], num_list[index + 1], num_list[index + 2], 0]
    # merge 3 lists
    tik_instance.vmrgsort4(input_ub[dest_pos_ub + offset * PROPOSAL_NUM], src_list, src_list_lengths,
                           if_exhausted_suspension=False, valid_bit="0111", repeat_times=1)
    # update the lists info : Merge the three element values and record them in a(num_list)
    num_list[index] = sum(num_list[index:index + 3])
    a = num_list[:index + 1:]
    b = num_list[index + 3::]
    a.extend(b)
    offset += a[index]

    return a, input_ub, offset


def merge2(tik_instance, num_list, input_ub, offset, src_pos_ub, index, dest_pos_ub):
    """
    Function: Merge 2 lists in UB.
    Modify : 2020-08-03

    Init base parameters
    Parameters
    ----------
    num_list: record the lists info
    offset: used space
    src_pos_ub, dest_pos_ub: position info
    input_ub: UB
    ----------
    """
    src_list = [input_ub[src_pos_ub + offset * PROPOSAL_NUM],
                input_ub[src_pos_ub + (offset + num_list[index]) * PROPOSAL_NUM],
                input_ub[0], input_ub[0]]
    src_list_lengths = [num_list[index], num_list[index + 1], 0, 0]
    # merge 2 lists
    tik_instance.vmrgsort4(input_ub[dest_pos_ub + offset * PROPOSAL_NUM], src_list, src_list_lengths,
                           if_exhausted_suspension=False, valid_bit="0011", repeat_times=1)

    # update the lists info : Merge the two element values and record them in num_list
    num_list[index] += num_list[index + 1]
    del num_list[index + 1]
    offset += num_list[index]

    return num_list, input_ub, offset


def vms4(tik_instance, num, total, input_ub, dest_pos_ub):
    """
    Function: Merge all lists into one.
    Modify : 2020-08-03

    Init base parameters
    Parameters
    ----------
    num: The number of effective object.
    total: The number of all object (16 alignment).
    input_ub: UB
    dest_pos_ub: The dest position in UB.
    ----------
    """
    # record the lists info
    length = total // 16
    num_list = [16] * length

    # over 4096
    if length > 256:
        # leftset rightset : num_list's valid room
        input_ub, _, num_list1 = vms4core(tik_instance, input_ub, dest_pos_ub, 0, length // 2, num_list)
        input_ub, _, num_list2 = vms4core(tik_instance, input_ub, dest_pos_ub, length // 2, length, num_list)

        num_list1.extend(num_list2)

        src_pos_ub, dest_pos_ub = dest_pos_ub, 0
        _, input_ub, _ = merge2(tik_instance, num_list1, input_ub, 0, src_pos_ub, 0, dest_pos_ub)
        return input_ub, dest_pos_ub

    else:
        input_ub, dest_pos_ub, num_list = vms4core(tik_instance, input_ub, dest_pos_ub, 0, length, num_list)
        return input_ub, dest_pos_ub


def vms4core(tik_instance, input_ub, dest_pos_ub, leftset, rightset, num_list):
    """
    Function: Merge core.
    Modify : 2020-08-03

    Init base parameters
    Parameters
    ----------
    input_ub: UB
    dest_pos_ub: The dest position in UB.
    leftset, rightset: The valid room.
    num_list : Lists info
    ----------
    """
    src_pos_ub = 0
    num_list = num_list[leftset:rightset]
    offset_temp = leftset * 16
    while len(num_list) > 1:
        src_pos_ub, dest_pos_ub = dest_pos_ub, src_pos_ub
        index = 0
        offset = offset_temp
        while True:
            res = len(num_list) - index
            if res > 3:
                num_list, input_ub, offset = merge4(tik_instance, num_list, input_ub,
                                                    offset, src_pos_ub, index, dest_pos_ub)
            elif res == 3:
                num_list, input_ub, offset = merge3(tik_instance, num_list, input_ub,
                                                    offset, src_pos_ub, index, dest_pos_ub)
            elif res == 2:
                num_list, input_ub, offset = merge2(tik_instance, num_list, input_ub,
                                                    offset, src_pos_ub, index, dest_pos_ub)
            elif res == 1:
                tik_instance.data_move(input_ub[dest_pos_ub + offset * PROPOSAL_NUM],
                                       input_ub[src_pos_ub + offset * PROPOSAL_NUM], 0, 1,
                                       num_list[index] * PROPOSAL_NUM // 16, 0, 0)
            else:
                break
            index += 1

    return input_ub, dest_pos_ub, num_list


def moveout(tik_instance, descending, total, num, data_out, index, input_ub, dest_pos_ub, data_indices, threadNum):
    """
    Function: Move UB to GM, and trans y2 from fp16 to int32.
    Modify : 2020-08-03

    Attention : This way is unstable (can't compare two scalar).
    Init base parameters
    Parameters
    ----------
    descending, index, total(add16+num), num, dest_pos_ub : for index compute
    data_out, input_ub, data_indices : for data move
    ----------
    """
    int_list = tik_instance.Tensor("int32", [total], name="data_indices_ub_list", scope=tik.scope_ubuf)

    src_pos_ub = total * PROPOSAL_NUM if dest_pos_ub == 0 else 0
    # ascend
    with tik_instance.if_scope(descending is False):
        # data is continuous in GM & gather scattered data together
        with tik_instance.for_range(0, num, thread_num=threadNum) as i2:
            input_ub[i2 + src_pos_ub].set_as(input_ub[(total - 1 - i2) * PROPOSAL_NUM + 4 + dest_pos_ub])
            input_ub[i2 + src_pos_ub + total].set_as(input_ub[(total - 1 - i2) * PROPOSAL_NUM + dest_pos_ub])
        # move output (float16) from UB to GM
        tik_instance.data_move(data_out[index], input_ub[src_pos_ub], 0, 1, total // 16, 0, 0)
        # conv indices (float16->int32) , and move from UB to GM
        if (total > 255 * 16):
            tik_instance.vec_conv(16, "round", int_list, input_ub[src_pos_ub + total], 255, 2, 1)
            tik_instance.vec_conv(16, "round", int_list[255 * 16], input_ub[src_pos_ub + total + 255 * 16],
                                  total // 16 - 255, 2, 1)
        else:
            tik_instance.vec_conv(16, "round", int_list, input_ub[src_pos_ub + total], total // 16, 2, 1)

        tik_instance.data_move(data_indices[index], int_list, 0, 1, total // 8, 0, 0)

    # descend
    with tik_instance.else_scope():
        # data is continuous in GM & gather scattered data together
        with tik_instance.for_range(0, num, thread_num=threadNum) as i2:
            input_ub[i2 + src_pos_ub].set_as(input_ub[i2 * PROPOSAL_NUM + 4 + dest_pos_ub])
            input_ub[i2 + src_pos_ub + total].set_as(input_ub[i2 * PROPOSAL_NUM + dest_pos_ub])
        # move output (float16) from UB to GM
        tik_instance.data_move(data_out[index], input_ub[src_pos_ub], 0, 1, total // 16, 0, 0)
        # conv indices (float16->int32) , and move from UB to GM
        if (total > 255 * 16):
            tik_instance.vec_conv(16, "round", int_list, input_ub[src_pos_ub + total], 255, 2, 1)
            tik_instance.vec_conv(16, "round", int_list[255 * 16], input_ub[src_pos_ub + total + 255 * 16],
                                  total // 16 - 255, 2, 1)
        else:
            tik_instance.vec_conv(16, "round", int_list, input_ub[src_pos_ub + total], total // 16, 2, 1)

        tik_instance.data_move(data_indices[index], int_list, 0, 1, total // 8, 0, 0)

    return data_out, data_indices


def sort_compute(tik_instance, dtype, total, i0, descending, num, distance, shape, big_distance, data_out, data_indices,
                 input_gm, L):
    """
    Function: sortcompute in UB.
    Modify : 2020-08-03

    Attention : This way is unstable (can't compare two scalar).
    Init base parameters
    Parameters
    ----------
    dtype, total, i0, descending, num, distance, shape, big_distance, L : for index compute
    data_out, data_indices, input_gm : for data move
    ----------
    """

    input_ub = tik_instance.Tensor(dtype, [total * PROPOSAL_NUM * 2], name="input_ub", scope=tik.scope_ubuf)
    data_out_ub_ = tik_instance.Tensor(dtype, [16], name="data_out_ub_", scope=tik.scope_ubuf)
    data_indices_ub_int_ = tik_instance.Tensor("int32", [16], name="data_indices_ub_int_", scope=tik.scope_ubuf)

    index = 0
    big_index = 0
    for i1 in range(L - 1):
        index += (i0 % shape[i1]) * distance[i1]
        big_index += (i0 % shape[i1]) * big_distance[i1]
        i0 = i0 // shape[i1]

    # 1. Move data from OUT to UB
    tik_instance.data_move(input_ub[0], input_gm[index], 0, 1, total // 16, 0, 0)
    threadNum = 2 if num > 1 else 1
    with tik_instance.for_range(0, num, thread_num=threadNum) as i2:
        input_ub[(num - 1 - i2) * PROPOSAL_NUM + 4].set_as(input_ub[(num - 1 - i2)])
        data_indices_ub_int_.set_as(num - 1 - i2)
        tik_instance.vec_conv(1, "none", data_out_ub_, data_indices_ub_int_, 1, 0, 0, deqscale=1.0)
        input_ub[(num - 1 - i2) * PROPOSAL_NUM].set_as(data_out_ub_[0])

    # 2. vbs16
    input_ub, dest_pos_ub = vbs16(tik_instance, num, total, input_ub, descending)

    # 3. vms4
    input_ub, dest_pos_ub = vms4(tik_instance, num, total, input_ub, dest_pos_ub)

    # 4. Move Data from UB to OUT
    data_out, data_indices = moveout(tik_instance, descending, total, num, data_out, big_index, input_ub,
                                     dest_pos_ub, data_indices, threadNum)

    return data_out, data_indices


@util.check_input_type(dict, dict, dict, int, bool, str)
def sort(x, y1, y2, axis=-1, descending=False, kernel_name="sort"):
    """
    Function: Sorts the elements of the input tensor along a given dimension in ascending order by value.
    Modify : 2020-08-03

    Init base parameters
    Parameters
    ----------
    input(x): dict
        data of input
    output(y1): dict
        data of output
    indices(y2): dict
        data of indices
    dim(axis): int
    descending: bool
    kernel_name: str
        the name of the operator
    ----------
    """
    shape, dtype, allnum, num = cheak(x, y1, y2, axis, kernel_name)

    tik_instance = tik.Tik(tik.Dprofile())

    add16 = (16 - (num % 16)) % 16
    total = num + add16

    big_shape = list(shape)
    big_shape[-1] = total

    input_gm = tik_instance.Tensor(dtype, shape, name="x", scope=tik.scope_gm)
    data_out = tik_instance.Tensor(dtype, big_shape, name="data_out", scope=tik.scope_gm, is_workspace=True)
    data_indices = tik_instance.Tensor("int32", big_shape, name="data_indices", scope=tik.scope_gm, is_workspace=True)
    data_out_ = tik_instance.Tensor(dtype, shape, name="data_out_", scope=tik.scope_gm)
    data_indices_ = tik_instance.Tensor("int32", shape, name="data_indices_", scope=tik.scope_gm)

    # to figure the index of input_gm
    L = len(shape)
    distance = []
    big_distance = []
    tmp = allnum
    big_tmp = allnum // num * total

    for i in range(L - 1):
        tmp = tmp // shape[i]
        distance.append(tmp)
        big_tmp = big_tmp // shape[i]
        big_distance.append(big_tmp)

    rounds = allnum // num

    available_aicore_num = tik.Dprofile().get_aicore_num()
    used_aicore_num = available_aicore_num if rounds > available_aicore_num else rounds
    batch_num_per_aicore_process = rounds // used_aicore_num
    batch_tail = rounds % used_aicore_num

    with tik_instance.for_range(0, used_aicore_num, block_num=used_aicore_num) as i:
        with tik_instance.for_range(0, batch_num_per_aicore_process) as k:
            data_out, data_indices = sort_compute(tik_instance, dtype, total, i + k * used_aicore_num, descending,
                                                  num, distance, shape, big_distance, data_out, data_indices, input_gm,
                                                  L)
        with tik_instance.if_scope(i < batch_tail):
            data_out, data_indices = sort_compute(tik_instance, dtype, total,
                                                  batch_num_per_aicore_process * used_aicore_num + i, descending, num,
                                                  distance, shape, big_distance, data_out, data_indices, input_gm, L)

    float_ub = tik_instance.Tensor("float16", [total], name="float_ub", scope=tik.scope_ubuf)
    int_ub = tik_instance.Tensor("int32", [total], name="int_ub", scope=tik.scope_ubuf)

    with tik_instance.for_range(0, rounds) as i:
        tik_instance.data_move(float_ub[0], data_out[i * total], 0, 1, total // 16, 0, 0)
        tik_instance.data_move(data_out_[i * num], float_ub[0], 0, 1, total // 16, 0, 0)

        tik_instance.data_move(int_ub[0], data_indices[i * total], 0, 1, total // 8, 0, 0)
        tik_instance.data_move(data_indices_[i * num], int_ub[0], 0, 1, total // 8, 0, 0)

    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[input_gm], outputs=[data_out_, data_indices_])

    return tik_instance
