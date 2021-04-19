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

map_index
"""

from te import tik
from te import platform as tbe_platform
from te.utils.op_utils import *


def map_index_vec_dup(tik_instance, mask, dst, const, length):
    """
    :param tik_instance:
    :param mask:
    :param dst:
    :param const:
    :param length:
    :return:
    """
    if length <= mask:
        tik_instance.vec_dup(length, dst[0], const, 1, 8)
    else:
        number = length // mask
        tail = length % mask
        tik_instance.vec_dup(mask, dst[0], const, number, 8)
        if tail > 0:
            tik_instance.vec_dup(tail, dst[number*mask], const, 1, 8)


def map_index_vec_sub(tik_instance, mask, dst, src0, src1, length):
    """
    :param tik_instance:
    :param mask:
    :param dst:
    :param src0:
    :param src1:
    :param length:
    :return:
    """
    if length <= mask:
        tik_instance.vec_sub(length, dst, src0, src1, 1, 8, 8, 8)
    else:
        number = length // mask
        tail = length % mask
        tik_instance.vec_sub(mask, dst, src0, src1, number, 8, 8, 8)
        if tail > 0:
            tik_instance.vec_sub(tail, dst[number*mask], src0[number*mask],
                                 src1, 1, 8, 8, 8)


class MapIndexProcess:
    """
    MapIndexProcess
    """
    def __init__(self, input_data):
        """
        :param input_data:
        """
        self.tik_instance = input_data[0]

        x_dic = input_data[1]
        x_shape = x_dic.get('shape')
        self.x_length = x_shape[0]

        data_seq_dic = input_data[2]
        data_seq_shape = data_seq_dic.get('shape')
        self.data_seq_length = data_seq_shape[0]

        y_dic = input_data[3]
        y_shape = y_dic.get('shape')

        if len(input_data) == 5:
            self.have_level_index = True
            level_index_dic = input_data[4]
            level_index_shape = level_index_dic.get('shape')
        else:
            self.have_level_index = False

        self.x = self.tik_instance.Tensor("int32", x_shape, name="x",
                                          scope=tik.scope_gm)
        self.data_seq = self.tik_instance.Tensor("int32", data_seq_shape,
                                                   name="data_seq",
                                                   scope=tik.scope_gm)
        if self.have_level_index == True:
            self.level_index = self.tik_instance.Tensor("int32",
                                                        level_index_shape,
                                                        name="level_index",
                                                        scope=tik.scope_gm)
            self.level_index_ub = self.tik_instance.Tensor(
                "int32", [((level_index_shape[0] + 7) // 8)*8],
                name="level_index",
                scope=tik.scope_ubuf)
            self.tik_instance.data_move(self.level_index_ub,
                                        self.level_index, 0, 1,
                                        (level_index_shape[0] + 7) // 8,
                                        0, 0, 0)

        self.y = self.tik_instance.Tensor("int32", y_shape, name="y",
                                          scope=tik.scope_gm)

        self.x_ub = self.tik_instance.Tensor("int32", [8], name="x_ub",
                                             scope=tik.scope_ubuf)
        self.data_seq_ub = self.tik_instance.Tensor("int32", data_seq_shape,
                                                    name="data_seq_ub",
                                                    scope=tik.scope_ubuf)
        self.y_ub = self.tik_instance.Tensor("int32", [8], name="y_ub",
                                          scope=tik.scope_ubuf)
        self.tik_instance.vec_dup(8, self.y_ub, -1, 1, 8)

        self.tik_instance.data_move(self.x_ub, self.x, 0, 1, 1, 0, 0, 0)
        self.tik_instance.data_move(self.data_seq_ub, self.data_seq, 0, 1,
                                    self.data_seq_length // 8, 0, 0, 0)


    def cce_map_index(self, kernel_name="map_index"):
        """
        :param kernel_name:
        :return:
        """
        tik_instance = self.tik_instance
        tik_name = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
        print("tik_name:", tik_name)
        length = self.data_seq_length // self.x_length
        max = ((length + 127) // 128)*128
        count = ((max // 16 + 15) // 16)*16
        res = tik_instance.Tensor("uint16", [count], name="res",
                                      scope=tik.scope_ubuf)
        map_index_vec_dup(tik_instance, 128, res, 0, count)

        with tik_instance.for_range(0, self.x_length) as i:
            # compute ===> tmp : vec_dup(x[i])
            tmp = tik_instance.Tensor("int32", [length], name="tmp",
                                                scope=tik.scope_ubuf)
            x_scalar = tik_instance.Scalar(dtype="int32")
            x_scalar.set_as(self.x_ub[i])

            # compute ===> data_seq[i*length]-x[i]
            map_index_vec_dup(tik_instance, 64, tmp, x_scalar, length)
            sub_value = tik_instance.Tensor("int32", [max], name="sub_value",
                                            scope=tik.scope_ubuf)
            map_index_vec_dup(tik_instance, 64, sub_value, 1, max)
            map_index_vec_sub(tik_instance, 64, sub_value,
                              self.data_seq_ub[i*length], tmp, length)

            # sub_value trans to FP16
            sub_fp16 = tik_instance.Tensor("float16", [max], name="sub_fp16",
                                           scope=tik.scope_ubuf)
            map_index_vec_dup(tik_instance, 128, sub_fp16, 1, max)
            if tik_name in ["Hi3796CV300CS"]:
                sub_int16 = tik_instance.Tensor("int16", [max],
                                                name="sub_int16",
                                                scope=tik.scope_ubuf)
                tik_instance.vcbd(64, sub_int16, sub_value, max // 64,
                                  1, 1, 4, 8)
                tik_instance.vconv(128, "", sub_fp16, sub_int16, max // 128,
                                   1, 1, 8, 8)
            else:
                tik_instance.vconv(64, "", sub_fp16, sub_value, max // 64,
                                   1, 1, 4, 8, 1.0)

            # sub_FP16 compv 0
            zero_ub = tik_instance.Tensor("float16", [max], name="tmp_fp16",
                                          scope=tik.scope_ubuf)
            map_index_vec_dup(tik_instance, 128, zero_ub, 0, max)

            cmp_res = tik_instance.Tensor("uint16", [count], name="cmp_res",
                                          scope=tik.scope_ubuf)
            map_index_vec_dup(tik_instance, 128, cmp_res, 0, count)
            with tik_instance.if_scope(i == 0):
                tik_instance.vec_cmpv_eq(res, sub_fp16, zero_ub,
                                         max // 128, 8, 8)
            with tik_instance.else_scope():
                tik_instance.vec_cmpv_eq(cmp_res, sub_fp16, zero_ub,
                                         max // 128, 8, 8)
                tik_instance.vec_and(16, res, cmp_res, res, count // 16,
                                     1, 1, 1)

        flag = tik_instance.Scalar(dtype="uint8")
        flag.set_as(0)
        with tik_instance.for_range(0, count) as i:
            with tik_instance.if_scope(res[i] > 0):
                with tik_instance.if_scope(flag == 0):
                    uint16_scalar = tik_instance.Scalar(dtype="uint16")
                    uint16_scalar.set_as(res[i])
                    src_scalar = tik_instance.Scalar(dtype="uint64")
                    src_scalar.set_as(uint16_scalar)

                    countbit1 = tik_instance.Scalar(dtype="uint64")
                    countbit1.set_as(0)
                    tik_instance.scalar_countleading0(countbit1, src_scalar)
                    with tik_instance.if_scope(countbit1 == 1):
                        flag.set_as(1)

                        dst_scalar = tik_instance.Scalar(dtype="uint64")
                        tik_instance.scalar_countleading0(dst_scalar,
                                                          src_scalar)
                        if self.have_level_index == True:
                            self.y_ub[0].set_as(
                                self.level_index_ub[i*16 + (63 - dst_scalar)])
                        else:
                            self.y_ub[0].set_as(i*16 + (63 - dst_scalar))
                    with tik_instance.else_scope():
                        with tik_instance.for_range(0, 16) as j:
                            with tik_instance.if_scope(flag == 0):
                                with tik_instance.if_scope(src_scalar % 2 != 0):
                                        flag.set_as(1)
                                        if self.have_level_index == True:
                                            self.y_ub[0].set_as(
                                                self.level_index_ub[i*16 + j])
                                        else:
                                            self.y_ub[0].set_as(i*16 + j)
                                with tik_instance.else_scope():
                                    src_scalar.set_as(src_scalar / 2)

        tik_instance.data_move(self.y, self.y_ub, 0, 1, 1, 0, 0, 0)

        if self.have_level_index == True:
            tik_instance.BuildCCE(
                kernel_name,
                inputs=[self.x, self.data_seq, self.level_index],
                outputs=[self.y])
        else:
            tik_instance.BuildCCE(
                kernel_name,
                inputs=[self.x, self.data_seq],
                outputs=[self.y])

        return self.tik_instance


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, OPTION_INPUT,
                 REQUIRED_OUTPUT, KERNEL_NAME)
def map_index(x_dic, data_seq_dic, level_index_dic, y_dic,
              kernel_name="map_index"):
    """
    :param x_dic:
    :param data_seq_dic:
    :param level_index_dic:
    :param y_dic:
    :param kernel_name:
    :return:
    """

    check_list = ["int32"]
    x_shape = x_dic.get("shape")
    x_dtype = x_dic.get("dtype")
    check_dtype(x_dtype.lower(), check_list, param_name="x")

    data_seq_shape = data_seq_dic.get("shape")
    data_seq_dtype = data_seq_dic.get("dtype")
    check_dtype(data_seq_dtype.lower(), check_list,
                param_name="data_seq")

    y_dtype = y_dic.get("dtype")
    check_dtype(y_dtype.lower(), check_list, param_name="y")

    if x_shape[0] > 8:
        raise RuntimeError("the length of x should "
                           "be less than or equal to 8")

    if data_seq_shape[0] % x_shape[0] != 0:
        raise RuntimeError("the length of data_seq must "
                           "be multiple of the length of x")

    tik_instance = tik.Tik(tik.Dprofile())

    if level_index_dic:
        level_index_dtype = level_index_dic.get("dtype")
        check_dtype(level_index_dtype.lower(), check_list,
                    param_name="level_index")

        map_index_result = MapIndexProcess((tik_instance, x_dic, data_seq_dic,
                                            y_dic, level_index_dic))
    else:
        map_index_result = MapIndexProcess((tik_instance, x_dic, data_seq_dic,
                                            y_dic))

    return map_index_result.cce_map_index(kernel_name)
