/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: Mse_loss
 * Create: 2020-07-22
 * Notes: This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
**/

/**
 * Copyright (C)  2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file mse_loss.h
 *
 * @brief
 *
 * @version 1.0
 *
 **/
/**
*
* @par Inputs:
* predict:A Tensor of type float16 or float32
* lable:A Tensor of type float16 or float32

* @par Attributes:
* @li operation:An optional str from sum, none, mean
* specifying the reduction algorithm. Defaults to "mean".

* @par Outputs:
* y: when reduction=none A Tensor. Has the same type as "predict".
     when reduction=sum/mean A Scalar
**/
#ifndef GE_OP_MSE_LOSS_H
#define GE_OP_MSE_LOSS_H
#include "graph/operator_reg.h"
namespace ge {
REG_OP(MseLoss)
.INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
.INPUT(label, TensorType({DT_FLOAT16, DT_FLOAT}))
.OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
.ATTR(reduction, String, "mean")
.OP_END_FACTORY_REG(MseLoss)
}
#endif // GE_OP_MSE_LOSS_H
