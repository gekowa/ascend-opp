/** Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use
 * this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file lp_loss.h
 *
 * @brief
 *
 * @version 1.0
 *
 */

#ifndef GE_OP_LP_LOSS_H
#define GE_OP_LP_LOSS_H
#include "graph/operator_reg.h"
namespace ge {
/**
* @brief Computes lp_loss.

* @par Inputs:
* @predict: An ND tensor of type float16, float32.
* @label: An ND tensor of type float16, float32.
*
* @par Attributes:
* @li p: A required int attribute that decides which loss to compute, now the p only can be 1 to compute l1_loss.
* @li reduction: An optional string.Defaults to "mean".

* @par Outputs:
* @y: An ND tensor tensor with the same shape and type as "predict".
*/
REG_OP(LpLoss)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(label, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(p, Int)
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(LpLoss)
}
#endif // GE_OP_LP_LOSS_H
