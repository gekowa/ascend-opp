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
 */

#ifndef GE_OP_MSE_LOSS_GRAD_H
#define GE_OP_MSE_LOSS_GRAD_H

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Computes gradients of mse loss.

* @par Inputs:
* @predict: An ND tensor of type float16, float32.
* @label: An ND tensor of type float16, float32.
* @dout: An ND tensor of type float16, float32.
*
* @par Attributes:
* @li reduction: An optional string.Defaults to "mean".

* @par Outputs:
* @y: An ND tensor tensor with the same shape and type as "predict".
*/
REG_OP(MseLossGrad)
    .INPUT(predict, TensorType({DT_FLOAT32, DT_FLOAT16}))
    .INPUT(label, TensorType({DT_FLOAT32, DT_FLOAT16}))
    .INPUT(dout, TensorType({DT_FLOAT32, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT32, DT_FLOAT16}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(MseLossGrad)
}

#endif
