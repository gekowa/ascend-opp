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
 * @sigmoid_cross_entropy_with_logits_grad_v2.h
 *
 * @brief
 *
 * @version 1.0
 *
 */

#ifndef GE_OP_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_H
#define GE_OP_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_H
#include "graph/operator_reg.h"
namespace ge {
/**
* @brief Computes gradients of sigmoid_cross_entropy_with_logits_v2.

* @par Inputs:
* @predict: An ND tensor of type float16, float32.
* @target: An ND tensor of type float16, float32.
* @dout: An ND tensor of type float16, float32.
* @weight: An optional ND tensor of type float16, float32.
* @pos_weight: An optional ND tensor of type float16, float32.
*
* @par Attributes:
* @li reduction: An optional string.Defaults to "mean".

* @par Outputs:
* @gradient: An ND tensor tensor with the same shape and type as "predict".
*/
REG_OP(SigmoidCrossEntropyWithLogitsGradV2)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dout, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(pos_weight, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(gradient, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SigmoidCrossEntropyWithLogitsGradV2)
}
#endif // GE_OP_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_H
