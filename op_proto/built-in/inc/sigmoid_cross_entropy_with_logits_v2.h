/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
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
 * @file sigmoid_cross_entropy_with_logitsv2.harderr
 *
 * @version 1.0
 */

#ifndef GE_OP_SIGMOID_CROSS_ENTROPY_WITH_LOGITSV2_H
#define GE_OP_SIGMOID_CROSS_ENTROPY_WITH_LOGITSV2_H

#include "graph/operator_reg.h"

/**
* @brief it measures Binary Cross Entropy between target and output logits. \n
*        this loss combines a Sigmoid layer and the BCELoss in one single class.
* @par Inputs:
*   4 inputs, including:
* @li predict: Dtype support float16 and float, Format support ND and NC1HWC0, required.
* @li target: Dtype support float16 and float, Format support ND and NC1HWC0, required.
* @li weight: Dtype support float16 and float, Format support ND and NC1HWC0, optional.
*             It supports to be broadcast into the same shape as predict.
* @li pow_weight: Dtype support float16 and float, Format support ND and NC1HWC0, optional.
*                 It supports to be broadcast into the same shape as predict.
*
* @par Outputs:
*   1 output, including:
* @li loss: Dtype support float, Format support ND and NC1HWC0, required.
*           if reduction == 'none', shape of it is same as predict
*           if reduction == 'mean' or 'sum', shape of it is 1
*
* @par Attributes:
* @li reduction: string. Specifies the reduction to apply to the output: 'none'、'mean'、'sum'.
*                optional, default is 'mean'.
*
* @attention Constraints:
* @li shape of target should be same as predict.
* @li Dtype and format of input shoule be same.
* @li Shape of weight or pos_weight should be same as predict's shape, \
*     or should meet the conditions for broadcasting to predict's shape
* @li When format is NC1HWC0, the len of shape of weight or pos_weight \
*     should be same as predict (if there is input weight or pos_weight), \
*     and ori shape of all inputs should be no bigger than 4. \
*     Not support broadcast in axis of C, when format is NC1HWC0.
*
*/
namespace ge {
REG_OP(SigmoidCrossEntropyWithLogitsV2)
    .INPUT(predict, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(target, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OPTIONAL_INPUT(pos_weight, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(loss, TensorType({DT_FLOAT}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SigmoidCrossEntropyWithLogitsV2)
}

#endif // GE_OP_SIGMOID_CROSS_ENTROPY_WITH_LOGITSV2_H
