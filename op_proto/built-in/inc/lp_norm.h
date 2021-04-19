/** Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: GE_OP_LP_NORM_H
 * Create: 2020-09-7
 * Notes: This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef GE_OP_LP_NORM_H
#define GE_OP_LP_NORM_H
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Computes Lp norm.

* @par Inputs:
* @x: An ND tensor of type float16, float32.
*
* @par Attributes:
* @p: Int, "inf" or "-inf", default value is 2.
* @axes: ListInt, {} means all axes will be computed.
* @keepdim: Bool, default is false.
* @epsilon: Float, default is 1e-12.

* @par Outputs:
* @y: An ND tensor of type float16, float32. 
*  y shape is depending on axes and keepdim.
*/
REG_OP(LpNorm)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(p, Int, 2)
    .ATTR(axes, ListInt, {})
    .ATTR(keepdim, Bool, false)
    .ATTR(epsilon, Float, 1e-12)
    .OP_END_FACTORY_REG(LpNorm)
}
#endif  // GE_OP_LP_NORM_H
