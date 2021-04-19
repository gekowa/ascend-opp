/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: op proto for StridedSliceV2
 * Create: 2020-08-10
 * Notes: This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */


#ifndef GE_OP_STRIDED_SLICE_V2_H
#define GE_OP_STRIDED_SLICE_V2_H

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Extracts a strided slice of a tensor. Roughly speaking, this op \n
*   extracts a slice of size (end-begin)/stride from the given input tensor. \n
*   Starting at the location specified by begin the slice continues by \n
*   adding stride to the index until all dimensions are not less than end. \n
*
* @par Inputs:
* Four inputs, including:
* @li x: A Tensor. Must be one of the following types: float32, float64, int32, uint8, int16, int8, \n
*     complex64, int64, qint8, quint8, qint32, qint16, quint16, uint16, \n
*     complex128, float16, uint32, uint64, complex64, complex128. \n
* @li begin: A Tensor of type int32 or int64, for the index of the first value to select.
*
* @li end: A Tensor of type int32 or int64, for the index of the last value to select.
*
* @li axes: A Tensor of type int32 or int64, indicate axis to be select.
*
* @li strides: A Tensor of type int32 or int64, for the increment.
*
* @par Attributes:
* @li begin_mask: A Tensor of type int32. \n
*     A bitmask where a bit "i" being "1" means to ignore the begin \n
*     value and instead use the largest interval possible.
* @li end_mask: A Tensor of type int32. \n
*     Analogous to "begin_mask".
* @li ellipsis_mask: A Tensor of type int32. \n
*     A bitmask where bit "i" being "1" means the "i"th position \n
*     is actually an ellipsis.
* @li new_axis_mask: A Tensor of type int32. \n
*     A bitmask where bit "i" being "1" means the "i"th \n
*     specification creates a new shape 1 dimension.
* @li shrink_axis_mask: A Tensor of type int32. \n
*     A bitmask where bit "i" implies that the "i"th \n
*     specification should shrink the dimensionality.
*
* @par Outputs:
* y: A Tensor. Has the same type as "x".
*
* @attention Constraints:
*
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator StridedSliceV2.
*/
REG_OP(StridedSliceV2)
    .INPUT(x, TensorType::BasicType())
    .INPUT(begin, TensorType::IndexNumberType())
    .INPUT(end, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(axes, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(strides, TensorType::IndexNumberType())
    .ATTR(begin_mask, Int, 0)
    .ATTR(end_mask, Int, 0)
    .ATTR(ellipsis_mask, Int, 0)
    .ATTR(new_axis_mask, Int, 0)
    .ATTR(shrink_axis_mask, Int, 0)
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(StridedSliceV2)
}
#endif // GE_OP_STRIDED_SLICE_V2_H
