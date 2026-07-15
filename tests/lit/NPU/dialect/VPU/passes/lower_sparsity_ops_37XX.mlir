//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --lower-sparsity-ops="fake-sparsify=false" %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!defaultType = tensor<1x16x16x16xf16, {order = #NHWC}>
!smType = tensor<1x16x16x16xi1, {order = #NHWC}>
!sparseType = !VPU.SparseTensor<data=!defaultType, sparsity_map=!smType>

// CHECK-LABEL: @LowerSparsifyOpF16
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16, {order = #NHWC}>, [[ARG_1:%[^:]+]]: tensor<16x1x1x4xsi32>, [[ARG_2:%[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>)
func.func @LowerSparsifyOpF16(%arg0: !defaultType, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> !defaultType {
    %0 = VPU.Sparsify(%arg0) : !defaultType -> !sparseType
    %1 = VPU.NCE.Convolution(%0, %weights, %wt) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            
            strides = [1, 1]
        } : !sparseType, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> !defaultType

    return %1 : !defaultType

    // CHECK:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<16x1x1x128xi1>
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK:       [[SPARSE_WEIGHTS:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]])
    // CHECK-SAME:      {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<16xi64>, alignment = 16 : i64>}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<16x16x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x128xi1>, is_weights,
    // CHECK-SAME:                           #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<16xi64>, alignment = 16 : i64>>
    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>
    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[ARG_0]], [[SPARSE_WEIGHTS]], [[CST_WEIGHTS_TABLE]]) rawFilterShape [16, 16, 1, 1] {
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      }
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x16x16x16xi1, {order = #NHWC}>>

    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], [[ARG_2]], [[ARG_1]])
    // CHECK:       return [[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!defaultType = tensor<1x16x16x16x!quant.uniform<u8:f16, 2.0>, {order = #NHWC}>
!smType = tensor<1x16x16x16xi1, {order = #NHWC}>
!sparseType = !VPU.SparseTensor<data=!defaultType, sparsity_map=!smType>

// CHECK-LABEL: @LowerSparsifyOpUniformQuant
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x16x16x16x!qElemType, {order = #NHWC}>
// CHECK-SAME:      [[ARG_1:%[^:]+]]: tensor<16x1x1x4xsi32>
// CHECK-SAME:      [[ARG_2:%[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>
func.func @LowerSparsifyOpUniformQuant(%arg0: !defaultType, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> !defaultType {
    %0 = VPU.Sparsify(%arg0) : !defaultType -> !sparseType
    %1 = VPU.NCE.Convolution(%0, %weights, %wt) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            
            strides = [1, 1]
        } : !sparseType, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> !defaultType

    return %1 : !defaultType

    // CHECK:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1, {order = #NHWC}>
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<16x1x1x128xi1>
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK:       [[SPARSE_WEIGHTS:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]])
    // CHECK-SAME:      {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<16xi64>, alignment = 16 : i64>}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<16x16x1x1x!qElemType1, {order = #NHWC}>, sparsity_map=tensor<16x1x1x128xi1>, is_weights,
    // CHECK-SAME:                           #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<16xi64>, alignment = 16 : i64>>
    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME:    : tensor<16x1x1x4xsi32>
    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[ARG_0]], [[SPARSE_WEIGHTS]], [[CST_WEIGHTS_TABLE]]) rawFilterShape [16, 16, 1, 1] {
    // CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:        strides = [1, 1]
    // CHECK-SAME:    }
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x16x16x16x!qElemType, {order = #NHWC}>, sparsity_map=tensor<1x16x16x16xi1, {order = #NHWC}>>

    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], [[ARG_2]], [[ARG_1]])
    // CHECK:       return [[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16:1, {1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}>
!defaultType = tensor<1x16x16x16x!qElemType, {order = #NHWC}>
!smType = tensor<1x16x16x16xi1, {order = #NHWC}>
!sparseType = !VPU.SparseTensor<data=!defaultType, sparsity_map=!smType>

// CHECK:  !qElemType = !quant.uniform<u8:f16:1, {1.000000e+00,2.000000e+00,3.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00}>
// CHECK:  !qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @LowerSparsifyOpPerAxisQuant
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16x!qElemType, {order = #NHWC}>, [[ARG_1:%[^:]+]]: tensor<16x1x1x4xsi32>, [[ARG_2:%[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>)
func.func @LowerSparsifyOpPerAxisQuant(%arg0: !defaultType, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> !defaultType {
    %0 = VPU.Sparsify(%arg0) : !defaultType -> !sparseType
    %1 = VPU.NCE.Convolution(%0, %weights, %wt) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            
            strides = [1, 1]
        } : !sparseType, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> !defaultType

    return %1 : !defaultType

    // CHECK:       [[VAL0:%.+]] = VPU.QuantizeCast([[ARG_0]]) {dstElemType = !qElemType1} : tensor<1x16x16x16x!qElemType, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType1, {order = #NHWC}>
    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1, {order = #NHWC}>
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<16x1x1x128xi1>
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK:       [[SPARSE_WEIGHTS:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]])
    // CHECK-SAME:      {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<16xi64>, alignment = 16 : i64>}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<16x16x1x1x!qElemType1, {order = #NHWC}>, sparsity_map=tensor<16x1x1x128xi1>, is_weights,
    // CHECK-SAME:                           #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<16xi64>, alignment = 16 : i64>>
    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME:    : tensor<16x1x1x4xsi32>
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], [[SPARSE_WEIGHTS]], [[CST_WEIGHTS_TABLE]]) rawFilterShape [16, 16, 1, 1] {
    // CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:        strides = [1, 1]
    // CHECK-SAME:    }
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x16x16x16x!qElemType1, {order = #NHWC}>, sparsity_map=tensor<1x16x16x16xi1, {order = #NHWC}>>
    // CHECK:       [[VAL2:%.+]] = VPU.QuantizeCast([[VAL1]]) {dstElemType = !qElemType}
    // CHECK-SAME:     : !VPU.SparseTensor<data=tensor<1x16x16x16x!qElemType1, {order = #NHWC}>, sparsity_map=tensor<1x16x16x16xi1, {order = #NHWC}>>
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x16x16x16x!qElemType, {order = #NHWC}>, sparsity_map=tensor<1x16x16x16xi1, {order = #NHWC}>>

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL2]], [[ARG_2]], [[ARG_1]])
    // CHECK:       return [[VAL3]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!defaultType = tensor<1x16x16x16xf16, {order = #NHWC}>
!smType = tensor<1x16x16x16xi1, {order = #NHWC}>
!sparseType = !VPU.SparseTensor<data=!defaultType, sparsity_map=!smType>

// CHECK-LABEL: @LowerDesparsifyOpF16
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16, {order = #NHWC}>, [[ARG_1:%[^:]+]]: tensor<16x1x1x4xsi32>, [[ARG_2:%[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>)
func.func @LowerDesparsifyOpF16(%arg0: !defaultType, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> !defaultType {
    %0 = VPU.NCE.Convolution(%arg0, %weights, %wt) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            
            strides = [1, 1]
        } : !defaultType, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> !sparseType
    %1 = VPU.Desparsify(%0) : !sparseType -> !defaultType

    return %1 : !defaultType

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[ARG_0]], [[ARG_2]], [[ARG_1]])
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Eltwise([[VAL0]], [[VAL0]])
    // CHECK-SAME:                   op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:                   clamp_low = -2147483648 : i64
    // CHECK-SAME:                   clamp_high = 2147483647 : i64
    // CHECK-NOT:                    !VPU.SparseTensor

    // CHECK:       return [[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!defaultType = tensor<1x16x16x16x!quant.uniform<u8:f16, 2.0>, {order = #NHWC}>
!smType = tensor<1x16x16x16xi1, {order = #NHWC}>
!sparseType = !VPU.SparseTensor<data=!defaultType, sparsity_map=!smType>

// CHECK-LABEL: @LowerDesparsifyOpQuantUniform
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16x!qElemType, {order = #NHWC}>, [[ARG_1:%[^:]+]]: tensor<16x1x1x4xsi32>, [[ARG_2:%[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>)
func.func @LowerDesparsifyOpQuantUniform(%arg0: !defaultType, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> !defaultType {
    %0 = VPU.NCE.Convolution(%arg0, %weights, %wt) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            
            strides = [1, 1]
        } : !defaultType, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> !sparseType
    %1 = VPU.Desparsify(%0) : !sparseType -> !defaultType

    return %1 : !defaultType

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[ARG_0]], [[ARG_2]], [[ARG_1]])
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Eltwise([[VAL0]], [[VAL0]])
    // CHECK-SAME:                   op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:                   clamp_low = 0
    // CHECK-SAME:                   clamp_high = 255
    // CHECK-NOT:                    !VPU.SparseTensor

    // CHECK:       return [[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16:1, {1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}>
!defaultType = tensor<1x16x16x16x!qElemType, {order = #NHWC}>
!smType = tensor<1x16x16x16xi1, {order = #NHWC}>
!sparseType = !VPU.SparseTensor<data=!defaultType, sparsity_map=!smType>

// CHECK:  !qElemType = !quant.uniform<u8:f16:1, {1.000000e+00,2.000000e+00,3.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00}>
// CHECK:  !qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @LowerDesparsifyOpPerAxisQuant
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16x!qElemType, {order = #NHWC}>, [[ARG_1:%[^:]+]]: tensor<16x1x1x4xsi32>, [[ARG_2:%[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>)
func.func @LowerDesparsifyOpPerAxisQuant(%arg0: !defaultType, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> !defaultType {
    %0 = VPU.NCE.Convolution(%arg0, %weights, %wt) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            
            strides = [1, 1]
        } : !defaultType, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> !sparseType
    %1 = VPU.Desparsify(%0) : !sparseType -> !defaultType

    return %1 : !defaultType

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[ARG_0]], [[ARG_2]], [[ARG_1]])
    // CHECK:       [[VAL1:%.+]] = VPU.QuantizeCast([[VAL0]]) {dstElemType = !qElemType1}
    // CHECK-SAME:       : !VPU.SparseTensor<data=tensor<1x16x16x16x!qElemType, {order = #NHWC}>, sparsity_map=tensor<1x16x16x16xi1, {order = #NHWC}>>
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x16x16x16x!qElemType1, {order = #NHWC}>, sparsity_map=tensor<1x16x16x16xi1, {order = #NHWC}>>
    // CHECK:       [[VAL2:%.+]] = VPU.NCE.Eltwise([[VAL1]], [[VAL1]])
    // CHECK-SAME:                   op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:                   clamp_low = 0
    // CHECK-SAME:                   clamp_high = 255
    // CHECK-NOT:                    !VPU.SparseTensor
    // CHECK:       [[VAL3:%.+]] = VPU.QuantizeCast([[VAL2]]) {dstElemType = !qElemType} : tensor<1x16x16x16x!qElemType1, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType, {order = #NHWC}>

    // CHECK:       return [[VAL3]]
}
