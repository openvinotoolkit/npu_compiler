//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --lower-sparsity-ops="fake-sparsify=false" %s | FileCheck %s
// REQUIRES: platform-NPU3720
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!inputType = tensor<1x4x120x110x!quant.uniform<u8:f16, 2.0>, {order = #NHWC}>
!smType = tensor<1x4x120x110xi1, {order = #NHWC}>
!sparseType = !VPU.SparseTensor<data=!inputType, sparsity_map=!smType>
!weightsType = tensor<64x1x1x16xf16, {order = #NHWC}>
!outputType = tensor<1x64x120x110x!quant.uniform<u8:f16, 2.0>, {order = #NHWC}>

// CHECK-LABEL: @LowerSparsifyOpUniformQuantUnalignedShape
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x4x120x110x!qElemType, {order = #NHWC}>, [[ARG_1:%[^:]+]]: tensor<64x1x1x4xsi32>, [[ARG_2:%[^:]+]]: tensor<64x1x1x16xf16, {order = #NHWC}>
func.func @LowerSparsifyOpUniformQuantUnalignedShape(%arg0: !inputType, %wt: tensor<64x1x1x4xsi32>, %weights: !weightsType) -> !outputType {
    %0 = VPU.Sparsify(%arg0) : !inputType -> !sparseType
    %1 = VPU.NCE.CompressConvolution(%0, %weights, %wt) rawFilterShape [64, 4, 1, 1] {
            cm_sp_pattern = 15 : i64,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            
            strides = [1, 1]
        } -> !outputType

    return %1 : !outputType

    // CHECK:       [[VAL0:%.+]] = VPU.Expand([[ARG_0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]}
    // CHECK-SAME:    : tensor<1x4x120x110x!qElemType, {order = #NHWC}> -> tensor<1x16x120x110x!qElemType, {order = #NHWC}>
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
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x16x120x110x!qElemType, {order = #NHWC}>, sparsity_map=tensor<1x16x120x110xi1, {order = #NHWC}>>
    // CHECK:       [[VAL2:%.+]] = VPU.Slice [[VAL1]] [0, 0, 0, 0] [1, 4, 120, 110]
    // CHECK-SAME:    : !VPU.SparseTensor<data=tensor<1x16x120x110x!qElemType, {order = #NHWC}>, sparsity_map=tensor<1x16x120x110xi1, {order = #NHWC}>> to
    // CHECK-SAME:      !VPU.SparseTensor<data=tensor<1x4x120x110x!qElemType, {order = #NHWC}>, sparsity_map=tensor<1x4x120x110xi1, {order = #NHWC}>>

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.CompressConvolution([[VAL2]], [[ARG_2]], [[ARG_1]])
    // CHECK:       return [[VAL3]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!inputType = tensor<1x4x31x31xf16, {order = #NHWC}>
!smType = tensor<1x4x31x31xi1, {order = #NHWC}>
!sparseType = !VPU.SparseTensor<data=!inputType, sparsity_map=!smType>
!weightsType = tensor<64x1x1x16xf16, {order = #NHWC}>
!outputType = tensor<1x64x31x31xf16, {order = #NHWC}>

// CHECK-LABEL: @LowerSparsifyOpFloatUnalignedShape
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x4x31x31xf16, {order = #NHWC}>, [[ARG_1:%[^:]+]]: tensor<64x1x1x4xsi32>, [[ARG_2:%[^:]+]]: tensor<64x1x1x16xf16, {order = #NHWC}>)
func.func @LowerSparsifyOpFloatUnalignedShape(%arg0: !inputType, %wt: tensor<64x1x1x4xsi32>, %weights: !weightsType) -> !outputType {
    %0 = VPU.Sparsify(%arg0) : !inputType -> !sparseType
    %1 = VPU.NCE.CompressConvolution(%0, %weights, %wt) rawFilterShape [64, 4, 1, 1] {
            cm_sp_pattern = 15 : i64,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            
            strides = [1, 1]
        } -> !outputType

    return %1 : !outputType

    // CHECK:       [[VAL0:%.+]] = VPU.Expand([[ARG_0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]}
    // CHECK-SAME:    : tensor<1x4x31x31xf16, {order = #NHWC}> -> tensor<1x16x31x31xf16, {order = #NHWC}>
    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<16x1x1x128xi1>
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK:       [[SPARSE_WEIGHTS:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]])
    // CHECK-SAME:      {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<16xi64>, alignment = 16 : i64>}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<16x16x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x128xi1>, is_weights,
    // CHECK-SAME:                           #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<16xi64>, alignment = 16 : i64>>
    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME:    : tensor<16x1x1x4xsi32>
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], [[SPARSE_WEIGHTS]], [[CST_WEIGHTS_TABLE]]) rawFilterShape [16, 16, 1, 1] {
    // CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:        strides = [1, 1]
    // CHECK-SAME:    }
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x16x31x31xf16, {order = #NHWC}>, sparsity_map=tensor<1x16x31x31xi1, {order = #NHWC}>>
    // CHECK:       [[VAL2:%.+]] = VPU.Slice [[VAL1]] [0, 0, 0, 0] [1, 4, 31, 31]
    // CHECK-SAME:    : !VPU.SparseTensor<data=tensor<1x16x31x31xf16, {order = #NHWC}>, sparsity_map=tensor<1x16x31x31xi1, {order = #NHWC}>> to
    // CHECK-SAME:      !VPU.SparseTensor<data=tensor<1x4x31x31xf16, {order = #NHWC}>, sparsity_map=tensor<1x4x31x31xi1, {order = #NHWC}>>

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.CompressConvolution([[VAL2]], [[ARG_2]], [[ARG_1]])
    // CHECK:       return [[VAL3]]
}
