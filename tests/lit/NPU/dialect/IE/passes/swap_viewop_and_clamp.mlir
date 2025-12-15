//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --swap-viewop-and-clamp %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

!qElemType = !quant.uniform<u8:f16, 0.0022939644607843138>
!qElemType1 = !quant.uniform<u8:f16, 0.0011469822303921569>

// CHECK: !qElemType = !quant.uniform<u8:f16, 0.0011469822303921569>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 0.0022939644607843138>

// CHECK-LABEL: @SwapQuantizeCast
// CHECK-SAME:    [[INPUT:%.*]]: tensor<1x16x8x8xf16>
func.func @SwapQuantizeCast(%arg0: tensor<1x16x8x8xf16>) -> tensor<1x16x8x8x!qElemType1> {
      %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x8x8xf16>, tensor<1x16x8x8xf16> -> tensor<1x16x8x8x!qElemType>
      %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType1} : tensor<1x16x8x8x!qElemType> -> tensor<1x16x8x8x!qElemType1>
      %2 = IE.Clamp(%1) {max = 5.000000e+00, min = 1.000000e+00} : tensor<1x16x8x8x!qElemType1> -> tensor<1x16x8x8x!qElemType1>

      return %2 : tensor<1x16x8x8x!qElemType1>

      // CHECK: [[VAR0:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
      // CHECK: [[VAR1:%.+]] = IE.Clamp([[VAR0]]) {max = 1.000000e+01 : f64, min = 2.000000e+00 : f64}
      // CHECK: [[VAR2:%.+]] = IE.QuantizeCast([[VAR1]]) {dstElemType = !qElemType}
      // CHECK: return [[VAR2]] : tensor<1x16x8x8x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0022939644607843138>
!qElemType1 = !quant.uniform<u8:f16, 0.0011469822303921569>

// CHECK: !qElemType = !quant.uniform<u8:f16, 0.0011469822303921569>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 0.0022939644607843138>

// CHECK-LABEL: @NoSwapTwoConsumers
// CHECK-SAME:    [[INPUT:%.*]]: tensor<1x16x8x8xf16>
func.func @NoSwapTwoConsumers(%arg0: tensor<1x16x8x8xf16>) -> (tensor<1x16x8x8x!qElemType1>, tensor<1x16x8x8x!qElemType1>) {
      %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x8x8xf16>, tensor<1x16x8x8xf16> -> tensor<1x16x8x8x!qElemType>
      %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType1} : tensor<1x16x8x8x!qElemType> -> tensor<1x16x8x8x!qElemType1>
      %2 = IE.Clamp(%1) {max = 5.000000e+00, min = 1.000000e+00} : tensor<1x16x8x8x!qElemType1> -> tensor<1x16x8x8x!qElemType1>

      return %1, %2 : tensor<1x16x8x8x!qElemType1>, tensor<1x16x8x8x!qElemType1>

      // CHECK: [[VAR0:%.+]] = IE.Add([[INPUT]], [[INPUT]])
      // CHECK: [[VAR1:%.+]] = IE.QuantizeCast([[VAR0]])
      // CHECK: [[VAR2:%.+]] = IE.Clamp([[VAR1]])

      // CHECK: return [[VAR1]], [[VAR2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0022939644607843138>

// CHECK-LABEL: @NoSwapWithOutNCEParent
// CHECK-SAME:    [[INPUT:%.*]]: tensor<1x16x8x8xui8>
func.func @NoSwapWithOutNCEParent(%arg0: tensor<1x16x8x8xui8>) -> tensor<1x16x8x8x!qElemType> {
      %0 = IE.QuantizeCast(%arg0) {dstElemType = !qElemType} : tensor<1x16x8x8xui8> -> tensor<1x16x8x8x!qElemType>
      %1 = IE.Clamp(%0) {max = 5.000000e+00, min = 1.000000e+00} : tensor<1x16x8x8x!qElemType> -> tensor<1x16x8x8x!qElemType>

      return %1 : tensor<1x16x8x8x!qElemType>

      // CHECK: [[VAR0:%.+]] = IE.QuantizeCast([[INPUT]])
      // CHECK: [[VAR1:%.+]] = IE.Clamp([[VAR0]])
      // CHECK: return [[VAR1]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0:123>
!qElemType1 = !quant.uniform<u8:f16, 1.0:120>
!qElemType2 = !quant.uniform<u8:f16, 2.0:121>
// CHECK-LABEL: @NotSwapDifferZeroPointQuantizeCastWithClamp
// CHECK-SAME:    [[INPUT:%.*]]: tensor<1x8x80x80x!qElemType>
func.func @NotSwapDifferZeroPointQuantizeCastWithClamp(%arg0: tensor<1x8x80x80x!qElemType>) -> tensor<1x8x80x80x!qElemType2> {
    %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x80x80x!qElemType>, tensor<1x8x80x80x!qElemType> -> tensor<1x8x80x80x!qElemType1>
    %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType2} : tensor<1x8x80x80x!qElemType1> -> tensor<1x8x80x80x!qElemType2>
    %2 = IE.Clamp(%1) {max = 4.000000e+00 : f64, min = -1.500000e+00 : f64} : tensor<1x8x80x80x!qElemType2> -> tensor<1x8x80x80x!qElemType2>
    return %2 : tensor<1x8x80x80x!qElemType2>

    // CHECK:   [[ADD:%.*]] = IE.Add([[INPUT]], [[INPUT]])
    // CHECK:   [[QUANTIZECAST:%.*]] = IE.QuantizeCast([[ADD]])
    // CHECK:   [[CLAMP:%.*]] = IE.Clamp([[QUANTIZECAST]])
    // CHECK:   return  [[CLAMP]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0:123>
!qElemType1 = !quant.uniform<u8:f16, 1.0:120>
// CHECK-LABEL: @swapShapeCastWithClamp
// CHECK-SAME:    [[INPUT:%.*]]: tensor<1x8x80x80x!qElemType>
func.func @swapShapeCastWithClamp(%arg0: tensor<1x8x80x80x!qElemType>) -> tensor<1x16x40x80x!qElemType1> {
    %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x80x80x!qElemType>, tensor<1x8x80x80x!qElemType> -> tensor<1x8x80x80x!qElemType1>
    %1 = IE.ShapeCast {shape = [1, 16, 40, 80]} inputs(%0 : tensor<1x8x80x80x!qElemType1>) -> tensor<1x16x40x80x!qElemType1>
    %2 = IE.Clamp(%1) {max = 4.000000e+00 : f64, min = -1.500000e+00 : f64} : tensor<1x16x40x80x!qElemType1> -> tensor<1x16x40x80x!qElemType1>
    return %2 : tensor<1x16x40x80x!qElemType1>

    // CHECK:   [[ADD:%.*]] = IE.Add([[INPUT]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x80x80x!qElemType>, tensor<1x8x80x80x!qElemType> -> tensor<1x8x80x80x!qElemType1>
    // CHECK:   [[CLAMP:%.*]] = IE.Clamp([[ADD]]) {max = 4.000000e+00 : f64, min = -1.500000e+00 : f64} : tensor<1x8x80x80x!qElemType1> -> tensor<1x8x80x80x!qElemType1>
    // CHECK:   [[SHAPECAST:%.*]] = IE.ShapeCast {shape = [1, 16, 40, 80]} inputs([[CLAMP]] : tensor<1x8x80x80x!qElemType1>) -> tensor<1x16x40x80x!qElemType1>
    // CHECK:   return  [[SHAPECAST]]
}


// -----

!qElemType = !quant.uniform<u8:f16, 1.0:123>
!qElemType1 = !quant.uniform<u8:f16, 1.0:120>
// CHECK-LABEL: @swapSliceWithClamp
// CHECK-SAME:    [[INPUT:%.*]]: tensor<1x16x80x80x!qElemType>
func.func @swapSliceWithClamp(%arg0: tensor<1x16x80x80x!qElemType>) -> tensor<1x8x80x80x!qElemType1> {
    %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x80x80x!qElemType>, tensor<1x16x80x80x!qElemType> -> tensor<1x16x80x80x!qElemType1>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 8, 80, 80] : tensor<1x16x80x80x!qElemType1> to tensor<1x8x80x80x!qElemType1>
    %2 = IE.Clamp(%1) {max = 4.000000e+00 : f64, min = -1.500000e+00 : f64} : tensor<1x8x80x80x!qElemType1> -> tensor<1x8x80x80x!qElemType1>
    return %2 : tensor<1x8x80x80x!qElemType1>

    // CHECK:   [[ADD:%.*]] = IE.Add([[INPUT]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x80x80x!qElemType>, tensor<1x16x80x80x!qElemType> -> tensor<1x16x80x80x!qElemType1>
    // CHECK:   [[CLAMP:%.*]] = IE.Clamp([[ADD]]) {max = 4.000000e+00 : f64, min = -1.500000e+00 : f64} : tensor<1x16x80x80x!qElemType1> -> tensor<1x16x80x80x!qElemType1>
    // CHECK:   [[SLICE:%.*]] = IE.Slice [[CLAMP]] [0, 0, 0, 0] [1, 8, 80, 80] : tensor<1x16x80x80x!qElemType1> to tensor<1x8x80x80x!qElemType1>
    // CHECK:   return  [[SLICE]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0:123>
!qElemType1 = !quant.uniform<u8:f16, 1.0:120>
// CHECK-LABEL: @notSwapWithNCEAlreadyHasPostOp
// CHECK-SAME:    [[INPUT:%.*]]: tensor<1x16x80x80x!qElemType>
func.func @notSwapWithNCEAlreadyHasPostOp(%arg0: tensor<1x16x80x80x!qElemType>) -> tensor<1x8x80x80x!qElemType1> {
    %0 = IE.AvgPool(%arg0) { kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Relu<>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1] }
            : tensor<1x16x80x80x!qElemType> -> tensor<1x16x80x80x!qElemType1>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 8, 80, 80] : tensor<1x16x80x80x!qElemType1> to tensor<1x8x80x80x!qElemType1>
    %2 = IE.Clamp(%1) {max = 4.000000e+00 : f64, min = -1.500000e+00 : f64} : tensor<1x8x80x80x!qElemType1> -> tensor<1x8x80x80x!qElemType1>
    return %2 : tensor<1x8x80x80x!qElemType1>

    // CHECK:   [[POOL:%.*]] = IE.AvgPool([[INPUT]])
    // CHECK:   [[SLICE:%.*]] = IE.Slice [[POOL]]
    // CHECK:   [[CLAMP:%.*]] = IE.Clamp([[SLICE]])
    // CHECK:   return  [[CLAMP]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @swapPermuteCastWithClamp
// CHECK-SAME:    [[INPUT:%.*]]: tensor<1x499x768x1xf16, {order = #NHWC}>
func.func @swapPermuteCastWithClamp(%arg0: tensor<1x499x768x1xf16, {order = #NHWC}>) -> tensor<1x768x1x1xf16> {
    %weights = const.Declare tensor<1x499x1x1xf16> = dense<1.0> : tensor<1x499x1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.Convolution(%arg0, %weights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x499x768x1xf16, {order = #NHWC}>, tensor<1x499x1x1xf16> -> tensor<1x1x768x1xf16, {order = #NHWC}>
    %1 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1x768x1xf16, {order = #NHWC}> -> tensor<1x768x1x1xf16>
    %2 = IE.Clamp(%1) {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64} : tensor<1x768x1x1xf16> -> tensor<1x768x1x1xf16>
    return %2 : tensor<1x768x1x1xf16>

    // CHECK:   [[WEIGHTS:%.*]] = const.Declare tensor<1x499x1x1xf16> = dense<1.000000e+00> : tensor<1x499x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:   [[CONVOLUTION:%.*]] = IE.Convolution([[INPUT]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x499x768x1xf16, {order = #NHWC}>, tensor<1x499x1x1xf16> -> tensor<1x1x768x1xf16, {order = #NHWC}>
    // CHECK:   [[CLAMP:%.*]] = IE.Clamp([[CONVOLUTION]]) {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64} : tensor<1x1x768x1xf16, {order = #NHWC}> -> tensor<1x1x768x1xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST:%.*]] = IE.PermuteCast([[CLAMP]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1x768x1xf16, {order = #NHWC}> -> tensor<1x768x1x1xf16>
    // CHECK:   return  [[PERMUTECAST]]
}
