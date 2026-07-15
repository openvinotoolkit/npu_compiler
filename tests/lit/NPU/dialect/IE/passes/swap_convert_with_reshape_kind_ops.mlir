//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --swap-convert-with-reshape-kind-ops --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapTransposeWithConvert
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x70x1x28xui8>) -> tensor<1x1x28x70xf16>
func.func @SwapTransposeWithConvert(%arg0: tensor<1x70x1x28xui8>) -> tensor<1x1x28x70xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16}
        : tensor<1x70x1x28xui8> -> tensor<1x70x1x28xf16>

    %1 = IE.Transpose(%0) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    return %1 : tensor<1x1x28x70xf16>

    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[ARG_0]]) {order_value = #NHWC}
    // CHECK-SAME:  : tensor<1x70x1x28xui8> -> tensor<1x1x28x70xui8>

    // CHECK:   [[CONVERT:%.+]] = IE.Convert([[TRANSPOSE]])
    // CHECK-SAME:  {dstElemType = f16}
    // CHECK-SAME:  : tensor<1x1x28x70xui8> -> tensor<1x1x28x70xf16>

    // CHECK:   return [[CONVERT]] : tensor<1x1x28x70xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapReshapeWithConvert
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x70x1x28xui8>) -> tensor<1x1x28x70xf16>
func.func @SwapReshapeWithConvert(%arg0: tensor<1x70x1x28xui8>) -> tensor<1x1x28x70xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16}
        : tensor<1x70x1x28xui8> -> tensor<1x70x1x28xf16>

    %1 = IE.Reshape(%0) {shape_value = [1, 1, 28, 70]} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    return %1 : tensor<1x1x28x70xf16>

    // CHECK:   [[RESHAPE:%.+]] = IE.Reshape([[ARG_0]]) {shape_value = [1, 1, 28, 70]}
    // CHECK-SAME:  : tensor<1x70x1x28xui8> -> tensor<1x1x28x70xui8>

    // CHECK:   [[CONVERT:%.+]] = IE.Convert([[RESHAPE]])
    // CHECK-SAME:  {dstElemType = f16}
    // CHECK-SAME:  : tensor<1x1x28x70xui8> -> tensor<1x1x28x70xf16>

    // CHECK:   return [[CONVERT]] : tensor<1x1x28x70xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapAffineReshapeWithConvert
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x70x1x28xui8>) -> tensor<1x1x28x70xf16>
func.func @SwapAffineReshapeWithConvert(%arg0: tensor<1x70x1x28xui8>) -> tensor<1x1x28x70xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16}
        : tensor<1x70x1x28xui8> -> tensor<1x70x1x28xf16>

    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [2], [2], [3], [3]], shape_value = [1, 1, 28, 70]} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    return %1 : tensor<1x1x28x70xf16>

    // CHECK:   [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[ARG_0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [2], [3], [3]], shape_value = [1, 1, 28, 70]}
    // CHECK-SAME:  : tensor<1x70x1x28xui8> -> tensor<1x1x28x70xui8>

    // CHECK:   [[CONVERT:%.+]] = IE.Convert([[AFFINERESHAPE]])
    // CHECK-SAME:  {dstElemType = f16}
    // CHECK-SAME:  : tensor<1x1x28x70xui8> -> tensor<1x1x28x70xf16>

    // CHECK:   return [[CONVERT]] : tensor<1x1x28x70xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapSqueezeWithConvert
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x1x70x28xui8>) -> tensor<70x28xf16>
func.func @SwapSqueezeWithConvert(%arg0: tensor<1x1x70x28xui8>) -> tensor<70x28xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16}
        : tensor<1x1x70x28xui8> -> tensor<1x1x70x28xf16>

    %1 = IE.Squeeze(%0) {axes_value = [0, 1]} : tensor<1x1x70x28xf16> -> tensor<70x28xf16>
    return %1 : tensor<70x28xf16>

    // CHECK:   [[SQUEEZE:%.+]] = IE.Squeeze([[ARG_0]]) {axes_value = [0, 1]}
    // CHECK-SAME:  : tensor<1x1x70x28xui8> -> tensor<70x28xui8>

    // CHECK:   [[CONVERT:%.+]] = IE.Convert([[SQUEEZE]])
    // CHECK-SAME:  {dstElemType = f16}
    // CHECK-SAME:  : tensor<70x28xui8> -> tensor<70x28xf16>

    // CHECK:   return [[CONVERT]] : tensor<70x28xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapUnsqueezeWithConvert
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<70x28xui8>) -> tensor<1x1x70x28xf16>
func.func @SwapUnsqueezeWithConvert(%arg0: tensor<70x28xui8>) -> tensor<1x1x70x28xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16}
        : tensor<70x28xui8> -> tensor<70x28xf16>

    %1 = IE.Unsqueeze(%0) {axes_value = [0, 1]} : tensor<70x28xf16> -> tensor<1x1x70x28xf16>
    return %1 : tensor<1x1x70x28xf16>

    // CHECK:   [[UNSQUEEZE:%.+]] = IE.Unsqueeze([[ARG_0]]) {axes_value = [0, 1]}
    // CHECK-SAME:  : tensor<70x28xui8> -> tensor<1x1x70x28xui8>

    // CHECK:   [[CONVERT:%.+]] = IE.Convert([[UNSQUEEZE]])
    // CHECK-SAME:  {dstElemType = f16}
    // CHECK-SAME:  : tensor<1x1x70x28xui8> -> tensor<1x1x70x28xf16>

    // CHECK:   return [[CONVERT]] : tensor<1x1x70x28xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.0>

// CHECK-LABEL: @DoNotSwapTransposeWithConvert
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x70x1x28xui8>) -> tensor<1x1x28x70xf16>
func.func @DoNotSwapTransposeWithConvert(%arg0: tensor<1x70x1x28xui8>) -> tensor<1x1x28x70xf16> {
    %0 = IE.QuantizeCast(%arg0) {dstElemType = !qElemType} : tensor<1x70x1x28xui8> -> tensor<1x70x1x28x!qElemType>
    %1 = IE.Add(%0, %0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x70x1x28x!qElemType>, tensor<1x70x1x28x!qElemType> -> tensor<1x70x1x28x!qElemType>
    %2 = IE.Convert(%1) {dstElemType = f16} : tensor<1x70x1x28x!qElemType> -> tensor<1x70x1x28xf16>

    %3 = IE.Transpose(%2) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    return %3 : tensor<1x1x28x70xf16>

    // CHECK:   [[VAR0:%.+]] = IE.QuantizeCast([[ARG_0]]) {dstElemType = !qElemType} :
    // CHECK-SAME:     tensor<1x70x1x28xui8> -> tensor<1x70x1x28x!qElemType>

    // CHECK:   [[ADD:%.+]] = IE.Add([[VAR0]], [[VAR0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:  : tensor<1x70x1x28x!qElemType>, tensor<1x70x1x28x!qElemType> -> tensor<1x70x1x28x!qElemType>

    // CHECK:   [[CONVERT:%.+]] = IE.Convert([[ADD]]) {dstElemType = f16} : tensor<1x70x1x28x!qElemType> -> tensor<1x70x1x28xf16>
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[CONVERT]]) {order_value = #NHWC}
    // CHECK-SAME:  : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>

    // CHECK:   return [[TRANSPOSE]] : tensor<1x1x28x70xf16>
}

// -----

// CHECK-LABEL: func @SwapConvertWithDepthToSpaceOutput
// CHECK-SAME:        [[INPUT:%[^:]+]]: tensor<1x16x800x1279xf32>
func.func @SwapConvertWithDepthToSpaceOutput(%arg0: tensor<1x16x800x1279xf32>) -> tensor<1x4x1600x2558xui8> {
    %in_low = const.Declare tensor<1x1x1x1xf32> = dense<-0.34410953521> : tensor<1x1x1x1xf32>
    %in_high = const.Declare tensor<1x1x1x1xf32> = dense<1.1431435> : tensor<1x1x1x1xf32>
    %out_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %out_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>
    %0 = IE.FakeQuantize(%arg0, %in_low, %in_high, %out_low, %out_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x800x1279xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x800x1279xf32>
    %1 = IE.Convert(%0) {dstElemType = ui8} : tensor<1x16x800x1279xf32> -> tensor<1x16x800x1279xui8>
    %2 = IE.DepthToSpace(%1) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x16x800x1279xui8> -> tensor<1x4x1600x2558xui8>
    return %2 : tensor<1x4x1600x2558xui8>

    // CHECK:   [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-0.344109535> : tensor<1x1x1x1xf32>
    // CHECK:   [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.14314353> : tensor<1x1x1x1xf32>
    // CHECK:   [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:   [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK:   [[FAKE_QUANT:%.+]] = IE.FakeQuantize([[INPUT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x800x1279xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x800x1279xf32>
    // CHECK:   [[DEPTH_TO_SPACE:%.+]] = IE.DepthToSpace([[FAKE_QUANT]])  {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x16x800x1279xf32> -> tensor<1x4x1600x2558xf32>
    // CHECK:   [[CONVERT:%.+]] = IE.Convert([[DEPTH_TO_SPACE]]) {dstElemType = ui8} : tensor<1x4x1600x2558xf32> -> tensor<1x4x1600x2558xui8>
    // CHECK:   return [[CONVERT]] : tensor<1x4x1600x2558xui8>

}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: func @SwapConvert2LayersOutput
// CHECK-SAME:        [[INPUT:%[^:]+]]: tensor<1x16x800x1279xf32>
func.func @SwapConvert2LayersOutput(%arg0: tensor<1x16x800x1279xf32>) -> tensor<1x2558x1600x4xui8> {
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<-0.34410953521> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<1.1431435> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>
    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x800x1279xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x800x1279xf32>
    %1 = IE.Convert(%0) {dstElemType = ui8} : tensor<1x16x800x1279xf32> -> tensor<1x16x800x1279xui8>
    %2 = IE.DepthToSpace(%1) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x16x800x1279xui8> -> tensor<1x4x1600x2558xui8>
    %3 = IE.Transpose(%2) {order_value = #NWHC} : tensor<1x4x1600x2558xui8> -> tensor<1x2558x1600x4xui8>
    return %3 : tensor<1x2558x1600x4xui8>

    // CHECK:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-0.344109535> : tensor<1x1x1x1xf32>
    // CHECK:   [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.14314353> : tensor<1x1x1x1xf32>
    // CHECK:   [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:   [[CST_3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK:   [[FAKE_QUANT:%.+]] = IE.FakeQuantize([[INPUT]], [[CST_0]], [[CST_1]], [[CST_2]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x800x1279xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x800x1279xf32>
    // CHECK:   [[DEPTH_TO_SPACE:%.+]] = IE.DepthToSpace([[FAKE_QUANT]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x16x800x1279xf32> -> tensor<1x4x1600x2558xf32>
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[DEPTH_TO_SPACE]]) {order_value = #NWHC} : tensor<1x4x1600x2558xf32> -> tensor<1x2558x1600x4xf32>
    // CHECK:   [[CONVERT:%.+]] = IE.Convert([[TRANSPOSE]]) {dstElemType = ui8} : tensor<1x2558x1600x4xf32> -> tensor<1x2558x1600x4xui8>
    // CHECK:   return [[CONVERT]] : tensor<1x2558x1600x4xui8>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: func @SwapConvert3LayersOutput
// CHECK-SAME:        [[INPUT:%[^:]+]]: tensor<1x16x800x1279xf32>
func.func @SwapConvert3LayersOutput(%arg0: tensor<1x16x800x1279xf32>) -> tensor<2558x1600x4xui8> {
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<-0.34410953521> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<1.1431435> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>
    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x800x1279xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x800x1279xf32>
    %1 = IE.Convert(%0) {dstElemType = ui8} : tensor<1x16x800x1279xf32> -> tensor<1x16x800x1279xui8>
    %2 = IE.DepthToSpace(%1) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x16x800x1279xui8> -> tensor<1x4x1600x2558xui8>
    %3 = IE.Transpose(%2) {order_value = #NWHC} : tensor<1x4x1600x2558xui8> -> tensor<1x2558x1600x4xui8>
    %4 = IE.Squeeze(%3) {axes_value = [0]} : tensor<1x2558x1600x4xui8> -> tensor<2558x1600x4xui8>
    return %4 : tensor<2558x1600x4xui8>

    // CHECK:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-0.344109535> : tensor<1x1x1x1xf32>
    // CHECK:   [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.14314353> : tensor<1x1x1x1xf32>
    // CHECK:   [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:   [[CST_3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK:   [[FAKE_QUANT:%.+]] = IE.FakeQuantize([[INPUT]], [[CST_0]], [[CST_1]], [[CST_2]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x800x1279xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x800x1279xf32>
    // CHECK:   [[DEPTH_TO_SPACE:%.+]] = IE.DepthToSpace([[FAKE_QUANT]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x16x800x1279xf32> -> tensor<1x4x1600x2558xf32>
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[DEPTH_TO_SPACE]]) {order_value = #NWHC} : tensor<1x4x1600x2558xf32> -> tensor<1x2558x1600x4xf32>
    // CHECK:   [[SQUEEZE:%.+]] = IE.Squeeze([[TRANSPOSE]]) {axes_value = [0]} : tensor<1x2558x1600x4xf32> -> tensor<2558x1600x4xf32>
    // CHECK:   [[CONVERT:%.+]] = IE.Convert([[SQUEEZE]]) {dstElemType = ui8} : tensor<2558x1600x4xf32> -> tensor<2558x1600x4xui8>
    // CHECK:   return [[CONVERT]] : tensor<2558x1600x4xui8>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>


// CHECK-LABEL: func @NotSwapConvertWithNonAgnosticOp
// CHECK-SAME:        [[INPUT:%[^:]+]]: tensor<1x16x800x1279xf32>
func.func @NotSwapConvertWithNonAgnosticOp(%arg0: tensor<1x16x800x1279xf32>) -> tensor<2558x1600x4xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<-0.34410953521> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<1.1431435> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>
    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x800x1279xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x800x1279xf32>
    %1 = IE.Convert(%0) {dstElemType = ui8} : tensor<1x16x800x1279xf32> -> tensor<1x16x800x1279xui8>
    %2 = IE.DepthToSpace(%1) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x16x800x1279xui8> -> tensor<1x4x1600x2558xui8>
    %3 = IE.Transpose(%2) {order_value = #NWHC} : tensor<1x4x1600x2558xui8> -> tensor<1x2558x1600x4xui8>
    %4 = IE.Squeeze(%3) {axes_value = [0]} : tensor<1x2558x1600x4xui8> -> tensor<2558x1600x4xui8>
    %5 = IE.Convert(%4) {dstElemType = f16} : tensor<2558x1600x4xui8> -> tensor<2558x1600x4xf16>
    return %5 : tensor<2558x1600x4xf16>

    // CHECK:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-0.344109535> : tensor<1x1x1x1xf32>
    // CHECK:   [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.14314353> : tensor<1x1x1x1xf32>
    // CHECK:   [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:   [[CST_3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK:   [[FAKE_QUANT:%.+]] = IE.FakeQuantize([[INPUT]], [[CST_0]], [[CST_1]], [[CST_2]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x800x1279xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x800x1279xf32>
    // CHECK:   [[CONVERT:%.+]] = IE.Convert([[FAKE_QUANT]]) {dstElemType = ui8} : tensor<1x16x800x1279xf32> -> tensor<1x16x800x1279xui8>
    // CHECK:   [[DEPTH_TO_SPACE:%.+]] = IE.DepthToSpace([[CONVERT]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x16x800x1279xui8> -> tensor<1x4x1600x2558xui8>
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[DEPTH_TO_SPACE]]) {order_value = #NWHC} : tensor<1x4x1600x2558xui8> -> tensor<1x2558x1600x4xui8>
    // CHECK:   [[SQUEEZE:%.+]] = IE.Squeeze([[TRANSPOSE]]) {axes_value = [0]} : tensor<1x2558x1600x4xui8> -> tensor<2558x1600x4xui8>
    // CHECK:   [[CONVERT_1:%.+]] = IE.Convert([[SQUEEZE]]) {dstElemType = f16} : tensor<2558x1600x4xui8> -> tensor<2558x1600x4xf16>
    // CHECK:   return [[CONVERT_1]] : tensor<2558x1600x4xf16>
}

// -----

// CHECK-LABEL: func @PropagateConvertForwardToFuse
// CHECK-SAME:        [[INPUT:%[^:]+]]: tensor<1x1152xsi64>
func.func @PropagateConvertForwardToFuse(%arg0: tensor<1x1152xsi64>) -> tensor<1x1x1x1152xi8> {
    %cst = const.Declare tensor<1x1x1x1152xsi32> = dense<1> : tensor<1x1x1x1152xsi32>
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 1152]} : tensor<1x1152xsi64> -> tensor<1x1x1x1152xsi64>
    %1 = IE.Convert(%0) {dstElemType = f16} : tensor<1x1x1x1152xsi64> -> tensor<1x1x1x1152xf16>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 1152]} : tensor<1x1x1x1152xf16> -> tensor<1x1152xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [0, 1]], shape_value = [1152, 1]} : tensor<1x1152xf16> -> tensor<1152x1xf16>
    %4 = IE.Gather(%3, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 4 : i64} : tensor<1152x1xf16>, tensor<1x1x1x1152xsi32> -> tensor<1x1x1x1152x1xf16>
    %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [1], [2], [3], [3]], shape_value = [1, 1, 1, 1152]} : tensor<1x1x1x1152x1xf16> -> tensor<1x1x1x1152xf16>
    %6 = IE.Convert(%5) {dstElemType = i8} : tensor<1x1x1x1152xf16> -> tensor<1x1x1x1152xi8>

    return %6 : tensor<1x1x1x1152xi8>

    // CHECK:   [[CST:%.+]] = const.Declare tensor<1x1x1x1152xsi32> = dense<1> : tensor<1x1x1x1152xsi32>
    // CHECK:   [[AFFINE_RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK:   [[CONVERT:%.+]] = IE.Convert([[AFFINE_RESHAPE_0]]) {dstElemType = i8} : tensor<1x1x1x1152xsi64> -> tensor<1x1x1x1152xi8>
    // CHECK:   [[AFFINE_RESHAPE_1:%.+]] = IE.AffineReshape([[CONVERT]])
    // CHECK:   [[AFFINE_RESHAPE_2:%.+]] = IE.AffineReshape([[AFFINE_RESHAPE_1]])
    // CHECK:   [[GATHER:%.+]] = IE.Gather([[AFFINE_RESHAPE_2]], [[CST]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 4 : i64} : tensor<1152x1xi8>, tensor<1x1x1x1152xsi32> -> tensor<1x1x1x1152x1xi8>
    // CHECK:   [[AFFINE_RESHAPE_3:%.+]] = IE.AffineReshape([[GATHER]])

    // CHECK:   return [[AFFINE_RESHAPE_3]] : tensor<1x1x1x1152xi8>
}

// -----

// CHECK-LABEL: func @PropagateConvertBackwardToFuse
// CHECK-SAME:        [[INPUT:%[^:]+]]: tensor<1x1152xi8>
func.func @PropagateConvertBackwardToFuse(%arg0: tensor<1x1152xi8>) -> tensor<1x1x1x1152xsi64> {
    %cst = const.Declare tensor<1x1x1x1152xsi32> = dense<1> : tensor<1x1x1x1152xsi32>
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 1152]} : tensor<1x1152xi8> -> tensor<1x1x1x1152xi8>
    %1 = IE.Convert(%0) {dstElemType = f16} : tensor<1x1x1x1152xi8> -> tensor<1x1x1x1152xf16>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 1152]} : tensor<1x1x1x1152xf16> -> tensor<1x1152xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [0, 1]], shape_value = [1152, 1]} : tensor<1x1152xf16> -> tensor<1152x1xf16>
    %4 = IE.Gather(%3, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 4 : i64} : tensor<1152x1xf16>, tensor<1x1x1x1152xsi32> -> tensor<1x1x1x1152x1xf16>
    %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [1], [2], [3], [3]], shape_value = [1, 1, 1, 1152]} : tensor<1x1x1x1152x1xf16> -> tensor<1x1x1x1152xf16>
    %6 = IE.Convert(%5) {dstElemType = si64} : tensor<1x1x1x1152xf16> -> tensor<1x1x1x1152xsi64>

    return %6 : tensor<1x1x1x1152xsi64>

    // CHECK:   [[CST:%.+]] = const.Declare tensor<1x1x1x1152xsi32> = dense<1> : tensor<1x1x1x1152xsi32>
    // CHECK:   [[AFFINE_RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK:   [[AFFINE_RESHAPE_1:%.+]] = IE.AffineReshape([[AFFINE_RESHAPE_0]])
    // CHECK:   [[AFFINE_RESHAPE_2:%.+]] = IE.AffineReshape([[AFFINE_RESHAPE_1]])
    // CHECK:   [[GATHER:%.+]] = IE.Gather([[AFFINE_RESHAPE_2]], [[CST]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 4 : i64} : tensor<1152x1xi8>, tensor<1x1x1x1152xsi32> -> tensor<1x1x1x1152x1xi8>
    // CHECK:   [[AFFINE_RESHAPE_3:%.+]] = IE.AffineReshape([[GATHER]])
    // CHECK:   [[CONVERT:%.+]] = IE.Convert([[AFFINE_RESHAPE_3]]) {dstElemType = si64} : tensor<1x1x1x1152xi8> -> tensor<1x1x1x1152xsi64>
    // CHECK:   return [[CONVERT]] : tensor<1x1x1x1152xsi64>
}

// -----

// CHECK-LABEL: func @NoPropagateConvertToFuse
// CHECK-SAME:        [[INPUT:%[^:]+]]: tensor<1x1152xsi64>
func.func @NoPropagateConvertToFuse(%arg0: tensor<1x1152xsi64>) -> tensor<1x1x1x1152xsi32> {
    %cst = const.Declare tensor<1x1x1x1152xsi32> = dense<1> : tensor<1x1x1x1152xsi32>
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 1152]} : tensor<1x1152xsi64> -> tensor<1x1x1x1152xsi64>
    %1 = IE.Convert(%0) {dstElemType = f16} : tensor<1x1x1x1152xsi64> -> tensor<1x1x1x1152xf16>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 1152]} : tensor<1x1x1x1152xf16> -> tensor<1x1152xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [0, 1]], shape_value = [1152, 1]} : tensor<1x1152xf16> -> tensor<1152x1xf16>
    %4 = IE.Gather(%3, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 4 : i64} : tensor<1152x1xf16>, tensor<1x1x1x1152xsi32> -> tensor<1x1x1x1152x1xf16>
    %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [1], [2], [3], [3]], shape_value = [1, 1, 1, 1152]} : tensor<1x1x1x1152x1xf16> -> tensor<1x1x1x1152xf16>
    %6 = IE.Convert(%5) {dstElemType = si32} : tensor<1x1x1x1152xf16> -> tensor<1x1x1x1152xsi32>

    return %6 : tensor<1x1x1x1152xsi32>

    // CHECK:   [[CST:%.+]] = const.Declare tensor<1x1x1x1152xsi32> = dense<1> : tensor<1x1x1x1152xsi32>
    // CHECK:   [[AFFINE_RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK:   [[CONVERT_0:%.+]] = IE.Convert([[AFFINE_RESHAPE_0]]) {dstElemType = f16} : tensor<1x1x1x1152xsi64> -> tensor<1x1x1x1152xf16>
    // CHECK:   [[AFFINE_RESHAPE_1:%.+]] = IE.AffineReshape([[CONVERT_0]])
    // CHECK:   [[AFFINE_RESHAPE_2:%.+]] = IE.AffineReshape([[AFFINE_RESHAPE_1]])
    // CHECK:   [[GATHER:%.+]] = IE.Gather([[AFFINE_RESHAPE_2]], [[CST]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 4 : i64} : tensor<1152x1xf16>, tensor<1x1x1x1152xsi32> -> tensor<1x1x1x1152x1xf16>
    // CHECK:   [[AFFINE_RESHAPE_3:%.+]] = IE.AffineReshape([[GATHER]])
    // CHECK:   [[CONVERT_1:%.+]] = IE.Convert([[AFFINE_RESHAPE_3]]) {dstElemType = si32} : tensor<1x1x1x1152xf16> -> tensor<1x1x1x1152xsi32>

    // CHECK:   return [[CONVERT_1]] : tensor<1x1x1x1152xsi32>
}

// -----

// CHECK-LABEL: func @PropagateConvertForwardNoInfiniteLoop
// CHECK-SAME:        [[INPUT:%[^:]+]]: tensor<1x1152xsi64>
func.func @PropagateConvertForwardNoInfiniteLoop(%arg0: tensor<1x1152xsi64>) -> tensor<1x1x576x2xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 1152]} : tensor<1x1152xsi64> -> tensor<1x1x1x1152xsi64>
    %1 = IE.Convert(%0) {dstElemType = si32} : tensor<1x1x1x1152xsi64> -> tensor<1x1x1x1152xsi32>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [2], [3, 4]], shape_value = [1, 1, 576, 2]} : tensor<1x1x1x1152xsi32> -> tensor<1x1x576x2xsi32>
    %3 = IE.Convert(%2) {dstElemType = f16} : tensor<1x1x576x2xsi32> -> tensor<1x1x576x2xf16>

    return %3 : tensor<1x1x576x2xf16>

    // CHECK:   [[AFFINE_RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME:  : tensor<1x1152xsi64> -> tensor<1x1x1x1152xsi64>
    // CHECK:   [[CONVERT:%.+]] = IE.Convert([[AFFINE_RESHAPE_0]]) {dstElemType = f16}
    // CHECK-SAME:  : tensor<1x1x1x1152xsi64> -> tensor<1x1x1x1152xf16>
    // CHECK:   [[AFFINE_RESHAPE_1:%.+]] = IE.AffineReshape([[CONVERT]])
    // CHECK-SAME:  : tensor<1x1x1x1152xf16> -> tensor<1x1x576x2xf16>
    // CHECK:   return [[AFFINE_RESHAPE_1]] : tensor<1x1x576x2xf16>
}

// -----

// Test EliminateConvertRoundTripThroughReshapeAndSlice pattern directly.
// Input is post-UniquifyBranches form: AffineReshape already merged before Slice.

// CHECK-LABEL: func @EliminateConvertRoundTripThroughReshapeAndSlice
// CHECK-SAME:        [[INPUT:%[^:]+]]: tensor<1x32x4096x1xf32>
func.func @EliminateConvertRoundTripThroughReshapeAndSlice(%arg0: tensor<1x32x4096x1xf32>) -> (tensor<1x32768x1x1xf32>, tensor<1x32768x1x1xf32>, tensor<1x32768x1x1xf32>, tensor<1x32768x1x1xf32>) {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x32x4096x1xf32> -> tensor<1x32x4096x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 131072, 1, 1]}
        : tensor<1x32x4096x1xf16> -> tensor<1x131072x1x1xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 32768, 1, 1] : tensor<1x131072x1x1xf16> to tensor<1x32768x1x1xf16>
    %3 = IE.Slice %1 [0, 32768, 0, 0] [1, 32768, 1, 1] : tensor<1x131072x1x1xf16> to tensor<1x32768x1x1xf16>
    %4 = IE.Slice %1 [0, 65536, 0, 0] [1, 32768, 1, 1] : tensor<1x131072x1x1xf16> to tensor<1x32768x1x1xf16>
    %5 = IE.Slice %1 [0, 98304, 0, 0] [1, 32768, 1, 1] : tensor<1x131072x1x1xf16> to tensor<1x32768x1x1xf16>
    %6 = IE.Convert(%2) {dstElemType = f32} : tensor<1x32768x1x1xf16> -> tensor<1x32768x1x1xf32>
    %7 = IE.Convert(%3) {dstElemType = f32} : tensor<1x32768x1x1xf16> -> tensor<1x32768x1x1xf32>
    %8 = IE.Convert(%4) {dstElemType = f32} : tensor<1x32768x1x1xf16> -> tensor<1x32768x1x1xf32>
    %9 = IE.Convert(%5) {dstElemType = f32} : tensor<1x32768x1x1xf16> -> tensor<1x32768x1x1xf32>
    return %6, %7, %8, %9 : tensor<1x32768x1x1xf32>, tensor<1x32768x1x1xf32>, tensor<1x32768x1x1xf32>, tensor<1x32768x1x1xf32>

    // CHECK:   [[RESHAPE:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 131072, 1, 1]}
    // CHECK-SAME:  : tensor<1x32x4096x1xf32> -> tensor<1x131072x1x1xf32>
    // CHECK:   [[SLICE0:%.+]] = IE.Slice [[RESHAPE]] [0, 0, 0, 0] [1, 32768, 1, 1] : tensor<1x131072x1x1xf32> to tensor<1x32768x1x1xf32>
    // CHECK:   [[SLICE1:%.+]] = IE.Slice [[RESHAPE]] [0, 32768, 0, 0] [1, 32768, 1, 1] : tensor<1x131072x1x1xf32> to tensor<1x32768x1x1xf32>
    // CHECK:   [[SLICE2:%.+]] = IE.Slice [[RESHAPE]] [0, 65536, 0, 0] [1, 32768, 1, 1] : tensor<1x131072x1x1xf32> to tensor<1x32768x1x1xf32>
    // CHECK:   [[SLICE3:%.+]] = IE.Slice [[RESHAPE]] [0, 98304, 0, 0] [1, 32768, 1, 1] : tensor<1x131072x1x1xf32> to tensor<1x32768x1x1xf32>
    // CHECK:   return [[SLICE0]], [[SLICE1]], [[SLICE2]], [[SLICE3]]
}

// -----

// CHECK-LABEL: func @NoEliminateConvertRoundTripSliceMultipleUsers
// CHECK-SAME:        [[INPUT:%[^:]+]]: tensor<1x32x4096x1xf32>
func.func @NoEliminateConvertRoundTripSliceMultipleUsers(%arg0: tensor<1x32x4096x1xf32>) -> (tensor<1x32768x1x1xf16>, tensor<1x32768x1x1xf32>) {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x32x4096x1xf32> -> tensor<1x32x4096x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 131072, 1, 1]}
        : tensor<1x32x4096x1xf16> -> tensor<1x131072x1x1xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 32768, 1, 1] : tensor<1x131072x1x1xf16> to tensor<1x32768x1x1xf16>
    %3 = IE.Slice %1 [0, 32768, 0, 0] [1, 32768, 1, 1] : tensor<1x131072x1x1xf16> to tensor<1x32768x1x1xf16>
    %4 = IE.Convert(%3) {dstElemType = f32} : tensor<1x32768x1x1xf16> -> tensor<1x32768x1x1xf32>
    return %2, %4 : tensor<1x32768x1x1xf16>, tensor<1x32768x1x1xf32>

    // The pattern should not fire because not all Slice users have Convert(f16->f32) as sole user.
    // Slice %2 is returned directly as f16, not converted back.
    // OpSwapConverter swaps Convert before AffineReshape, resulting in AffineReshape(f32) -> Convert(f32->f16).

    // CHECK:   [[RESHAPE:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 131072, 1, 1]}
    // CHECK-SAME:  : tensor<1x32x4096x1xf32> -> tensor<1x131072x1x1xf32>
    // CHECK:   [[CONVERT:%.+]] = IE.Convert([[RESHAPE]]) {dstElemType = f16}
    // CHECK-SAME:  : tensor<1x131072x1x1xf32> -> tensor<1x131072x1x1xf16>
    // CHECK:   [[SLICE0:%.+]] = IE.Slice [[CONVERT]] [0, 0, 0, 0] [1, 32768, 1, 1] : tensor<1x131072x1x1xf16> to tensor<1x32768x1x1xf16>
    // CHECK:   [[SLICE1:%.+]] = IE.Slice [[CONVERT]] [0, 32768, 0, 0] [1, 32768, 1, 1] : tensor<1x131072x1x1xf16> to tensor<1x32768x1x1xf16>
    // CHECK:   [[CONVERT_BACK:%.+]] = IE.Convert([[SLICE1]]) {dstElemType = f32}
    // CHECK-SAME:  : tensor<1x32768x1x1xf16> -> tensor<1x32768x1x1xf32>
    // CHECK:   return [[SLICE0]], [[CONVERT_BACK]]
}
