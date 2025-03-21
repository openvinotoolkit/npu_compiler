//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --mlir-print-elementsattrs-with-hex-if-larger=-1 --fuse-reshape-mvn %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseReorderWithMVN1
module @FuseReorderWithMVN1 {

// CHECK-LABEL: @main
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x256x256x256xf16, {order = #NHWC}>)
  func.func @main(%arg0: tensor<1x256x256x256xf16, {order = #NHWC}>) -> tensor<1x256x256x256xf16, {order = #NHWC}> {
    %2 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x256x256x256xf16, {order = #NHWC}> -> tensor<1x256x256x256xf16>
    %3 = IE.Reshape(%2) {shape_value = [1, 32, 524288, 1]} : tensor<1x256x256x256xf16> -> tensor<1x32x524288x1xf16>
    %4 = IE.Reorder(%3) {dstOrder = #NHWC} : tensor<1x32x524288x1xf16> -> tensor<1x32x524288x1xf16, {order = #NHWC}>
    %5 = IE.MVN(%4) {across_channels = false, eps = 5.000000e-01 : f64, normalize_variance = true} : tensor<1x32x524288x1xf16, {order = #NHWC}> -> tensor<1x32x524288x1xf16, {order = #NHWC}>
    %6 = IE.Reorder(%5) {dstOrder = #NCHW} : tensor<1x32x524288x1xf16, {order = #NHWC}> -> tensor<1x32x524288x1xf16>
    %7 = IE.Reshape(%6) {shape_value = [1, 256, 256, 256]} : tensor<1x32x524288x1xf16> -> tensor<1x256x256x256xf16>
    %8 = IE.Reorder(%7) {dstOrder = #NHWC} : tensor<1x256x256x256xf16> -> tensor<1x256x256x256xf16, {order = #NHWC}>
    return %8 : tensor<1x256x256x256xf16, {order = #NHWC}>

    // CHECK: [[VAR0:%.*]] = IE.MVN([[INPUT]]) {across_channels = false, eps = 5.000000e-01 : f64, internal_reshape = [1, 32, 524288, 1], normalize_variance = true} : tensor<1x256x256x256xf16, {order = #NHWC}> -> tensor<1x256x256x256xf16, {order = #NHWC}>
    // CHECK: return [[VAR0]] : tensor<1x256x256x256xf16, {order = #NHWC}>
  }

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MoveGroupConvPostFuseReorderWithMVN
module @MoveGroupConvPostFuseReorderWithMVN {

// CHECK-LABEL: @main
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x256x256x256xf16, {order = #NHWC}>)
  func.func @main(%arg0: tensor<1x256x256x256xf16, {order = #NHWC}>) -> tensor<1x256x256x256xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x1x1x1xf16, {order = #NHWC}> = dense<9.000000e+00> : tensor<32x1x1x1xf16, {order = #NHWC}>
    %2 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x256x256x256xf16, {order = #NHWC}> -> tensor<1x256x256x256xf16>
    %3 = IE.Reshape(%2) {shape_value = [1, 32, 524288, 1]} : tensor<1x256x256x256xf16> -> tensor<1x32x524288x1xf16>
    %4 = IE.Reorder(%3) {dstOrder = #NHWC} : tensor<1x32x524288x1xf16> -> tensor<1x32x524288x1xf16, {order = #NHWC}>
    %5 = IE.MVN(%4) {across_channels = false, eps = 5.000000e-01 : f64, normalize_variance = true} : tensor<1x32x524288x1xf16, {order = #NHWC}> -> tensor<1x32x524288x1xf16, {order = #NHWC}>
    %6 = IE.Reorder(%5) {dstOrder = #NCHW} : tensor<1x32x524288x1xf16, {order = #NHWC}> -> tensor<1x32x524288x1xf16>
    %7 = IE.AffineReshape(%6) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 32, 1, 524288]} : tensor<1x32x524288x1xf16> -> tensor<1x32x1x524288xf16>
    %8 = IE.Reorder(%7) {dstOrder = #NHWC} : tensor<1x32x1x524288xf16> -> tensor<1x32x1x524288xf16, {order = #NHWC}>
    %9 = IE.GroupConvolution(%8, %cst_0) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x1x524288xf16, {order = #NHWC}>, tensor<32x1x1x1xf16, {order = #NHWC}> -> tensor<1x32x1x524288xf16, {order = #NHWC}>
    %10 = IE.Reorder(%9) {dstOrder = #NCHW} : tensor<1x32x1x524288xf16, {order = #NHWC}> -> tensor<1x32x1x524288xf16>
    %11 = IE.Reshape(%10) {shape_value = [1, 256, 256, 256]} : tensor<1x32x1x524288xf16> -> tensor<1x256x256x256xf16>
    %12 = IE.Reorder(%11) {dstOrder = #NHWC} : tensor<1x256x256x256xf16> -> tensor<1x256x256x256xf16, {order = #NHWC}>

    return %12 : tensor<1x256x256x256xf16, {order = #NHWC}>

    // CHECK: [[CST:%.*]] = const.Declare tensor<256x1x1x1xf16, {order = #NHWC}> = dense<9.000000e+00> : tensor<256x1x1x1xf16, {order = #NHWC}>
    // CHECK: [[VAR0:%.*]] = IE.MVN([[INPUT]]) {across_channels = false, eps = 5.000000e-01 : f64, internal_reshape = [1, 32, 524288, 1], normalize_variance = true} : tensor<1x256x256x256xf16, {order = #NHWC}> -> tensor<1x256x256x256xf16, {order = #NHWC}>
    // CHECK: [[VAR1:%.*]] = IE.GroupConvolution([[VAR0]], [[CST]]) {dilations = [1, 1], groups = 256 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x256x256xf16, {order = #NHWC}>, tensor<256x1x1x1xf16, {order = #NHWC}> -> tensor<1x256x256x256xf16, {order = #NHWC}>
    // CHECK: return [[VAR1]] : tensor<1x256x256x256xf16, {order = #NHWC}>
  }

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseReshapeMVNWithConcatAndGroupConv
// CHECK-SAME: [[INPUT_0:%arg[0-9]]]: tensor<1x128x256x256xf16>
// CHECK-SAME: [[INPUT_1:%arg[0-9]]]: tensor<1x128x256x256xf16>
func.func @FuseReshapeMVNWithConcatAndGroupConv(%arg0: tensor<1x128x256x256xf16>, %arg1: tensor<1x128x256x256xf16>
              ) -> (tensor<1x256x256x256xf16, {order = #NHWC}>, tensor<1x256x256x256xf16, {order = #NHWC}>) {
    %cst_0 = const.Declare tensor<32x1x1x1xf16, {order = #NHWC}> = dense<[
                  [[[1.000000e+00]]],[[[2.000000e+00]]],[[[3.000000e+00]]],[[[4.000000e+00]]],
                  [[[5.000000e+00]]],[[[6.000000e+00]]],[[[7.000000e+00]]],[[[8.000000e+00]]],
                  [[[9.000000e+00]]],[[[10.00000e+00]]],[[[11.00000e+00]]],[[[12.00000e+00]]],
                  [[[13.00000e+00]]],[[[14.00000e+00]]],[[[15.00000e+00]]],[[[16.00000e+00]]]
                ]> : tensor<16x1x1x1xf16>, [#const.Broadcast<1 : i64, 2 : i64>, #const.Reshape<[32, 1, 1, 1]>, #const.Reorder<#NHWC>]

    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]} : tensor<1x128x256x256xf16>, tensor<1x128x256x256xf16> -> tensor<1x256x256x256xf16>
    %1 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x256x256x256xf16> -> tensor<1x256x256x256xf16, {order = #NHWC}>

    %2 = IE.Reshape(%0) {shape_value = [1, 32, 524288, 1]} : tensor<1x256x256x256xf16> -> tensor<1x32x524288x1xf16>
    %3 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<1x32x524288x1xf16> -> tensor<1x32x524288x1xf16, {order = #NHWC}>
    %4 = IE.MVN(%3) {across_channels = false, eps = 5.000000e-01 : f64, normalize_variance = true} : tensor<1x32x524288x1xf16, {order = #NHWC}> -> tensor<1x32x524288x1xf16, {order = #NHWC}>
    %5 = IE.Reorder(%4) {dstOrder = #NCHW} : tensor<1x32x524288x1xf16, {order = #NHWC}> -> tensor<1x32x524288x1xf16>
    %6 = IE.AffineReshape(%5) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 32, 1, 524288]} : tensor<1x32x524288x1xf16> -> tensor<1x32x1x524288xf16>
    %7 = IE.Reorder(%6) {dstOrder = #NHWC} : tensor<1x32x1x524288xf16> -> tensor<1x32x1x524288xf16, {order = #NHWC}>
    %8 = IE.GroupConvolution(%7, %cst_0) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x1x524288xf16, {order = #NHWC}>, tensor<32x1x1x1xf16, {order = #NHWC}> -> tensor<1x32x1x524288xf16, {order = #NHWC}>
    %9 = IE.Reorder(%8) {dstOrder = #NCHW} : tensor<1x32x1x524288xf16, {order = #NHWC}> -> tensor<1x32x1x524288xf16>
    %10 = IE.Reshape(%9) {shape_value = [1, 256, 256, 256]} : tensor<1x32x1x524288xf16> -> tensor<1x256x256x256xf16>
    %11 = IE.Reorder(%10) {dstOrder = #NHWC} : tensor<1x256x256x256xf16> -> tensor<1x256x256x256xf16, {order = #NHWC}>

    return %1, %11 : tensor<1x256x256x256xf16, {order = #NHWC}>, tensor<1x256x256x256xf16, {order = #NHWC}>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<256x1x1x1xf16, {order = #NHWC}> = dense<[
    // CHECK-SAME{LITERAL}:     [[[1.000000e+00]]], [[[1.000000e+00]]], [[[1.000000e+00]]], [[[1.000000e+00]]],
    // CHECK-SAME{LITERAL}:     [[[1.000000e+00]]], [[[1.000000e+00]]], [[[1.000000e+00]]], [[[1.000000e+00]]],
    // CHECK-SAME{LITERAL}:     [[[1.000000e+00]]], [[[1.000000e+00]]], [[[1.000000e+00]]], [[[1.000000e+00]]],
    // CHECK-SAME{LITERAL}:     [[[1.000000e+00]]], [[[1.000000e+00]]], [[[1.000000e+00]]], [[[1.000000e+00]]],
    // CHECK-SAME{LITERAL}:     [[[2.000000e+00]]], [[[2.000000e+00]]], [[[2.000000e+00]]], [[[2.000000e+00]]],
    // CHECK-SAME{LITERAL}:     [[[2.000000e+00]]], [[[2.000000e+00]]], [[[2.000000e+00]]], [[[2.000000e+00]]],
    // CHECK-SAME{LITERAL}:     [[[2.000000e+00]]], [[[2.000000e+00]]], [[[2.000000e+00]]], [[[2.000000e+00]]],
    // CHECK-SAME{LITERAL}:     [[[2.000000e+00]]], [[[2.000000e+00]]], [[[2.000000e+00]]], [[[2.000000e+00]]],
    // CHECK-SAME{LITERAL}:   ]> : tensor<256x1x1x1xf16, {order = #NHWC}>

    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME{LITERAL}:   static_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]} : tensor<1x128x256x256xf16>, tensor<1x128x256x256xf16> -> tensor<1x256x256x256xf16>
    // CHECK:   [[REORDER:%.+]] = IE.Reorder([[CONCAT]]) {dstOrder = #NHWC} : tensor<1x256x256x256xf16> -> tensor<1x256x256x256xf16, {order = #NHWC}>

    // CHECK:   [[MVN:%.+]] = IE.MVN([[REORDER]]) {
    // CHECK-SAME:            across_channels = false, eps = 5.000000e-01 : f64, internal_reshape = [1, 32, 524288, 1], normalize_variance = true} : tensor<1x256x256x256xf16, {order = #NHWC}> -> tensor<1x256x256x256xf16, {order = #NHWC}>
    // CHECK:   [[CONV:%.+]] = IE.GroupConvolution([[MVN]], [[FILTER]]) {
    // CHECK-SAME:            dilations = [1, 1], groups = 256 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x256x256xf16, {order = #NHWC}>, tensor<256x1x1x1xf16, {order = #NHWC}> -> tensor<1x256x256x256xf16, {order = #NHWC}>

    // CHECK: return [[REORDER]], [[CONV]] : tensor<1x256x256x256xf16, {order = #NHWC}>, tensor<1x256x256x256xf16, {order = #NHWC}>
              }

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MoveGroupConvPostFuseReorderWithMVNDequantized
module @MoveGroupConvPostFuseReorderWithMVNDequantized {

// CHECK-LABEL: @main
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x256x256x256xf16, {order = #NHWC}>)
  func.func @main(%arg0: tensor<1x256x256x256xf16, {order = #NHWC}>) -> tensor<1x256x256x256xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x1x1x1x!quant.uniform<u8:f16, 0.0039836037392709765>> = dense<9> : tensor<32xui8>, [#const.Reshape<[1, 1, 1, 32]>, #const.CastElemType<f32>,
             #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!quant.uniform<u8:f16, 0.0039836037392709765>>, #const.Reshape<[32, 1, 1, 1]>]
    %2 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x256x256x256xf16, {order = #NHWC}> -> tensor<1x256x256x256xf16>
    %3 = IE.Reshape(%2) {shape_value = [1, 32, 524288, 1]} : tensor<1x256x256x256xf16> -> tensor<1x32x524288x1xf16>
    %4 = IE.Reorder(%3) {dstOrder = #NHWC} : tensor<1x32x524288x1xf16> -> tensor<1x32x524288x1xf16, {order = #NHWC}>
    %5 = IE.MVN(%4) {across_channels = false, eps = 5.000000e-01 : f64, normalize_variance = true} : tensor<1x32x524288x1xf16, {order = #NHWC}> -> tensor<1x32x524288x1xf16, {order = #NHWC}>
    %6 = IE.Reorder(%5) {dstOrder = #NCHW} : tensor<1x32x524288x1xf16, {order = #NHWC}> -> tensor<1x32x524288x1xf16>
    %7 = IE.AffineReshape(%6) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 32, 1, 524288]} : tensor<1x32x524288x1xf16> -> tensor<1x32x1x524288xf16>
    %8 = IE.Reorder(%7) {dstOrder = #NHWC} : tensor<1x32x1x524288xf16> -> tensor<1x32x1x524288xf16, {order = #NHWC}>
    %dequantize = IE.Dequantize(%cst_0) {dstElemType = f16} : tensor<32x1x1x1x!quant.uniform<u8:f16, 0.0039836037392709765>> -> tensor<32x1x1x1xf16>
    %reorder = IE.Reorder(%dequantize) {dstOrder = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<32x1x1x1xf16> -> tensor<32x1x1x1xf16, {order = #NHWC}>
    %9 = IE.GroupConvolution(%8, %reorder) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x1x524288xf16, {order = #NHWC}>, tensor<32x1x1x1xf16, {order = #NHWC}> -> tensor<1x32x1x524288xf16, {order = #NHWC}>
    %10 = IE.Reorder(%9) {dstOrder = #NCHW} : tensor<1x32x1x524288xf16, {order = #NHWC}> -> tensor<1x32x1x524288xf16>
    %11 = IE.Reshape(%10) {shape_value = [1, 256, 256, 256]} : tensor<1x32x1x524288xf16> -> tensor<1x256x256x256xf16>
    %12 = IE.Reorder(%11) {dstOrder = #NHWC} : tensor<1x256x256x256xf16> -> tensor<1x256x256x256xf16, {order = #NHWC}>

    return %12 : tensor<1x256x256x256xf16, {order = #NHWC}>

    // CHECK: [[CST:%.+]] = const.Declare tensor<256x1x1x1xf16, {order = #NHWC}> = dense<3.585820e-02> : tensor<256x1x1x1xf16, {order = #NHWC}>
    // CHECK: [[VAR0:%.+]] = IE.MVN([[INPUT]]) {across_channels = false, eps = 5.000000e-01 : f64, internal_reshape = [1, 32, 524288, 1], normalize_variance = true} : tensor<1x256x256x256xf16, {order = #NHWC}> -> tensor<1x256x256x256xf16, {order = #NHWC}>
    // CHECK: [[VAR1:%.+]] = IE.GroupConvolution([[VAR0]], [[CST]]) {dilations = [1, 1], groups = 256 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x256x256xf16, {order = #NHWC}>, tensor<256x1x1x1xf16, {order = #NHWC}> -> tensor<1x256x256x256xf16, {order = #NHWC}>
    // CHECK: return [[VAR1]] : tensor<1x256x256x256xf16, {order = #NHWC}>
  }

}
