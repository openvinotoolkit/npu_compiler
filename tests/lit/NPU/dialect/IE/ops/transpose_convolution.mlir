//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
 
//CHECK:  func.func @DepthToSpaceDynamicHeightAndWidth([[ARG0:%.*]]: tensor<1x144x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 144, 3, 3]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 9, 9]> : tensor<4xsi64>, order = #NHWC}> {
func.func @DepthToSpaceDynamicHeightAndWidth(%arg0: tensor<1x144x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 144, 3, 3]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 9, 9]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> {
    %cst = const.Declare tensor<16x144x3x3xf16> = dense<1> : tensor<16x144x3x3xui8, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, [#const.CastElemType<f16>, #const.Reorder<#NCHW>]
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    %0 = IE.FakeQuantize(%cst, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 2 : i64} : tensor<16x144x3x3xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<16x144x3x3xf16>
    %1 = IE.TransposedConvolution(%arg0, %0) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [3, 3]} : tensor<1x144x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 144, 3, 3]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<16x144x3x3xf16> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 9, 9]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    return %1 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 9, 9]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[VAL1:%.*]] = IE.TransposedConvolution([[ARG0]], %0) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [3, 3]} : tensor<1x144x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 144, 3, 3]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x144x3x3xf16> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 9, 9]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: return [[VAL1]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 9, 9]> : tensor<4xsi64>, order = #NHWC}>
}


