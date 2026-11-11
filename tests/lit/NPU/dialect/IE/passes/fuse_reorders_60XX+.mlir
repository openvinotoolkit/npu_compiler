//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-reorders %s | FileCheck %s
// REQUIRES: platform-NPU6010

!qElemType = !quant.uniform<u16:f16, 1.2090807877116612E-5>
!qElemType1 = !quant.uniform<u8:f16, 3.4662022017965132E-4:128>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @FuseReorderToConvU16Input
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x512x120x16x!qElemType, {order = #NHWC}>) -> tensor<1x512x120x16xf32> {
func.func @FuseReorderToConvU16Input(%arg0: tensor<1x512x120x16x!qElemType, {order = #NHWC}>) -> tensor<1x512x120x16xf32> {
    %cst = const.Declare tensor<512x512x1x1x!qElemType1, {order = #NHWC}> = dense<1> : tensor<512x512x1x1xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<1x512x1x1xf16> = dense<1.000000e+00> : tensor<1x512x1x1xf32>, [#const.CastElemType<f16>]
    %Conv = IE.Convolution(%arg0, %cst, %cst_0) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x512x120x16x!qElemType, {order = #NHWC}>, tensor<512x512x1x1x!qElemType1, {order = #NHWC}>, tensor<1x512x1x1xf16> -> tensor<1x512x120x16xf32, {order = #NHWC}>
    %Reorder = IE.Reorder(%Conv) {dstOrder = #NCHW} : tensor<1x512x120x16xf32, {order = #NHWC}> -> tensor<1x512x120x16xf32>
    return %Reorder : tensor<1x512x120x16xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<512x512x1x1x!qElemType1, {order = #NHWC}>
    // CHECK-SAME:      = dense<1> : tensor<512x512x1x1xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x512x1x1xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<1x512x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[CST]], [[CST_0]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x512x120x16x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:  tensor<512x512x1x1x!qElemType1, {order = #NHWC}>,
    // CHECK-SAME:  tensor<1x512x1x1xf16> -> tensor<1x512x120x16xf32>
    // CHECK-NOT:       [[REORDER:%.+]] =  IE.Reorder([[CONV]]) {
    // CHECK:       return [[CONV]] : tensor<1x512x120x16xf32>
}

// -----

!qElemType = !quant.uniform<u16:f16, 1.2090807877116612E-5>
!qElemType1 = !quant.uniform<u8:f16, 3.4662022017965132E-4:128>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotFusedReorderToConvU16
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x512x120x16x!qElemType, {order = #NHWC}>) -> tensor<1x512x120x16xf32> {
func.func @NotFusedReorderToConvU16(%arg0: tensor<1x512x120x16x!qElemType, {order = #NHWC}>) -> tensor<1x512x120x16xf32> {
    %cst = const.Declare tensor<512x512x1x1x!qElemType1, {order = #NHWC}> = dense<1> : tensor<512x512x1x1xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<1x512x1x1xf16> = dense<1.000000e+00> : tensor<1x512x1x1xf32>, [#const.CastElemType<f16>]
    %Conv = IE.Convolution(%arg0, %cst, %cst_0) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x512x120x16x!qElemType, {order = #NHWC}>, tensor<512x512x1x1x!qElemType1, {order = #NHWC}>, tensor<1x512x1x1xf16> -> tensor<1x512x120x16xf32, {order = #NHWC}>
    %Reorder = IE.Reorder(%Conv) {dstOrder = #NCHW} : tensor<1x512x120x16xf32, {order = #NHWC}> -> tensor<1x512x120x16xf32>
    %MaxPool = IE.MaxPool(%Reorder) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x512x120x16xf32> -> tensor<1x512x120x16xf32>
    return %MaxPool : tensor<1x512x120x16xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<512x512x1x1x!qElemType1, {order = #NHWC}>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x512x1x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[CST]], [[CST_0]]) {
    // CHECK-SAME:  } : tensor<1x512x120x16x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:  tensor<512x512x1x1x!qElemType1, {order = #NHWC}>,
    // CHECK-SAME:  tensor<1x512x1x1xf16> -> tensor<1x512x120x16xf32, {order = #NHWC}>
    // CHECK:       [[REORDER:%.+]] = IE.Reorder([[CONV]]) {dstOrder = #NCHW}
    // CHECK-SAME:  : tensor<1x512x120x16xf32, {order = #NHWC}> -> tensor<1x512x120x16xf32>
    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[REORDER]])
    // CHECK:       return [[MAXPOOL]] : tensor<1x512x120x16xf32>
}
