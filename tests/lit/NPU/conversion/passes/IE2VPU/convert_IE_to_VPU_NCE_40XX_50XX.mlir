//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --convert-IE-to-VPU-NCE %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertCompressConvolutionWithDynamicShapes
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
func.func @ConvertCompressConvolutionWithDynamicShapes(
    %arg0: tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
) -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 270, 480]> : tensor<4xsi64>, order = #NHWC}> {

    %weights = const.Declare tensor<32x4x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x4x3x3xf16>, [#const.Reorder<#NHWC>]
    %conv = IE.Convolution(%arg0, %weights) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [0, 0],
        strides = [2, 2]
    } : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 540, 960]> : tensor<4xsi64>, order = #NHWC}>,
        tensor<32x4x3x3xf16, {order = #NHWC}>
     -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 270, 480]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK-DAG: [[CST_WEIGHTS:%.+]] = const.Declare tensor<32x1x1x48xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x4x3x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[32, 1, 1, 36]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 12]>]
    // CHECK-DAG: [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<"{{.+}}"> : tensor<32x1x1x4xsi32>
    // CHECK: [[COMPRESS_CONV:%.+]] = VPU.NCE.CompressConvolution([[ARG0]], [[CST_WEIGHTS]], [[CST_WEIGHTS_TABLE]]) rawFilterShape [32, 4, 3, 3] {
    // CHECK-SAME: cm_sp_pattern = 15 : i64
    // CHECK-SAME: pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>
    // CHECK-SAME: strides = [2, 2]
    // CHECK-SAME: -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 270, 480]> : tensor<4xsi64>, order = #NHWC}>

    return %conv : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 270, 480]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK: return [[COMPRESS_CONV]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertCompressConvolutionWithSmallDynamicBounds
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 10, 10]> : tensor<4xsi64>, order = #NHWC}>
func.func @ConvertCompressConvolutionWithSmallDynamicBounds(
    %arg0: tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 10, 10]> : tensor<4xsi64>, order = #NHWC}>
) -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 5, 5]> : tensor<4xsi64>, order = #NHWC}> {

    %weights = const.Declare tensor<32x4x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x4x3x3xf16>, [#const.Reorder<#NHWC>]
    %conv = IE.Convolution(%arg0, %weights) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [0, 0],
        strides = [2, 2]
    } : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 10, 10]> : tensor<4xsi64>, order = #NHWC}>,
        tensor<32x4x3x3xf16, {order = #NHWC}>
     -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 5, 5]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK-DAG: [[CST_WEIGHTS:%.+]] = const.Declare tensor<32x1x1x48xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x4x3x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[32, 1, 1, 36]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 12]>]
    // CHECK-DAG: [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<"{{.+}}"> : tensor<32x1x1x4xsi32>
    // CHECK: [[COMPRESS_CONV:%.+]] = VPU.NCE.CompressConvolution([[ARG0]], [[CST_WEIGHTS]], [[CST_WEIGHTS_TABLE]]) rawFilterShape [32, 4, 3, 3] {
    // CHECK-SAME: cm_sp_pattern = 15 : i64
    // CHECK-SAME: pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>
    // CHECK-SAME: strides = [2, 2]
    // CHECK-SAME: -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 5, 5]> : tensor<4xsi64>, order = #NHWC}>

    return %conv : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 5, 5]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK: return [[COMPRESS_CONV]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertCompressConvolutionWithStaticShapes
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x4x540x960xf16, {order = #NHWC}>
func.func @ConvertCompressConvolutionWithStaticShapes(
    %arg0: tensor<1x4x540x960xf16, {order = #NHWC}>
) -> tensor<1x32x270x480xf16, {order = #NHWC}> {

    %weights = const.Declare tensor<32x4x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x4x3x3xf16>, [#const.Reorder<#NHWC>]
    %conv = IE.Convolution(%arg0, %weights) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [0, 0],
        strides = [2, 2]
    } : tensor<1x4x540x960xf16, {order = #NHWC}>,
        tensor<32x4x3x3xf16, {order = #NHWC}>
     -> tensor<1x32x270x480xf16, {order = #NHWC}>

    // CHECK-DAG: [[CST_WEIGHTS:%.+]] = const.Declare tensor<32x1x1x48xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x4x3x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[32, 1, 1, 36]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 12]>]
    // CHECK-DAG: [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<"{{.+}}"> : tensor<32x1x1x4xsi32>
    // CHECK: [[COMPRESS_CONV:%.+]] = VPU.NCE.CompressConvolution([[ARG0]], [[CST_WEIGHTS]], [[CST_WEIGHTS_TABLE]]) rawFilterShape [32, 4, 3, 3] {
    // CHECK-SAME: cm_sp_pattern = 15 : i64
    // CHECK-SAME: pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>
    // CHECK-SAME: strides = [2, 2]
    // CHECK-SAME: -> tensor<1x32x270x480xf16, {order = #NHWC}>

    return %conv : tensor<1x32x270x480xf16, {order = #NHWC}>

    // CHECK: return [[COMPRESS_CONV]]
}
