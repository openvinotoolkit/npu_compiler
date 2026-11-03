//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --adjust-layouts --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @AdjustTransposedConvolutionLayout
module @AdjustTransposedConvolutionLayout {

config.PipelineOptions @Options {
        config.Option @config.EnableSEPtrsOperations : true 
}

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x16x23x30xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x16x46x60xf16>
    }

// CHECK: func.func @main([[INPUT:%.+]]: tensor<1x16x23x30xf16>) -> tensor<1x16x46x60xf16> {
func.func @main(%input: tensor<1x16x23x30xf16>) -> tensor<1x16x46x60xf16> {
    %weights = const.Declare tensor<16x16x2x2xf16> = dense<1.000000e+00> : tensor<16x16x2x2xf16>
    %output = IE.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x16x23x30xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x46x60xf16>
    return %output : tensor<1x16x46x60xf16>

    // CHECK:       [[WEIGHTS_REORDERED:%.+]] = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x2x2xf16>, [#const.Reorder<#NHWC>]

    // CHECK:       [[INPUT_REORDERED:%.+]] = IE.Reorder([[INPUT]]) {dstOrder = #NHWC} : tensor<1x16x23x30xf16> -> tensor<1x16x23x30xf16, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]] = IE.TransposedConvolution([[INPUT_REORDERED]], [[WEIGHTS_REORDERED]]) {
    // CHECK-SAME:          dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]
    // CHECK-SAME:      -> tensor<1x16x46x60xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = IE.Reorder([[CONV]]) {dstOrder = #NCHW} : tensor<1x16x46x60xf16, {order = #NHWC}> -> tensor<1x16x46x60xf16>

    // CHECK:       return [[OUTPUT]]

}
}
