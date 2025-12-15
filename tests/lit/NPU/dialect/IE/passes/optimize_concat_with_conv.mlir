//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --optimize-concat-with-conv %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK-LABEL: @OptimizeConcatWithConv
module @OptimizeConcatWithConv{

config.Resources 2 of @NCE at 1.700000e+03 MHz
net.NetworkInfo entryPoint : @main
inputsInfo : {
    DataInfo "input0" : tensor<1x128x1x1xf16>
    DataInfo "input1" : tensor<1x128x1x1xf16>
} outputsInfo : {
    DataInfo "output" : tensor<1x128x2x1xf16>
}

// CHECK: func.func @main([[INPUT0:%.+]]: tensor<1x128x1x1xf16>, [[INPUT1:%.+]]: tensor<1x128x1x1xf16>)
func.func @main(%arg0: tensor<1x128x1x1xf16>, %arg1: tensor<1x128x1x1xf16>) -> tensor<1x128x2x1xf16> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0]]}
        : tensor<1x128x1x1xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x2x1xf16>
    return %0 : tensor<1x128x2x1xf16>

    //CHECK:   [[WEIGHTS:%.+]] = const.Declare tensor<64x32x5x1xf16, {order = #NHWC}> = dense<"0x
    //CHECK-SAME:      003C0000000000000000{{([0000]{160})}}
    //CHECK-SAME:      0000000000000000003C{{([0000]{160})}}
    //CHECK-SAME:      00000000000000000000003C0000000000000000

    //CHECK:      [[RESHAPE0:%.+]] = IE.Reshape([[INPUT0]]) {shape_value = [1, 4, 1, 32]}
    //CHECK-SAME:     : tensor<1x128x1x1xf16> -> tensor<1x4x1x32xf16>
    //CHECK:      [[PERMUTECAST0:%.+]] = IE.PermuteCast([[RESHAPE0]]) {dst_order = #NHWC, mem_perm = #NCHW}
    //CHECK-SAME:     : tensor<1x4x1x32xf16> -> tensor<1x32x4x1xf16, {order = #NHWC}>

    //CHECK:      [[RESHAPE1:%.+]] = IE.Reshape([[INPUT1]]) {shape_value = [1, 4, 1, 32]}
    //CHECK-SAME:     : tensor<1x128x1x1xf16> -> tensor<1x4x1x32xf16>
    //CHECK:      [[PERMUTECAST1:%.+]] = IE.PermuteCast([[RESHAPE1]]) {dst_order = #NHWC, mem_perm = #NCHW}

    //CHECK-SAME:     : tensor<1x4x1x32xf16> -> tensor<1x32x4x1xf16, {order = #NHWC}>
    //CHECK:      [[CONCAT:%.+]] = IE.Concat([[PERMUTECAST0]], [[PERMUTECAST1]]) {per_axis = #IE.Concat<axis = 2 : i64>}
    //CHECK-SAME:     : tensor<1x32x4x1xf16, {order = #NHWC}>, tensor<1x32x4x1xf16, {order = #NHWC}> -> tensor<1x32x8x1xf16, {order = #NHWC}>

    //CHECK:      [[CONV:%.+]] = IE.Convolution([[CONCAT]], [[WEIGHTS]])
    //CHECK-SAME:     {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    //CHECK-SAME:     : tensor<1x32x8x1xf16, {order = #NHWC}>, tensor<64x32x5x1xf16, {order = #NHWC}>
    //CHECK-SAME:     -> tensor<1x64x4x1xf16, {order = #NHWC}>
    //CHECK:      [[PERMUTECAST2:%.+]] = IE.PermuteCast([[CONV]]) {dst_order = #NCHW, mem_perm = #NCHW}
    //CHECK-SAME:     : tensor<1x64x4x1xf16, {order = #NHWC}> -> tensor<1x4x1x64xf16>
    //CHECK:      [[RESHAPE2:%.+]] = IE.Reshape([[PERMUTECAST2]]) {shape_value = [1, 128, 2, 1]}
    //CHECK-SAME:     : tensor<1x4x1x64xf16> -> tensor<1x128x2x1xf16>
    //CHECK:      return [[RESHAPE2]] : tensor<1x128x2x1xf16>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeConcatWithConvAndAdd
// CHECK-SAME:     ([[INPUT0:%.+]]: tensor<1x2x272x480xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x1x272x480xf16, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<1x1x272x480xf16, {order = #NHWC}>)
func.func @OptimizeConcatWithConvAndAdd(%arg0: tensor<1x2x272x480xf16, {order = #NHWC}>, %arg1: tensor<1x1x272x480xf16, {order = #NHWC}>, %arg2: tensor<1x1x272x480xf16, {order = #NHWC}>) -> tensor<1x4x272x480xf16, {order = #NHWC}> {
    %0 = IE.Concat(%arg0, %arg1, %arg2) {static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]]} : tensor<1x2x272x480xf16, {order = #NHWC}>, tensor<1x1x272x480xf16, {order = #NHWC}>, tensor<1x1x272x480xf16, {order = #NHWC}> -> tensor<1x4x272x480xf16, {order = #NHWC}>

    return %0 : tensor<1x4x272x480xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<4x1x1x1xf16, {order = #NHWC}> =
    // CHECK-SAME{LITERAL}:    dense<[[[[0.000000e+00]]], [[[0.000000e+00]]], [[[0.000000e+00]]], [[[1.000000e+00]]]]> : tensor<4x1x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK-DAG:       [[WEIGHTS_0:%.+]] = const.Declare tensor<4x1x1x1xf16, {order = #NHWC}> =
    // CHECK-SAME{LITERAL}:    dense<[[[[0.000000e+00]]], [[[0.000000e+00]]], [[[1.000000e+00]]], [[[0.000000e+00]]]]> : tensor<4x1x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK-DAG:       [[WEIGHTS_1:%.+]] = const.Declare tensor<4x2x1x1xf16, {order = #NHWC}> =
    // CHECK-SAME{LITERAL}:    dense<[[[[1.000000e+00]], [[0.000000e+00]]], [[[0.000000e+00]], [[1.000000e+00]]], [[[0.000000e+00]], [[0.000000e+00]]], [[[0.000000e+00]], [[0.000000e+00]]]]> : tensor<4x2x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    //CHECK:   [[CONV_0:%.+]] = IE.Convolution([[INPUT0]], [[WEIGHTS_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x2x272x480xf16, {order = #NHWC}>, tensor<4x2x1x1xf16, {order = #NHWC}> -> tensor<1x4x272x480xf16, {order = #NHWC}>
    //CHECK:   [[CONV_1:%.+]] = IE.Convolution([[INPUT1]], [[WEIGHTS_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x272x480xf16, {order = #NHWC}>, tensor<4x1x1x1xf16, {order = #NHWC}> -> tensor<1x4x272x480xf16, {order = #NHWC}>
    //CHECK:   [[CONV_2:%.+]] = IE.Convolution([[INPUT2]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x272x480xf16, {order = #NHWC}>, tensor<4x1x1x1xf16, {order = #NHWC}> -> tensor<1x4x272x480xf16, {order = #NHWC}>

    //CHECK:   [[ADD_0:%.+]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x272x480xf16, {order = #NHWC}>, tensor<1x4x272x480xf16, {order = #NHWC}> -> tensor<1x4x272x480xf16, {order = #NHWC}>
    //CHECK:   [[ADD_1:%.+]] = IE.Add([[ADD_0]], [[CONV_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x272x480xf16, {order = #NHWC}>, tensor<1x4x272x480xf16, {order = #NHWC}> -> tensor<1x4x272x480xf16, {order = #NHWC}>

    //CHECK:   return [[ADD_1]] : tensor<1x4x272x480xf16, {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotOptimizeConcatWithBigChannel
// CHECK-SAME:     ([[INPUT0:%.+]]: tensor<1x6x272x480xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x6x272x480xf16, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<1x6x272x480xf16, {order = #NHWC}>)
func.func @NotOptimizeConcatWithBigChannel(%arg0: tensor<1x6x272x480xf16, {order = #NHWC}>, %arg1: tensor<1x6x272x480xf16, {order = #NHWC}>, %arg2: tensor<1x6x272x480xf16, {order = #NHWC}>) -> tensor<1x18x272x480xf16, {order = #NHWC}> {
    %0 = IE.Concat(%arg0, %arg1, %arg2) {static_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0]]} : tensor<1x6x272x480xf16, {order = #NHWC}>, tensor<1x6x272x480xf16, {order = #NHWC}>, tensor<1x6x272x480xf16, {order = #NHWC}> -> tensor<1x18x272x480xf16, {order = #NHWC}>

    return %0 : tensor<1x18x272x480xf16, {order = #NHWC}>

    //CHECK:   [[CONCAT:%.+]] = IE.Concat([[INPUT0]], [[INPUT1]], [[INPUT2]])

    //CHECK:   return [[CONCAT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotOptimizeConcatIfConvCannotShapeCasted
// CHECK-SAME:     ([[INPUT0:%.+]]: tensor<1x1x256x480xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x4x256x480xf16, {order = #NHWC}>)
func.func @NotOptimizeConcatIfConvCannotShapeCasted(%arg0: tensor<1x1x256x480xf16, {order = #NHWC}>, %arg1: tensor<1x4x256x480xf16, {order = #NHWC}>) -> tensor<1x5x256x480xf16, {order = #NHWC}> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x256x480xf16, {order = #NHWC}>, tensor<1x4x256x480xf16, {order = #NHWC}> -> tensor<1x5x256x480xf16, {order = #NHWC}>

    return %0 : tensor<1x5x256x480xf16, {order = #NHWC}>

    //CHECK:   [[CONCAT:%.+]] = IE.Concat([[INPUT0]], [[INPUT1]])

    //CHECK:   return [[CONCAT]]
}

// -----

// CHECK-LABEL: @NotOptimizeConcatIfElementTypeNotSupported
// CHECK-SAME:     ([[INPUT0:%.+]]: tensor<1x128x1x1xsi32>, [[INPUT1:%.+]]: tensor<1x128x1x1xsi32>)
func.func @NotOptimizeConcatIfElementTypeNotSupported(%arg0: tensor<1x128x1x1xsi32>, %arg1: tensor<1x128x1x1xsi32>) -> tensor<1x128x2x1xsi32> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0]]} : tensor<1x128x1x1xsi32>, tensor<1x128x1x1xsi32> -> tensor<1x128x2x1xsi32>
    return %0 : tensor<1x128x2x1xsi32>
    //CHECK:   [[CONCAT:%.+]] = IE.Concat([[INPUT0]], [[INPUT1]])
    //CHECK:   return [[CONCAT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeConcatWithConvAndAddIfAlreadyAligned
// CHECK-SAME:     ([[INPUT0:%.+]]: tensor<1x16x512x512xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x16x512x512xf16, {order = #NHWC}>)
func.func @OptimizeConcatWithConvAndAddIfAlreadyAligned(%arg0: tensor<1x16x512x512xf16, {order = #NHWC}>, %arg1: tensor<1x16x512x512xf16, {order = #NHWC}>) -> tensor<1x32x512x512xf16, {order = #NHWC}> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x512x512xf16, {order = #NHWC}>, tensor<1x16x512x512xf16, {order = #NHWC}> -> tensor<1x32x512x512xf16, {order = #NHWC}>

    return %0 : tensor<1x32x512x512xf16, {order = #NHWC}>

    //CHECK:   [[WEIGHTS:%.+]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<"0x
    //CHECK-SAME:      {{([0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000]{20})}}
    //CHECK-SAME:      {{([803F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000]{15})}}
    //CHECK-SAME:      803F"> : tensor<32x16x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    //CHECK:   [[WEIGHTS_0:%.+]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<"0x
    //CHECK-SAME:      {{([0000803F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000]{16})}}
    //CHECK-SAME:      {{([0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000]{19})}}
    //CHECK-SAME:      00000000000000000000"> : tensor<32x16x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    //CHECK:   [[CONV_0:%.+]] = IE.Convolution([[INPUT0]], [[WEIGHTS_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x512x512xf16, {order = #NHWC}>, tensor<32x16x1x1xf16, {order = #NHWC}> -> tensor<1x32x512x512xf16, {order = #NHWC}>
    //CHECK:   [[CONV_1:%.+]] = IE.Convolution([[INPUT1]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x512x512xf16, {order = #NHWC}>, tensor<32x16x1x1xf16, {order = #NHWC}> -> tensor<1x32x512x512xf16, {order = #NHWC}>

    //CHECK:   [[ADD:%.+]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x512x512xf16, {order = #NHWC}>, tensor<1x32x512x512xf16, {order = #NHWC}> -> tensor<1x32x512x512xf16, {order = #NHWC}>

    //CHECK:   return [[ADD]] : tensor<1x32x512x512xf16, {order = #NHWC}>
}
