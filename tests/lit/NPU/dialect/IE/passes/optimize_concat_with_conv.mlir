//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --optimize-concat-with-conv %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK-LABEL: @OptimizeConcatWithConv
module @OptimizeConcatWithConv{

IE.TileResource 2 of @NCE at 1.700000e+03 MHz
IE.CNNNetwork entryPoint : @main
inputsInfo : {
    DataInfo "input0" : tensor<1x128x1x1xf16>
    DataInfo "input1" : tensor<1x128x1x1xf16>
} outputsInfo : {
    DataInfo "output" : tensor<1x128x2x1xf16>
}

// CHECK: func.func @main([[INPUT0:%.+]]: tensor<1x128x1x1xf16>, [[INPUT1:%.+]]: tensor<1x128x1x1xf16>)
func.func @main(%arg0: tensor<1x128x1x1xf16>, %arg1: tensor<1x128x1x1xf16>) -> tensor<1x128x2x1xf16> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0]]} : tensor<1x128x1x1xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x2x1xf16>
    return %0 : tensor<1x128x2x1xf16>
    //CHECK:   [[WEIGHTS:%.+]] = const.Declare tensor<64x32x5x1xf16, {order = #NHWC}> = dense<"0x
    //CHECK-SAME:      003C0000000000000000{{([0000]{160})}}
    //CHECK-SAME:      0000000000000000003C{{([0000]{160})}}
    //CHECK-SAME:      00000000000000000000003C0000000000000000
    //CHECK:   [[RESHAPE0:%.+]] = IE.Reshape([[INPUT0]]) {shape_value = [1, 32, 4, 1]} : tensor<1x128x1x1xf16> -> tensor<1x32x4x1xf16>
    //CHECK:   [[LAYOUTCAST0:%.+]] = IE.LayoutCast([[RESHAPE0]]) {dst_order = #NHWC} : tensor<1x32x4x1xf16> -> tensor<1x32x4x1xf16, {order = #NHWC}>
    //CHECK:   [[RESHAPE1:%.+]] = IE.Reshape([[INPUT1]]) {shape_value = [1, 32, 4, 1]} : tensor<1x128x1x1xf16> -> tensor<1x32x4x1xf16>
    //CHECK:   [[LAYOUTCAST1:%.+]] = IE.LayoutCast([[RESHAPE1]]) {dst_order = #NHWC} : tensor<1x32x4x1xf16> -> tensor<1x32x4x1xf16, {order = #NHWC}>
    //CHECK:   [[CONCAT:%.+]] = IE.Concat([[LAYOUTCAST0]], [[LAYOUTCAST1]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x32x4x1xf16, {order = #NHWC}>, tensor<1x32x4x1xf16, {order = #NHWC}> -> tensor<1x32x8x1xf16, {order = #NHWC}>
    //CHECK:   [[CONV:%.+]] = IE.Convolution([[CONCAT]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x8x1xf16, {order = #NHWC}>, tensor<64x32x5x1xf16, {order = #NHWC}> -> tensor<1x64x4x1xf16, {order = #NHWC}>
    //CHECK:   [[LAYOUTCAST2:%.+]] = IE.LayoutCast([[CONV]]) {dst_order = #NCHW} : tensor<1x64x4x1xf16, {order = #NHWC}> -> tensor<1x64x4x1xf16>
    //CHECK:   [[RESHAPE2:%.+]] = IE.Reshape([[LAYOUTCAST2]]) {shape_value = [1, 128, 2, 1]} : tensor<1x64x4x1xf16> -> tensor<1x128x2x1xf16>
    //CHECK:   return [[RESHAPE2]] : tensor<1x128x2x1xf16>
}
}
