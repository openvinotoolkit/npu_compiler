//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --handle-large-kernels %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @HandleLargeKernelsConv
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x1x32000xf16>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

func.func @HandleLargeKernelsConv(%arg0 : tensor<1x1x1x32000xf16>) -> tensor<1x64x1x2000xf16> {
    %cst = const.Declare tensor<64x1x1x33xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 16], pads_end = [0, 16], strides = [1, 16]} : tensor<1x1x1x32000xf16>, tensor<64x1x1x33xf16> -> tensor<1x64x1x2000xf16>

    return %conv : tensor<1x64x1x2000xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<64x1x1x33xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<64x1x1x15xf16> = dense<0.000000e+00> : tensor<64x1x1x15xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CST]], [[CST_0]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<64x1x1x33xf16>, tensor<64x1x1x15xf16> -> tensor<64x1x1x48xf16>
    // CHECK: [[RESHAPE_INPUT:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 2000, 1, 16]} : tensor<1x1x1x32000xf16> -> tensor<1x2000x1x16xf16>
    // CHECK: [[TRANSPOSE_INPUT:%.+]] = IE.Transpose([[RESHAPE_INPUT]]) {order_value = #NWHC} : tensor<1x2000x1x16xf16> -> tensor<1x16x1x2000xf16>
    // CHECK: [[RESHAPE_FILTER:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [64, 3, 1, 16]} : tensor<64x1x1x48xf16> -> tensor<64x3x1x16xf16>
    // CHECK: [[TRANSPOSE_FILTER:%.+]] = IE.Transpose([[RESHAPE_FILTER]]) {order_value = #NWHC} : tensor<64x3x1x16xf16> -> tensor<64x16x1x3xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[TRANSPOSE_INPUT]], [[TRANSPOSE_FILTER]]) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x16x1x2000xf16>, tensor<64x16x1x3xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: return [[CONV]] : tensor<1x64x1x2000xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsConvWithPostOp
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x1x32000xf16>
func.func @HandleLargeKernelsConvWithPostOp(%arg0 : tensor<1x1x1x32000xf16>) -> tensor<1x64x1x2000xf16> {
    %cst = const.Declare tensor<64x1x1x33xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 16], pads_end = [0, 16], post_op = #IE.Relu<>, strides = [1, 16]} : tensor<1x1x1x32000xf16>, tensor<64x1x1x33xf16> -> tensor<1x64x1x2000xf16>

    return %conv : tensor<1x64x1x2000xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<64x1x1x33xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<64x1x1x15xf16> = dense<0.000000e+00> : tensor<64x1x1x15xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CST]], [[CST_0]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<64x1x1x33xf16>, tensor<64x1x1x15xf16> -> tensor<64x1x1x48xf16>
    // CHECK: [[RESHAPE_INPUT:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 2000, 1, 16]} : tensor<1x1x1x32000xf16> -> tensor<1x2000x1x16xf16>
    // CHECK: [[TRANSPOSE_INPUT:%.+]] = IE.Transpose([[RESHAPE_INPUT]]) {order_value = #NWHC} : tensor<1x2000x1x16xf16> -> tensor<1x16x1x2000xf16>
    // CHECK: [[RESHAPE_FILTER:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [64, 3, 1, 16]} : tensor<64x1x1x48xf16> -> tensor<64x3x1x16xf16>
    // CHECK: [[TRANSPOSE_FILTER:%.+]] = IE.Transpose([[RESHAPE_FILTER]]) {order_value = #NWHC} : tensor<64x3x1x16xf16> -> tensor<64x16x1x3xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[TRANSPOSE_INPUT]], [[TRANSPOSE_FILTER]]) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], post_op = #IE.Relu<>, strides = [1, 1]} : tensor<1x16x1x2000xf16>, tensor<64x16x1x3xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: return [[CONV]] : tensor<1x64x1x2000xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsConvWithBias
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x1x32000xf16>
func.func @HandleLargeKernelsConvWithBias(%arg0 : tensor<1x1x1x32000xf16>) -> tensor<1x64x1x2000xf16> {
    %cst = const.Declare tensor<64x1x1x33xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>
    %bias = const.Declare tensor<1x64x1x1xf16> = dense<1.000000e+00> : tensor<1x64x1x1xf16>

    %conv = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 16], pads_end = [0, 16], strides = [1, 16]} : tensor<1x1x1x32000xf16>, tensor<64x1x1x33xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x1x2000xf16>

    return %conv : tensor<1x64x1x2000xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<64x1x1x33xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x64x1x1xf16> = dense<1.000000e+00> : tensor<1x64x1x1xf16>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<64x1x1x15xf16> = dense<0.000000e+00> : tensor<64x1x1x15xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CST]], [[CST_1]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<64x1x1x33xf16>, tensor<64x1x1x15xf16> -> tensor<64x1x1x48xf16>
    // CHECK: [[RESHAPE_INPUT:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 2000, 1, 16]} : tensor<1x1x1x32000xf16> -> tensor<1x2000x1x16xf16>
    // CHECK: [[TRANSPOSE_INPUT:%.+]] = IE.Transpose([[RESHAPE_INPUT]]) {order_value = #NWHC} : tensor<1x2000x1x16xf16> -> tensor<1x16x1x2000xf16>
    // CHECK: [[RESHAPE_FILTER:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [64, 3, 1, 16]} : tensor<64x1x1x48xf16> -> tensor<64x3x1x16xf16>
    // CHECK: [[TRANSPOSE_FILTER:%.+]] = IE.Transpose([[RESHAPE_FILTER]]) {order_value = #NWHC} : tensor<64x3x1x16xf16> -> tensor<64x16x1x3xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[TRANSPOSE_INPUT]], [[TRANSPOSE_FILTER]], [[CST_0]]) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x16x1x2000xf16>, tensor<64x16x1x3xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: return [[CONV]] : tensor<1x64x1x2000xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsConv2DimsSplit
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x32000x32000xf16>
func.func @HandleLargeKernelsConv2DimsSplit(%arg0 : tensor<1x1x32000x32000xf16>) -> tensor<1x64x2001x2001xf16> {
    %cst = const.Declare tensor<64x1x22x22xf16> = dense<1.000000e+00> : tensor<64x1x22x22xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [16, 16], pads_end = [16, 16], strides = [16, 16]} : tensor<1x1x32000x32000xf16>, tensor<64x1x22x22xf16> -> tensor<1x64x2001x2001xf16>

    return %conv : tensor<1x64x2001x2001xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<64x1x11x11xf16> = dense<1.000000e+00> : tensor<64x1x22x22xf16>, [#const.SubView<[0, 0, 0, 0], [64, 1, 11, 11]>]
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<64x1x11x11xf16> = dense<1.000000e+00> : tensor<64x1x22x22xf16>, [#const.SubView<[0, 0, 0, 11], [64, 1, 11, 11]>]
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<64x1x11x11xf16> = dense<1.000000e+00> : tensor<64x1x22x22xf16>, [#const.SubView<[0, 0, 11, 0], [64, 1, 11, 11]>]
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<64x1x11x11xf16> = dense<1.000000e+00> : tensor<64x1x22x22xf16>, [#const.SubView<[0, 0, 11, 11], [64, 1, 11, 11]>]
    // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<1x1x32000x16xf16> = dense<0.000000e+00> : tensor<1x1x32000x16xf16>
    // CHECK-DAG: [[CST_4:%.+]] = const.Declare tensor<1x1x16x32032xf16> = dense<0.000000e+00> : tensor<1x1x16x32032xf16>
    // CHECK: [[CONCAT0:%.+]] = IE.Concat([[CST_3]], [[INPUT]], [[CST_3]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x1x32000x16xf16>, tensor<1x1x32000x32000xf16>, tensor<1x1x32000x16xf16> -> tensor<1x1x32000x32032xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CST_4]], %0, [[CST_4]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x1x16x32032xf16>, tensor<1x1x32000x32032xf16>, tensor<1x1x16x32032xf16> -> tensor<1x1x32032x32032xf16>

    // CHECK: [[SLICEACT0:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 0] [1, 1, 32011, 32011] : tensor<1x1x32032x32032xf16> to tensor<1x1x32011x32011xf16>
    // CHECK: [[CONV0:%.+]] = IE.Convolution([[SLICEACT0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x1x32011x32011xf16>, tensor<64x1x11x11xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[SLICEACT1:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 11] [1, 1, 32011, 32011] : tensor<1x1x32032x32032xf16> to tensor<1x1x32011x32011xf16>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[SLICEACT1]], [[CST_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x1x32011x32011xf16>, tensor<64x1x11x11xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[Add0:%.+]] = IE.Add([[CONV0]], [[CONV1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x2001x2001xf16>, tensor<1x64x2001x2001xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[SLICEACT2:%.+]] = IE.Slice [[CONCAT]] [0, 0, 11, 0] [1, 1, 32011, 32011] : tensor<1x1x32032x32032xf16> to tensor<1x1x32011x32011xf16>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[SLICEACT2]], [[CST_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x1x32011x32011xf16>, tensor<64x1x11x11xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[Add1:%.+]] = IE.Add([[Add0]], [[CONV2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x2001x2001xf16>, tensor<1x64x2001x2001xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[SLICEACT3:%.+]] = IE.Slice [[CONCAT]] [0, 0, 11, 11] [1, 1, 32011, 32011] : tensor<1x1x32032x32032xf16> to tensor<1x1x32011x32011xf16>
    // CHECK: [[CONV3:%.+]] = IE.Convolution([[SLICEACT3]], [[CST_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x1x32011x32011xf16>, tensor<64x1x11x11xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[Add2:%.+]] = IE.Add([[Add1]], [[CONV3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x2001x2001xf16>, tensor<1x64x2001x2001xf16> -> tensor<1x64x2001x2001xf16>

    // CHECK: return [[Add2]] : tensor<1x64x2001x2001xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsConv2DimsUnevenSplit
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x32000x32000xf16>
func.func @HandleLargeKernelsConv2DimsUnevenSplit(%arg0 : tensor<1x1x32000x32000xf16>) -> tensor<1x64x2001x2001xf16> {
    %cst = const.Declare tensor<64x1x18x18xf16> = dense<1.000000e+00> : tensor<64x1x18x18xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [16, 16], pads_end = [16, 16], strides = [16, 16]} : tensor<1x1x32000x32000xf16>, tensor<64x1x18x18xf16> -> tensor<1x64x2001x2001xf16>

    return %conv : tensor<1x64x2001x2001xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<64x1x11x11xf16> = dense<1.000000e+00> : tensor<64x1x18x18xf16>, [#const.SubView<[0, 0, 0, 0], [64, 1, 11, 11]>]
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<64x1x11x7xf16> = dense<1.000000e+00> : tensor<64x1x18x18xf16>, [#const.SubView<[0, 0, 0, 11], [64, 1, 11, 7]>]
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<64x1x7x11xf16> = dense<1.000000e+00> : tensor<64x1x18x18xf16>, [#const.SubView<[0, 0, 11, 0], [64, 1, 7, 11]>]
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<64x1x7x7xf16> = dense<1.000000e+00> : tensor<64x1x18x18xf16>, [#const.SubView<[0, 0, 11, 11], [64, 1, 7, 7]>]
    // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<1x1x32000x16xf16> = dense<0.000000e+00> : tensor<1x1x32000x16xf16>
    // CHECK-DAG: [[CST_4:%.+]] = const.Declare tensor<1x1x16x32032xf16> = dense<0.000000e+00> : tensor<1x1x16x32032xf16>
    // CHECK: [[CONCAT0:%.+]] = IE.Concat([[CST_3]], [[INPUT]], [[CST_3]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x1x32000x16xf16>, tensor<1x1x32000x32000xf16>, tensor<1x1x32000x16xf16> -> tensor<1x1x32000x32032xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CST_4]], [[CONCAT0]], [[CST_4]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x1x16x32032xf16>, tensor<1x1x32000x32032xf16>, tensor<1x1x16x32032xf16> -> tensor<1x1x32032x32032xf16>
    // CHECK: [[SLICEACT0:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 0] [1, 1, 32011, 32011] : tensor<1x1x32032x32032xf16> to tensor<1x1x32011x32011xf16>
    // CHECK: [[CONV0:%.+]] = IE.Convolution([[SLICEACT0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x1x32011x32011xf16>, tensor<64x1x11x11xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[SLICEACT1:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 11] [1, 1, 32011, 32007] : tensor<1x1x32032x32032xf16> to tensor<1x1x32011x32007xf16>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[SLICEACT1]], [[CST_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x1x32011x32007xf16>, tensor<64x1x11x7xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[Add0:%.+]] = IE.Add([[CONV0]], [[CONV1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x2001x2001xf16>, tensor<1x64x2001x2001xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[SLICEACT2:%.+]] = IE.Slice [[CONCAT]] [0, 0, 11, 0] [1, 1, 32007, 32011] : tensor<1x1x32032x32032xf16> to tensor<1x1x32007x32011xf16>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[SLICEACT2]], [[CST_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x1x32007x32011xf16>, tensor<64x1x7x11xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[Add1:%.+]] = IE.Add([[Add0]], [[CONV2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x2001x2001xf16>, tensor<1x64x2001x2001xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[SLICEACT3:%.+]] = IE.Slice [[CONCAT]] [0, 0, 11, 11] [1, 1, 32007, 32007] : tensor<1x1x32032x32032xf16> to tensor<1x1x32007x32007xf16>
    // CHECK: [[CONV3:%.+]] = IE.Convolution([[SLICEACT3]], [[CST_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x1x32007x32007xf16>, tensor<64x1x7x7xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[Add2:%.+]] = IE.Add([[Add1]], [[CONV3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x2001x2001xf16>, tensor<1x64x2001x2001xf16> -> tensor<1x64x2001x2001xf16>

    // CHECK: return [[Add2]] : tensor<1x64x2001x2001xf16>
}

// -----

// CHECK-LABEL: @HandleLargePrimeKernelsConvWithOneDimOnH
// CHECK-SAME: [[INPUT:%.+]]: tensor<32x1x80000x1xf16>
func.func @HandleLargePrimeKernelsConvWithOneDimOnH(%arg0 : tensor<32x1x80000x1xf16>) -> tensor<32x80x7975x1xf16> {
    %cst = const.Declare tensor<80x1x251x1xf16> = dense<1.000000e+00> : tensor<80x1x251x1xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [10, 1]} : tensor<32x1x80000x1xf16>, tensor<80x1x251x1xf16> -> tensor<32x80x7975x1xf16>
    return %conv : tensor<32x80x7975x1xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<80x10x3x1xf16> = dense<1.000000e+00> : tensor<80x1x251x1xf16>, [#const.SubView<[0, 0, 0, 0], [80, 1, 250, 1]>, #const.Reshape<[80, 25, 10, 1]>, #const.SubView<[0, 22, 0, 0], [80, 3, 10, 1]>, #const.Transpose<#NHCW>]
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<80x10x11x1xf16> = dense<1.000000e+00> : tensor<80x1x251x1xf16>, [#const.SubView<[0, 0, 0, 0], [80, 1, 250, 1]>, #const.Reshape<[80, 25, 10, 1]>, #const.SubView<[0, 11, 0, 0], [80, 11, 10, 1]>, #const.Transpose<#NHCW>]
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<80x10x11x1xf16> = dense<1.000000e+00> : tensor<80x1x251x1xf16>, [#const.SubView<[0, 0, 0, 0], [80, 1, 250, 1]>, #const.Reshape<[80, 25, 10, 1]>, #const.SubView<[0, 0, 0, 0], [80, 11, 10, 1]>, #const.Transpose<#NHCW>]
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<80x1x1x1xf16> = dense<1.000000e+00> : tensor<80x1x251x1xf16>, [#const.SubView<[0, 0, 250, 0], [80, 1, 1, 1]>]

    // CHECK: [[SLICEACT0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [32, 1, 79990, 1] : tensor<32x1x80000x1xf16> to tensor<32x1x79990x1xf16>
    // CHECK: [[RESHAPE:%.+]] = IE.Reshape([[SLICEACT0]]) {shape_value = [32, 7999, 10, 1]} : tensor<32x1x79990x1xf16> -> tensor<32x7999x10x1xf16>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]]) {order_value = #NHCW} : tensor<32x7999x10x1xf16> -> tensor<32x10x7999x1xf16>
    // CHECK: [[SLICEACT1:%.+]] = IE.Slice [[TRANSPOSE]] [0, 0, 0, 0] [32, 10, 7985, 1] : tensor<32x10x7999x1xf16> to tensor<32x10x7985x1xf16>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[SLICEACT1]], [[CST_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<32x10x7985x1xf16>, tensor<80x10x11x1xf16> -> tensor<32x80x7975x1xf16>
    // CHECK: [[SLICEACT2:%.+]] = IE.Slice [[TRANSPOSE]] [0, 0, 11, 0] [32, 10, 7985, 1] : tensor<32x10x7999x1xf16> to tensor<32x10x7985x1xf16>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[SLICEACT2]], [[CST_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<32x10x7985x1xf16>, tensor<80x10x11x1xf16> -> tensor<32x80x7975x1xf16>
    // CHECK: [[ADD0:%.+]] = IE.Add([[CONV1]], [[CONV2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<32x80x7975x1xf16>, tensor<32x80x7975x1xf16> -> tensor<32x80x7975x1xf16>
    // CHECK: [[SLICEACT3:%.+]] = IE.Slice [[TRANSPOSE]] [0, 0, 22, 0] [32, 10, 7977, 1] : tensor<32x10x7999x1xf16> to tensor<32x10x7977x1xf16>
    // CHECK: [[CONV3:%.+]] = IE.Convolution([[SLICEACT3]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<32x10x7977x1xf16>, tensor<80x10x3x1xf16> -> tensor<32x80x7975x1xf16>
    // CHECK: [[ADD1:%.+]] = IE.Add([[ADD0]], [[CONV3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<32x80x7975x1xf16>, tensor<32x80x7975x1xf16> -> tensor<32x80x7975x1xf16>
    // CHECK: [[SLICEACT4:%.+]] = IE.Slice [[INPUT]] [0, 0, 250, 0] [32, 1, 79741, 1] : tensor<32x1x80000x1xf16> to tensor<32x1x79741x1xf16>
    // CHECK: [[CONV4:%.+]] = IE.Convolution([[SLICEACT4]], [[CST_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [10, 1]} : tensor<32x1x79741x1xf16>, tensor<80x1x1x1xf16> -> tensor<32x80x7975x1xf16>
    // CHECK: [[ADD2:%.+]] = IE.Add([[ADD1]], [[CONV4]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<32x80x7975x1xf16>, tensor<32x80x7975x1xf16> -> tensor<32x80x7975x1xf16>

    // CHECK: return [[ADD2]] : tensor<32x80x7975x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargePrimeKernelsConvWithOneDimOnW
// CHECK-SAME: [[INPUT:%.+]]: tensor<32x1x1x80000xf16>
func.func @HandleLargePrimeKernelsConvWithOneDimOnW(%arg0 : tensor<32x1x1x80000xf16>) -> tensor<32x80x1x7975xf16> {
    %cst = const.Declare tensor<80x1x1x251xf16> = dense<1.000000e+00> : tensor<80x1x1x251xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 10]} : tensor<32x1x1x80000xf16>, tensor<80x1x1x251xf16> -> tensor<32x80x1x7975xf16>
    return %conv : tensor<32x80x1x7975xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<80x10x1x3xf16> = dense<1.000000e+00> : tensor<80x1x1x251xf16>, [#const.SubView<[0, 0, 0, 0], [80, 1, 1, 250]>, #const.Reshape<[80, 25, 1, 10]>, #const.SubView<[0, 22, 0, 0], [80, 3, 1, 10]>, #const.Transpose<#NWHC>]
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<80x10x1x11xf16> = dense<1.000000e+00> : tensor<80x1x1x251xf16>, [#const.SubView<[0, 0, 0, 0], [80, 1, 1, 250]>, #const.Reshape<[80, 25, 1, 10]>, #const.SubView<[0, 11, 0, 0], [80, 11, 1, 10]>, #const.Transpose<#NWHC>]
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<80x10x1x11xf16> = dense<1.000000e+00> : tensor<80x1x1x251xf16>, [#const.SubView<[0, 0, 0, 0], [80, 1, 1, 250]>, #const.Reshape<[80, 25, 1, 10]>, #const.SubView<[0, 0, 0, 0], [80, 11, 1, 10]>, #const.Transpose<#NWHC>]
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<80x1x1x1xf16> = dense<1.000000e+00> : tensor<80x1x1x251xf16>, [#const.SubView<[0, 0, 0, 250], [80, 1, 1, 1]>]

    // CHECK: [[SLICEACT0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [32, 1, 1, 79990] : tensor<32x1x1x80000xf16> to tensor<32x1x1x79990xf16>
    // CHECK: [[RESHAPE:%.+]] = IE.Reshape([[SLICEACT0]]) {shape_value = [32, 7999, 1, 10]} : tensor<32x1x1x79990xf16> -> tensor<32x7999x1x10xf16>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]]) {order_value = #NWHC} : tensor<32x7999x1x10xf16> -> tensor<32x10x1x7999xf16>
    // CHECK: [[SLICEACT1:%.+]] = IE.Slice [[TRANSPOSE]] [0, 0, 0, 0] [32, 10, 1, 7985] : tensor<32x10x1x7999xf16> to tensor<32x10x1x7985xf16>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[SLICEACT1]], [[CST_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<32x10x1x7985xf16>, tensor<80x10x1x11xf16> -> tensor<32x80x1x7975xf16>
    // CHECK: [[SLICEACT2:%.+]] = IE.Slice [[TRANSPOSE]] [0, 0, 0, 11] [32, 10, 1, 7985] : tensor<32x10x1x7999xf16> to tensor<32x10x1x7985xf16>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[SLICEACT2]], [[CST_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<32x10x1x7985xf16>, tensor<80x10x1x11xf16> -> tensor<32x80x1x7975xf16>
    // CHECK: [[ADD0:%.+]] = IE.Add([[CONV1]], [[CONV2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<32x80x1x7975xf16>, tensor<32x80x1x7975xf16> -> tensor<32x80x1x7975xf16>
    // CHECK: [[SLICEACT3:%.+]] = IE.Slice [[TRANSPOSE]] [0, 0, 0, 22] [32, 10, 1, 7977] : tensor<32x10x1x7999xf16> to tensor<32x10x1x7977xf16>
    // CHECK: [[CONV3:%.+]] = IE.Convolution([[SLICEACT3]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<32x10x1x7977xf16>, tensor<80x10x1x3xf16> -> tensor<32x80x1x7975xf16>
    // CHECK: [[ADD1:%.+]] = IE.Add([[ADD0]], [[CONV3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<32x80x1x7975xf16>, tensor<32x80x1x7975xf16> -> tensor<32x80x1x7975xf16>
    // CHECK: [[SLICEACT4:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 250] [32, 1, 1, 79741] : tensor<32x1x1x80000xf16> to tensor<32x1x1x79741xf16>
    // CHECK: [[CONV4:%.+]] = IE.Convolution([[SLICEACT4]], [[CST_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 10]} : tensor<32x1x1x79741xf16>, tensor<80x1x1x1xf16> -> tensor<32x80x1x7975xf16>
    // CHECK: [[ADD2:%.+]] = IE.Add([[ADD1]], [[CONV4]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<32x80x1x7975xf16>, tensor<32x80x1x7975xf16> -> tensor<32x80x1x7975xf16>

    // CHECK: return [[ADD2]] : tensor<32x80x1x7975xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelConvWithGCDWhenOneDimOnW
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x1x80000xf16>
func.func @HandleLargeKernelConvWithGCDWhenOneDimOnW(%arg0 : tensor<1x1x1x80000xf16>) -> tensor<1x257x1x497xf16> {
    %cst = const.Declare tensor<257x1x1x512xf16> = dense<1.000000e+00> : tensor<257x1x1x512xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 160]} : tensor<1x1x1x80000xf16>, tensor<257x1x1x512xf16> -> tensor<1x257x1x497xf16>
    return %conv : tensor<1x257x1x497xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<257x1x1x512xf16> = dense<1.000000e+00> : tensor<257x1x1x512xf16>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<257x1x1x128xf16> = dense<0.000000e+00> : tensor<257x1x1x128xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CST]], [[CST_0]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<257x1x1x512xf16>, tensor<257x1x1x128xf16> -> tensor<257x1x1x640xf16>
    // CHECK: [[RESHAPE_INPUT:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 1000, 1, 80]} : tensor<1x1x1x80000xf16> -> tensor<1x1000x1x80xf16>
    // CHECK: [[TRANSPOSE_INPUT:%.+]] = IE.Transpose([[RESHAPE_INPUT]]) {order_value = #NWHC} : tensor<1x1000x1x80xf16> -> tensor<1x80x1x1000xf16>
    // CHECK: [[RESHAPE_FILTER:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [257, 8, 1, 80]} : tensor<257x1x1x640xf16> -> tensor<257x8x1x80xf16>
    // CHECK: [[TRANSPOSE_FILTER:%.+]] = IE.Transpose([[RESHAPE_FILTER]]) {order_value = #NWHC} : tensor<257x8x1x80xf16> -> tensor<257x80x1x8xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[TRANSPOSE_INPUT]], [[TRANSPOSE_FILTER]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]} : tensor<1x80x1x1000xf16>, tensor<257x80x1x8xf16> -> tensor<1x257x1x497xf16>
    // CHECK: return [[CONV]] : tensor<1x257x1x497xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelConvWithGCDWhenOneDimOnH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x80000x1xf16>
func.func @HandleLargeKernelConvWithGCDWhenOneDimOnH(%arg0 : tensor<1x1x80000x1xf16>) -> tensor<1x257x497x1xf16> {
    %cst = const.Declare tensor<257x1x512x1xf16> = dense<1.000000e+00> : tensor<257x1x512x1xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [160, 1]} : tensor<1x1x80000x1xf16>, tensor<257x1x512x1xf16> -> tensor<1x257x497x1xf16>
    return %conv : tensor<1x257x497x1xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<257x1x512x1xf16> = dense<1.000000e+00> : tensor<257x1x512x1xf16>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<257x1x128x1xf16> = dense<0.000000e+00> : tensor<257x1x128x1xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CST]], [[CST_0]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<257x1x512x1xf16>, tensor<257x1x128x1xf16> -> tensor<257x1x640x1xf16>
    // CHECK: [[RESHAPE_INPUT:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 1000, 80, 1]} : tensor<1x1x80000x1xf16> -> tensor<1x1000x80x1xf16>
    // CHECK: [[TRANSPOSE_INPUT:%.+]] = IE.Transpose([[RESHAPE_INPUT]]) {order_value = #NHCW} : tensor<1x1000x80x1xf16> -> tensor<1x80x1000x1xf16>
    // CHECK: [[RESHAPE_FILTER:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [257, 8, 80, 1]} : tensor<257x1x640x1xf16> -> tensor<257x8x80x1xf16>
    // CHECK: [[TRANSPOSE_FILTER:%.+]] = IE.Transpose([[RESHAPE_FILTER]]) {order_value = #NHCW} : tensor<257x8x80x1xf16> -> tensor<257x80x8x1xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[TRANSPOSE_INPUT]], [[TRANSPOSE_FILTER]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} : tensor<1x80x1000x1xf16>, tensor<257x80x8x1xf16> -> tensor<1x257x497x1xf16>
    // CHECK: return [[CONV]] : tensor<1x257x497x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelConvWithHighStrideGCD
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x1x49964xf16>
func.func @HandleLargeKernelConvWithHighStrideGCD(%arg0 : tensor<1x1x1x49964xf16>) -> tensor<1x1025x1x199xf16> {
    %cst = const.Declare tensor<1025x1x1x2048xf16> = dense<1.000000e+00> : tensor<1025x1x1x2048xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 242]} : tensor<1x1x1x49964xf16>, tensor<1025x1x1x2048xf16> -> tensor<1x1025x1x199xf16>
    return %conv : tensor<1x1025x1x199xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1025x1x1x130xf16> = dense<0.000000e+00> : tensor<1025x1x1x130xf16>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1025x1x1x2048xf16> = dense<1.000000e+00> : tensor<1025x1x1x2048xf16>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x130xf16> = dense<0.000000e+00> : tensor<1x1x1x130xf16>
    // CHECK: [[CONCAT_INPUT:%.+]] = IE.Concat([[INPUT]], [[CST_1]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x1x1x49964xf16>, tensor<1x1x1x130xf16> -> tensor<1x1x1x50094xf16>
    // CHECK: [[CONCAT_FILTER:%.+]] = IE.Concat([[CST_0]], [[CST]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1025x1x1x2048xf16>, tensor<1025x1x1x130xf16> -> tensor<1025x1x1x2178xf16>
    // CHECK: [[RESHAPE_INPUT:%.+]] = IE.Reshape([[CONCAT_INPUT]]) {shape_value = [1, 207, 1, 242]} : tensor<1x1x1x50094xf16> -> tensor<1x207x1x242xf16>
    // CHECK: [[TRANSPOSE_INPUT:%.+]] = IE.Transpose([[RESHAPE_INPUT]]) {order_value = #NWHC} : tensor<1x207x1x242xf16> -> tensor<1x242x1x207xf16>
    // CHECK: [[RESHAPE_FILTER:%.+]] = IE.Reshape([[CONCAT_FILTER]]) {shape_value = [1025, 9, 1, 242]} : tensor<1025x1x1x2178xf16> -> tensor<1025x9x1x242xf16>
    // CHECK: [[TRANSPOSE_FILTER:%.+]] = IE.Transpose([[RESHAPE_FILTER]]) {order_value = #NWHC} : tensor<1025x9x1x242xf16> -> tensor<1025x242x1x9xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[TRANSPOSE_INPUT]], [[TRANSPOSE_FILTER]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x242x1x207xf16>, tensor<1025x242x1x9xf16> -> tensor<1x1025x1x199xf16>
    // CHECK: return [[CONV]] : tensor<1x1025x1x199xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelConvWithOutputSlicing
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x1x50100xf16>
func.func @HandleLargeKernelConvWithOutputSlicing(%arg0 : tensor<1x1x1x50100xf16>) -> tensor<1x512x1x199xf16> {
    %cst = const.Declare tensor<512x1x1x2050xf16> = dense<1.000000e+00> : tensor<512x1x1x2050xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 242]} : tensor<1x1x1x50100xf16>, tensor<512x1x1x2050xf16> -> tensor<1x512x1x199xf16>
    return %conv : tensor<1x512x1x199xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<512x1x1x128xf16> = dense<0.000000e+00> : tensor<512x1x1x128xf16>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<512x1x1x2050xf16> = dense<1.000000e+00> : tensor<512x1x1x2050xf16>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x236xf16> = dense<0.000000e+00> : tensor<1x1x1x236xf16>
    // CHECK: [[CONCAT_INPUT:%.+]] = IE.Concat([[INPUT]], [[CST_1]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x1x1x50100xf16>, tensor<1x1x1x236xf16> -> tensor<1x1x1x50336xf16>
    // CHECK: [[CONCAT_FILTER:%.+]] = IE.Concat([[CST_0]], [[CST]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<512x1x1x2050xf16>, tensor<512x1x1x128xf16> -> tensor<512x1x1x2178xf16>
    // CHECK: [[RESHAPE_INPUT:%.+]] = IE.Reshape([[CONCAT_INPUT]]) {shape_value = [1, 208, 1, 242]} : tensor<1x1x1x50336xf16> -> tensor<1x208x1x242xf16>
    // CHECK: [[TRANSPOSE_INPUT:%.+]] = IE.Transpose([[RESHAPE_INPUT]]) {order_value = #NWHC} : tensor<1x208x1x242xf16> -> tensor<1x242x1x208xf16>
    // CHECK: [[RESHAPE_FILTER:%.+]] = IE.Reshape([[CONCAT_FILTER]]) {shape_value = [512, 9, 1, 242]} : tensor<512x1x1x2178xf16> -> tensor<512x9x1x242xf16>
    // CHECK: [[TRANSPOSE_FILTER:%.+]] = IE.Transpose([[RESHAPE_FILTER]]) {order_value = #NWHC} : tensor<512x9x1x242xf16> -> tensor<512x242x1x9xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[TRANSPOSE_INPUT]], [[TRANSPOSE_FILTER]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x242x1x208xf16>, tensor<512x242x1x9xf16> -> tensor<1x512x1x200xf16>
    // CHECK: [[SLICE:%.+]] = IE.Slice [[CONV]] [0, 0, 0, 0] [1, 512, 1, 199] : tensor<1x512x1x200xf16> to tensor<1x512x1x199xf16>
    // CHECK: return [[SLICE]] : tensor<1x512x1x199xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsConvWithDifferentPadding
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x19x1024x1024xf16>
func.func @HandleLargeKernelsConvWithDifferentPadding(%arg0: tensor<1x19x1024x1024xf16>) -> tensor<1x19x1024x1024xf16> {
    %cst = const.Declare tensor<19x19x16x16xf16> = dense<1.000000e+00> : tensor<19x19x16x16xf32>, [#const.CastElemType<f16>]

    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [7, 7], pads_end = [8, 8], strides = [1, 1]} : tensor<1x19x1024x1024xf16>, tensor<19x19x16x16xf16> -> tensor<1x19x1024x1024xf16>
    return %conv : tensor<1x19x1024x1024xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x19x8x1039xf16> = dense<0.000000e+00> : tensor<1x19x8x1039xf16>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x19x7x1039xf16> = dense<0.000000e+00> : tensor<1x19x7x1039xf16>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<19x19x11x11xf16> = dense<1.000000e+00> : tensor<19x19x16x16xf32>, [#const.SubView<[0, 0, 0, 0], [19, 19, 11, 11]>, #const.CastElemType<f16>]
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<19x19x11x5xf16> = dense<1.000000e+00> : tensor<19x19x16x16xf32>, [#const.SubView<[0, 0, 0, 11], [19, 19, 11, 5]>, #const.CastElemType<f16>]
    // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<19x19x5x11xf16> = dense<1.000000e+00> : tensor<19x19x16x16xf32>, [#const.SubView<[0, 0, 11, 0], [19, 19, 5, 11]>, #const.CastElemType<f16>]
    // CHECK-DAG: [[CST_4:%.+]] = const.Declare tensor<19x19x5x5xf16> = dense<1.000000e+00> : tensor<19x19x16x16xf32>, [#const.SubView<[0, 0, 11, 11], [19, 19, 5, 5]>, #const.CastElemType<f16>]
    // CHECK-DAG: [[CST_5:%.+]] = const.Declare tensor<1x19x1024x7xf16> = dense<0.000000e+00> : tensor<1x19x1024x7xf16>
    // CHECK-DAG: [[CST_6:%.+]] = const.Declare tensor<1x19x1024x8xf16> = dense<0.000000e+00> : tensor<1x19x1024x8xf16>
    
    // CHECK: [[CONCAT_0:%.+]] = IE.Concat([[CST_5]], [[INPUT]], [[CST_6]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x19x1024x7xf16>, tensor<1x19x1024x1024xf16>, tensor<1x19x1024x8xf16> -> tensor<1x19x1024x1039xf16>
    // CHECK: [[CONCAT_1:%.+]] = IE.Concat([[CST_0]], [[CONCAT_0]], [[CST]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x19x7x1039xf16>, tensor<1x19x1024x1039xf16>, tensor<1x19x8x1039xf16> -> tensor<1x19x1039x1039xf16>
    // CHECK: [[SLICE_0:%.+]] = IE.Slice [[CONCAT_1]] [0, 0, 0, 0] [1, 19, 1034, 1034] : tensor<1x19x1039x1039xf16> to tensor<1x19x1034x1034xf16>
    // CHECK: [[CONV_0:%.+]] = IE.Convolution([[SLICE_0]], [[CST_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x19x1034x1034xf16>, tensor<19x19x11x11xf16> -> tensor<1x19x1024x1024xf16>
    // CHECK: [[SLICE_1:%.+]] = IE.Slice [[CONCAT_1]] [0, 0, 0, 11] [1, 19, 1034, 1028] : tensor<1x19x1039x1039xf16> to tensor<1x19x1034x1028xf16>
    // CHECK: [[CONV_1:%.+]] = IE.Convolution([[SLICE_1]], [[CST_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x19x1034x1028xf16>, tensor<19x19x11x5xf16> -> tensor<1x19x1024x1024xf16>
    // CHECK: [[ADD_0:%.+]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x19x1024x1024xf16>, tensor<1x19x1024x1024xf16> -> tensor<1x19x1024x1024xf16>
    // CHECK: [[SLICE_2:%.+]] = IE.Slice [[CONCAT_1]] [0, 0, 11, 0] [1, 19, 1028, 1034] : tensor<1x19x1039x1039xf16> to tensor<1x19x1028x1034xf16>
    // CHECK: [[CONV_2:%.+]] = IE.Convolution([[SLICE_2]], [[CST_3]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x19x1028x1034xf16>, tensor<19x19x5x11xf16> -> tensor<1x19x1024x1024xf16>
    // CHECK: [[ADD_1:%.+]] = IE.Add([[ADD_0]], [[CONV_2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x19x1024x1024xf16>, tensor<1x19x1024x1024xf16> -> tensor<1x19x1024x1024xf16>
    // CHECK: [[SLICE_3:%.+]] = IE.Slice [[CONCAT_1]] [0, 0, 11, 11] [1, 19, 1028, 1028] : tensor<1x19x1039x1039xf16> to tensor<1x19x1028x1028xf16>
    // CHECK: [[CONV_3:%.+]] = IE.Convolution([[SLICE_3]], [[CST_4]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x19x1028x1028xf16>, tensor<19x19x5x5xf16> -> tensor<1x19x1024x1024xf16>
    // CHECK: [[ADD_2:%.+]] = IE.Add([[ADD_1]], [[CONV_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x19x1024x1024xf16>, tensor<1x19x1024x1024xf16> -> tensor<1x19x1024x1024xf16>

    // CHECK: return [[ADD_2]] : tensor<1x19x1024x1024xf16>
}


// -----

// CHECK-LABEL: @HandleGlobalConv
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<512x3x16x16xf16>
// CHECK-SAME:  [[INPUT2:%.+]]: tensor<1024x3x16x16xf16>
// CHECK-SAME:  [[INPUT3:%.+]]: tensor<1x1024x1x1xf16>

func.func @HandleGlobalConv(%arg0 : tensor<512x3x16x16xf16>, %arg1 : tensor<1024x3x16x16xf16>,  %arg2 : tensor<1x1024x1x1xf16>) -> tensor<512x1024x1x1xf16> {
    %conv = IE.Convolution(%arg0, %arg1, %arg2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<512x3x16x16xf16>, tensor<1024x3x16x16xf16>, tensor<1x1024x1x1xf16> -> tensor<512x1024x1x1xf16>
    return %conv : tensor<512x1024x1x1xf16>

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.Reshape([[INPUT1]]) {shape_value = [512, 768, 1, 1]} : tensor<512x3x16x16xf16> -> tensor<512x768x1x1xf16>
    // CHECK:   [[RESHAPE_FILTER:%.+]] = IE.Reshape([[INPUT2]]) {shape_value = [1024, 768, 1, 1]} : tensor<1024x3x16x16xf16> -> tensor<1024x768x1x1xf16>
    // CHECK:   [[CONV:%.+]] = IE.Convolution([[RESHAPE_INPUT]], [[RESHAPE_FILTER]], [[INPUT3]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<512x768x1x1xf16>, tensor<1024x768x1x1xf16>, tensor<1x1024x1x1xf16> -> tensor<512x1024x1x1xf16>
    // CHECK:   return [[CONV]] : tensor<512x1024x1x1xf16>
}


// -----

// CHECK-LABEL: @HandleGlobalConvKxBiggerThanLimitation
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<512x3x16x8xf16>
// CHECK-SAME:  [[INPUT2:%.+]]: tensor<1024x3x16x8xf16>

func.func @HandleGlobalConvKxBiggerThanLimitation(%arg0 : tensor<512x3x16x8xf16>, %arg1 : tensor<1024x3x16x8xf16>) -> tensor<512x1024x1x1xf16> {
    %conv = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 8]} : tensor<512x3x16x8xf16>, tensor<1024x3x16x8xf16> -> tensor<512x1024x1x1xf16>
    return %conv : tensor<512x1024x1x1xf16>

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.Reshape([[INPUT1]]) {shape_value = [512, 384, 1, 1]} : tensor<512x3x16x8xf16> -> tensor<512x384x1x1xf16>
    // CHECK:   [[RESHAPE_FILTER:%.+]] = IE.Reshape([[INPUT2]]) {shape_value = [1024, 384, 1, 1]} : tensor<1024x3x16x8xf16> -> tensor<1024x384x1x1xf16>
    // CHECK:   [[CONV:%.+]] = IE.Convolution([[RESHAPE_INPUT]], [[RESHAPE_FILTER]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<512x384x1x1xf16>, tensor<1024x384x1x1xf16> -> tensor<512x1024x1x1xf16>
    // CHECK:   return [[CONV]] : tensor<512x1024x1x1xf16>
}


// -----

// CHECK-LABEL: @HandleGlobalConvWithPadding
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<512x3x16x16xf16>
// CHECK-SAME:  [[INPUT2:%.+]]: tensor<1024x3x16x16xf16>

func.func @HandleGlobalConvWithPadding(%arg0 : tensor<512x3x16x16xf16>, %arg1 : tensor<1024x3x16x16xf16>) -> tensor<512x1024x1x1xf16> {
    %conv = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [16, 16]} : tensor<512x3x16x16xf16>, tensor<1024x3x16x16xf16> -> tensor<512x1024x1x1xf16>
    return %conv : tensor<512x1024x1x1xf16>

    // CHECK:   [[CST:%.+]] = const.Declare tensor<512x3x1x18xf16> = dense<0.000000e+00> : tensor<512x3x1x18xf16>
    // CHECK:   [[CST_0:%.+]] = const.Declare tensor<512x3x16x1xf16> = dense<0.000000e+00> : tensor<512x3x16x1xf16>
    // CHECK:   [[SLICE1:%.+]] = IE.Slice [[ARG1:%.+]] [0, 0, 0, 0] [1024, 3, 11, 11] : tensor<1024x3x16x16xf16> to tensor<1024x3x11x11xf16>
    // CHECK:   [[SLICE2:%.+]] = IE.Slice [[ARG1]] [0, 0, 0, 11] [1024, 3, 11, 5] : tensor<1024x3x16x16xf16> to tensor<1024x3x11x5xf16>
    // CHECK:   [[SLICE3:%.+]] = IE.Slice [[ARG1]] [0, 0, 11, 0] [1024, 3, 5, 11] : tensor<1024x3x16x16xf16> to tensor<1024x3x5x11xf16>
    // CHECK:   [[SLICE4:%.+]] = IE.Slice [[ARG1]] [0, 0, 11, 11] [1024, 3, 5, 5] : tensor<1024x3x16x16xf16> to tensor<1024x3x5x5xf16>
    // CHECK:   [[CONCAT1:%.+]] = IE.Concat([[CST_0]], [[ARG0:%.+]], [[CST_0]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<512x3x16x1xf16>, tensor<512x3x16x16xf16>, tensor<512x3x16x1xf16> -> tensor<512x3x16x18xf16>
    // CHECK:   [[CONCAT2:%.+]] = IE.Concat([[CST]], [[CONCAT1]], [[CST]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<512x3x1x18xf16>, tensor<512x3x16x18xf16>, tensor<512x3x1x18xf16> -> tensor<512x3x18x18xf16>
    // CHECK:   [[SLICE5:%.+]] = IE.Slice [[CONCAT2]] [0, 0, 0, 0] [512, 3, 11, 11] : tensor<512x3x18x18xf16> to tensor<512x3x11x11xf16>
    // CHECK:   [[CONV1:%.+]] = IE.Convolution([[SLICE5]], [[SLICE1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<512x3x11x11xf16>, tensor<1024x3x11x11xf16> -> tensor<512x1024x1x1xf16>
    // CHECK:   [[SLICE6:%.+]] = IE.Slice [[CONCAT2]] [0, 0, 0, 11] [512, 3, 11, 5] : tensor<512x3x18x18xf16> to tensor<512x3x11x5xf16>
    // CHECK:   [[CONV2:%.+]] = IE.Convolution([[SLICE6]], [[SLICE2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<512x3x11x5xf16>, tensor<1024x3x11x5xf16> -> tensor<512x1024x1x1xf16>
    // CHECK:   [[ADD1:%.+]] = IE.Add([[CONV1]], [[CONV2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<512x1024x1x1xf16>, tensor<512x1024x1x1xf16> -> tensor<512x1024x1x1xf16>
    // CHECK:   [[SLICE7:%.+]] = IE.Slice [[CONCAT2]] [0, 0, 11, 0] [512, 3, 5, 11] : tensor<512x3x18x18xf16> to tensor<512x3x5x11xf16>
    // CHECK:   [[CONV3:%.+]] = IE.Convolution([[SLICE7]], [[SLICE3]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<512x3x5x11xf16>, tensor<1024x3x5x11xf16> -> tensor<512x1024x1x1xf16>
    // CHECK:   [[ADD2:%.+]] = IE.Add([[ADD1]], [[CONV3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<512x1024x1x1xf16>, tensor<512x1024x1x1xf16> -> tensor<512x1024x1x1xf16>
    // CHECK:   [[SLICE8:%.+]] = IE.Slice [[CONCAT2]] [0, 0, 11, 11] [512, 3, 5, 5] : tensor<512x3x18x18xf16> to tensor<512x3x5x5xf16>
    // CHECK:   [[CONV4:%.+]] = IE.Convolution([[SLICE8]], [[SLICE4]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<512x3x5x5xf16>, tensor<1024x3x5x5xf16> -> tensor<512x1024x1x1xf16>
    // CHECK:   [[ADD3:%.+]] = IE.Add([[ADD2]], [[CONV4]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<512x1024x1x1xf16>, tensor<512x1024x1x1xf16> -> tensor<512x1024x1x1xf16>

    // CHECK:   return [[ADD3]] : tensor<512x1024x1x1xf16>
}
