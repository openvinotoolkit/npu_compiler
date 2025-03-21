//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-convolution-weights %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @AdjustConvWeightsCase1
func.func @AdjustConvWeightsCase1(%arg0: tensor<1x16x40x40xf16>) -> tensor<1x32x40x40xf16> {
    %filter = const.Declare tensor<32x16x1x1xf16> = dense<1.000000e+00> : tensor<32x16x1x1xf16>
    %bias = const.Declare tensor<1x32x1x1xf16> = dense<1.000000e+00> : tensor<1x32x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x40x40xf16>, tensor<32x16x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x40x40xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 29, 40, 40] : tensor<1x32x40x40xf16> to tensor<1x29x40x40xf16>
    %2 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x40x40xf16>, tensor<32x16x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x40x40xf16>
    %3 = IE.Concat(%1, %2) {static_offsets = [[0, 0, 0, 0], [0, 29, 0, 0]]} : tensor<1x29x40x40xf16>, tensor<1x32x40x40xf16> -> tensor<1x61x40x40xf16>
    %4 = IE.Expand(%3) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x61x40x40xf16> -> tensor<1x64x40x40xf16>
    %filter1 = const.Declare tensor<32x64x1x1xf16> = dense<1.000000e+00> : tensor<32x64x1x1xf16>
    %5 = IE.Convolution(%4, %filter1, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x40x40xf16>, tensor<32x64x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x40x40xf16>
    return %5 : tensor<1x32x40x40xf16>

    // CHECK-DAG:   [[FILTER0:%.*]] = const.Declare tensor<32x32x1x1xf16> = dense<1.000000e+00> : tensor<32x64x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [32, 29, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 3, 0, 0]>]
    // CHECK-DAG:   [[FILTER1:%.*]] = const.Declare tensor<32x32x1x1xf16> = dense<1.000000e+00> : tensor<32x64x1x1xf16>, [#const.SubView<[0, 29, 0, 0], [32, 32, 1, 1]>]
    // CHECK-DAG:   [[FILTER2:%.*]] = const.Declare tensor<32x16x1x1xf16> = dense<1.000000e+00> : tensor<32x16x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x32x1x1xf16> = dense<1.000000e+00> : tensor<1x32x1x1xf16>
    // CHECK:       [[CONV0:%.*]] = IE.Convolution({{[^:]+}}, [[FILTER2]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x40x40xf16>, tensor<32x16x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x40x40xf16>
    // CHECK:       [[CONV1:%.*]] = IE.Convolution({{[^:]+}}, [[FILTER2]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x40x40xf16>, tensor<32x16x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x40x40xf16>
    // CHECK:       [[CONCAT0:%.*]] = IE.Concat([[CONV0]], [[CONV1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x32x40x40xf16>, tensor<1x32x40x40xf16> -> tensor<1x64x40x40xf16>
    // CHECK:       [[CONCAT1:%.*]] = IE.Concat([[FILTER0]], [[FILTER1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<32x32x1x1xf16>, tensor<32x32x1x1xf16> -> tensor<32x64x1x1xf16>
    // CHECK:       [[CONV2:%.*]] = IE.Convolution([[CONCAT0]], [[CONCAT1]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x40x40xf16>, tensor<32x64x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x40x40xf16>
    // CHECK:       return [[CONV2]] : tensor<1x32x40x40xf16>
}

// CHECK-LABEL: @AdjustConvWeightsCase2
func.func @AdjustConvWeightsCase2(%arg0: tensor<1x16x40x40xf16>) -> tensor<1x32x40x40xf16> {
    %filter = const.Declare tensor<32x16x1x1xf16> = dense<1.000000e+00> : tensor<32x16x1x1xf16>
    %bias = const.Declare tensor<1x32x1x1xf16> = dense<1.000000e+00> : tensor<1x32x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x40x40xf16>, tensor<32x16x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x40x40xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 17, 40, 40] : tensor<1x32x40x40xf16> to tensor<1x17x40x40xf16>
    %2 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x40x40xf16>, tensor<32x16x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x40x40xf16>
    %3 = IE.Slice %2 [0, 0, 0, 0] [1, 17, 40, 40] : tensor<1x32x40x40xf16> to tensor<1x17x40x40xf16>
    %4 = IE.Concat(%1, %3) {static_offsets = [[0, 0, 0, 0], [0, 17, 0, 0]]} : tensor<1x17x40x40xf16>, tensor<1x17x40x40xf16> -> tensor<1x34x40x40xf16>
    %5 = IE.Expand(%4) {pads_begin = [0, 0, 0, 0], pads_end = [0, 14, 0, 0]} : tensor<1x34x40x40xf16> -> tensor<1x48x40x40xf16>
    %filter1 = const.Declare tensor<32x48x1x1xf16> = dense<1.000000e+00> : tensor<32x48x1x1xf16>
    %6 = IE.Convolution(%5, %filter1, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x48x40x40xf16>, tensor<32x48x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x40x40xf16>
    return %6 : tensor<1x32x40x40xf16>

    // CHECK-DAG:   [[FILTER0:%.*]] = const.Declare tensor<32x32x1x1xf16> = dense<1.000000e+00> : tensor<32x48x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [32, 17, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>]
    // CHECK-DAG:   [[FILTER1:%.*]] = const.Declare tensor<32x32x1x1xf16> = dense<1.000000e+00> : tensor<32x48x1x1xf16>, [#const.SubView<[0, 17, 0, 0], [32, 17, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>]
    // CHECK-DAG:   [[FILTER2:%.*]] = const.Declare tensor<32x16x1x1xf16> = dense<1.000000e+00> : tensor<32x16x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x32x1x1xf16> = dense<1.000000e+00> : tensor<1x32x1x1xf16>
    // CHECK:       [[CONV0:%.*]] = IE.Convolution({{[^:]+}}, [[FILTER2]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x40x40xf16>, tensor<32x16x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x40x40xf16>
    // CHECK:       [[CONV1:%.*]] = IE.Convolution({{[^:]+}}, [[FILTER2]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x40x40xf16>, tensor<32x16x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x40x40xf16>
    // CHECK:       [[CONCAT0:%.*]] = IE.Concat([[CONV0]], [[CONV1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x32x40x40xf16>, tensor<1x32x40x40xf16> -> tensor<1x64x40x40xf16>
    // CHECK:       [[CONCAT1:%.*]] = IE.Concat([[FILTER0]], [[FILTER1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<32x32x1x1xf16>, tensor<32x32x1x1xf16> -> tensor<32x64x1x1xf16>
    // CHECK:       [[CONV2:%.*]] = IE.Convolution([[CONCAT0]], [[CONCAT1]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x40x40xf16>, tensor<32x64x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x40x40xf16>
    // CHECK:       return [[CONV2]] : tensor<1x32x40x40xf16>
}

// CHECK-LABEL: @AdjustConvWeightsNotConvertSliceOnH
func.func @AdjustConvWeightsNotConvertSliceOnH(%arg0: tensor<1x16x41x41xf16>, %arg1: tensor<1x16x40x40xf16>) -> tensor<1x32x40x40xf16> {
    %filter = const.Declare tensor<32x16x1x1xf16> = dense<1.000000e+00> : tensor<32x16x1x1xf16>
    %bias = const.Declare tensor<1x32x1x1xf16> = dense<1.000000e+00> : tensor<1x32x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x41x41xf16>, tensor<32x16x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x41x41xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 17, 40, 40] : tensor<1x32x41x41xf16> to tensor<1x17x40x40xf16>
    %2 = IE.Convolution(%arg1, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x40x40xf16>, tensor<32x16x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x40x40xf16>
    %3 = IE.Slice %2 [0, 0, 0, 0] [1, 17, 40, 40] : tensor<1x32x40x40xf16> to tensor<1x17x40x40xf16>
    %4 = IE.Concat(%1, %3) {static_offsets = [[0, 0, 0, 0], [0, 17, 0, 0]]} : tensor<1x17x40x40xf16>, tensor<1x17x40x40xf16> -> tensor<1x34x40x40xf16>
    %5 = IE.Expand(%4) {pads_begin = [0, 0, 0, 0], pads_end = [0, 14, 0, 0]} : tensor<1x34x40x40xf16> -> tensor<1x48x40x40xf16>
    %filter1 = const.Declare tensor<32x48x1x1xf16> = dense<1.000000e+00> : tensor<32x48x1x1xf16>
    %6 = IE.Convolution(%5, %filter1, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x48x40x40xf16>, tensor<32x48x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x40x40xf16>
    return %6 : tensor<1x32x40x40xf16>

    // CHECK-DAG:   [[FILTER0:%.*]] = const.Declare tensor<32x48x1x1xf16> = dense<1.000000e+00> : tensor<32x48x1x1xf16>
    // CHECK-DAG:   [[FILTER1:%.*]] = const.Declare tensor<32x16x1x1xf16> = dense<1.000000e+00> : tensor<32x16x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x32x1x1xf16> = dense<1.000000e+00> : tensor<1x32x1x1xf16>
    // CHECK:       [[CONV0:%.*]] = IE.Convolution({{[^:]+}}, [[FILTER1]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x41x41xf16>, tensor<32x16x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x41x41xf16>
    // CHECK:       [[SLICE0:%.*]] = IE.Slice [[CONV0]] [0, 0, 0, 0] [1, 17, 40, 40] : tensor<1x32x41x41xf16> to tensor<1x17x40x40xf16>
    // CHECK:       [[CONV1:%.*]]  = IE.Convolution({{[^:]+}}, [[FILTER1]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x40x40xf16>, tensor<32x16x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x40x40xf16>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice [[CONV1]] [0, 0, 0, 0] [1, 17, 40, 40] : tensor<1x32x40x40xf16> to tensor<1x17x40x40xf16>
    // CHECK:       [[CONCAT:%.*]]  = IE.Concat([[SLICE0]], [[SLICE1]])
    //CHECK-SAME{LITERAL}           {static_offsets = [[0, 0, 0, 0], [0, 17, 0, 0]]} : tensor<1x17x40x40xf16>, tensor<1x17x40x40xf16> -> tensor<1x34x40x40xf16>
    // CHECK:       [[EXPAND:%.*]]  = IE.Expand([[CONCAT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 14, 0, 0]} : tensor<1x34x40x40xf16> -> tensor<1x48x40x40xf16>
    // CHECK:       [[CONV2:%.*]] = IE.Convolution([[EXPAND]], [[FILTER0]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x48x40x40xf16>, tensor<32x48x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x40x40xf16>
    // CHECK:       return [[CONV2]] : tensor<1x32x40x40xf16>
}

// CHECK-LABEL: @AdjustConvWeightsMultiOuputSlice
func.func @AdjustConvWeightsMultiOuputSlice(%arg0: tensor<1x16x40x40xf16>) -> (tensor<1x17x40x40xf16>, tensor<1x32x40x40xf16>) {
    %filter = const.Declare tensor<32x16x1x1xf16> = dense<1.000000e+00> : tensor<32x16x1x1xf16>
    %bias = const.Declare tensor<1x32x1x1xf16> = dense<1.000000e+00> : tensor<1x32x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x40x40xf16>, tensor<32x16x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x40x40xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 17, 40, 40] : tensor<1x32x40x40xf16> to tensor<1x17x40x40xf16>
    %2 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x40x40xf16>, tensor<32x16x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x40x40xf16>
    %3 = IE.Slice %2 [0, 0, 0, 0] [1, 17, 40, 40] : tensor<1x32x40x40xf16> to tensor<1x17x40x40xf16>
    %4 = IE.Concat(%1, %3) {static_offsets = [[0, 0, 0, 0], [0, 17, 0, 0]]} : tensor<1x17x40x40xf16>, tensor<1x17x40x40xf16> -> tensor<1x34x40x40xf16>
    %5 = IE.Expand(%4) {pads_begin = [0, 0, 0, 0], pads_end = [0, 14, 0, 0]} : tensor<1x34x40x40xf16> -> tensor<1x48x40x40xf16>
    %filter1 = const.Declare tensor<32x48x1x1xf16> = dense<1.000000e+00> : tensor<32x48x1x1xf16>
    %6 = IE.Convolution(%5, %filter1, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x48x40x40xf16>, tensor<32x48x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x40x40xf16>
    return %3, %6 : tensor<1x17x40x40xf16>, tensor<1x32x40x40xf16>

    // CHECK-DAG:   [[FILTER0:%.*]] = const.Declare tensor<32x32x1x1xf16> = dense<1.000000e+00> : tensor<32x48x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [32, 17, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>]
    // CHECK-DAG:   [[FILTER1:%.*]] = const.Declare tensor<32x32x1x1xf16> = dense<1.000000e+00> : tensor<32x48x1x1xf16>, [#const.SubView<[0, 17, 0, 0], [32, 17, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>]
    // CHECK-DAG:   [[FILTER2:%.*]] = const.Declare tensor<32x16x1x1xf16> = dense<1.000000e+00> : tensor<32x16x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x32x1x1xf16> = dense<1.000000e+00> : tensor<1x32x1x1xf16>
    // CHECK:       [[CONV0:%.*]] = IE.Convolution({{[^:]+}}, [[FILTER2]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x40x40xf16>, tensor<32x16x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x40x40xf16>
    // CHECK:       [[CONV1:%.*]] = IE.Convolution({{[^:]+}}, [[FILTER2]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x40x40xf16>, tensor<32x16x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x40x40xf16>
    // CHECK:       [[SLICE:%.*]]  = IE.Slice [[CONV1]] [0, 0, 0, 0] [1, 17, 40, 40] : tensor<1x32x40x40xf16> to tensor<1x17x40x40xf16>
    // CHECK:       [[CONCAT0:%.*]] = IE.Concat([[CONV0]], [[CONV1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x32x40x40xf16>, tensor<1x32x40x40xf16> -> tensor<1x64x40x40xf16>
    // CHECK:       [[CONCAT1:%.*]] = IE.Concat([[FILTER0]], [[FILTER1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<32x32x1x1xf16>, tensor<32x32x1x1xf16> -> tensor<32x64x1x1xf16>
    // CHECK:       [[CONV2:%.*]] = IE.Convolution([[CONCAT0]], [[CONCAT1]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x40x40xf16>, tensor<32x64x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x40x40xf16>
    // CHECK:       return [[SLICE]], [[CONV2]] : tensor<1x17x40x40xf16>, tensor<1x32x40x40xf16>

}

// CHECK-LABEL: @AdjustConvWeightsCompressConvCase
func.func @AdjustConvWeightsCompressConvCase(%arg0: tensor<1x16x40x40xf16>) -> tensor<1x16x40x40xf16> {
    %filter = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x40x40xf16>, tensor<16x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x40x40xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 1, 40, 40] : tensor<1x16x40x40xf16> to tensor<1x1x40x40xf16>
    %2 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x40x40xf16>, tensor<16x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x40x40xf16>
    %3 = IE.Slice %2 [0, 0, 0, 0] [1, 2, 40, 40] : tensor<1x16x40x40xf16> to tensor<1x2x40x40xf16>
    %4 = IE.Concat(%1, %3) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x40x40xf16>, tensor<1x2x40x40xf16> -> tensor<1x3x40x40xf16>
    %5 = IE.Expand(%4) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x40x40xf16> -> tensor<1x4x40x40xf16>
    %filter1 = const.Declare tensor<16x4x1x1xf16> = dense<1.000000e+00> : tensor<16x4x1x1xf16>
    %6 = IE.Convolution(%5, %filter1, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x40x40xf16>, tensor<16x4x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x40x40xf16>

    return %6 : tensor<1x16x40x40xf16>

    // CHECK-DAG:   [[FILTER0:%.*]] = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x4x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>]
    // CHECK-DAG:   [[FILTER1:%.*]] = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x4x1x1xf16>, [#const.SubView<[0, 1, 0, 0], [16, 2, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 14, 0, 0]>]
    // CHECK-DAG:   [[FILTER2:%.*]] = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
    // CHECK:       [[CONV0:%.*]] = IE.Convolution({{[^:]+}}, [[FILTER2]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x40x40xf16>, tensor<16x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x40x40xf16>
    // CHECK:       [[CONV1:%.*]] = IE.Convolution({{[^:]+}}, [[FILTER2]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x40x40xf16>, tensor<16x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x40x40xf16>
    // CHECK:       [[CONCAT0:%.*]] = IE.Concat([[CONV0]], [[CONV1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x16x40x40xf16>, tensor<1x16x40x40xf16> -> tensor<1x32x40x40xf16>
    // CHECK:       [[CONCAT1:%.*]] = IE.Concat([[FILTER0]], [[FILTER1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<16x16x1x1xf16>, tensor<16x16x1x1xf16> -> tensor<16x32x1x1xf16>
    // CHECK:       [[CONV2:%.*]] = IE.Convolution([[CONCAT0]], [[CONCAT1]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x40x40xf16>, tensor<16x32x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x40x40xf16>
    // CHECK:       return [[CONV2]] : tensor<1x16x40x40xf16>
}
