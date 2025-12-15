//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --swap-mem-permute-and-expand %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MemPermuteWithExpandNeedSwap
// CHECK-SAME:       ([[ARG0:%arg[0-9]+]]: tensor<1x5x512x512xf16>)
func.func @MemPermuteWithExpandNeedSwap(%arg0: tensor<1x5x512x512xf16>) -> tensor<1x16x512x512xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %1 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x5x512x512xf16> -> tensor<1x5x512x512xf16, {order = #NHWC}>
    %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 11, 0, 0]} : tensor<1x5x512x512xf16, {order = #NHWC}> -> tensor<1x16x512x512xf16, {order = #NHWC}>
    %3 = IE.Convolution(%2, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Relu<>, strides = [1, 1]} : tensor<1x16x512x512xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x512x512xf16, {order = #NHWC}>

    return %3 : tensor<1x16x512x512xf16, {order = #NHWC}>
    // CHECK-DAG:       [[CST:%.*]] = const.Declare
    // CHECK-SAME:      tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]

    // CHECK:       [[EXPAND:%.*]] = IE.Expand([[ARG0]])
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0]
    // CHECK-SAME:      pads_end = [0, 11, 0, 0]
    // CHECK-SAME:      : tensor<1x5x512x512xf16> -> tensor<1x16x512x512xf16>

    // CHECK:       [[PERMUTE:%.*]] = IE.MemPermute([[EXPAND]])
    // CHECK-SAME:      {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x512x512xf16> -> tensor<1x16x512x512xf16, {order = #NHWC}>

    // CHECK:       [[CONV:%.*]] = IE.Convolution([[PERMUTE]], [[CST]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      post_op = #IE.Relu<>
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x16x512x512xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x512x512xf16, {order = #NHWC}>

    // CHECK:       return [[CONV]] : tensor<1x16x512x512xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @MemPermuteWithExpandAndSlice
// CHECK-SAME:       ([[ARG0:%arg[0-9]+]]: tensor<1x5x512x1xf16>)
func.func @MemPermuteWithExpandAndSlice(%arg0: tensor<1x5x512x1xf16>) -> tensor<1x1x512x5xf16> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x5x512x1xf16> -> tensor<1x5x512x1xf16, {order = #NCWH}>
    %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 11, 0, 0]} : tensor<1x5x512x1xf16, {order = #NCWH}> -> tensor<1x16x512x1xf16, {order = #NCWH}>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 5, 512, 1] : tensor<1x16x512x1xf16, {order = #NCWH}> to tensor<1x5x512x1xf16, {order = #NCWH}>
    %3 = IE.MemPermute(%2) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x5x512x1xf16, {order = #NCWH}> -> tensor<1x1x512x5xf16>
    return %3 : tensor<1x1x512x5xf16>

    // CHECK;      [[EXPAND:%.*]] = IE.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 11, 0, 0]} : tensor<1x5x512x1xf16> -> tensor<1x16x512x1xf16>
    // CHECK;      [[PERMUTE0:%.*]] = IE.MemPermute([[EXPAND]]) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x16x512x1xf16> -> tensor<1x16x512x1xf16, {order = #NCWH}>
    // CHECK;      [[SLICE:%.*]] = IE.Slice [[PERMUTE0]] [0, 0, 0, 0] [1, 5, 512, 1] : tensor<1x16x512x1xf16, {order = #NCWH}> to tensor<1x5x512x1xf16, {order = #NCWH}>
    // CHECK;      [[PERMUTE1:%.*]] = IE.MemPermute([[SLICE]]) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x5x512x1xf16, {order = #NCWH}> -> tensor<1x1x512x5xf16>
    // CHECK;      return [[PERMUTE1]] : tensor<1x1x512x5xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>


// CHECK-LABEL: @DoNotSwapMemPermuteWithExpandIDUAutopad
// CHECK-SAME:       ([[ARG0:%arg[0-9]+]]: tensor<1x5x512x512xf16>)
func.func @DoNotSwapMemPermuteWithExpandIDUAutopad(%arg0: tensor<1x5x512x512xf16>) -> tensor<1x16x512x512xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %1 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x5x512x512xf16> -> tensor<1x5x512x512xf16, {order = #NHWC}>
    %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 11, 0, 0]} : tensor<1x5x512x512xf16, {order = #NHWC}> -> tensor<1x16x512x512xf16, {order = #NHWC}>
    %3 = IE.Convolution(%2, %0) {input_padding = [0, 11, 0, 0], dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Relu<>, strides = [1, 1]} : tensor<1x16x512x512xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x512x512xf16, {order = #NHWC}>

    return %3 : tensor<1x16x512x512xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.*]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK:       [[PERMUTE:%.*]] = IE.MemPermute([[ARG0]])
    // CHECK:       [[EXPAND:%.*]] = IE.Expand([[PERMUTE]])
    // CHECK:       [[CONV:%.*]] = IE.Convolution([[EXPAND]], [[CST]])
    // CHECK:       return [[CONV]]
}
