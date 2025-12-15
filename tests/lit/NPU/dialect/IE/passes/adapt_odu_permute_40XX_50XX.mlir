//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adapt-odu-permute  %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @DoNotChangePermuteODU
func.func @DoNotChangePermuteODU(%arg0: tensor<1x16x1x64xf16, {order = #NHWC}>) -> tensor<1x16x1x64xf16, {order = #NCWH}> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x1x64xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x1x64xf16, {order = #NCWH}>

    return %0 : tensor<1x16x1x64xf16, {order = #NCWH}>

    // CHECK:   [[CONV:%.+]] = IE.Convolution
    // CHECK-SAME: -> tensor<1x16x1x64xf16, {order = #NCWH}>

    // CHECK:   return [[CONV]] : tensor<1x16x1x64xf16, {order = #NCWH}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @PermuteODUNWCH
func.func @PermuteODUNWCH(%arg0: tensor<1x16x1x64xf16, {order = #NHWC}>) -> tensor<1x16x1x64xf16, {order = #NWCH}> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x1x64xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x1x64xf16, {order = #NWCH}>

    return %0 : tensor<1x16x1x64xf16, {order = #NWCH}>

    // CHECK:   [[CONV:%.+]] = IE.Convolution
    // CHECK-SAME: -> tensor<1x16x1x64xf16, {order = #NHWC}>

    // CHECK:   [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[CONV]]) {dst_order = #NWCH, mem_perm = #NHWC}
    // CHECK-SAME:  tensor<1x16x1x64xf16, {order = #NHWC}>
    // CHECK-SAME:  -> tensor<1x16x1x64xf16, {order = #NWCH}>
    // CHECK:   return [[PERMUTE_CAST]] : tensor<1x16x1x64xf16, {order = #NWCH}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @PermuteODUNHCW
func.func @PermuteODUNHCW(%arg0: tensor<1x16x64x1xf16, {order = #NHWC}>) -> tensor<1x16x64x1xf16, {order = #NHCW}> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x64x1xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x64x1xf16, {order = #NHCW}>

    return %0 : tensor<1x16x64x1xf16, {order = #NHCW}>

    // CHECK:   [[CONV:%.+]] = IE.Convolution
    // CHECK-SAME: -> tensor<1x16x64x1xf16, {order = #NWHC}>

    // CHECK:   [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[CONV]]) {dst_order = #NHCW, mem_perm = #NHWC}
    // CHECK-SAME:  tensor<1x16x64x1xf16, {order = #NWHC}>
    // CHECK-SAME:  -> tensor<1x16x64x1xf16, {order = #NHCW}>
    // CHECK:   return [[PERMUTE_CAST]] : tensor<1x16x64x1xf16, {order = #NHCW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @PermuteODUWithPermuteCastBefore
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x16x1x64xf16, {order = #NHWC}>) -> tensor<1x16x1x33xf16> {
func.func @PermuteODUWithPermuteCastBefore(%arg0: tensor<1x16x1x64xf16, {order = #NHWC}>) -> tensor<1x16x1x33xf16> {
    %0 = IE.MaxPool(%arg0) {
        kernel_size = [1, 2],
        input_padding = [0, 0, 0, 1],
        output_padding = [0, 0, 0, 1],
        pads_begin = [0, 1],
        pads_end = [0, 1],
        strides = [1, 2],
        rounding_type = #IE.rounding_type<FLOOR>,
        post_op = #IE.Clamp<min = 0.000000e+00 : f64, max = 6.000000e+00 : f64>
    } : tensor<1x16x1x64xf16, {order = #NHWC}>
            -> tensor<1x16x1x33xf16>

    return %0 : tensor<1x16x1x33xf16>


    // CHECK:   [[PERMUTE_CAST_IN:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NHWC, mem_perm = #NHCW}
    // CHECK-SAME:      : tensor<1x16x1x64xf16, {order = #NHWC}> -> tensor<1x16x64x1xf16, {order = #NHWC}>
    // CHECK:   [[CONV:%.+]] = IE.MaxPool([[PERMUTE_CAST_IN]]
    // CHECK-SAME:          input_padding = [0, 0, 1, 0],
    // CHECK-SAME:          kernel_size = [2, 1]
    // CHECK-SAME:          output_padding = [0, 0, 1, 0],
    // CHECK-SAME:          pads_begin = [1, 0],
    // CHECK-SAME:          pads_end = [1, 0],
    // CHECK-SAME:          strides = [2, 1]
    // CHECK-SAME:      tensor<1x16x64x1xf16, {order = #NHWC}> -> tensor<1x16x33x1xf16, {order = #NCWH}>

    // CHECK:   [[PERMUTE_CAST_OUT:%.+]] = IE.PermuteCast([[CONV]]) {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME:      tensor<1x16x33x1xf16, {order = #NCWH}>
    // CHECK-SAME:      -> tensor<1x16x1x33xf16>
    // CHECK:   return [[PERMUTE_CAST_OUT]] : tensor<1x16x1x33xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @AddPermuteODUWithNoPermuteCastBefore
func.func @AddPermuteODUWithNoPermuteCastBefore(%arg0: tensor<1x16x1x64xf16, {order = #NHWC}>) -> tensor<1x16x1x64xf16, {order = #NWCH}> {
    %0 = IE.Add(%arg0, %arg0) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x1x64xf16, {order = #NHWC}>,
        tensor<1x16x1x64xf16, {order = #NHWC}>
            -> tensor<1x16x1x64xf16, {order = #NWCH}>

    return %0 : tensor<1x16x1x64xf16, {order = #NWCH}>

    // CHECK:   [[ADD:%.+]] = IE.Add
    // CHECK-SAME: -> tensor<1x16x1x64xf16, {order = #NHWC}>

    // CHECK:   [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[ADD]]) {dst_order = #NWCH, mem_perm = #NHWC}
    // CHECK-SAME:  tensor<1x16x1x64xf16, {order = #NHWC}>
    // CHECK-SAME:  -> tensor<1x16x1x64xf16, {order = #NWCH}>
    // CHECK:   return [[PERMUTE_CAST]] : tensor<1x16x1x64xf16, {order = #NWCH}>
}
