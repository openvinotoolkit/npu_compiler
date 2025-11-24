//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-mem-permute  %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @MemPermuteNWHC
func.func @MemPermuteNWHC(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x16x64x32xf16> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x32x64xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NCHW,
        mem_perm = #NWHC
    } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x16x64x32xf16>

    return %1 : tensor<1x16x64x32xf16>

    // CHECK:   [[CONV:%.*]] = IE.Convolution
    // CHECK-SAME: -> tensor<1x16x32x64xf16, {order = #NCWH}>
    // CHECK-NOT: IE.MemPermute

    // CHECK:   [[PERMUTE_CAST:%.*]] = IE.PermuteCast([[CONV]]) {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME:  tensor<1x16x32x64xf16, {order = #NCWH}>
    // CHECK-SAME:  -> tensor<1x16x64x32xf16>
    // CHECK:   return [[PERMUTE_CAST]] : tensor<1x16x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MemPermuteNWCH
func.func @MemPermuteNWCH(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x16x32x64xf16> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x32x64xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NCHW,
        mem_perm = #NWCH
    } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x16x32x64xf16>

    return %1 : tensor<1x16x32x64xf16>

    // CHECK:   [[CONV:%.*]] = IE.Convolution
    // CHECK-SAME: -> tensor<1x16x32x64xf16>
    // CHECK-NOT: IE.MemPermute

    // CHECK:   return [[CONV]] : tensor<1x16x32x64xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @MemPermuteNHCW
func.func @MemPermuteNHCW(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x64x32x16xf16> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x32x64xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NCHW,
        mem_perm = #NHCW
    } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x64x32x16xf16>

    return %1 : tensor<1x64x32x16xf16>

    // CHECK:   [[CONV:%.*]] = IE.Convolution
    // CHECK-SAME: -> tensor<1x16x32x64xf16, {order = #NWHC}>
    // CHECK-NOT: IE.MemPermute

    // CHECK:   [[PERMUTE_CAST:%.*]] = IE.PermuteCast([[CONV]]) {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME:  tensor<1x16x32x64xf16, {order = #NWHC}>
    // CHECK-SAME:  -> tensor<1x64x32x16xf16>
    // CHECK:   return [[PERMUTE_CAST]] : tensor<1x64x32x16xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MemPermuteNHWC
func.func @MemPermuteNHWC(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x64x16x32xf16> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x32x64xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NCHW,
        mem_perm = #NHWC
    } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x64x16x32xf16>

    return %1 : tensor<1x64x16x32xf16>

    // CHECK:   [[CONV:%.*]] = IE.Convolution
    // CHECK-SAME: -> tensor<1x16x32x64xf16, {order = #NWCH}>
    // CHECK-NOT: IE.MemPermute

    // CHECK:   [[PERMUTE_CAST:%.*]] = IE.PermuteCast([[CONV]]) {dst_order = #NCHW, mem_perm = #NCHW} :
    // CHECK-SAME:  tensor<1x16x32x64xf16, {order = #NWCH}>
    // CHECK-SAME:  -> tensor<1x64x16x32xf16>
    // CHECK:   return [[PERMUTE_CAST]] : tensor<1x64x16x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @MemPermuteNCWH
func.func @MemPermuteNCWH(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x32x16x64xf16> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x32x64xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NCHW,
        mem_perm = #NCWH
    } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x32x16x64xf16>

    return %1 : tensor<1x32x16x64xf16>

    // CHECK:   [[CONV:%.*]] = IE.Convolution
    // CHECK-SAME: -> tensor<1x16x32x64xf16, {order = #NHCW}>
    // CHECK-NOT: IE.MemPermute

    // CHECK:   [[PERMUTE_CAST:%.*]] = IE.PermuteCast([[CONV]]) {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME:  tensor<1x16x32x64xf16, {order = #NHCW}>
    // CHECK-SAME:  -> tensor<1x32x16x64xf16>
    // CHECK:   return [[PERMUTE_CAST]] : tensor<1x32x16x64xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @MemPermuteNWHCOrderNHWC
func.func @MemPermuteNWHCOrderNHWC(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x32x16x64xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x32x64xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NHWC,
        mem_perm = #NWHC
    } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x32x16x64xf16, {order = #NHWC}>

    return %1 : tensor<1x32x16x64xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution
    // CHECK-SAME: -> tensor<1x16x32x64xf16, {order = #NCWH}>
    // CHECK-NOT: IE.MemPermute

    // CHECK:   [[PERMUTE_CAST:%.*]] = IE.PermuteCast([[CONV]]) {dst_order = #NHWC, mem_perm = #NCHW}
    // CHECK-SAME:  tensor<1x16x32x64xf16, {order = #NCWH}>
    // CHECK-SAME:  -> tensor<1x32x16x64xf16, {order = #NHWC}>
    // CHECK:   return [[PERMUTE_CAST]] : tensor<1x32x16x64xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MemPermuteNWCHOrderNHWC
func.func @MemPermuteNWCHOrderNHWC(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x64x16x32xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x32x64xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NHWC,
        mem_perm = #NWCH
    } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x64x16x32xf16, {order = #NHWC}>

    return %1 : tensor<1x64x16x32xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution
    // CHECK-SAME: -> tensor<1x16x32x64xf16>
    // CHECK-NOT: IE.MemPermute

    // CHECK:   [[PERMUTE_CAST:%.*]] = IE.PermuteCast([[CONV]]) {dst_order = #NHWC, mem_perm = #NCHW}
    // CHECK-SAME:  tensor<1x16x32x64xf16>
    // CHECK-SAME:  -> tensor<1x64x16x32xf16, {order = #NHWC}>
    // CHECK:   return [[PERMUTE_CAST]] : tensor<1x64x16x32xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @MemPermuteNHCWOrderNHWC
func.func @MemPermuteNHCWOrderNHWC(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x16x64x32xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x32x64xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NHWC,
        mem_perm = #NHCW
    } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x16x64x32xf16, {order = #NHWC}>

    return %1 : tensor<1x16x64x32xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution
    // CHECK-SAME: -> tensor<1x16x32x64xf16, {order = #NWHC}>
    // CHECK-NOT: IE.MemPermute

    // CHECK:   [[PERMUTE_CAST:%.*]] = IE.PermuteCast([[CONV]]) {dst_order = #NHWC, mem_perm = #NCHW}
    // CHECK-SAME:  tensor<1x16x32x64xf16, {order = #NWHC}>
    // CHECK-SAME:  -> tensor<1x16x64x32xf16, {order = #NHWC}>
    // CHECK:   return [[PERMUTE_CAST]] : tensor<1x16x64x32xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MemPermuteNHWCOrderNHWC
func.func @MemPermuteNHWCOrderNHWC(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x32x64x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x32x64xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x32x64x16xf16, {order = #NHWC}>

    return %1 : tensor<1x32x64x16xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution
    // CHECK-SAME: -> tensor<1x16x32x64xf16, {order = #NWCH}>
    // CHECK-NOT: IE.MemPermute

    // CHECK:   [[PERMUTE_CAST:%.*]] = IE.PermuteCast([[CONV]]) {dst_order = #NHWC, mem_perm = #NCHW}
    // CHECK-SAME:  tensor<1x16x32x64xf16, {order = #NWCH}>
    // CHECK-SAME:  -> tensor<1x32x64x16xf16, {order = #NHWC}>
    // CHECK:   return [[PERMUTE_CAST]] : tensor<1x32x64x16xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @MemPermuteNCWHOrderNHWC
func.func @MemPermuteNCWHOrderNHWC(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x64x32x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x32x64xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NHWC,
        mem_perm = #NCWH
    } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x64x32x16xf16, {order = #NHWC}>

    return %1 : tensor<1x64x32x16xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution
    // CHECK-SAME: -> tensor<1x16x32x64xf16, {order = #NHCW}>
    // CHECK-NOT: IE.MemPermute

    // CHECK:   [[PERMUTE_CAST:%.*]] = IE.PermuteCast(%0) {dst_order = #NHWC, mem_perm = #NCHW}
    // CHECK-SAME:  tensor<1x16x32x64xf16, {order = #NHCW}>
    // CHECK-SAME:  -> tensor<1x64x32x16xf16, {order = #NHWC}>
    // CHECK:   return [[PERMUTE_CAST]] : tensor<1x64x32x16xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MultiplyMemPermuteNHWC
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x3x16x16xf16>
func.func @MultiplyMemPermuteNHWC(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x3x16x16xf16>
        = dense<12.000000e+00> : tensor<1x3x16x16xf16>

    %0 = IE.Multiply(%arg0, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>

    %1 = IE.MemPermute(%0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16, {order = #NHWC}>

    return %1 : tensor<1x3x16x16xf16, {order = #NHWC}>

    // CHECK-DAG: %[[VAL_1:.*]] = const.Declare tensor<1x3x16x16xf16> = dense<1.200000e+01> : tensor<1x3x16x16xf16>
    // CHECK:   %[[MUL:.*]] = IE.Multiply(%[[VAL_0]], %[[VAL_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
    // CHECK:   %[[RESULT:.*]] = IE.MemPermute(%[[MUL]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16, {order = #NHWC}>
    // return   %[[RESULT]]

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SubtractMemPermuteNHWC
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x3x16x16xf16>
func.func @SubtractMemPermuteNHWC(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x3x16x16xf16>
        = dense<12.000000e+00> : tensor<1x3x16x16xf16>

    %0 = IE.Subtract(%arg0, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>

    %1 = IE.MemPermute(%0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16, {order = #NHWC}>

    return %1 : tensor<1x3x16x16xf16, {order = #NHWC}>

    // CHECK-DAG: %[[VAL_1:.*]] = const.Declare tensor<1x3x16x16xf16> = dense<1.200000e+01> : tensor<1x3x16x16xf16>
    // CHECK:   %[[MUL:.*]] = IE.Subtract(%[[VAL_0]], %[[VAL_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
    // CHECK:   %[[RESULT:.*]] = IE.MemPermute(%[[MUL]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16, {order = #NHWC}>
    // return   %[[RESULT]]

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AndMemPermuteNHWC
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x3x16x16xf16>
func.func @AndMemPermuteNHWC(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x3x16x16xf16>
        = dense<12.000000e+00> : tensor<1x3x16x16xf16>

    %0 = IE.And(%arg0, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>

    %1 = IE.MemPermute(%0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16, {order = #NHWC}>

    return %1 : tensor<1x3x16x16xf16, {order = #NHWC}>

    // CHECK-DAG: %[[VAL_1:.*]] = const.Declare tensor<1x3x16x16xf16> = dense<1.200000e+01> : tensor<1x3x16x16xf16>
    // CHECK:   %[[MUL:.*]] = IE.And(%[[VAL_0]], %[[VAL_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
    // CHECK:   %[[RESULT:.*]] = IE.MemPermute(%[[MUL]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16, {order = #NHWC}>
    // return   %[[RESULT]]

}


// -----

#HNCW = affine_map<(d0, d1, d2, d3) -> (d2, d0, d1, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PoolMemPermuteDstOrderNHWC
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x3x16x32xf16>
func.func @PoolMemPermuteDstOrderNHWC(%arg0: tensor<1x3x16x32xf16>) -> tensor<16x32x1x3xf16, {order = #HNCW}> {
    %0 = IE.AvgPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x3x16x32xf16> -> tensor<1x3x16x32xf16>

    %1 = IE.MemPermute(%0) {
        dst_order = #HNCW,
        mem_perm = #NHWC
    } : tensor<1x3x16x32xf16> -> tensor<16x32x1x3xf16, {order = #HNCW}>

    return %1 : tensor<16x32x1x3xf16, {order = #HNCW}>

    // CHECK:   [[POOL:%.*]] = IE.AvgPool(%arg0) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x16x32xf16> -> tensor<1x3x16x32xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTE_CAST:%.*]] = IE.PermuteCast([[POOL]]) {dst_order = #map, mem_perm = #NCHW} : tensor<1x3x16x32xf16, {order = #NHWC}> -> tensor<16x32x1x3xf16, {order = #map}>
    // CHECK:   return     [[PERMUTE_CAST]]

}


// -----

#HNCW = affine_map<(d0, d1, d2, d3) -> (d2, d0, d1, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PoolMemPermuteMemPermuteNHWC
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x3x16x32xf16, {order = #NHWC}>
func.func @PoolMemPermuteMemPermuteNHWC(%arg0: tensor<1x3x16x32xf16, {order = #NHWC}>) -> tensor<32x3x1x16xf16, {order = #NHWC}> {
   %0 = IE.AvgPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x3x16x32xf16, {order = #NHWC}> -> tensor<1x3x16x32xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NHWC,
        mem_perm = #HNCW
    } : tensor<1x3x16x32xf16, {order = #NHWC}> -> tensor<32x3x1x16xf16, {order = #NHWC}>

    return %1 : tensor<32x3x1x16xf16, {order = #NHWC}>

    // CHECK:   [[POOL:%.*]] = IE.AvgPool(%arg0) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x16x32xf16, {order = #NHWC}> -> tensor<1x3x16x32xf16, {order = #NWHC}>
    // CHECK:   [[PERMUTE_CAST:%.*]] = IE.PermuteCast([[POOL]]) {dst_order = #NHWC, mem_perm = #map} : tensor<1x3x16x32xf16, {order = #NWHC}> -> tensor<32x3x1x16xf16, {order = #NHWC}>
    // CHECK:   return     [[PERMUTE_CAST]]

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @TransposedConvWithMemPermuteNWHC
func.func @TransposedConvWithMemPermuteNWHC(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x16x129x65xf16> {
    %cst = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x2x2xf16, {order = #NHWC}>

    %0 = IE.TransposedConvolution(%arg0, %cst) {
        dilations = [1, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 0>,
        spatial_output_padding = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [2, 2]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x2x2xf16, {order = #NHWC}>
            -> tensor<1x16x65x129xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NCHW,
        mem_perm = #NWHC
    } : tensor<1x16x65x129xf16, {order = #NHWC}> -> tensor<1x16x129x65xf16>

    return %1 : tensor<1x16x129x65xf16>

    // CHECK:   [[CST:%.*]] = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x2x2xf16, {order = #NHWC}>
    // CHECK:   [[TransposedConv:%.*]] = IE.TransposedConvolution(%arg0, [[CST]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 0, 0>,
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      spatial_output_padding = [1, 1],
    // CHECK-SAME:      strides = [2, 2]
    // CHECK-SAME:  } : tensor<1x16x32x64xf16, {order = #NHWC}>, tensor<16x16x2x2xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x65x129xf16, {order = #NCWH}>

    // CHECK:   [[PERMUTE_CAST:%.*]] = IE.PermuteCast([[TransposedConv]]) {
    // CHECK-SAME:      dst_order = #NCHW, mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x16x65x129xf16, {order = #NCWH}>
    // CHECK-SAME:      -> tensor<1x16x129x65xf16>

    // CHECK:   return [[PERMUTE_CAST]] : tensor<1x16x129x65xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @TransposedConvWithMemPermuteNWCH
func.func @TransposedConvWithMemPermuteNWCH(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x16x65x129xf16> {
    %cst = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x2x2xf16, {order = #NHWC}>

    %0 = IE.TransposedConvolution(%arg0, %cst) {
        dilations = [1, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 0>,
        spatial_output_padding = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [2, 2]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x2x2xf16, {order = #NHWC}>
            -> tensor<1x16x65x129xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NCHW,
        mem_perm = #NWCH
    } : tensor<1x16x65x129xf16, {order = #NHWC}> -> tensor<1x16x65x129xf16>

    return %1 : tensor<1x16x65x129xf16>

    // CHECK:   [[CST:%.*]] = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x2x2xf16, {order = #NHWC}>
    // CHECK:   [[TransposedConv:%.*]] = IE.TransposedConvolution(%arg0, [[CST]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 0, 0>,
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      spatial_output_padding = [1, 1],
    // CHECK-SAME:      strides = [2, 2]
    // CHECK-SAME:  } : tensor<1x16x32x64xf16, {order = #NHWC}>, tensor<16x16x2x2xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x65x129xf16>

    // CHECK:   return [[TransposedConv]] : tensor<1x16x65x129xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @TransposedConvWithMemPermuteNHCW
func.func @TransposedConvWithMemPermuteNHCW(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x129x65x16xf16> {
    %cst = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x2x2xf16, {order = #NHWC}>

    %0 = IE.TransposedConvolution(%arg0, %cst) {
        dilations = [1, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 0>,
        spatial_output_padding = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [2, 2]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x2x2xf16, {order = #NHWC}>
            -> tensor<1x16x65x129xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NCHW,
        mem_perm = #NHCW
    } : tensor<1x16x65x129xf16, {order = #NHWC}> -> tensor<1x129x65x16xf16>

    return %1 : tensor<1x129x65x16xf16>

    // CHECK:   [[CST:%.*]] = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x2x2xf16, {order = #NHWC}>
    // CHECK:   [[TransposedConv:%.*]] = IE.TransposedConvolution(%arg0, [[CST]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 0, 0>,
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      spatial_output_padding = [1, 1],
    // CHECK-SAME:      strides = [2, 2]
    // CHECK-SAME:  } : tensor<1x16x32x64xf16, {order = #NHWC}>, tensor<16x16x2x2xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x65x129xf16, {order = #NWHC}>

    // CHECK:   [[PERMUTE_CAST:%.*]] = IE.PermuteCast([[TransposedConv]]) {
    // CHECK-SAME:      dst_order = #NCHW, mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x16x65x129xf16, {order = #NWHC}>
    // CHECK-SAME:      -> tensor<1x129x65x16xf16>

    // CHECK:   return [[PERMUTE_CAST]] : tensor<1x129x65x16xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TransposedConvWithMemPermuteNHWC
func.func @TransposedConvWithMemPermuteNHWC(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x129x16x65xf16> {
    %cst = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x2x2xf16, {order = #NHWC}>

    %0 = IE.TransposedConvolution(%arg0, %cst) {
        dilations = [1, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 0>,
        spatial_output_padding = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [2, 2]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x2x2xf16, {order = #NHWC}>
            -> tensor<1x16x65x129xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NCHW,
        mem_perm = #NHWC
    } : tensor<1x16x65x129xf16, {order = #NHWC}> -> tensor<1x129x16x65xf16>

    return %1 : tensor<1x129x16x65xf16>

    // CHECK:   [[CST:%.*]] = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x2x2xf16, {order = #NHWC}>
    // CHECK:   [[TransposedConv:%.*]] = IE.TransposedConvolution(%arg0, [[CST]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 0, 0>,
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      spatial_output_padding = [1, 1],
    // CHECK-SAME:      strides = [2, 2]
    // CHECK-SAME:  } : tensor<1x16x32x64xf16, {order = #NHWC}>, tensor<16x16x2x2xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x65x129xf16, {order = #NWCH}>

    // CHECK:   [[PERMUTE_CAST:%.*]] = IE.PermuteCast([[TransposedConv]]) {
    // CHECK-SAME:      dst_order = #NCHW, mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x16x65x129xf16, {order = #NWCH}>
    // CHECK-SAME:      -> tensor<1x129x16x65xf16>

    // CHECK:   return [[PERMUTE_CAST]] : tensor<1x129x16x65xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @TransposedConvWithMemPermuteNCWH
func.func @TransposedConvWithMemPermuteNCWH(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x65x16x129xf16> {
    %cst = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x2x2xf16, {order = #NHWC}>

    %0 = IE.TransposedConvolution(%arg0, %cst) {
        dilations = [1, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 0>,
        spatial_output_padding = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [2, 2]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x2x2xf16, {order = #NHWC}>
            -> tensor<1x16x65x129xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NCHW,
        mem_perm = #NCWH
    } : tensor<1x16x65x129xf16, {order = #NHWC}> -> tensor<1x65x16x129xf16>

    return %1 : tensor<1x65x16x129xf16>

    // CHECK:   [[CST:%.*]] = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x2x2xf16, {order = #NHWC}>
    // CHECK:   [[TransposedConv:%.*]] = IE.TransposedConvolution(%arg0, [[CST]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 0, 0>,
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      spatial_output_padding = [1, 1],
    // CHECK-SAME:      strides = [2, 2]
    // CHECK-SAME:  } : tensor<1x16x32x64xf16, {order = #NHWC}>, tensor<16x16x2x2xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x65x129xf16, {order = #NHCW}>

    // CHECK:   [[PERMUTE_CAST:%.*]] = IE.PermuteCast([[TransposedConv]]) {
    // CHECK-SAME:      dst_order = #NCHW, mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x16x65x129xf16, {order = #NHCW}>
    // CHECK-SAME:      -> tensor<1x65x16x129xf16>

    // CHECK:   return [[PERMUTE_CAST]] : tensor<1x65x16x129xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 5.000000e-01>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @MemPermuteAcrossQuantizeCast
// CHECK-SAME:      [[INPUT:%arg[0-9]]]:  tensor<1x16x64x64xf16, {order = #NHWC}>
func.func @MemPermuteAcrossQuantizeCast(%arg0: tensor<1x16x64x64xf16, {order = #NHWC}>) -> tensor<1x64x64x16x!qElemType, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x64x64xf16, {order = #NHWC}>, tensor<1x16x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64x!qElemType1, {order = #NHWC}>
    %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType} : tensor<1x16x64x64x!qElemType1, {order = #NHWC}> -> tensor<1x16x64x64x!qElemType, {order = #NHWC}>
    %2 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x64x64x!qElemType, {order = #NHWC}> -> tensor<1x64x64x16x!qElemType, {order = #NHWC}>

    return %2 : tensor<1x64x64x16x!qElemType, {order = #NHWC}>

    // CHECK:       [[ADD:%.*]] = IE.Add([[INPUT]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x64x64xf16, {order = #NHWC}>, tensor<1x16x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64x!qElemType1, {order = #NWCH}>
    // CHECK:       [[PERMUTE_CAST:%.*]] = IE.PermuteCast([[ADD]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x64x64x!qElemType1, {order = #NWCH}> -> tensor<1x64x64x16x!qElemType1, {order = #NHWC}>
    // CHECK:       [[QUANTIZE_CAST:%.*]] = IE.QuantizeCast([[PERMUTE_CAST]]) {dstElemType = !qElemType} : tensor<1x64x64x16x!qElemType1, {order = #NHWC}> -> tensor<1x64x64x16x!qElemType, {order = #NHWC}>
    // CHECK:       return [[QUANTIZE_CAST]] : tensor<1x64x64x16x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#WNHC = affine_map<(d0, d1, d2, d3) -> (d3, d0, d2, d1)>

// CHECK-LABEL: @MemPermuteToPoolAffineReshape
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x64x29x1xf16, {order = #NHWC}>
func.func @MemPermuteToPoolAffineReshape(%arg0: tensor<1x64x29x1xf16, {order = #NHWC}>) -> tensor<64x29x1x1xf16, {order = #NHWC}> {
   %0 = IE.MaxPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x29x1xf16, {order = #NHWC}> -> tensor<1x64x29x1xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NHWC,
        mem_perm = #WNHC
    } : tensor<1x64x29x1xf16, {order = #NHWC}> -> tensor<64x29x1x1xf16, {order = #NHWC}>

    return %1 : tensor<64x29x1x1xf16, {order = #NHWC}>

    // CHECK:                 [[POOL:%.+]] = IE.MaxPool([[INPUT]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x64x29x1xf16, {order = #NHWC}> -> tensor<1x64x29x1xf16, {order = #NCWH}>
    // CHECK:                 [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[POOL]])
    // CHECK-SAME{LITERAL}:       {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [64, 29, 1, 1]} : tensor<1x64x29x1xf16, {order = #NCWH}> -> tensor<64x29x1x1xf16, {order = #NHWC}>
    // CHECK:                 return     [[AFFINE_RESHAPE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#WNHC = affine_map<(d0, d1, d2, d3) -> (d3, d0, d2, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @NotMemPermuteToPoolAffineReshape
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x64x29x8xf16, {order = #NHWC}>
func.func @NotMemPermuteToPoolAffineReshape(%arg0: tensor<1x64x29x8xf16, {order = #NHWC}>) -> tensor<64x29x1x8xf16, {order = #NHWC}> {
   %0 = IE.MaxPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x29x8xf16, {order = #NHWC}> -> tensor<1x64x29x8xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NHWC,
        mem_perm = #WNHC
    } : tensor<1x64x29x8xf16, {order = #NHWC}> -> tensor<64x29x1x8xf16, {order = #NHWC}>

    return %1 : tensor<64x29x1x8xf16, {order = #NHWC}>

    // CHECK:                 [[POOL:%.+]] = IE.MaxPool([[INPUT]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x64x29x8xf16, {order = #NHWC}> -> tensor<1x64x29x8xf16, {order = #NCWH}>
    // CHECK:                 [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[POOL]])
    // CHECK-SAME:                {dst_order = #NHWC, mem_perm = #map} : tensor<1x64x29x8xf16, {order = #NCWH}> -> tensor<64x29x1x8xf16, {order = #NHWC}>
    // CHECK:                 return     [[PERMUTE_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @FuseMemPermuteToConvThroughViewOps
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x256x8192x4xf16, {order = #NHWC}>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<4096x256x1x1xf16, {order = #NHWC}>
func.func @FuseMemPermuteToConvThroughViewOps(%arg0: tensor<1x256x8192x4xf16, {order = #NHWC}>, %arg1: tensor<4096x256x1x1xf16, {order = #NHWC}>) -> tensor<1x8x4096x4096xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x256x8192x4xf16, {order = #NHWC}>, tensor<4096x256x1x1xf16, {order = #NHWC}> -> tensor<1x4096x8192x4xf16, {order = #NHWC}>

    %1 = IE.AffineReshape(%0) {
        dim_mapping = [[0], [1], [2], [2, 3]],
        shape_value = [1, 4096, 32768, 1]
    } : tensor<1x4096x8192x4xf16, {order = #NHWC}> -> tensor<1x4096x32768x1xf16, {order = #NHWC}>

    %2 = IE.PermuteCast(%1) {
        dst_order = #NCHW,
        mem_perm = #map1
    } : tensor<1x4096x32768x1xf16, {order = #NHWC}> -> tensor<32768x4096x1x1xf16>

    %3 = IE.AffineReshape(%2) {
        dim_mapping = [[0, 1, 2], [3], [3], [3]],
        shape_value = [1, 4096, 8, 4096]
    } : tensor<32768x4096x1x1xf16> -> tensor<1x4096x8x4096xf16>

    %4 = IE.MemPermute(%3) {
        dst_order = #NCHW,
        mem_perm = #map2} : tensor<1x4096x8x4096xf16> -> tensor<1x8x4096x4096xf16>

    return %4 : tensor<1x8x4096x4096xf16>

    // CHECK:               [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 256, 4096, 8]} inputs([[INPUT_0]] : tensor<1x256x8192x4xf16, {order = #NHWC}>) -> tensor<1x256x4096x8xf16, {order = #NHWC}>
    // CHECK:               [[CONV:%.+]] = IE.Convolution([[SHAPE_CAST]], [[INPUT_1]]) {
    // CHECK-SAME:                  dilations = [1, 1],
    // CHECK-SAME:                  pads_begin = [0, 0],
    // CHECK-SAME:                  pads_end = [0, 0],
    // CHECK-SAME:                  strides = [1, 1]
    // CHECK-SAME:          } : tensor<1x256x4096x8xf16, {order = #NHWC}>, tensor<4096x256x1x1xf16, {order = #NHWC}> -> tensor<1x4096x4096x8xf16, {order = #NWHC}>
    // CHECK:               [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[CONV]]) {
    // CHECK-SAME:                  dst_order = #NCHW,
    // CHECK-SAME:                  mem_perm = #NCHW
    // CHECK-SAME:          } : tensor<1x4096x4096x8xf16, {order = #NWHC}> -> tensor<1x8x4096x4096xf16>

    // CHECK:               return     [[PERMUTE_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d1, d3, d4)>

// CHECK-LABEL: @NotFuseNon4DMemPermuteToConvThroughViewOps
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x256x8192x4xf16, {order = #NHWC}>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<4096x256x1x1xf16, {order = #NHWC}>
func.func @NotFuseNon4DMemPermuteToConvThroughViewOps(%arg0: tensor<1x256x8192x4xf16, {order = #NHWC}>, %arg1: tensor<4096x256x1x1xf16, {order = #NHWC}>) -> tensor<1x2048x2x8x4096xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x256x8192x4xf16, {order = #NHWC}>, tensor<4096x256x1x1xf16, {order = #NHWC}> -> tensor<1x4096x8192x4xf16, {order = #NHWC}>

    %1 = IE.AffineReshape(%0) {
        dim_mapping = [[0], [1], [2], [2, 3]],
        shape_value = [1, 4096, 32768, 1]
    } : tensor<1x4096x8192x4xf16, {order = #NHWC}> -> tensor<1x4096x32768x1xf16, {order = #NHWC}>

    %2 = IE.PermuteCast(%1) {
        dst_order = #NCHW,
        mem_perm = #map1
    } : tensor<1x4096x32768x1xf16, {order = #NHWC}> -> tensor<32768x4096x1x1xf16>

    %3 = IE.AffineReshape(%2) {
        dim_mapping = [[0, 1, 2, 3], [4], [4], [4]],
        shape_value = [1, 2, 2048, 8, 4096]
    } : tensor<32768x4096x1x1xf16> -> tensor<1x2x2048x8x4096xf16>

    %4 = IE.MemPermute(%3) {
        dst_order = #NCDHW,
        mem_perm = #map2} : tensor<1x2x2048x8x4096xf16> -> tensor<1x2048x2x8x4096xf16>

    return %4 : tensor<1x2048x2x8x4096xf16>

    // CHECK:               [[CONV:%.+]] = IE.Convolution([[INPUT_0]], [[INPUT_1]])
    // CHECK:               [[RESHAPE_0:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK:               [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[RESHAPE_0]])
    // CHECK:               [[RESHAPE_1:%.+]] = IE.AffineReshape([[PERMUTE_CAST]])
    // CHECK:               [[MEM_PERMUTE:%.+]] = IE.MemPermute([[RESHAPE_1]])

    // CHECK:               return     [[MEM_PERMUTE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @NotFuseBatchMemPermuteToConvThroughViewOps
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x256x8192x4xf16, {order = #NHWC}>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<4096x256x1x1xf16, {order = #NHWC}>
func.func @NotFuseBatchMemPermuteToConvThroughViewOps(%arg0: tensor<1x256x8192x4xf16, {order = #NHWC}>, %arg1: tensor<4096x256x1x1xf16, {order = #NHWC}>) -> tensor<2x8x2048x4096xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x256x8192x4xf16, {order = #NHWC}>, tensor<4096x256x1x1xf16, {order = #NHWC}> -> tensor<1x4096x8192x4xf16, {order = #NHWC}>

    %1 = IE.AffineReshape(%0) {
        dim_mapping = [[0], [1], [2], [2, 3]],
        shape_value = [1, 4096, 32768, 1]
    } : tensor<1x4096x8192x4xf16, {order = #NHWC}> -> tensor<1x4096x32768x1xf16, {order = #NHWC}>

    %2 = IE.PermuteCast(%1) {
        dst_order = #NCHW,
        mem_perm = #map1
    } : tensor<1x4096x32768x1xf16, {order = #NHWC}> -> tensor<32768x4096x1x1xf16>

    %3 = IE.AffineReshape(%2) {
        dim_mapping = [[0, 1, 2], [3], [3], [3]],
        shape_value = [2, 2048, 8, 4096]
    } : tensor<32768x4096x1x1xf16> -> tensor<2x2048x8x4096xf16>

    %4 = IE.MemPermute(%3) {
        dst_order = #NCHW,
        mem_perm = #map2} : tensor<2x2048x8x4096xf16> -> tensor<2x8x2048x4096xf16>

    return %4 : tensor<2x8x2048x4096xf16>

    // CHECK:               [[CONV:%.+]] = IE.Convolution([[INPUT_0]], [[INPUT_1]])
    // CHECK:               [[RESHAPE_0:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK:               [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[RESHAPE_0]])
    // CHECK:               [[RESHAPE_1:%.+]] = IE.AffineReshape([[PERMUTE_CAST]])
    // CHECK:               [[MEM_PERMUTE:%.+]] = IE.MemPermute([[RESHAPE_1]])

    // CHECK:               return     [[MEM_PERMUTE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @FuseMemPermuteToConvThroughViewOpsAndSliceOp
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x16x262144x4xf16, {order = #NHWC}>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<16x16x1x1xf16, {order = #NHWC}>
func.func @FuseMemPermuteToConvThroughViewOpsAndSliceOp(%arg0: tensor<1x16x262144x4xf16, {order = #NHWC}>, %arg1: tensor<16x16x1x1xf16, {order = #NHWC}>) -> tensor<1x4096x7x256xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x262144x4xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x262144x4xf16, {order = #NHWC}>

    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 7, 262144, 4] : tensor<1x16x262144x4xf16, {order = #NHWC}> to tensor<1x7x262144x4xf16, {order = #NHWC}>

    %2 = IE.AffineReshape(%1) {
        dim_mapping = [[0], [1], [2], [2, 3]],
        shape_value = [1, 7, 1048576, 1]
    } : tensor<1x7x262144x4xf16, {order = #NHWC}> -> tensor<1x7x1048576x1xf16, {order = #NHWC}>

    %3 = IE.PermuteCast(%2) {dst_order = #NCHW, mem_perm = #map1} : tensor<1x7x1048576x1xf16, {order = #NHWC}> -> tensor<1048576x7x1x1xf16>

    %4 = IE.AffineReshape(%3) {
        dim_mapping = [[0, 1, 2], [3], [3], [3]],
        shape_value = [1, 4096, 256, 7]
    } : tensor<1048576x7x1x1xf16> -> tensor<1x4096x256x7xf16>

    %5 = IE.MemPermute(%4) {dst_order = #NCHW, mem_perm = #map2} : tensor<1x4096x256x7xf16> -> tensor<1x4096x7x256xf16>

    return %5 : tensor<1x4096x7x256xf16>

    // CHECK:               [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 16, 4096, 256]} inputs([[INPUT_0]] : tensor<1x16x262144x4xf16, {order = #NHWC}>) -> tensor<1x16x4096x256xf16, {order = #NHWC}>
    // CHECK:               [[CONV:%.+]] = IE.Convolution([[SHAPE_CAST]], [[INPUT_1]]) {
    // CHECK-SAME:                  dilations = [1, 1],
    // CHECK-SAME:                  pads_begin = [0, 0],
    // CHECK-SAME:                  pads_end = [0, 0],
    // CHECK-SAME:                  strides = [1, 1]
    // CHECK-SAME:          } : tensor<1x16x4096x256xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x4096x256xf16, {order = #NHCW}>
    // CHECK:               [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[CONV]]) {
    // CHECK-SAME:                  dst_order = #NCHW,
    // CHECK-SAME:                  mem_perm = #NCHW
    // CHECK-SAME:          } : tensor<1x16x4096x256xf16, {order = #NHCW}> -> tensor<1x4096x16x256xf16>
    // CHECK:               [[SLICE:%.+]] = IE.Slice [[PERMUTE_CAST]] [0, 0, 0, 0] [1, 4096, 7, 256] : tensor<1x4096x16x256xf16> to tensor<1x4096x7x256xf16>

    // CHECK:               return     [[SLICE]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @MemPermuteInterpolate
func.func @MemPermuteInterpolate(%arg0: tensor<1x16x5x5xf16, {order = #NHWC}>) -> tensor<1x16x10x10xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [60, 60]
         } : tensor<1x16x5x5xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NCHW,
        mem_perm = #NWHC
    } : tensor<1x16x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16>

    return %1 : tensor<1x16x10x10xf16>

    // CHECK:   [[INTERPOLATE:%.+]] = IE.Interpolate
    // CHECK-SAME: -> tensor<1x16x10x10xf16, {order = #NCWH}>
    // CHECK-NOT: IE.MemPermute

    // CHECK:   [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[INTERPOLATE]]) {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME:  tensor<1x16x10x10xf16, {order = #NCWH}>
    // CHECK-SAME:  -> tensor<1x16x10x10xf16>
    // CHECK:   return [[PERMUTE_CAST]] : tensor<1x16x10x10xf16>
}
