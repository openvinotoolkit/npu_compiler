//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-reorders %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithSubView
module @ReorderWithSubView {

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x8x4x2xf16>)
func.func @main(%arg0: tensor<1x8x4x2xf16>) -> tensor<1x4x4x2xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x8x4x2xf16> -> tensor<1x8x4x2xf16, {order = #NHWC}>
    %1 = IE.Slice %0 [0, 2, 0, 0] [1, 4, 4, 2] : tensor<1x8x4x2xf16, {order = #NHWC}> to tensor<1x4x4x2xf16, {order = #NHWC}>
    %2 = IE.Reorder(%1) {dstOrder = #NCHW} : tensor<1x4x4x2xf16, {order = #NHWC}> -> tensor<1x4x4x2xf16>
    return %2 : tensor<1x4x4x2xf16>

    // CHECK:       [[VAR0:%.+]] = IE.Slice [[ARG0]]
    // CHECK-SAME:      tensor<1x8x4x2xf16> to tensor<1x4x4x2xf16>
    // CHECK:       return [[VAR0]] : tensor<1x4x4x2xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithTwoUsersSubView
module @ReorderWithTwoUsersSubView {

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x8x4x2xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x8x4x2xf16, {order = #NHWC}>) -> (tensor<1x4x4x2xf16, {order = #NHWC}>, tensor<1x5x4x2xf16, {order = #NHWC}>) {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x8x4x2xf16, {order = #NHWC}> -> tensor<1x8x4x2xf16>
    %1 = IE.Slice %0 [0, 2, 0, 0] [1, 4, 4, 2] : tensor<1x8x4x2xf16> to tensor<1x4x4x2xf16>
    %2 = IE.Slice %0 [0, 2, 0, 0] [1, 5, 4, 2] : tensor<1x8x4x2xf16> to tensor<1x5x4x2xf16>
    %3 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x4x4x2xf16> -> tensor<1x4x4x2xf16, {order = #NHWC}>
    %4 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<1x5x4x2xf16> -> tensor<1x5x4x2xf16, {order = #NHWC}>
    return %3, %4 : tensor<1x4x4x2xf16, {order = #NHWC}>, tensor<1x5x4x2xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%.+]] = IE.Slice [[ARG0]]
    // CHECK-SAME:      tensor<1x8x4x2xf16, {order = #NHWC}> to tensor<1x4x4x2xf16, {order = #NHWC}>
    // CHECK:       [[VAR1:%.+]] = IE.Slice [[ARG0]]
    // CHECK-SAME:      tensor<1x8x4x2xf16, {order = #NHWC}> to tensor<1x5x4x2xf16, {order = #NHWC}>
    // CHECK:       return [[VAR0]], [[VAR1]] : tensor<1x4x4x2xf16, {order = #NHWC}>, tensor<1x5x4x2xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d1, d4)>

// CHECK-LABEL: @ReorderWithSlicesSubViewNoSwap
module @ReorderWithSlicesSubViewNoSwap {

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x3x9x16x438xf16, {order = #map}>)
func.func @main(%arg0: tensor<1x3x9x16x438xf16, {order = #map}>) -> tensor<1x3x9x7008xf16, {order = #NCHW}> {

    %0 = IE.Reorder(%arg0) {dstOrder = #NCDHW} : tensor<1x3x9x16x438xf16, {order = #map}> -> tensor<1x3x9x16x438xf16>

    %1 = IE.Slice %0 [0, 0, 0, 0, 0] [1, 3, 9, 16, 1] : tensor<1x3x9x16x438xf16> to tensor<1x3x9x16x1xf16>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [2], [3], [3]], shape_value = [1, 3, 9, 16]} : tensor<1x3x9x16x1xf16> -> tensor<1x3x9x16xf16, {order = #NCHW}>
    %3 = IE.Sigmoid(%2) : tensor<1x3x9x16xf16, {order = #NCHW}> -> tensor<1x3x9x16xf16, {order = #NCHW}>

    %4 = IE.Slice %0 [0, 0, 0, 0, 1] [1, 3, 9, 16, 436] : tensor<1x3x9x16x438xf16> to tensor<1x3x9x16x436xf16>
    %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [1], [2], [3], [3]], shape_value = [1, 3, 9, 6976]} : tensor<1x3x9x16x436xf16> -> tensor<1x3x9x6976xf16, {order = #NCHW}>

    %6 = IE.Slice %0 [0, 0, 0, 0, 437] [1, 3, 9, 16, 1] : tensor<1x3x9x16x438xf16> to tensor<1x3x9x16x1xf16>
    %7 = IE.AffineReshape(%6) {dim_mapping = [[0], [1], [2], [3], [3]], shape_value = [1, 3, 9, 16]} : tensor<1x3x9x16x1xf16> -> tensor<1x3x9x16xf16, {order = #NCHW}>
    %8 = IE.Exp(%7) : tensor<1x3x9x16xf16, {order = #NCHW}> -> tensor<1x3x9x16xf16, {order = #NCHW}>

    %9 = IE.Concat(%3, %5, %8) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 16], [0, 0, 0, 6992]]} : tensor<1x3x9x16xf16, {order = #NCHW}>, tensor<1x3x9x6976xf16, {order = #NCHW}>, tensor<1x3x9x16xf16, {order = #NCHW}> -> tensor<1x3x9x7008xf16, {order = #NCHW}>

    return %9 : tensor<1x3x9x7008xf16, {order = #NCHW}>

    // CHECK:       [[VAR0:%.*]] = IE.Reorder([[ARG0]])
    // CHECK-SAME:      tensor<1x3x9x16x438xf16, {order = #map}> -> tensor<1x3x9x16x438xf16>

    // CHECK:       [[VAR1:%.*]] = IE.Slice [[VAR0]]
    // CHECK-SAME:      tensor<1x3x9x16x438xf16> to tensor<1x3x9x16x1xf16>
    // CHECK:       [[VAR2:%.*]] = IE.AffineReshape([[VAR1]])
    // CHECK-SAME:      tensor<1x3x9x16x1xf16> -> tensor<1x3x9x16xf16, {order = #NCHW}>
    // CHECK:       [[VAR3:%.*]] = IE.Sigmoid([[VAR2]])
    // CHECK-SAME:      tensor<1x3x9x16xf16, {order = #NCHW}> -> tensor<1x3x9x16xf16, {order = #NCHW}>

    // CHECK:       [[VAR4:%.*]] = IE.Slice [[VAR0]]
    // CHECK-SAME:      tensor<1x3x9x16x438xf16> to tensor<1x3x9x16x436xf16>
    // CHECK:       [[VAR5:%.*]] = IE.AffineReshape([[VAR4]])
    // CHECK-SAME:      tensor<1x3x9x16x436xf16> -> tensor<1x3x9x6976xf16, {order = #NCHW}>

    // CHECK:       [[VAR6:%.*]] = IE.Slice [[VAR0]]
    // CHECK-SAME:      tensor<1x3x9x16x438xf16> to tensor<1x3x9x16x1xf16>
    // CHECK:       [[VAR7:%.*]] = IE.AffineReshape([[VAR6]])
    // CHECK-SAME:      tensor<1x3x9x16x1xf16> -> tensor<1x3x9x16xf16, {order = #NCHW}>
    // CHECK:       [[VAR8:%.*]] = IE.Exp([[VAR7]])
    // CHECK-SAME:      tensor<1x3x9x16xf16, {order = #NCHW}> -> tensor<1x3x9x16xf16, {order = #NCHW}>

    // CHECK:       [[VAR9:%.*]] = IE.Concat([[VAR3]], [[VAR5]], [[VAR8]])
    // CHECK-SAME:            tensor<1x3x9x16xf16, {order = #NCHW}>, tensor<1x3x9x6976xf16, {order = #NCHW}>, tensor<1x3x9x16xf16, {order = #NCHW}>
    // CHECK-SAME:            -> tensor<1x3x9x7008xf16, {order = #NCHW}>

    // CHECK:       return [[VAR9]] : tensor<1x3x9x7008xf16, {order = #NCHW}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d1, d4)>

// CHECK-LABEL: @ReorderWithSlicesSubViewSwap
module @ReorderWithSlicesSubViewSwap {

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x3x9x16x438xf16, {order = #map}>)
func.func @main(%arg0: tensor<1x3x9x16x438xf16, {order = #map}>) -> tensor<1x3x144x438xf16, {order = #NHWC}> {

    %0 = IE.Reorder(%arg0) {dstOrder = #NCDHW} : tensor<1x3x9x16x438xf16, {order = #map}> -> tensor<1x3x9x16x438xf16>

    %1 = IE.Slice %0 [0, 0, 0, 0, 0] [1, 3, 9, 16, 1] : tensor<1x3x9x16x438xf16> to tensor<1x3x9x16x1xf16>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [2], [3], [3]], shape_value = [1, 3, 9, 16]} : tensor<1x3x9x16x1xf16> -> tensor<1x3x9x16xf16, {order = #NCHW}>
    %3 = IE.Sigmoid(%2) : tensor<1x3x9x16xf16, {order = #NCHW}> -> tensor<1x3x9x16xf16, {order = #NCHW}>
    %4 = IE.AffineReshape(%3) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 3, 144, 1]} : tensor<1x3x9x16xf16, {order = #NCHW}> -> tensor<1x3x144x1xf16, {order = #NCHW}>
    %5 = IE.Reorder(%4) {dstOrder = #NHWC} : tensor<1x3x144x1xf16, {order = #NCHW}> -> tensor<1x3x144x1xf16, {order = #NHWC}>

    %6 = IE.Slice %0 [0, 0, 0, 0, 1] [1, 3, 9, 16, 437] : tensor<1x3x9x16x438xf16> to tensor<1x3x9x16x437xf16>
    %7 = IE.AffineReshape(%6) {dim_mapping = [[0], [1], [2], [2], [3]], shape_value = [1, 3, 144, 437]} : tensor<1x3x9x16x437xf16> -> tensor<1x3x144x437xf16, {order = #NCHW}>
    %8 = IE.Reorder(%7) {dstOrder = #NHWC} : tensor<1x3x144x437xf16, {order = #NCHW}> -> tensor<1x3x144x437xf16, {order = #NHWC}>

    %9 = IE.Concat(%5, %8) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1]]} : tensor<1x3x144x1xf16, {order = #NHWC}>, tensor<1x3x144x437xf16, {order = #NHWC}> -> tensor<1x3x144x438xf16, {order = #NHWC}>

    return %9 : tensor<1x3x144x438xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%.*]] = IE.Slice [[ARG0]]
    // CHECK-SAME:      tensor<1x3x9x16x438xf16, {order = #map}> to tensor<1x3x9x16x1xf16, {order = #map}>
    // CHECK:       [[VAR1:%.*]] = IE.Reorder([[VAR0]])
    // CHECK-SAME:      tensor<1x3x9x16x1xf16, {order = #map}> -> tensor<1x3x9x16x1xf16>
    // CHECK:       [[VAR2:%.*]] = IE.AffineReshape([[VAR1]])
    // CHECK-SAME:      tensor<1x3x9x16x1xf16> -> tensor<1x3x9x16xf16, {order = #NCHW}>
    // CHECK:       [[VAR3:%.*]] = IE.Sigmoid([[VAR2]])
    // CHECK-SAME:      tensor<1x3x9x16xf16, {order = #NCHW}> -> tensor<1x3x9x16xf16, {order = #NCHW}>
    // CHECK:       [[VAR4:%.*]] = IE.AffineReshape([[VAR3]])
    // CHECK-SAME:      tensor<1x3x9x16xf16, {order = #NCHW}> -> tensor<1x3x144x1xf16, {order = #NCHW}>
    // CHECK:       [[VAR5:%.*]] = IE.Reorder([[VAR4]])
    // CHECK-SAME:      tensor<1x3x144x1xf16, {order = #NCHW}> -> tensor<1x3x144x1xf16, {order = #NHWC}>

    // CHECK:       [[VAR6:%.*]] = IE.Slice [[ARG0]]
    // CHECK-SAME:      tensor<1x3x9x16x438xf16, {order = #map}> to tensor<1x3x9x16x437xf16, {order = #map}>
    // CHECK:       [[VAR7:%.*]] = IE.AffineReshape([[VAR6]])
    // CHECK-SAME:      tensor<1x3x9x16x437xf16, {order = #map}> -> tensor<1x3x144x437xf16, {order = #NHCW}>
    // CHECK:       [[VAR8:%.*]] = IE.Reorder([[VAR7]])
    // CHECK-SAME:      tensor<1x3x144x437xf16, {order = #NHCW}> -> tensor<1x3x144x437xf16, {order = #NHWC}>

    // CHECK:       [[VAR9:%.*]] = IE.Concat([[VAR5]], [[VAR8]])
    // CHECK-SAME:            tensor<1x3x144x1xf16, {order = #NHWC}>, tensor<1x3x144x437xf16, {order = #NHWC}>
    // CHECK-SAME:            -> tensor<1x3x144x438xf16, {order = #NHWC}>

    // CHECK:       return [[VAR9]] : tensor<1x3x144x438xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithExpandNeedSwap
module @ReorderWithExpandNeedSwap {

// CHECK:       func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x5x512x512xf16>)
func.func @main(%arg0: tensor<1x5x512x512xf16>) -> tensor<1x16x512x512xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %1 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x5x512x512xf16> -> tensor<1x5x512x512xf16, {order = #NHWC}>
    %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 11, 0, 0]} : tensor<1x5x512x512xf16, {order = #NHWC}> -> tensor<1x16x512x512xf16, {order = #NHWC}>
    %3 = IE.Convolution(%2, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Relu<>, strides = [1, 1]} : tensor<1x16x512x512xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x512x512xf16, {order = #NHWC}>

    return %3 : tensor<1x16x512x512xf16, {order = #NHWC}>
    // CHECK-DAG:       [[VAR0:%.*]] = const.Declare
    // CHECK-SAME:      tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]

    // CHECK:       [[VAR1:%.*]] = IE.Expand(%arg0)
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0]
    // CHECK-SAME:      pads_end = [0, 11, 0, 0]
    // CHECK-SAME:      : tensor<1x5x512x512xf16> -> tensor<1x16x512x512xf16>

    // CHECK:       [[VAR2:%.*]] = IE.Reorder([[VAR1]])
    // CHECK-SAME:      tensor<1x16x512x512xf16> -> tensor<1x16x512x512xf16, {order = #NHWC}>

    // CHECK:       [[VAR3:%.*]] = IE.Convolution([[VAR2]], [[VAR0]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      post_op = #IE.Relu<>
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x16x512x512xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x512x512xf16, {order = #NHWC}>

    // CHECK:       return [[VAR3]] : tensor<1x16x512x512xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithExpandAndSliceNeedSwap
module @ReorderWithExpandAndSliceNeedSwap {

// CHECK:       func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x1x1x32032xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x1x1x32032xf16, {order = #NHWC}>) -> tensor<1x16x1x31995xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>} : tensor<1x1x1x32032xf16, {order = #NHWC}> -> tensor<1x1x1x32032xf16>
    %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x32032xf16> -> tensor<1x16x1x32032xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 16, 1, 31995] : tensor<1x16x1x32032xf16> to tensor<1x16x1x31995xf16>
    %3 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<1x16x1x31995xf16> -> tensor<1x16x1x31995xf16, {order = #NHWC}>

    return %3 : tensor<1x16x1x31995xf16, {order = #NHWC}>

    // CHECK:       [[EXPAND:%.*]] = IE.Expand(%arg0)
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0]
    // CHECK-SAME:      pads_end = [0, 15, 0, 0]
    // CHECK-SAME:      : tensor<1x1x1x32032xf16, {order = #NHWC}> -> tensor<1x16x1x32032xf16, {order = #NHWC}>

    // CHECK:       [[SLICE:%.*]] = IE.Slice [[EXPAND]] [0, 0, 0, 0] [1, 16, 1, 31995] :
    // CHECK-SAME:      tensor<1x16x1x32032xf16, {order = #NHWC}> to tensor<1x16x1x31995xf16, {order = #NHWC}>

    // CHECK:      return [[SLICE]] : tensor<1x16x1x31995xf16, {order = #NHWC}>

}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithExpandAndMultiSliceNeedSwap
module @ReorderWithExpandAndMultiSliceNeedSwap {

// CHECK:       func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x1x1x32032xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x1x1x32032xf16, {order = #NHWC}>) -> (tensor<1x16x1x31995xf16, {order = #NHWC}>, tensor<1x16x1x31995xf16, {order = #NHWC}>) {
    %0 = IE.Reorder(%arg0) {dstOrder = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>} : tensor<1x1x1x32032xf16, {order = #NHWC}> -> tensor<1x1x1x32032xf16>
    %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x32032xf16> -> tensor<1x16x1x32032xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 16, 1, 31995] : tensor<1x16x1x32032xf16> to tensor<1x16x1x31995xf16>
    %3 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<1x16x1x31995xf16> -> tensor<1x16x1x31995xf16, {order = #NHWC}>
    %4 = IE.Slice %1 [0, 0, 0, 11] [1, 16, 1, 31995] : tensor<1x16x1x32032xf16> to tensor<1x16x1x31995xf16>
    %5 = IE.Reorder(%4) {dstOrder = #NHWC} : tensor<1x16x1x31995xf16> -> tensor<1x16x1x31995xf16, {order = #NHWC}>

    return %3, %5 : tensor<1x16x1x31995xf16, {order = #NHWC}>, tensor<1x16x1x31995xf16, {order = #NHWC}>

    // CHECK:       [[EXPAND:%.*]] = IE.Expand(%arg0)
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0]
    // CHECK-SAME:      pads_end = [0, 15, 0, 0]
    // CHECK-SAME:      : tensor<1x1x1x32032xf16, {order = #NHWC}> -> tensor<1x16x1x32032xf16, {order = #NHWC}>

    // CHECK:       [[SLICE_0:%.*]] = IE.Slice [[EXPAND]] [0, 0, 0, 0] [1, 16, 1, 31995] :
    // CHECK-SAME:      tensor<1x16x1x32032xf16, {order = #NHWC}> to tensor<1x16x1x31995xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_1:%.*]] = IE.Slice [[EXPAND]] [0, 0, 0, 11] [1, 16, 1, 31995] :
    // CHECK-SAME:      tensor<1x16x1x32032xf16, {order = #NHWC}> to tensor<1x16x1x31995xf16, {order = #NHWC}>

    // CHECK:      return [[SLICE_0]], [[SLICE_1]] : tensor<1x16x1x31995xf16, {order = #NHWC}>, tensor<1x16x1x31995xf16, {order = #NHWC}>

}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithExpandNoSwap
module @ReorderWithExpandNoSwap {

// CHECK:       func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x1x512x512xf16>)
func.func @main(%arg0: tensor<1x1x512x512xf16>) -> tensor<1x16x512x512xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %1 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x1x512x512xf16> -> tensor<1x1x512x512xf16, {order = #NHWC}>
    %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x512x512xf16, {order = #NHWC}> -> tensor<1x16x512x512xf16, {order = #NHWC}>
    %3 = IE.Convolution(%2, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Relu<>, strides = [1, 1]} : tensor<1x16x512x512xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x512x512xf16, {order = #NHWC}>

    return %3 : tensor<1x16x512x512xf16, {order = #NHWC}>
    // CHECK-DAG:       [[VAR0:%.*]] = const.Declare
    // CHECK-SAME:      tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]

    // CHECK:       [[VAR1:%.*]] = IE.Reorder(%arg0)
    // CHECK-SAME:      tensor<1x1x512x512xf16> -> tensor<1x1x512x512xf16, {order = #NHWC}>

    // CHECK:       [[VAR2:%.*]] = IE.Expand([[VAR1]])
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0]
    // CHECK-SAME:      pads_end = [0, 15, 0, 0]
    // CHECK-SAME:      : tensor<1x1x512x512xf16, {order = #NHWC}> -> tensor<1x16x512x512xf16, {order = #NHWC}>

    // CHECK:       [[VAR3:%.*]] = IE.Convolution([[VAR2]], [[VAR0]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      post_op = #IE.Relu<>
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x16x512x512xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x512x512xf16, {order = #NHWC}>

    // CHECK:       return [[VAR3]] : tensor<1x16x512x512xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithExpand
module @ReorderWithExpand {

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x3x30x30xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x3x15x13xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x3x30x30xf16>

    %1 = IE.Expand(%0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x30x30xf16> -> tensor<1x16x30x30xf16>

    %2 = IE.MaxPool(%1) {
        kernel_size = [5, 5],
        pads_begin = [2, 0],
        pads_end = [2, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [2, 2]
    } : tensor<1x16x30x30xf16> -> tensor<1x16x15x13xf16>

    %3 = IE.Slice %2 [0, 0, 0, 0] [1, 3, 15, 13] : tensor<1x16x15x13xf16> to tensor<1x3x15x13xf16>

    %4 = IE.Reorder(%3) {dstOrder = #NHWC} : tensor<1x3x15x13xf16> -> tensor<1x3x15x13xf16, {order = #NHWC}>

    return %4 : tensor<1x3x15x13xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%.+]] = IE.Expand([[ARG0]]
    // CHECK-SAME:      tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x16x30x30xf16, {order = #NHWC}>

    // CHECK:       [[VAR1:%.+]] = IE.MaxPool([[VAR0]])
    // CHECK-SAME:      tensor<1x16x30x30xf16, {order = #NHWC}> -> tensor<1x16x15x13xf16, {order = #NHWC}>

    // CHECK:       [[VAR2:%.+]] = IE.Slice [[VAR1]]
    // CHECK-SAME:      tensor<1x16x15x13xf16, {order = #NHWC}> to tensor<1x3x15x13xf16, {order = #NHWC}>

    // CHECK:       return [[VAR2]] : tensor<1x3x15x13xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8<0:254>:f16:1, {5.0750492125984249E-4:127, 0.0013264333169291339:127,9.8713551919291337E-4:127}>
!qElemType1 = !quant.uniform<u8<0:254>:f16:1, {5.0750492125984249E-4:127, 0.0013264333169291339:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127}>

module @ReorderWithQuantExpandAndSlice {

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x3x30x30x!qElemType>)
func.func @main(%arg0: tensor<1x3x30x30x!qElemType>) -> tensor<1x3x15x13x!qElemType> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x3x30x30x!qElemType> -> tensor<1x3x30x30x!qElemType, {order = #NHWC}>

    %1 = IE.Expand(%0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x30x30x!qElemType, {order = #NHWC}> -> tensor<1x16x30x30x!qElemType1, {order = #NHWC}>

    %2 = IE.MaxPool(%1) {
        kernel_size = [5, 5],
        pads_begin = [2, 0],
        pads_end = [2, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [2, 2]
    } : tensor<1x16x30x30x!qElemType1, {order = #NHWC}> -> tensor<1x16x15x13x!qElemType1, {order = #NHWC}>

    %3 = IE.Reorder(%2) {dstOrder = #NCHW} : tensor<1x16x15x13x!qElemType1, {order = #NHWC}> -> tensor<1x16x15x13x!qElemType1>

    %4 = IE.Slice %3 [0, 0, 0, 0] [1, 3, 15, 13] : tensor<1x16x15x13x!qElemType1> to tensor<1x3x15x13x!qElemType>

    return %4 : tensor<1x3x15x13x!qElemType>

    // CHECK: [[VAR0:%.+]] = IE.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} :
    // CHECK-SAME:     tensor<1x3x30x30x!qElemType> ->
    // CHECK-SAME:     tensor<1x16x30x30x!qElemType1>

    // CHECK: [[VAR1:%.+]] = IE.Reorder([[VAR0]]) {dstOrder = #NHWC} :
    // CHECK-SAME:     tensor<1x16x30x30x!qElemType1> ->
    // CHECK-SAME:     tensor<1x16x30x30x!qElemType1, {order = #NHWC}>

    // CHECK: [[VAR2:%.+]] = IE.MaxPool([[VAR1]])
    // CHECK-SAME:     tensor<1x16x30x30x!qElemType1, {order = #NHWC}> -> tensor<1x16x15x13x!qElemType1, {order = #NHWC}>

    // CHECK: [[VAR3:%.+]] = IE.Slice [[VAR2]] [0, 0, 0, 0] [1, 3, 15, 13] :
    // CHECK-SAME:     tensor<1x16x15x13x!qElemType1, {order = #NHWC}> to
    // CHECK-SAME:     tensor<1x3x15x13x!qElemType, {order = #NHWC}>

    // CHECK: [[VAR4:%.+]] = IE.Reorder([[VAR3]]) {dstOrder = #NCHW} :
    // CHECK-SAME:     tensor<1x3x15x13x!qElemType, {order = #NHWC}> ->
    // CHECK-SAME:     tensor<1x3x15x13x!qElemType>

    // CHECK: return [[VAR4]] : tensor<1x3x15x13x!qElemType>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithSplit
module @ReorderWithSplit {

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x3x30x30xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) ->
        (tensor<1x1x30x30xf16, {order = #NHWC}>, tensor<1x1x30x30xf16, {order = #NHWC}>, tensor<1x1x30x30xf16, {order = #NHWC}>){
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x3x30x30xf16>

    %1:3 = IE.Split(%0) {axis_value = 1, num_splits = 3} :
        tensor<1x3x30x30xf16> -> tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>

    %2 = IE.Reorder(%1#0) {dstOrder = #NHWC} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30xf16, {order = #NHWC}>
    %3 = IE.Reorder(%1#1) {dstOrder = #NHWC} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30xf16, {order = #NHWC}>
    %4 = IE.Reorder(%1#2) {dstOrder = #NHWC} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30xf16, {order = #NHWC}>

    return %2, %3, %4 : tensor<1x1x30x30xf16, {order = #NHWC}>, tensor<1x1x30x30xf16, {order = #NHWC}>, tensor<1x1x30x30xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%[0-9]+]]:3 = IE.Split([[ARG0]])
    // CHECK-SAME:      tensor<1x3x30x30xf16, {order = #NHWC}> ->
    // CHECK-SAME:          tensor<1x1x30x30xf16, {order = #NHWC}>,
    // CHECK-SAME:          tensor<1x1x30x30xf16, {order = #NHWC}>,
    // CHECK-SAME:          tensor<1x1x30x30xf16, {order = #NHWC}>

    // CHECK:       return [[VAR0]]#0, [[VAR0]]#1, [[VAR0]]#2
    // CHECK-SAME:      tensor<1x1x30x30xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x30x30xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x30x30xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithSplitMultipleUses
module @ReorderWithSplitMultipleUses {

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x3x30x30xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) ->
        (tensor<1x1x30x30xf16, {order = #NHWC}>, tensor<1x1x30x30xf16, {order = #NHWC}>){
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x3x30x30xf16>

    %1:3 = IE.Split(%0) {axis_value = 1, num_splits = 3} :
        tensor<1x3x30x30xf16> -> tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>

    %2 = IE.Reorder(%1#1) {dstOrder = #NHWC} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30xf16, {order = #NHWC}>
    %3 = IE.Reorder(%1#1) {dstOrder = #NHWC} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30xf16, {order = #NHWC}>

    return %2, %3 : tensor<1x1x30x30xf16, {order = #NHWC}>, tensor<1x1x30x30xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%[0-9]+]]:3 = IE.Split([[ARG0]])
    // CHECK-SAME:      tensor<1x3x30x30xf16, {order = #NHWC}> ->
    // CHECK-SAME:          tensor<1x1x30x30xf16, {order = #NHWC}>,
    // CHECK-SAME:          tensor<1x1x30x30xf16, {order = #NHWC}>,
    // CHECK-SAME:          tensor<1x1x30x30xf16, {order = #NHWC}>

    // CHECK:       return [[VAR0]]#1, [[VAR0]]#1
    // CHECK-SAME:      tensor<1x1x30x30xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x30x30xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithConcat
module @ReorderWithConcat {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<1x1x30x30xf16, {order = #NHWC}>,
// CHECK-SAME:      [[ARG1:%arg[0-9]+]]: tensor<1x1x30x30xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x1x30x30xf16, {order = #NHWC}>, %arg1: tensor<1x1x30x30xf16, {order = #NHWC}>)
        -> tensor<1x2x30x30xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x1x30x30xf16, {order = #NHWC}> -> tensor<1x1x30x30xf16>
    %1 = IE.Reorder(%arg1) {dstOrder = #NCHW} : tensor<1x1x30x30xf16, {order = #NHWC}> -> tensor<1x1x30x30xf16>
    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 1>} : tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x2x30x30xf16>
    %3 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<1x2x30x30xf16> -> tensor<1x2x30x30xf16, {order = #NHWC}>
    return %3 : tensor<1x2x30x30xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%.+]] = IE.Concat([[ARG0]], [[ARG1]]) {per_axis = #IE.Concat<axis = 1 : i64>}
    // CHECK-SAME:      tensor<1x1x30x30xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x30x30xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x2x30x30xf16, {order = #NHWC}>
    // CHECK:       return [[VAR0]] : tensor<1x2x30x30xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithConcatWithConsts
module @ReorderWithConcatWithConsts {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<1x1x30x30xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x1x30x30xf16, {order = #NHWC}>)
        -> tensor<1x2x30x30xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x1x30x30xf16, {order = #NHWC}> -> tensor<1x1x30x30xf16>
    %1 = const.Declare tensor<1x1x30x30xf16> = dense<1.0> : tensor<1x1x30x30xf16>
    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 1>} : tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x2x30x30xf16>
    %3 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<1x2x30x30xf16> -> tensor<1x2x30x30xf16, {order = #NHWC}>
    return %3 : tensor<1x2x30x30xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST0:%.*]] = const.Declare
    // CHECK-SAME:      #const.Reorder<#NHWC>
    // CHECK:       [[VAR0:%.+]] = IE.Concat([[ARG0]], [[CST0]]) {per_axis = #IE.Concat<axis = 1 : i64>}
    // CHECK-SAME:      tensor<1x1x30x30xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x30x30xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x2x30x30xf16, {order = #NHWC}>
    // CHECK:       return [[VAR0]] : tensor<1x2x30x30xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithConcatWithArgs
module @ReorderWithConcatWithArgs {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<1x1x30x30xf16, {order = #NHWC}>,
// CHECK-SAME:      [[ARG1:%arg[0-9]+]]: tensor<1x3x16x16xf16>)
func.func @main(%arg0: tensor<1x1x30x30xf16, {order = #NHWC}>, %arg1: tensor<1x3x16x16xf16>)
        -> tensor<1x4x16x16xf16, {order = #NHWC}> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 16, 16] : tensor<1x1x30x30xf16,  {order = #NHWC}> to tensor<1x1x16x16xf16, {order = #NHWC}>
    %1 = IE.Reorder(%0) {dstOrder = #NCHW} : tensor<1x1x16x16xf16, {order = #NHWC}> -> tensor<1x1x16x16xf16>
    %2 = IE.Concat(%1, %arg1) {per_axis = #IE.Concat<axis = 1>} : tensor<1x1x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x4x16x16xf16>
    %3 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<1x4x16x16xf16> -> tensor<1x4x16x16xf16, {order = #NHWC}>
    return %3 : tensor<1x4x16x16xf16, {order = #NHWC}>

    // CHECK:  [[ARG_REORDER:%.*]] = IE.Reorder(%arg1) {dstOrder = #NHWC} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16, {order = #NHWC}>
    // CHECK:  [[REORDER_INPUT:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 16, 16]
    // CHECK:  [[CONCAT:%.*]] = IE.Concat([[REORDER_INPUT]], [[ARG_REORDER]]) {per_axis = #IE.Concat<axis = 1 : i64>} :
    // CHECK-SAME:       tensor<1x1x16x16xf16, {order = #NHWC}>,
    // CHECK-SAME:       tensor<1x3x16x16xf16, {order = #NHWC}>
    // CHECK-SAME:       -> tensor<1x4x16x16xf16, {order = #NHWC}>
    // CHECK:  return [[CONCAT]] : tensor<1x4x16x16xf16, {order = #NHWC}>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotReorderWithConcatWithTwoArgs
module @NotReorderWithConcatWithTwoArgs {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<1x1x30x30xf16, {order = #NHWC}>,
// CHECK-SAME:      [[ARG1:%arg[0-9]+]]: tensor<1x3x16x16xf16>,
// CHECK-SAME:      [[ARG2:%arg[0-9]+]]: tensor<1x3x16x16xf16>)
func.func @main(%arg0: tensor<1x1x30x30xf16, {order = #NHWC}>, %arg1: tensor<1x3x16x16xf16>, %arg2: tensor<1x3x16x16xf16>)
        -> tensor<1x7x16x16xf16, {order = #NHWC}> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 16, 16] : tensor<1x1x30x30xf16,  {order = #NHWC}> to tensor<1x1x16x16xf16, {order = #NHWC}>
    %1 = IE.Reorder(%0) {dstOrder = #NCHW} : tensor<1x1x16x16xf16, {order = #NHWC}> -> tensor<1x1x16x16xf16>
    %2 = IE.Concat(%1, %arg1, %arg2) {per_axis = #IE.Concat<axis = 1>} : tensor<1x1x16x16xf16>, tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x7x16x16xf16>
    %3 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<1x7x16x16xf16> -> tensor<1x7x16x16xf16, {order = #NHWC}>
    return %3 : tensor<1x7x16x16xf16, {order = #NHWC}>

    // CHECK:  [[REORDER_INPUT:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 16, 16]
    // CHECK:  [[REORDER_SLICE:%.*]] = IE.Reorder([[REORDER_INPUT]])
    // CHECK:  [[CONCAT:%.*]] = IE.Concat([[REORDER_SLICE]], %arg1, %arg2) {per_axis = #IE.Concat<axis = 1 : i64>} :
    // CHECK:  [[REORDER_OUTPUT:%.*]] = IE.Reorder([[CONCAT]])
    // CHECK:  return [[REORDER_OUTPUT]] : tensor<1x7x16x16xf16, {order = #NHWC}>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderPropagationWithConcat
module @ReorderPropagationWithConcat {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<1x1x30x30xf16, {order = #NHWC}>,
// CHECK-SAME:      [[ARG1:%arg[0-9]+]]: tensor<1x1x30x30xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x1x30x30xf16, {order = #NHWC}>, %arg1: tensor<1x1x30x30xf16, {order = #NHWC}>)
        -> tensor<1x2x29x30xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x1x30x30xf16, {order = #NHWC}> -> tensor<1x1x30x30xf16>
    %1 = IE.Reorder(%arg1) {dstOrder = #NCHW} : tensor<1x1x30x30xf16, {order = #NHWC}> -> tensor<1x1x30x30xf16>
    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 1>} : tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x2x30x30xf16>
    %3 = IE.Slice %2 [0, 0, 1, 0] [1, 2, 29, 30] : tensor<1x2x30x30xf16> to tensor<1x2x29x30xf16>
    %4 = IE.Reorder(%3) {dstOrder = #NHWC} : tensor<1x2x29x30xf16> -> tensor<1x2x29x30xf16, {order = #NHWC}>
    return %4 : tensor<1x2x29x30xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%.+]] = IE.Concat([[ARG0]], [[ARG1]]) {per_axis = #IE.Concat<axis = 1 : i64>}
    // CHECK-SAME:      tensor<1x1x30x30xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x30x30xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x2x30x30xf16, {order = #NHWC}>
    // CHECK:       [[VAR1:%.+]] = IE.Slice [[VAR0]]
    // CHECK-SAME:      to tensor<1x2x29x30xf16, {order = #NHWC}>
    // CHECK:       return [[VAR1]] : tensor<1x2x29x30xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithExpandTwoBranches
module @ReorderWithExpandTwoBranches {

// CHECK:       func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x24x56x56xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x24x56x56xf16, {order = #NHWC}>) -> tensor<1x32x56x56xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %1 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x24x56x56xf16, {order = #NHWC}> -> tensor<1x24x56x56xf16>
    %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x24x56x56xf16> -> tensor<1x32x56x56xf16>
    %3 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<1x32x56x56xf16> -> tensor<1x32x56x56xf16, {order = #NHWC}>
    %4 = IE.Convolution(%3, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Relu<>, strides = [1, 1]} : tensor<1x32x56x56xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x56x56xf16, {order = #NHWC}>
    %5 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x24x56x56xf16> -> tensor<1x32x56x56xf16>
    %6 = IE.Reorder(%5) {dstOrder = #NHWC} : tensor<1x32x56x56xf16> -> tensor<1x32x56x56xf16, {order = #NHWC}>
    %7 = IE.Add(%6, %4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x56x56xf16, {order = #NHWC}>, tensor<1x32x56x56xf16, {order = #NHWC}> -> tensor<1x32x56x56xf16, {order = #NHWC}>

    return %7 : tensor<1x32x56x56xf16, {order = #NHWC}>
    // CHECK-DAG:       [[VAR0:%.*]] = const.Declare
    // CHECK-SAME:      tensor<32x32x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]

    // CHECK:       [[VAR1:%.*]] = IE.Expand([[ARG0]])
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0]
    // CHECK-SAME:      pads_end = [0, 8, 0, 0]
    // CHECK-SAME:      : tensor<1x24x56x56xf16, {order = #NHWC}> -> tensor<1x32x56x56xf16, {order = #NHWC}>

    // CHECK:       [[VAR2:%.*]] = IE.Convolution([[VAR1]], [[VAR0]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      post_op = #IE.Relu<>
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x32x56x56xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x32x56x56xf16, {order = #NHWC}>

    // CHECK:       [[VAR3:%.*]] = IE.Expand([[ARG0]])
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0]
    // CHECK-SAME:      pads_end = [0, 8, 0, 0]
    // CHECK-SAME:      : tensor<1x24x56x56xf16, {order = #NHWC}> -> tensor<1x32x56x56xf16, {order = #NHWC}>


    // CHECK:       [[VAR4:%.*]] = IE.Add([[VAR3]], [[VAR2]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x32x56x56xf16, {order = #NHWC}>, tensor<1x32x56x56xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x32x56x56xf16, {order = #NHWC}>

    // CHECK:       return [[VAR4]] : tensor<1x32x56x56xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithLayer
module @ReorderWithLayer {

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x3x30x30xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x3x30x30xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x3x30x30xf16>
    %1 = IE.SoftMax(%0) { axisInd = 1 } : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16, {order = #NHWC}>
    return %2 : tensor<1x3x30x30xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%.+]] = IE.SoftMax([[ARG0]]
    // CHECK-SAME:      tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x3x30x30xf16, {order = #NHWC}>

    // CHECK:       return [[VAR0]] : tensor<1x3x30x30xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

// CHECK-LABEL: @ReorderWithQuantizeCast
module @ReorderWithQuantizeCast {

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x30x30xui8, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x3x30x30xui8, {order = #NHWC}>) -> tensor<1x3x30x30x!qElemType, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x3x30x30xui8, {order = #NHWC}> -> tensor<1x3x30x30xui8>
    %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType} : tensor<1x3x30x30xui8> -> tensor<1x3x30x30x!qElemType>

    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x3x30x30x!qElemType> -> tensor<1x3x30x30x!qElemType, {order = #NHWC}>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
            tensor<1x3x30x30x!qElemType, {order = #NHWC}>, tensor<1x3x30x30x!qElemType, {order = #NHWC}>
            -> tensor<1x3x30x30x!qElemType, {order = #NHWC}>

    return %3 : tensor<1x3x30x30x!qElemType, {order = #NHWC}>

    // CHECK-NOT:  IE.Reorder

    // CHECK:      [[VAR0:%.+]] = IE.QuantizeCast([[ARG0:%.+]]) {dstElemType = !qElemType} :
    // CHECK-SAME:     tensor<1x3x30x30xui8, {order = #NHWC}> -> tensor<1x3x30x30x!qElemType, {order = #NHWC}>

    // CHECK-NOT:  IE.Reorder

    // CHECK:      [[VAR1:%.+]] = IE.Add([[VAR0]], [[VAR0]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
    // CHECK-SAME:     tensor<1x3x30x30x!qElemType, {order = #NHWC}>, tensor<1x3x30x30x!qElemType, {order = #NHWC}> -> tensor<1x3x30x30x!qElemType, {order = #NHWC}>

    // CHECK:      return [[VAR1]] : tensor<1x3x30x30x!qElemType, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DontReorderEltwiseAndOp
module @DontReorderEltwiseAndOp {

func.func @main(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x3x30x30xf16, {order = #NHWC}> {
    // CHECK:       ([[ARG0:%.+]]: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x3x30x30xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x30x30xf16> = dense<9.997550e-02> : tensor<1x1x30x30xf16>
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x3x30x30xf16>
    %1 = IE.And(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x3x30x30xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16, {order = #NHWC}>
    return %2 : tensor<1x3x30x30xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x1x30x30xf16> = dense<9.997550e-02> : tensor<1x1x30x30xf16>
    // CHECK:       [[REORDER_IN:%.+]] = IE.Reorder([[ARG0]]) {dstOrder = #NCHW} : tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x3x30x30xf16>
    // CHECK:       [[AND:%.+]] = IE.And([[REORDER_IN]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       [[REORDER_OUT:%.+]] = IE.Reorder([[AND]]) {dstOrder = #NHWC} : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16, {order = #NHWC}>

    // CHECK:       return [[REORDER_OUT]] : tensor<1x3x30x30xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.51323526419845278:128>
!qElemType1 = !quant.uniform<u8:f16, 0.25661763209922639:128>
!qElemType2 = !quant.uniform<u8:f16, 0.12830881604961319:128>

module @ReorderWithQuantizeCastTwoBranches {

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x48x14x14x!qElemType, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x48x14x14x!qElemType, {order = #NHWC}>) -> (tensor<1x48x14x14x!qElemType2, {order = #NHWC}>, tensor<1x14x14x40x!qElemType1>) {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x48x14x14x!qElemType, {order = #NHWC}> -> tensor<1x48x14x14x!qElemType>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 40, 14, 14] : tensor<1x48x14x14x!qElemType> to tensor<1x40x14x14x!qElemType>
    %2 = IE.QuantizeCast(%1) {dstElemType = !qElemType1} : tensor<1x40x14x14x!qElemType> -> tensor<1x40x14x14x!qElemType1>
    %3 = IE.QuantizeCast(%2) {dstElemType = !qElemType2} : tensor<1x40x14x14x!qElemType1> -> tensor<1x40x14x14x!qElemType2>
    %4 = IE.Expand(%3) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x40x14x14x!qElemType2> -> tensor<1x48x14x14x!qElemType2>
    %5 = IE.Reorder(%4) {dstOrder = #NHWC} : tensor<1x48x14x14x!qElemType2> -> tensor<1x48x14x14x!qElemType2, {order = #NHWC}>
    %6 = IE.Reshape(%2) {shape_value = [1, 14, 14, 40]} : tensor<1x40x14x14x!qElemType1> -> tensor<1x14x14x40x!qElemType1>

   return %5, %6 : tensor<1x48x14x14x!qElemType2, {order = #NHWC}>, tensor<1x14x14x40x!qElemType1>

    // CHECK-NOT:  IE.Reorder
    // CHECK:      [[SLICE:%.+]] = IE.Slice %arg0
    // CHECK-SAME:  tensor<1x48x14x14x!qElemType, {order = #NHWC}> to tensor<1x40x14x14x!qElemType, {order = #NHWC}>
    // CHECK:      [[QCAST0:%.+]] = IE.QuantizeCast([[SLICE]]) {dstElemType = !qElemType2}
    // CHECK-SAME: tensor<1x40x14x14x!qElemType, {order = #NHWC}> -> tensor<1x40x14x14x!qElemType2, {order = #NHWC}>
    // CHECK:      [[REORDER:%.+]] = IE.Reorder([[QCAST0]]) {dstOrder = #NCHW}
    // CHECK-SAME: tensor<1x40x14x14x!qElemType2, {order = #NHWC}> -> tensor<1x40x14x14x!qElemType2>
    // CHECK:      [[QCAST1:%.+]] = IE.QuantizeCast([[QCAST0]]) {dstElemType = !qElemType1}
    // CHECK-SAME: tensor<1x40x14x14x!qElemType2, {order = #NHWC}> -> tensor<1x40x14x14x!qElemType1, {order = #NHWC}>
    // CHECK:      [[RESULT0:%.+]] = IE.Expand([[QCAST1]])
    // CHECK-SAME: tensor<1x40x14x14x!qElemType1, {order = #NHWC}> -> tensor<1x48x14x14x!qElemType1, {order = #NHWC}>
    // CHECK:      [[RESULT1:%.+]] = IE.Reshape([[REORDER]])
    // CHECK-SAME: tensor<1x40x14x14x!qElemType2> -> tensor<1x14x14x40x!qElemType2>
    // CHECK:      return [[RESULT0]], [[RESULT1]] : tensor<1x48x14x14x!qElemType1, {order = #NHWC}>, tensor<1x14x14x40x!qElemType2>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CMajorToZMajorConv
module @CMajorToZMajorConv {

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x3x32x32xf16, {order = #NHWC}>) -> tensor<1x16x32x32xf16, {order = #NHWC}> {
func.func @main(%arg0: tensor<1x3x32x32xf16, {order = #NHWC}>) -> tensor<1x16x32x32xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x3x1x1xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x3x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x3x32x32xf16, {order = #NHWC}> -> tensor<1x3x32x32xf16>

    %1 = IE.Convolution(%0, %cst) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x32x32xf16>, tensor<16x3x1x1xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>

    return %1 : tensor<1x16x32x32xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<16x3x1x1xf16, {order = #NHWC}>
    // CHECK:       [[VAR0:%.+]] = IE.Convolution([[ARG0]], [[CST]])
    // CHECK-SAME:       -> tensor<1x16x32x32xf16, {order = #NHWC}>
    // CHECK:       return [[VAR0]] : tensor<1x16x32x32xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2, d3)>

// CHECK-LABEL: @ReorderPermuteCastChainWithSlice
module @ReorderPermuteCastChainWithSlice {

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x128x30x30xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x128x30x30xf16, {order = #NHWC}>) -> (tensor<1x4x30x30x16xf16>, tensor<1x4x30x30x16xf16>) {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x128x30x30xf16, {order = #NHWC}> -> tensor<1x128x30x30xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1, 2], [3], [4]], shape_value = [1, 4, 32, 30, 30]} :
            tensor<1x128x30x30xf16> -> tensor<1x4x32x30x30xf16>
    %2 = IE.PermuteCast(%1) {dst_order = #map1, mem_perm = #NCDHW} : tensor<1x4x32x30x30xf16> -> tensor<1x4x30x30x32xf16, {order = #map1}>
    %3 = IE.Reorder(%2) {dstOrder = #NCDHW} : tensor<1x4x30x30x32xf16, {order = #map1}> -> tensor<1x4x30x30x32xf16, {order = #NCDHW}>
    %4 = IE.Slice %3 [0, 0, 0, 0, 0] [1, 4, 30, 30, 16] : tensor<1x4x30x30x32xf16, {order = #NCDHW}> to tensor<1x4x30x30x16xf16>
    %5 = IE.Slice %3 [0, 0, 0, 0, 1] [1, 4, 30, 30, 16] : tensor<1x4x30x30x32xf16, {order = #NCDHW}> to tensor<1x4x30x30x16xf16>
    return %4, %5 : tensor<1x4x30x30x16xf16>, tensor<1x4x30x30x16xf16>

    // CHECK:       [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME(LITERAL):       {dim_mapping = [[0], [1, 2], [3], [4]], shape_value = [1, 4, 32, 30, 30]}
    // CHECK-SAME:      -> tensor<1x4x32x30x30xf16, {order = #map}>
    // CHECK:       [[PERMUTECAST:%.+]] = IE.PermuteCast([[AFFINERESHAPE]]) {dst_order = #map1, mem_perm = #NCDHW}
    // CHECK-SAME:      -> tensor<1x4x30x30x32xf16, {order = #map1}>
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[PERMUTECAST]]
    // CHECK-SAME:      tensor<1x4x30x30x32xf16, {order = #map1}> to tensor<1x4x30x30x16xf16, {order = #map1}>
    // CHECK:       [[REORDER0:%.+]] = IE.Reorder([[SLICE0]]) {dstOrder = #NCDHW}
    // CHECK-SAME:      -> tensor<1x4x30x30x16xf16>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[PERMUTECAST]]
    // CHECK-SAME:      tensor<1x4x30x30x32xf16, {order = #map1}> to tensor<1x4x30x30x16xf16, {order = #map1}>
    // CHECK:       [[REORDER1:%.+]] = IE.Reorder([[SLICE1]]) {dstOrder = #NCDHW}
    // CHECK-SAME:      -> tensor<1x4x30x30x16xf16>
    // CHECK:       return [[REORDER0]], [[REORDER1]]
    // CHECK-SAME:      tensor<1x4x30x30x16xf16>,
    // CHECK-SAME:      tensor<1x4x30x30x16xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @ReorderWithPermuteCastSubView
module @ReorderWithPermuteCastSubView {

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x8x1x128xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x8x1x128xf16, {order = #NHWC}>) -> (tensor<1x128x3x8xf16, {order = #NHWC}>, tensor<1x8x2x128xf16>) {
    %cst = const.Declare tensor<1x8x2x128xf16> = dense<0.000000e+00> : tensor<1x8x2x128xf32>, [#const.CastElemType<f16>]
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x8x1x128xf16, {order = #NHWC}> -> tensor<1x8x1x128xf16>
    %1 = IE.Concat(%cst, %0) {static_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]} : tensor<1x8x2x128xf16>, tensor<1x8x1x128xf16> -> tensor<1x8x3x128xf16>
    %2 = IE.PermuteCast(%1) {dst_order = #NWHC, mem_perm = #NCHW} : tensor<1x8x3x128xf16> -> tensor<1x128x3x8xf16, {order = #NWHC}>
    %3 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<1x128x3x8xf16, {order = #NWHC}> -> tensor<1x128x3x8xf16, {order = #NHWC}>
    %4 = IE.Slice %1 [0, 0, 1, 0] [1, 8, 2, 128] : tensor<1x8x3x128xf16> to tensor<1x8x2x128xf16>
    return %3, %4 : tensor<1x128x3x8xf16, {order = #NHWC}>, tensor<1x8x2x128xf16>
}

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x8x2x128xf16, {order = #NHWC}>
    // CHECK-SAME:      tensor<1x8x2x128xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    // CHECK:       [[CONCAT0:%.+]] = IE.Concat([[CST]], [[ARG0]])
    // CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]} :
    // CHECK-SAME:            tensor<1x8x2x128xf16, {order = #NHWC}>, tensor<1x8x1x128xf16, {order = #NHWC}>
    // CHECK-SAME:            -> tensor<1x8x3x128xf16, {order = #NHWC}>

    // CHECK:       [[PERMUTECAST:%.+]] = IE.PermuteCast([[CONCAT0]]) {dst_order = #NHCW, mem_perm = #NCHW}
    // CHECK-SAME:      -> tensor<1x128x3x8xf16, {order = #NHCW}>
    // CHECK:       [[REORDER0:%.+]] = IE.Reorder([[PERMUTECAST]]) {dstOrder = #NHWC}
    // CHECK-SAME:      -> tensor<1x128x3x8xf16, {order = #NHWC}>

    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[CONCAT0]]
    // CHECK-SAME:          [0, 0, 1, 0] [1, 8, 2, 128] : tensor<1x8x3x128xf16, {order = #NHWC}> to tensor<1x8x2x128xf16, {order = #NHWC}>
    // CHECK:       [[REORDER1:%.+]] = IE.Reorder([[SLICE0]]) {dstOrder = #NCHW}
    // CHECK-SAME:      -> tensor<1x8x2x128xf16>

    // CHECK:       return [[REORDER0]], [[REORDER1]] : tensor<1x128x3x8xf16, {order = #NHWC}>, tensor<1x8x2x128xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @ReorderWithReadValueAndAssign
module @ReorderWithReadValueAndAssign {

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x16x1x128xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x16x1x128xf16, {order = #NHWC}>) -> tensor<1x16x3x128xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x16x2x128xf16> = dense<0.000000e+00> : tensor<1x16x2x128xf32>, [#const.CastElemType<f16>]
    %0 = IE.ReadValue(%cst) {name = "MemoryCellId-2"} : tensor<1x16x2x128xf16> -> tensor<1x16x2x128xf16>
    %1 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x16x1x128xf16, {order = #NHWC}> -> tensor<1x16x1x128xf16>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]} : tensor<1x16x2x128xf16>, tensor<1x16x1x128xf16> -> tensor<1x16x3x128xf16>
    %3 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<1x16x3x128xf16> -> tensor<1x16x3x128xf16, {order = #NHWC}>
    %4 = IE.Slice %2 [0, 0, 1, 0] [1, 16, 2, 128] : tensor<1x16x3x128xf16> to tensor<1x16x2x128xf16>
    %5 = IE.Assign(%4) {name = "MemoryCellId-2"} : tensor<1x16x2x128xf16> -> tensor<1x16x2x128xf16>
    return %3 : tensor<1x16x3x128xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x16x2x128xf16, {order = #NHWC}>
    // CHECK-SAME:      tensor<1x16x2x128xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[READVALUE0:%.+]] = IE.ReadValue([[CST]]) {name = "MemoryCellId-2"}
    // CHECK-SAME:      -> tensor<1x16x2x128xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT0:%.+]] = IE.Concat([[READVALUE0]], [[ARG0]])
    // CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]} :
    // CHECK-SAME:            tensor<1x16x2x128xf16, {order = #NHWC}>, tensor<1x16x1x128xf16, {order = #NHWC}>
    // CHECK-SAME:            -> tensor<1x16x3x128xf16, {order = #NHWC}>

    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[CONCAT0]]
    // CHECK-SAME:          [0, 0, 1, 0] [1, 16, 2, 128] : tensor<1x16x3x128xf16, {order = #NHWC}> to tensor<1x16x2x128xf16, {order = #NHWC}>
    // CHECK:       [[ASSIGN0:%.+]] = IE.Assign([[SLICE0]]) {name = "MemoryCellId-2"}
    // CHECK-SAME:      -> tensor<1x16x2x128xf16, {order = #NHWC}>

    // CHECK:       return [[CONCAT0]] : tensor<1x16x3x128xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @ReorderWithReadValueAndAssignNegative
module @ReorderWithReadValueAndAssignNegative {

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x16x1x128xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x16x1x128xf16, {order = #NHWC}>) -> tensor<1x16x3x128xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x16x2x128xf16> = dense<0.000000e+00> : tensor<1x16x2x128xf32>, [#const.CastElemType<f16>]
    %0 = IE.ReadValue(%cst) {name = "MemoryCellId-2"} : tensor<1x16x2x128xf16> -> tensor<1x16x2x128xf16>
    %1 = IE.Negative(%0) : tensor<1x16x2x128xf16> -> tensor<1x16x2x128xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x16x2x128xf16> -> tensor<1x16x2x128xf16, {order = #NHWC}>
    %3 = IE.Concat(%2, %arg0) {static_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]} : tensor<1x16x2x128xf16, {order = #NHWC}>, tensor<1x16x1x128xf16, {order = #NHWC}> -> tensor<1x16x3x128xf16, {order = #NHWC}>
    %4 = IE.Slice %3[0, 0, 1, 0] [1, 16, 2, 128] : tensor<1x16x3x128xf16, {order = #NHWC}> to tensor<1x16x2x128xf16, {order = #NHWC}>
    %5 = IE.Reorder(%4) {dstOrder = #NCHW} : tensor<1x16x2x128xf16, {order = #NHWC}> -> tensor<1x16x2x128xf16>
    %6 = IE.Assign(%5) {name = "MemoryCellId-2"} : tensor<1x16x2x128xf16> -> tensor<1x16x2x128xf16>
    return %3 : tensor<1x16x3x128xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x16x2x128xf16>
    // CHECK-SAME:     tensor<1x16x2x128xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[READVALUE0:%.+]] = IE.ReadValue([[CST]]) {name = "MemoryCellId-2"}
    // CHECK-SAME:     -> tensor<1x16x2x128xf16>
    // CHECK:       [[NEGATIVE:%.+]] = IE.Negative([[READVALUE0]]) : tensor<1x16x2x128xf16>
    // CHECK-SAME:     -> tensor<1x16x2x128xf16>

    // CHECK:       [[REORDER0:%.+]] = IE.Reorder([[NEGATIVE]]) {dstOrder = #NHWC}
    // CHECK:       [[CONCAT0:%.+]] = IE.Concat([[REORDER0]], [[ARG0]])

    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[CONCAT0]]
    // CHECK-SAME:          [0, 0, 1, 0] [1, 16, 2, 128] : tensor<1x16x3x128xf16, {order = #NHWC}> to tensor<1x16x2x128xf16, {order = #NHWC}>
    // CHECK:       [[REORDER1:%.+]] = IE.Reorder([[SLICE0]]) {dstOrder = #NCHW}
    // CHECK-SAME:      -> tensor<1x16x2x128xf16>
    // CHECK:       [[ASSIGN0:%.+]] = IE.Assign([[REORDER1]]) {name = "MemoryCellId-2"}
    // CHECK-SAME:     -> tensor<1x16x2x128xf16>
    // CHECK:       return [[CONCAT0]] : tensor<1x16x3x128xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithHwAddNoChange
module @ReorderWithHwAddNoChange {

func.func @main(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x30x30xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x30x30xf16, {order = #NHWC}>, tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x3x30x30xf16, {order = #NHWC}>
    %2 = IE.Reorder(%1) {dstOrder = #NCHW} : tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x3x30x30xf16>
    return %2 : tensor<1x3x30x30xf16>

    // CHECK:       [[VAR0:%.+]] = IE.Reorder(%arg0) {dstOrder = #NHWC}
    // CHECK:       [[VAR1:%.+]] = IE.Add([[VAR0]], [[VAR0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:       [[VAR2:%.+]] = IE.Reorder([[VAR1]]) {dstOrder = #NCHW}

    // CHECK:       return [[VAR2]] : tensor<1x3x30x30xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithSwAddHasChange
module @ReorderWithSwAddHasChange {

func.func @main(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x3x30x30xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x30x30xf16> = dense<1.0> : tensor<1x1x30x30xf16>
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x3x30x30xf16>
    %1 = IE.Add(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x3x30x30xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16, {order = #NHWC}>
    return %2 : tensor<1x3x30x30xf16, {order = #NHWC}>

    // CHECK-DAG:       [[VAR0:%.+]] = const.Declare tensor<1x1x30x30xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x30x30xf16>, [#const.Reorder<#NHWC>]
    // CHECK:       [[VAR1:%.+]] = IE.Add(%arg0, [[VAR0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x30x30xf16, {order = #NHWC}>, tensor<1x1x30x30xf16, {order = #NHWC}> -> tensor<1x3x30x30xf16, {order = #NHWC}>

    // CHECK:       return [[VAR1]] : tensor<1x3x30x30xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithSwMultiplyHasChange
module @ReorderWithSwMultiplyHasChange {

func.func @main(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x3x30x30xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x30x30xf16> = dense<0.1> : tensor<1x1x30x30xf16>
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x3x30x30xf16>
    %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x3x30x30xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16, {order = #NHWC}>
    return %2 : tensor<1x3x30x30xf16, {order = #NHWC}>

    // CHECK-DAG:       [[VAR0:%.+]] = const.Declare tensor<1x1x30x30xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x30x30xf16>, [#const.Reorder<#NHWC>]
    // CHECK:       [[VAR1:%.+]] = IE.Multiply(%arg0, [[VAR0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x30x30xf16, {order = #NHWC}>, tensor<1x1x30x30xf16, {order = #NHWC}> -> tensor<1x3x30x30xf16, {order = #NHWC}>

    // CHECK:       return [[VAR1]] : tensor<1x3x30x30xf16, {order = #NHWC}>
}

}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotSwapExpandWithReorder
module @NotSwapExpandWithReorder {

func.func @main(%arg0: tensor<1x1x8x9xf16, {order = #NCWH}>) -> tensor<1x16x8x9xf16, {order = #NHWC}> {
    %1 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x1x8x9xf16, {order = #NCWH}> -> tensor<1x1x8x9xf16, {order = #NHWC}>
    %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x8x9xf16, {order = #NHWC}> -> tensor<1x16x8x9xf16, {order = #NHWC}>
    return %2 : tensor<1x16x8x9xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%.+]] = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x1x8x9xf16, {order = #NCWH}> -> tensor<1x1x8x9xf16, {order = #NHWC}>
    // CHECK:       [[VAR1:%.+]] = IE.Expand([[VAR0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x8x9xf16, {order = #NHWC}> -> tensor<1x16x8x9xf16, {order = #NHWC}>

    // CHECK:       return [[VAR1]] : tensor<1x16x8x9xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapReorderWithStridedSlice
module @SwapReorderWithStridedSlice {

func.func @main(%arg0: tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x3x320x640xf16> {
    %begins = const.Declare tensor<4xsi64> = dense<[0, 0, 0, 0]> : tensor<4xsi64>
    %ends = const.Declare tensor<4xsi64> = dense<[0, 0, 2147483647, 0]> : tensor<4xsi64>
    %strides = const.Declare tensor<4xsi64> = dense<[1, 1, 2, 1]> : tensor<4xsi64>

    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x3x640x640xf16, {order = #NHWC}> -> tensor<1x3x640x640xf16>
    %1 = IE.StridedSlice(%0, %begins, %ends, %strides) {
        begin_mask = [1, 1, 1, 1],
        end_mask = [1, 1, 0, 1],
        new_axis_mask = [0, 0, 0, 0],
        shrink_axis_mask = [0, 0, 0, 0],
        ellipsis_mask = [0, 0, 0, 0],
        operandSegmentSizes = array<i32: 1, 1, 1, 1>
    } : tensor<1x3x640x640xf16>, tensor<4xsi64>, tensor<4xsi64>, tensor<4xsi64> -> tensor<1x3x320x640xf16>
    return %1 : tensor<1x3x320x640xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<4xsi64> = dense<0> : tensor<4xsi64>
    // CHECK:       [[CST_0:%.+]] = const.Declare tensor<4xsi64> = dense<[0, 0, 2147483647, 0]> : tensor<4xsi64>
    // CHECK:       [[CST_1:%.+]] = const.Declare tensor<4xsi64> = dense<[1, 1, 2, 1]> : tensor<4xsi64>
    // CHECK:       [[VAR0:%.+]] = IE.StridedSlice(%arg0, [[CST]], [[CST_0]], [[CST_1]]) {
    // CHECK-SAME:    begin_mask =
    // CHECK-SAME:    [1, 1, 1, 1],
    // CHECK-SAME:    ellipsis_mask =
    // CHECK-SAME:    [0, 0, 0, 0],
    // CHECK-SAME:    end_mask =
    // CHECK-SAME:    [1, 1, 0, 1],
    // CHECK-SAME:    new_axis_mask = [0, 0, 0, 0],
    // CHECK-SAME:    operandSegmentSizes = array<i32: 1, 1, 1, 1>, shrink_axis_mask =
    // CHECK-SAME:    [0, 0, 0, 0]
    // CHECK-SAME:    } : tensor<1x3x640x640xf16, {order = #NHWC}>, tensor<4xsi64>, tensor<4xsi64>, tensor<4xsi64> ->
    // CHECK-SAME:    tensor<1x3x320x640xf16, {order = #NHWC}>
    // CHECK:       [[VAR1:%.+]] = IE.Reorder([[VAR0]]) {dstOrder = #NCHW} : tensor<1x3x320x640xf16, {order = #NHWC}> -> tensor<1x3x320x640xf16>

    // CHECK:       return [[VAR1]] : tensor<1x3x320x640xf16>
}

}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#NCWDH = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2, d3)>

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK:     #map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2, d3)>

// CHECK-LABEL: @MoveReorderThroughConvertAndViewLikeOps
module @MoveReorderThroughConvertAndViewLikeOps {

func.func @main(%arg0: tensor<1x3x80x80x85xf16, {order = #NCWDH}>) -> tensor<1x3x80x80x85xf32> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCDHW} : tensor<1x3x80x80x85xf16, {order = #NCWDH}> -> tensor<1x3x80x80x85xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [2], [2], [3]], shape_value = [1, 3, 6400, 85]} : tensor<1x3x80x80x85xf16> -> tensor<1x3x6400x85xf16>
    %2 = IE.Convert(%1) {dstElemType = f32} : tensor<1x3x6400x85xf16> -> tensor<1x3x6400x85xf32>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 3, 80, 80, 85]} : tensor<1x3x6400x85xf32> -> tensor<1x3x80x80x85xf32>
    %4 = IE.SoftMax(%3) { axisInd = 1 } : tensor<1x3x80x80x85xf32> -> tensor<1x3x80x80x85xf32>

    return %4 : tensor<1x3x80x80x85xf32>

    // CHECK:               [[RESHAPE0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [2], [2], [3]], shape_value = [1, 3, 6400, 85]} : tensor<1x3x80x80x85xf16, {order = #map}> -> tensor<1x3x6400x85xf16, {order = #NCWH}>
    // CHECK:               [[CONVERT:%.*]] = IE.Convert([[RESHAPE0]]) {dstElemType = f32} : tensor<1x3x6400x85xf16, {order = #NCWH}> -> tensor<1x3x6400x85xf32, {order = #NCWH}>
    // CHECK:               [[RESHAPE1:%.*]] = IE.AffineReshape([[CONVERT]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 3, 80, 80, 85]} : tensor<1x3x6400x85xf32, {order = #NCWH}> -> tensor<1x3x80x80x85xf32, {order = #map}>
    // CHECK:               [[SOFTMAX:%.*]] = IE.SoftMax([[RESHAPE1]]) {axisInd = 1 : i64} : tensor<1x3x80x80x85xf32, {order = #map}> -> tensor<1x3x80x80x85xf32, {order = #map}>
    // CHECK:               [[REORDER:%.*]] = IE.Reorder([[SOFTMAX]]) {dstOrder = #NCDHW} : tensor<1x3x80x80x85xf32, {order = #map}> -> tensor<1x3x80x80x85xf32>

    // CHECK:               return [[REORDER]] : tensor<1x3x80x80x85xf32>
}

}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#NCWDH = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2, d3)>

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK:     #map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2, d3)>

// CHECK-LABEL: @NotMoveReorderThroughConvertAndViewLikeOpsIfConsumerIsReturnOp
module @NotMoveReorderThroughConvertAndViewLikeOpsIfConsumerIsReturnOp {

func.func @main(%arg0: tensor<1x3x80x80x85xf16, {order = #NCWDH}>) -> tensor<1x3x80x80x85xf32> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCDHW} : tensor<1x3x80x80x85xf16, {order = #NCWDH}> -> tensor<1x3x80x80x85xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [2], [2], [3]], shape_value = [1, 3, 6400, 85]} : tensor<1x3x80x80x85xf16> -> tensor<1x3x6400x85xf16>
    %2 = IE.Convert(%1) {dstElemType = f32} : tensor<1x3x6400x85xf16> -> tensor<1x3x6400x85xf32>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 3, 80, 80, 85]} : tensor<1x3x6400x85xf32> -> tensor<1x3x80x80x85xf32>
    return %3 : tensor<1x3x80x80x85xf32>

    // CHECK:               [[REORDER:%.*]] = IE.Reorder(%arg0) {dstOrder = #NCDHW} : tensor<1x3x80x80x85xf16, {order = #map}> -> tensor<1x3x80x80x85xf16>
    // CHECK:               [[RESHAPE0:%.*]] = IE.AffineReshape([[REORDER]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [2], [2], [3]], shape_value = [1, 3, 6400, 85]} : tensor<1x3x80x80x85xf16> -> tensor<1x3x6400x85xf16>
    // CHECK:               [[CONVERT:%.*]] = IE.Convert([[RESHAPE0]]) {dstElemType = f32} : tensor<1x3x6400x85xf16> -> tensor<1x3x6400x85xf32>
    // CHECK:               [[RESHAPE1:%.*]] = IE.AffineReshape([[CONVERT]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 3, 80, 80, 85]} : tensor<1x3x6400x85xf32> -> tensor<1x3x80x80x85xf32>

    // CHECK:               return [[RESHAPE1]] : tensor<1x3x80x80x85xf32>
}

}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#NCWDH = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2, d3)>

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK:     #map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2, d3)>

// CHECK-LABEL: @MoveReorderThroughConvertAndViewLikeOpsIfElemTypeDecreased
module @MoveReorderThroughConvertAndViewLikeOpsIfElemTypeDecreased {

func.func @main(%arg0: tensor<1x3x80x80x85xf32, {order = #NCWDH}>) -> tensor<1x3x80x80x85xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCDHW} : tensor<1x3x80x80x85xf32, {order = #NCWDH}> -> tensor<1x3x80x80x85xf32>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [2], [2], [3]], shape_value = [1, 3, 6400, 85]} : tensor<1x3x80x80x85xf32> -> tensor<1x3x6400x85xf32>
    %2 = IE.Convert(%1) {dstElemType = f16} : tensor<1x3x6400x85xf32> -> tensor<1x3x6400x85xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 3, 80, 80, 85]} : tensor<1x3x6400x85xf16> -> tensor<1x3x80x80x85xf16>
    return %3 : tensor<1x3x80x80x85xf16>

    // CHECK:               [[RESHAPE0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [2], [2], [3]], shape_value = [1, 3, 6400, 85]} : tensor<1x3x80x80x85xf32, {order = #map}> -> tensor<1x3x6400x85xf32, {order = #NCWH}>
    // CHECK:               [[CONVERT:%.*]] = IE.Convert([[RESHAPE0]]) {dstElemType = f16} : tensor<1x3x6400x85xf32, {order = #NCWH}> -> tensor<1x3x6400x85xf16, {order = #NCWH}>
    // CHECK:               [[REORDER:%.*]] = IE.Reorder([[CONVERT]]) {dstOrder = #NCHW} : tensor<1x3x6400x85xf16, {order = #NCWH}> -> tensor<1x3x6400x85xf16>
    // CHECK:               [[RESHAPE1:%.*]] = IE.AffineReshape([[REORDER]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 3, 80, 80, 85]} : tensor<1x3x6400x85xf16> -> tensor<1x3x80x80x85xf16>

    // CHECK:               return [[RESHAPE1]] : tensor<1x3x80x80x85xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2, d3)>

// CHECK:     #map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2, d3)>

// CHECK-LABEL: @NotMoveReorderThroughViewLikeOpWithReturnConsumer
module @NotMoveReorderThroughViewLikeOpWithReturnConsumer {

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x128x30x30xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x128x30x30xf16, {order = #NHWC}>) -> (tensor<1x4x30x30x32xf16, {order = #map1}>) {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x128x30x30xf16, {order = #NHWC}> -> tensor<1x128x30x30xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1, 2], [3], [4]], shape_value = [1, 4, 32, 30, 30]} :
            tensor<1x128x30x30xf16> -> tensor<1x4x32x30x30xf16>
    %2 = IE.PermuteCast(%1) {dst_order = #map1, mem_perm = #NCDHW} : tensor<1x4x32x30x30xf16> -> tensor<1x4x30x30x32xf16, {order = #map1}>
    return %2 : tensor<1x4x30x30x32xf16, {order = #map1}>

    // CHECK:       [[REORDER:%.+]] = IE.Reorder([[ARG0]]) {dstOrder = #NCHW}
    // CHECK-SAME:      -> tensor<1x128x30x30xf16>
    // CHECK:       [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[REORDER]])
    // CHECK-SAME(LITERAL):       {dim_mapping = [[0], [1, 2], [3], [4]], shape_value = [1, 4, 32, 30, 30]}
    // CHECK-SAME:      -> tensor<1x4x32x30x30xf16>
    // CHECK:       [[PERMUTECAST:%.+]] = IE.PermuteCast([[AFFINERESHAPE]]) {dst_order = #map, mem_perm = #NCDHW}
    // CHECK-SAME:      -> tensor<1x4x30x30x32xf16, {order = #map}>

    // CHECK:       return [[PERMUTECAST]]
    // CHECK-SAME:      tensor<1x4x30x30x32xf16, {order = #map}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>

module @ReorderWithReshape {
// CHECK-LABEL: @MoveReorderThroughReshapeWhenReshapeInImmuatbleGroup
func.func @MoveReorderThroughReshapeWhenReshapeInImmuatbleGroup(%arg0: tensor<1x32x8x2xf16, {order = #NHWC}>) -> tensor<1x32x4x4xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x32x8x2xf16, {order = #NHWC}> -> tensor<1x32x8x2xf16>
    %1 = IE.Reshape(%0) {shape_value = [1, 32, 4, 4]} : tensor<1x32x8x2xf16> -> tensor<1x32x4x4xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x32x4x4xf16> -> tensor<1x32x4x4xf16, {order = #NHWC}>
    return %2 : tensor<1x32x4x4xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%.+]] = IE.ShapeCast {shape = [1, 32, 4, 4]}
    // CHECK-SAME:      inputs(%arg0 : tensor<1x32x8x2xf16, {order = #NHWC}>) -> tensor<1x32x4x4xf16, {order = #NHWC}>
    // CHECK:       return [[VAR0]] : tensor<1x32x4x4xf16, {order = #NHWC}>
}

// CHECK-LABEL: @NotMoveReorderThroughReshapeNotInImmuatbleGroup
func.func @NotMoveReorderThroughReshapeNotInImmuatbleGroup(%arg0: tensor<1x32x8x2xf16, {order = #NHWC}>) -> tensor<1x16x16x2xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x32x8x2xf16, {order = #NHWC}> -> tensor<1x32x8x2xf16>
    %1 = IE.Reshape(%0) {shape_value = [1, 16, 16, 2]} : tensor<1x32x8x2xf16> -> tensor<1x16x16x2xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x16x16x2xf16> -> tensor<1x16x16x2xf16, {order = #NHWC}>
    return %2 : tensor<1x16x16x2xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%.+]] = IE.Reorder(%arg0)
    // CHECK:       [[VAR1:%.+]] = IE.Reshape([[VAR0]])
    // CHECK:       [[VAR2:%.+]] = IE.Reorder([[VAR1]])
    // CHECK:       return [[VAR2]] : tensor<1x16x16x2xf16, {order = #NHWC}>
}

// CHECK-LABEL: @MoveReorderThroughReshapeWithIdenticalMemShapeAndPerm
func.func @MoveReorderThroughReshapeWithIdenticalMemShapeAndPerm(%arg0: tensor<1x1x64x128xf16, {order = #NWHC}>) -> tensor<1x64x128x1xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x1x64x128xf16, {order = #NWHC}> -> tensor<1x1x64x128xf16>
    %1 = IE.Reshape(%0) {shape_value = [1, 64, 128, 1]} : tensor<1x1x64x128xf16> -> tensor<1x64x128x1xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x64x128x1xf16> -> tensor<1x64x128x1xf16, {order = #NHWC}>
    return %2 : tensor<1x64x128x1xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%.+]] = IE.ShapeCast {shape = [1, 64, 128, 1]}
    // CHECK-SAME:      inputs(%arg0 : tensor<1x1x64x128xf16, {order = #NWHC}>) -> tensor<1x64x128x1xf16, {order = #NWHC}>
    // CHECK:       [[VAR1:%.+]] = IE.Reorder([[VAR0]]) {dstOrder = #NHWC} :
    // CHECK-SAME:      tensor<1x64x128x1xf16, {order = #NWHC}> -> tensor<1x64x128x1xf16, {order = #NHWC}>
    // CHECK:       return [[VAR1]]
}

// CHECK-LABEL: @NotMoveReorderThroughReshapeWithNoIdenticalMemShape
func.func @NotMoveReorderThroughReshapeWithNoIdenticalMemShape(%arg0: tensor<1x1x64x128xf16, {order = #NWHC}>) -> tensor<64x1x128x1xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x1x64x128xf16, {order = #NWHC}> -> tensor<1x1x64x128xf16>
    %1 = IE.Reshape(%0) {shape_value = [64, 1, 128, 1]} : tensor<1x1x64x128xf16> -> tensor<64x1x128x1xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<64x1x128x1xf16> -> tensor<64x1x128x1xf16, {order = #NHWC}>
    return %2 : tensor<64x1x128x1xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%.+]] = IE.Reorder(%arg0)
    // CHECK:       [[VAR1:%.+]] = IE.Reshape([[VAR0]])
    // CHECK:       [[VAR2:%.+]] = IE.Reorder([[VAR1]])
    // CHECK:       return [[VAR2]] : tensor<64x1x128x1xf16, {order = #NHWC}>
}

// CHECK-LABEL: @NotMoveReorderThroughReshapeWithNoIdenticalPerm
func.func @NotMoveReorderThroughReshapeWithNoIdenticalPerm(%arg0: tensor<1x1x9x9xf16, {order = #NCWH}>) -> tensor<9x9x1x1xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x1x9x9xf16, {order = #NCWH}> -> tensor<1x1x9x9xf16>
    %1 = IE.Reshape(%0) {shape_value = [9, 9, 1, 1]} : tensor<1x1x9x9xf16> -> tensor<9x9x1x1xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<9x9x1x1xf16> -> tensor<9x9x1x1xf16, {order = #NHWC}>
    return %2 : tensor<9x9x1x1xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%.+]] = IE.Reorder(%arg0)
    // CHECK:       [[VAR1:%.+]] = IE.Reshape([[VAR0]])
    // CHECK:       [[VAR2:%.+]] = IE.Reorder([[VAR1]])
    // CHECK:       return [[VAR2]] : tensor<9x9x1x1xf16, {order = #NHWC}>
}

// CHECK-LABEL: @MoveReorderThroughReshapeWithContinuousMem
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<8000x256x1x1xf16, {order = #map}>)
func.func @MoveReorderThroughReshapeWithContinuousMem(%arg0: tensor<8000x256x1x1xf16, {order = #map}>) -> tensor<1x8000x16x16xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<8000x256x1x1xf16, {order = #map}> -> tensor<8000x256x1x1xf16>
    %1 = IE.Reshape(%0) {shape_value = [1, 8000, 16, 16]} : tensor<8000x256x1x1xf16> -> tensor<1x8000x16x16xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x8000x16x16xf16> -> tensor<1x8000x16x16xf16, {order = #NHWC}>
    return %2 : tensor<1x8000x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%.+]] = IE.ShapeCast {shape = [8000, 16, 1, 16]}
    // CHECK-SAME:      inputs([[INPUT]] : tensor<8000x256x1x1xf16, {order = #map}>) -> tensor<8000x16x1x16xf16, {order = #map}>
    // CHECK:       [[VAR1:%.+]] = IE.PermuteCast([[VAR0]]) {dst_order = #NCHW, mem_perm = #NCHW} :
    // CHECK-SAME:      tensor<8000x16x1x16xf16, {order = #map}> -> tensor<1x8000x16x16xf16>
    // CHECK:       [[VAR2:%.+]] = IE.Reorder([[VAR1]])
    // CHECK:       return [[VAR2]] : tensor<1x8000x16x16xf16, {order = #NHWC}>
}

// CHECK-LABEL: @NotMoveReorderThroughReshapeWithNoContinuousMem
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x8000x16x16xf16, {order = #NHWC}>)
func.func @NotMoveReorderThroughReshapeWithNoContinuousMem(%arg0: tensor<1x8000x16x16xf16, {order = #NHWC}>) -> tensor<1x4000x32x16xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x8000x16x16xf16, {order = #NHWC}> -> tensor<1x8000x16x16xf16>
    %1 = IE.Reshape(%0) {shape_value = [1, 4000, 32, 16]} : tensor<1x8000x16x16xf16> -> tensor<1x4000x32x16xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x4000x32x16xf16> -> tensor<1x4000x32x16xf16, {order = #NHWC}>
    return %2 : tensor<1x4000x32x16xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%.+]] = IE.Reorder([[INPUT]])
    // CHECK:       [[VAR1:%.+]] = IE.Reshape([[VAR0]])
    // CHECK:       [[VAR2:%.+]] = IE.Reorder([[VAR1]])
    // CHECK:       return [[VAR2]] : tensor<1x4000x32x16xf16, {order = #NHWC}>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

module @ReorderWithAffineReshape {

// CHECK-LABEL: @MoveReorderThroughAffineReshapeWithCompatibleMem
func.func @MoveReorderThroughAffineReshapeWithCompatibleMem(%arg0: tensor<1x1x64x128xf16, {order = #NWHC}>) -> tensor<1x64x128x1xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x1x64x128xf16, {order = #NWHC}> -> tensor<1x1x64x128xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 64, 128, 1]} : tensor<1x1x64x128xf16> -> tensor<1x64x128x1xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x64x128x1xf16> -> tensor<1x64x128x1xf16, {order = #NHWC}>
    return %2 : tensor<1x64x128x1xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%.+]] = IE.ShapeCast {shape = [1, 64, 128, 1]}
    // CHECK-SAME:      inputs(%arg0 : tensor<1x1x64x128xf16, {order = #NWHC}>) -> tensor<1x64x128x1xf16, {order = #NWHC}>
    // CHECK:       [[VAR1:%.+]] = IE.Reorder([[VAR0]]) {dstOrder = #NHWC} :
    // CHECK-SAME:      tensor<1x64x128x1xf16, {order = #NWHC}> -> tensor<1x64x128x1xf16, {order = #NHWC}>
    // CHECK:       return [[VAR1]]
}

// CHECK-LABEL: @NotMoveReorderThroughAffineReshapeWithnotCompatibleMem
func.func @NotMoveReorderThroughAffineReshapeWithnotCompatibleMem(%arg0: tensor<1x1x9x9xf16, {order = #NCWH}>) -> tensor<9x9x1x1xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x1x9x9xf16, {order = #NCWH}> -> tensor<1x1x9x9xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [9, 9, 1, 1]} : tensor<1x1x9x9xf16> -> tensor<9x9x1x1xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<9x9x1x1xf16> -> tensor<9x9x1x1xf16, {order = #NHWC}>
    return %2 : tensor<9x9x1x1xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%.+]] = IE.Reorder(%arg0)
    // CHECK:       [[VAR1:%.+]] = IE.AffineReshape([[VAR0]])
    // CHECK:       [[VAR2:%.+]] = IE.Reorder([[VAR1]])
    // CHECK:       return [[VAR2]] : tensor<9x9x1x1xf16, {order = #NHWC}>
}

}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d2, d0, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @ReorderWithShapeCast {

// CHECK-LABEL: @MoveReorderThroughShapeCast
func.func @MoveReorderThroughShapeCast(%arg0: tensor<1x32x2x4xf16, {order = #map}>) -> tensor<2x8x8x2xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NWCH} : tensor<1x32x2x4xf16, {order = #map}> -> tensor<1x32x2x4xf16, {order = #NWCH}>
    %1 = IE.ShapeCast {shape = [2, 8, 8, 2]} inputs(%0 : tensor<1x32x2x4xf16, {order = #NWCH}>) -> tensor<2x8x8x2xf16, {order = #NWCH}>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<2x8x8x2xf16, {order = #NWCH}> -> tensor<2x8x8x2xf16, {order = #NHWC}>
    return %2 : tensor<2x8x8x2xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%.+]] = IE.ShapeCast {shape = [2, 8, 8, 2]}
    // CHECK-SAME:      inputs(%arg0 : tensor<1x32x2x4xf16, {order = #map}>) -> tensor<2x8x8x2xf16, {order = #map}>
    // CHECK:       [[VAR1:%.+]] = IE.Reorder([[VAR0]]) {dstOrder = #NHWC}
    // CHECK-SAME:      : tensor<2x8x8x2xf16, {order = #map}> -> tensor<2x8x8x2xf16, {order = #NHWC}>
    // CHECK:       return [[VAR1]] : tensor<2x8x8x2xf16, {order = #NHWC}>
}

// CHECK-LABEL: @NotMoveReorderThroughShapeCast
func.func @NotMoveReorderThroughShapeCast(%arg0: tensor<1x32x8x2xf16, {order = #NCWH}>) -> tensor<1x32x4x4xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NWCH} : tensor<1x32x8x2xf16, {order = #NCWH}> -> tensor<1x32x8x2xf16, {order = #NWCH}>
    %1 = IE.ShapeCast {shape = [1, 32, 4, 4]} inputs(%0 : tensor<1x32x8x2xf16, {order = #NWCH}>) -> tensor<1x32x4x4xf16, {order = #NWCH}>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x32x4x4xf16, {order = #NWCH}> -> tensor<1x32x4x4xf16, {order = #NHWC}>
    return %2 : tensor<1x32x4x4xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%.+]] = IE.Reorder(%arg0)
    // CHECK:       [[VAR1:%.+]] = IE.ShapeCast
    // CHECK:       [[VAR2:%.+]] = IE.Reorder([[VAR1]])
    // CHECK:       return [[VAR2]] : tensor<1x32x4x4xf16, {order = #NHWC}>
}

}

// -----

#NDHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4, d1)>
#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2, d3)>

// CHECK-LABEL: @ReorderWithPermuteCast
module @ReorderWithPermuteCast {

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x33x120x1x64xf16, {order = #NDHWC}>) -> tensor<1x120x2x64x33xf16> {
func.func @main(%arg0: tensor<1x33x120x1x64xf16, {order = #NDHWC}>) -> (tensor<1x120x2x64x33xf16>) {
    %0 = IE.Concat(%arg0, %arg0) {static_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 1, 0]]} : tensor<1x33x120x1x64xf16, {order = #NDHWC}>, tensor<1x33x120x1x64xf16, {order = #NDHWC}> -> tensor<1x33x120x2x64xf16, {order = #NDHWC}>
    %1 = IE.Reorder(%0) {dstOrder = #NCDHW} : tensor<1x33x120x2x64xf16, {order = #NDHWC}> -> tensor<1x33x120x2x64xf16>
    %2 = IE.PermuteCast(%1) {dst_order = #map, mem_perm = #NCDHW} : tensor<1x33x120x2x64xf16> -> tensor<1x120x2x64x33xf16, {order = #map}>
    %3 = IE.Reorder(%2) {dstOrder = #NCDHW} : tensor<1x120x2x64x33xf16, {order = #map}> -> tensor<1x120x2x64x33xf16>
    return %3 : tensor<1x120x2x64x33xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[ARG0]], [[ARG0]])
    // CHECK:       [[PERMUTECAST:%.+]] = IE.PermuteCast([[CONCAT]]) {dst_order = #NCDHW, mem_perm = #NCDHW} : tensor<1x33x120x2x64xf16, {order = #NDHWC}>
    // CHECK-SAME:      -> tensor<1x120x2x64x33xf16>

    // CHECK:       return [[PERMUTECAST]] : tensor<1x120x2x64x33xf16>
}

}



// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithConcatWithMoreReorderInputs
module @ReorderWithConcatWithMoreReorderInputs {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<1x1x16x16xf16>,
// CHECK-SAME:      [[ARG1:%arg[0-9]+]]: tensor<1x4x16x16xf16, {order = #NHWC}>,
// CHECK-SAME:      [[ARG2:%arg[0-9]+]]: tensor<1x4x16x16xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x1x16x16xf16>, %arg1: tensor<1x4x16x16xf16, {order = #NHWC}>, %arg2: tensor<1x4x16x16xf16, {order = #NHWC}>)
        -> tensor<1x7x16x16xf16, {order = #NHWC}> {
    %0 = IE.Slice %arg1 [0, 0, 0, 0] [1, 3, 16, 16] : tensor<1x4x16x16xf16, {order = #NHWC}> to tensor<1x3x16x16xf16, {order = #NHWC}>
    %1 = IE.Slice %arg2 [0, 0, 0, 0] [1, 3, 16, 16] : tensor<1x4x16x16xf16, {order = #NHWC}> to tensor<1x3x16x16xf16, {order = #NHWC}>
    %2 = IE.Reorder(%0) {dstOrder = #NCHW} : tensor<1x3x16x16xf16, {order = #NHWC}> -> tensor<1x3x16x16xf16>
    %3 = IE.Reorder(%1) {dstOrder = #NCHW} : tensor<1x3x16x16xf16, {order = #NHWC}> -> tensor<1x3x16x16xf16>
    %4 = IE.Concat(%arg0, %2, %3) {per_axis = #IE.Concat<axis = 1>} : tensor<1x1x16x16xf16>, tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x7x16x16xf16>
    %5 = IE.Reorder(%4) {dstOrder = #NHWC} : tensor<1x7x16x16xf16> -> tensor<1x7x16x16xf16, {order = #NHWC}>
    return %5 : tensor<1x7x16x16xf16, {order = #NHWC}>

    // CHECK:  [[REORDER_INPUT0:%.*]] = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x1x16x16xf16> -> tensor<1x1x16x16xf16, {order = #NHWC}>
    // CHECK:  [[REORDER_SLICE0:%.*]] = IE.Slice %arg1 [0, 0, 0, 0] [1, 3, 16, 16] : tensor<1x4x16x16xf16, {order = #NHWC}> to tensor<1x3x16x16xf16, {order = #NHWC}>
    // CHECK:  [[REORDER_SLICE1:%.*]] = IE.Slice %arg2 [0, 0, 0, 0] [1, 3, 16, 16] : tensor<1x4x16x16xf16, {order = #NHWC}> to tensor<1x3x16x16xf16, {order = #NHWC}>
    // CHECK:  [[CONCAT:%.*]] = IE.Concat([[REORDER_INPUT0]], [[REORDER_SLICE0]], [[REORDER_SLICE1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x16x16xf16, {order = #NHWC}>, tensor<1x3x16x16xf16, {order = #NHWC}>, tensor<1x3x16x16xf16, {order = #NHWC}> -> tensor<1x7x16x16xf16, {order = #NHWC}>
    // CHECK:  return [[CONCAT]] : tensor<1x7x16x16xf16, {order = #NHWC}>
}
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotOptReorderWithConcatWithLessReorderInputs
module @NotOptReorderWithConcatWithLessReorderInputs {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<1x1x16x16xf16>,
// CHECK-SAME:      [[ARG1:%arg[0-9]+]]: tensor<1x4x16x16xf16>,
// CHECK-SAME:      [[ARG2:%arg[0-9]+]]: tensor<1x4x16x16xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x1x16x16xf16>, %arg1: tensor<1x4x16x16xf16>, %arg2: tensor<1x4x16x16xf16, {order = #NHWC}>)
        -> tensor<1x7x16x16xf16, {order = #NHWC}> {
    %0 = IE.Slice %arg1 [0, 0, 0, 0] [1, 3, 16, 16] : tensor<1x4x16x16xf16> to tensor<1x3x16x16xf16>
    %1 = IE.Slice %arg2 [0, 0, 0, 0] [1, 3, 16, 16] : tensor<1x4x16x16xf16, {order = #NHWC}> to tensor<1x3x16x16xf16, {order = #NHWC}>
    %2 = IE.Reorder(%1) {dstOrder = #NCHW} : tensor<1x3x16x16xf16, {order = #NHWC}> -> tensor<1x3x16x16xf16>
    %3 = IE.Concat(%arg0, %0, %2) {per_axis = #IE.Concat<axis = 1>} : tensor<1x1x16x16xf16>, tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x7x16x16xf16>
    %4 = IE.Reorder(%3) {dstOrder = #NHWC} : tensor<1x7x16x16xf16> -> tensor<1x7x16x16xf16, {order = #NHWC}>
    return %4 : tensor<1x7x16x16xf16, {order = #NHWC}>

    // CHECK:  [[REORDER_SLICE0:%.*]] = IE.Slice %arg1 [0, 0, 0, 0] [1, 3, 16, 16] : tensor<1x4x16x16xf16> to tensor<1x3x16x16xf16>
    // CHECK:  [[REORDER_SLICE1:%.*]] = IE.Slice %arg2 [0, 0, 0, 0] [1, 3, 16, 16] : tensor<1x4x16x16xf16, {order = #NHWC}> to tensor<1x3x16x16xf16, {order = #NHWC}>
    // CHECK:  [[REORDER_INPUT0:%.*]] = IE.Reorder([[REORDER_SLICE1]]) {dstOrder = #NCHW} : tensor<1x3x16x16xf16, {order = #NHWC}> -> tensor<1x3x16x16xf16>
    // CHECK:  [[CONCAT:%.*]] = IE.Concat(%arg0, [[REORDER_SLICE0]], [[REORDER_INPUT0]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x16x16xf16>, tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x7x16x16xf16>
    // CHECK:  [[REORDER_OUTPUT:%.*]] = IE.Reorder([[CONCAT]]) {dstOrder = #NHWC} : tensor<1x7x16x16xf16> -> tensor<1x7x16x16xf16, {order = #NHWC}>
    // CHECK:  return [[REORDER_OUTPUT]] : tensor<1x7x16x16xf16, {order = #NHWC}>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

module @ReorderWithTile {

// CHECK-LABEL: @SwitchReorderWithTileForBetterTileDMAPerfCase1
func.func @SwitchReorderWithTileForBetterTileDMAPerfCase1(%arg0: tensor<1x1x11x11xf16>) -> tensor<1x64x11x11xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x1x11x11xf16> -> tensor<1x1x11x11xf16, {order = #NHWC}>
    %1 = IE.Tile(%0) {repeats_values = [1, 64, 1, 1]} : tensor<1x1x11x11xf16, {order = #NHWC}> -> tensor<1x64x11x11xf16, {order = #NHWC}>
    return %1 : tensor<1x64x11x11xf16, {order = #NHWC}>

    // CHECK:       [[TILE:%.+]] = IE.Tile({{[^:]+}}) {repeats_values = [1, 64, 1, 1]} : tensor<1x1x11x11xf16> -> tensor<1x64x11x11xf16>
    // CHECK:       [[REORDER:%.+]] = IE.Reorder([[TILE]]) {dstOrder = #NHWC} : tensor<1x64x11x11xf16> -> tensor<1x64x11x11xf16, {order = #NHWC}>
    // CHECK:       return [[REORDER]] : tensor<1x64x11x11xf16, {order = #NHWC}>
}

// CHECK-LABEL: @SwitchReorderWithTileForBetterTileDMAPerfCase2
func.func @SwitchReorderWithTileForBetterTileDMAPerfCase2(%arg0: tensor<1x64x1x1xf16>) -> tensor<1x64x11x11xf16, {order = #NHWC}> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 11, 11]} : tensor<1x64x1x1xf16> -> tensor<1x64x11x11xf16>
    %1 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x64x11x11xf16> -> tensor<1x64x11x11xf16, {order = #NHWC}>
    return %1 : tensor<1x64x11x11xf16, {order = #NHWC}>

    // CHECK:       [[REORDER:%.+]] = IE.Reorder({{[^:]+}}) {dstOrder = #NHWC} : tensor<1x64x1x1xf16> -> tensor<1x64x1x1xf16, {order = #NHWC}>
    // CHECK:       [[TILE:%.+]] = IE.Tile([[REORDER]]) {repeats_values = [1, 1, 11, 11]} : tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x11x11xf16, {order = #NHWC}>
    // CHECK:       return [[TILE]] : tensor<1x64x11x11xf16, {order = #NHWC}>
}

// CHECK-LABEL: @FusingNonTrivialReorderAroundTile
func.func @FusingNonTrivialReorderAroundTile(%arg0: tensor<1x1x11x11xf16>) -> tensor<1x64x11x11xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x1x11x11xf16> -> tensor<1x1x11x11xf16, {order = #NHWC}>
    %1 = IE.Tile(%0) {repeats_values = [1, 64, 1, 1]} : tensor<1x1x11x11xf16, {order = #NHWC}> -> tensor<1x64x11x11xf16, {order = #NHWC}>
    %2 = IE.Reorder(%1) {dstOrder = #NCHW} : tensor<1x64x11x11xf16, {order = #NHWC}> -> tensor<1x64x11x11xf16>
    return %2 : tensor<1x64x11x11xf16>

    // CHECK:       [[TILE:%.+]] = IE.Tile({{[^:]+}}) {repeats_values = [1, 64, 1, 1]} : tensor<1x1x11x11xf16> -> tensor<1x64x11x11xf16>
    // CHECK:       return [[TILE]] : tensor<1x64x11x11xf16>
}

// CHECK-LABEL: @NotSwitchReorderWithTile
func.func @NotSwitchReorderWithTile(%arg0: tensor<1x64x1x1xf16>) -> tensor<1x64x11x11xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x64x1x1xf16> -> tensor<1x64x1x1xf16, {order = #NHWC}>
    %1 = IE.Tile(%0) {repeats_values = [1, 1, 11, 11]} : tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x11x11xf16, {order = #NHWC}>
    return %1 : tensor<1x64x11x11xf16, {order = #NHWC}>

    // CHECK:       [[REORDER:%.+]] = IE.Reorder({{[^:]+}}) {dstOrder = #NHWC} : tensor<1x64x1x1xf16> -> tensor<1x64x1x1xf16, {order = #NHWC}>
    // CHECK:       [[TILE:%.+]] = IE.Tile([[REORDER]]) {repeats_values = [1, 1, 11, 11]} : tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x11x11xf16, {order = #NHWC}>
    // CHECK:       return [[TILE]] : tensor<1x64x11x11xf16, {order = #NHWC}>

}

// CHECK-LABEL: @NotSwitchTrivialReorderWithTile
func.func @NotSwitchTrivialReorderWithTile(%arg0: tensor<1x64x1x1xf16>) -> tensor<1x64x11x11xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x64x1x1xf16> -> tensor<1x64x1x1xf16, {order = #NHWC}>
    %1 = IE.Tile(%0) {repeats_values = [1, 1, 11, 11]} : tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x11x11xf16, {order = #NHWC}>
    %2 = IE.Reorder(%1) {dstOrder = #NCHW} : tensor<1x64x11x11xf16, {order = #NHWC}> -> tensor<1x64x11x11xf16>
    return %2 : tensor<1x64x11x11xf16>

    // CHECK:       [[REORDER:%.+]] = IE.Reorder({{[^:]+}}) {dstOrder = #NHWC} : tensor<1x64x1x1xf16> -> tensor<1x64x1x1xf16, {order = #NHWC}>
    // CHECK:       [[TILE:%.+]] = IE.Tile([[REORDER]]) {repeats_values = [1, 1, 11, 11]} : tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x11x11xf16, {order = #NHWC}>
    // CHECK:       [[RES:%.+]] = IE.Reorder([[TILE]]) {dstOrder = #NCHW} : tensor<1x64x11x11xf16, {order = #NHWC}> -> tensor<1x64x11x11xf16>
    // CHECK:       return [[RES]] : tensor<1x64x11x11xf16>
}

}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptReorderIfNonReorderInputHasMultiUsers
module @OptReorderIfNonReorderInputHasMultiUsers {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<1x1x16x16xf16>,
// CHECK-SAME:      [[ARG1:%arg[0-9]+]]: tensor<1x4x16x16xf16, {order = #NHWC}>,
// CHECK-SAME:      [[ARG2:%arg[0-9]+]]: tensor<1x4x16x16xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x1x16x16xf16>, %arg1: tensor<1x4x16x16xf16, {order = #NHWC}>, %arg2: tensor<1x4x16x16xf16, {order = #NHWC}>)
        -> (tensor<1x1x8x32xf16>, tensor<1x7x16x16xf16, {order = #NHWC}>) {
    %0 = IE.Slice %arg1 [0, 0, 0, 0] [1, 3, 16, 16] : tensor<1x4x16x16xf16, {order = #NHWC}> to tensor<1x3x16x16xf16, {order = #NHWC}>
    %1 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1, 8, 32]} : tensor<1x1x16x16xf16> -> tensor<1x1x8x32xf16>
    %2 = IE.Slice %arg2 [0, 0, 0, 0] [1, 3, 16, 16] : tensor<1x4x16x16xf16, {order = #NHWC}> to tensor<1x3x16x16xf16, {order = #NHWC}>
    %3 = IE.Reorder(%0) {dstOrder = #NCHW} : tensor<1x3x16x16xf16, {order = #NHWC}> -> tensor<1x3x16x16xf16>
    %4 = IE.Reorder(%2) {dstOrder = #NCHW} : tensor<1x3x16x16xf16, {order = #NHWC}> -> tensor<1x3x16x16xf16>
    %5 = IE.Concat(%arg0, %3, %4) {per_axis = #IE.Concat<axis = 1>} : tensor<1x1x16x16xf16>, tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x7x16x16xf16>
    %6 = IE.Reorder(%5) {dstOrder = #NHWC} : tensor<1x7x16x16xf16> -> tensor<1x7x16x16xf16, {order = #NHWC}>
    return %1, %6 : tensor<1x1x8x32xf16>, tensor<1x7x16x16xf16, {order = #NHWC}>

    // CHECK:  [[INPUT_REORDER:%.*]] = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x1x16x16xf16> -> tensor<1x1x16x16xf16, {order = #NHWC}>
    // CHECK:  [[INPUT_SLICE_1:%.*]] = IE.Slice %arg1 [0, 0, 0, 0] [1, 3, 16, 16] : tensor<1x4x16x16xf16, {order = #NHWC}> to tensor<1x3x16x16xf16, {order = #NHWC}>
    // CHECK:  [[AFFINERESHAPE_OUTPUT:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1, 8, 32]} : tensor<1x1x16x16xf16> -> tensor<1x1x8x32xf16>
    // CHECK:  [[INPUT_SLICE_2:%.*]] = IE.Slice %arg2 [0, 0, 0, 0] [1, 3, 16, 16] : tensor<1x4x16x16xf16, {order = #NHWC}> to tensor<1x3x16x16xf16, {order = #NHWC}>
    // CHECK:  [[CONCAT_OUTPUT:%.*]] = IE.Concat([[INPUT_REORDER]], [[INPUT_SLICE_1]], [[INPUT_SLICE_2]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x16x16xf16, {order = #NHWC}>, tensor<1x3x16x16xf16, {order = #NHWC}>, tensor<1x3x16x16xf16, {order = #NHWC}> -> tensor<1x7x16x16xf16, {order = #NHWC}>
    // CHECK:  return [[AFFINERESHAPE_OUTPUT]], [[CONCAT_OUTPUT]]   : tensor<1x1x8x32xf16>, tensor<1x7x16x16xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithExpandSlice
module @ReorderWithExpandSlice {

// CHECK:       func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x1x1x32032xf16>)
func.func @main(%arg0: tensor<1x1x1x32032xf16>) -> tensor<1x16x1x31995xf16, {order = #NHWC}> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x32032xf16> -> tensor<1x16x1x32032xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 16, 1, 31995] : tensor<1x16x1x32032xf16> to tensor<1x16x1x31995xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x16x1x31995xf16> -> tensor<1x16x1x31995xf16, {order = #NHWC}>

    return %2 : tensor<1x16x1x31995xf16, {order = #NHWC}>

    // CHECK:       [[REORDER_0:%.*]] = IE.Reorder(%arg0) {dstOrder = #NHWC}
    // CHECK-SAME:      tensor<1x1x1x32032xf16> -> tensor<1x1x1x32032xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.*]] = IE.Expand([[REORDER_0]])
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0]
    // CHECK-SAME:      pads_end = [0, 15, 0, 0]
    // CHECK-SAME:      : tensor<1x1x1x32032xf16, {order = #NHWC}> -> tensor<1x16x1x32032xf16, {order = #NHWC}>

    // CHECK:       [[SLICE:%.*]] = IE.Slice [[EXPAND]] [0, 0, 0, 0] [1, 16, 1, 31995] :
    // CHECK-SAME:      tensor<1x16x1x32032xf16, {order = #NHWC}> to tensor<1x16x1x31995xf16, {order = #NHWC}>

    // CHECK:      return [[SLICE]] : tensor<1x16x1x31995xf16, {order = #NHWC}>

}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithExpandMultiSlices
module @ReorderWithExpandMultiSlices {

// CHECK:       func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x1x1x32032xf16>)
func.func @main(%arg0: tensor<1x1x1x32032xf16>) -> (tensor<1x16x1x31995xf16, {order = #NHWC}>, tensor<1x16x1x31995xf16, {order = #NHWC}>) {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x32032xf16> -> tensor<1x16x1x32032xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 16, 1, 31995] : tensor<1x16x1x32032xf16> to tensor<1x16x1x31995xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x16x1x31995xf16> -> tensor<1x16x1x31995xf16, {order = #NHWC}>
    %3 = IE.Slice %0 [0, 0, 0, 11] [1, 16, 1, 31995] : tensor<1x16x1x32032xf16> to tensor<1x16x1x31995xf16>
    %4 = IE.Reorder(%3) {dstOrder = #NHWC} : tensor<1x16x1x31995xf16> -> tensor<1x16x1x31995xf16, {order = #NHWC}>

    return %2, %4 : tensor<1x16x1x31995xf16, {order = #NHWC}>, tensor<1x16x1x31995xf16, {order = #NHWC}>

    // CHECK:       [[REORDER_0:%.*]] = IE.Reorder(%arg0) {dstOrder = #NHWC}
    // CHECK-SAME:      tensor<1x1x1x32032xf16> -> tensor<1x1x1x32032xf16, {order = #NHWC}>

    // CHECK:       [[EXPAND:%.*]] = IE.Expand([[REORDER_0]])
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0]
    // CHECK-SAME:      pads_end = [0, 15, 0, 0]
    // CHECK-SAME:      : tensor<1x1x1x32032xf16, {order = #NHWC}> -> tensor<1x16x1x32032xf16, {order = #NHWC}>

    // CHECK:       [[SLICE_0:%.*]] = IE.Slice [[EXPAND]] [0, 0, 0, 11] [1, 16, 1, 31995] :
    // CHECK-SAME:      tensor<1x16x1x32032xf16, {order = #NHWC}> to tensor<1x16x1x31995xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_1:%.*]] = IE.Slice [[EXPAND]] [0, 0, 0, 0] [1, 16, 1, 31995] :
    // CHECK-SAME:      tensor<1x16x1x32032xf16, {order = #NHWC}> to tensor<1x16x1x31995xf16, {order = #NHWC}>

    // CHECK:      return [[SLICE_1]], [[SLICE_0]] : tensor<1x16x1x31995xf16, {order = #NHWC}>, tensor<1x16x1x31995xf16, {order = #NHWC}>

}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotCleanUpReorderWithExpandSlice
module @NotCleanUpReorderWithExpandSlice {

// CHECK:       func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x1x1x32032xf16>)
func.func @main(%arg0: tensor<1x1x1x32032xf16>) -> (tensor<1x16x1x200xf16, {order = #NHWC}>, tensor<1x16x1x200xf16, {order = #NHWC}>) {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x32032xf16> -> tensor<1x16x1x32032xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 16, 1, 200] : tensor<1x16x1x32032xf16> to tensor<1x16x1x200xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x16x1x200xf16> -> tensor<1x16x1x200xf16, {order = #NHWC}>
    %3 = IE.Slice %0 [0, 0, 0, 11] [1, 16, 1, 200] : tensor<1x16x1x32032xf16> to tensor<1x16x1x200xf16>
    %4 = IE.Reorder(%3) {dstOrder = #NHWC} : tensor<1x16x1x200xf16> -> tensor<1x16x1x200xf16, {order = #NHWC}>

    return %2, %4 : tensor<1x16x1x200xf16, {order = #NHWC}>, tensor<1x16x1x200xf16, {order = #NHWC}>

    // CHECK:       [[EXPAND:%.*]] = IE.Expand(%arg0)
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0]
    // CHECK-SAME:      pads_end = [0, 15, 0, 0]
    // CHECK-SAME:      :  tensor<1x1x1x32032xf16> -> tensor<1x16x1x32032xf16>

    // CHECK:       [[SLICE_0:%.*]] = IE.Slice [[EXPAND]] [0, 0, 0, 0] [1, 16, 1, 200] :
    // CHECK-SAME:      tensor<1x16x1x32032xf16> to tensor<1x16x1x200xf16>
    // CHECK:       [[REORDER_0:%.*]] = IE.Reorder([[SLICE_0]]) {dstOrder = #NHWC}
    // CHECK-SAME:      tensor<1x16x1x200xf16> -> tensor<1x16x1x200xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_1:%.*]] = IE.Slice [[EXPAND]] [0, 0, 0, 11] [1, 16, 1, 200] :
    // CHECK-SAME:      tensor<1x16x1x32032xf16> to tensor<1x16x1x200xf16>
    // CHECK:       [[REORDER_1:%.*]] = IE.Reorder([[SLICE_1]]) {dstOrder = #NHWC}
    // CHECK-SAME:      tensor<1x16x1x200xf16> -> tensor<1x16x1x200xf16, {order = #NHWC}>
    // CHECK:      return [[REORDER_0]], [[REORDER_1]] : tensor<1x16x1x200xf16, {order = #NHWC}>, tensor<1x16x1x200xf16, {order = #NHWC}>

}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithPartOfExpandSlicePattern
module @ReorderWithPartOfExpandSlicePattern {

// CHECK:       func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x1x1x32032xf16>)
func.func @main(%arg0: tensor<1x1x1x32032xf16>) -> (tensor<1x16x1x31995xf16, {order = #NHWC}>, tensor<1x16x1x32032xf32>) {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x32032xf16> -> tensor<1x16x1x32032xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 16, 1, 31995] : tensor<1x16x1x32032xf16> to tensor<1x16x1x31995xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x16x1x31995xf16> -> tensor<1x16x1x31995xf16, {order = #NHWC}>
    %3 = IE.Convert(%0) {dstElemType = f32} : tensor<1x16x1x32032xf16> -> tensor<1x16x1x32032xf32>


    return %2, %3 : tensor<1x16x1x31995xf16, {order = #NHWC}>, tensor<1x16x1x32032xf32>

    // CHECK:       [[EXPAND:%.*]] = IE.Expand(%arg0)
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0]
    // CHECK-SAME:      pads_end = [0, 15, 0, 0]
    // CHECK-SAME:      :  tensor<1x1x1x32032xf16> -> tensor<1x16x1x32032xf16>

    // CHECK:       [[SLICE_0:%.*]] = IE.Slice [[EXPAND]] [0, 0, 0, 0] [1, 16, 1, 31995] :
    // CHECK-SAME:      tensor<1x16x1x32032xf16> to tensor<1x16x1x31995xf16>
    // CHECK:       [[REORDER_0:%.*]] = IE.Reorder([[SLICE_0]]) {dstOrder = #NHWC}
    // CHECK-SAME:     tensor<1x16x1x31995xf16> -> tensor<1x16x1x31995xf16, {order = #NHWC}>
    // CHECK:       [[CONVERT_0:%.*]] = IE.Convert([[EXPAND]]) {dstElemType = f32} : tensor<1x16x1x32032xf16> -> tensor<1x16x1x32032xf32>

    // CHECK:      return [[REORDER_0]], [[CONVERT_0]] : tensor<1x16x1x31995xf16, {order = #NHWC}>, tensor<1x16x1x32032xf32>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotSwapReorderWithConvert
module @NotSwapReorderWithConvert{
// CHECK:       func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x16x1x31995xui8>)
func.func @main(%arg0: tensor<1x16x1x31995xui8>) -> tensor<1x16x1x31995xf16, {order = #NHWC}> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x16x1x31995xui8> -> tensor<1x16x1x31995xf16>
    %1 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x16x1x31995xf16> -> tensor<1x16x1x31995xf16, {order = #NHWC}>
    return %1 : tensor<1x16x1x31995xf16, {order = #NHWC}>

    // CHECK:       [[CONVERT:%.*]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x16x1x31995xui8> -> tensor<1x16x1x31995xf16>
    // CHECK:       [[REORDER:%.*]] = IE.Reorder([[CONVERT]]) {dstOrder = #NHWC} : tensor<1x16x1x31995xf16> -> tensor<1x16x1x31995xf16, {order = #NHWC}>
    // CHECK:       return [[REORDER]] : tensor<1x16x1x31995xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithoutExpandSlicePattern
module @ReorderWithoutExpandSlicePattern {

// CHECK:       func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x1x1x31995xf32>)
func.func @main(%arg0: tensor<1x1x1x31995xf32>) -> (tensor<1x16x1x31995xf16, {order = #NHWC}>, tensor<1x16x1x31995xf16, {order = #NHWC}>) {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x31995xf32> -> tensor<1x16x1x31995xf32>
    %1 = IE.Convert(%0) {dstElemType = f16} : tensor<1x16x1x31995xf32> -> tensor<1x16x1x31995xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x16x1x31995xf16> -> tensor<1x16x1x31995xf16, {order = #NHWC}>
    %3 = IE.Convert(%0) {dstElemType = f16} : tensor<1x16x1x31995xf32> -> tensor<1x16x1x31995xf16>
    %4 = IE.Reorder(%3) {dstOrder = #NHWC} : tensor<1x16x1x31995xf16> -> tensor<1x16x1x31995xf16, {order = #NHWC}>

    return %2, %4 : tensor<1x16x1x31995xf16, {order = #NHWC}>, tensor<1x16x1x31995xf16, {order = #NHWC}>

    // CHECK:       [[EXPAND:%.*]] = IE.Expand(%arg0)
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0]
    // CHECK-SAME:      pads_end = [0, 15, 0, 0]
    // CHECK-SAME:      :  tensor<1x1x1x31995xf32> -> tensor<1x16x1x31995xf32>

    // CHECK:       [[CONVERT_0:%.*]] = IE.Convert([[EXPAND]]) {dstElemType = f16} : tensor<1x16x1x31995xf32> -> tensor<1x16x1x31995xf16>
    // CHECK:       [[REORDER_0:%.*]] = IE.Reorder([[CONVERT_0]]) {dstOrder = #NHWC}
    // CHECK-SAME:      tensor<1x16x1x31995xf16> -> tensor<1x16x1x31995xf16, {order = #NHWC}>
    // CHECK:       [[CONVERT_1:%.*]] = IE.Convert([[EXPAND]]) {dstElemType = f16} : tensor<1x16x1x31995xf32> -> tensor<1x16x1x31995xf16>
    // CHECK:       [[REORDER_1:%.*]] = IE.Reorder([[CONVERT_1]]) {dstOrder = #NHWC}
    // CHECK-SAME:      tensor<1x16x1x31995xf16> -> tensor<1x16x1x31995xf16, {order = #NHWC}>
    // CHECK:      return [[REORDER_0]], [[REORDER_1]] : tensor<1x16x1x31995xf16, {order = #NHWC}>, tensor<1x16x1x31995xf16, {order = #NHWC}>
}
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#HWC = affine_map<(d0, d1, d2) -> (d1, d2, d0)>

// CHECK-LABEL: @ReorderWithHeterogeneousUsers
func.func @ReorderWithHeterogeneousUsers(%arg0: tensor<1x65x4620xf16, {order = #HWC}>) -> (tensor<1x64x4620xf16>, tensor<65x4620xf16>) {
    %0 = IE.Reorder(%arg0) {dstOrder = #CHW} : tensor<1x65x4620xf16, {order = #HWC}> -> tensor<1x65x4620xf16>
    %1 = IE.Slice %0 [0, 0, 0] [1, 64, 4620] : tensor<1x65x4620xf16> to tensor<1x64x4620xf16>
    %2 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1]], shape_value = [65, 4620]} : tensor<1x65x4620xf16> -> tensor<65x4620xf16>

    return %1, %2 : tensor<1x64x4620xf16>, tensor<65x4620xf16>

    // CHECK:  [[REORDER_0:%.*]] = IE.Reorder(%arg0)
    // CHECK-SAME:      tensor<1x65x4620xf16, {order = #HWC}> -> tensor<1x65x4620xf16>
    // CHECK:  [[SLICE_0:%.*]] = IE.Slice %arg0 [0, 0, 0] [1, 64, 4620] : tensor<1x65x4620xf16, {order = #HWC}> to tensor<1x64x4620xf16, {order = #HWC}>
    // CHECK:  [[REORDER_1:%.*]] = IE.Reorder([[SLICE_0]])
    // CHECK-SAME:      tensor<1x64x4620xf16, {order = #HWC}> -> tensor<1x64x4620xf16>
    // CHECK:  [[RESHAPE:%.*]] = IE.AffineReshape([[REORDER_0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1]], shape_value = [65, 4620]} : tensor<1x65x4620xf16> -> tensor<65x4620xf16>

    // CHECK:       return [[REORDER_1]], [[RESHAPE]] : tensor<1x64x4620xf16>, tensor<65x4620xf16>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithLayerOtherInputAddTrivialReorder
func.func @ReorderWithLayerOtherInputAddTrivialReorder(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>, %arg1: tensor<1x1x30x30xf16>)
        -> tensor<1x3x30x30xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x3x30x30xf16>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x3x30x30xf16>
    return %1 : tensor<1x3x30x30xf16>

    // CHECK:       [[VAR0:%.+]] = IE.Reorder(%arg1) {dstOrder = #NHWC} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30xf16, {order = #NHWC}>
    // CHECK:       [[VAR1:%.+]] = IE.Multiply(%arg0, [[VAR0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x3x30x30xf16, {order = #NHWC}>, tensor<1x1x30x30xf16, {order = #NHWC}> -> tensor<1x3x30x30xf16, {order = #NHWC}>
    // CHECK:       [[VAR2:%.+]] = IE.Reorder([[VAR1]]) {dstOrder = #NCHW} : tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x3x30x30xf16>
    // CHECK:       return [[VAR2]] : tensor<1x3x30x30xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @NotReorderMultiUsersWithAffineReshape
func.func @NotReorderMultiUsersWithAffineReshape(%arg0: tensor<1x3x96x8xf16, {order = #NCWH}>)
        -> (tensor<1x3x8x12x8xf16>, tensor<1x3x96x8xf16>) {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x3x96x8xf16, {order = #NCWH}> -> tensor<1x3x96x8xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 3, 8, 12, 8]} : tensor<1x3x96x8xf16> -> tensor<1x3x8x12x8xf16>
    %2 = IE.Sigmoid(%0) : tensor<1x3x96x8xf16> -> tensor<1x3x96x8xf16>
    return %1, %2 : tensor<1x3x8x12x8xf16>, tensor<1x3x96x8xf16>

    // CHECK:       [[VAR0:%.+]] = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x3x96x8xf16, {order = #NCWH}> -> tensor<1x3x96x8xf16>
    // CHECK:       [[VAR1:%.+]] = IE.AffineReshape([[VAR0]])
    // CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 3, 8, 12, 8]} : tensor<1x3x96x8xf16> -> tensor<1x3x8x12x8xf16>
    // CHECK:       [[VAR2:%.+]] = IE.Sigmoid([[VAR0]]) : tensor<1x3x96x8xf16> -> tensor<1x3x96x8xf16>
    // CHECK:       return [[VAR1]], [[VAR2]] : tensor<1x3x8x12x8xf16>, tensor<1x3x96x8xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CleanUpInputOutputReordersWithTile
func.func @CleanUpInputOutputReordersWithTile(%arg0: tensor<1x128x1x3xf16, {order = #NHWC}>)
        -> tensor<1x128x1x249xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x128x1x3xf16, {order = #NHWC}> -> tensor<1x128x1x3xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 128, 3, 1]} : tensor<1x128x1x3xf16> -> tensor<1x128x3x1xf16>
    %2 = IE.Tile(%1) {repeats_values = [1, 1, 1, 83]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x83xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 128, 1, 249]} : tensor<1x128x3x83xf16> -> tensor<1x128x1x249xf16>
    %4 = IE.Reorder(%3) {dstOrder = #NHWC} : tensor<1x128x1x249xf16> -> tensor<1x128x1x249xf16, {order = #NHWC}>
    return %4 : tensor<1x128x1x249xf16, {order = #NHWC}>

    // CHECK:       [[SHAPECAST_1:%.+]] = IE.ShapeCast {shape = [1, 128, 3, 1]}
    // CHECK-SAME:      inputs({{[^:]+}} : tensor<1x128x1x3xf16, {order = #NHWC}>) -> tensor<1x128x3x1xf16, {order = #NHWC}>
    // CHECK:       [[TILE:%.+]] = IE.Tile([[SHAPECAST_1]]) {repeats_values = [1, 1, 1, 83]} : tensor<1x128x3x1xf16, {order = #NHWC}> -> tensor<1x128x3x83xf16, {order = #NHWC}>
    // CHECK:       [[SHAPECAST_2:%.+]] = IE.ShapeCast {shape = [1, 128, 1, 249]}
    // CHECK-SAME:      inputs([[TILE]] : tensor<1x128x3x83xf16, {order = #NHWC}>) -> tensor<1x128x1x249xf16, {order = #NHWC}>
    // CHECK:       return [[SHAPECAST_2]] : tensor<1x128x1x249xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotCleanUpInputOutputReordersWithTilePartialPattern
func.func @NotCleanUpInputOutputReordersWithTilePartialPattern(%arg0: tensor<1x128x1x3xf16, {order = #NHWC}>)
        -> tensor<1x128x1x249xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x128x1x3xf16, {order = #NHWC}> -> tensor<1x128x1x3xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 128, 3, 1]} : tensor<1x128x1x3xf16> -> tensor<1x128x3x1xf16>
    %2 = IE.Tile(%1) {repeats_values = [1, 1, 1, 83]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x83xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 128, 1, 249]} : tensor<1x128x3x83xf16> -> tensor<1x128x1x249xf16>
    return %3 : tensor<1x128x1x249xf16>

    // CHECK:       [[REORDER:%.+]] = IE.Reorder({{[^:]+}}) {dstOrder = #NCHW} : tensor<1x128x1x3xf16, {order = #NHWC}> -> tensor<1x128x1x3xf16>
    // CHECK:       [[AFFINERESHAPE_1:%.+]] = IE.AffineReshape([[REORDER]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 128, 3, 1]} : tensor<1x128x1x3xf16> -> tensor<1x128x3x1xf16>
    // CHECK:       [[TILE:%.+]] = IE.Tile([[AFFINERESHAPE_1]])  {repeats_values = [1, 1, 1, 83]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x83xf16>
    // CHECK:       [[AFFINERESHAPE_2:%.+]] = IE.AffineReshape([[TILE]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 128, 1, 249]} : tensor<1x128x3x83xf16> -> tensor<1x128x1x249xf16>
    // CHECK:       return [[AFFINERESHAPE_2]] : tensor<1x128x1x249xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotCleanUpInputOutputReordersWithMultiInputReorderOuputs
func.func @NotCleanUpInputOutputReordersWithMultiInputReorderOuputs(%arg0: tensor<1x128x1x3xf16, {order = #NHWC}>)
        -> (tensor<1x128x1x249xf16, {order = #NHWC}>, tensor<1x128x3x1xf16>){
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x128x1x3xf16, {order = #NHWC}> -> tensor<1x128x1x3xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 128, 3, 1]} : tensor<1x128x1x3xf16> -> tensor<1x128x3x1xf16>
    %2 = IE.Tile(%1) {repeats_values = [1, 1, 1, 83]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x83xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 128, 1, 249]} : tensor<1x128x3x83xf16> -> tensor<1x128x1x249xf16>
    %4 = IE.Reorder(%3) {dstOrder = #NHWC} : tensor<1x128x1x249xf16> -> tensor<1x128x1x249xf16, {order = #NHWC}>
    %5 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 128, 3, 1]} : tensor<1x128x1x3xf16> -> tensor<1x128x3x1xf16>

    return %4, %5 : tensor<1x128x1x249xf16, {order = #NHWC}>, tensor<1x128x3x1xf16>

    // CHECK:       [[REORDER:%.+]] = IE.Reorder({{[^:]+}}) {dstOrder = #NCHW} : tensor<1x128x1x3xf16, {order = #NHWC}> -> tensor<1x128x1x3xf16>
    // CHECK:       [[AFFINERESHAPE_1:%.+]] = IE.AffineReshape([[REORDER]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 128, 3, 1]} : tensor<1x128x1x3xf16> -> tensor<1x128x3x1xf16>
    // CHECK:       [[TILE:%.+]] = IE.Tile([[AFFINERESHAPE_1]])  {repeats_values = [1, 1, 1, 83]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x83xf16>
    // CHECK:       [[AFFINERESHAPE_2:%.+]] = IE.AffineReshape([[TILE]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 128, 1, 249]} : tensor<1x128x3x83xf16> -> tensor<1x128x1x249xf16>
    // CHECK:       [[REORDER_2:%.+]] = IE.Reorder([[AFFINERESHAPE_2]]) {dstOrder = #NHWC} : tensor<1x128x1x249xf16> -> tensor<1x128x1x249xf16, {order = #NHWC}>
    // CHECK:       [[AFFINERESHAPE_3:%.+]] = IE.AffineReshape([[REORDER]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 128, 3, 1]} : tensor<1x128x1x3xf16> -> tensor<1x128x3x1xf16>

    // CHECK:       return [[REORDER_2]], [[AFFINERESHAPE_3]] : tensor<1x128x1x249xf16, {order = #NHWC}>, tensor<1x128x3x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotCleanUpInputOutputReordersWithMultiInputAffineReshapeOuputs
func.func @NotCleanUpInputOutputReordersWithMultiInputAffineReshapeOuputs(%arg0: tensor<1x128x1x3xf16, {order = #NHWC}>)
        -> (tensor<1x128x1x249xf16, {order = #NHWC}>, tensor<1x128x3x1xf16>){
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x128x1x3xf16, {order = #NHWC}> -> tensor<1x128x1x3xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 128, 3, 1]} : tensor<1x128x1x3xf16> -> tensor<1x128x3x1xf16>
    %2 = IE.Tile(%1) {repeats_values = [1, 1, 1, 83]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x83xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 128, 1, 249]} : tensor<1x128x3x83xf16> -> tensor<1x128x1x249xf16>
    %4 = IE.Reorder(%3) {dstOrder = #NHWC} : tensor<1x128x1x249xf16> -> tensor<1x128x1x249xf16, {order = #NHWC}>
    %5 = IE.Sigmoid(%1) : tensor<1x128x3x1xf16> -> tensor<1x128x3x1xf16>

    return %4, %5 : tensor<1x128x1x249xf16, {order = #NHWC}>, tensor<1x128x3x1xf16>

    // CHECK:       [[REORDER:%.+]] = IE.Reorder({{[^:]+}}) {dstOrder = #NCHW} : tensor<1x128x1x3xf16, {order = #NHWC}> -> tensor<1x128x1x3xf16>
    // CHECK:       [[AFFINERESHAPE_1:%.+]] = IE.AffineReshape([[REORDER]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 128, 3, 1]} : tensor<1x128x1x3xf16> -> tensor<1x128x3x1xf16>
    // CHECK:       [[TILE:%.+]] = IE.Tile([[AFFINERESHAPE_1]])  {repeats_values = [1, 1, 1, 83]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x83xf16>
    // CHECK:       [[AFFINERESHAPE_2:%.+]] = IE.AffineReshape([[TILE]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 128, 1, 249]} : tensor<1x128x3x83xf16> -> tensor<1x128x1x249xf16>
    // CHECK:       [[REORDER_2:%.+]] = IE.Reorder([[AFFINERESHAPE_2]]) {dstOrder = #NHWC} : tensor<1x128x1x249xf16> -> tensor<1x128x1x249xf16, {order = #NHWC}>
    // CHECK:       [[SIGMOID:%.+]] = IE.Sigmoid([[AFFINERESHAPE_1]]) : tensor<1x128x3x1xf16> -> tensor<1x128x3x1xf16>

    // CHECK:       return [[REORDER_2]], [[SIGMOID]] : tensor<1x128x1x249xf16, {order = #NHWC}>, tensor<1x128x3x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotCleanUpInputOutputReordersWithMultiOutputAffineReshapeOuputs
func.func @NotCleanUpInputOutputReordersWithMultiOutputAffineReshapeOuputs(%arg0: tensor<1x128x1x3xf16, {order = #NHWC}>)
        -> (tensor<1x128x1x249xf16, {order = #NHWC}>, tensor<1x128x1x249xf16>){
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x128x1x3xf16, {order = #NHWC}> -> tensor<1x128x1x3xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 128, 3, 1]} : tensor<1x128x1x3xf16> -> tensor<1x128x3x1xf16>
    %2 = IE.Tile(%1) {repeats_values = [1, 1, 1, 83]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x83xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 128, 1, 249]} : tensor<1x128x3x83xf16> -> tensor<1x128x1x249xf16>
    %4 = IE.Reorder(%3) {dstOrder = #NHWC} : tensor<1x128x1x249xf16> -> tensor<1x128x1x249xf16, {order = #NHWC}>
    %5 = IE.Sigmoid(%3) : tensor<1x128x1x249xf16> -> tensor<1x128x1x249xf16>

    return %4, %5 : tensor<1x128x1x249xf16, {order = #NHWC}>, tensor<1x128x1x249xf16>

    // CHECK:       [[REORDER:%.+]] = IE.Reorder({{[^:]+}}) {dstOrder = #NCHW} : tensor<1x128x1x3xf16, {order = #NHWC}> -> tensor<1x128x1x3xf16>
    // CHECK:       [[AFFINERESHAPE_1:%.+]] = IE.AffineReshape([[REORDER]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 128, 3, 1]} : tensor<1x128x1x3xf16> -> tensor<1x128x3x1xf16>
    // CHECK:       [[TILE:%.+]] = IE.Tile([[AFFINERESHAPE_1]])  {repeats_values = [1, 1, 1, 83]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x83xf16>
    // CHECK:       [[AFFINERESHAPE_2:%.+]] = IE.AffineReshape([[TILE]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 128, 1, 249]} : tensor<1x128x3x83xf16> -> tensor<1x128x1x249xf16>
    // CHECK:       [[REORDER_2:%.+]] = IE.Reorder([[AFFINERESHAPE_2]]) {dstOrder = #NHWC} : tensor<1x128x1x249xf16> -> tensor<1x128x1x249xf16, {order = #NHWC}>
    // CHECK:       [[SIGMOID:%.+]] = IE.Sigmoid([[AFFINERESHAPE_2]]) : tensor<1x128x1x249xf16> -> tensor<1x128x1x249xf16>

    // CHECK:       return [[REORDER_2]], [[SIGMOID]] : tensor<1x128x1x249xf16, {order = #NHWC}>, tensor<1x128x1x249xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0058469066432878082>
!qElemType1 = !quant.uniform<u8:f16, 0.0029234533216439041>

// CHECK-LABEL: @ReorderAffineReshapeQuantizeCastReorder
func.func @ReorderAffineReshapeQuantizeCastReorder(%arg0: tensor<1x128x1x249x!qElemType, {order = #NHWC}>)
        -> tensor<1x128x83x3x!qElemType1, {order = #NHWC}>{
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x128x1x249x!qElemType, {order = #NHWC}> -> tensor<1x128x1x249x!qElemType>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 128, 83, 3]} : tensor<1x128x1x249x!qElemType> -> tensor<1x128x83x3x!qElemType>
    %2 = IE.QuantizeCast(%1) {dstElemType = !qElemType1} : tensor<1x128x83x3x!qElemType> -> tensor<1x128x83x3x!qElemType1>
    %3 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<1x128x83x3x!qElemType1> -> tensor<1x128x83x3x!qElemType1, {order = #NHWC}>
    return %3 : tensor<1x128x83x3x!qElemType1, {order = #NHWC}>

    // CHECK:       [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 128, 83, 3]}
    // CHECK-SAME:      inputs({{[^:]+}} : tensor<1x128x1x249x!qElemType, {order = #NHWC}>) -> tensor<1x128x83x3x!qElemType, {order = #NHWC}>
    // CHECK:      [[QCAST:%.+]] = IE.QuantizeCast([[SHAPECAST]]) {dstElemType = !qElemType1}
    // CHECK-SAME:      tensor<1x128x83x3x!qElemType, {order = #NHWC}> -> tensor<1x128x83x3x!qElemType1, {order = #NHWC}>
    // CHECK:       return [[QCAST]] : tensor<1x128x83x3x!qElemType1, {order = #NHWC}>
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderAddReorder
func.func @ReorderAddReorder(%arg0: tensor<1x1x77x768xf16>) -> tensor<1x1x77x768xf16> {
    %cst = const.Declare tensor<1x1x77x768xf16, {order = #NHWC}> = dense<1.0> : tensor<1x1x77x768xf16, {order = #NHWC}>
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x1x77x768xf16> -> tensor<1x1x77x768xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x77x768xf16, {order = #NHWC}>, tensor<1x1x77x768xf16, {order = #NHWC}> -> tensor<1x1x77x768xf16, {order = #NHWC}>
    %2 = IE.Reorder(%1) {dstOrder = #NCHW} : tensor<1x1x77x768xf16, {order = #NHWC}> -> tensor<1x1x77x768xf16>
    return %2 : tensor<1x1x77x768xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x77x768xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x77x768xf16, {order = #NHWC}>, [#const.Reorder<#NCHW>, #const.LayoutCast<#NHWC>]
    // CHECK: [[LAYOUTCAST_1:%.+]] = IE.LayoutCast([[ARG0:%.+]]) {dst_order = #NHWC} : tensor<1x1x77x768xf16> -> tensor<1x1x77x768xf16, {order = #NHWC}>
    // CHECK: [[ADD:%.+]] = IE.Add([[LAYOUTCAST_1]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x77x768xf16, {order = #NHWC}>, tensor<1x1x77x768xf16, {order = #NHWC}> -> tensor<1x1x77x768xf16, {order = #NHWC}>
    // CHECK: [[LAYOUTCAST_1:%.+]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW} : tensor<1x1x77x768xf16, {order = #NHWC}> -> tensor<1x1x77x768xf16>
    // CHECK: return [[LAYOUTCAST_1]] : tensor<1x1x77x768xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0053023436490227194:122>
// CHECK-LABEL: @ReorderAddReorderMVN
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x512x768x1x!qElemType>
func.func @ReorderAddReorderMVN(%arg0: tensor<1x512x768x1x!qElemType>) -> tensor<1x512x768x1xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x512x768x1x!qElemType> -> tensor<1x512x768x1x!qElemType, {order = #NHWC}>
    %1 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x512x768x1x!qElemType> -> tensor<1x512x768x1x!qElemType, {order = #NHWC}>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x768x1x!qElemType, {order = #NHWC}>, tensor<1x512x768x1x!qElemType, {order = #NHWC}> -> tensor<1x512x768x1xf16, {order = #NHWC}>
    %3 = IE.Reorder(%2) {dstOrder = #NCHW} : tensor<1x512x768x1xf16, {order = #NHWC}> -> tensor<1x512x768x1xf16>
    %4 = IE.MVN(%3) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x512x768x1xf16> -> tensor<1x512x768x1xf16>
    return %4 : tensor<1x512x768x1xf16>

    // CHECK: [[LAYOUTCAST_0:%.+]] = IE.LayoutCast([[INPUT]]) {dst_order = #NHWC} : tensor<1x512x768x1x!qElemType> -> tensor<1x512x768x1x!qElemType, {order = #NHWC}>
    // CHECK: [[LAYOUTCAST_1:%.+]] = IE.LayoutCast([[INPUT]]) {dst_order = #NHWC} : tensor<1x512x768x1x!qElemType> -> tensor<1x512x768x1x!qElemType, {order = #NHWC}>
    // CHECK: [[ADD:%.+]] = IE.Add([[LAYOUTCAST_0]], [[LAYOUTCAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x768x1x!qElemType, {order = #NHWC}>, tensor<1x512x768x1x!qElemType, {order = #NHWC}> -> tensor<1x512x768x1xf16, {order = #NHWC}>
    // CHECK: [[LAYOUTCAST_2:%.+]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW} : tensor<1x512x768x1xf16, {order = #NHWC}> -> tensor<1x512x768x1xf16>
    // CHECK: [[MVN:%.+]] = IE.MVN([[LAYOUTCAST_2]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x512x768x1xf16> -> tensor<1x512x768x1xf16>
    // CHECK: return [[MVN]] : tensor<1x512x768x1xf16>
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-LABEL: @DoNotPropagateReorderNCWHWithGatherElementsOp
module @DoNotPropagateReorderNCWHWithGatherElementsOp {

// CHECK: func.func @main([[DATA:%.+]]: tensor<1x8400x4xf16, {order = #map}>, [[INDEX:%.+]]: tensor<1x300x4xsi32>) -> tensor<1x300x4xf16>
func.func @main(%data: tensor<1x8400x4xf16, {order = #map}>, %index: tensor<1x300x4xsi32>) -> tensor<1x300x4xf16> {
    %reordered_data = IE.Reorder(%data) {dstOrder = #CHW} : tensor<1x8400x4xf16, {order = #map}> -> tensor<1x8400x4xf16>
    %out = IE.GatherElements(%reordered_data, %index) {axis = 1 : i64} : tensor<1x8400x4xf16>, tensor<1x300x4xsi32> -> tensor<1x300x4xf16>
    return %out : tensor<1x300x4xf16>

    // CHECK:           [[REORDER:%.+]] = IE.Reorder([[DATA]]) {dstOrder = #CHW}
    // CHECK-SAME:          tensor<1x8400x4xf16, {order = #map}> -> tensor<1x8400x4xf16>
    // CHECK:           [[GATHER:%.+]] = IE.GatherElements([[REORDER]], [[INDEX]]) {axis = 1 : i64}
    // CHECK-SAME:          tensor<1x8400x4xf16>, tensor<1x300x4xsi32> -> tensor<1x300x4xf16>
    // CHECK:           return [[GATHER]] : tensor<1x300x4xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>

// CHECK-LABEL: @ReorderWithEltwiseNCHW
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x512x6x235xf16, {order = #map}>
func.func @ReorderWithEltwiseNCHW(%arg0: tensor<1x512x6x235xf16, {order = #map}>) -> tensor<1x512x6x235xf16> {
    %in_reorder = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x512x6x235xf16, {order = #map}> -> tensor<1x512x6x235xf16>
    %gelu = IE.Gelu(%in_reorder) : tensor<1x512x6x235xf16> -> tensor<1x512x6x235xf16>

    return %gelu : tensor<1x512x6x235xf16>

    // CHECK: [[PERMUTE_CAST_0:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x6x235xf16, {order = #map}> -> tensor<235x1x512x6xf16>
    // CHECK: [[GELU:%.+]] = IE.Gelu([[PERMUTE_CAST_0]]) : tensor<235x1x512x6xf16> -> tensor<235x1x512x6xf16>
    // CHECK: [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast([[GELU]]) {dst_order = #map, mem_perm = #NCHW} : tensor<235x1x512x6xf16> -> tensor<1x512x6x235xf16, {order = #map}>
    // CHECK: [[REORDER:%.+]] = IE.Reorder([[PERMUTE_CAST_1]]) {dstOrder = #NCHW} : tensor<1x512x6x235xf16, {order = #map}> -> tensor<1x512x6x235xf16>
    // CHECK: return [[REORDER]] : tensor<1x512x6x235xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>

// CHECK-LABEL: @ReorderWithEltwiseNHWC
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x512x6x235xf16, {order = #map}>
func.func @ReorderWithEltwiseNHWC(%arg0: tensor<1x512x6x235xf16, {order = #map}>) -> tensor<1x512x6x235xf16, {order = #NHWC}> {
    %in_reorder = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x512x6x235xf16, {order = #map}> -> tensor<1x512x6x235xf16, {order = #NHWC}>
    %gelu = IE.Gelu(%in_reorder) : tensor<1x512x6x235xf16, {order = #NHWC}> -> tensor<1x512x6x235xf16, {order = #NHWC}>

    return %gelu : tensor<1x512x6x235xf16, {order = #NHWC}>

    // CHECK: [[PERMUTE_CAST_0:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x6x235xf16, {order = #map}> -> tensor<235x1x512x6xf16>
    // CHECK: [[GELU:%.+]] = IE.Gelu([[PERMUTE_CAST_0]]) : tensor<235x1x512x6xf16> -> tensor<235x1x512x6xf16>
    // CHECK: [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast([[GELU]]) {dst_order = #map, mem_perm = #NCHW} : tensor<235x1x512x6xf16> -> tensor<1x512x6x235xf16, {order = #map}>
    // CHECK: [[REORDER:%.+]] = IE.Reorder([[PERMUTE_CAST_1]]) {dstOrder = #NHWC} : tensor<1x512x6x235xf16, {order = #map}> -> tensor<1x512x6x235xf16, {order = #NHWC}>
    // CHECK: return [[REORDER]] : tensor<1x512x6x235xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderNHWCToNCHWWithEltwise
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x512x6x235xf16, {order = #NHWC}>
func.func @ReorderNHWCToNCHWWithEltwise(%arg0: tensor<1x512x6x235xf16, {order = #NHWC}>) -> tensor<1x512x6x235xf16> {
    %in_reorder = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x512x6x235xf16, {order = #NHWC}> -> tensor<1x512x6x235xf16>
    %gelu = IE.Gelu(%in_reorder) : tensor<1x512x6x235xf16> -> tensor<1x512x6x235xf16>

    return %gelu : tensor<1x512x6x235xf16>

    //CHECK: [[GELU:%.+]] = IE.Gelu([[INPUT]]) : tensor<1x512x6x235xf16, {order = #NHWC}> -> tensor<1x512x6x235xf16, {order = #NHWC}>
    //CHECK: [[REORDER:%.+]] = IE.Reorder([[GELU]]) {dstOrder = #NCHW} : tensor<1x512x6x235xf16, {order = #NHWC}> -> tensor<1x512x6x235xf16>
    //CHECK: return [[REORDER]] : tensor<1x512x6x235xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>
!qElemType = !quant.uniform<i4:f16:1, {1.000000e+00,2.000000e+00}>
!qElemType1 = !quant.uniform<i4:f16:0, {1.000000e+00,2.000000e+00}>

// CHECK-LABEL: @NotMoveReorderThroughAffineReshapeForPerAxisQuant
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x2x1x128x!qElemType, {order = #NCWH}>
func.func @NotMoveReorderThroughAffineReshapeForPerAxisQuant(%arg0: tensor<1x2x1x128x!qElemType, {order = #NCWH}>) -> tensor<2x128x1x1x!qElemType1, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x2x1x128x!qElemType, {order = #NCWH}> -> tensor<1x2x1x128x!quant.uniform<i4:f16:1, {1.0, 2.0}>, {order = #NCHW}>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [2, 128, 1, 1]}  : tensor<1x2x1x128x!quant.uniform<i4:f16:1, {1.0, 2.0}>, {order = #NCHW}> -> tensor<2x128x1x1x!qElemType1, {order = #NCHW}>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<2x128x1x1x!qElemType1, {order = #NCHW}> -> tensor<2x128x1x1x!qElemType1, {order = #NHWC}>
    return %2 : tensor<2x128x1x1x!qElemType1, {order = #NHWC}>

    // CHECK:       [[REORDER_0:%.+]] = IE.Reorder([[INPUT]]) {dstOrder = #NCHW} : tensor<1x2x1x128x!qElemType, {order = #NCWH}> -> tensor<1x2x1x128x!qElemType, {order = #NCHW}>
    // CHECK:       [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[REORDER_0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [2, 128, 1, 1]} : tensor<1x2x1x128x!qElemType, {order = #NCHW}> -> tensor<2x128x1x1x!qElemType1, {order = #NCHW}>
    // CHECK:       [[REORDER_1:%.+]] = IE.Reorder([[AFFINERESHAPE]]) {dstOrder = #NHWC} : tensor<2x128x1x1x!qElemType1, {order = #NCHW}> -> tensor<2x128x1x1x!qElemType1, {order = #NHWC}>
    // CHECK:       return [[REORDER_1]] : tensor<2x128x1x1x!qElemType1, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>


// CHECK-LABEL: @ReorderWithConcatNoSwap
module @ReorderWithConcatNoSwap {

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x768x64x64xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x768x64x64xf16, {order = #NHWC}>) ->  tensor<1x64x70x768xf16> {
    %cst = const.Declare tensor<768x1x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x1x1x768xf16>, [#const.Transpose<#NWCH>, #const.Reshape<[768, 1, 1, 1]>, #const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<1x768x1x1xf16> = dense<1.250000e-01> : tensor<1x1x1x768xf16>, [#const.Transpose<#NWCH>]
    %cst_1 = const.Declare tensor<1x64x6x768xf16> = dense<0.000000e+00> : tensor<1x64x6x768xf16>
    %0 = IE.GroupConvolution(%arg0, %cst, %cst_0) {dilations = [1, 1], groups = 768 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x768x64x64xf16, {order = #NHWC}>, tensor<768x1x1x1xf16, {order = #NHWC}>, tensor<1x768x1x1xf16> -> tensor<1x768x64x64xf16, {order = #NHWC}>
    %1 = IE.Reorder(%0) {dstOrder = #NCHW} : tensor<1x768x64x64xf16, {order = #NHWC}> -> tensor<1x768x64x64xf16>
    %2 = IE.PermuteCast(%1) {dst_order = #NWCH, mem_perm = #NCHW} : tensor<1x768x64x64xf16> -> tensor<1x64x64x768xf16, {order = #NWCH}>
    %3 = IE.Reorder(%2) {dstOrder = #NCHW} : tensor<1x64x64x768xf16, {order = #NWCH}> -> tensor<1x64x64x768xf16>
    %4 = IE.Concat(%3, %cst_1) {static_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]]} : tensor<1x64x64x768xf16>, tensor<1x64x6x768xf16> -> tensor<1x64x70x768xf16>
    return %4 : tensor<1x64x70x768xf16>

    // CHECK-DAG:       [[CST:%.*]] = const.Declare
    // CHECK-SAME:      tensor<768x1x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x1x1x768xf16>, [#const.Transpose<#NWCH>, #const.Reshape<[768, 1, 1, 1]>, #const.Reorder<#NHWC>]

    // CHECK-DAG:       [[CST_0:%.*]] = const.Declare
    // CHECK-SAME:      tensor<1x768x1x1xf16> = dense<1.250000e-01> : tensor<1x1x1x768xf16>, [#const.Transpose<#NWCH>]

    // CHECK-DAG:       [[CST_1:%.*]] = const.Declare
    // CHECK-SAME:      tensor<1x64x6x768xf16> = dense<0.000000e+00> : tensor<1x64x6x768xf16>

    // CHECK:       [[GROUP_CONV:%.*]] = IE.GroupConvolution([[ARG0]], [[CST]], [[CST_0]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 768 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      tensor<1x768x64x64xf16, {order = #NHWC}>, tensor<768x1x1x1xf16, {order = #NHWC}>, tensor<1x768x1x1xf16> -> tensor<1x768x64x64xf16, {order = #NHWC}>
    // CHECK:       [[REORDER_0:%.*]] = IE.Reorder([[GROUP_CONV]])
    // CHECK-SAME:      {dstOrder = #NCHW} : tensor<1x768x64x64xf16, {order = #NHWC}> -> tensor<1x768x64x64xf16>
    // CHECK:       [[PERMUTE_CAST:%.*]] = IE.PermuteCast([[REORDER_0]])
    // CHECK-SAME:      {dst_order = #NWCH, mem_perm = #NCHW} : tensor<1x768x64x64xf16> -> tensor<1x64x64x768xf16, {order = #NWCH}>
    // CHECK:       [[REORDER_1:%.*]] = IE.Reorder([[PERMUTE_CAST]])
    // CHECK-SAME:      {dstOrder = #NCHW} : tensor<1x64x64x768xf16, {order = #NWCH}> -> tensor<1x64x64x768xf16>

    // CHECK:       [[CONCAT:%.*]] = IE.Concat([[REORDER_1]], [[CST_1]])
    // CHECK-SAME:      tensor<1x64x64x768xf16>, tensor<1x64x6x768xf16> -> tensor<1x64x70x768xf16>

    // CHECK:       return [[CONCAT]] : tensor<1x64x70x768xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwitchReorderWithTileForBetterTileDMAPerfCase2
func.func @SwitchReorderWithTileForBetterTileDMAPerfCase2(%arg0: tensor<1x64x1x1xf16, {order = #NHWC}>) -> tensor<1x64x11x11xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x1x1xf16>
    %1 = IE.Tile(%0) {repeats_values = [1, 1, 11, 11]} : tensor<1x64x1x1xf16> -> tensor<1x64x11x11xf16>
    return %1 : tensor<1x64x11x11xf16>

    // CHECK:       [[TILE:%.+]] = IE.Tile({{[^:]+}}) {repeats_values = [1, 1, 11, 11]} : tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x11x11xf16, {order = #NHWC}>
    // CHECK:       [[REORDER:%.+]] = IE.Reorder([[TILE]]) {dstOrder = #NCHW} : tensor<1x64x11x11xf16, {order = #NHWC}> -> tensor<1x64x11x11xf16>
    // CHECK:       return [[REORDER]] : tensor<1x64x11x11xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @FusingNonTrivialReorderAroundTile
func.func @FusingNonTrivialReorderAroundTile(%arg0: tensor<1x1x11x11xf16, {order = #NCWH}> ) -> tensor<1x64x11x11xf16, {order = #NCWH}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x1x11x11xf16, {order = #NCWH}> -> tensor<1x1x11x11xf16>
    %1 = IE.Tile(%0) {repeats_values = [1, 64, 1, 1]} : tensor<1x1x11x11xf16> -> tensor<1x64x11x11xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NCWH} : tensor<1x64x11x11xf16> -> tensor<1x64x11x11xf16, {order = #NCWH}>
    return %2 : tensor<1x64x11x11xf16, {order = #NCWH}>

    // CHECK:       [[TILE:%.+]] = IE.Tile({{[^:]+}}) {repeats_values = [1, 64, 1, 1]} : tensor<1x1x11x11xf16, {order = #NCWH}> -> tensor<1x64x11x11xf16, {order = #NCWH}>
    // CHECK:       return [[TILE]] : tensor<1x64x11x11xf16, {order = #NCWH}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithSubtract
module @ReorderWithSubtract {

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x32x28x1xf16, {order = #NHWC}>) -> tensor<1x32x28x1xf16> {
func.func @main(%arg0: tensor<1x32x28x1xf16, {order = #NHWC}>) -> tensor<1x32x28x1xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x32x28x1xf16, {order = #NHWC}> -> tensor<1x32x28x1xf16>
    %1 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x32x28x1xf16, {order = #NHWC}> -> tensor<1x32x28x1xf16>
    %2 = IE.Subtract(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x28x1xf16>, tensor<1x32x28x1xf16> -> tensor<1x32x28x1xf16>

    return %2 : tensor<1x32x28x1xf16>

    // CHECK:       [[SUB:%.+]] = IE.Subtract([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x28x1xf16, {order = #NHWC}>, tensor<1x32x28x1xf16, {order = #NHWC}> -> tensor<1x32x28x1xf16, {order = #NHWC}>
    // CHECK:       [[REORDER:%.+]] = IE.Reorder([[SUB]]) {dstOrder = #NCHW} : tensor<1x32x28x1xf16, {order = #NHWC}> -> tensor<1x32x28x1xf16>
    // CHECK:       return [[REORDER]] : tensor<1x32x28x1xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithSwDivideHasChange
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x3x30x30xf16, {order = #NHWC}>
func.func @ReorderWithSwDivideHasChange(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x3x30x30xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x30x30xf16> = dense<0.1> : tensor<1x1x30x30xf16>
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x3x30x30xf16>
    %1 = IE.Divide(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x3x30x30xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16, {order = #NHWC}>
    return %2 : tensor<1x3x30x30xf16, {order = #NHWC}>

    // CHECK-DAG:       [[VAR0:%.+]] = const.Declare tensor<1x1x30x30xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x30x30xf16>, [#const.Reorder<#NHWC>]
    // CHECK:           [[VAR1:%.+]] = IE.Divide([[INPUT]], [[VAR0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x30x30xf16, {order = #NHWC}>, tensor<1x1x30x30xf16, {order = #NHWC}> -> tensor<1x3x30x30xf16, {order = #NHWC}>

    // CHECK:           return [[VAR1]] : tensor<1x3x30x30xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AlignInputsLayoutMVN6Op
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x512x10xf16, {order = #NHWC}>)
func.func @AlignInputsLayoutMVN6Op(%inp: tensor<1x1x512x10xf16, {order = #NHWC}>) -> tensor<1x1x512x10xf16,{order = #NHWC}> {

    %scl = const.Declare tensor<1x1x1x10xf16> = dense<[[[[1.00098,1.56348,1.19336,1.80859,1.58496,1.47949,1.35059,1.89551,1.82324,1.74707]]]]> : tensor<1x1x1x10xf16>
    %0 = IE.Reorder(%inp) {dstOrder = #NCHW} : tensor<1x1x512x10xf16, {order = #NHWC}> -> tensor<1x1x512x10xf16>
    %1 = IE.MVN6(%0, %scl) {axes_value = [2], eps = 1.000000e-05 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 1, 0, 0>}
         : tensor<1x1x512x10xf16>, tensor<1x1x1x10xf16> -> tensor<1x1x512x10xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x1x512x10xf16> -> tensor<1x1x512x10xf16, {order = #NHWC}>
    return %2 : tensor<1x1x512x10xf16, {order = #NHWC}>

    // CHECK:      [[SCALE:%.+]]  = const.Declare tensor<1x1x1x10xf16, {order = #NHWC}>
    // CHECK:      [[OUTPUT:%.+]] = IE.MVN6([[INPUT]], [[SCALE]]) {axes_value = [2], eps = 1.000000e-05 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 1, 0, 0>} :
    // CHECK-SAME:                  tensor<1x1x512x10xf16, {order = #NHWC}>, tensor<1x1x1x10xf16, {order = #NHWC}> -> tensor<1x1x512x10xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x1x512x10xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithSwGreaterEqualHasChange
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x1x1xf32, {order = #NHWC}>
func.func @ReorderWithSwGreaterEqualHasChange(%arg0: tensor<1x1x1x1xf32, {order = #NHWC}>) -> tensor<1x1x1x1xf32, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>]
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x1x1x1xf32, {order = #NHWC}> -> tensor<1x1x1x1xf32>
    %1 = IE.GreaterEqual(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32, {order = #NHWC}>
    return %2 : tensor<1x1x1x1xf32, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]]  = const.Declare tensor<1x1x1x1xf32, {order = #NHWC}> = dense<1.000000e+00> : tensor<1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK:           [[GREATER_EQUAL:%.+]] = IE.GreaterEqual([[INPUT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32, {order = #NHWC}>, tensor<1x1x1x1xf32, {order = #NHWC}> -> tensor<1x1x1x1xf32, {order = #NHWC}>
    // CHECK:           return [[GREATER_EQUAL]] : tensor<1x1x1x1xf32, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithSwGreaterHasChange
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x1x1xf32, {order = #NHWC}>
func.func @ReorderWithSwGreaterHasChange(%arg0: tensor<1x1x1x1xf32, {order = #NHWC}>) -> tensor<1x1x1x1xi8, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>]
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x1x1x1xf32, {order = #NHWC}> -> tensor<1x1x1x1xf32>
    %1 = IE.Greater(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x1xi8>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x1x1x1xi8> -> tensor<1x1x1x1xi8, {order = #NHWC}>
    return %2 : tensor<1x1x1x1xi8, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]]  = const.Declare tensor<1x1x1x1xf32, {order = #NHWC}> = dense<1.000000e+00> : tensor<1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK:           [[GREATER:%.+]] = IE.Greater([[INPUT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32, {order = #NHWC}>, tensor<1x1x1x1xf32, {order = #NHWC}> -> tensor<1x1x1x1xi8, {order = #NHWC}>
    // CHECK:           return [[GREATER]] : tensor<1x1x1x1xi8, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithSwLessHasChange
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x1x1xf32, {order = #NHWC}>
func.func @ReorderWithSwLessHasChange(%arg0: tensor<1x1x1x1xf32, {order = #NHWC}>) -> tensor<1x1x1x1xi8, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>]
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x1x1x1xf32, {order = #NHWC}> -> tensor<1x1x1x1xf32>
    %1 = IE.Less(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x1xi8>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x1x1x1xi8> -> tensor<1x1x1x1xi8, {order = #NHWC}>
    return %2 : tensor<1x1x1x1xi8, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]]  = const.Declare tensor<1x1x1x1xf32, {order = #NHWC}> = dense<1.000000e+00> : tensor<1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK:           [[LESS:%.+]] = IE.Less([[INPUT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32, {order = #NHWC}>, tensor<1x1x1x1xf32, {order = #NHWC}> -> tensor<1x1x1x1xi8, {order = #NHWC}>
    // CHECK:           return [[LESS]] : tensor<1x1x1x1xi8, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithSwLessEqualHasChange
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x1x1xf32, {order = #NHWC}>
func.func @ReorderWithSwLessEqualHasChange(%arg0: tensor<1x1x1x1xf32, {order = #NHWC}>) -> tensor<1x1x1x1xi8, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>]
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x1x1x1xf32, {order = #NHWC}> -> tensor<1x1x1x1xf32>
    %1 = IE.LessEqual(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x1xi8>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x1x1x1xi8> -> tensor<1x1x1x1xi8, {order = #NHWC}>
    return %2 : tensor<1x1x1x1xi8, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]]  = const.Declare tensor<1x1x1x1xf32, {order = #NHWC}> = dense<1.000000e+00> : tensor<1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK:           [[LESS_EQUAL:%.+]] = IE.LessEqual([[INPUT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32, {order = #NHWC}>, tensor<1x1x1x1xf32, {order = #NHWC}> -> tensor<1x1x1x1xi8, {order = #NHWC}>
    // CHECK:           return [[LESS_EQUAL]] : tensor<1x1x1x1xi8, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithSwEqualHasChange
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x1x1xf32, {order = #NHWC}>
func.func @ReorderWithSwEqualHasChange(%arg0: tensor<1x1x1x1xf32, {order = #NHWC}>) -> tensor<1x1x1x1xi8, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>]
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x1x1x1xf32, {order = #NHWC}> -> tensor<1x1x1x1xf32>
    %1 = IE.Equal(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x1xi8>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x1x1x1xi8> -> tensor<1x1x1x1xi8, {order = #NHWC}>
    return %2 : tensor<1x1x1x1xi8, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]]  = const.Declare tensor<1x1x1x1xf32, {order = #NHWC}> = dense<1.000000e+00> : tensor<1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK:           [[EQUAL:%.+]] = IE.Equal([[INPUT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32, {order = #NHWC}>, tensor<1x1x1x1xf32, {order = #NHWC}> -> tensor<1x1x1x1xi8, {order = #NHWC}>
    // CHECK:           return [[EQUAL]] : tensor<1x1x1x1xi8, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithSwNotEqualHasChange
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x1x1xf32, {order = #NHWC}>
func.func @ReorderWithSwNotEqualHasChange(%arg0: tensor<1x1x1x1xf32, {order = #NHWC}>) -> tensor<1x1x1x1xi8, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>]
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x1x1x1xf32, {order = #NHWC}> -> tensor<1x1x1x1xf32>
    %1 = IE.NotEqual(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x1xi8>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x1x1x1xi8> -> tensor<1x1x1x1xi8, {order = #NHWC}>
    return %2 : tensor<1x1x1x1xi8, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]]  = const.Declare tensor<1x1x1x1xf32, {order = #NHWC}> = dense<1.000000e+00> : tensor<1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK:           [[NOT_EQUAL:%.+]] = IE.NotEqual([[INPUT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32, {order = #NHWC}>, tensor<1x1x1x1xf32, {order = #NHWC}> -> tensor<1x1x1x1xi8, {order = #NHWC}>
    // CHECK:           return [[NOT_EQUAL]] : tensor<1x1x1x1xi8, {order = #NHWC}>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithGroupConvToLayoutCast
// CHECK-SAME:  [[INPUT0:%.+]]: tensor<1x2x1152x96xf16>
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<1x1x1x1xf16, {order = #NHWC}>
func.func @ReorderWithGroupConvToLayoutCast(%arg0: tensor<1x2x1152x96xf16>, %arg1: tensor<1x1x1x1xf16, {order = #NHWC}>) -> tensor<1x2x1152x96xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x2x1152x96xf16> -> tensor<1x2x1152x96xf16, {order = #NHWC}>
    %1 = IE.Tile(%arg1) {repeats_values = [1, 2, 1, 1]} : tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x2x1x1xf16, {order = #NHWC}>
    %2 = IE.ShapeCast {shape = [2, 1, 1, 1]} inputs(%1 : tensor<1x2x1x1xf16, {order = #NHWC}>) -> tensor<2x1x1x1xf16, {order = #NHWC}>
    %3 = IE.GroupConvolution(%0, %2) {dilations = [1, 1], groups = 2 : i64, pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Relu<>, strides = [1, 1]} : tensor<1x2x1152x96xf16, {order = #NHWC}>, tensor<2x1x1x1xf16, {order = #NHWC}>-> tensor<1x2x1152x96xf16, {order = #NHWC}>
    %4 = IE.Reorder(%3) {dstOrder = #NCHW} : tensor<1x2x1152x96xf16, {order = #NHWC}> -> tensor<1x2x1152x96xf16>

    return %4 : tensor<1x2x1152x96xf16>

    // CHECK:        [[TILE:%.+]] = IE.Tile([[INPUT1]]) {repeats_values = [1, 2, 1, 1]} : tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x2x1x1xf16, {order = #NHWC}>
    // CHECK:        [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [2, 1, 1, 1]} inputs([[TILE]] : tensor<1x2x1x1xf16, {order = #NHWC}>) -> tensor<2x1x1x1xf16, {order = #NHWC}>
    // CHECK:        [[IN_LAYOUTCAST:%.+]] = IE.LayoutCast([[INPUT0]]) {dst_order = #NHWC} : tensor<1x2x1152x96xf16> -> tensor<1x2x1152x96xf16, {order = #NHWC}>
    // CHECK:        [[GROUPCONV:%.+]] = IE.GroupConvolution([[IN_LAYOUTCAST]], [[SHAPECAST]]) {dilations = [1, 1], groups = 2 : i64, pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Relu<>, strides = [1, 1]} : tensor<1x2x1152x96xf16, {order = #NHWC}>, tensor<2x1x1x1xf16, {order = #NHWC}> -> tensor<1x2x1152x96xf16, {order = #NHWC}>
    // CHECK:        [[OUT_LAYOUTCAST:%.+]] = IE.LayoutCast([[GROUPCONV]]) {dst_order = #NCHW} : tensor<1x2x1152x96xf16, {order = #NHWC}> -> tensor<1x2x1152x96xf16>
    // CHECK:        return [[OUT_LAYOUTCAST]] : tensor<1x2x1152x96xf16>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithGroupConvToPermuteCast
// CHECK-SAME:  [[INPUT0:%.+]]: tensor<1x32x1152x96xf16>
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<1x1x1x1xf16>
func.func @ReorderWithGroupConvToPermuteCast(%arg0: tensor<1x32x1152x96xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x32x1152x96xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x32x1152x96xf16> -> tensor<1x32x1152x96xf16, {order = #NHWC}>
    %1 = IE.Tile(%arg1) {repeats_values = [32, 1, 1, 1]} : tensor<1x1x1x1xf16> -> tensor<32x1x1x1xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<32x1x1x1xf16> -> tensor<32x1x1x1xf16, {order = #NHWC}>
    %3 = IE.GroupConvolution(%0, %2) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Relu<>, strides = [1, 1]} : tensor<1x32x1152x96xf16, {order = #NHWC}>, tensor<32x1x1x1xf16, {order = #NHWC}> -> tensor<1x32x1152x96xf16, {order = #NHWC}>
    %4 = IE.Reorder(%3) {dstOrder = #NCHW} : tensor<1x32x1152x96xf16, {order = #NHWC}> -> tensor<1x32x1152x96xf16>

    return %4 : tensor<1x32x1152x96xf16>

    // CHECK:        [[IN_PERMUTECAST:%.+]] = IE.PermuteCast([[INPUT0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x32x1152x96xf16> -> tensor<1x96x32x1152xf16, {order = #NHWC}>
    // CHECK:        [[TILE:%.+]] = IE.Tile([[INPUT1]]) {repeats_values = [96, 1, 1, 1]} : tensor<1x1x1x1xf16> -> tensor<96x1x1x1xf16>
    // CHECK:        [[REORDER:%.+]] = IE.Reorder([[TILE]]) {dstOrder = #NHWC} : tensor<96x1x1x1xf16> -> tensor<96x1x1x1xf16, {order = #NHWC}>
    // CHECK:        [[GROUPCONV:%.+]] = IE.GroupConvolution([[IN_PERMUTECAST]], [[REORDER]]) {dilations = [1, 1], groups = 96 : i64, pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Relu<>, strides = [1, 1]} : tensor<1x96x32x1152xf16, {order = #NHWC}>, tensor<96x1x1x1xf16, {order = #NHWC}> -> tensor<1x96x32x1152xf16, {order = #NHWC}>
    // CHECK:        [[OUT_PERMUTECAST:%.+]] = IE.PermuteCast([[GROUPCONV]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x96x32x1152xf16, {order = #NHWC}> -> tensor<1x32x1152x96xf16>
    // CHECK:        return [[OUT_PERMUTECAST]] : tensor<1x32x1152x96xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithGroupConvToPermuteCastNoFilterTile
// CHECK-SAME:  [[INPUT0:%.+]]: tensor<1x1x1152x8xf16>
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<1x1x1x1xf16, {order = #NHWC}>
func.func @ReorderWithGroupConvToPermuteCastNoFilterTile(%arg0: tensor<1x1x1152x8xf16>, %arg1: tensor<1x1x1x1xf16, {order = #NHWC}>) -> tensor<1x1x1152x8xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x1x1152x8xf16> -> tensor<1x1x1152x8xf16, {order = #NHWC}>
    %1 = IE.GroupConvolution(%0, %arg1) {dilations = [1, 1], groups = 1 : i64, pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Relu<>, strides = [1, 1]} : tensor<1x1x1152x8xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x1152x8xf16, {order = #NHWC}>
    %2 = IE.Reorder(%1) {dstOrder = #NCHW} : tensor<1x1x1152x8xf16, {order = #NHWC}> -> tensor<1x1x1152x8xf16>

    return %2 : tensor<1x1x1152x8xf16>

    // CHECK:        [[IN_PERMUTECAST:%.+]]  = IE.PermuteCast([[INPUT0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1x1152x8xf16> -> tensor<1x8x1x1152xf16, {order = #NHWC}>
    // CHECK:        [[TILE:%.+]] = IE.Tile([[INPUT1]]) {repeats_values = [8, 1, 1, 1]} : tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<8x1x1x1xf16, {order = #NHWC}>
    // CHECK:        [[GROUPCONV:%.+]]  = IE.GroupConvolution([[IN_PERMUTECAST]], [[TILE]]) {dilations = [1, 1], groups = 8 : i64, pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Relu<>, strides = [1, 1]} : tensor<1x8x1x1152xf16, {order = #NHWC}>, tensor<8x1x1x1xf16, {order = #NHWC}> -> tensor<1x8x1x1152xf16, {order = #NHWC}>
    // CHECK:        [[OUT_PERMUTECAST:%.+]] = IE.PermuteCast([[GROUPCONV]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x8x1x1152xf16, {order = #NHWC}> -> tensor<1x1x1152x8xf16>
  
    // CHECK:        return [[OUT_PERMUTECAST]] : tensor<1x1x1152x8xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithGroupConvWithConcat
// CHECK-SAME:  [[INPUT0:%.+]]: tensor<1x1x1x1xf16, {order = #NHWC}>
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<1x2x1152x32xf16, {order = #NHWC}>
// CHECK-SAME:  [[INPUT2:%.+]]: tensor<1x2x1152x64xf16>
func.func @ReorderWithGroupConvWithConcat(%arg0: tensor<1x1x1x1xf16, {order = #NHWC}>, %arg1: tensor<1x2x1152x32xf16, {order = #NHWC}>, %arg2: tensor<1x2x1152x64xf16>) -> tensor<1x2x1152x96xf16> {

    %0 = IE.Reorder(%arg1) {dstOrder = #NCHW} : tensor<1x2x1152x32xf16, {order = #NHWC}> -> tensor<1x2x1152x32xf16>
    %1 = IE.Concat(%0, %arg2) {per_axis = #IE.Concat<axis = 3>} : tensor<1x2x1152x32xf16>, tensor<1x2x1152x64xf16> -> tensor<1x2x1152x96xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x2x1152x96xf16> -> tensor<1x2x1152x96xf16, {order = #NHWC}>
    %3 = IE.Tile(%arg0) {repeats_values = [1, 2, 1, 1]} : tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x2x1x1xf16, {order = #NHWC}>
    %4 = IE.ShapeCast {shape = [2, 1, 1, 1]} inputs(%3 : tensor<1x2x1x1xf16, {order = #NHWC}>) -> tensor<2x1x1x1xf16, {order = #NHWC}>
    %5 = IE.GroupConvolution(%2, %4) {dilations = [1, 1], groups = 2 : i64, pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Relu<>, strides = [1, 1]} : tensor<1x2x1152x96xf16, {order = #NHWC}>, tensor<2x1x1x1xf16, {order = #NHWC}> -> tensor<1x2x1152x96xf16, {order = #NHWC}>
    %6 = IE.Reorder(%5) {dstOrder = #NCHW} : tensor<1x2x1152x96xf16, {order = #NHWC}> -> tensor<1x2x1152x96xf16>

    return %6 : tensor<1x2x1152x96xf16>

    // CHECK:        [[REORDER:%.+]] = IE.Reorder([[INPUT1]]) {dstOrder = #NCHW} : tensor<1x2x1152x32xf16, {order = #NHWC}> -> tensor<1x2x1152x32xf16>
    // CHECK:        [[CONCAT:%.+]] = IE.Concat([[REORDER]], [[INPUT2]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x2x1152x32xf16>, tensor<1x2x1152x64xf16> -> tensor<1x2x1152x96xf16>
    // CHECK:        [[TILE:%.+]] = IE.Tile([[INPUT0]]) {repeats_values = [1, 2, 1, 1]} : tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x2x1x1xf16, {order = #NHWC}>
    // CHECK:        [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [2, 1, 1, 1]} inputs([[TILE]] : tensor<1x2x1x1xf16, {order = #NHWC}>) -> tensor<2x1x1x1xf16, {order = #NHWC}>
    // CHECK:        [[IN_LAYOUTCAST:%.+]] = IE.LayoutCast([[CONCAT]]) {dst_order = #NHWC} : tensor<1x2x1152x96xf16> -> tensor<1x2x1152x96xf16, {order = #NHWC}>
    // CHECK:        [[GROUPCONV:%.+]] = IE.GroupConvolution([[IN_LAYOUTCAST]], [[SHAPECAST]]) {dilations = [1, 1], groups = 2 : i64, pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Relu<>, strides = [1, 1]} : tensor<1x2x1152x96xf16, {order = #NHWC}>, tensor<2x1x1x1xf16, {order = #NHWC}> -> tensor<1x2x1152x96xf16, {order = #NHWC}>
    // CHECK:        [[OUT_LAYOUTCAST:%.+]] = IE.LayoutCast([[GROUPCONV]]) {dst_order = #NCHW} : tensor<1x2x1152x96xf16, {order = #NHWC}> -> tensor<1x2x1152x96xf16>
    // CHECK:        return [[OUT_LAYOUTCAST]] : tensor<1x2x1152x96xf16>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotOptReorderWithGroupConvForBias
// CHECK-SAME:  [[INPUT0:%.+]]: tensor<1x32x1152x96xf16>
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<1x1x1x1xf16, {order = #NHWC}>
// CHECK-SAME:  [[INPUT2:%.+]]: tensor<1x32x1x1xf16, {order = #NHWC}>
func.func @NotOptReorderWithGroupConvForBias(%arg0: tensor<1x32x1152x96xf16>, %arg1: tensor<1x1x1x1xf16, {order = #NHWC}>, %arg2: tensor<1x32x1x1xf16, {order = #NHWC}>) -> tensor<1x32x1152x96xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x32x1152x96xf16> -> tensor<1x32x1152x96xf16, {order = #NHWC}>
    %1 = IE.Tile(%arg1) {repeats_values = [1, 32, 1, 1]} : tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x32x1x1xf16, {order = #NHWC}>
    %2 = IE.ShapeCast {shape = [32, 1, 1, 1]} inputs(%1 : tensor<1x32x1x1xf16, {order = #NHWC}>) -> tensor<32x1x1x1xf16, {order = #NHWC}>
    %3 = IE.GroupConvolution(%0, %2, %arg2) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x1152x96xf16, {order = #NHWC}>, tensor<32x1x1x1xf16, {order = #NHWC}>, tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x1152x96xf16, {order = #NHWC}>
    %4 = IE.Reorder(%3) {dstOrder = #NCHW} : tensor<1x32x1152x96xf16, {order = #NHWC}> -> tensor<1x32x1152x96xf16>

    return %4 : tensor<1x32x1152x96xf16>

    // CHECK:        [[IN_REORDER:%.+]] = IE.Reorder([[INPUT0]])
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i4:f16:1, {0.1,0.2}>

// CHECK-LABEL: @NotOptReorderWithGroupConvForPerAxisQuant
// CHECK-SAME:  [[INPUT0:%.+]]: tensor<1x2x1152x96xf16>
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<1x1x1x1xf16, {order = #NHWC}>
func.func @NotOptReorderWithGroupConvForPerAxisQuant(%arg0: tensor<1x2x1152x96xf16>, %arg1: tensor<1x1x1x1xf16, {order = #NHWC}>) -> tensor<1x2x1152x96x!qElemType> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x2x1152x96xf16> -> tensor<1x2x1152x96xf16, {order = #NHWC}>
    %1 = IE.Tile(%arg1) {repeats_values = [1, 2, 1, 1]} : tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x2x1x1xf16, {order = #NHWC}>
    %2 = IE.ShapeCast {shape = [2, 1, 1, 1]} inputs(%1 : tensor<1x2x1x1xf16, {order = #NHWC}>) -> tensor<2x1x1x1xf16, {order = #NHWC}>
    %3 = IE.GroupConvolution(%0, %2) {dilations = [1, 1], groups = 2 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x2x1152x96xf16, {order = #NHWC}>, tensor<2x1x1x1xf16, {order = #NHWC}> -> tensor<1x2x1152x96x!qElemType, {order = #NHWC}>
    %4 = IE.Reorder(%3) {dstOrder = #NCHW} : tensor<1x2x1152x96x!qElemType, {order = #NHWC}> -> tensor<1x2x1152x96x!qElemType>

    return %4 : tensor<1x2x1152x96x!qElemType>

    // CHECK:        [[IN_REORDER:%.+]] = IE.Reorder([[INPUT0]])
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotOptReorderWithGroupConvForPerChannelPostOp
// CHECK-SAME:  [[INPUT0:%.+]]: tensor<1x32x1152x96xf16>
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<1x1x1x1xf16, {order = #NHWC}>
func.func @NotOptReorderWithGroupConvForPerChannelPostOp(%arg0: tensor<1x32x1152x96xf16>, %arg1: tensor<1x1x1x1xf16, {order = #NHWC}>) -> tensor<1x32x1152x96xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x32x1152x96xf16> -> tensor<1x32x1152x96xf16, {order = #NHWC}>
    %1 = IE.Tile(%arg1) {repeats_values = [1, 32, 1, 1]} : tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x32x1x1xf16, {order = #NHWC}>
    %2 = IE.ShapeCast {shape = [32, 1, 1, 1]} inputs(%1 : tensor<1x32x1x1xf16, {order = #NHWC}>) -> tensor<32x1x1x1xf16, {order = #NHWC}>
    %3 = IE.GroupConvolution(%0, %2) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PRelu<negative_slope=[1.000000e-01, 2.000000e-01]>, strides = [1, 1]} : tensor<1x32x1152x96xf16, {order = #NHWC}>, tensor<32x1x1x1xf16, {order = #NHWC}> -> tensor<1x32x1152x96xf16, {order = #NHWC}>
    %4 = IE.Reorder(%3) {dstOrder = #NCHW} : tensor<1x32x1152x96xf16, {order = #NHWC}> -> tensor<1x32x1152x96xf16>

    return %4 : tensor<1x32x1152x96xf16>

    // CHECK:        [[IN_REORDER:%.+]] = IE.Reorder([[INPUT0]])
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0053023436490227194:122>
!qElemType1 = !quant.uniform<u8:f16, 0.010899822384703392:128>

// CHECK-LABEL: @ReorderAddQuantCastSlice
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x2x1x1024xf16>
func.func @ReorderAddQuantCastSlice(%arg0: tensor<1x2x1x1024xf16>) -> (tensor<1x1x1x1024x!qElemType, {order = #NHWC}>, tensor<1x1x1x1024x!qElemType, {order = #NHWC}>) {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x2x1x1024xf16> -> tensor<1x2x1x1024xf16, {order = #NHWC}>
    %1 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x2x1x1024xf16> -> tensor<1x2x1x1024xf16, {order = #NHWC}>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x2x1x1024xf16, {order = #NHWC}>, tensor<1x2x1x1024xf16, {order = #NHWC}> -> tensor<1x2x1x1024x!qElemType1, {order = #NHWC}>
    %3 = IE.QuantizeCast(%2) {dstElemType = !qElemType} : tensor<1x2x1x1024x!qElemType1, {order = #NHWC}> -> tensor<1x2x1x1024x!qElemType, {order = #NHWC}>
    %4 = IE.Slice %3 [0, 0, 0, 0] [1, 1, 1, 1024] : tensor<1x2x1x1024x!qElemType, {order = #NHWC}> to tensor<1x1x1x1024x!qElemType, {order = #NHWC}>
    %5 = IE.Slice %3 [0, 1, 0, 0] [1, 1, 1, 1024] : tensor<1x2x1x1024x!qElemType, {order = #NHWC}> to tensor<1x1x1x1024x!qElemType, {order = #NHWC}>
    return %4, %5 : tensor<1x1x1x1024x!qElemType, {order = #NHWC}>, tensor<1x1x1x1024x!qElemType, {order = #NHWC}>

    // CHECK: [[LAYOUTCAST_0:%.+]] = IE.LayoutCast([[INPUT]]) {dst_order = #NHWC} : tensor<1x2x1x1024xf16> -> tensor<1x2x1x1024xf16, {order = #NHWC}>
    // CHECK: [[LAYOUTCAST_1:%.+]] = IE.LayoutCast([[INPUT]]) {dst_order = #NHWC} : tensor<1x2x1x1024xf16> -> tensor<1x2x1x1024xf16, {order = #NHWC}>
    // CHECK: [[ADD:%.+]] = IE.Add([[LAYOUTCAST_0]], [[LAYOUTCAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x2x1x1024xf16, {order = #NHWC}>, tensor<1x2x1x1024xf16, {order = #NHWC}> -> tensor<1x2x1x1024x!qElemType1, {order = #NHWC}>
    // CHECK: [[LAYOUTCAST_2:%.+]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW} : tensor<1x2x1x1024x!qElemType1, {order = #NHWC}> -> tensor<1x2x1x1024x!qElemType1>
    // CHECK: [[QUANTIZECAST:%.+]] = IE.QuantizeCast([[LAYOUTCAST_2]]) {dstElemType = !qElemType} : tensor<1x2x1x1024x!qElemType1> -> tensor<1x2x1x1024x!qElemType>
    // CHECK: [[SLICE_0:%.+]] = IE.Slice [[QUANTIZECAST]] [0, 0, 0, 0] [1, 1, 1, 1024] : tensor<1x2x1x1024x!qElemType> to tensor<1x1x1x1024x!qElemType>
    // CHECK: [[REORDER_0:%.+]] = IE.Reorder([[SLICE_0]]) {dstOrder = #NHWC} : tensor<1x1x1x1024x!qElemType> -> tensor<1x1x1x1024x!qElemType, {order = #NHWC}>
    // CHECK: [[SLICE_1:%.+]] = IE.Slice [[QUANTIZECAST]] [0, 1, 0, 0] [1, 1, 1, 1024] : tensor<1x2x1x1024x!qElemType> to tensor<1x1x1x1024x!qElemType>
    // CHECK: [[REORDER_1:%.+]] = IE.Reorder([[SLICE_1]]) {dstOrder = #NHWC} : tensor<1x1x1x1024x!qElemType> -> tensor<1x1x1x1024x!qElemType, {order = #NHWC}>
    // CHECK: return [[REORDER_0]], [[REORDER_1]] : tensor<1x1x1x1024x!qElemType, {order = #NHWC}>, tensor<1x1x1x1024x!qElemType, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0053023436490227194:122>

// CHECK-LABEL: @ReorderSameInputsAddSlice
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x2x1x1024xf16>
func.func @ReorderSameInputsAddSlice(%arg0: tensor<1x2x1x1024xf16>) -> (tensor<1x1x1x1024x!qElemType, {order = #NHWC}>, tensor<1x1x1x1024x!qElemType, {order = #NHWC}>) {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x2x1x1024xf16> -> tensor<1x2x1x1024xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x2x1x1024xf16, {order = #NHWC}>, tensor<1x2x1x1024xf16, {order = #NHWC}> -> tensor<1x2x1x1024x!qElemType, {order = #NHWC}>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 1, 1, 1024] : tensor<1x2x1x1024x!qElemType, {order = #NHWC}> to tensor<1x1x1x1024x!qElemType, {order = #NHWC}>
    %3 = IE.Slice %1 [0, 1, 0, 0] [1, 1, 1, 1024] : tensor<1x2x1x1024x!qElemType, {order = #NHWC}> to tensor<1x1x1x1024x!qElemType, {order = #NHWC}>
    return %2, %3 : tensor<1x1x1x1024x!qElemType, {order = #NHWC}>, tensor<1x1x1x1024x!qElemType, {order = #NHWC}>

    // CHECK: [[LAYOUTCAST_0:%.+]] = IE.LayoutCast([[INPUT]]) {dst_order = #NHWC} : tensor<1x2x1x1024xf16> -> tensor<1x2x1x1024xf16, {order = #NHWC}>
    // CHECK: [[ADD:%.+]] = IE.Add([[LAYOUTCAST_0]], [[LAYOUTCAST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x2x1x1024xf16, {order = #NHWC}>, tensor<1x2x1x1024xf16, {order = #NHWC}> -> tensor<1x2x1x1024x!qElemType, {order = #NHWC}>
    // CHECK: [[LAYOUTCAST_1:%.+]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW} : tensor<1x2x1x1024x!qElemType, {order = #NHWC}> -> tensor<1x2x1x1024x!qElemType>
    // CHECK: [[SLICE_0:%.+]] = IE.Slice [[LAYOUTCAST_1]] [0, 0, 0, 0] [1, 1, 1, 1024] : tensor<1x2x1x1024x!qElemType> to tensor<1x1x1x1024x!qElemType>
    // CHECK: [[REORDER_0:%.+]] = IE.Reorder([[SLICE_0]]) {dstOrder = #NHWC} : tensor<1x1x1x1024x!qElemType> -> tensor<1x1x1x1024x!qElemType, {order = #NHWC}>
    // CHECK: [[SLICE_1:%.+]] = IE.Slice [[LAYOUTCAST_1]] [0, 1, 0, 0] [1, 1, 1, 1024] : tensor<1x2x1x1024x!qElemType> to tensor<1x1x1x1024x!qElemType>
    // CHECK: [[REORDER_1:%.+]] = IE.Reorder([[SLICE_1]]) {dstOrder = #NHWC} : tensor<1x1x1x1024x!qElemType> -> tensor<1x1x1x1024x!qElemType, {order = #NHWC}>
    // CHECK: return [[REORDER_0]], [[REORDER_1]] : tensor<1x1x1x1024x!qElemType, {order = #NHWC}>, tensor<1x1x1x1024x!qElemType, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0053023436490227194:122>

// CHECK-LABEL: @NotOptReorderAddSliceDueToBranch
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x2x1x1024xf16>
func.func @NotOptReorderAddSliceDueToBranch(%arg0: tensor<1x2x1x1024xf16>) -> (tensor<1x1x1x1024x!qElemType, {order = #NHWC}>, tensor<1x1x1x1024x!qElemType, {order = #NHWC}>, tensor<1x2x1x1024xf16, {order = #NHWC}>) {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x2x1x1024xf16> -> tensor<1x2x1x1024xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x2x1x1024xf16, {order = #NHWC}>, tensor<1x2x1x1024xf16, {order = #NHWC}> -> tensor<1x2x1x1024x!qElemType, {order = #NHWC}>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 1, 1, 1024] : tensor<1x2x1x1024x!qElemType, {order = #NHWC}> to tensor<1x1x1x1024x!qElemType, {order = #NHWC}>
    %3 = IE.Slice %1 [0, 1, 0, 0] [1, 1, 1, 1024] : tensor<1x2x1x1024x!qElemType, {order = #NHWC}> to tensor<1x1x1x1024x!qElemType, {order = #NHWC}>
    %4 = IE.Add(%0, %0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x2x1x1024xf16, {order = #NHWC}>, tensor<1x2x1x1024xf16, {order = #NHWC}> -> tensor<1x2x1x1024xf16, {order = #NHWC}>
    return %2, %3, %4 : tensor<1x1x1x1024x!qElemType, {order = #NHWC}>, tensor<1x1x1x1024x!qElemType, {order = #NHWC}>, tensor<1x2x1x1024xf16, {order = #NHWC}>

    // CHECK: IE.Reorder([[INPUT]])
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0053023436490227194:122>

// CHECK-LABEL: @NotOptReorderAddSliceDueToNonTrivialReorder
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x1x1024xf16>
func.func @NotOptReorderAddSliceDueToNonTrivialReorder(%arg0: tensor<1x16x1x1024xf16>) -> (tensor<1x8x1x1024x!qElemType, {order = #NHWC}>, tensor<1x8x1x1024x!qElemType, {order = #NHWC}>) {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x16x1x1024xf16> -> tensor<1x16x1x1024xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x1x1024xf16, {order = #NHWC}>, tensor<1x16x1x1024xf16, {order = #NHWC}> -> tensor<1x16x1x1024x!qElemType, {order = #NHWC}>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 8, 1, 1024] : tensor<1x16x1x1024x!qElemType, {order = #NHWC}> to tensor<1x8x1x1024x!qElemType, {order = #NHWC}>
    %3 = IE.Slice %1 [0, 8, 0, 0] [1, 8, 1, 1024] : tensor<1x16x1x1024x!qElemType, {order = #NHWC}> to tensor<1x8x1x1024x!qElemType, {order = #NHWC}>
    return %2, %3 : tensor<1x8x1x1024x!qElemType, {order = #NHWC}>, tensor<1x8x1x1024x!qElemType, {order = #NHWC}>

    // CHECK: IE.Reorder([[INPUT]])
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0053023436490227194:122>

// CHECK-LABEL: @NotOptReorderAddSliceDueToWorseSlice
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1024x1x2xf16>
func.func @NotOptReorderAddSliceDueToWorseSlice(%arg0: tensor<1x1024x1x2xf16>) -> (tensor<1x1024x1x1x!qElemType, {order = #NHWC}>, tensor<1x1024x1x1x!qElemType, {order = #NHWC}>) {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x1024x1x2xf16> -> tensor<1x1024x1x2xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1024x1x2xf16, {order = #NHWC}>, tensor<1x1024x1x2xf16, {order = #NHWC}> -> tensor<1x1024x1x2x!qElemType, {order = #NHWC}>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 1024, 1, 1] : tensor<1x1024x1x2x!qElemType, {order = #NHWC}> to tensor<1x1024x1x1x!qElemType, {order = #NHWC}>
    %3 = IE.Slice %1 [0, 0, 0, 1] [1, 1024, 1, 1] : tensor<1x1024x1x2x!qElemType, {order = #NHWC}> to tensor<1x1024x1x1x!qElemType, {order = #NHWC}>
    return %2, %3 : tensor<1x1024x1x1x!qElemType, {order = #NHWC}>, tensor<1x1024x1x1x!qElemType, {order = #NHWC}>

    // CHECK: IE.Reorder([[INPUT]])
}
