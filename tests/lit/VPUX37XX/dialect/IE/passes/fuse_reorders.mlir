//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --fuse-reorders %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FusePermuteToConvolution
func @FusePermuteToConvolution(%arg0: tensor<1x11x16x23xf16, {order = #NHWC}>) -> tensor<1x11x15x21xf16> {
    %cst = const.Declare tensor<11x11x2x3xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<11x11x2x3xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]

    %CONV = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x11x16x23xf16, {order = #NHWC}>,
        tensor<11x11x2x3xf16, {order = #NHWC}> -> tensor<1x11x15x21xf16, {order = #NHWC}>

    %REORDER = IE.Reorder(%CONV) {
        dstOrder = #NCHW
    } : tensor<1x11x15x21xf16, {order = #NHWC}> -> tensor<1x11x15x21xf16>

    return %REORDER : tensor<1x11x15x21xf16>

    // CHECK:   [[CST:%.*]] = const.Declare tensor<11x11x2x3xf16, {order = #NHWC}> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<11x11x2x3xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]

    // CHECK:   [[CONV:%.*]] = IE.Convolution(%arg0, [[CST]]) {
    // CHECK-SAME:  dilations = [1, 1],
    // CHECK-SAME:  pads_begin = [0, 0],
    // CHECK-SAME:  pads_end = [0, 0],
    // CHECK-SAME:  strides = [1, 1]
    // CHECK-SAME: } : tensor<1x11x16x23xf16, {order = #NHWC}>,
    // CHECK-SAME: tensor<11x11x2x3xf16, {order = #NHWC}> -> tensor<1x11x15x21xf16>

    // CHECK:   return [[CONV]] : tensor<1x11x15x21xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FusePermuteToMaxPool
func @FusePermuteToMaxPool(%arg0: tensor<1x11x16x23xf16, {order = #NHWC}>) -> tensor<1x11x14x21xf16> {
    %MAX_POOL = IE.MaxPool(%arg0) {
        kernel_size = [3, 3],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = "FLOOR",
        strides = [1, 1]
    } : tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x14x21xf16, {order = #NHWC}>

    %REORDER = IE.Reorder(%MAX_POOL) {
        dstOrder = #NCHW
    } : tensor<1x11x14x21xf16, {order = #NHWC}> -> tensor<1x11x14x21xf16>

    return %REORDER : tensor<1x11x14x21xf16>

    // CHECK:   [[POOL:%.*]] = IE.MaxPool(%arg0) {
    // CHECK-SAME:  kernel_size = [3, 3],
    // CHECK-SAME:  pads_begin = [0, 0],
    // CHECK-SAME:  pads_end = [0, 0],
    // CHECK-SAME:  rounding_type = "FLOOR",
    // CHECK-SAME:  strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x14x21xf16>

    // CHECK:   return [[POOL]] : tensor<1x11x14x21xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FusePermuteToGroupConv
func @FusePermuteToGroupConv(%arg0: tensor<1x11x16x23xf16, {order = #NHWC}>) -> tensor<1x11x15x21xf16> {
    %cst = const.Declare tensor<11x1x2x3xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<11x1x1x2x3xf32>,
        [#const.Reshape<[11, 1, 2, 3]>, #const.ConvertElemType<f16>, #const.Reorder<#NHWC>]

    %DWCONV = IE.GroupConvolution(%arg0, %cst) {
        dilations = [1, 1],
        groups = 11 : i64,
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x11x16x23xf16, {order = #NHWC}>,
        tensor<11x1x2x3xf16, {order = #NHWC}> -> tensor<1x11x15x21xf16, {order = #NHWC}>

    %REORDER = IE.Reorder(%DWCONV) {
        dstOrder = #NCHW
    } : tensor<1x11x15x21xf16, {order = #NHWC}> -> tensor<1x11x15x21xf16>

    return %REORDER : tensor<1x11x15x21xf16>

    // CHECK:   [[CST:%.*]] = const.Declare tensor<11x1x2x3xf16, {order = #NHWC}> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<11x1x1x2x3xf32>,
    // CHECK-SAME:  [#const.Reshape<[11, 1, 2, 3]>, #const.ConvertElemType<f16>, #const.Reorder<#NHWC>]

    // CHECK:   [[DWCONV:%.*]] = IE.GroupConvolution(%arg0, [[CST]]) {
    // CHECK-SAME:  dilations = [1, 1],
    // CHECK-SAME:  groups = 11 : i64,
    // CHECK-SAME:  pads_begin = [0, 0],
    // CHECK-SAME:  pads_end = [0, 0],
    // CHECK-SAME:  strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x11x16x23xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<11x1x2x3xf16, {order = #NHWC}> -> tensor<1x11x15x21xf16>
    // CHECK:   return [[DWCONV]] : tensor<1x11x15x21xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FusePermuteToAvgPool
func @FusePermuteToAvgPool(%arg0: tensor<1x11x16x23xf16, {order = #NHWC}>) -> tensor<1x11x14x21xf16> {
    %AVG_POOL = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [3, 3],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = "FLOOR",
        strides = [1, 1]
    } : tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x14x21xf16, {order = #NHWC}>

    %REORDER = IE.Reorder(%AVG_POOL) {
        dstOrder = #NCHW
    } : tensor<1x11x14x21xf16, {order = #NHWC}> -> tensor<1x11x14x21xf16>

    return %REORDER : tensor<1x11x14x21xf16>

    // CHECK:   [[AVG_POOL:%.*]] = IE.AvgPool(%arg0) {
    // CHECK-SAME:  exclude_pads,
    // CHECK-SAME:  kernel_size = [3, 3],
    // CHECK-SAME:  pads_begin = [0, 0],
    // CHECK-SAME:  pads_end = [0, 0],
    // CHECK-SAME:  rounding_type = "FLOOR",
    // CHECK-SAME:  strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x14x21xf16>

    // CHECK:   return [[AVG_POOL]] : tensor<1x11x14x21xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FusePermuteToAdd
func @FusePermuteToAdd(%arg0: tensor<1x11x16x23xf16, {order = #NHWC}>) -> tensor<1x11x16x23xf16> {
    %cst = const.Declare tensor<1x11x16x23xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<1x11x16x23xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]

    %ADD = IE.Add(%arg0, %cst) {
        auto_broadcast = "NUMPY"
    } : tensor<1x11x16x23xf16, {order = #NHWC}>,
        tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NHWC}>

    %REORDER = IE.Reorder(%ADD) {
        dstOrder = #NCHW
    } : tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16>

    return %REORDER : tensor<1x11x16x23xf16>

    // CHECK:   [[CST:%.*]] = const.Declare tensor<1x11x16x23xf16, {order = #NHWC}> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<1x11x16x23xf32>,
    // CHECK-SAME:  [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]

    // CHECK:   [[ADD:%.*]] = IE.Add(%arg0, [[CST]]) {
    // CHECK-SAME:  auto_broadcast = "NUMPY"
    // CHECK-SAME:  } : tensor<1x11x16x23xf16, {order = #NHWC}>,
    // CHECK-SAME:  tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16>

    // CHECK:   return [[ADD]] : tensor<1x11x16x23xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseNCWHPermuteToAdd
func @FuseNCWHPermuteToAdd(%arg0: tensor<1x11x16x23xf16, {order = #NHWC}>) -> tensor<1x11x16x23xf16, {order = #NCWH}> {
    %cst = const.Declare tensor<1x11x16x23xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<1x11x16x23xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]

    %ADD = IE.Add(%arg0, %cst) {
        auto_broadcast = "NUMPY"
    } : tensor<1x11x16x23xf16, {order = #NHWC}>,
        tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NHWC}>

    %REORDER = IE.Reorder(%ADD) {
        dstOrder = #NCWH
    } : tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NCWH}>

    return %REORDER : tensor<1x11x16x23xf16, {order = #NCWH}>

    // CHECK:   [[CST:%.*]] = const.Declare tensor<1x11x16x23xf16, {order = #NHWC}> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<1x11x16x23xf32>,
    // CHECK-SAME:  [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]

    // CHECK:   [[ADD:%.*]] = IE.Add(%arg0, [[CST]]) {
    // CHECK-SAME:  auto_broadcast = "NUMPY"
    // CHECK-SAME:  } : tensor<1x11x16x23xf16, {order = #NHWC}>,
    // CHECK-SAME:  tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NCWH}>

    // CHECK:   return [[ADD]] : tensor<1x11x16x23xf16, {order = #NCWH}>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseNHCWPermuteToAdd
func @FuseNHCWPermuteToAdd(%arg0: tensor<1x11x16x23xf16, {order = #NHWC}>) -> tensor<1x11x16x23xf16, {order = #NHCW}> {
    %cst = const.Declare tensor<1x11x16x23xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<1x11x16x23xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]

    %ADD = IE.Add(%arg0, %cst) {
        auto_broadcast = "NUMPY"
    } : tensor<1x11x16x23xf16, {order = #NHWC}>,
        tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NHWC}>

    %REORDER = IE.Reorder(%ADD) {
        dstOrder = #NHCW
    } : tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NHCW}>

    return %REORDER : tensor<1x11x16x23xf16, {order = #NHCW}>

    // CHECK:   [[CST:%.*]] = const.Declare tensor<1x11x16x23xf16, {order = #NHWC}> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<1x11x16x23xf32>,
    // CHECK-SAME:  [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]

    // CHECK:   [[ADD:%.*]] = IE.Add(%arg0, [[CST]]) {
    // CHECK-SAME:  auto_broadcast = "NUMPY"
    // CHECK-SAME:  } : tensor<1x11x16x23xf16, {order = #NHWC}>,
    // CHECK-SAME:  tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NHCW}>

    // CHECK:   return [[ADD]] : tensor<1x11x16x23xf16, {order = #NHCW}>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseNWHCPermuteToAdd
func @FuseNWHCPermuteToAdd(%arg0: tensor<1x11x16x23xf16, {order = #NHWC}>) -> tensor<1x11x16x23xf16, {order = #NWHC}> {
    %cst = const.Declare tensor<1x11x16x23xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<1x11x16x23xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]

    %ADD = IE.Add(%arg0, %cst) {
        auto_broadcast = "NUMPY"
    } : tensor<1x11x16x23xf16, {order = #NHWC}>,
        tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NHWC}>

    %REORDER = IE.Reorder(%ADD) {
        dstOrder = #NWHC
    } : tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NWHC}>

    return %REORDER : tensor<1x11x16x23xf16, {order = #NWHC}>

    // CHECK:   [[CST:%.*]] = const.Declare tensor<1x11x16x23xf16, {order = #NHWC}> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<1x11x16x23xf32>,
    // CHECK-SAME:  [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]

    // CHECK:   [[ADD:%.*]] = IE.Add(%arg0, [[CST]]) {
    // CHECK-SAME:  auto_broadcast = "NUMPY"
    // CHECK-SAME:  } : tensor<1x11x16x23xf16, {order = #NHWC}>,
    // CHECK-SAME:  tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NWHC}>

    // CHECK:   return [[ADD]] : tensor<1x11x16x23xf16, {order = #NWHC}>
}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseNWCHPermuteToAdd
func @FuseNWCHPermuteToAdd(%arg0: tensor<1x11x16x23xf16, {order = #NHWC}>) -> tensor<1x11x16x23xf16, {order = #NWCH}> {
    %cst = const.Declare tensor<1x11x16x23xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<1x11x16x23xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]

    %ADD = IE.Add(%arg0, %cst) {
        auto_broadcast = "NUMPY"
    } : tensor<1x11x16x23xf16, {order = #NHWC}>,
        tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NHWC}>

    %REORDER = IE.Reorder(%ADD) {
        dstOrder = #NWCH
    } : tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NWCH}>

    return %REORDER : tensor<1x11x16x23xf16, {order = #NWCH}>

    // CHECK:   [[CST:%.*]] = const.Declare tensor<1x11x16x23xf16, {order = #NHWC}> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<1x11x16x23xf32>,
    // CHECK-SAME:  [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]

    // CHECK:   [[ADD:%.*]] = IE.Add(%arg0, [[CST]]) {
    // CHECK-SAME:  auto_broadcast = "NUMPY"
    // CHECK-SAME:  } : tensor<1x11x16x23xf16, {order = #NHWC}>,
    // CHECK-SAME:  tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NWCH}>

    // CHECK:   return [[ADD]] : tensor<1x11x16x23xf16, {order = #NWCH}>
}
