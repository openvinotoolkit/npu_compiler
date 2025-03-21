//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-reorders %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FusePermuteToConvolution
func.func @FusePermuteToConvolution(%arg0: tensor<1x11x16x23xf16, {order = #NHWC}>) -> tensor<1x11x15x21xf16> {
    %cst = const.Declare tensor<11x11x2x3xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<11x11x2x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

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

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<11x11x2x3xf16, {order = #NHWC}> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<11x11x2x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

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

// CHECK-LABEL: @FusePermuteToConvolutionWithConvertOp
func.func @FusePermuteToConvolutionWithConvertOp(%arg0: tensor<1x11x16x23xf16, {order = #NHWC}>) -> tensor<1x11x15x21xf32> {
    %cst = const.Declare tensor<11x11x2x3xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<11x11x2x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

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

    %CONVERT = IE.Convert(%REORDER) {dstElemType = f32} : tensor<1x11x15x21xf16> -> tensor<1x11x15x21xf32>

    return %CONVERT : tensor<1x11x15x21xf32>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<11x11x2x3xf16, {order = #NHWC}> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<11x11x2x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    // CHECK:   [[CONV:%.*]] = IE.Convolution(%arg0, [[CST]]) {
    // CHECK-SAME:  dilations = [1, 1],
    // CHECK-SAME:  pads_begin = [0, 0],
    // CHECK-SAME:  pads_end = [0, 0],
    // CHECK-SAME:  strides = [1, 1]
    // CHECK-SAME: } : tensor<1x11x16x23xf16, {order = #NHWC}>,
    // CHECK-SAME: tensor<11x11x2x3xf16, {order = #NHWC}> -> tensor<1x11x15x21xf16>

    // CHECK:   [[CONVERT:%.*]] =  IE.Convert([[CONV]]) {dstElemType = f32}
    // CHECK-SAME: tensor<1x11x15x21xf16> -> tensor<1x11x15x21xf32>

    // CHECK:   return [[CONVERT]] : tensor<1x11x15x21xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FusePermuteToMaxPool
func.func @FusePermuteToMaxPool(%arg0: tensor<1x11x16x23xf16, {order = #NHWC}>) -> tensor<1x11x14x21xf16> {
    %MAX_POOL = IE.MaxPool(%arg0) {
        kernel_size = [3, 3],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
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
    // CHECK-SAME:  rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:  strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x14x21xf16>

    // CHECK:   return [[POOL]] : tensor<1x11x14x21xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FusePermuteToGroupConv
func.func @FusePermuteToGroupConv(%arg0: tensor<1x11x16x23xf16, {order = #NHWC}>) -> tensor<1x11x15x21xf16> {
    %cst = const.Declare tensor<11x1x2x3xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<11x1x1x2x3xf32>,
        [#const.Reshape<[11, 1, 2, 3]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]

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

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<11x1x2x3xf16, {order = #NHWC}> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<11x1x1x2x3xf32>,
    // CHECK-SAME:  [#const.Reshape<[11, 1, 2, 3]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]

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
func.func @FusePermuteToAvgPool(%arg0: tensor<1x11x16x23xf16, {order = #NHWC}>) -> tensor<1x11x14x21xf16> {
    %AVG_POOL = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [3, 3],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
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
    // CHECK-SAME:  rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:  strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x14x21xf16>

    // CHECK:   return [[AVG_POOL]] : tensor<1x11x14x21xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FusePermuteToAdd
func.func @FusePermuteToAdd(%arg0: tensor<1x11x16x23xf16, {order = #NHWC}>) -> tensor<1x11x16x23xf16> {
    %cst = const.Declare tensor<1x11x16x23xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<1x11x16x23xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %ADD = IE.Add(%arg0, %cst) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x11x16x23xf16, {order = #NHWC}>,
        tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NHWC}>

    %REORDER = IE.Reorder(%ADD) {
        dstOrder = #NCHW
    } : tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16>

    return %REORDER : tensor<1x11x16x23xf16>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<1x11x16x23xf16, {order = #NHWC}> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<1x11x16x23xf32>,
    // CHECK-SAME:  [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    // CHECK:   [[ADD:%.*]] = IE.Add(%arg0, [[CST]]) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x11x16x23xf16, {order = #NHWC}>,
    // CHECK-SAME:  tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16>

    // CHECK:   return [[ADD]] : tensor<1x11x16x23xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseNCWHPermuteToAdd
func.func @FuseNCWHPermuteToAdd(%arg0: tensor<1x11x16x23xf16, {order = #NHWC}>) -> tensor<1x11x16x23xf16, {order = #NCWH}> {
    %cst = const.Declare tensor<1x11x16x23xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<1x11x16x23xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %ADD = IE.Add(%arg0, %cst) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x11x16x23xf16, {order = #NHWC}>,
        tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NHWC}>

    %REORDER = IE.Reorder(%ADD) {
        dstOrder = #NCWH
    } : tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NCWH}>

    return %REORDER : tensor<1x11x16x23xf16, {order = #NCWH}>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<1x11x16x23xf16, {order = #NHWC}> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<1x11x16x23xf32>,
    // CHECK-SAME:  [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    // CHECK:   [[ADD:%.*]] = IE.Add(%arg0, [[CST]]) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x11x16x23xf16, {order = #NHWC}>,
    // CHECK-SAME:  tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NCWH}>

    // CHECK:   return [[ADD]] : tensor<1x11x16x23xf16, {order = #NCWH}>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseNHCWPermuteToAdd
func.func @FuseNHCWPermuteToAdd(%arg0: tensor<1x11x16x23xf16, {order = #NHWC}>) -> tensor<1x11x16x23xf16, {order = #NHCW}> {
    %cst = const.Declare tensor<1x11x16x23xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<1x11x16x23xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %ADD = IE.Add(%arg0, %cst) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x11x16x23xf16, {order = #NHWC}>,
        tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NHWC}>

    %REORDER = IE.Reorder(%ADD) {
        dstOrder = #NHCW
    } : tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NHCW}>

    return %REORDER : tensor<1x11x16x23xf16, {order = #NHCW}>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<1x11x16x23xf16, {order = #NHWC}> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<1x11x16x23xf32>,
    // CHECK-SAME:  [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    // CHECK:   [[ADD:%.*]] = IE.Add(%arg0, [[CST]]) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x11x16x23xf16, {order = #NHWC}>,
    // CHECK-SAME:  tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NHCW}>

    // CHECK:   return [[ADD]] : tensor<1x11x16x23xf16, {order = #NHCW}>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseNWHCPermuteToAdd
func.func @FuseNWHCPermuteToAdd(%arg0: tensor<1x11x16x23xf16, {order = #NHWC}>) -> tensor<1x11x16x23xf16, {order = #NWHC}> {
    %cst = const.Declare tensor<1x11x16x23xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<1x11x16x23xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %ADD = IE.Add(%arg0, %cst) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x11x16x23xf16, {order = #NHWC}>,
        tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NHWC}>

    %REORDER = IE.Reorder(%ADD) {
        dstOrder = #NWHC
    } : tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NWHC}>

    return %REORDER : tensor<1x11x16x23xf16, {order = #NWHC}>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<1x11x16x23xf16, {order = #NHWC}> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<1x11x16x23xf32>,
    // CHECK-SAME:  [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    // CHECK:   [[ADD:%.*]] = IE.Add(%arg0, [[CST]]) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x11x16x23xf16, {order = #NHWC}>,
    // CHECK-SAME:  tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NWHC}>

    // CHECK:   return [[ADD]] : tensor<1x11x16x23xf16, {order = #NWHC}>
}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseNWCHPermuteToAdd
func.func @FuseNWCHPermuteToAdd(%arg0: tensor<1x11x16x23xf16, {order = #NHWC}>) -> tensor<1x11x16x23xf16, {order = #NWCH}> {
    %cst = const.Declare tensor<1x11x16x23xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<1x11x16x23xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %ADD = IE.Add(%arg0, %cst) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x11x16x23xf16, {order = #NHWC}>,
        tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NHWC}>

    %REORDER = IE.Reorder(%ADD) {
        dstOrder = #NWCH
    } : tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NWCH}>

    return %REORDER : tensor<1x11x16x23xf16, {order = #NWCH}>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<1x11x16x23xf16, {order = #NHWC}> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<1x11x16x23xf32>,
    // CHECK-SAME:  [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    // CHECK:   [[ADD:%.*]] = IE.Add(%arg0, [[CST]]) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x11x16x23xf16, {order = #NHWC}>,
    // CHECK-SAME:  tensor<1x11x16x23xf16, {order = #NHWC}> -> tensor<1x11x16x23xf16, {order = #NWCH}>

    // CHECK:   return [[ADD]] : tensor<1x11x16x23xf16, {order = #NWCH}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FusePermuteToConvolutionBeforeInterpolate
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @FusePermuteToConvolutionBeforeInterpolate(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x2x256x256xf16> {
    %cst = const.Declare tensor<2x32x1x1xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<2x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %CONV = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<2x32x1x1xf16, {order = #NHWC}> -> tensor<1x2x64x64xf16, {order = #NHWC}>
    %REORDER = IE.Reorder(%CONV) {dstOrder = #NCHW} : tensor<1x2x64x64xf16, {order = #NHWC}> -> tensor<1x2x64x64xf16>
    %INTERPOLATE = IE.Interpolate(%REORDER)
                       {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <FLOOR>, antialias = false,
                       pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
                       operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], sizes_attr = [1, 2, 256, 256]
                       } : tensor<1x2x64x64xf16> -> tensor<1x2x256x256xf16>
    return %INTERPOLATE : tensor<1x2x256x256xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<2x32x1x1xf16, {order = #NHWC}> =
    // CHECK-SAME:  dense<-1.000000e+00> : tensor<2x32x1x1xf16>, [#const.Reorder<#NHWC>]

    // CHECK:   [[CONV:%.+]] = IE.Convolution([[INPUT]], [[CST]]) {
    // CHECK-SAME:  dilations = [1, 1],
    // CHECK-SAME:  pads_begin = [0, 0],
    // CHECK-SAME:  pads_end = [0, 0],
    // CHECK-SAME:  strides = [1, 1]
    // CHECK-SAME: } : tensor<1x32x64x64xf16, {order = #NHWC}>,
    // CHECK-SAME: tensor<2x32x1x1xf16, {order = #NHWC}> -> tensor<1x2x64x64xf16>

    // CHECK:      [[INTERPOLATE:%.+]] = IE.Interpolate([[CONV]])
    // CHECK-SAME: tensor<1x2x64x64xf16> -> tensor<1x2x256x256xf16>

    // CHECK:   return [[INTERPOLATE]] : tensor<1x2x256x256xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FusePermuteToTransposedConv
func.func @FusePermuteToTransposedConv(%arg0: tensor<1x2x256x144xf16, {order = #NHWC}>) -> tensor<1x2x512x288xf16> {
    %cst = const.Declare tensor<2x2x4x4xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<2x2x4x4xf16>, [#const.Reorder<#NHWC>]

    %TransposedConv = IE.TransposedConvolution(%arg0, %cst) {
        dilations = [1, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 0>,
        output_padding = [0, 0],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [2, 2]
    } : tensor<1x2x256x144xf16, {order = #NHWC}>,
        tensor<2x2x4x4xf16, {order = #NHWC}> -> tensor<1x2x512x288xf16, {order = #NHWC}>

    %Reorder = IE.Reorder(%TransposedConv) {
        dstOrder = #NCHW
    } : tensor<1x2x512x288xf16, {order = #NHWC}> -> tensor<1x2x512x288xf16>

    return %Reorder : tensor<1x2x512x288xf16>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<2x2x4x4xf16, {order = #NHWC}> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<2x2x4x4xf16>, [#const.Reorder<#NHWC>]

    // CHECK:   [[TransposedConv:%.*]] = IE.TransposedConvolution(%arg0, [[CST]]) {
    // CHECK-SAME:  dilations = [1, 1],
    // CHECK-SAME:  operandSegmentSizes = array<i32: 1, 1, 0, 0>,
    // CHECK-SAME:  output_padding = [0, 0],
    // CHECK-SAME:  pads_begin = [1, 1],
    // CHECK-SAME:  pads_end = [1, 1],
    // CHECK-SAME:  strides = [2, 2]
    // CHECK-SAME: } : tensor<1x2x256x144xf16, {order = #NHWC}>,
    // CHECK-SAME: tensor<2x2x4x4xf16, {order = #NHWC}> -> tensor<1x2x512x288xf16>

    // CHECK:   return [[TransposedConv]] : tensor<1x2x512x288xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotFusePermuteToAddWithHigherPrecision
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x3x1024x1024xf32, {order = #NHWC}>
func.func @NotFusePermuteToAddWithHigherPrecision(%arg0: tensor<1x3x1024x1024xf32, {order = #NHWC}>) -> tensor<1x3x1024x1024xf32> {
    %cst = const.Declare tensor<1x3x1024x1024xf32, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x3x1024x1024xf32>, [#const.Reorder<#NHWC>]

    %ADD = IE.Add(%arg0, %cst) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x3x1024x1024xf32, {order = #NHWC}>, tensor<1x3x1024x1024xf32, {order = #NHWC}> -> tensor<1x3x1024x1024xf32, {order = #NHWC}>

    %REORDER = IE.Reorder(%ADD) {
        dstOrder = #NCHW
    } : tensor<1x3x1024x1024xf32, {order = #NHWC}> -> tensor<1x3x1024x1024xf32>

    return %REORDER : tensor<1x3x1024x1024xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x3x1024x1024xf32, {order = #NHWC}> =
    // CHECK-SAME:      dense<1.000000e+00> : tensor<1x3x1024x1024xf32>, [#const.Reorder<#NHWC>]

    // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT]], [[CST]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x3x1024x1024xf32, {order = #NHWC}>, tensor<1x3x1024x1024xf32, {order = #NHWC}> -> tensor<1x3x1024x1024xf32, {order = #NHWC}>

    // CHECK:       [[REORDER:%.+]] = IE.Reorder([[ADD]]) {
    // CHECK-SAME:      dstOrder = #NCHW
    // CHECK-SAME:  } : tensor<1x3x1024x1024xf32, {order = #NHWC}> -> tensor<1x3x1024x1024xf32>

    // CHECK:       return [[REORDER]] : tensor<1x3x1024x1024xf32>
}
