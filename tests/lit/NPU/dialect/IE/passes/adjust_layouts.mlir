//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-layouts --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @InOutNHCW
module @InOutNHCW {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x8x4x2xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x8x4x2xf16>
    }

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x8x4x2xf16, {order = #NHCW}>) -> tensor<1x8x4x2xf16, {order = #NHCW}> {
func.func @main(%arg0: tensor<1x8x4x2xf16, {order = #NHCW}>) -> tensor<1x8x4x2xf16, {order = #NHCW}> {
    %0 = IE.GRN(%arg0) {bias = 1.0} : tensor<1x8x4x2xf16, {order = #NHCW}> -> tensor<1x8x4x2xf16, {order = #NHCW}>
    %1 = IE.SoftMax(%0) {axisInd = 1} : tensor<1x8x4x2xf16, {order = #NHCW}> -> tensor<1x8x4x2xf16, {order = #NHCW}>
    %2 = IE.GRN(%1) {bias = 1.0} : tensor<1x8x4x2xf16, {order = #NHCW}> -> tensor<1x8x4x2xf16, {order = #NHCW}>
    return %2 : tensor<1x8x4x2xf16, {order = #NHCW}>

    // CHECK: [[VAR0:%.+]] = IE.Reorder([[ARG0]]) {dstOrder = #NCHW} : tensor<1x8x4x2xf16, {order = #NHCW}> -> tensor<1x8x4x2xf16>
    // CHECK: [[VAR1:%.+]] = IE.GRN([[VAR0]]) {bias = 1.000000e+00 : f64} : tensor<1x8x4x2xf16> -> tensor<1x8x4x2xf16>
    // CHECK: [[VAR2:%.+]] = IE.Reorder([[VAR1]]) {dstOrder = #NHCW} : tensor<1x8x4x2xf16> -> tensor<1x8x4x2xf16, {order = #NHCW}>

    // CHECK: [[VAR3:%.+]] = IE.SoftMax([[VAR2]]) {axisInd = 1 : i64} : tensor<1x8x4x2xf16, {order = #NHCW}> -> tensor<1x8x4x2xf16, {order = #NHCW}>

    // CHECK: [[VAR4:%.+]] = IE.Reorder([[VAR3]]) {dstOrder = #NCHW} : tensor<1x8x4x2xf16, {order = #NHCW}> -> tensor<1x8x4x2xf16>
    // CHECK: [[VAR5:%.+]] = IE.GRN([[VAR4]]) {bias = 1.000000e+00 : f64} : tensor<1x8x4x2xf16> -> tensor<1x8x4x2xf16>
    // CHECK: [[VAR6:%.+]] = IE.Reorder([[VAR5]]) {dstOrder = #NHCW} : tensor<1x8x4x2xf16> -> tensor<1x8x4x2xf16, {order = #NHCW}>

    // CHECK: return [[VAR6]] : tensor<1x8x4x2xf16, {order = #NHCW}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DifferentOrders
module @DifferentOrders {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x8x4x2xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x8x4x2xf16>
    }

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x8x4x2xf16>) -> tensor<1x8x4x2xf16, {order = #NHWC}> {
func.func @main(%arg0: tensor<1x8x4x2xf16>) -> tensor<1x8x4x2xf16, {order = #NHWC}> {
    %0 = IE.GRN(%arg0) {bias = 1.0} : tensor<1x8x4x2xf16> -> tensor<1x8x4x2xf16, {order = #NHWC}>
    return %0 : tensor<1x8x4x2xf16, {order = #NHWC}>

    // CHECK: [[VAR0:%.+]] = IE.GRN([[ARG0]]) {bias = 1.000000e+00 : f64} : tensor<1x8x4x2xf16> -> tensor<1x8x4x2xf16>
    // CHECK: [[VAR1:%.+]] = IE.Reorder([[VAR0]]) {dstOrder = #NHWC} : tensor<1x8x4x2xf16> -> tensor<1x8x4x2xf16, {order = #NHWC}>
    // CHECK: return [[VAR1]] : tensor<1x8x4x2xf16, {order = #NHWC}>
}

}

// -----

// CHECK-LABEL: @HwOp
module @HwOp {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x16x30x30xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x16x15x13xf16>
    }

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x16x30x30xf16>) -> tensor<1x16x15x13xf16> {
func.func @main(%arg0: tensor<1x16x30x30xf16>) -> tensor<1x16x15x13xf16> {
   %0 = IE.MaxPool(%arg0) {
        kernel_size = [5, 5],
        pads_begin = [2, 0],
        pads_end = [2, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [2, 2]
    } : tensor<1x16x30x30xf16> -> tensor<1x16x15x13xf16>
    return %0 : tensor<1x16x15x13xf16>

    // CHECK:       [[VAR0:%.+]] = IE.Reorder([[ARG0]]) {dstOrder = #NHWC} : tensor<1x16x30x30xf16> -> tensor<1x16x30x30xf16, {order = #NHWC}>
    // CHECK:       [[VAR1:%.+]] = IE.MaxPool([[VAR0]])
    // CHECK-SAME:      tensor<1x16x30x30xf16, {order = #NHWC}> -> tensor<1x16x15x13xf16, {order = #NHWC}>
    // CHECK:       [[VAR2:%.+]] = IE.Reorder([[VAR1]]) {dstOrder = #NCHW} : tensor<1x16x15x13xf16, {order = #NHWC}> -> tensor<1x16x15x13xf16>
    // CHECK:       return [[VAR2]] : tensor<1x16x15x13xf16>
}

}

// -----

// CHECK-LABEL: @HwOpSameInputs
module @HwOpSameInputs {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x16x30x25xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x16x30x25xf16>
    }

// CHECK-LABEL: @main
func.func @main(%arg0: tensor<1x16x30x25xf16>) -> tensor<1x16x30x25xf16> {
    %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x30x25xf16>, tensor<1x16x30x25xf16> -> tensor<1x16x30x25xf16>
    %1 = IE.Add(%0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x30x25xf16>, tensor<1x16x30x25xf16> -> tensor<1x16x30x25xf16>
    return %1 : tensor<1x16x30x25xf16>

    // CHECK:    [[VAR0:%.+]] = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x16x30x25xf16> -> tensor<1x16x30x25xf16, {order = #NHWC}>
    // CHECK:    [[VAR1:%.+]] = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x16x30x25xf16> -> tensor<1x16x30x25xf16, {order = #NHWC}>

    // CHECK:    [[VAR2:%.+]] = IE.Add([[VAR0]], [[VAR1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME:     tensor<1x16x30x25xf16, {order = #NHWC}>, tensor<1x16x30x25xf16, {order = #NHWC}> -> tensor<1x16x30x25xf16, {order = #NHWC}>

    // CHECK:    [[VAR3:%.+]] = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x16x30x25xf16> -> tensor<1x16x30x25xf16, {order = #NHWC}>
    // CHECK:    [[VAR4:%.+]] = IE.Add([[VAR2]], [[VAR3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME:     tensor<1x16x30x25xf16, {order = #NHWC}>, tensor<1x16x30x25xf16, {order = #NHWC}> -> tensor<1x16x30x25xf16, {order = #NHWC}>

    // CHECK:    [[VAR5:%.+]] = IE.Reorder([[VAR4]]) {dstOrder = #NCHW} : tensor<1x16x30x25xf16, {order = #NHWC}> -> tensor<1x16x30x25xf16>
    // CHECK:    return [[VAR5]] : tensor<1x16x30x25xf16>
}

}

// -----

// CHECK-LABEL: @SwAddOp
module @SwAddOp {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x3x676x2xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x3x676x2xf16>
    }

// CHECK-LABEL: @main
func.func @main(%arg0: tensor<1x3x676x2xf16>) -> tensor<1x3x676x2xf16> {
    %cst = const.Declare tensor<1x1x676x2xf16> = dense<1.0> : tensor<1x1x676x2xf16>
    %0 = IE.Add(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x676x2xf16>, tensor<1x1x676x2xf16> -> tensor<1x3x676x2xf16>
    return %0 : tensor<1x3x676x2xf16>

    // CHECK-DAG:    [[CST:%.+]] = const.Declare tensor<1x1x676x2xf16> = dense<1.000000e+00> : tensor<1x1x676x2xf16>
    // CHECK:    [[ADD:%.+]] = IE.Add(%arg0, [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME:     tensor<1x3x676x2xf16>, tensor<1x1x676x2xf16> -> tensor<1x3x676x2xf16>
    // CHECK:    return [[ADD]] : tensor<1x3x676x2xf16>
}

}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @HwOpDifferentDstOrder
module @HwOpDifferentDstOrder {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x16x30x25xf16>
    }
    outputsInfo : {
        DataInfo "prob1" : tensor<1x16x30x25xf16>
        DataInfo "prob2" : tensor<1x16x30x25xf16>
    }

// CHECK-LABEL: @main
func.func @main(%arg0: tensor<1x16x30x25xf16, {order = #NHCW}>) -> (tensor<1x16x30x25xf16, {order = #NHCW}>, tensor<1x16x30x25xf16, {order = #NHCW}>) {
    %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x30x25xf16, {order = #NHCW}>, tensor<1x16x30x25xf16, {order = #NHCW}> -> tensor<1x16x30x25xf16, {order = #NHCW}>
    %1 = IE.GRN(%arg0) {bias = 1.0} : tensor<1x16x30x25xf16, {order = #NHCW}> -> tensor<1x16x30x25xf16, {order = #NHCW}>
    return %0, %1 : tensor<1x16x30x25xf16, {order = #NHCW}>, tensor<1x16x30x25xf16, {order = #NHCW}>

    // CHECK:    [[VAR0:%.+]] = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x16x30x25xf16, {order = #NHCW}> -> tensor<1x16x30x25xf16, {order = #NHWC}>
    // CHECK:    [[VAR1:%.+]] = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x16x30x25xf16, {order = #NHCW}> -> tensor<1x16x30x25xf16, {order = #NHWC}>
    // CHECK:    [[VAR2:%.+]] = IE.Add([[VAR0]], [[VAR1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME:     tensor<1x16x30x25xf16, {order = #NHWC}>, tensor<1x16x30x25xf16, {order = #NHWC}> -> tensor<1x16x30x25xf16, {order = #NHWC}>
    // CHECK:    [[VAR3:%.+]] = IE.Reorder([[VAR2]]) {dstOrder = #NHCW} : tensor<1x16x30x25xf16, {order = #NHWC}> -> tensor<1x16x30x25xf16, {order = #NHCW}>

    // CHECK:    [[VAR4:%.+]] = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x16x30x25xf16, {order = #NHCW}> -> tensor<1x16x30x25xf16>
    // CHECK:    [[VAR5:%.+]] = IE.GRN([[VAR4]]) {bias = 1.000000e+00 : f64} : tensor<1x16x30x25xf16> -> tensor<1x16x30x25xf16>
    // CHECK:    [[VAR6:%.+]] = IE.Reorder([[VAR5]]) {dstOrder = #NHCW} : tensor<1x16x30x25xf16> -> tensor<1x16x30x25xf16, {order = #NHCW}>

    // CHECK:    return [[VAR3]], [[VAR6]] : tensor<1x16x30x25xf16, {order = #NHCW}>, tensor<1x16x30x25xf16, {order = #NHCW}>
}

}

// -----

// CHECK-LABEL: @ZMajorConv
module @ZMajorConv {

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x16x30x30xf16>) -> tensor<1x16x30x30xf16> {
func.func @main(%arg0: tensor<1x16x30x30xf16>) -> tensor<1x16x30x30xf16> {
    %cst = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x16x30x30xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x30x30xf16>

    return %0 : tensor<1x16x30x30xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK:       [[VAR0:%.+]] = IE.Reorder([[ARG0]]) {dstOrder = #NHWC}
    // CHECK:       [[VAR1:%.+]] = IE.Convolution([[VAR0]], [[CST]])
    // CHECK-SAME:       -> tensor<1x16x30x30xf16, {order = #NHWC}>
    // CHECK:       [[VAR2:%.+]] = IE.Reorder([[VAR1]]) {dstOrder = #NCHW}
    // CHECK:       return [[VAR2]] : tensor<1x16x30x30xf16>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithScatterNDUpdate
module @ReorderWithScatterNDUpdate {

func.func @main(%arg0: tensor<1x50x56x56xf16, {order = #NHWC}>, %arg1: tensor<1x35x56x56xf16>) -> tensor<1x50x56x56xf16> {
    %cst = const.Declare tensor<1x35x2xsi32> = dense<1> : tensor<1x35x2xsi64>, [#const.CastElemType<si32>]
    %1 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x50x56x56xf16, {order = #NHWC}>, tensor<1x35x2xsi32>, tensor<1x35x56x56xf16> -> tensor<1x50x56x56xf16>
    return %1 : tensor<1x50x56x56xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x35x2xsi32> = dense<1> : tensor<1x35x2xsi64>, [#const.CastElemType<si32>]
    // CHECK:       [[VAR0:%.+]] = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x50x56x56xf16, {order = #NHWC}> -> tensor<1x50x56x56xf16>
    // CHECK:       [[VAR1:%.+]] = IE.ScatterNDUpdate([[VAR0]], [[CST]], %arg1) : tensor<1x50x56x56xf16>, tensor<1x35x2xsi32>, tensor<1x35x56x56xf16> -> tensor<1x50x56x56xf16>
    // CHECK:       return [[VAR1]] : tensor<1x50x56x56xf16>
}
}

// -----

// CHECK-LABEL: @DoNotAdjustInterpolateNearestLayout
module @DoNotAdjustInterpolateNearestLayout {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x3x30x30xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x3x60x60xf16>
    }

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x3x30x30xf16>) -> tensor<1x3x60x60xf16> {
func.func @main(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x60x60xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [60, 60]
         } : tensor<1x3x30x30xf16> -> tensor<1x3x60x60xf16>

    return %0 : tensor<1x3x60x60xf16>

    // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[ARG0]])
    // CHECK-SAME:      attr = #IE.Interpolate<mode = <NEAREST>,
    // CHECK-SAME:                             shape_calc_mode = <SCALES>,
    // CHECK-SAME:                             coord_mode = <ASYMMETRIC>,
    // CHECK-SAME:                             nearest_mode = <FLOOR>,
    // CHECK-SAME:                             antialias = false,
    // CHECK-SAME:                             pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:                             pads_end = [0, 0, 0, 0],
    // CHECK-SAME:                             cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:      axes_attr = [2, 3],
    // CHECK-SAME:      scales_attr = [2.000000e+00, 2.000000e+00],
    // CHECK-SAME:      sizes_attr = [60, 60]
    // CHECK-SAME:      -> tensor<1x3x60x60xf16>

    // CHECK:       return [[INTERP]]
}
}

// -----

// CHECK-LABEL: @DoNotAdjustInterpolateLinearLayout
module @DoNotAdjustInterpolateLinearLayout {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x3x30x30xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x3x60x60xf16>
    }

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x3x30x30xf16>) -> tensor<1x3x60x60xf16> {
func.func @main(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x60x60xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [60, 60]
         } : tensor<1x3x30x30xf16> -> tensor<1x3x60x60xf16>

    return %0 : tensor<1x3x60x60xf16>

    // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[ARG0]])
    // CHECK-SAME:      attr = #IE.Interpolate<mode = <LINEAR>,
    // CHECK-SAME:                             shape_calc_mode = <SCALES>,
    // CHECK-SAME:                             coord_mode = <ASYMMETRIC>,
    // CHECK-SAME:                             nearest_mode = <FLOOR>,
    // CHECK-SAME:                             antialias = false,
    // CHECK-SAME:                             pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:                             pads_end = [0, 0, 0, 0],
    // CHECK-SAME:                             cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:      axes_attr = [2, 3],
    // CHECK-SAME:      scales_attr = [2.000000e+00, 2.000000e+00],
    // CHECK-SAME:      sizes_attr = [60, 60]
    // CHECK-SAME:      -> tensor<1x3x60x60xf16>

    // CHECK:       return [[INTERP]]
}
}

// -----

#HWC = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#NC = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: NonZeroMemoryLayout
func.func  @NonZeroMemoryLayout(%arg0: tensor<1x3x3xsi32, {order = #HWC}>) -> tensor<3x?xsi32, {bounds = #const.OpaqueI64Elements<[3, 9]> : tensor<2xsi64>, order = #NC}> {
    // CHECK:       [[INPUT:%arg.*]]: tensor<1x3x3xsi32, {order = #HWC}>

    %0 = IE.NonZero(%arg0) {dstElemType = si32} : tensor<1x3x3xsi32, {order = #HWC}> -> tensor<3x?xsi32, {bounds = #const.OpaqueI64Elements<[3, 9]> : tensor<2xsi64>, order = #NC}>
    return %0 : tensor<3x?xsi32, {bounds = #const.OpaqueI64Elements<[3, 9]> : tensor<2xsi64>, order = #NC}>

    // CHECK:       [[CHW_INPUT:%.*]] = IE.Reorder([[INPUT]])
    // CHECK-SAME:     dstOrder = #CHW
    // CHECK:       [[NON_ZERO:%.*]] = IE.NonZero([[CHW_INPUT]])
    // CHECK:       return [[NON_ZERO]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MvnLayoutForReshapeFuse
func.func @MvnLayoutForReshapeFuse(%arg0: tensor<1x512x128x128xf16>) -> tensor<1x512x128x128xf16> {

    %cst_0 = const.Declare tensor<512x512x3x3xf16> = dense<1.0> : tensor<512x512x3x3xf16>
    %cst_1 = const.Declare tensor<1x512x1x1xf16> = dense<1.5> : tensor<1x512x1x1xf16>
    %cst_2 = const.Declare tensor<512x1x1x1xf16> = dense<2.0> : tensor<1x512x1x1xf16>, [#const.Reshape<[512, 1, 1, 1]>]
    %cst_3 = const.Declare tensor<1x512x1x1xf16> = dense<2.5> : tensor<1x512x1x1xf16>

    %103 = IE.Convolution(%arg0, %cst_0, %cst_1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x512x128x128xf16>, tensor<512x512x3x3xf16>, tensor<1x512x1x1xf16> -> tensor<1x512x128x128xf16>
    %104 = IE.Reshape(%103) {shape_value = [1, 32, 262144, 1]} : tensor<1x512x128x128xf16> -> tensor<1x32x262144x1xf16>
    %105 = IE.MVN(%104) {across_channels = false, eps = 9.9999999747524271E-7 : f64, normalize_variance = true} : tensor<1x32x262144x1xf16> -> tensor<1x32x262144x1xf16>
    %106 = IE.Reshape(%105) {shape_value = [1, 512, 128, 128]} : tensor<1x32x262144x1xf16> -> tensor<1x512x128x128xf16>
    %107 = IE.GroupConvolution(%106, %cst_2, %cst_3) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x128x128xf16>, tensor<512x1x1x1xf16>, tensor<1x512x1x1xf16> -> tensor<1x512x128x128xf16>

    return %107 : tensor<1x512x128x128xf16>

    // CHECK:  [[MVN_OUT:%.+]] = IE.MVN
    // CHECK-SAME: -> tensor<1x32x262144x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MvnWithAffineReshapeLayoutForReshapeFuse
func.func @MvnWithAffineReshapeLayoutForReshapeFuse(%arg0: tensor<1x512x128x128xf16>) -> tensor<1x512x128x128xf16> {

    %cst_0 = const.Declare tensor<512x512x3x3xf16> = dense<1.0> : tensor<512x512x3x3xf16>
    %cst_1 = const.Declare tensor<1x512x1x1xf16> = dense<1.5> : tensor<1x512x1x1xf16>
    %cst_2 = const.Declare tensor<512x1x1x1xf16> = dense<2.0> : tensor<1x512x1x1xf16>, [#const.Reshape<[512, 1, 1, 1]>]
    %cst_3 = const.Declare tensor<1x512x1x1xf16> = dense<2.5> : tensor<1x512x1x1xf16>

    %103 = IE.Convolution(%arg0, %cst_0, %cst_1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x512x128x128xf16>, tensor<512x512x3x3xf16>, tensor<1x512x1x1xf16> -> tensor<1x512x128x128xf16>
    %104 = IE.AffineReshape(%103) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 32, 16, 16384]} : tensor<1x512x128x128xf16> -> tensor<1x32x16x16384xf16>
    %105 = IE.MVN(%104) {across_channels = false, eps = 9.9999999747524271E-7 : f64, normalize_variance = true} : tensor<1x32x16x16384xf16> -> tensor<1x32x16x16384xf16>
    %106 = IE.AffineReshape(%105) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 512, 128, 128]} : tensor<1x32x16x16384xf16> -> tensor<1x512x128x128xf16>
    %107 = IE.GroupConvolution(%106, %cst_2, %cst_3) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x128x128xf16>, tensor<512x1x1x1xf16>, tensor<1x512x1x1xf16> -> tensor<1x512x128x128xf16>

    return %107 : tensor<1x512x128x128xf16>

    // CHECK:  [[MVN_OUT:%.+]] = IE.MVN
    // CHECK-SAME: -> tensor<1x32x16x16384xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>

// CHECK-LABEL:   @AdjustForGeluWNCH
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x512x6x235xf16, {order = #map}>)
func.func @AdjustForGeluWNCH(%arg0: tensor<1x512x6x235xf16, {order = #map}>) -> tensor<1x512x6x235xf16, {order = #map}> {
    %0 = IE.Gelu(%arg0) : tensor<1x512x6x235xf16, {order = #map}> -> tensor<1x512x6x235xf16, {order = #map}>
    return %0 : tensor<1x512x6x235xf16, {order = #map}>

    // CHECK: [[REORDER0:%.+]] = IE.Reorder([[INPUT]]) {dstOrder = #NCHW} : tensor<1x512x6x235xf16, {order = #map}> -> tensor<1x512x6x235xf16>
    // CHECK: [[GELU:%.+]] = IE.Gelu([[REORDER0]]) : tensor<1x512x6x235xf16> -> tensor<1x512x6x235xf16>
    // CHECK: [[REORDER1:%.+]] = IE.Reorder([[GELU]]) {dstOrder = #map} : tensor<1x512x6x235xf16> -> tensor<1x512x6x235xf16, {order = #map}>
    // CHECK: return [[REORDER1]] : tensor<1x512x6x235xf16, {order = #map}>
}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @AdjustForPReluNWCH
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x64x1x41xf16, {order = #NWCH}>, [[INPUT1:%.+]]: tensor<1x64x1x1xf16, {order = #NWCH}>)
func.func @AdjustForPReluNWCH(%arg0: tensor<1x64x1x41xf16, {order = #NWCH}>,
                              %arg1: tensor<1x64x1x1xf16, {order = #NWCH}>)
                              -> tensor<1x64x1x41xf16, {order = #NWCH}> {
    %0 = IE.PRelu(%arg0, %arg1) : tensor<1x64x1x41xf16, {order = #NWCH}>, tensor<1x64x1x1xf16, {order = #NWCH}> -> tensor<1x64x1x41xf16, {order = #NWCH}>

    return %0 : tensor<1x64x1x41xf16, {order = #NWCH}>

    // CHECK: [[REORDER_IN0:%.+]] = IE.Reorder([[INPUT0]]) {dstOrder = #NCHW} : tensor<1x64x1x41xf16, {order = #NWCH}> -> tensor<1x64x1x41xf16>
    // CHECK: [[REORDER_IN1:%.+]] = IE.Reorder([[INPUT1]]) {dstOrder = #NCHW} : tensor<1x64x1x1xf16, {order = #NWCH}> -> tensor<1x64x1x1xf16>
    // CHECK: [[PRELU:%.+]] = IE.PRelu([[REORDER_IN0]], [[REORDER_IN1]]) : tensor<1x64x1x41xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x1x41xf16>
    // CHECK: [[REORDER_OUT:%.+]] = IE.Reorder([[PRELU]]) {dstOrder = #NWCH} : tensor<1x64x1x41xf16> -> tensor<1x64x1x41xf16, {order = #NWCH}>

    // CHECK: return [[REORDER_OUT]] : tensor<1x64x1x41xf16, {order = #NWCH}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @AdjustForExternalKernel
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x64x1x41xf16>, [[INPUT1:%.+]]: tensor<1x64x1x1xf16>)
func.func @AdjustForExternalKernel(%arg0: tensor<1x64x1x41xf16>,
                                   %arg1: tensor<1x64x1x1xf16>)
                              -> tensor<1x64x1x41xf16> {
    %0 = IE.ExternalKernel "dummy_kernel" at_path("dummy_path") inputs(%arg0, %arg1 : tensor<1x64x1x41xf16>, tensor<1x64x1x1xf16>) attrs({}) -> tensor<1x64x1x41xf16>
    return %0 : tensor<1x64x1x41xf16>

    // CHECK:      [[REORDER_IN0:%.+]] = IE.Reorder([[INPUT0]]) {dstOrder = #NHWC} : tensor<1x64x1x41xf16> -> tensor<1x64x1x41xf16, {order = #NHWC}>
    // CHECK:      [[REORDER_IN1:%.+]] = IE.Reorder([[INPUT1]]) {dstOrder = #NHWC} : tensor<1x64x1x1xf16> -> tensor<1x64x1x1xf16, {order = #NHWC}>
    // CHECK:      [[EXT_KERNEL:%.+]] = IE.ExternalKernel "dummy_kernel" at_path("dummy_path")
    // CHECK-SAME: inputs([[REORDER_IN0]], [[REORDER_IN1]] : tensor<1x64x1x41xf16, {order = #NHWC}>, tensor<1x64x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME: -> tensor<1x64x1x41xf16, {order = #NHWC}>
    // CHECK:      [[REORDER_OUT:%.+]] = IE.Reorder([[EXT_KERNEL]]) {dstOrder = #NCHW} : tensor<1x64x1x41xf16, {order = #NHWC}> -> tensor<1x64x1x41xf16>
    // CHECK:      return [[REORDER_OUT]] : tensor<1x64x1x41xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NoAdjustForExternalKernel
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x64x1x41xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
func.func @NoAdjustForExternalKernel(%arg0: tensor<1x64x1x41xf16, {order = #NHWC}>,
                                   %arg1: tensor<1x64x1x1xf16, {order = #NHWC}>)
                              -> tensor<1x64x1x41xf16, {order = #NHWC}> {
    %0 = IE.ExternalKernel "dummy_kernel" at_path("dummy_path") inputs(%arg0, %arg1 : tensor<1x64x1x41xf16, {order = #NHWC}>, tensor<1x64x1x1xf16, {order = #NHWC}>) attrs({}) -> tensor<1x64x1x41xf16, {order = #NHWC}>
    return %0 : tensor<1x64x1x41xf16, {order = #NHWC}>

    // CHECK-NOT:  IE.Reorder
    // CHECK:      [[EXT_KERNEL:%.+]] = IE.ExternalKernel "dummy_kernel" at_path("dummy_path")
    // CHECK-SAME: inputs([[INPUT0]], [[INPUT1]] : tensor<1x64x1x41xf16, {order = #NHWC}>, tensor<1x64x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME: -> tensor<1x64x1x41xf16, {order = #NHWC}>
    // CHECK-NOT:  IE.Reorder
    // CHECK:      return [[EXT_KERNEL]] : tensor<1x64x1x41xf16, {order = #NHWC}>
}
