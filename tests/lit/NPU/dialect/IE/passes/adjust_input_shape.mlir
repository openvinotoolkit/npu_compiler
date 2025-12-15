//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-input-shape --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ExpandAddToShapeCastAddWithTwoExpands
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x3x32x32xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x3x32x32xf16>
func.func @ExpandAddToShapeCastAddWithTwoExpands(%arg0: tensor<1x3x32x32xf16>, %arg1: tensor<1x3x32x32xf16>) -> tensor<1x16x32x32xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    %1 = IE.Expand(%arg1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x32x32xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
    return %2 : tensor<1x16x32x32xf16>

    // CHECK-NOT:   IE.Expand
    // CHECK-DAG:   [[CAST1:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 12]} inputs([[INPUT1]] : tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16>
    // CHECK-DAG:   [[CAST2:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 12]} inputs([[INPUT2]] : tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16>

    // CHECK:       [[ADD:%.+]] = IE.Add([[CAST1]], [[CAST2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x16x12xf16>, tensor<1x16x16x12xf16> -> tensor<1x16x16x12xf16>
    // CHECK:       [[CAST_OUTPUT:%.+]] = IE.ShapeCast {shape = [1, 3, 32, 32]} inputs([[ADD]] : tensor<1x16x16x12xf16>) -> tensor<1x3x32x32xf16>
    // CHECK:       [[EXPAND_OUTPUT:%.+]] = IE.Expand([[CAST_OUTPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       return [[EXPAND_OUTPUT]]
}

// -----

// CHECK-LABEL: @ExpandAddToShapeCastAddWithTwoExpandAndSlice
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x3x32x32xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x3x32x32xf16>
func.func @ExpandAddToShapeCastAddWithTwoExpandAndSlice(%arg0: tensor<1x3x32x32xf16>, %arg1: tensor<1x3x32x32xf16>) -> tensor<1x3x32x32xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    %1 = IE.Expand(%arg1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x32x32xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
    %3 = IE.Slice %2 [0, 0, 0, 0] [1, 3, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x3x32x32xf16>
    return %3 : tensor<1x3x32x32xf16>

    // CHECK-NOT:   IE.Expand
    // CHECK-DAG:   [[CAST1:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 12]} inputs([[INPUT1]] : tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16>
    // CHECK-DAG:   [[CAST2:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 12]} inputs([[INPUT2]] : tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16>

    // CHECK:       [[ADD:%.+]] = IE.Add([[CAST1]], [[CAST2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x16x12xf16>, tensor<1x16x16x12xf16> -> tensor<1x16x16x12xf16>
    // CHECK:       [[CAST_OUTPUT:%.+]] = IE.ShapeCast {shape = [1, 3, 32, 32]} inputs([[ADD]] : tensor<1x16x16x12xf16>) -> tensor<1x3x32x32xf16>
    // CHECK:       [[EXPAND_OUTPUT:%.+]] = IE.Expand([[CAST_OUTPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       [[SLICE_OUTPUT:%.+]] = IE.Slice [[EXPAND_OUTPUT]] [0, 0, 0, 0] [1, 3, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x3x32x32xf16>
    // CHECK:       return [[SLICE_OUTPUT]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.5>
!qElemType1 = !quant.uniform<u8:f16, 0.25>

// CHECK-LABEL: @ExpandAddToShapeCastAddWithQuantizeCast
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x3x32x32x!qElemType>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x3x32x32x!qElemType>
func.func @ExpandAddToShapeCastAddWithQuantizeCast(%arg0: tensor<1x3x32x32x!qElemType>, %arg1: tensor<1x3x32x32x!qElemType>) -> tensor<1x16x32x32x!qElemType1> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32x!qElemType> -> tensor<1x16x32x32x!qElemType>
    %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType1} :
        tensor<1x16x32x32x!qElemType> -> tensor<1x16x32x32x!qElemType1>
    %2 = IE.Expand(%arg1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32x!qElemType> -> tensor<1x16x32x32x!qElemType>
    %3 = IE.QuantizeCast(%2) {dstElemType = !qElemType1} :
        tensor<1x16x32x32x!qElemType> -> tensor<1x16x32x32x!qElemType1>
    %4 = IE.Add(%1, %3) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x32x32x!qElemType1>, tensor<1x16x32x32x!qElemType1> -> tensor<1x16x32x32x!qElemType1>
    return %4 : tensor<1x16x32x32x!qElemType1>

    // CHECK-NOT:   IE.Expand
    // CHECK:       [[CAST1:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 12]} inputs([[INPUT1]] : tensor<1x3x32x32x!qElemType>) -> tensor<1x16x16x12x!qElemType>
    // CHECK:       [[QUANTIZE1:%.+]] = IE.QuantizeCast([[CAST1]]) {dstElemType = !qElemType1} : tensor<1x16x16x12x!qElemType> -> tensor<1x16x16x12x!qElemType1>
    // CHECK:       [[CAST2:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 12]} inputs([[INPUT2]] : tensor<1x3x32x32x!qElemType>) -> tensor<1x16x16x12x!qElemType>
    // CHECK:       [[QUANTIZE2:%.+]] = IE.QuantizeCast([[CAST2]]) {dstElemType = !qElemType1} : tensor<1x16x16x12x!qElemType> -> tensor<1x16x16x12x!qElemType1>

    // CHECK:       [[ADD:%.+]] = IE.Add([[QUANTIZE1]], [[QUANTIZE2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x16x12x!qElemType1>, tensor<1x16x16x12x!qElemType1> -> tensor<1x16x16x12x!qElemType1>
    // CHECK:       [[CAST_OUTPUT:%.+]] = IE.ShapeCast {shape = [1, 3, 32, 32]} inputs([[ADD]] : tensor<1x16x16x12x!qElemType1>) -> tensor<1x3x32x32x!qElemType1>
    // CHECK:       [[EXPAND_OUTPUT:%.+]] = IE.Expand([[CAST_OUTPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32x!qElemType1> -> tensor<1x16x32x32x!qElemType1>
    // CHECK:       return [[EXPAND_OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandAddToShapeCastAddWithSingleDenseValue
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x1x512xf16, {order = #NHWC}>
func.func @ExpandAddToShapeCastAddWithSingleDenseValue(%arg0: tensor<1x1x1x512xf16, {order = #NHWC}>) -> tensor<1x16x1x512xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x16x1x512xf16, {order = #NHWC}> = dense<1.0> : tensor<512xf16>, [#const.Reshape<[1, 1, 1, 512]>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>]
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x512xf16, {order = #NHWC}> -> tensor<1x16x1x512xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x512xf16, {order = #NHWC}>, tensor<1x16x1x512xf16, {order = #NHWC}> -> tensor<1x16x1x512xf16, {order = #NHWC}>
    return %1 : tensor<1x16x1x512xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x16x8x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512xf16>, [#const.Reshape<[1, 1, 1, 512]>, #const.Reorder<#NHWC>, #const.Reshape<[1, 16, 8, 4]>]
    // CHECK:       [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 16, 8, 4]} inputs([[INPUT]] : tensor<1x1x1x512xf16, {order = #NHWC}>) -> tensor<1x16x8x4xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[SHAPE_CAST]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x8x4xf16, {order = #NHWC}>, tensor<1x16x8x4xf16, {order = #NHWC}> -> tensor<1x16x8x4xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 1, 1, 512]} inputs([[ADD]] : tensor<1x16x8x4xf16, {order = #NHWC}>) -> tensor<1x1x1x512xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[SHAPE_CAST_OUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x512xf16, {order = #NHWC}> -> tensor<1x16x1x512xf16, {order = #NHWC}>
    // CHECK:       return [[EXPAND]] : tensor<1x16x1x512xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandAddToShapeCastAddWithMultiDenseValue
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x1x16xf16, {order = #NHWC}>
func.func @ExpandAddToShapeCastAddWithMultiDenseValue(%arg0: tensor<1x1x1x16xf16, {order = #NHWC}>) -> tensor<1x16x1x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x16x1x16xf16, {order = #NHWC}> =
            dense<[[[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]]]>
            : tensor<1x1x1x16xf16>, [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>]
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x16xf16, {order = #NHWC}> -> tensor<1x16x1x16xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x16xf16, {order = #NHWC}>, tensor<1x16x1x16xf16, {order = #NHWC}> -> tensor<1x16x1x16xf16, {order = #NHWC}>
    return %1 : tensor<1x16x1x16xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<
    // CHECK-SAME:              1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00,
    // CHECK-SAME:              1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00
    // CHECK-SAME:              : tensor<1x1x1x16xf16>, [#const.Reorder<#NHWC>,
    // CHECK-SAME:              #const.Reshape<[1, 16, 1, 1]>]
    // CHECK:       [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 16, 1, 1]} inputs([[INPUT]] : tensor<1x1x1x16xf16, {order = #NHWC}>) -> tensor<1x16x1x1xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[SHAPE_CAST]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x1x1xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 1, 1, 16]} inputs([[ADD]] : tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<1x1x1x16xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[SHAPE_CAST_OUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x16xf16, {order = #NHWC}> -> tensor<1x16x1x16xf16, {order = #NHWC}>
    // CHECK:       return [[EXPAND]] : tensor<1x16x1x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandAddToShapeCastAddWithMultiDenseBroadcastValue
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x1x16xf16, {order = #NHWC}>
func.func @ExpandAddToShapeCastAddWithMultiDenseBroadcastValue(%arg0: tensor<1x1x1x16xf16, {order = #NHWC}>) -> tensor<1x16x1x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x16x1x16xf16, {order = #NHWC}> =
            dense<[[[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]]]>
            : tensor<1x1x1x16xf16>, [#const.Broadcast<0 : i64, 2 : i64>, #const.SubView<[1, 0, 0, 0], [1, 1, 1, 16]>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>]
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x16xf16, {order = #NHWC}> -> tensor<1x16x1x16xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x16xf16, {order = #NHWC}>, tensor<1x16x1x16xf16, {order = #NHWC}> -> tensor<1x16x1x16xf16, {order = #NHWC}>
    return %1 : tensor<1x16x1x16xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<
    // CHECK-SAME:              1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00,
    // CHECK-SAME:              1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00
    // CHECK-SAME:              : tensor<1x1x1x16xf16>, [#const.Broadcast<0 : i64, 2 : i64>, #const.SubView<[1, 0, 0, 0], [1, 1, 1, 16]>,
    // CHECK-SAME:              #const.Reorder<#NHWC>, #const.Reshape<[1, 16, 1, 1]>]
    // CHECK:       [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 16, 1, 1]} inputs([[INPUT]] : tensor<1x1x1x16xf16, {order = #NHWC}>) -> tensor<1x16x1x1xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[SHAPE_CAST]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x1x1xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 1, 1, 16]} inputs([[ADD]] : tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<1x1x1x16xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[SHAPE_CAST_OUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x16xf16, {order = #NHWC}> -> tensor<1x16x1x16xf16, {order = #NHWC}>
    // CHECK:       return [[EXPAND]] : tensor<1x16x1x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandAddToShapeCastAddWithSingleConstInputV1
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x1x512xf16, {order = #NHWC}>
func.func @ExpandAddToShapeCastAddWithSingleConstInputV1(%arg0: tensor<1x1x1x512xf16, {order = #NHWC}>) -> tensor<1x16x1x512xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x16x1x512xf16, {order = #NHWC}> = dense<1.0> : tensor<1x1x1x1xf16>, [#const.Broadcast<3 : i64, 512 : i64>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>]
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x512xf16, {order = #NHWC}> -> tensor<1x16x1x512xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x512xf16, {order = #NHWC}>, tensor<1x16x1x512xf16, {order = #NHWC}> -> tensor<1x16x1x512xf16, {order = #NHWC}>
    return %1 : tensor<1x16x1x512xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x16x8x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<3 : i64, 512 : i64>, #const.Reorder<#NHWC>, #const.Reshape<[1, 16, 8, 4]>]
    // CHECK:       [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 16, 8, 4]} inputs([[INPUT]] : tensor<1x1x1x512xf16, {order = #NHWC}>) -> tensor<1x16x8x4xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[SHAPE_CAST]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x8x4xf16, {order = #NHWC}>, tensor<1x16x8x4xf16, {order = #NHWC}> -> tensor<1x16x8x4xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 1, 1, 512]} inputs([[ADD]] : tensor<1x16x8x4xf16, {order = #NHWC}>) -> tensor<1x1x1x512xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[SHAPE_CAST_OUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x512xf16, {order = #NHWC}> -> tensor<1x16x1x512xf16, {order = #NHWC}>
    // CHECK:       return [[EXPAND]] : tensor<1x16x1x512xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandAddToShapeCastAddWithSingleConstInputV2
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x1x512xf16, {order = #NHWC}>
func.func @ExpandAddToShapeCastAddWithSingleConstInputV2(%arg0: tensor<1x1x1x512xf16, {order = #NHWC}>) -> tensor<1x16x1x512xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x16x1x512xf16, {order = #NHWC}> = dense<1.0> : tensor<1xf16>, [#const.Reshape<[1, 1, 1, 1]>, #const.Broadcast<3 : i64, 512 : i64>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>]
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x512xf16, {order = #NHWC}> -> tensor<1x16x1x512xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x512xf16, {order = #NHWC}>, tensor<1x16x1x512xf16, {order = #NHWC}> -> tensor<1x16x1x512xf16, {order = #NHWC}>
    return %1 : tensor<1x16x1x512xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x16x8x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1xf16>, [#const.Reshape<[1, 1, 1, 1]>, #const.Broadcast<3 : i64, 512 : i64>, #const.Reorder<#NHWC>, #const.Reshape<[1, 16, 8, 4]>]
    // CHECK:       [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 16, 8, 4]} inputs([[INPUT]] : tensor<1x1x1x512xf16, {order = #NHWC}>) -> tensor<1x16x8x4xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[SHAPE_CAST]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x8x4xf16, {order = #NHWC}>, tensor<1x16x8x4xf16, {order = #NHWC}> -> tensor<1x16x8x4xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 1, 1, 512]} inputs([[ADD]] : tensor<1x16x8x4xf16, {order = #NHWC}>) -> tensor<1x1x1x512xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[SHAPE_CAST_OUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x512xf16, {order = #NHWC}> -> tensor<1x16x1x512xf16, {order = #NHWC}>
    // CHECK:       return [[EXPAND]] : tensor<1x16x1x512xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.5>
!qElemType1 = !quant.uniform<u8:f16, 0.25>

// CHECK-LABEL: @ExpandAddToShapeCastAddWithQuantizeDequantizeSequence
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x44x44xf16, {order = #NHWC}>
func.func @ExpandAddToShapeCastAddWithQuantizeDequantizeSequence(%arg0: tensor<1x1x44x44xf16, {order = #NHWC}>) -> tensor<1x1x44x44xf16, {order = #NHWC}> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x44x44xf16, {order = #NHWC}> -> tensor<1x16x44x44xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x44x44xf16, {order = #NHWC}>, tensor<1x16x44x44xf16, {order = #NHWC}> -> tensor<1x16x44x44x!qElemType, {order = #NHWC}>
    %2 = IE.QuantizeCast(%1) {dstElemType = !qElemType1} : tensor<1x16x44x44x!qElemType, {order = #NHWC}> -> tensor<1x16x44x44x!qElemType1, {order = #NHWC}>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x44x44x!qElemType1, {order = #NHWC}>, tensor<1x16x44x44x!qElemType1, {order = #NHWC}> -> tensor<1x16x44x44xf16, {order = #NHWC}>
    %4 = IE.Slice %3 [0, 0, 0, 0] [1, 1, 44, 44] : tensor<1x16x44x44xf16, {order = #NHWC}> to tensor<1x1x44x44xf16, {order = #NHWC}>
    return %4 : tensor<1x1x44x44xf16, {order = #NHWC}>


    // CHECK-NOT:   IE.Expand
    // CHECK:       [[CAST1:%.+]] = IE.ShapeCast {shape = [1, 16, 11, 11]} inputs([[INPUT]] : tensor<1x1x44x44xf16, {order = #NHWC}>) -> tensor<1x16x11x11xf16, {order = #NHWC}>
    // CHECK:       [[ADD1:%.+]] = IE.Add([[CAST1]], [[CAST1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x11x11xf16, {order = #NHWC}>, tensor<1x16x11x11xf16, {order = #NHWC}> -> tensor<1x16x11x11x!qElemType, {order = #NHWC}>
    // CHECK:       [[QUANTIZE:%.+]] = IE.QuantizeCast([[ADD1]]) {dstElemType = !qElemType1} : tensor<1x16x11x11x!qElemType, {order = #NHWC}> -> tensor<1x16x11x11x!qElemType1, {order = #NHWC}>
    // CHECK:       [[ADD2:%.+]] = IE.Add([[QUANTIZE]], [[QUANTIZE]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x11x11x!qElemType1, {order = #NHWC}>, tensor<1x16x11x11x!qElemType1, {order = #NHWC}> -> tensor<1x16x11x11xf16, {order = #NHWC}>
    // CHECK:       [[CAST2:%.+]] = IE.ShapeCast {shape = [1, 1, 44, 44]} inputs([[ADD2]] : tensor<1x16x11x11xf16, {order = #NHWC}>) -> tensor<1x1x44x44xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[CAST2]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x44x44xf16, {order = #NHWC}> -> tensor<1x16x44x44xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[EXPAND]] [0, 0, 0, 0] [1, 1, 44, 44] : tensor<1x16x44x44xf16, {order = #NHWC}> to tensor<1x1x44x44xf16, {order = #NHWC}>
    // CHECK:       return [[SLICE]]
}

// -----

// CHECK-LABEL: @ExpandAddUnsupportedShape
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x3x11x11xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x3x11x11xf16>
func.func @ExpandAddUnsupportedShape(%arg0: tensor<1x3x11x11xf16>, %arg1: tensor<1x3x11x11xf16>) -> tensor<1x16x11x11xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x11x11xf16> -> tensor<1x16x11x11xf16>
    %1 = IE.Expand(%arg1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x11x11xf16> -> tensor<1x16x11x11xf16>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x11x11xf16>, tensor<1x16x11x11xf16> -> tensor<1x16x11x11xf16>
    return %2 : tensor<1x16x11x11xf16>

    // Nothing should be changed
    // the total size 3x11x11 is not divisible by the alignment 16
    // expansion is necessary
    // CHECK-NOT:   IE.ShapeCast
    // CHECK:       [[EXPAND1:%.+]] = IE.Expand([[INPUT1]])
    // CHECK:       [[EXPAND2:%.+]] = IE.Expand([[INPUT2]])

    // CHECK:       [[ADD:%.+]] = IE.Add([[EXPAND1]], [[EXPAND2]])
    // CHECK:       return [[ADD]]
}

// -----

// CHECK-LABEL: @ExpandAddUnsupportedInput
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x3x32x32xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x16x32x32xf16>
func.func @ExpandAddUnsupportedInput(%arg0: tensor<1x3x32x32xf16>, %arg1: tensor<1x16x32x32xf16>) -> tensor<1x16x32x32xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    %2 = IE.Add(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x32x32xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
    return %2 : tensor<1x16x32x32xf16>

    // Nothing should be changed
    // cases are not supported when any of the inputs is not expand
    // CHECK-NOT:   IE.ShapeCast
    // CHECK:       [[EXPAND1:%.+]] = IE.Expand([[INPUT1]])
    // CHECK:       [[ADD:%.+]] = IE.Add([[EXPAND1]], [[INPUT2]])
    // CHECK:       return [[ADD]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandOverWAddToShapeCastAdd
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x16x32x22xf16, {order = #NHWC}>
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x16x32x22xf16, {order = #NHWC}>
func.func @ExpandOverWAddToShapeCastAdd(%arg0: tensor<1x16x32x22xf16, {order = #NHWC}>, %arg1: tensor<1x16x32x22xf16, {order = #NHWC}>) -> tensor<1x16x32x32xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 10]} : tensor<1x16x32x22xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>
    %1 = IE.Expand(%arg1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 10]} : tensor<1x16x32x22xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x32x32xf16, {order = #NHWC}>, tensor<1x16x32x32xf16, {order = #NHWC}>
            -> tensor<1x16x32x32xf16>
    return %2 : tensor<1x16x32x32xf16>

    // Nothing should be changed
    // Eltwise ops with different input and output layouts are not supported
    // CHECK:       [[EXPAND_1:%.+]] = IE.Expand([[INPUT1]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 10]} : tensor<1x16x32x22xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND_2:%.+]] = IE.Expand([[INPUT2]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 10]} : tensor<1x16x32x22xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[EXPAND_1]], [[EXPAND_2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x32x32xf16, {order = #NHWC}>, tensor<1x16x32x32xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16>
    // CHECK:       return [[ADD]] : tensor<1x16x32x32xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AddWithUnsupportedConstInput
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x1x512xf16, {order = #NHWC}>
func.func @AddWithUnsupportedConstInput(%arg0: tensor<1x1x1x512xf16, {order = #NHWC}>) -> tensor<1x16x1x512xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x16x1x512xf16, {order = #NHWC}> = dense<1.0> : tensor<1x8x1x512xf16>, [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 8, 0, 0]>]
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x512xf16, {order = #NHWC}> -> tensor<1x16x1x512xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x512xf16, {order = #NHWC}>, tensor<1x16x1x512xf16, {order = #NHWC}> -> tensor<1x16x1x512xf16, {order = #NHWC}>
    return %1 : tensor<1x16x1x512xf16, {order = #NHWC}>

    // Nothing should be changed
    // cases are not supported when the constant input's real size doesn't equal to another input's size
    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x16x1x512xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x8x1x512xf16>, [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 8, 0, 0]>]
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x512xf16, {order = #NHWC}> -> tensor<1x16x1x512xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[EXPAND]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x512xf16, {order = #NHWC}>, tensor<1x16x1x512xf16, {order = #NHWC}> -> tensor<1x16x1x512xf16, {order = #NHWC}>
    // CHECK:       return [[ADD]] : tensor<1x16x1x512xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @ExpandGroupConvToShapeCastGroupConvDenseWeights
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x32x32xf16>
func.func @ExpandGroupConvToShapeCastGroupConvDenseWeights(%arg0: tensor<1x3x32x32xf16>) -> tensor<1x16x32x32xf16> {
    %filters = const.Declare tensor<16x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 16 : i64>]
    %bias = const.Declare tensor<1x16x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>]
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    %2 = IE.GroupConvolution(%0, %filters, %bias) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf16>, tensor<16x1x1x1xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x32x32xf16>
    return %2 : tensor<1x16x32x32xf16>

    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>, #const.Reshape<[1, 16, 1, 1]>]
    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 16 : i64>, #const.Reshape<[16, 1, 1, 1]>]

    // CHECK-NOT:   IE.Expand
    // CHECK:       [[SHAPE_CAST_IN:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 12]} inputs([[INPUT]] : tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16>
    // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[SHAPE_CAST_IN]], [[FILTER]], [[BIAS]])
    // CHECK-SAME:      dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    // CHECK-SAME:          -> tensor<1x16x16x12xf16>
    // CHECK:       [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 3, 32, 32]} inputs([[GROUP_CONV]] : tensor<1x16x16x12xf16>) -> tensor<1x3x32x32xf16>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[SHAPE_CAST_OUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>

    // CHECK:       return [[EXPAND]]
}

// -----

// CHECK-LABEL: @ExpandGroupConvToShapeCastGroupConvOpaqueWeights
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x32x32xf16>
func.func @ExpandGroupConvToShapeCastGroupConvOpaqueWeights(%arg0: tensor<1x3x32x32xf16>) -> tensor<1x16x32x32xf16> {
    %filters = const.Declare tensor<16x1x1x1xf32> = dense<42.0> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 16 : i64>]
    %bias = const.Declare tensor<1x16x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>]
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    %2 = IE.GroupConvolution(%0, %filters, %bias) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf16>, tensor<16x1x1x1xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x32x32xf16>
    return %2 : tensor<1x16x32x32xf16>

    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>, #const.Reshape<[1, 16, 1, 1]>]
    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<4.200000e+01> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 16 : i64>, #const.Reshape<[16, 1, 1, 1]>]

    // CHECK-NOT:   IE.Expand
    // CHECK:       [[SHAPE_CAST_IN:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 12]} inputs([[INPUT]] : tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16>
    // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[SHAPE_CAST_IN]], [[FILTER]], [[BIAS]])
    // CHECK-SAME:      dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    // CHECK-SAME:          -> tensor<1x16x16x12xf16>
    // CHECK:       [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 3, 32, 32]} inputs([[GROUP_CONV]] : tensor<1x16x16x12xf16>) -> tensor<1x3x32x32xf16>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[SHAPE_CAST_OUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>

    // CHECK:       return [[EXPAND]]
}

// -----

// CHECK-LABEL: @ExpandGroupConvToShapeCastGroupConvWithSingleDenseWeights
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x32x32xf16>
func.func @ExpandGroupConvToShapeCastGroupConvWithSingleDenseWeights(%arg0: tensor<1x3x32x32xf16>) -> tensor<1x16x32x32xf16> {
    %filters = const.Declare tensor<16x1x1x1xf32> = dense<-1.000000e+00> : tensor<3x1x1x1xf32>, [#const.PadWithZero<[0, 0, 0, 0], [13, 0, 0, 0]>]
    %bias = const.Declare tensor<1x16x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>]
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    %2 = IE.GroupConvolution(%0, %filters, %bias) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf16>, tensor<16x1x1x1xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x32x32xf16>
    return %2 : tensor<1x16x32x32xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<-1.000000e+00> : tensor<3x1x1x1xf32>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 1]>, #const.Broadcast<0 : i64, 16 : i64>, #const.Reshape<[16, 1, 1, 1]>]
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>, #const.Reshape<[1, 16, 1, 1]>]

    // CHECK-NOT:   IE.Expand
    // CHECK:       [[SHAPE_CAST_IN:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 12]} inputs([[INPUT]] : tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16>
    // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[SHAPE_CAST_IN]], [[FILTER]], [[BIAS]])
    // CHECK-SAME:      dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    // CHECK-SAME:          -> tensor<1x16x16x12xf16>
    // CHECK:       [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 3, 32, 32]} inputs([[GROUP_CONV]] : tensor<1x16x16x12xf16>) -> tensor<1x3x32x32xf16>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[SHAPE_CAST_OUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>

    // CHECK:       return [[EXPAND]]
}

// -----

// CHECK-LABEL: @SharedWeightsGroupConv
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x32x32xf16>
// CHECK-SAME:        [[INPUT_1:%arg[0-9]]]: tensor<1x16x32x32xf16>
func.func @SharedWeightsGroupConv(%arg0: tensor<1x3x32x32xf16>, %arg1: tensor<1x16x32x32xf16>) -> (tensor<1x16x32x32xf16>, tensor<1x16x32x32xf16>) {
    %filters = const.Declare tensor<16x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 16 : i64>]
    %bias = const.Declare tensor<1x16x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>]
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    %2 = IE.GroupConvolution(%0, %filters, %bias) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf16>, tensor<16x1x1x1xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x32x32xf16>
    %3 = IE.GroupConvolution(%arg1, %filters, %bias) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf16>, tensor<16x1x1x1xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x32x32xf16>
    return %2, %3 : tensor<1x16x32x32xf16>, tensor<1x16x32x32xf16>

    // CHECK-DAG:   [[BIAS_0:%.+]] = const.Declare tensor<1x16x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>, #const.Reshape<[1, 16, 1, 1]>]
    // CHECK-DAG:   [[FILTER_0:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 16 : i64>, #const.Reshape<[16, 1, 1, 1]>]

    // CHECK-DAG:   [[BIAS_1:%.+]] = const.Declare tensor<1x16x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>]
    // CHECK-DAG:   [[FILTER_1:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 16 : i64>]

    // CHECK-NOT:   IE.Expand
    // CHECK:       [[SHAPE_CAST_IN:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 12]} inputs([[INPUT]] : tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16>
    // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[SHAPE_CAST_IN]], [[FILTER_0]], [[BIAS_0]])
    // CHECK-SAME:      dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    // CHECK-SAME:          -> tensor<1x16x16x12xf16>
    // CHECK:       [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 3, 32, 32]} inputs([[GROUP_CONV]] : tensor<1x16x16x12xf16>) -> tensor<1x3x32x32xf16>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[SHAPE_CAST_OUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>

    // CHECK:       [[GROUP_CONV_1:%.+]] = IE.GroupConvolution([[INPUT_1]], [[FILTER_1]], [[BIAS_1]])
    // CHECK-SAME:      dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    // CHECK-SAME:          -> tensor<1x16x32x32xf16>

    // CHECK:       return [[EXPAND]], [[GROUP_CONV_1:%.+]]
}


// -----

!qElemType = !quant.uniform<u8:f16, 0.013957817414227655:161>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandGroupConvToShapeCastGroupConvWithReshapeAttribute
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x25x56x56xf16, {order = #NHWC}>
func.func @ExpandGroupConvToShapeCastGroupConvWithReshapeAttribute(%arg0: tensor<1x25x56x56xf16, {order = #NHWC}>) -> tensor<1x32x56x56x!quant.uniform<u8:f16, 0.013957817414227655:161>, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x1x1x1xf16, {order = #NHWC}> = dense<0.0232320447> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>, #const.Broadcast<1 : i64, 25 : i64>, #const.Reshape<[25, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [7, 0, 0, 0]>]
    %cst_1 = const.Declare tensor<1x32x1x1xf16> = dense<-2.4650476> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>, #const.Broadcast<1 : i64, 25 : i64>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>]

    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 7, 0, 0]} : tensor<1x25x56x56xf16, {order = #NHWC}> -> tensor<1x32x56x56xf16, {order = #NHWC}>
    %1 = IE.GroupConvolution(%0, %cst_0, %cst_1) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x56x56xf16, {order = #NHWC}>, tensor<32x1x1x1xf16, {order = #NHWC}>, tensor<1x32x1x1xf16> -> tensor<1x32x56x56x!quant.uniform<u8:f16, 0.013957817414227655:161>, {order = #NHWC}>

    return %1 : tensor<1x32x56x56x!quant.uniform<u8:f16, 0.013957817414227655:161>, {order = #NHWC}>

    // CHECK-DAG:    [[CST:%.+]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<0.0232320447> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>, #const.Reshape<[16, 1, 1, 1]>]
    // CHECK-DAG:    [[CST_1:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<-2.4650476> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>, #const.Broadcast<1 : i64, 16 : i64>, #const.Reshape<[1, 16, 1, 1]>]
    // CHECK:        [[SHAPECAST_1:%.+]] = IE.ShapeCast {shape = [1, 16, 70, 70]} inputs(%arg0 : tensor<1x25x56x56xf16, {order = #NHWC}>) -> tensor<1x16x70x70xf16, {order = #NHWC}>

    // CHECK:        [[GROUP_CONV:%.+]] = IE.GroupConvolution([[SHAPECAST_1]], [[CST]], [[CST_1]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      groups = 16 : i64, pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0], strides = [1, 1]
    // CHECK-SAME:      } : tensor<1x16x70x70xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16>
    // CHECK-SAME:          -> tensor<1x16x70x70x!qElemType, {order = #NHWC}>
    // CHECK:        [[SHAPECAST_2:%.+]] = IE.ShapeCast {shape = [1, 25, 56, 56]} inputs([[GROUP_CONV]] : tensor<1x16x70x70x!qElemType, {order = #NHWC}>) -> tensor<1x25x56x56x!qElemType, {order = #NHWC}>
    // CHECK:        [[EXPAND:%.+]] = IE.Expand([[SHAPECAST_2]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 7, 0, 0]} : tensor<1x25x56x56x!qElemType, {order = #NHWC}> -> tensor<1x32x56x56x!qElemType, {order = #NHWC}>

    // CHECK:       return [[EXPAND]]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandGroupConvToShapeCastGroupConvWithAffineReshapeAttribute
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x12x56x56xf16>
func.func @ExpandGroupConvToShapeCastGroupConvWithAffineReshapeAttribute(%arg0: tensor<1x12x56x56xf16>) -> tensor<1x16x56x56xf16> {
    %cst_0 = const.Declare tensor<16x1x1x1xf16> = dense<0.0232320447> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 12 : i64>, #const.AffineReshape<[[0], [0], [1], [2, 3]], [12, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [4, 0, 0, 0]>]
    %cst_1 = const.Declare tensor<1x16x1x1xf16> = dense<-2.4650476> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 16 : i64>]

    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 4, 0, 0]} : tensor<1x12x56x56xf16> -> tensor<1x16x56x56xf16>
    %1 = IE.GroupConvolution(%0, %cst_0, %cst_1) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x56x56xf16>, tensor<16x1x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x56x56xf16>

    return %1 : tensor<1x16x56x56xf16>

    // CHECK-DAG:    [[CST:%.+]] = const.Declare tensor<16x1x1x1xf16> = dense<2.323910e-02> : tensor<1x1x1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>, #const.Broadcast<0 : i64, 16 : i64>, #const.Reshape<[16, 1, 1, 1]>]
    // CHECK-DAG:    [[CST_1:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<-2.464840e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 16 : i64>, #const.Reshape<[1, 16, 1, 1]>]
    // CHECK:        [[SHAPECAST_1:%.+]] = IE.ShapeCast {shape = [1, 16, 49, 48]} inputs(%arg0 : tensor<1x12x56x56xf16>) -> tensor<1x16x49x48xf16>

    // CHECK:        [[GROUP_CONV:%.+]] = IE.GroupConvolution([[SHAPECAST_1]], [[CST]], [[CST_1]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      groups = 16 : i64, pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0], strides = [1, 1]
    // CHECK-SAME:      } : tensor<1x16x49x48xf16>, tensor<16x1x1x1xf16>, tensor<1x16x1x1xf16>
    // CHECK-SAME:          -> tensor<1x16x49x48xf16>
    // CHECK:        [[SHAPECAST_2:%.+]] = IE.ShapeCast {shape = [1, 12, 56, 56]} inputs([[GROUP_CONV]] : tensor<1x16x49x48xf16>) -> tensor<1x12x56x56xf16>
    // CHECK:        [[EXPAND:%.+]] = IE.Expand([[SHAPECAST_2]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 4, 0, 0]} : tensor<1x12x56x56xf16> -> tensor<1x16x56x56xf16>

    // CHECK:       return [[EXPAND]]

}

// -----

// CHECK-LABEL: @NotAdjustGroupConvToShapeCastGroupConv
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x32x32xf16>
func.func @NotAdjustGroupConvToShapeCastGroupConv(%arg0: tensor<1x3x32x32xf16>) -> tensor<1x16x31x31xf16> {
    %filters = const.Declare tensor<16x1x2x2xf32> = dense<1.0> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 16 : i64>, #const.Broadcast<2 : i64, 2 : i64>, #const.Broadcast<3 : i64, 2 : i64>]
    %bias = const.Declare tensor<1x16x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>]
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    %2 = IE.GroupConvolution(%0, %filters, %bias) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf16>, tensor<16x1x2x2xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x31x31xf16>
    return %2 : tensor<1x16x31x31xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<16x1x2x2xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 16 : i64>, #const.Broadcast<2 : i64, 2 : i64>, #const.Broadcast<3 : i64, 2 : i64>]
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>]
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[EXPAND]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf16>, tensor<16x1x2x2xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x31x31xf16>
    // CHECK:       return [[GROUP_CONV]] : tensor<1x16x31x31xf16>
}

// -----

// CHECK-LABEL: @NotAdjustNonEltwiseGroupConvStrides2x1
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x32x32xf16>
func.func @NotAdjustNonEltwiseGroupConvStrides2x1(%arg0: tensor<1x3x32x32xf16>) -> tensor<1x16x16x32xf16> {
    %filters = const.Declare tensor<16x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 16 : i64>]
    %bias = const.Declare tensor<1x16x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>]
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    %2 = IE.GroupConvolution(%0, %filters, %bias) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} : tensor<1x16x32x32xf16>, tensor<16x1x1x1xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x16x32xf16>
    return %2 : tensor<1x16x16x32xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 16 : i64>]
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>]
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[EXPAND]], [[FILTER]], [[BIAS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      groups = 16 : i64, pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0], strides = [2, 1]
    // CHECK-SAME:      } : tensor<1x16x32x32xf16>, tensor<16x1x1x1xf32>, tensor<1x16x1x1xf32>
    // CHECK-SAME:          -> tensor<1x16x16x32xf16>

    // CHECK:       return [[GROUP_CONV]] : tensor<1x16x16x32xf16>
}

// -----

// CHECK-LABEL: @NotAdjustNonEltwiseGroupConvKernel2x1
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x32x32xf16>
func.func @NotAdjustNonEltwiseGroupConvKernel2x1(%arg0: tensor<1x3x32x32xf16>) -> tensor<1x16x31x32xf16> {
    %filters = const.Declare tensor<16x1x2x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 16 : i64>, #const.Broadcast<2 : i64, 2 : i64>]
    %bias = const.Declare tensor<1x16x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>]
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    %2 = IE.GroupConvolution(%0, %filters, %bias) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf16>, tensor<16x1x2x1xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x31x32xf16>
    return %2 : tensor<1x16x31x32xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<16x1x2x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 16 : i64>, #const.Broadcast<2 : i64, 2 : i64>]
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>]
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[EXPAND]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf16>, tensor<16x1x2x1xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x31x32xf16>
    // CHECK:       return [[GROUP_CONV]] : tensor<1x16x31x32xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AdjustInputShapeForPermuteQuantize
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1280x8x8xf16>
func.func @AdjustInputShapeForPermuteQuantize(%arg0: tensor<1x1280x8x8xf16>) -> tensor<1x1280x8x8xf16, {order = #NHWC}> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 8]} : tensor<1x1280x8x8xf16> -> tensor<1x1280x8x16xf16>
    %1 = IE.PermuteQuantize(%0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1280x8x16xf16> -> tensor<1x1280x8x16xf16, {order = #NHWC}>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 1280, 8, 8] : tensor<1x1280x8x16xf16, {order = #NHWC}> to tensor<1x1280x8x8xf16, {order = #NHWC}>
    return %2 : tensor<1x1280x8x8xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.Expand
    // CHECK:       [[CAST:%.+]] = IE.ShapeCast {shape = [1, 1280, 4, 16]} inputs([[INPUT]] : tensor<1x1280x8x8xf16>) -> tensor<1x1280x4x16xf16>
    // CHECK:       [[PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize([[CAST]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1280x4x16xf16> -> tensor<1x1280x4x16xf16, {order = #NHWC}>
    // CHECK:       [[CAST_OUTPUT:%.+]] = IE.ShapeCast {shape = [1, 1280, 8, 8]} inputs([[PERMUTEQUANTIZE]] : tensor<1x1280x4x16xf16, {order = #NHWC}>) -> tensor<1x1280x8x8xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND_OUTPUT:%.+]] = IE.Expand([[CAST_OUTPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 8]} : tensor<1x1280x8x8xf16, {order = #NHWC}> -> tensor<1x1280x8x16xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_OUTPUT:%.+]] = IE.Slice [[EXPAND_OUTPUT]] [0, 0, 0, 0] [1, 1280, 8, 8] : tensor<1x1280x8x16xf16, {order = #NHWC}> to tensor<1x1280x8x8xf16, {order = #NHWC}>
    // CHECK:       return [[SLICE_OUTPUT]]
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @NotAdjustInputShapeForPermuteQuantizeWithInvalidLayouts
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1280x8x8xf16>
func.func @NotAdjustInputShapeForPermuteQuantizeWithInvalidLayouts(%arg0: tensor<1x1280x8x8xf16>) -> tensor<1x1280x8x8xf16, {order = #NHCW}> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 8]} : tensor<1x1280x8x8xf16> -> tensor<1x1280x8x16xf16>
    %1 = IE.PermuteQuantize(%0) {dstElemType = f16, dst_order = #NHCW, mem_perm = #NHCW, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1280x8x16xf16> -> tensor<1x1280x8x16xf16, {order = #NHCW}>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 1280, 8, 8] : tensor<1x1280x8x16xf16, {order = #NHCW}> to tensor<1x1280x8x8xf16, {order = #NHCW}>
    return %2 : tensor<1x1280x8x8xf16, {order = #NHCW}>

    // Nothing should be changed
    // cases are not supported when the layouts are invalid
    // CHECK-NOT:   IE.ShapeCast
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]])
    // CHECK:       [[PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize([[EXPAND]]) {dstElemType = f16, dst_order = #NHCW, mem_perm = #NHCW, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1280x8x16xf16> -> tensor<1x1280x8x16xf16, {order = #NHCW}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[PERMUTEQUANTIZE]] [0, 0, 0, 0] [1, 1280, 8, 8] : tensor<1x1280x8x16xf16, {order = #NHCW}> to tensor<1x1280x8x8xf16, {order = #NHCW}>
    // CHECK:       return [[SLICE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotAdjustInputShapeForPermuteQuantizeWithInvalidExpand
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x384x672xf16>
func.func @NotAdjustInputShapeForPermuteQuantizeWithInvalidExpand(%arg0: tensor<1x3x384x672xf16>) -> tensor<1x16x384x672xf16, {order = #NHWC}> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x384x672xf16> -> tensor<1x16x384x672xf16>
    %1 = IE.PermuteQuantize(%0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x16x384x672xf16> -> tensor<1x16x384x672xf16, {order = #NHWC}>
    return %1 : tensor<1x16x384x672xf16, {order = #NHWC}>

    // Nothing should be changed as it should only handle width expanding
    // CHECK-NOT:   IE.ShapeCast
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]])
    // CHECK:       [[PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize([[EXPAND]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x16x384x672xf16> -> tensor<1x16x384x672xf16, {order = #NHWC}>
    // CHECK:       return [[PERMUTEQUANTIZE]]
}

// -----

// CHECK-LABEL: @AdjustAvgPoolingToShapeCastAvgPooling
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x1x2048xf16>
func.func @AdjustAvgPoolingToShapeCastAvgPooling(%arg0: tensor<1x1x1x2048xf16>) -> tensor<1x1x1x2048xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x2048xf16> -> tensor<1x16x1x2048xf16>
    %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.10000000149011612 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x1x2048xf16> -> tensor<1x16x1x2048xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 1, 1, 2048] : tensor<1x16x1x2048xf16> to tensor<1x1x1x2048xf16>
    return %2 : tensor<1x1x1x2048xf16>

    // ExpandPoolingRewriter will be used instead of ExpandSingleChannelPoolingRewriter due to benefit level setting
    // CHECK:   [[SHAPECAST0:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 8]} inputs([[INPUT]] : tensor<1x1x1x2048xf16>) -> tensor<1x16x16x8xf16>
    // CHECK:   [[POOLING:%.+]] = IE.AvgPool([[SHAPECAST0]])
    // CHECK:   [[SHAPECAST0:%.+]] = IE.ShapeCast {shape = [1, 1, 1, 2048]} inputs([[POOLING]] : tensor<1x16x16x8xf16>) -> tensor<1x1x1x2048xf16>
    // CHECK:   [[EXPAND:%.+]] = IE.Expand([[SHAPECAST0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x2048xf16> -> tensor<1x16x1x2048xf16>
    // CHECK:   [[SLICE:%.+]] = IE.Slice [[EXPAND]] [0, 0, 0, 0] [1, 1, 1, 2048] : tensor<1x16x1x2048xf16> to tensor<1x1x1x2048xf16>
    // CHECK:       return [[SLICE]] : tensor<1x1x1x2048xf16>
}

// -----

// CHECK-LABEL: @NotAdjustAvgPoolingWrongKernel
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x1x2048xf16>
func.func @NotAdjustAvgPoolingWrongKernel(%arg0: tensor<1x1x1x2048xf16>) -> tensor<1x1x1x2047xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x2048xf16> -> tensor<1x16x1x2048xf16>
    %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [1, 2], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.10000000149011612 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x1x2048xf16> -> tensor<1x16x1x2047xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 1, 1, 2047] : tensor<1x16x1x2047xf16> to tensor<1x1x1x2047xf16>
    return %2 : tensor<1x1x1x2047xf16>

    // CHECK:   [[EXPAND:%.+]] = IE.Expand
    // CHECK:   [[POOLING:%.+]] = IE.AvgPool
    // CHECK:   [[SLICE:%.+]] = IE.Slice
    // CHECK:       return [[SLICE]] : tensor<1x1x1x2047xf16>
}


// -----

// CHECK-LABEL: @NotAdjustAvgPoolingWrongPad
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x1x2048xf16>
func.func @NotAdjustAvgPoolingWrongPad(%arg0: tensor<1x1x1x2048xf16>) -> tensor<1x1x2x2049xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x2048xf16> -> tensor<1x16x1x2048xf16>
    %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [1, 0], pads_end = [0, 1], post_op = #IE.LeakyRelu<negative_slope = 0.10000000149011612 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x1x2048xf16> -> tensor<1x16x2x2049xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 1, 2, 2049] : tensor<1x16x2x2049xf16> to tensor<1x1x2x2049xf16>
    return %2 : tensor<1x1x2x2049xf16>

    // CHECK:   [[EXPAND:%.+]] = IE.Expand
    // CHECK:   [[POOLING:%.+]] = IE.AvgPool
    // CHECK:   [[SLICE:%.+]] = IE.Slice
    // CHECK:       return [[SLICE]] : tensor<1x1x2x2049xf16>
}


// -----

// CHECK-LABEL: @NotAdjustAvgPoolingWrongStride
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x1x2048xf16>
func.func @NotAdjustAvgPoolingWrongStride(%arg0: tensor<1x1x1x2048xf16>) -> tensor<1x1x1x1024xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x2048xf16> -> tensor<1x16x1x2048xf16>
    %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.10000000149011612 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 2]} : tensor<1x16x1x2048xf16> -> tensor<1x16x1x1024xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 1, 1, 1024] : tensor<1x16x1x1024xf16> to tensor<1x1x1x1024xf16>
    return %2 : tensor<1x1x1x1024xf16>

    // CHECK:   [[EXPAND:%.+]] = IE.Expand
    // CHECK:   [[POOLING:%.+]] = IE.AvgPool
    // CHECK:   [[SLICE:%.+]] = IE.Slice
    // CHECK:       return [[SLICE]] : tensor<1x1x1x1024xf16>
}


// -----

// CHECK-LABEL: @AdjustMaxPoolingToShapeCastMaxPooling
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x1x2048xf16>
func.func @AdjustMaxPoolingToShapeCastMaxPooling(%arg0: tensor<1x1x1x2048xf16>) -> tensor<1x1x1x2048xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x2048xf16> -> tensor<1x16x1x2048xf16>
    %1 = IE.MaxPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.10000000149011612 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x1x2048xf16> -> tensor<1x16x1x2048xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 1, 1, 2048] : tensor<1x16x1x2048xf16> to tensor<1x1x1x2048xf16>
    return %2 : tensor<1x1x1x2048xf16>

    // CHECK:   [[SHAPECAST0:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 8]} inputs([[INPUT]] : tensor<1x1x1x2048xf16>) -> tensor<1x16x16x8xf16>
    // CHECK:   [[POOLING:%.+]] = IE.MaxPool([[SHAPECAST0]])
    // CHECK:   [[SHAPECAST0:%.+]] = IE.ShapeCast {shape = [1, 1, 1, 2048]} inputs([[POOLING]] : tensor<1x16x16x8xf16>) -> tensor<1x1x1x2048xf16>
    // CHECK:   [[EXPAND:%.+]] = IE.Expand([[SHAPECAST0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x2048xf16> -> tensor<1x16x1x2048xf16>
    // CHECK:   [[SLICE:%.+]] = IE.Slice [[EXPAND]] [0, 0, 0, 0] [1, 1, 1, 2048] : tensor<1x16x1x2048xf16> to tensor<1x1x1x2048xf16>
    // CHECK:       return [[SLICE]] : tensor<1x1x1x2048xf16>
}


// -----

// CHECK-LABEL: @NotAdjustMaxPoolingWrongKernel
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x1x2048xf16>
func.func @NotAdjustMaxPoolingWrongKernel(%arg0: tensor<1x1x1x2048xf16>) -> tensor<1x1x1x2047xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x2048xf16> -> tensor<1x16x1x2048xf16>
    %1 = IE.MaxPool(%0) {exclude_pads, kernel_size = [1, 2], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.10000000149011612 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x1x2048xf16> -> tensor<1x16x1x2047xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 1, 1, 2047] : tensor<1x16x1x2047xf16> to tensor<1x1x1x2047xf16>
    return %2 : tensor<1x1x1x2047xf16>

    // CHECK:   [[EXPAND:%.+]] = IE.Expand
    // CHECK:   [[POOLING:%.+]] = IE.MaxPool
    // CHECK:   [[SLICE:%.+]] = IE.Slice
    // CHECK:       return [[SLICE]] : tensor<1x1x1x2047xf16>
}


// -----

// CHECK-LABEL: @NotAdjustMaxPoolingWrongPad
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x1x2048xf16>
func.func @NotAdjustMaxPoolingWrongPad(%arg0: tensor<1x1x1x2048xf16>) -> tensor<1x1x2x2049xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x2048xf16> -> tensor<1x16x1x2048xf16>
    %1 = IE.MaxPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [1, 0], pads_end = [0, 1], post_op = #IE.LeakyRelu<negative_slope = 0.10000000149011612 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x1x2048xf16> -> tensor<1x16x2x2049xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 1, 2, 2049] : tensor<1x16x2x2049xf16> to tensor<1x1x2x2049xf16>
    return %2 : tensor<1x1x2x2049xf16>

    // CHECK:   [[EXPAND:%.+]] = IE.Expand
    // CHECK:   [[POOLING:%.+]] = IE.MaxPool
    // CHECK:   [[SLICE:%.+]] = IE.Slice
    // CHECK:       return [[SLICE]] : tensor<1x1x2x2049xf16>
}


// -----

// CHECK-LABEL: @NotAdjustMaxPoolingWrongStride
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x1x2048xf16>
func.func @NotAdjustMaxPoolingWrongStride(%arg0: tensor<1x1x1x2048xf16>) -> tensor<1x1x1x1024xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x2048xf16> -> tensor<1x16x1x2048xf16>
    %1 = IE.MaxPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.10000000149011612 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 2]} : tensor<1x16x1x2048xf16> -> tensor<1x16x1x1024xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 1, 1, 1024] : tensor<1x16x1x1024xf16> to tensor<1x1x1x1024xf16>
    return %2 : tensor<1x1x1x1024xf16>

    // CHECK:   [[EXPAND:%.+]] = IE.Expand
    // CHECK:   [[POOLING:%.+]] = IE.MaxPool
    // CHECK:   [[SLICE:%.+]] = IE.Slice
    // CHECK:       return [[SLICE]] : tensor<1x1x1x1024xf16>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotAdjustMaxPoolingDifferentLayout
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x8x8xf16, {order = #NHWC}>
func.func @NotAdjustMaxPoolingDifferentLayout(%arg0: tensor<1x3x8x8xf16, {order = #NHWC}>) -> tensor<1x3x8x8xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x8x8xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
    %1 = IE.MaxPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.10000000149011612 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x8x8xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 3, 8, 8] : tensor<1x16x8x8xf16> to tensor<1x3x8x8xf16>
    return %2 : tensor<1x3x8x8xf16>

    // CHECK:   [[EXPAND:%.+]] = IE.Expand
    // CHECK:   [[POOLING:%.+]] = IE.MaxPool
    // CHECK:   [[SLICE:%.+]] = IE.Slice
    // CHECK:       return [[SLICE]] : tensor<1x3x8x8xf16>
}

// -----

// CHECK-LABEL: @NotAdjustNonEltwiseGroupConv
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x32x32xf16>
func.func @NotAdjustNonEltwiseGroupConv(%arg0: tensor<1x3x32x32xf16>) -> tensor<1x16x16x32xf16> {
    %filters = const.Declare tensor<16x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 16 : i64>]
    %bias = const.Declare tensor<1x16x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>]
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    %2 = IE.GroupConvolution(%0, %filters, %bias) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} : tensor<1x16x32x32xf16>, tensor<16x1x1x1xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x16x32xf16>
    return %2 : tensor<1x16x16x32xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 16 : i64>]
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>]
    // CHECK:       [[EXPAND:%.+]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[EXPAND]], [[FILTER]], [[BIAS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      groups = 16 : i64, pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0], strides = [2, 1]
    // CHECK-SAME:      } : tensor<1x16x32x32xf16>, tensor<16x1x1x1xf32>, tensor<1x16x1x1xf32>
    // CHECK-SAME:          -> tensor<1x16x16x32xf16>

    // CHECK:       return [[GROUP_CONV]] : tensor<1x16x16x32xf16>
}

// -----

// CHECK-LABEL: @KeepConstReshapeAttrIfIsDimShrinkWhenShapeCastEltwise
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x8x2xf16>
func.func @KeepConstReshapeAttrIfIsDimShrinkWhenShapeCastEltwise(%arg0: tensor<1x3x8x2xf16>) -> tensor<1x16x8x2xf16> {
    %cst = const.Declare tensor<1x16x8x2xf16> = dense<42.0>
     : tensor<1x3x2x4x2xf32>, [#const.Reshape<[1, 3, 8, 2]>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]
    %expand = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x8x2xf16> -> tensor<1x16x8x2xf16>
    %add = IE.Add(%expand, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x8x2xf16>, tensor<1x16x8x2xf16> -> tensor<1x16x8x2xf16>
    return %add : tensor<1x16x8x2xf16>

    // CHECK-DAG:    [[CST:%.+]] = const.Declare tensor<1x16x3x1xf16> = dense<4.200000e+01> : tensor<1x3x2x4x2xf32>,
    // CHECK-SAME:      [#const.Reshape<[1, 16, 3, 1]>, #const.CastElemType<f16>]
    // CHECK:        [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 16, 3, 1]} inputs(%arg0 : tensor<1x3x8x2xf16>) -> tensor<1x16x3x1xf16>
    // CHECK:        [[ADD:%.+]] = IE.Add([[SHAPECAST_IN]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x3x1xf16>, tensor<1x16x3x1xf16> -> tensor<1x16x3x1xf16>
    // CHECK:        [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 3, 8, 2]} inputs([[ADD]] : tensor<1x16x3x1xf16>) -> tensor<1x3x8x2xf16>
    // CHECK:        [[EXPAND:%.+]] = IE.Expand([[SHAPECAST_OUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x8x2xf16> -> tensor<1x16x8x2xf16>

    // CHECK:       return [[EXPAND]]
}

// -----

 #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AddWithBoradCastAndPaddingConstInput
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x8x1x512xf16, {order = #NHWC}>
func.func @AddWithBoradCastAndPaddingConstInput(%arg0: tensor<1x8x1x512xf16, {order = #NHWC}>) -> tensor<1x16x1x512xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x16x1x512xf16, {order = #NHWC}> = dense<1.0> : tensor<1x1x1x512xf16>, [#const.Broadcast<1 : i64, 8 : i64>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 8, 0, 0]>]
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x8x1x512xf16, {order = #NHWC}> -> tensor<1x16x1x512xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x512xf16, {order = #NHWC}>, tensor<1x16x1x512xf16, {order = #NHWC}> -> tensor<1x16x1x512xf16, {order = #NHWC}>
    return %1 : tensor<1x16x1x512xf16, {order = #NHWC}>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x16x16x16xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x512xf16>, [#const.Broadcast<1 : i64, 8 : i64>, #const.Reorder<#NHWC>, #const.Reshape<[1, 16, 16, 16]>]
    // CHECK:       [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 16]} inputs([[INPUT]] : tensor<1x8x1x512xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[SHAPECAST_IN]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:       [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 8, 1, 512]} inputs([[ADD]] : tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x8x1x512xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[SHAPECAST_OUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x8x1x512xf16, {order = #NHWC}> -> tensor<1x16x1x512xf16, {order = #NHWC}>
    // CHECK:       return [[EXPAND]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AddWithComplexConstInput
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x8x1x512xf16, {order = #NHWC}>
func.func @AddWithComplexConstInput(%arg0: tensor<1x8x1x512xf16, {order = #NHWC}>) -> tensor<1x16x1x512xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x16x1x512xf16, {order = #NHWC}> = dense<1.0> : tensor<1x1x1x512xf16>, [#const.PadWithZero<[0, 0, 0, 0], [0, 8, 0, 0]>, #const.SubView<[0, 0, 0, 0], [1, 8, 1, 512]>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 8, 0, 0]>]
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x8x1x512xf16, {order = #NHWC}> -> tensor<1x16x1x512xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x512xf16, {order = #NHWC}>, tensor<1x16x1x512xf16, {order = #NHWC}> -> tensor<1x16x1x512xf16, {order = #NHWC}>
    return %1 : tensor<1x16x1x512xf16, {order = #NHWC}>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x16x16x16xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x512xf16>,
    // CHECK-SAME:    [#const.PadWithZero<[0, 0, 0, 0], [0, 8, 0, 0]>, #const.SubView<[0, 0, 0, 0], [1, 8, 1, 512]>, #const.Reorder<#NHWC>, #const.Reshape<[1, 16, 16, 16]>]
    // CHECK:       [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 16]} inputs([[INPUT]] : tensor<1x8x1x512xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[SHAPECAST_IN]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:       [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 8, 1, 512]} inputs([[ADD]] : tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x8x1x512xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[SHAPECAST_OUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x8x1x512xf16, {order = #NHWC}> -> tensor<1x16x1x512xf16, {order = #NHWC}>
    // CHECK:       return [[EXPAND]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NotAdjustForDifferentPaddingAttr
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x8x1x512xf16, {order = #NHWC}>
func.func @NotAdjustForDifferentPaddingAttr(%arg0: tensor<1x8x1x512xf16, {order = #NHWC}>) -> tensor<1x16x1x512xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x16x1x512xf16, {order = #NHWC}> = dense<1.0> : tensor<1x1x1x512xf16>, [#const.PadWithZero<[0, 0, 0, 0], [0, 8, 0, 0]>, #const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [1, 8, 1, 512]>, #const.PadWithZero<[0, 8, 0, 0], [0, 0, 0, 0]>]
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x8x1x512xf16, {order = #NHWC}> -> tensor<1x16x1x512xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x512xf16, {order = #NHWC}>, tensor<1x16x1x512xf16, {order = #NHWC}> -> tensor<1x16x1x512xf16, {order = #NHWC}>
    return %1 : tensor<1x16x1x512xf16, {order = #NHWC}>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x16x1x512xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]])
    // CHECK:       [[ADD:%.+]] = IE.Add([[EXPAND]], [[CST]])
    // CHECK:       return [[ADD]]
}

// -----

// CHECK-LABEL: @ExpandSingleAddToShapeCastAdd
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x3x32x32xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x16x32x32xf16>
func.func @ExpandSingleAddToShapeCastAdd(%arg0: tensor<1x3x32x32xf16>, %arg1: tensor<1x16x32x32xf16>) -> tensor<1x3x32x32xf16> {
    %0 = IE.Add(%arg1, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x32x32xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
    %1 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x32x32xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
    %3 = IE.Slice %2 [0, 0, 0, 0] [1, 3, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x3x32x32xf16>
    return %3 : tensor<1x3x32x32xf16>

    // CHECK:        [[ADD_1:%.+]] = IE.Add(%arg1, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x32x32xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
    // CHECK:        [[SLICE:%.+]] = IE.Slice [[ADD_1]] [0, 0, 0, 0] [1, 3, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x3x32x32xf16>
    // CHECK:        [[SHAPECAST_1:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 12]} inputs([[SLICE]] : tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16>
    // CHECK:        [[SHAPECAST_2:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 12]} inputs(%arg0 : tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16>
    // CHECK:        [[ADD:%.+]] = IE.Add([[SHAPECAST_1]], [[SHAPECAST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x16x12xf16>, tensor<1x16x16x12xf16> -> tensor<1x16x16x12xf16>
    // CHECK:        [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 3, 32, 32]} inputs([[ADD]] : tensor<1x16x16x12xf16>) -> tensor<1x3x32x32xf16>
    // CHECK:        [[EXPAND:%.+]] = IE.Expand([[SHAPECAST_OUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    // CHECK:        [[SLICE_OUT:%.+]] = IE.Slice [[EXPAND]] [0, 0, 0, 0] [1, 3, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x3x32x32xf16>

    // CHECK:       return [[SLICE_OUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @PropagateShapeCastBeforeEltwise
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x16x256x192xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x16x256x192xf16, {order = #NHWC}>
func.func @PropagateShapeCastBeforeEltwise(
        %input_0: tensor<1x16x256x192xf16, {order = #NHWC}>,
        %input_1: tensor<1x16x256x192xf16, {order = #NHWC}>)
            -> tensor<1x48x512x32xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x48x3x3xf16>, [#const.Reorder<#NHWC>]
    %eltwise = IE.Add(%input_0, %input_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x16x256x192xf16, {order = #NHWC}>,
        tensor<1x16x256x192xf16, {order = #NHWC}> -> tensor<1x16x256x192xf16, {order = #NHWC}>
    %shapecast = IE.ShapeCast {shape = [1, 48, 512, 32]}
        inputs(%eltwise : tensor<1x16x256x192xf16, {order = #NHWC}>) -> tensor<1x48x512x32xf16, {order = #NHWC}>
    %conv = IE.Convolution(%shapecast, %weights) {
        dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} :
        tensor<1x48x512x32xf16, {order = #NHWC}>, tensor<48x48x3x3xf16, {order = #NHWC}>
            -> tensor<1x48x512x32xf16, {order = #NHWC}>

    return %conv : tensor<1x48x512x32xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_1:%.+]] = IE.ShapeCast {shape = [1, 48, 512, 32]}
    // CHECK-SAME:      inputs([[INPUT1]] : tensor<1x16x256x192xf16, {order = #NHWC}>) -> tensor<1x48x512x32xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_2:%.+]] = IE.ShapeCast {shape = [1, 48, 512, 32]}
    // CHECK-SAME:      inputs([[INPUT2]] : tensor<1x16x256x192xf16, {order = #NHWC}>) -> tensor<1x48x512x32xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[SHAPE_CAST_1]], [[SHAPE_CAST_2]])
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[ADD]], [[WEIGHTS]])
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @PropagateShapeCastAfterEltwise
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x48x512x32xf16, {order = #NHWC}>
func.func @PropagateShapeCastAfterEltwise(%input: tensor<1x48x512x32xf16, {order = #NHWC}>)
            -> tensor<1x16x256x192xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x48x3x3xf16>, [#const.Reorder<#NHWC>]
    %conv = IE.Convolution(%input, %weights) {
        dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} :
        tensor<1x48x512x32xf16, {order = #NHWC}>, tensor<48x48x3x3xf16, {order = #NHWC}>
            -> tensor<1x48x512x32xf16, {order = #NHWC}>
    %shapecast = IE.ShapeCast {shape = [1, 16, 256, 192]}
        inputs(%conv : tensor<1x48x512x32xf16, {order = #NHWC}>) -> tensor<1x16x256x192xf16, {order = #NHWC}>
    %eltwise = IE.Add(%shapecast, %shapecast) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x16x256x192xf16, {order = #NHWC}>,
        tensor<1x16x256x192xf16, {order = #NHWC}> -> tensor<1x16x256x192xf16, {order = #NHWC}>

    return %eltwise : tensor<1x16x256x192xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]])
    // CHECK:       [[ADD:%.+]] = IE.Add([[CONV]], [[CONV]])
    // CHECK:       [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 16, 256, 192]}
    // CHECK-SAME:      inputs([[ADD]] : tensor<1x48x512x32xf16, {order = #NHWC}>) -> tensor<1x16x256x192xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @DoNotPropagateShapeCastWithUnalignedShape
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x16x256x256xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x16x256x256xf16, {order = #NHWC}>
func.func @DoNotPropagateShapeCastWithUnalignedShape(
        %input_0: tensor<1x16x256x256xf16, {order = #NHWC}>,
        %input_1: tensor<1x16x256x256xf16, {order = #NHWC}>)
            -> tensor<1x16x512x512xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x4x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x4x3x3xf16>, [#const.Reorder<#NHWC>]
    %eltwise = IE.Add(%input_0, %input_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x16x256x256xf16, {order = #NHWC}>,
        tensor<1x16x256x256xf16, {order = #NHWC}> -> tensor<1x16x256x256xf16, {order = #NHWC}>
    %shapecast = IE.ShapeCast {shape = [1, 4, 512, 512]}
        inputs(%eltwise : tensor<1x16x256x256xf16, {order = #NHWC}>) -> tensor<1x4x512x512xf16, {order = #NHWC}>
    %conv = IE.Convolution(%shapecast, %weights) {
        dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} :
        tensor<1x4x512x512xf16, {order = #NHWC}>, tensor<16x4x3x3xf16, {order = #NHWC}>
            -> tensor<1x16x512x512xf16, {order = #NHWC}>

    return %conv : tensor<1x16x512x512xf16, {order = #NHWC}>

    // CHECK:   [[WEIGHTS:%.+]] = const.Declare tensor<16x4x3x3xf16, {order = #NHWC}>
    // CHECK-NOT:   IE.ShapeCast

    // CHECK:       [[ADD:%.+]] =  IE.Add([[INPUT1]], [[INPUT2]])
    // CHECK:       [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 4, 512, 512]}
    // CHECK-SAME:      inputs([[ADD]] : tensor<1x16x256x256xf16, {order = #NHWC}>) -> tensor<1x4x512x512xf16, {order = #NHWC}>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[SHAPE_CAST]], [[WEIGHTS]])
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @ExpandPoolingOpWithSingleChannel
// CHECK-SAME:        [[INPUT0:%arg[0-9]]]: tensor<1x1x48x4040xf16, {order = #NHWC}>
func.func @ExpandPoolingOpWithSingleChannel(%input_0: tensor<1x1x48x4040xf16, {order = #NHWC}>)
            -> tensor<1x1x6x4040xf16, {order = #NHWC}> {
    %expand = IE.Expand(%input_0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x48x4040xf16, {order = #NHWC}> -> tensor<1x16x48x4040xf16, {order = #NHWC}>
    %pool = IE.AvgPool(%expand) {exclude_pads, kernel_size = [8, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 1]} : tensor<1x16x48x4040xf16, {order = #NHWC}> -> tensor<1x16x6x4040xf16, {order = #NHWC}>
    %slice = IE.Slice %pool [0, 0, 0, 0] [1, 1, 6, 4040] : tensor<1x16x6x4040xf16, {order = #NHWC}> to tensor<1x1x6x4040xf16, {order = #NHWC}>
    return %slice : tensor<1x1x6x4040xf16, {order = #NHWC}>

    // CHECK:       [[SHAPECAST0:%.*]] = IE.ShapeCast {shape = [1, 4040, 48, 1]} inputs([[INPUT0]] : tensor<1x1x48x4040xf16, {order = #NHWC}>) -> tensor<1x4040x48x1xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND0:%.*]] = IE.Expand([[SHAPECAST0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x4040x48x1xf16, {order = #NHWC}> -> tensor<1x4048x48x1xf16, {order = #NHWC}>
    // CHECK:       [[POOL:%.*]] = IE.AvgPool([[EXPAND0]]) {exclude_pads, kernel_size = [8, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 1]} : tensor<1x4048x48x1xf16, {order = #NHWC}> -> tensor<1x4048x6x1xf16, {order = #NHWC}>
    // CHECK:       [[SHAPECAST1:%.*]] = IE.ShapeCast {shape = [1, 1, 6, 4048]} inputs([[POOL]] : tensor<1x4048x6x1xf16, {order = #NHWC}>) -> tensor<1x1x6x4048xf16, {order = #NHWC}>
    // CHECK:       [[SLICE0:%.*]] = IE.Slice [[SHAPECAST1]] [0, 0, 0, 0] [1, 1, 6, 4040] : tensor<1x1x6x4048xf16, {order = #NHWC}> to tensor<1x1x6x4040xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND1:%.*]] = IE.Expand([[SLICE0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x6x4040xf16, {order = #NHWC}> -> tensor<1x16x6x4040xf16, {order = #NHWC}>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice [[EXPAND1]] [0, 0, 0, 0] [1, 1, 6, 4040] : tensor<1x16x6x4040xf16, {order = #NHWC}> to tensor<1x1x6x4040xf16, {order = #NHWC}>

    // CHECK:       return [[SLICE1]] : tensor<1x1x6x4040xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @ExpandTwoPoolingOpsWithSingleChannel
// CHECK-SAME:        [[INPUT0:%arg[0-9]]]: tensor<1x1x48x4040xf16, {order = #NHWC}>
func.func @ExpandTwoPoolingOpsWithSingleChannel(%input_0: tensor<1x1x48x4040xf16, {order = #NHWC}>)
            -> tensor<1x1x6x4040xf16, {order = #NHWC}> {
    %expand = IE.Expand(%input_0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x48x4040xf16, {order = #NHWC}> -> tensor<1x16x48x4040xf16, {order = #NHWC}>
    %pool0 = IE.AvgPool(%expand) {exclude_pads, kernel_size = [8, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 1]} : tensor<1x16x48x4040xf16, {order = #NHWC}> -> tensor<1x16x6x4040xf16, {order = #NHWC}>
    %pool1 = IE.AvgPool(%pool0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.10000000149011612 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x6x4040xf16, {order = #NHWC}> -> tensor<1x16x6x4040xf16, {order = #NHWC}>
    %slice = IE.Slice %pool1 [0, 0, 0, 0] [1, 1, 6, 4040] : tensor<1x16x6x4040xf16, {order = #NHWC}> to tensor<1x1x6x4040xf16, {order = #NHWC}>
    return %slice : tensor<1x1x6x4040xf16, {order = #NHWC}>

    // CHECK:       [[SHAPECAST0:%.*]] = IE.ShapeCast {shape = [1, 4040, 48, 1]} inputs([[INPUT0]] : tensor<1x1x48x4040xf16, {order = #NHWC}>) -> tensor<1x4040x48x1xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND0:%.*]] = IE.Expand([[SHAPECAST0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x4040x48x1xf16, {order = #NHWC}> -> tensor<1x4048x48x1xf16, {order = #NHWC}>
    // CHECK:       [[POOL0:%.*]] = IE.AvgPool([[EXPAND0]]) {exclude_pads, kernel_size = [8, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 1]} : tensor<1x4048x48x1xf16, {order = #NHWC}> -> tensor<1x4048x6x1xf16, {order = #NHWC}>
    // CHECK:       [[SHAPECAST1:%.*]] = IE.ShapeCast {shape = [1, 1, 6, 4048]} inputs([[POOL0]] : tensor<1x4048x6x1xf16, {order = #NHWC}>) -> tensor<1x1x6x4048xf16, {order = #NHWC}>
    // CHECK:       [[SLICE0:%.*]] = IE.Slice [[SHAPECAST1]] [0, 0, 0, 0] [1, 1, 6, 4040] : tensor<1x1x6x4048xf16, {order = #NHWC}> to tensor<1x1x6x4040xf16, {order = #NHWC}>
    // CHECK:       [[SHAPECAST2:%.*]] = IE.ShapeCast {shape = [1, 16, 101, 15]} inputs(%4 : tensor<1x1x6x4040xf16, {order = #NHWC}>) -> tensor<1x16x101x15xf16, {order = #NHWC}>
    // CHECK:       [[POOL1:%.*]] = IE.AvgPool([[SHAPECAST2]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.10000000149011612 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x101x15xf16, {order = #NHWC}> -> tensor<1x16x101x15xf16, {order = #NHWC}>
    // CHECK:       [[SHAPECAST3:%.*]] = IE.ShapeCast {shape = [1, 1, 6, 4040]} inputs([[POOL1]] : tensor<1x16x101x15xf16, {order = #NHWC}>) -> tensor<1x1x6x4040xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND1:%.*]] = IE.Expand([[SHAPECAST3]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x6x4040xf16, {order = #NHWC}> -> tensor<1x16x6x4040xf16, {order = #NHWC}>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice [[EXPAND1]] [0, 0, 0, 0] [1, 1, 6, 4040] : tensor<1x16x6x4040xf16, {order = #NHWC}> to tensor<1x1x6x4040xf16, {order = #NHWC}>
    // CHECK:       return [[SLICE1]] : tensor<1x1x6x4040xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @DoNotChangeExpandPoolingOpWithSingleChannel
// CHECK-SAME:        [[INPUT0:%arg[0-9]]]: tensor<1x1x48x64xf16, {order = #NHWC}>
func.func @DoNotChangeExpandPoolingOpWithSingleChannel(%input_0: tensor<1x1x48x64xf16, {order = #NHWC}>)
            -> tensor<1x1x6x32xf16, {order = #NHWC}> {
    %expand = IE.Expand(%input_0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x48x64xf16, {order = #NHWC}> -> tensor<1x16x48x64xf16, {order = #NHWC}>
    %pool = IE.AvgPool(%expand) {exclude_pads, kernel_size = [8, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 2]} : tensor<1x16x48x64xf16, {order = #NHWC}> -> tensor<1x16x6x32xf16, {order = #NHWC}>
    %slice = IE.Slice %pool [0, 0, 0, 0] [1, 1, 6, 32] : tensor<1x16x6x32xf16, {order = #NHWC}> to tensor<1x1x6x32xf16, {order = #NHWC}>
    return %slice : tensor<1x1x6x32xf16, {order = #NHWC}>

    // CHECK:       [[EXPAND:%.*]] = IE.Expand([[INPUT0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x48x64xf16, {order = #NHWC}> -> tensor<1x16x48x64xf16, {order = #NHWC}>
    // CHECK:       [[POOL:%.*]] = IE.AvgPool([[EXPAND]]) {exclude_pads, kernel_size = [8, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 2]} : tensor<1x16x48x64xf16, {order = #NHWC}> -> tensor<1x16x6x32xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.*]] = IE.Slice [[POOL]] [0, 0, 0, 0] [1, 1, 6, 32] : tensor<1x16x6x32xf16, {order = #NHWC}> to tensor<1x1x6x32xf16, {order = #NHWC}>
    // CHECK:       return [[SLICE]] : tensor<1x1x6x32xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0031202451855528589>
!qElemType1 = !quant.uniform<u8:f16, 0.0062404903711057178>

// CHECK-LABEL: @PropagateAffineReshapeBeforeEltwise
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x144x576x4xf16, {order = #NHWC}>
func.func @PropagateAffineReshapeBeforeEltwise(%arg0: tensor<1x144x576x4xf16, {order = #NHWC}>) -> tensor<1x32x576x4x!qElemType, {order = #NHWC}> {
    %weights = const.Declare tensor<32x144x1x1x!qElemType, {order = #NHWC}> = dense<1.0> :
        tensor<32x144x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

    %affine_reshape1 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 144, 2304, 1]} :
        tensor<1x144x576x4xf16, {order = #NHWC}> -> tensor<1x144x2304x1xf16, {order = #NHWC}>
    %eltwise = IE.Add(%affine_reshape1, %affine_reshape1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
        tensor<1x144x2304x1xf16, {order = #NHWC}>,
        tensor<1x144x2304x1xf16, {order = #NHWC}> -> tensor<1x144x2304x1x!qElemType1, {order = #NHWC}>
    %quantize_cast = IE.QuantizeCast(%eltwise) {dstElemType = !qElemType} :
        tensor<1x144x2304x1x!qElemType1, {order = #NHWC}> -> tensor<1x144x2304x1x!qElemType, {order = #NHWC}>
    %affine_reshape2 = IE.AffineReshape(%quantize_cast) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 144, 576, 4]} :
        tensor<1x144x2304x1x!qElemType, {order = #NHWC}> -> tensor<1x144x576x4x!qElemType, {order = #NHWC}>
    %conv = IE.Convolution(%affine_reshape2, %weights) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
        tensor<1x144x576x4x!qElemType, {order = #NHWC}>, tensor<32x144x1x1x!qElemType, {order = #NHWC}>
            -> tensor<1x32x576x4x!qElemType, {order = #NHWC}>

    return %conv : tensor<1x32x576x4x!qElemType, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<32x144x1x1x!qElemType, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT]], [[INPUT]])
    // CHECK:       [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[ADD]]) {dstElemType = !qElemType}
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[QUANTIZE_CAST]], [[WEIGHTS]])
    // CHECK:       return [[CONV]] : tensor<1x32x576x4x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0031202451855528589>
!qElemType1 = !quant.uniform<u8:f16, 0.0062404903711057178>

// CHECK-LABEL: @ConvPropagateAffineReshapeAfterEltwise
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x48x512x32xf16, {order = #NHWC}>
func.func @ConvPropagateAffineReshapeAfterEltwise(%input: tensor<1x48x512x32xf16, {order = #NHWC}>)
            -> tensor<1x48x16384x1x!qElemType, {order = #NHWC}> {
    %weights = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x48x3x3xf16>, [#const.Reorder<#NHWC>]

    %conv = IE.Convolution(%input, %weights) {
        dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} :
        tensor<1x48x512x32xf16, {order = #NHWC}>, tensor<48x48x3x3xf16, {order = #NHWC}>
            -> tensor<1x48x512x32xf16, {order = #NHWC}>
    %affine_reshape = IE.AffineReshape(%conv) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 48, 16384, 1]} :
        tensor<1x48x512x32xf16, {order = #NHWC}> -> tensor<1x48x16384x1xf16, {order = #NHWC}>
    %eltwise = IE.Add(%affine_reshape, %affine_reshape) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x48x16384x1xf16, {order = #NHWC}>,
        tensor<1x48x16384x1xf16, {order = #NHWC}> -> tensor<1x48x16384x1x!qElemType1, {order = #NHWC}>
    %quantize_cast = IE.QuantizeCast(%eltwise) {dstElemType = !qElemType} :
        tensor<1x48x16384x1x!qElemType1, {order = #NHWC}> -> tensor<1x48x16384x1x!qElemType, {order = #NHWC}>

    return %quantize_cast : tensor<1x48x16384x1x!qElemType, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]])
    // CHECK:       [[ADD:%.+]] = IE.Add([[CONV]], [[CONV]])
    // CHECK:       [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[ADD]]) {dstElemType = !qElemType}
    // CHECK:       [[AFFINE_RESHAPE:%.+]] =  IE.AffineReshape([[QUANTIZE_CAST]])
    // CHECK-SAME:      tensor<1x48x512x32x!qElemType, {order = #NHWC}> -> tensor<1x48x16384x1x!qElemType, {order = #NHWC}>
    // CHECK:       return [[AFFINE_RESHAPE]] : tensor<1x48x16384x1x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0031202451855528589>
!qElemType1 = !quant.uniform<u8:f16, 0.0062404903711057178>

// CHECK-LABEL: @AddActivationPropagateAffineReshapeAfterEltwise
// CHECK-SAME:        [[INPUT0:%arg[0-9]]]: tensor<1x48x512x32xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x48x512x32xf16, {order = #NHWC}>
func.func @AddActivationPropagateAffineReshapeAfterEltwise(%arg0: tensor<1x48x512x32xf16, {order = #NHWC}>, %arg1: tensor<1x48x512x32xf16, {order = #NHWC}>)
            -> tensor<1x48x16384x1x!qElemType, {order = #NHWC}> {
    %add = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x48x512x32xf16, {order = #NHWC}>, tensor<1x48x512x32xf16, {order = #NHWC}>
            -> tensor<1x48x512x32xf16, {order = #NHWC}>
    %sin = IE.Sin(%add) : tensor<1x48x512x32xf16, {order = #NHWC}> -> tensor<1x48x512x32xf16, {order = #NHWC}>
    %affine_reshape = IE.AffineReshape(%sin) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 48, 16384, 1]} :
        tensor<1x48x512x32xf16, {order = #NHWC}> -> tensor<1x48x16384x1xf16, {order = #NHWC}>
    %eltwise = IE.Add(%affine_reshape, %affine_reshape) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x48x16384x1xf16, {order = #NHWC}>,
        tensor<1x48x16384x1xf16, {order = #NHWC}> -> tensor<1x48x16384x1x!qElemType1, {order = #NHWC}>
    %quantize_cast = IE.QuantizeCast(%eltwise) {dstElemType = !qElemType} :
        tensor<1x48x16384x1x!qElemType1, {order = #NHWC}> -> tensor<1x48x16384x1x!qElemType, {order = #NHWC}>

    return %quantize_cast : tensor<1x48x16384x1x!qElemType, {order = #NHWC}>

    // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT0]], [[INPUT1]])
    // CHECK:       [[SIN:%.+]] = IE.Sin([[ADD]])
    // CHECK:       [[ELTWISE:%.+]] = IE.Add([[SIN]], [[SIN]])
    // CHECK:       [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[ELTWISE]]) {dstElemType = !qElemType}
    // CHECK:       [[AFFINE_RESHAPE:%.+]] =  IE.AffineReshape([[QUANTIZE_CAST]])
    // CHECK-SAME:      tensor<1x48x512x32x!qElemType, {order = #NHWC}> -> tensor<1x48x16384x1x!qElemType, {order = #NHWC}>
    // CHECK:       return [[AFFINE_RESHAPE]] : tensor<1x48x16384x1x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PropagateShapeCastBeforeMultiEltwise
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x256x192xf16, {order = #NHWC}>
func.func @PropagateShapeCastBeforeMultiEltwise(%arg0: tensor<1x16x256x192xf16, {order = #NHWC}>)
            -> tensor<1x48x512x32xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x16x256x192xf16, {order = #NHWC}> = dense<1.100000e+00> : tensor<1x16x256x192xf16>, [#const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x48x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Add(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x192xf16, {order = #NHWC}>, tensor<1x16x256x192xf16, {order = #NHWC}> -> tensor<1x16x256x192xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x192xf16, {order = #NHWC}>, tensor<1x16x256x192xf16, {order = #NHWC}> -> tensor<1x16x256x192xf16, {order = #NHWC}>
    %2 = IE.Add(%1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x192xf16, {order = #NHWC}>, tensor<1x16x256x192xf16, {order = #NHWC}> -> tensor<1x16x256x192xf16, {order = #NHWC}>
    %3 = IE.Add(%2, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x192xf16, {order = #NHWC}>, tensor<1x16x256x192xf16, {order = #NHWC}> -> tensor<1x16x256x192xf16, {order = #NHWC}>
    %4 = IE.Add(%3, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x192xf16, {order = #NHWC}>, tensor<1x16x256x192xf16, {order = #NHWC}> -> tensor<1x16x256x192xf16, {order = #NHWC}>
    %5 = IE.Add(%4, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x192xf16, {order = #NHWC}>, tensor<1x16x256x192xf16, {order = #NHWC}> -> tensor<1x16x256x192xf16, {order = #NHWC}>
    %6 = IE.Add(%5, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x192xf16, {order = #NHWC}>, tensor<1x16x256x192xf16, {order = #NHWC}> -> tensor<1x16x256x192xf16, {order = #NHWC}>
    %7 = IE.Add(%6, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x192xf16, {order = #NHWC}>, tensor<1x16x256x192xf16, {order = #NHWC}> -> tensor<1x16x256x192xf16, {order = #NHWC}>
    %8 = IE.Add(%7, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x192xf16, {order = #NHWC}>, tensor<1x16x256x192xf16, {order = #NHWC}> -> tensor<1x16x256x192xf16, {order = #NHWC}>
    %9 = IE.Add(%8, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x192xf16, {order = #NHWC}>, tensor<1x16x256x192xf16, {order = #NHWC}> -> tensor<1x16x256x192xf16, {order = #NHWC}>
    %10 = IE.Add(%9, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x192xf16, {order = #NHWC}>, tensor<1x16x256x192xf16, {order = #NHWC}> -> tensor<1x16x256x192xf16, {order = #NHWC}>
    %11 = IE.ShapeCast {shape = [1, 48, 512, 32]} inputs(%10 : tensor<1x16x256x192xf16, {order = #NHWC}>) -> tensor<1x48x512x32xf16, {order = #NHWC}>
    %12 = IE.Convolution(%11, %cst_0) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} :
        tensor<1x48x512x32xf16, {order = #NHWC}>, tensor<48x48x3x3xf16, {order = #NHWC}> -> tensor<1x48x512x32xf16, {order = #NHWC}>

    return %12 : tensor<1x48x512x32xf16, {order = #NHWC}>

    // CHECK-DAG:   [[CST_IN:%.+]] = const.Declare tensor<1x48x512x32xf16, {order = #NHWC}>
    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 48, 512, 32]}
    // CHECK-SAME:      inputs([[INPUT]] : tensor<1x16x256x192xf16, {order = #NHWC}>) -> tensor<1x48x512x32xf16, {order = #NHWC}>
    // CHECK:       [[ADD_0:%.+]] = IE.Add([[SHAPE_CAST]], [[CST_IN]])
    // CHECK:       [[ADD_1:%.+]] = IE.Add([[ADD_0]], [[CST_IN]])
    // CHECK:       [[ADD_2:%.+]] = IE.Add([[ADD_1]], [[CST_IN]])
    // CHECK:       [[ADD_3:%.+]] = IE.Add([[ADD_2]], [[CST_IN]])
    // CHECK:       [[ADD_4:%.+]] = IE.Add([[ADD_3]], [[CST_IN]])
    // CHECK:       [[ADD_5:%.+]] = IE.Add([[ADD_4]], [[CST_IN]])
    // CHECK:       [[ADD_6:%.+]] = IE.Add([[ADD_5]], [[CST_IN]])
    // CHECK:       [[ADD_7:%.+]] = IE.Add([[ADD_6]], [[CST_IN]])
    // CHECK:       [[ADD_8:%.+]] = IE.Add([[ADD_7]], [[CST_IN]])
    // CHECK:       [[ADD_9:%.+]] = IE.Add([[ADD_8]], [[CST_IN]])
    // CHECK:       [[ADD_10:%.+]] = IE.Add([[ADD_9]], [[CST_IN]])
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[ADD_10]], [[WEIGHTS]])
}

// -----

// CHECK-LABEL: @SkipSingleExpandAddAndChannelAlignedOutput
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x3x32x32xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x16x32x32xf16>
func.func @SkipSingleExpandAddAndChannelAlignedOutput(%arg0: tensor<1x3x32x32xf16>, %arg1: tensor<1x16x32x32xf16>) -> tensor<1x16x32x32xf16> {
    %0 = IE.Add(%arg1, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x32x32xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
    %1 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x32x32xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
    %3= IE.AvgPool(%2) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.10000000149011612 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>

    return %3 : tensor<1x16x32x32xf16>

    // CHECK:       [[ADD_IN:%.+]] = IE.Add([[INPUT2]], [[INPUT2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x32x32xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       [[EXPAND_IN:%.+]] = IE.Expand([[INPUT1]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       [[ADD:%.+]] = IE.Add([[ADD_IN]], [[EXPAND_IN]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x32x32xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       [[AVG:%.+]] = IE.AvgPool([[ADD]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.10000000149011612 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>

    // CHECK:       return [[AVG]] : tensor<1x16x32x32xf16>
}

// -----

// CHECK-LABEL: @AdjustSingleExpandAddOpsChain
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x3x32x32xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x16x32x32xf16>
func.func @AdjustSingleExpandAddOpsChain(%arg0: tensor<1x3x32x32xf16>, %arg1: tensor<1x16x32x32xf16>) -> tensor<1x16x32x32xf16> {
    %0 = IE.Add(%arg1, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x32x32xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
    %1 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x32x32xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x32x32xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>

    return %3 : tensor<1x16x32x32xf16>

    // CHECK:       [[ADD_IN:%.+]] = IE.Add([[INPUT2]], [[INPUT2]])  {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x32x32xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       [[SLICE_IN:%.+]] = IE.Slice [[ADD_IN]] [0, 0, 0, 0] [1, 3, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x3x32x32xf16>
    // CHECK:       [[SHAPE_CAST1:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 12]} inputs([[SLICE_IN]] : tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16>
    // CHECK:       [[SHAPE_CAST2:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 12]} inputs([[INPUT1]] : tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16>
    // CHECK:       [[ADD_1:%.+]] = IE.Add([[SHAPE_CAST1]], [[SHAPE_CAST2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x16x12xf16>, tensor<1x16x16x12xf16
    // CHECK:       [[ADD_2:%.+]] = IE.Add([[ADD_1]], [[ADD_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x16x12xf16>, tensor<1x16x16x12xf16
    // CHECK:       [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 3, 32, 32]} inputs([[ADD_2]] : tensor<1x16x16x12xf16>) -> tensor<1x3x32x32xf16>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[SHAPE_CAST_OUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>

    // CHECK:       return [[EXPAND]] : tensor<1x16x32x32xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AlignEltwiseShapeWithProducer
// CHECK-SAME:        [[INPUT0:%arg[0-9]]]: tensor<1x576x576xf16>
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x128x576x1xf16, {order = #NHWC}>
func.func @AlignEltwiseShapeWithProducer(%arg0: tensor<1x576x576xf16>, %arg1: tensor<1x128x576x1xf16, {order = #NHWC}>)
            -> tensor<1x16x576x576xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<576x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<576x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 576, 576]} : tensor<1x576x576xf16> -> tensor<1x1x576x576xf16>
    %1 = IE.PermuteCast(%0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x576x576xf16> -> tensor<1x1x576x576xf16, {order = #NHWC}>
    %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x576x576xf16, {order = #NHWC}> -> tensor<1x16x576x576xf16, {order = #NHWC}>

    %3 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 128, 144, 4]} : tensor<1x128x576x1xf16, {order = #NHWC}> -> tensor<1x128x144x4xf16, {order = #NHWC}>
    %4 = IE.Convolution(%3, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x144x4xf16, {order = #NHWC}>, tensor<576x128x1x1xf16, {order = #NHWC}> -> tensor<1x576x144x4xf16, {order = #NHWC}>
    %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 576, 576, 1]} : tensor<1x576x144x4xf16, {order = #NHWC}> -> tensor<1x576x576x1xf16, {order = #NHWC}>
    %6 = IE.PermuteCast(%5) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>} : tensor<1x576x576x1xf16, {order = #NHWC}> -> tensor<576x576x1x1xf16>
    %7 = IE.AffineReshape(%6) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 576, 576]} : tensor<576x576x1x1xf16> -> tensor<1x1x576x576xf16>
    %8 = IE.PermuteCast(%7) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x576x576xf16> -> tensor<1x1x576x576xf16, {order = #NHWC}>
    %9 = IE.Expand(%8) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x576x576xf16, {order = #NHWC}> -> tensor<1x16x576x576xf16, {order = #NHWC}>

    %10 = IE.Add(%2, %9) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x576x576xf16, {order = #NHWC}>, tensor<1x16x576x576xf16, {order = #NHWC}> -> tensor<1x16x576x576xf16, {order = #NHWC}>

    return %10 : tensor<1x16x576x576xf16, {order = #NHWC}>

    // CHECK-DAG:    [[CST:%.+]] = const.Declare tensor<576x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<576x128x1x1xf16>, [#const.Reorder<#NHWC>]

    // CHECK:        [[RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT0]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 576, 576]} : tensor<1x576x576xf16> -> tensor<1x1x576x576xf16>
    // CHECK:        [[PERMUTE_0:%.+]] = IE.PermuteCast([[RESHAPE_0]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x576x576xf16> -> tensor<1x1x576x576xf16, {order = #NHWC}>
    // CHECK:        [[ADD_IN_0:%.+]] = IE.ShapeCast {shape = [1, 576, 144, 4]} inputs([[PERMUTE_0]] : tensor<1x1x576x576xf16, {order = #NHWC}>) -> tensor<1x576x144x4xf16, {order = #NHWC}>

    // CHECK:        [[RESHAPE_1:%.+]] = IE.AffineReshape([[INPUT1]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 128, 144, 4]} : tensor<1x128x576x1xf16, {order = #NHWC}> -> tensor<1x128x144x4xf16, {order = #NHWC}>
    // CHECK:        [[CONV:%.+]] = IE.Convolution([[RESHAPE_1]], [[CST]]) {
    // CHECK-SAME:              dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    // CHECK-SAME:          } : tensor<1x128x144x4xf16, {order = #NHWC}>, tensor<576x128x1x1xf16, {order = #NHWC}> -> tensor<1x576x144x4xf16, {order = #NHWC}>
    // CHECK:        [[RESHAPE_2:%.+]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 576, 576, 1]} : tensor<1x576x144x4xf16, {order = #NHWC}> -> tensor<1x576x576x1xf16, {order = #NHWC}>
    // CHECK:        [[PERMUTE_1:%.+]] = IE.PermuteCast([[RESHAPE_2]]) {dst_order = #NCHW, mem_perm = #map} : tensor<1x576x576x1xf16, {order = #NHWC}> -> tensor<576x576x1x1xf16>
    // CHECK:        [[RESHAPE_3:%.+]] = IE.AffineReshape([[PERMUTE_1]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 576, 576]} : tensor<576x576x1x1xf16> -> tensor<1x1x576x576xf16>
    // CHECK:        [[PERMUTE_2:%.+]] = IE.PermuteCast([[RESHAPE_3]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x576x576xf16> -> tensor<1x1x576x576xf16, {order = #NHWC}>
    // CHECK:        [[ADD_IN_1:%.+]] = IE.ShapeCast {shape = [1, 576, 144, 4]} inputs([[PERMUTE_2]] : tensor<1x1x576x576xf16, {order = #NHWC}>) -> tensor<1x576x144x4xf16, {order = #NHWC}>

    // CHECK:        [[ADD:%.+]] = IE.Add([[ADD_IN_0]], [[ADD_IN_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x576x144x4xf16, {order = #NHWC}>, tensor<1x576x144x4xf16, {order = #NHWC}> -> tensor<1x576x144x4xf16, {order = #NHWC}>
    // CHECK:        [[ADD_OUT:%.+]] = IE.ShapeCast {shape = [1, 1, 576, 576]} inputs([[ADD]] : tensor<1x576x144x4xf16, {order = #NHWC}>) -> tensor<1x1x576x576xf16, {order = #NHWC}>
    // CHECK:        [[EXPAND:%.+]] = IE.Expand([[ADD_OUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x576x576xf16, {order = #NHWC}> -> tensor<1x16x576x576xf16, {order = #NHWC}>

    // CHECK:       return [[EXPAND]] : tensor<1x16x576x576xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.5>

// CHECK-LABEL: @AdjustInputShapeForPermuteQuantizeDueToBigW
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x1x245760xf16>
func.func @AdjustInputShapeForPermuteQuantizeDueToBigW(%arg0: tensor<1x1x1x245760xf16>) -> tensor<1x1x1x245760x!qElemType, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {dstElemType = !qElemType, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1x1x245760xf16> -> tensor<1x1x1x245760x!qElemType, {order = #NHWC}>
    return %0 : tensor<1x1x1x245760x!qElemType, {order = #NHWC}>

    // CHECK:       [[CAST:%.+]] = IE.ShapeCast {shape = [1, 1, 30, 8192]} inputs([[INPUT]] : tensor<1x1x1x245760xf16>) -> tensor<1x1x30x8192xf16>
    // CHECK:       [[PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize([[CAST]]) {dstElemType = !qElemType, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1x30x8192xf16> -> tensor<1x1x30x8192x!qElemType, {order = #NHWC}>
    // CHECK:       [[CAST_OUTPUT:%.+]] = IE.ShapeCast {shape = [1, 1, 1, 245760]} inputs([[PERMUTEQUANTIZE]] : tensor<1x1x30x8192x!qElemType, {order = #NHWC}>) -> tensor<1x1x1x245760x!qElemType, {order = #NHWC}>
    // CHECK:       return [[CAST_OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AdjustInputShapeForPermuteQuantizeDueToBigH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x8x245760x16xf16>
func.func @AdjustInputShapeForPermuteQuantizeDueToBigH(%arg0: tensor<1x8x245760x16xf16>) -> tensor<1x8x245760x16xf16, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x245760x16xf16> -> tensor<1x8x245760x16xf16, {order = #NHWC}>
    return %0 : tensor<1x8x245760x16xf16, {order = #NHWC}>

    // CHECK:       [[CAST:%.+]] = IE.ShapeCast {shape = [1, 8, 8192, 480]} inputs([[INPUT]] : tensor<1x8x245760x16xf16>) -> tensor<1x8x8192x480xf16>
    // CHECK:       [[PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize([[CAST]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x8192x480xf16> -> tensor<1x8x8192x480xf16, {order = #NHWC}>
    // CHECK:       [[CAST_OUTPUT:%.+]] = IE.ShapeCast {shape = [1, 8, 245760, 16]} inputs([[PERMUTEQUANTIZE]] : tensor<1x8x8192x480xf16, {order = #NHWC}>) -> tensor<1x8x245760x16xf16, {order = #NHWC}>
    // CHECK:       return [[CAST_OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotAdjustInputShapeForPermuteQuantizeDueToNonePad
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x8x245760x16xf16>
func.func @NotAdjustInputShapeForPermuteQuantizeDueToNonePad(%arg0: tensor<1x8x245760x16xf16>) -> tensor<1x16x245760x16xf16, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x8x245760x16xf16> -> tensor<1x16x245760x16xf16, {order = #NHWC}>
    return %0 : tensor<1x16x245760x16xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.ShapeCast
    // CHECK:       [[PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x8x245760x16xf16> -> tensor<1x16x245760x16xf16, {order = #NHWC}>
    // CHECK:       return [[PERMUTEQUANTIZE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotAdjustInputShapeForPermuteQuantizeDueToBigW
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x8x245760x2048xf16>
func.func @NotAdjustInputShapeForPermuteQuantizeDueToBigW(%arg0: tensor<1x8x245760x2048xf16>) -> tensor<1x8x245760x2048xf16, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x245760x2048xf16> -> tensor<1x8x245760x2048xf16, {order = #NHWC}>
    return %0 : tensor<1x8x245760x2048xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.ShapeCast
    // CHECK:       [[PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x245760x2048xf16> -> tensor<1x8x245760x2048xf16, {order = #NHWC}>
    // CHECK:       return [[PERMUTEQUANTIZE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotAdjustInputShapeForPermuteQuantizeDueToBigH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x8x2048x245760xf16>
func.func @NotAdjustInputShapeForPermuteQuantizeDueToBigH(%arg0: tensor<1x8x2048x245760xf16>) -> tensor<1x8x2048x245760xf16, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x2048x245760xf16> -> tensor<1x8x2048x245760xf16, {order = #NHWC}>
    return %0 : tensor<1x8x2048x245760xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.ShapeCast
    // CHECK:       [[PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x2048x245760xf16> -> tensor<1x8x2048x245760xf16, {order = #NHWC}>
    // CHECK:       return [[PERMUTEQUANTIZE]]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AdjustAvgPoolingWithBatchSizeNot1
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1024x77x1x1xf16, {order = #NHWC}>
func.func @AdjustAvgPoolingWithBatchSizeNot1(%arg0: tensor<1024x77x1x1xf16, {order = #NHWC}>) -> tensor<1024x77x1x1xf16, {order = #NHWC}> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]}
        : tensor<1024x77x1x1xf16, {order = #NHWC}> -> tensor<1024x80x1x1xf16, {order = #NHWC}>
    %1 = IE.AvgPool(%0) {
        exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
        post_op = #IE.LeakyRelu<negative_slope = 0.10000000149011612 : f64>,
        rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
        : tensor<1024x80x1x1xf16, {order = #NHWC}> -> tensor<1024x80x1x1xf16, {order = #NHWC}>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1024, 77, 1, 1] : tensor<1024x80x1x1xf16, {order = #NHWC}> to tensor<1024x77x1x1xf16, {order = #NHWC}>
    return %2 : tensor<1024x77x1x1xf16, {order = #NHWC}>

    // CHECK:   [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 16, 77, 64]} inputs([[INPUT]] : tensor<1024x77x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x16x77x64xf16, {order = #NHWC}>

    // CHECK:   [[POOLING:%.+]] = IE.AvgPool([[SHAPECAST_IN]])
    // CHECK-SAME:      : tensor<1x16x77x64xf16, {order = #NHWC}> -> tensor<1x16x77x64xf16, {order = #NHWC}>

    // CHECK:   [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [1024, 77, 1, 1]} inputs([[POOLING]] : tensor<1x16x77x64xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1024x77x1x1xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND:%.+]] = IE.Expand([[SHAPECAST_OUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]}
    // CHECK-SAME:      : tensor<1024x77x1x1xf16, {order = #NHWC}> -> tensor<1024x80x1x1xf16, {order = #NHWC}>
    // CHECK:   [[SLICE:%.+]] = IE.Slice [[EXPAND]] [0, 0, 0, 0] [1024, 77, 1, 1]
    // CHECK-SAME:      : tensor<1024x80x1x1xf16, {order = #NHWC}> to tensor<1024x77x1x1xf16, {order = #NHWC}>

    // CHECK:       return [[SLICE]] : tensor<1024x77x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AdjustInputShapeWithMinimalDimChange
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x4x1600x2560xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x4x1600x2560xf16, {order = #NHWC}>
func.func @AdjustInputShapeWithMinimalDimChange(%arg0: tensor<1x4x1600x2560xf16, {order = #NHWC}>, %arg1: tensor<1x4x1600x2560xf16, {order = #NHWC}>) -> tensor<1x16x1600x2560xf16, {order = #NHWC}> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]} : tensor<1x4x1600x2560xf16, {order = #NHWC}> -> tensor<1x16x1600x2560xf16, {order = #NHWC}>
    %1 = IE.Expand(%arg1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]} : tensor<1x4x1600x2560xf16, {order = #NHWC}> -> tensor<1x16x1600x2560xf16, {order = #NHWC}>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x1600x2560xf16, {order = #NHWC}>, tensor<1x16x1600x2560xf16, {order = #NHWC}> -> tensor<1x16x1600x2560xf16, {order = #NHWC}>
    return %2 : tensor<1x16x1600x2560xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.Expand
    // CHECK-DAG:   [[CAST1:%.+]] = IE.ShapeCast {shape = [1, 16, 1600, 640]} inputs([[INPUT1]] : tensor<1x4x1600x2560xf16, {order = #NHWC}>) -> tensor<1x16x1600x640xf16, {order = #NHWC}>
    // CHECK-DAG:   [[CAST2:%.+]] = IE.ShapeCast {shape = [1, 16, 1600, 640]} inputs([[INPUT2]] : tensor<1x4x1600x2560xf16, {order = #NHWC}>) -> tensor<1x16x1600x640xf16, {order = #NHWC}>

    // CHECK:       [[ADD:%.+]] = IE.Add([[CAST1]], [[CAST2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x1600x640xf16, {order = #NHWC}>, tensor<1x16x1600x640xf16, {order = #NHWC}> -> tensor<1x16x1600x640xf16, {order = #NHWC}>
    // CHECK:       [[CAST_OUTPUT:%.+]] = IE.ShapeCast {shape = [1, 4, 1600, 2560]} inputs([[ADD]] : tensor<1x16x1600x640xf16, {order = #NHWC}>) -> tensor<1x4x1600x2560xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND_OUTPUT:%.+]] = IE.Expand([[CAST_OUTPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]} : tensor<1x4x1600x2560xf16, {order = #NHWC}> -> tensor<1x16x1600x2560xf16, {order = #NHWC}>
    // CHECK:       return [[EXPAND_OUTPUT]]
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128}>
!qElemType1 = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.567:128, 0.567:128, 0.567:128}>

// CHECK-LABEL: @NotExpandGroupConvToShapeCastForAxisQuantizedType
// CHECK-SAME:        [[INPUT:%arg0]]: tensor<1x3x32x32x!qElemType>
func.func @NotExpandGroupConvToShapeCastForAxisQuantizedType(%arg0: tensor<1x3x32x32x!qElemType>) -> tensor<1x6x32x32x!qElemType1> {
    %filters = const.Declare tensor<6x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 6 : i64>]
    %bias = const.Declare tensor<1x6x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 6 : i64>]
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x3x32x32x!qElemType> -> tensor<1x6x32x32x!qElemType1>
    %1 = IE.GroupConvolution(%0, %filters, %bias) {dilations = [1, 1], groups = 6 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x6x32x32x!qElemType1>, tensor<6x1x1x1xf32>, tensor<1x6x1x1xf32> -> tensor<1x6x32x32x!qElemType1>
    return %1 : tensor<1x6x32x32x!qElemType1>

    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x6x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 6 : i64>]
    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<6x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 6 : i64>]

    // CHECK-NOT:   IE.ShapeCast
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x3x32x32x!qElemType> -> tensor<1x6x32x32x!qElemType1>
    // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[EXPAND]], [[FILTER]], [[BIAS]])
    // CHECK-SAME:      dilations = [1, 1], groups = 6 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    // CHECK-SAME:          -> tensor<1x6x32x32x!qElemType1>

    // CHECK:       return [[GROUP_CONV]]
}
