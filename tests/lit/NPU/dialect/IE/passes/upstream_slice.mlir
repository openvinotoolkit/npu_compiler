//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --upstream-slice --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @UpstreamSlice
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x64x4xf16>
// CHECK-SAME: [[ARG_1:%[^:]+]]: tensor<1x16x17x2xf16>)
!qElemType = !quant.uniform<u8:f16, 0.0078431377223893706:128>
!qElemType1 = !quant.uniform<i8:f16, 0.0078431377223893706>
!qElemType2 = !quant.uniform<u8:f16, 0.0078431372549019607:128>
!qElemType3 = !quant.uniform<u8:f16, 0.015686274509803921>
!qElemType4 = !quant.uniform<u8:f16, 0.031372549019607843>
func.func @UpstreamSlice(%arg0: tensor<1x16x64x4xf16>, %arg1: tensor<1x16x17x2xf16>) -> tensor<1x32x16x4xf16> {
    %cst = const.Declare tensor<1x32x1x1xf16> = dense<1.0> : tensor<1x32x1x1xf32>, [#const.CastElemType<f16>]
    %cst_0 = const.Declare tensor<1x64x1x1xf16> = dense<1.0> : tensor<1x64x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<32x16x3x1x!qElemType> = dense<1.0> : tensor<32x16x3x1xf32>, [#const.CastElemType<f16>, #const.CastElemType<si8>, #const.CastElemType<!qElemType1>, #const.CastElemType<si8>, #const.CastElemType<i32>, #const.Add<1.280000e+02 : f64>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %cst_2 = const.Declare tensor<64x16x3x1x!qElemType> = dense<1.0> : tensor<64x16x3x1xf32>, [#const.CastElemType<f16>, #const.CastElemType<si8>, #const.CastElemType<!qElemType1>, #const.CastElemType<si8>, #const.CastElemType<i32>, #const.Add<1.280000e+02 : f64>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType2} : tensor<1x16x64x4xf16> -> tensor<1x16x64x4x!qElemType2>
    %1 = IE.Convolution(%0, %cst_1, %cst) {dilations = [1, 1], pads_begin = [1, 0], pads_end = [1, 0], strides = [1, 1]} : tensor<1x16x64x4x!qElemType2>, tensor<32x16x3x1x!qElemType>, tensor<1x32x1x1xf16> -> tensor<1x32x64x4x!qElemType3>
    %2 = IE.Slice %1 [0, 0, 47, 0] [1, 32, 17, 4] : tensor<1x32x64x4x!qElemType3> to tensor<1x32x17x4x!qElemType3>
    %3 = IE.Quantize(%arg1) {dstElemType = !qElemType2} : tensor<1x16x17x2xf16> -> tensor<1x16x17x2x!qElemType2>
    %4 = IE.Convolution(%3, %cst_2, %cst_0) {dilations = [1, 1], pads_begin = [1, 0], pads_end = [1, 0], strides = [1, 1]} : tensor<1x16x17x2x!qElemType2>, tensor<64x16x3x1x!qElemType>, tensor<1x64x1x1xf16> -> tensor<1x64x17x2x!qElemType3>
    %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x64x17x2x!qElemType3> -> tensor<1x64x17x2xf16>
    %6 = IE.Reshape(%5) {shape_value = [1, 32, 17, 4]} : tensor<1x64x17x2xf16> -> tensor<1x32x17x4xf16>
    %7 = IE.Quantize(%6) {dstElemType = !qElemType3} : tensor<1x32x17x4xf16> -> tensor<1x32x17x4x!qElemType3>
    %8 = IE.Add(%2, %7) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x17x4x!qElemType3>, tensor<1x32x17x4x!qElemType3> -> tensor<1x32x17x4x!qElemType4>
    %9 = IE.Dequantize(%8) {dstElemType = f16} : tensor<1x32x17x4x!qElemType4> -> tensor<1x32x17x4xf16>
    %10 = IE.Slice %9 [0, 0, 1, 0] [1, 32, 16, 4] : tensor<1x32x17x4xf16> to tensor<1x32x16x4xf16>
    return %10 : tensor<1x32x16x4xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x32x1x1xf16>
    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<1x64x1x1xf16>
    // CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<32x16x3x1x!qElemType>
    // CHECK-DAG:   [[CST2:%.+]] = const.Declare tensor<64x16x3x1x!qElemType>

    // CHECK:       [[QUANT0:%.+]] = IE.Quantize([[ARG_0]])
    // CHECK:       [[CONV0:%.+]] = IE.Convolution([[QUANT0]], [[CST1]], [[CST]])
    // CHECK:       [[QUANT1:%.+]] = IE.Quantize([[ARG_1]])
    // CHECK:       [[CONV1:%.+]] = IE.Convolution([[QUANT1]], [[CST2]], [[CST0]])
    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[CONV1]])
    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[DEQUANT]])
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[RESHAPE]]
    // CHECK-SAME:    [0, 0, 1, 0] [1, 32, 16, 4] : tensor<1x32x17x4xf16> to tensor<1x32x16x4xf16>
    // CHECK:       [[QUANT:%.+]] = IE.Quantize([[SLICE1]])
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[CONV0]]
    // CHECK-SAME:    [0, 0, 48, 0] [1, 32, 16, 4] : tensor<1x32x64x4x!qElemType3> to tensor<1x32x16x4x!qElemType3>
    // CHECK:       [[ADD:%.+]] = IE.Add([[SLICE0]], [[QUANT]])
    // CHECK-SAME:    {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x16x4x!qElemType3>, tensor<1x32x16x4x!qElemType3> -> tensor<1x32x16x4x!qElemType4>
    // CHECK:       [[DEQUANT1:%.+]] = IE.Dequantize([[ADD]])
    // CHECK-SAME:   {dstElemType = f16} : tensor<1x32x16x4x!qElemType4> -> tensor<1x32x16x4xf16>
    // CHECK:       return  [[DEQUANT1]] : tensor<1x32x16x4xf16>
}

// -----

// CHECK-LABEL: @UpstreamStridedSlice
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x64x4xf16>
// CHECK-SAME: [[ARG_1:%[^:]+]]: tensor<1x16x17x2xf16>)
!qElemType = !quant.uniform<u8:f16, 0.0078431377223893706:128>
!qElemType1 = !quant.uniform<i8:f16, 0.0078431377223893706>
!qElemType2 = !quant.uniform<u8:f16, 0.0078431372549019607:128>
!qElemType3 = !quant.uniform<u8:f16, 0.015686274509803921>
!qElemType4 = !quant.uniform<u8:f16, 0.031372549019607843>
func.func @UpstreamStridedSlice(%arg0: tensor<1x16x64x4xf16>, %arg1: tensor<1x16x17x2xf16>) -> tensor<1x32x16x4xf16> {
    %cst = const.Declare tensor<1x32x1x1xf16> = dense<1.0> : tensor<1x32x1x1xf32>, [#const.CastElemType<f16>]
    %cst_0 = const.Declare tensor<1x64x1x1xf16> = dense<1.0> : tensor<1x64x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<32x16x3x1x!qElemType> = dense<1.0> : tensor<32x16x3x1xf32>, [#const.CastElemType<f16>, #const.CastElemType<si8>, #const.CastElemType<!qElemType1>, #const.CastElemType<si8>, #const.CastElemType<i32>, #const.Add<1.280000e+02 : f64>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %cst_2 = const.Declare tensor<64x16x3x1x!qElemType> = dense<1.0> : tensor<64x16x3x1xf32>, [#const.CastElemType<f16>, #const.CastElemType<si8>, #const.CastElemType<!qElemType1>, #const.CastElemType<si8>, #const.CastElemType<i32>, #const.Add<1.280000e+02 : f64>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType2} : tensor<1x16x64x4xf16> -> tensor<1x16x64x4x!qElemType2>
    %1 = IE.Convolution(%0, %cst_1, %cst) {dilations = [1, 1], pads_begin = [1, 0], pads_end = [1, 0], strides = [1, 1]} : tensor<1x16x64x4x!qElemType2>, tensor<32x16x3x1x!qElemType>, tensor<1x32x1x1xf16> -> tensor<1x32x64x4x!qElemType3>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x32x64x4x!qElemType3> -> tensor<1x32x64x4xf16>
    %3 = IE.StridedSlice(%2) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 47, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 32, 64, 4], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 1]} : tensor<1x32x64x4xf16> -> tensor<1x32x17x4xf16>
    %4 = IE.Quantize(%3) {dstElemType = !qElemType3} : tensor<1x32x17x4xf16> -> tensor<1x32x17x4x!qElemType3>
    %5 = IE.Quantize(%arg1) {dstElemType = !qElemType2} : tensor<1x16x17x2xf16> -> tensor<1x16x17x2x!qElemType2>
    %6 = IE.Convolution(%5, %cst_2, %cst_0) {dilations = [1, 1], pads_begin = [1, 0], pads_end = [1, 0], strides = [1, 1]} : tensor<1x16x17x2x!qElemType2>, tensor<64x16x3x1x!qElemType>, tensor<1x64x1x1xf16> -> tensor<1x64x17x2x!qElemType3>
    %7 = IE.Dequantize(%6) {dstElemType = f16} : tensor<1x64x17x2x!qElemType3> -> tensor<1x64x17x2xf16>
    %8 = IE.Reshape(%7) {shape_value = [1, 32, 17, 4]} : tensor<1x64x17x2xf16> -> tensor<1x32x17x4xf16>
    %9 = IE.Quantize(%8) {dstElemType = !qElemType3} : tensor<1x32x17x4xf16> -> tensor<1x32x17x4x!qElemType3>
    %10 = IE.Add(%4, %9) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x17x4x!qElemType3>, tensor<1x32x17x4x!qElemType3> -> tensor<1x32x17x4x!qElemType4>
    %11 = IE.Dequantize(%10) {dstElemType = f16} : tensor<1x32x17x4x!qElemType4> -> tensor<1x32x17x4xf16>
    %12 = IE.StridedSlice(%11) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 32, 17, 4], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 1]} : tensor<1x32x17x4xf16> -> tensor<1x32x16x4xf16>
    return %12 : tensor<1x32x16x4xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x32x1x1xf16>
    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<1x64x1x1xf16>
    // CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<32x16x3x1x!qElemType>
    // CHECK-DAG:   [[CST2:%.+]] = const.Declare tensor<64x16x3x1x!qElemType>

    // CHECK:       [[QUANT0:%.+]] = IE.Quantize([[ARG_0]])
    // CHECK:       [[CONV0:%.+]] = IE.Convolution([[QUANT0]], [[CST1]], [[CST]])
    // CHECK:       [[DEQUANT0:%.+]] = IE.Dequantize([[CONV0]])
    // CHECK:       [[SS1:%.+]] = IE.StridedSlice([[DEQUANT0]])
    // CHECK-SAME:   {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 47, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 32, 64, 4], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 1]}
    // CHECK:       [[QUANT0:%.+]] = IE.Quantize([[SS1]])
    // CHECK:       [[QUANT1:%.+]] = IE.Quantize([[ARG_1]])
    // CHECK:       [[CONV1:%.+]] = IE.Convolution([[QUANT1]], [[CST2]], [[CST0]])
    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[CONV1]])
    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[DEQUANT]])
    // CHECK:       [[QUANT2:%.+]] = IE.Quantize([[RESHAPE]])
    // CHECK:       [[ADD:%.+]] = IE.Add([[QUANT0]], [[QUANT2]])
    // CHECK:       [[DEQUANT1:%.+]] = IE.Dequantize([[ADD]])
    // CHECK:       [[SS2:%.+]] = IE.StridedSlice([[DEQUANT1]])
    // CHECK-SAME:   {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 32, 17, 4], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 1]}
    // CHECK:       return  [[SS2]] : tensor<1x32x16x4xf16>
}

// -----

// CHECK-LABEL: @UpstreamSliceFakeQuantizeDifferentAxis
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<4x4x5x5xf16>)
func.func @UpstreamSliceFakeQuantizeDifferentAxis(%arg0: tensor<4x4x5x5xf16>) -> tensor<4x4x3x3xf16> {
    %cst = const.Declare tensor<4x4x1x1xf16> = dense<[[[[0.227172852]], [[0.158569336]], [[0.136474609]], [[0.226318359]]], [[[0.198608398]], [[0.166625977]], [[0.126342773]], [[0.18359375]]], [[[0.154541016]], [[0.25390625]], [[0.42578125]], [[0.231689453]]], [[[0.130615234]], [[0.235107422]], [[0.141723633]], [[0.139892578]]]]> : tensor<4x4x1x1xf32>, [#const.CastElemType<f16>]
    %cst_0 = const.Declare tensor<4x4x1x1xf16> = dense<[[[[-0.145141602]], [[-0.189819336]], [[-0.168823242]], [[-0.149536133]]], [[[-0.128173828]], [[-0.174438477]], [[-0.159912109]], [[-0.163330078]]], [[[-0.850585938]], [[-0.151855469]], [[-0.0876464843]], [[-0.23059082]]], [[[-0.225708008]], [[-0.207641602]], [[-0.211303711]], [[-0.249389648]]]]> : tensor<4x4x1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 255 : i64} : tensor<4x4x5x5xf16>, tensor<4x4x1x1xf16>, tensor<4x4x1x1xf16>, tensor<4x4x1x1xf16>, tensor<4x4x1x1xf16> -> tensor<4x4x5x5xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [4, 4, 3, 3] : tensor<4x4x5x5xf16> to tensor<4x4x3x3xf16>

    return %1 : tensor<4x4x3x3xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<4x4x1x1xf16> =
    // CHECK-SAME{LITERAL}:      dense<[[[[0.227172852]], [[0.158569336]], [[0.136474609]], [[0.226318359]]], [[[0.198608398]], [[0.166625977]], [[0.126342773]], [[0.18359375]]], [[[0.154541016]], [[0.25390625]], [[0.42578125]], [[0.231689453]]], [[[0.130615234]], [[0.235107422]], [[0.141723633]], [[0.139892578]]]]> : tensor<4x4x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<4x4x1x1xf16> =
    // CHECK-SAME{LITERAL}:      dense<[[[[-0.145141602]], [[-0.189819336]], [[-0.168823242]], [[-0.149536133]]], [[[-0.128173828]], [[-0.174438477]], [[-0.159912109]], [[-0.163330078]]], [[[-0.850585938]], [[-0.151855469]], [[-0.0876464843]], [[-0.23059082]]], [[[-0.225708008]], [[-0.207641602]], [[-0.211303711]], [[-0.249389648]]]]> : tensor<4x4x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG_0]] [0, 0, 0, 0] [4, 4, 3, 3] : tensor<4x4x5x5xf16> to tensor<4x4x3x3xf16>
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[SLICE]], [[CST0]], [[CST]], [[CST0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 255 : i64} : tensor<4x4x3x3xf16>, tensor<4x4x1x1xf16>, tensor<4x4x1x1xf16>, tensor<4x4x1x1xf16>, tensor<4x4x1x1xf16> -> tensor<4x4x3x3xf16>

    // CHECK: return [[FQ]] : tensor<4x4x3x3xf16>
}

// -----

// CHECK-LABEL: @DoNotUpstreamSliceFakeQuantizeSameAxis
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x5x5xf16>)
func.func @DoNotUpstreamSliceFakeQuantizeSameAxis(%arg0: tensor<1x16x5x5xf16>) -> tensor<1x8x5x5xf16> {
    %cst = const.Declare tensor<1x16x1x1xf16> = dense<[[[[0.227172852]], [[0.158569336]], [[0.136474609]], [[0.226318359]], [[0.198608398]], [[0.166625977]], [[0.126342773]], [[0.18359375]], [[0.154541016]], [[0.25390625]], [[0.42578125]], [[0.231689453]], [[0.130615234]], [[0.235107422]], [[0.141723633]], [[0.139892578]]]]> : tensor<1x16x1x1xf32>, [#const.CastElemType<f16>]
    %cst_0 = const.Declare tensor<1x16x1x1xf16> = dense<[[[[-0.145141602]], [[-0.189819336]], [[-0.168823242]], [[-0.149536133]], [[-0.128173828]], [[-0.174438477]], [[-0.159912109]], [[-0.163330078]], [[-0.850585938]], [[-0.151855469]], [[-0.0876464843]], [[-0.23059082]], [[-0.225708008]], [[-0.207641602]], [[-0.211303711]], [[-0.249389648]]]]> : tensor<1x16x1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 255 : i64} : tensor<1x16x5x5xf16>, tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x5x5xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 8, 5, 5] : tensor<1x16x5x5xf16> to tensor<1x8x5x5xf16>

    return %1 : tensor<1x8x5x5xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x16x1x1xf16> =
    // CHECK-SAME{LITERAL}:      dense<[[[[0.227172852]], [[0.158569336]], [[0.136474609]], [[0.226318359]], [[0.198608398]], [[0.166625977]], [[0.126342773]], [[0.18359375]], [[0.154541016]], [[0.25390625]], [[0.42578125]], [[0.231689453]], [[0.130615234]], [[0.235107422]], [[0.141723633]], [[0.139892578]]]]> : tensor<1x16x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<1x16x1x1xf16> =
    // CHECK-SAME{LITERAL}:      dense<[[[[-0.145141602]], [[-0.189819336]], [[-0.168823242]], [[-0.149536133]], [[-0.128173828]], [[-0.174438477]], [[-0.159912109]], [[-0.163330078]], [[-0.850585938]], [[-0.151855469]], [[-0.0876464843]], [[-0.23059082]], [[-0.225708008]], [[-0.207641602]], [[-0.211303711]], [[-0.249389648]]]]> : tensor<1x16x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[ARG_0]], [[CST0]], [[CST]], [[CST0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 255 : i64} : tensor<1x16x5x5xf16>, tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x5x5xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[FQ]] [0, 0, 0, 0] [1, 8, 5, 5] : tensor<1x16x5x5xf16> to tensor<1x8x5x5xf16>

    // CHECK:       return [[SLICE]] : tensor<1x8x5x5xf16>
}

// -----

#C = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @DoNotUpstreamDynamicSlices
func.func @DoNotUpstreamDynamicSlices(
        %DATA: tensor<32xf16>,
        %SHAPE: tensor<1xsi32>
) -> tensor<?xf16, {bounds = #const.OpaqueI64Elements<[32]> : tensor<1xsi64>, order = #C}> {
    // CHECK:   [[DATA:%.+]]: tensor<32xf16>, [[SHAPE:%.+]]: tensor<1xsi32>
    %GELU = IE.Gelu(%DATA) : tensor<32xf16> -> tensor<32xf16>
    // CHECK:   [[GELU:%.+]] = IE.Gelu([[DATA]]) : tensor<32xf16> -> tensor<32xf16>

    %SLICE = IE.StridedSlice(%GELU, %SHAPE) {
        begin_mask = [],
        begins_attr = [0],
        ellipsis_mask = [],
        end_mask = [],
        new_axis_mask = [],
        operandSegmentSizes = array<i32: 1, 0, 1, 0>,
        shrink_axis_mask = [],
        strides_attr = [1]
    } : tensor<32xf16>, tensor<1xsi32> -> tensor<?xf16, {bounds = #const.OpaqueI64Elements<[32]> : tensor<1xsi64>, order = #C}>
    // CHECK:   [[SLICE:%.+]] = IE.StridedSlice([[GELU]], [[SHAPE]])

    return %SLICE : tensor<?xf16, {bounds = #const.OpaqueI64Elements<[32]> : tensor<1xsi64>, order = #C}>
    // CHECK:   return [[SLICE]] : tensor<?xf16, {bounds = #const.OpaqueI64Elements<[32]> : tensor<1xsi64>, order = #C}>
}

// -----

// CHECK-LABEL: @SliceUpstreamWithAffineReshape
!qElemType = !quant.uniform<u8:f16, 0.022586843079211664:128>
!qElemType1 = !quant.uniform<u8:f16, 0.021841353061152438:128>
!qElemType2 = !quant.uniform<u8:f16, 0.045173686158423328:128>
!qElemType3 = !quant.uniform<u8:f16, 0.043682706122304876:128>

// CHECK-SAME:      [[ARG:%arg[0-9]+]]: tensor<1x1024x192x96xf16>
func.func @SliceUpstreamWithAffineReshape(%arg0: tensor<1x1024x192x96xf16>) -> (tensor<1x1024x64x96x!qElemType>, tensor<1x1024x64x96x!qElemType1>) {
    %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1024x192x96xf16>, tensor<1x1024x192x96xf16> -> tensor<1x1024x192x96x!qElemType2>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1, 2, 3], [4]], shape_value = [1024, 64, 3, 1, 96]} : tensor<1x1024x192x96x!qElemType2> -> tensor<1024x64x3x1x96x!qElemType2>
    %2 = IE.QuantizeCast(%1) {dstElemType = !qElemType} : tensor<1024x64x3x1x96x!qElemType2> -> tensor<1024x64x3x1x96x!qElemType>
    %3 = IE.Slice %2 [0, 0, 0, 0, 0] [1024, 64, 1, 1, 96] : tensor<1024x64x3x1x96x!qElemType> to tensor<1024x64x1x1x96x!qElemType>
    %4 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1024x192x96xf16>, tensor<1x1024x192x96xf16> -> tensor<1x1024x192x96x!qElemType3>
    %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [0], [1, 2, 3], [4]], shape_value = [1024, 64, 3, 1, 96]} : tensor<1x1024x192x96x!qElemType3> -> tensor<1024x64x3x1x96x!qElemType3>
    %6 = IE.QuantizeCast(%5) {dstElemType = !qElemType1} : tensor<1024x64x3x1x96x!qElemType3> -> tensor<1024x64x3x1x96x!qElemType1>
    %7 = IE.Slice %6 [0, 0, 1, 0, 0] [1024, 64, 1, 1, 96] : tensor<1024x64x3x1x96x!qElemType1> to tensor<1024x64x1x1x96x!qElemType1>
    %8 = IE.AffineReshape(%3) {dim_mapping = [[0, 1], [2], [2], [2], [3]], shape_value = [1, 1024, 64, 96]} : tensor<1024x64x1x1x96x!qElemType> -> tensor<1x1024x64x96x!qElemType>
    %9 = IE.AffineReshape(%7) {dim_mapping = [[0, 1], [2], [2], [2], [3]], shape_value = [1, 1024, 64, 96]} : tensor<1024x64x1x1x96x!qElemType1> -> tensor<1x1024x64x96x!qElemType1>
    return %8, %9 : tensor<1x1024x64x96x!qElemType>, tensor<1x1024x64x96x!qElemType1>

    // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape([[ARG]])
    // CHECK-SAME{LITERAL}:         {dim_mapping = [[0], [0], [1, 2, 3], [4]], shape_value = [1024, 64, 3, 1, 96]} : tensor<1x1024x192x96xf16> -> tensor<1024x64x3x1x96xf16>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[RESHAPE_0]] [0, 0, 0, 0, 0] [1024, 64, 1, 1, 96] : tensor<1024x64x3x1x96xf16> to tensor<1024x64x1x1x96xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[SLICE_0]])
    // CHECK-SAME{LITERAL}:         {dim_mapping = [[0, 1], [2], [2], [2], [3]], shape_value = [1, 1024, 64, 96]} : tensor<1024x64x1x1x96xf16> -> tensor<1x1024x64x96xf16>
    // CHECK:       [[ADD_0:%.+]] = IE.Add([[RESHAPE_1]], [[RESHAPE_1]])
    // CHECK:          : tensor<1x1024x64x96xf16>, tensor<1x1024x64x96xf16> -> tensor<1x1024x64x96x!qElemType2>
    // CHECK:       [[CAST_0:%.+]] = IE.QuantizeCast([[ADD_0]]) {dstElemType = !qElemType} : tensor<1x1024x64x96x!qElemType2> -> tensor<1x1024x64x96x!qElemType>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.AffineReshape([[ARG]])
    // CHECK-SAME{LITERAL}:         {dim_mapping = [[0], [0], [1, 2, 3], [4]], shape_value = [1024, 64, 3, 1, 96]} : tensor<1x1024x192x96xf16> -> tensor<1024x64x3x1x96xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[RESHAPE_2]] [0, 0, 1, 0, 0] [1024, 64, 1, 1, 96] : tensor<1024x64x3x1x96xf16> to tensor<1024x64x1x1x96xf16>
    // CHECK:       [[RESHAPE_3:%.+]] = IE.AffineReshape([[SLICE_1]])
    // CHECK-SAME{LITERAL}:         {dim_mapping = [[0, 1], [2], [2], [2], [3]], shape_value = [1, 1024, 64, 96]} : tensor<1024x64x1x1x96xf16> -> tensor<1x1024x64x96xf16>
    // CHECK:       [[ADD_1:%.+]] = IE.Add([[RESHAPE_3]], [[RESHAPE_3]])
    // CHECK:          : tensor<1x1024x64x96xf16>, tensor<1x1024x64x96xf16> -> tensor<1x1024x64x96x!qElemType3>
    // CHECK:       [[CAST_1:%.+]] = IE.QuantizeCast([[ADD_1]]) {dstElemType = !qElemType1} : tensor<1x1024x64x96x!qElemType3> -> tensor<1x1024x64x96x!qElemType1>
    // CHECK:       return [[CAST_0]], [[CAST_1]] : tensor<1x1024x64x96x!qElemType>, tensor<1x1024x64x96x!qElemType1>
}

// -----

// CHECK-LABEL: @SliceUpstreamWithAffineReshapeWithExtraReshape
!qElemType = !quant.uniform<u8:f16, 0.044086302962957645:128>
!qElemType1 = !quant.uniform<u8:f16, 0.05178938846962125:128>
!qElemType2 = !quant.uniform<u8:f16, 0.08817260592591529:128>
!qElemType3 = !quant.uniform<u8:f16, 0.1035787769392425:128>

// CHECK-SAME:      [[ARG:%arg[0-9]+]]: tensor<1x1024x192x192xf16>
func.func @SliceUpstreamWithAffineReshapeWithExtraReshape(%arg0: tensor<1x1024x192x192xf16>) -> (tensor<1024x8x8x192x!qElemType>, tensor<1024x64x2x96x!qElemType1>) {
    %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1024x192x192xf16>, tensor<1x1024x192x192xf16> -> tensor<1x1024x192x192x!qElemType2>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1, 2], [3, 4]], shape_value = [1024, 64, 3, 2, 96]} : tensor<1x1024x192x192x!qElemType2> -> tensor<1024x64x3x2x96x!qElemType2>
    %2 = IE.QuantizeCast(%1) {dstElemType = !qElemType} : tensor<1024x64x3x2x96x!qElemType2> -> tensor<1024x64x3x2x96x!qElemType>
    %3 = IE.Slice %2 [0, 0, 0, 0, 0] [1024, 64, 1, 2, 96] : tensor<1024x64x3x2x96x!qElemType> to tensor<1024x64x1x2x96x!qElemType>
    %4 = IE.AffineReshape(%3) {dim_mapping = [[0], [1, 2], [3], [3], [3]], shape_value = [1024, 8, 8, 192]} : tensor<1024x64x1x2x96x!qElemType> -> tensor<1024x8x8x192x!qElemType>
    %5 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1024x192x192xf16>, tensor<1x1024x192x192xf16> -> tensor<1x1024x192x192x!qElemType3>
    %6 = IE.AffineReshape(%5) {dim_mapping = [[0], [0], [1, 2], [3, 4]], shape_value = [1024, 64, 3, 2, 96]} : tensor<1x1024x192x192x!qElemType3> -> tensor<1024x64x3x2x96x!qElemType3>
    %7 = IE.QuantizeCast(%6) {dstElemType = !qElemType1} : tensor<1024x64x3x2x96x!qElemType3> -> tensor<1024x64x3x2x96x!qElemType1>
    %8 = IE.Slice %7 [0, 0, 1, 0, 0] [1024, 64, 1, 2, 96] : tensor<1024x64x3x2x96x!qElemType1> to tensor<1024x64x1x2x96x!qElemType1>
    %9 = IE.AffineReshape(%8) {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1024, 64, 2, 96]} : tensor<1024x64x1x2x96x!qElemType1> -> tensor<1024x64x2x96x!qElemType1>
    return %4, %9 : tensor<1024x8x8x192x!qElemType>, tensor<1024x64x2x96x!qElemType1>

    // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape([[ARG]])
    // CHECK-SAME{LITERAL}:         {dim_mapping = [[0], [0], [1, 2], [3, 4]], shape_value = [1024, 64, 3, 2, 96]} : tensor<1x1024x192x192xf16> -> tensor<1024x64x3x2x96xf16>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[RESHAPE_0]] [0, 0, 0, 0, 0] [1024, 64, 1, 2, 96] : tensor<1024x64x3x2x96xf16> to tensor<1024x64x1x2x96xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[SLICE_0]])
    // CHECK-SAME{LITERAL}:         {dim_mapping = [[0, 1], [2], [2], [3], [3]], shape_value = [1, 1024, 64, 192]} : tensor<1024x64x1x2x96xf16> -> tensor<1x1024x64x192xf16>
    // CHECK:       [[ADD_0:%.+]] = IE.Add([[RESHAPE_1]], [[RESHAPE_1]])
    // CHECK:          : tensor<1x1024x64x192xf16>, tensor<1x1024x64x192xf16> -> tensor<1x1024x64x192x!qElemType2>
    // CHECK:       [[CAST_0:%.+]] = IE.QuantizeCast([[ADD_0]]) {dstElemType = !qElemType} : tensor<1x1024x64x192x!qElemType2> -> tensor<1x1024x64x192x!qElemType>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.AffineReshape([[CAST_0]])
    // CHECK-SAME{LITERAL}:         {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [1024, 8, 8, 192]} : tensor<1x1024x64x192x!qElemType> -> tensor<1024x8x8x192x!qElemType>
    // CHECK:       [[RESHAPE_3:%.+]] = IE.AffineReshape([[ARG]])
    // CHECK-SAME{LITERAL}:         {dim_mapping = [[0], [0], [1, 2], [3, 4]], shape_value = [1024, 64, 3, 2, 96]} : tensor<1x1024x192x192xf16> -> tensor<1024x64x3x2x96xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[RESHAPE_3]] [0, 0, 1, 0, 0] [1024, 64, 1, 2, 96] : tensor<1024x64x3x2x96xf16> to tensor<1024x64x1x2x96xf16>
    // CHECK:       [[RESHAPE_4:%.+]] = IE.AffineReshape([[SLICE_1]])
    // CHECK-SAME{LITERAL}:         {dim_mapping = [[0, 1], [2], [2], [3], [3]], shape_value = [1, 1024, 64, 192]} : tensor<1024x64x1x2x96xf16> -> tensor<1x1024x64x192xf16>
    // CHECK:       [[ADD_1:%.+]] = IE.Add([[RESHAPE_4]], [[RESHAPE_4]])
    // CHECK:          : tensor<1x1024x64x192xf16>, tensor<1x1024x64x192xf16> -> tensor<1x1024x64x192x!qElemType3>
    // CHECK:       [[CAST_1:%.+]] = IE.QuantizeCast([[ADD_1]]) {dstElemType = !qElemType1} : tensor<1x1024x64x192x!qElemType3> -> tensor<1x1024x64x192x!qElemType1>
    // CHECK:       [[RESHAPE_5:%.+]] = IE.AffineReshape([[CAST_1]])
    // CHECK-SAME{LITERAL}:         {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1024, 64, 2, 96]} : tensor<1x1024x64x192x!qElemType1> -> tensor<1024x64x2x96x!qElemType1>
    // CHECK:       return [[RESHAPE_2]], [[RESHAPE_5]] : tensor<1024x8x8x192x!qElemType>, tensor<1024x64x2x96x!qElemType1>
}

// -----

// CHECK-LABEL: @SliceUpstreamWithAffineReshapeWithoutReshape
!qElemType = !quant.uniform<u8:f16, 0.08817260592591529:128>
!qElemType1 = !quant.uniform<u8:f16, 0.1035787769392425:128>

// CHECK-SAME:      [[ARG:%arg[0-9]+]]: tensor<1024x64x3x2x96xf16>
func.func @SliceUpstreamWithAffineReshapeWithoutReshape(%arg0: tensor<1024x64x3x2x96xf16>) -> (tensor<1024x64x1x2x96x!qElemType>, tensor<1024x64x1x2x96x!qElemType1>) {
    %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1024x64x3x2x96xf16>, tensor<1024x64x3x2x96xf16> -> tensor<1024x64x3x2x96x!qElemType>
    %1 = IE.Slice %0 [0, 0, 0, 0, 0] [1024, 64, 1, 2, 96] : tensor<1024x64x3x2x96x!qElemType> to tensor<1024x64x1x2x96x!qElemType>
    %2 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1024x64x3x2x96xf16>, tensor<1024x64x3x2x96xf16> -> tensor<1024x64x3x2x96x!qElemType1>
    %3 = IE.Slice %2 [0, 0, 1, 0, 0] [1024, 64, 1, 2, 96] : tensor<1024x64x3x2x96x!qElemType1> to tensor<1024x64x1x2x96x!qElemType1>
    return %1, %3 : tensor<1024x64x1x2x96x!qElemType>, tensor<1024x64x1x2x96x!qElemType1>

    // CHECK: [[SLICE_1:%.+]] = IE.Slice [[ARG]] [0, 0, 0, 0, 0] [1024, 64, 1, 2, 96]
    // CHECK: [[ADD_1:%.+]] = IE.Add([[SLICE_1]], [[SLICE_1]])
    // CHECK: [[SLICE_2:%.+]] = IE.Slice [[ARG]] [0, 0, 1, 0, 0] [1024, 64, 1, 2, 96]
    // CHECK: [[ADD_2:%.+]] = IE.Add([[SLICE_2]], [[SLICE_2]])
    // CHECK: return [[ADD_1]], [[ADD_2]]
}

// -----

// CHECK-LABEL: @SliceUpstreamWithAffineReshapeWithQuantCastOrReshape
!qElemType1 = !quant.uniform<u8:f16, 0.05178938846962125:128>
!qElemType2 = !quant.uniform<u8:f16, 0.08817260592591529:128>
!qElemType3 = !quant.uniform<u8:f16, 0.1035787769392425:128>

// CHECK-SAME:      [[ARG:%arg[0-9]+]]: tensor<1x1024x192x192xf16>
func.func @SliceUpstreamWithAffineReshapeWithQuantCastOrReshape(%arg0: tensor<1x1024x192x192xf16>) -> (tensor<1024x8x8x192x!qElemType2>, tensor<1x1024x64x192x!qElemType1>) {
    %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1024x192x192xf16>, tensor<1x1024x192x192xf16> -> tensor<1x1024x192x192x!qElemType2>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1, 2], [3, 4]], shape_value = [1024, 64, 3, 2, 96]} : tensor<1x1024x192x192x!qElemType2> -> tensor<1024x64x3x2x96x!qElemType2>
    %2 = IE.Slice %1 [0, 0, 0, 0, 0] [1024, 64, 1, 2, 96] : tensor<1024x64x3x2x96x!qElemType2> to tensor<1024x64x1x2x96x!qElemType2>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1, 2], [3], [3], [3]], shape_value = [1024, 8, 8, 192]} : tensor<1024x64x1x2x96x!qElemType2> -> tensor<1024x8x8x192x!qElemType2>
    %4 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1024x192x192xf16>, tensor<1x1024x192x192xf16> -> tensor<1x1024x192x192x!qElemType3>
    %5 = IE.QuantizeCast(%4) {dstElemType = !qElemType1} : tensor<1x1024x192x192x!qElemType3> -> tensor<1x1024x192x192x!qElemType1>
    %6 = IE.Slice %5 [0, 0, 128, 0] [1, 1024, 64, 192] : tensor<1x1024x192x192x!qElemType1> to tensor<1x1024x64x192x!qElemType1>
    return %3, %6 : tensor<1024x8x8x192x!qElemType2>, tensor<1x1024x64x192x!qElemType1>

    // CHECK: [[RESHAPE_1:%.+]] = IE.AffineReshape([[ARG]])
    // CHECK-SAME{LITERAL}:         {dim_mapping = [[0], [0], [1, 2], [3, 4]], shape_value = [1024, 64, 3, 2, 96]} : tensor<1x1024x192x192xf16> -> tensor<1024x64x3x2x96xf16>
    // CHECK: [[SLICE:%.+]] = IE.Slice [[RESHAPE_1]] [0, 0, 0, 0, 0] [1024, 64, 1, 2, 96] : tensor<1024x64x3x2x96xf16> to tensor<1024x64x1x2x96xf16>
    // CHECK: [[RESHAPE_2:%.+]] = IE.AffineReshape([[SLICE]])
    // CHECK-SAME{LITERAL}:         {dim_mapping = [[0, 1], [2], [2], [3], [3]], shape_value = [1, 1024, 64, 192]} : tensor<1024x64x1x2x96xf16> -> tensor<1x1024x64x192xf16>
    // CHECK: [[ADD_1:%.+]] = IE.Add([[RESHAPE_2]], [[RESHAPE_2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1024x64x192xf16>, tensor<1x1024x64x192xf16> -> tensor<1x1024x64x192x!qElemType>
    // CHECK: [[RESHAPE_3:%.+]] = IE.AffineReshape([[ADD_1]])
    // CHECK-SAME{LITERAL}:         {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [1024, 8, 8, 192]} : tensor<1x1024x64x192x!qElemType> -> tensor<1024x8x8x192x!qElemType>
    // CHECK: [[SLICE_1:%.+]] = IE.Slice [[ARG]] [0, 0, 128, 0] [1, 1024, 64, 192] : tensor<1x1024x192x192xf16> to tensor<1x1024x64x192xf16>
    // CHECK: [[ADD_2:%.+]] = IE.Add([[SLICE_1]], [[SLICE_1]])  {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1024x64x192xf16>, tensor<1x1024x64x192xf16> -> tensor<1x1024x64x192x!qElemType2>
    // CHECK: [[CAST:%.+]] = IE.QuantizeCast([[ADD_2]]) {dstElemType = !qElemType1} : tensor<1x1024x64x192x!qElemType2> -> tensor<1x1024x64x192x!qElemType1>
    // CHECK: return [[RESHAPE_3]], [[CAST]]
}
