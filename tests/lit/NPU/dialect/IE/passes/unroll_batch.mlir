//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-batch %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @UnrollFullyConnectedBatch
func.func @UnrollFullyConnectedBatch(%arg0: tensor<2x16xf32>) -> tensor<2x64xf32> {
    %cst = const.Declare tensor<64x16xf16> = dense<1.0> : tensor<64x16xf32>, [#const.CastElemType<f16>]
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<2x16xf32> -> tensor<2x16xf16>
    %1 = IE.FullyConnected(%0, %cst) : tensor<2x16xf16>, tensor<64x16xf16> -> tensor<2x64xf16>
    %2 = IE.Convert(%1) {dstElemType = f32} : tensor<2x64xf16> -> tensor<2x64xf32>

    return %2 : tensor<2x64xf32>

    // CHECK-DAG:       %[[WEIGHTS:.*]] = const.Declare tensor<64x16xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x16xf32>, [#const.CastElemType<f16>]
    // CHECK:       %[[INPUT:.*]] = IE.Convert(%arg0) {dstElemType = f16} : tensor<2x16xf32> -> tensor<2x16xf16>
    // CHECK:       %[[INPUT_SLICE_1:.*]] = IE.Slice %[[INPUT]] [0, 0] [1, 16] : tensor<2x16xf16> to tensor<1x16xf16>
    // CHECK:       %[[FC_1:.*]] = IE.FullyConnected(%[[INPUT_SLICE_1]], %[[WEIGHTS]]) : tensor<1x16xf16>, tensor<64x16xf16> -> tensor<1x64xf16>
    // CHECK:       %[[INPUT_SLICE_2:.*]] = IE.Slice %[[INPUT]] [1, 0] [1, 16] : tensor<2x16xf16> to tensor<1x16xf16>
    // CHECK:       %[[FC_2:.*]] = IE.FullyConnected(%[[INPUT_SLICE_2]], %[[WEIGHTS]]) : tensor<1x16xf16>, tensor<64x16xf16> -> tensor<1x64xf16>
    // CHECK:       %[[FC_CONCAT:.*]] = IE.Concat(%[[FC_1]], %[[FC_2]])
    // CHECK-SAME:      {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x64xf16>, tensor<1x64xf16> -> tensor<2x64xf16>
    // CHECK:       %[[OUT:.*]] = IE.Convert(%[[FC_CONCAT]]) {dstElemType = f32} : tensor<2x64xf16> -> tensor<2x64xf32>
    // CHECK:       return %[[OUT]] : tensor<2x64xf32>
}

!qElemType = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @UnrollEltwiseAddBatch
func.func @UnrollEltwiseAddBatch(%arg0: tensor<2x128x40x8xf16>) -> tensor<2x128x40x8xf16> {
    %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<2x128x40x8xf16>, tensor<2x128x40x8xf16> -> tensor<2x128x40x8x!qElemType>
    %1 = IE.Add(%0, %0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<2x128x40x8x!qElemType>, tensor<2x128x40x8x!qElemType> -> tensor<2x128x40x8xf16>
    return %1 : tensor<2x128x40x8xf16>

    // CHECK: [[SLICE0_ARG0:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 128, 40, 8] : tensor<2x128x40x8xf16> to tensor<1x128x40x8xf16>
    // CHECK: [[AND0:%.+]] = IE.Add([[SLICE0_ARG0]], [[SLICE0_ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
    // CHECK-SAME: tensor<1x128x40x8xf16>, tensor<1x128x40x8xf16> -> tensor<1x128x40x8x!qElemType>
    // CHECK: [[SLICE1_ARG0:%.+]] = IE.Slice %arg0 [1, 0, 0, 0] [1, 128, 40, 8] : tensor<2x128x40x8xf16> to tensor<1x128x40x8xf16>
    // CHECK: [[AND1:%.+]] = IE.Add([[SLICE1_ARG0]], [[SLICE1_ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
    // CHECK-SAME: tensor<1x128x40x8xf16>, tensor<1x128x40x8xf16> -> tensor<1x128x40x8x!qElemType>
    // CHECK: [[CONCAT0:%.+]] = IE.Concat([[AND0]], [[AND1]]) {per_axis = #IE.Concat<axis = 0 : i64>} :
    // CHECK-SAME: tensor<1x128x40x8x!qElemType>, tensor<1x128x40x8x!qElemType> -> tensor<2x128x40x8x!qElemType>
    // CHECK: [[SLICE2:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 0, 0] [1, 128, 40, 8] : tensor<2x128x40x8x!qElemType> to tensor<1x128x40x8x!qElemType>
    // CHECK: [[AND2:%.+]] = IE.Add([[SLICE2]], [[SLICE2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
    // CHECK-SAME: tensor<1x128x40x8x!qElemType>, tensor<1x128x40x8x!qElemType> -> tensor<1x128x40x8xf16>
    // CHECK: [[SLICE3:%.+]] = IE.Slice [[CONCAT0]] [1, 0, 0, 0] [1, 128, 40, 8] : tensor<2x128x40x8x!qElemType> to tensor<1x128x40x8x!qElemType>
    // CHECK: [[AND3:%.+]] = IE.Add([[SLICE3]], [[SLICE3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
    // CHECK-SAME: tensor<1x128x40x8x!qElemType>, tensor<1x128x40x8x!qElemType> -> tensor<1x128x40x8xf16>
    // CHECK: [[CONCAT1:%.+]] = IE.Concat([[AND2]], [[AND3]]) {per_axis = #IE.Concat<axis = 0 : i64>} :
    // CHECK-SAME: tensor<1x128x40x8xf16>, tensor<1x128x40x8xf16> -> tensor<2x128x40x8xf16>
    // CHECK: return [[CONCAT1]] : tensor<2x128x40x8xf16>
}

// -----

func.func @UnrollSigmoidBatch(%arg0: tensor<3x9x16x1xf16>) -> tensor<3x9x16x1xf16> {
    %0 = IE.Sigmoid(%arg0) : tensor<3x9x16x1xf16> -> tensor<3x9x16x1xf16>
    return %0 : tensor<3x9x16x1xf16>
    // CHECK: [[SLICE0_ARG0:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 9, 16, 1] : tensor<3x9x16x1xf16> to tensor<1x9x16x1xf16>
    // CHECK: [[SIGMOID0:%.+]] = IE.Sigmoid([[SLICE0_ARG0]]) :
    // CHECK-SAME: tensor<1x9x16x1xf16> -> tensor<1x9x16x1xf16>
    // CHECK: [[SLICE1_ARG0:%.+]] = IE.Slice %arg0 [1, 0, 0, 0] [1, 9, 16, 1] : tensor<3x9x16x1xf16> to tensor<1x9x16x1xf16>
    // CHECK: [[SIGMOID1:%.+]] = IE.Sigmoid([[SLICE1_ARG0]]) :
    // CHECK-SAME: tensor<1x9x16x1xf16> -> tensor<1x9x16x1xf16>
    // CHECK: [[SLICE2_ARG0:%.+]] = IE.Slice %arg0 [2, 0, 0, 0] [1, 9, 16, 1] : tensor<3x9x16x1xf16> to tensor<1x9x16x1xf16>
    // CHECK: [[SIGMOID2:%.+]] = IE.Sigmoid([[SLICE2_ARG0]]) :
    // CHECK-SAME: tensor<1x9x16x1xf16> -> tensor<1x9x16x1xf16>
    // CHECK: [[CONCAT0:%.+]] = IE.Concat([[SIGMOID0]], [[SIGMOID1]], [[SIGMOID2]]) {per_axis = #IE.Concat<axis = 0 : i64>} :
    // CHECK-SAME: tensor<1x9x16x1xf16>, tensor<1x9x16x1xf16>, tensor<1x9x16x1xf16> -> tensor<3x9x16x1xf16>
    // CHECK: return [[CONCAT0]] : tensor<3x9x16x1xf16>
}

// -----

func.func @UnrollExpBatch(%arg0: tensor<3x9x16x1xf16>) -> tensor<3x9x16x1xf16> {
    %0 = IE.Exp(%arg0) : tensor<3x9x16x1xf16> -> tensor<3x9x16x1xf16>
    return %0 : tensor<3x9x16x1xf16>
    // CHECK: [[SLICE0_ARG0:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 9, 16, 1] : tensor<3x9x16x1xf16> to tensor<1x9x16x1xf16>
    // CHECK: [[EXP0:%.+]] = IE.Exp([[SLICE0_ARG0]]) :
    // CHECK-SAME: tensor<1x9x16x1xf16> -> tensor<1x9x16x1xf16>
    // CHECK: [[SLICE1_ARG0:%.+]] = IE.Slice %arg0 [1, 0, 0, 0] [1, 9, 16, 1] : tensor<3x9x16x1xf16> to tensor<1x9x16x1xf16>
    // CHECK: [[EXP1:%.+]] = IE.Exp([[SLICE1_ARG0]]) :
    // CHECK-SAME: tensor<1x9x16x1xf16> -> tensor<1x9x16x1xf16>
    // CHECK: [[SLICE2_ARG0:%.+]] = IE.Slice %arg0 [2, 0, 0, 0] [1, 9, 16, 1] : tensor<3x9x16x1xf16> to tensor<1x9x16x1xf16>
    // CHECK: [[EXP2:%.+]] = IE.Exp([[SLICE2_ARG0]]) :
    // CHECK-SAME: tensor<1x9x16x1xf16> -> tensor<1x9x16x1xf16>
    // CHECK: [[CONCAT0:%.+]] = IE.Concat([[EXP0]], [[EXP1]], [[EXP2]]) {per_axis = #IE.Concat<axis = 0 : i64>} :
    // CHECK-SAME: tensor<1x9x16x1xf16>, tensor<1x9x16x1xf16>, tensor<1x9x16x1xf16> -> tensor<3x9x16x1xf16>
    // CHECK: return [[CONCAT0]] : tensor<3x9x16x1xf16>
}

// -----

func.func @UnrollGroupConvolutionBatch(%arg0: tensor<3x9x16x1xf16>, %arg1: tensor<9x1x1x1xf16>) -> tensor<3x9x16x1xf16> {
    %0 = IE.GroupConvolution(%arg0, %arg1) {dilations = [1, 1], groups = 9 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<3x9x16x1xf16>, tensor<9x1x1x1xf16> -> tensor<3x9x16x1xf16>
    return %0 : tensor<3x9x16x1xf16>
    // CHECK: [[SLICE0_ARG0:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 9, 16, 1] : tensor<3x9x16x1xf16> to tensor<1x9x16x1xf16>
    // CHECK: [[GROUPCONVOLUTION0:%.+]] = IE.GroupConvolution([[SLICE0_ARG0]], %arg1) {dilations = [1, 1], groups = 9 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME: tensor<1x9x16x1xf16>, tensor<9x1x1x1xf16> -> tensor<1x9x16x1xf16>
    // CHECK: [[SLICE1_ARG0:%.+]] = IE.Slice %arg0 [1, 0, 0, 0] [1, 9, 16, 1] : tensor<3x9x16x1xf16> to tensor<1x9x16x1xf16>
    // CHECK: [[GROUPCONVOLUTION1:%.+]] = IE.GroupConvolution([[SLICE1_ARG0]], %arg1) {dilations = [1, 1], groups = 9 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME: tensor<1x9x16x1xf16>, tensor<9x1x1x1xf16> -> tensor<1x9x16x1xf16>
    // CHECK: [[SLICE2_ARG0:%.+]] = IE.Slice %arg0 [2, 0, 0, 0] [1, 9, 16, 1] : tensor<3x9x16x1xf16> to tensor<1x9x16x1xf16>
    // CHECK: [[GROUPCONVOLUTION2:%.+]] = IE.GroupConvolution([[SLICE2_ARG0]], %arg1) {dilations = [1, 1], groups = 9 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME: tensor<1x9x16x1xf16>, tensor<9x1x1x1xf16> -> tensor<1x9x16x1xf16>
    // CHECK: [[CONCAT0:%.+]] = IE.Concat([[GROUPCONVOLUTION0]], [[GROUPCONVOLUTION1]], [[GROUPCONVOLUTION2]]) {per_axis = #IE.Concat<axis = 0 : i64>} :
    // CHECK-SAME: tensor<1x9x16x1xf16>, tensor<1x9x16x1xf16>, tensor<1x9x16x1xf16> -> tensor<3x9x16x1xf16>
    // CHECK: return [[CONCAT0]] : tensor<3x9x16x1xf16>
}

// -----

func.func @UnrollMaxPoolingBatch(%arg0: tensor<2x128x32x64xf16>) -> tensor<2x128x32x64xf16> {
    %MAX_POOL = IE.MaxPool(%arg0) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<2x128x32x64xf16> -> tensor<2x128x32x64xf16>

    return %MAX_POOL : tensor<2x128x32x64xf16>

    // CHECK:   [[SLICE_0:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 128, 32, 64] :
    // CHECK-SAME:      tensor<2x128x32x64xf16> to tensor<1x128x32x64xf16>

    // CHECK:   [[MAX_POOL_0:%.+]] = IE.MaxPool([[SLICE_0]]) {
    // CHECK-SAME:      kernel_size = [3, 3],
    // CHECK-SAME:      pads_begin = [1, 1],
    // CHECK-SAME:      pads_end = [1, 1],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x128x32x64xf16> -> tensor<1x128x32x64xf16>

    // CHECK:   [[SLICE_1:%.+]] = IE.Slice %arg0 [1, 0, 0, 0] [1, 128, 32, 64] :
    // CHECK-SAME:      tensor<2x128x32x64xf16> to tensor<1x128x32x64xf16>

    // CHECK:   [[MAX_POOL_1:%.+]] = IE.MaxPool([[SLICE_1]]) {
    // CHECK-SAME:      kernel_size = [3, 3],
    // CHECK-SAME:      pads_begin = [1, 1],
    // CHECK-SAME:      pads_end = [1, 1],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x128x32x64xf16> -> tensor<1x128x32x64xf16>

    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[MAX_POOL_0]], [[MAX_POOL_1]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 0 : i64>
    // CHECK-SAME:  } : tensor<1x128x32x64xf16>, tensor<1x128x32x64xf16> -> tensor<2x128x32x64xf16>

    // CHECK:   return [[CONCAT]] : tensor<2x128x32x64xf16>
}

// -----

// CHECK-LABEL: @UnrollInterpolateBatch
// CHECK-SAME:      [[INPUT:%.+]]: tensor<2x3x10x10xf32>
func.func @UnrollInterpolateBatch(%arg0: tensor<2x3x10x10xf32>) -> tensor<2x3x20x15xf32> {
    %0 = const.Declare tensor<2xsi64> = dense<[20, 15]> : tensor<2xsi64>
    %1 = const.Declare tensor<2xf32>  = dense<[2.000000e+00, 1.500000e+00]> : tensor<2xf32>
    %2 = const.Declare tensor<2xsi64> = dense<[2, 3]> : tensor<2xsi64>
    %3 = IE.Interpolate(%arg0, %0, %1, %2) {
            attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <NEAREST>, nearest_mode = <ROUND_PREFER_FLOOR>,
            pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, operandSegmentSizes = array<i32: 1, 1, 1, 1>
        } : tensor<2x3x10x10xf32>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<2x3x20x15xf32>

    return %3 : tensor<2x3x20x15xf32>

    // CHECK-DAG:   [[SHAPE_CST:%.+]] = const.Declare tensor<2xsi64> = dense<[20, 15]> : tensor<2xsi64>
    // CHECK-DAG:   [[SCALE_CST:%.+]] = const.Declare tensor<2xf32> = dense<[2.000000e+00, 1.500000e+00]> : tensor<2xf32>
    // CHECK-DAG:   [[AXES_CST:%.+]] = const.Declare tensor<2xsi64> = dense<[2, 3]> : tensor<2xsi64>

    // CHECK:       [[INPUT_0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 3, 10, 10] : tensor<2x3x10x10xf32> to tensor<1x3x10x10xf32>
    // CHECK:       [[INTERPOLATE_0:%.+]] = IE.Interpolate([[INPUT_0]], [[SHAPE_CST]], [[SCALE_CST]], [[AXES_CST]])
    // CHECK-SAME:      {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, operandSegmentSizes = array<i32: 1, 1, 1, 1>}
    // CHECK-SAME:      : tensor<1x3x10x10xf32>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x3x20x15xf32>
    // CHECK:       [[INPUT_1:%.+]] = IE.Slice [[INPUT]] [1, 0, 0, 0] [1, 3, 10, 10] : tensor<2x3x10x10xf32> to tensor<1x3x10x10xf32>
    // CHECK:       [[INTERPOLATE_1:%.+]] = IE.Interpolate([[INPUT_1]], [[SHAPE_CST]], [[SCALE_CST]], [[AXES_CST]])
    // CHECK-SAME:      {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, operandSegmentSizes = array<i32: 1, 1, 1, 1>}
    // CHECK-SAME:      : tensor<1x3x10x10xf32>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x3x20x15xf32>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INTERPOLATE_0]], [[INTERPOLATE_1]])
    // CHECK-SAME:      {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x3x20x15xf32>, tensor<1x3x20x15xf32> -> tensor<2x3x20x15xf32>

    // CHECK:       return [[CONCAT]] : tensor<2x3x20x15xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UnrollMemPermuteBatch
// CHECK-SAME:      [[INPUT:%.+]]: tensor<2x2x289x289xf16, {order = #NHWC}>
func.func @UnrollMemPermuteBatch(%arg0: tensor<2x2x289x289xf16, {order = #NHWC}>) -> tensor<4x1x289x289xf16> {
    %0 = IE.ShapeCast {shape = [4, 1, 289, 289]} inputs(%arg0 : tensor<2x2x289x289xf16, {order = #NHWC}>) -> tensor<4x1x289x289xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<4x1x289x289xf16, {order = #NHWC}> -> tensor<4x1x289x289xf16>

    return %1 : tensor<4x1x289x289xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = IE.ShapeCast {shape = [4, 1, 289, 289]} inputs([[INPUT]] : tensor<2x2x289x289xf16, {order = #NHWC}>) -> tensor<4x1x289x289xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_0:%.+]] = IE.Slice [[RESHAPE_IN]] [0, 0, 0, 0] [1, 1, 289, 289] : tensor<4x1x289x289xf16, {order = #NHWC}> to tensor<1x1x289x289xf16, {order = #NHWC}>
    // CHECK:       [[PERMUTE_0:%.+]] = IE.MemPermute([[INPUT_0]])
    // CHECK-SAME{LITERAL}:     {dst_order = #NCHW, mem_perm = #NWHC}
    // CHECK-SAME:     tensor<1x1x289x289xf16, {order = #NHWC}> -> tensor<1x1x289x289xf16>

    // CHECK:       [[INPUT_1:%.+]] = IE.Slice [[RESHAPE_IN]] [1, 0, 0, 0] [1, 1, 289, 289] : tensor<4x1x289x289xf16, {order = #NHWC}> to tensor<1x1x289x289xf16, {order = #NHWC}>
    // CHECK:       [[PERMUTE_1:%.+]] = IE.MemPermute([[INPUT_1]])
    // CHECK-SAME{LITERAL}:     {dst_order = #NCHW, mem_perm = #NWHC}
    // CHECK-SAME:     tensor<1x1x289x289xf16, {order = #NHWC}> -> tensor<1x1x289x289xf16>

    // CHECK:       [[INPUT_2:%.+]] = IE.Slice [[RESHAPE_IN]] [2, 0, 0, 0] [1, 1, 289, 289] : tensor<4x1x289x289xf16, {order = #NHWC}> to tensor<1x1x289x289xf16, {order = #NHWC}>
    // CHECK:       [[PERMUTE_2:%.+]] = IE.MemPermute([[INPUT_2]])
    // CHECK-SAME{LITERAL}:     {dst_order = #NCHW, mem_perm = #NWHC}
    // CHECK-SAME:     tensor<1x1x289x289xf16, {order = #NHWC}> -> tensor<1x1x289x289xf16>

    // CHECK:       [[INPUT_3:%.+]] = IE.Slice [[RESHAPE_IN]] [3, 0, 0, 0] [1, 1, 289, 289] : tensor<4x1x289x289xf16, {order = #NHWC}> to tensor<1x1x289x289xf16, {order = #NHWC}>
    // CHECK:       [[PERMUTE_3:%.+]] = IE.MemPermute([[INPUT_3]])
    // CHECK-SAME{LITERAL}:     {dst_order = #NCHW, mem_perm = #NWHC}
    // CHECK-SAME:     tensor<1x1x289x289xf16, {order = #NHWC}> -> tensor<1x1x289x289xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[PERMUTE_0]], [[PERMUTE_1]], [[PERMUTE_2]], [[PERMUTE_3]])
    // CHECK-SAME:     {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x1x289x289xf16>, tensor<1x1x289x289xf16>, tensor<1x1x289x289xf16>, tensor<1x1x289x289xf16> -> tensor<4x1x289x289xf16>

    // CHECK:       return [[CONCAT]] : tensor<4x1x289x289xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotUnrollMemPermuteWhenConstInput
func.func @NotUnrollMemPermuteWhenConstInput() -> tensor<4x1x289x289xf16> {
    %cst_in = const.Declare tensor<4x1x289x289xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<4x1x289x289xf16>, [#const.Reorder<#NHWC>]
    %1 = IE.MemPermute(%cst_in) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<4x1x289x289xf16, {order = #NHWC}> -> tensor<4x1x289x289xf16>
    return %1 : tensor<4x1x289x289xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<4x1x289x289xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4x1x289x289xf16>, [#const.Reorder<#NHWC>]
    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute([[CST]])
    // CHECK-SAME{LITERAL}:     {dst_order = #NCHW, mem_perm = #NWHC}
    // CHECK-SAME:     tensor<4x1x289x289xf16, {order = #NHWC}> -> tensor<4x1x289x289xf16>

    // CHECK:       return [[MEM_PERMUTE]] : tensor<4x1x289x289xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotUnrollMemPermuteIfCannotConvertToPolling
// CHECK-SAME:      [[INPUT:%.+]]: tensor<2x2x124x124xf16, {order = #NHWC}>
func.func @NotUnrollMemPermuteIfCannotConvertToPolling(%arg0: tensor<2x2x124x124xf16, {order = #NHWC}>) -> tensor<4x1x124x124xf16> {
    %0 = IE.ShapeCast {shape = [4, 1, 124, 124]} inputs(%arg0 : tensor<2x2x124x124xf16, {order = #NHWC}>) -> tensor<4x1x124x124xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<4x1x124x124xf16, {order = #NHWC}> -> tensor<4x1x124x124xf16>

    return %1 : tensor<4x1x124x124xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = IE.ShapeCast {shape = [4, 1, 124, 124]} inputs(%arg0 : tensor<2x2x124x124xf16, {order = #NHWC}>) -> tensor<4x1x124x124xf16, {order = #NHWC}>
    // CHECK:       [[PERMUTE:%.+]] = IE.MemPermute([[RESHAPE_IN]]) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<4x1x124x124xf16, {order = #NHWC}> -> tensor<4x1x124x124xf16>

    // CHECK:       return [[PERMUTE]] : tensor<4x1x124x124xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#CHNW = affine_map<(d0, d1, d2, d3) -> (d1, d2, d0, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotUnrollMemPermuteDueToDimNChanged
// CHECK-SAME:      [[INPUT:%.+]]: tensor<2x2x289x289xf16, {order = #NHWC}>
func.func @NotUnrollMemPermuteDueToDimNChanged(%arg0: tensor<2x2x289x289xf16, {order = #NHWC}>) -> tensor<289x289x4x1xf16> {
    %0 = IE.ShapeCast {shape = [4, 1, 289, 289]} inputs(%arg0 : tensor<2x2x289x289xf16, {order = #NHWC}>) -> tensor<4x1x289x289xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #CHNW} : tensor<4x1x289x289xf16, {order = #NHWC}> -> tensor<289x289x4x1xf16>

    return %1 : tensor<289x289x4x1xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = IE.ShapeCast {shape = [4, 1, 289, 289]} inputs(%arg0 : tensor<2x2x289x289xf16, {order = #NHWC}>) -> tensor<4x1x289x289xf16, {order = #NHWC}>
    // CHECK:       [[PERMUTE:%.+]] = IE.MemPermute([[RESHAPE_IN]]) {dst_order = #NCHW, mem_perm = #map} : tensor<4x1x289x289xf16, {order = #NHWC}> -> tensor<289x289x4x1xf16>

    // CHECK:       return [[PERMUTE]] : tensor<289x289x4x1xf16>
}

// -----

// CHECK-LABEL:  func.func @UnrollMultiply
// CHECK-SAME:      [[IN1:%.+]]: tensor<2x3x576x576xf16>
// CHECK-SAME:      [[IN2:%.+]]: tensor<2x1x576x576xf16>
func.func @UnrollMultiply(%input1 : tensor<2x3x576x576xf16>, %input2 : tensor<2x1x576x576xf16>) -> tensor<2x3x576x576xf16> {
  %res = IE.Multiply(%input1, %input2) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> }
    : tensor<2x3x576x576xf16>, tensor<2x1x576x576xf16> -> tensor<2x3x576x576xf16>
  return %res : tensor<2x3x576x576xf16>

    // CHECK: [[SLICE0_ARG0:%.+]] = IE.Slice [[IN1]] [0, 0, 0, 0] [1, 3, 576, 576] :
    // CHECK-SAME:      tensor<2x3x576x576xf16> to tensor<1x3x576x576xf16>

    // CHECK: [[SLICE0_ARG1:%.+]] = IE.Slice [[IN2]] [0, 0, 0, 0] [1, 1, 576, 576] :
    // CHECK-SAME:      tensor<2x1x576x576xf16> to tensor<1x1x576x576xf16>

    // CHECK: [[MUL1:%.+]] = IE.Multiply([[SLICE0_ARG0]], [[SLICE0_ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME: tensor<1x3x576x576xf16>, tensor<1x1x576x576xf16> -> tensor<1x3x576x576xf16>

    // CHECK: [[SLICE1_ARG0:%.+]] = IE.Slice [[IN1]] [1, 0, 0, 0] [1, 3, 576, 576] :
    // CHECK-SAME: tensor<2x3x576x576xf16> to tensor<1x3x576x576xf16>

    // CHECK: [[SLICE1_ARG1:%.+]] = IE.Slice [[IN2]] [1, 0, 0, 0] [1, 1, 576, 576] :
    // CHECK-SAME: tensor<2x1x576x576xf16> to tensor<1x1x576x576xf16>

    // CHECK: [[MUL2:%.+]] = IE.Multiply([[SLICE1_ARG0]], [[SLICE1_ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME: tensor<1x3x576x576xf16>, tensor<1x1x576x576xf16> -> tensor<1x3x576x576xf16>

    // CHECK: [[CONCAT:%.+]] = IE.Concat(%2, %5) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 0 : i64>
    // CHECK-SAME:  } : tensor<1x3x576x576xf16>, tensor<1x3x576x576xf16> -> tensor<2x3x576x576xf16>

    // CHECK:   return [[CONCAT]] : tensor<2x3x576x576xf16>
}

// -----

// CHECK-LABEL: @UnrollAveragePoolingBatch
func.func @UnrollAveragePoolingBatch(%arg0: tensor<2x128x32x64xf16>) -> tensor<2x128x32x64xf16> {
    %AVG_POOL = IE.AvgPool(%arg0) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<2x128x32x64xf16> -> tensor<2x128x32x64xf16>

    return %AVG_POOL : tensor<2x128x32x64xf16>

    // CHECK:   [[SLICE_0:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 128, 32, 64] :
    // CHECK-SAME:      tensor<2x128x32x64xf16> to tensor<1x128x32x64xf16>

    // CHECK:   [[AVG_POOL_0:%.+]] = IE.AvgPool([[SLICE_0]]) {
    // CHECK-SAME:      kernel_size = [3, 3],
    // CHECK-SAME:      pads_begin = [1, 1],
    // CHECK-SAME:      pads_end = [1, 1],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x128x32x64xf16> -> tensor<1x128x32x64xf16>

    // CHECK:   [[SLICE_1:%.+]] = IE.Slice %arg0 [1, 0, 0, 0] [1, 128, 32, 64] :
    // CHECK-SAME:      tensor<2x128x32x64xf16> to tensor<1x128x32x64xf16>

    // CHECK:   [[AVG_POOL_1:%.+]] = IE.AvgPool([[SLICE_1]]) {
    // CHECK-SAME:      kernel_size = [3, 3],
    // CHECK-SAME:      pads_begin = [1, 1],
    // CHECK-SAME:      pads_end = [1, 1],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x128x32x64xf16> -> tensor<1x128x32x64xf16>

    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[AVG_POOL_0]], [[AVG_POOL_1]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 0 : i64>
    // CHECK-SAME:  } : tensor<1x128x32x64xf16>, tensor<1x128x32x64xf16> -> tensor<2x128x32x64xf16>

    // CHECK:   return [[CONCAT]] : tensor<2x128x32x64xf16>
}
