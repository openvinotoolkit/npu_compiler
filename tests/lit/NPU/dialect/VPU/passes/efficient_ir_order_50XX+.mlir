//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --efficient-ir-order="enable-reorder-concat-branches=true" %s | FileCheck %s
// REQUIRES: arch-NPU50XX

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ReorderSymmetricConcatBranches
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<4096x4096x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT2:%arg[0-9]]]: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT3:%arg[0-9]]]: tensor<4096x4096x1x1xf16, {order = #NHWC}>
func.func @ReorderSymmetricConcatBranches(%arg0: tensor<1x4096x1024x4xf16, {order = #NHWC}>, %arg1: tensor<4096x4096x1x1xf16, {order = #NHWC}>, %arg2: tensor<1x4096x1024x4xf16, {order = #NHWC}>, %arg3: tensor<4096x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x2x4096x40xf16, {order = #NHCW}> {
    %cst = const.Declare tensor<48x4096x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<48x4096x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %arg1) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [4096, 4096, 1, 1], strides = [1, 1], tilingStrategy = [1, 1, 20, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<4096x4096x1x1xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
    %1 = VPU.SoftMax(%0) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, padSize = 3 : i64} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%arg2, %arg3) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [4096, 4096, 1, 1], strides = [1, 1], tilingStrategy = [1, 1, 21, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<4096x4096x1x1xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
    %3 = VPU.SoftMax(%2) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, padSize = 3 : i64} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
    %4 = VPU.NCE.Convolution(%1, %cst) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [48, 4096, 1, 1], strides = [1, 1], tilingStrategy = [1, 1, 22, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
    %5 = VPU.Slice %4 [0, 0, 0, 0] [1, 40, 1024, 4] : tensor<1x48x1024x4xf16, {order = #NHWC}> to tensor<1x40x1024x4xf16, {order = #NHWC}>
    %6 = VPU.AffineReshape(%5) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 40, 4096, 1]} : tensor<1x40x1024x4xf16, {order = #NHWC}> -> tensor<1x40x4096x1xf16, {order = #NHWC}>
    %7 = VPU.PermuteCast(%6) {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>} : tensor<1x40x4096x1xf16, {order = #NHWC}> -> tensor<4096x40x1x1xf16>
    %8 = VPU.NCE.Convolution(%3, %cst) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [48, 4096, 1, 1], strides = [1, 1], tilingStrategy = [1, 1, 23, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
    %9 = VPU.Slice %8 [0, 0, 0, 0] [1, 40, 1024, 4] : tensor<1x48x1024x4xf16, {order = #NHWC}> to tensor<1x40x1024x4xf16, {order = #NHWC}>
    %10 = VPU.AffineReshape(%9) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 40, 4096, 1]} : tensor<1x40x1024x4xf16, {order = #NHWC}> -> tensor<1x40x4096x1xf16, {order = #NHWC}>
    %11 = VPU.PermuteCast(%10) {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>} : tensor<1x40x4096x1xf16, {order = #NHWC}> -> tensor<4096x40x1x1xf16>
    %12 = VPU.AffineReshape(%11) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 4096, 1, 40]} : tensor<4096x40x1x1xf16> -> tensor<1x4096x1x40xf16>
    %13 = VPU.PermuteCast(%12) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x4096x1x40xf16> -> tensor<1x1x4096x40xf16, {order = #NHCW}>
    %14 = VPU.AffineReshape(%7) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 4096, 1, 40]} : tensor<4096x40x1x1xf16> -> tensor<1x4096x1x40xf16>
    %15 = VPU.PermuteCast(%14) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x4096x1x40xf16> -> tensor<1x1x4096x40xf16, {order = #NHCW}>
    %16 = VPU.Concat(%13, %15) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x4096x40xf16, {order = #NHCW}>, tensor<1x1x4096x40xf16, {order = #NHCW}> -> tensor<1x2x4096x40xf16, {order = #NHCW}>

    return %16 : tensor<1x2x4096x40xf16, {order = #NHCW}>

    // CHECK: [[CST:%.+]] = const.Declare tensor<48x4096x1x1xf16, {order = #NHWC}>

    // CHECK: [[CONV_1_1:%.+]] = VPU.NCE.Convolution([[INPUT0]], [[INPUT1]])
    // CHECK: [[SOFTMAX_1:%.+]] = VPU.SoftMax([[CONV_1_1]])
    // CHECK: [[CONV_1_2:%.+]] = VPU.NCE.Convolution([[SOFTMAX_1]], [[CST]])
    // CHECK: [[SLICE_1:%.+]] = VPU.Slice [[CONV_1_2]]
    // CHECK: [[RESHAPE_1_1:%.+]] = VPU.AffineReshape([[SLICE_1]])
    // CHECK: [[PERMUTECAST_1_1:%.+]] = VPU.PermuteCast([[RESHAPE_1_1]])
    // CHECK: [[RESHAPE_1_2:%.+]] = VPU.AffineReshape([[PERMUTECAST_1_1]])
    // CHECK: [[PERMUTECAST_1_2:%.+]] = VPU.PermuteCast([[RESHAPE_1_2]])

    // CHECK: [[CONV_2_1:%.+]] = VPU.NCE.Convolution([[INPUT2]], [[INPUT3]])
    // CHECK: [[SOFTMAX_2:%.+]] = VPU.SoftMax([[CONV_2_1]])
    // CHECK: [[CONV_2_2:%.+]] = VPU.NCE.Convolution([[SOFTMAX_2]], [[CST]])
    // CHECK: [[SLICE_2:%.+]] = VPU.Slice [[CONV_2_2]]
    // CHECK: [[RESHAPE_2_1:%.+]] = VPU.AffineReshape([[SLICE_2]])
    // CHECK: [[PERMUTECAST_2_1:%.+]] = VPU.PermuteCast([[RESHAPE_2_1]])
    // CHECK: [[RESHAPE_2_2:%.+]] = VPU.AffineReshape([[PERMUTECAST_2_1]])
    // CHECK: [[PERMUTECAST_2_2:%.+]] = VPU.PermuteCast([[RESHAPE_2_2]])

    // CHECK: [[CONCAT:%.+]] = VPU.Concat([[PERMUTECAST_2_2]], [[PERMUTECAST_1_2]])
    // CHECK: return [[CONCAT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @NotReorderSymmetricConcatBranchesIfTaskCanBeParallel
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<1x16x1024x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<16x16x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT2:%arg[0-9]]]: tensor<1x16x1024x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT3:%arg[0-9]]]: tensor<16x16x1x1xf16, {order = #NHWC}>
func.func @NotReorderSymmetricConcatBranchesIfTaskCanBeParallel(%arg0: tensor<1x16x1024x4xf16, {order = #NHWC}>, %arg1: tensor<16x16x1x1xf16, {order = #NHWC}>, %arg2: tensor<1x16x1024x4xf16, {order = #NHWC}>, %arg3: tensor<16x16x1x1xf16, {order = #NHWC}>) -> tensor<1x2x16x40xf16, {order = #NHCW}> {
    %cst = const.Declare tensor<48x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<48x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %arg1) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [16, 16, 1, 1], strides = [1, 1], tilingStrategy = [1, 1, 20, 1]} : tensor<1x16x1024x4xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x1024x4xf16, {order = #NHWC}>
    %1 = VPU.SoftMax(%0) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, padSize = 3 : i64} : tensor<1x16x1024x4xf16, {order = #NHWC}> -> tensor<1x16x1024x4xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%arg2, %arg3) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [16, 16, 1, 1], strides = [1, 1], tilingStrategy = [1, 1, 21, 1]} : tensor<1x16x1024x4xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x1024x4xf16, {order = #NHWC}>
    %3 = VPU.SoftMax(%2) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, padSize = 3 : i64} : tensor<1x16x1024x4xf16, {order = #NHWC}> -> tensor<1x16x1024x4xf16, {order = #NHWC}>
    %4 = VPU.NCE.Convolution(%1, %cst) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [48, 16, 1, 1], strides = [1, 1], tilingStrategy = [1, 1, 22, 1]} : tensor<1x16x1024x4xf16, {order = #NHWC}>, tensor<48x16x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
    %5 = VPU.Slice %4 [0, 0, 0, 0] [1, 40, 1024, 4] : tensor<1x48x1024x4xf16, {order = #NHWC}> to tensor<1x40x1024x4xf16, {order = #NHWC}>
    %6 = VPU.AffineReshape(%5) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 40, 16, 1]} : tensor<1x40x1024x4xf16, {order = #NHWC}> -> tensor<1x40x16x1xf16, {order = #NHWC}>
    %7 = VPU.PermuteCast(%6) {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>} : tensor<1x40x16x1xf16, {order = #NHWC}> -> tensor<16x40x1x1xf16>
    %8 = VPU.NCE.Convolution(%3, %cst) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [48, 16, 1, 1], strides = [1, 1], tilingStrategy = [1, 1, 23, 1]} : tensor<1x16x1024x4xf16, {order = #NHWC}>, tensor<48x16x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
    %9 = VPU.Slice %8 [0, 0, 0, 0] [1, 40, 1024, 4] : tensor<1x48x1024x4xf16, {order = #NHWC}> to tensor<1x40x1024x4xf16, {order = #NHWC}>
    %10 = VPU.AffineReshape(%9) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 40, 16, 1]} : tensor<1x40x1024x4xf16, {order = #NHWC}> -> tensor<1x40x16x1xf16, {order = #NHWC}>
    %11 = VPU.PermuteCast(%10) {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>} : tensor<1x40x16x1xf16, {order = #NHWC}> -> tensor<16x40x1x1xf16>
    %12 = VPU.AffineReshape(%11) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 16, 1, 40]} : tensor<16x40x1x1xf16> -> tensor<1x16x1x40xf16>
    %13 = VPU.PermuteCast(%12) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x16x1x40xf16> -> tensor<1x1x16x40xf16, {order = #NHCW}>
    %14 = VPU.AffineReshape(%7) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 16, 1, 40]} : tensor<16x40x1x1xf16> -> tensor<1x16x1x40xf16>
    %15 = VPU.PermuteCast(%14) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x16x1x40xf16> -> tensor<1x1x16x40xf16, {order = #NHCW}>
    %16 = VPU.Concat(%13, %15) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x16x40xf16, {order = #NHCW}>, tensor<1x1x16x40xf16, {order = #NHCW}> -> tensor<1x2x16x40xf16, {order = #NHCW}>

    return %16 : tensor<1x2x16x40xf16, {order = #NHCW}>

    // CHECK: [[CST:%.+]] = const.Declare tensor<48x16x1x1xf16, {order = #NHWC}>

    // CHECK: [[CONV_1_1:%.+]] = VPU.NCE.Convolution([[INPUT0]], [[INPUT1]])
    // CHECK: [[SOFTMAX_1:%.+]] = VPU.SoftMax([[CONV_1_1]])
    // CHECK: [[CONV_2_1:%.+]] = VPU.NCE.Convolution([[INPUT2]], [[INPUT3]])
    // CHECK: [[SOFTMAX_2:%.+]] = VPU.SoftMax([[CONV_2_1]])
    // CHECK: [[CONV_1_2:%.+]] = VPU.NCE.Convolution([[SOFTMAX_1]], [[CST]])
    // CHECK: [[SLICE_1:%.+]] = VPU.Slice [[CONV_1_2]]
    // CHECK: [[RESHAPE_1:%.+]] = VPU.AffineReshape([[SLICE_1]])
    // CHECK: [[PERMUTECAST_1:%.+]] = VPU.PermuteCast([[RESHAPE_1]])
    // CHECK: [[CONV_2_2:%.+]] = VPU.NCE.Convolution([[SOFTMAX_2]], [[CST]])
    // CHECK: [[SLICE_2:%.+]] = VPU.Slice [[CONV_2_2]]
    // CHECK: [[RESHAPE_2:%.+]] = VPU.AffineReshape([[SLICE_2]])
    // CHECK: [[PERMUTECAST_2:%.+]] = VPU.PermuteCast([[RESHAPE_2]])
    // CHECK: [[RESHAPE_3:%.+]] = VPU.AffineReshape([[PERMUTECAST_2]])
    // CHECK: [[PERMUTECAST_3:%.+]] = VPU.PermuteCast([[RESHAPE_3]])
    // CHECK: [[RESHAPE_4:%.+]] = VPU.AffineReshape([[PERMUTECAST_1]])
    // CHECK: [[PERMUTECAST_4:%.+]] = VPU.PermuteCast([[RESHAPE_4]])
    // CHECK: [[CONCAT:%.+]] = VPU.Concat([[PERMUTECAST_3]], [[PERMUTECAST_4]])
    // CHECK: return [[CONCAT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @NotReorderAsymmetricConcatBranches
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<48x4096x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT2:%arg[0-9]]]: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT3:%arg[0-9]]]: tensor<1x48x1024x4xf16, {order = #NHWC}>
func.func @NotReorderAsymmetricConcatBranches(%arg0: tensor<1x4096x1024x4xf16, {order = #NHWC}>, %arg1: tensor<48x4096x1x1xf16, {order = #NHWC}>, %arg2: tensor<1x4096x1024x4xf16, {order = #NHWC}>, %arg3: tensor<1x48x1024x4xf16, {order = #NHWC}>) -> tensor<1x2x4096x40xf16, {order = #NHCW}> {
    %0 = VPU.NCE.Convolution(%arg0, %arg1) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [48, 4096, 1, 1], strides = [1, 1], tilingStrategy = [1, 1, 22, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
    %1 = VPU.Slice %0 [0, 0, 0, 0] [1, 40, 1024, 4] : tensor<1x48x1024x4xf16, {order = #NHWC}> to tensor<1x40x1024x4xf16, {order = #NHWC}>
    %2 = VPU.AffineReshape(%1) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 40, 4096, 1]} : tensor<1x40x1024x4xf16, {order = #NHWC}> -> tensor<1x40x4096x1xf16, {order = #NHWC}>
    %3 = VPU.PermuteCast(%2) {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>} : tensor<1x40x4096x1xf16, {order = #NHWC}> -> tensor<4096x40x1x1xf16>
    %5 = VPU.Slice %arg3 [0, 0, 0, 0] [1, 40, 1024, 4] : tensor<1x48x1024x4xf16, {order = #NHWC}> to tensor<1x40x1024x4xf16, {order = #NHWC}>
    %6 = VPU.AffineReshape(%5) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 40, 4096, 1]} : tensor<1x40x1024x4xf16, {order = #NHWC}> -> tensor<1x40x4096x1xf16, {order = #NHWC}>
    %7 = VPU.PermuteCast(%6) {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>} : tensor<1x40x4096x1xf16, {order = #NHWC}> -> tensor<4096x40x1x1xf16>
    %8 = VPU.AffineReshape(%3) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 4096, 1, 40]} : tensor<4096x40x1x1xf16> -> tensor<1x4096x1x40xf16>
    %9 = VPU.PermuteCast(%8) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x4096x1x40xf16> -> tensor<1x1x4096x40xf16, {order = #NHCW}>
    %10 = VPU.AffineReshape(%7) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 4096, 1, 40]} : tensor<4096x40x1x1xf16> -> tensor<1x4096x1x40xf16>
    %11 = VPU.PermuteCast(%10) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x4096x1x40xf16> -> tensor<1x1x4096x40xf16, {order = #NHCW}>
    %12 = VPU.Concat(%9, %11) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x4096x40xf16, {order = #NHCW}>, tensor<1x1x4096x40xf16, {order = #NHCW}> -> tensor<1x2x4096x40xf16, {order = #NHCW}>

    return %12 : tensor<1x2x4096x40xf16, {order = #NHCW}>

    // CHECK: [[CONV_1:%.+]] = VPU.NCE.Convolution([[INPUT0]], [[INPUT1]])
    // CHECK: [[SLICE_1:%.+]] = VPU.Slice [[CONV_1]]
    // CHECK: [[RESHAPE_1:%.+]] = VPU.AffineReshape([[SLICE_1]])
    // CHECK: [[PERMUTECAST_1:%.+]] = VPU.PermuteCast([[RESHAPE_1]])
    // CHECK: [[SLICE_2:%.+]] = VPU.Slice [[INPUT3]]
    // CHECK: [[RESHAPE_2:%.+]] = VPU.AffineReshape([[SLICE_2]])
    // CHECK: [[PERMUTECAST_2:%.+]] = VPU.PermuteCast([[RESHAPE_2]])
    // CHECK: [[RESHAPE_3:%.+]] = VPU.AffineReshape([[PERMUTECAST_1]])
    // CHECK: [[PERMUTECAST_3:%.+]] = VPU.PermuteCast([[RESHAPE_3]])
    // CHECK: [[RESHAPE_4:%.+]] = VPU.AffineReshape([[PERMUTECAST_2]])
    // CHECK: [[PERMUTECAST_4:%.+]] = VPU.PermuteCast([[RESHAPE_4]])
    // CHECK: [[CONCAT:%.+]] = VPU.Concat([[PERMUTECAST_3]], [[PERMUTECAST_4]])
    // CHECK: return [[CONCAT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderAsymmetricConcatBranches
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<1x96x32x32xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<1x96x32x32xf16>
func.func @ReorderAsymmetricConcatBranches(%arg0: tensor<1x96x32x32xf16, {order = #NHWC}>, %arg1: tensor<1x96x32x32xf16>) -> tensor<1x192x32x32xf16> {
  %0 = VPU.SoftMax(%arg0) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x96x32x32xf16, {order = #NHWC}> -> tensor<1x96x32x32xf16, {order = #NHWC}>
  %1 = VPU.NCE.AveragePool(%0) {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <LRELU>, clamp_low = 0.000000e+00 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [-0.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1]} -> tensor<1x96x32x32xf16> 
  %2 = VPU.PermuteCast(%arg1) {dst_order = #NHWC, mem_perm = #NCHW } : tensor<1x96x32x32xf16> -> tensor<1x32x96x32xf16, {order = #NHWC}>
  %3 = VPU.NCE.MaxPool(%2) {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1]} -> tensor<1x32x96x32xf16, {order = #NHCW}> 
  %4 = VPU.PermuteCast(%3) {dst_order = #NCWH, mem_perm = #NCHW } : tensor<1x32x96x32xf16, {order = #NHCW}> -> tensor<1x96x32x32xf16, {order = #NCWH}>
  %5 = VPU.SoftMax(%4) {axisInd = 2 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x96x32x32xf16, {order = #NCWH}> -> tensor<1x96x32x32xf16, {order = #NCWH}>
  %6 = VPU.PermuteCast(%5) {dst_order = #NHWC, mem_perm = #NCHW } : tensor<1x96x32x32xf16, {order = #NCWH}> -> tensor<1x32x96x32xf16, {order = #NHWC}>
  %7 = VPU.NCE.MaxPool(%6) {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1]} -> tensor<1x32x96x32xf16, {order = #NHCW}> 
  %8 = VPU.PermuteCast(%7) {dst_order = #NHWC, mem_perm = #NCHW } : tensor<1x32x96x32xf16, {order = #NHCW}> -> tensor<1x32x96x32xf16, {order = #NHWC}>
  %9 = VPU.NCE.AveragePool(%8) {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <LRELU>, clamp_low = 0.000000e+00 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [-0.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1]} -> tensor<1x32x96x32xf16, {order = #NHWC}> 
  %10 = VPU.PermuteCast(%9) {dst_order = #NCHW , mem_perm = #NCHW } : tensor<1x32x96x32xf16, {order = #NHWC}> -> tensor<1x96x32x32xf16>
  %11 = VPU.Concat(%1, %10) {static_offsets = [[0, 0, 0, 0], [0, 96, 0, 0]]} : tensor<1x96x32x32xf16>, tensor<1x96x32x32xf16> -> tensor<1x192x32x32xf16>
  return %11 : tensor<1x192x32x32xf16>

  // CHECK: [[SOFTMAX_0:%.+]] = VPU.SoftMax([[INPUT0]])
  // CHECK: [[PERMUTECAST_0:%.+]] = VPU.PermuteCast([[INPUT1]])
  // CHECK: [[MAXPOOL_0:%.+]] = VPU.NCE.MaxPool([[PERMUTECAST_0]])
  // CHECK: [[PERMUTECAST_1:%.+]] = VPU.PermuteCast([[MAXPOOL_0]])
  // CHECK: [[SOFTMAX_1:%.+]] = VPU.SoftMax([[PERMUTECAST_1]])
  // CHECK: [[PERMUTECAST_2:%.+]] = VPU.PermuteCast([[SOFTMAX_1]])
  // CHECK: [[MAXPOOL_1:%.+]] = VPU.NCE.MaxPool([[PERMUTECAST_2]])
  // CHECK: [[PERMUTECAST_3:%.+]] = VPU.PermuteCast([[MAXPOOL_1]])
  // CHECK: [[AVGPOOL_0:%.+]] = VPU.NCE.AveragePool([[PERMUTECAST_3]])
  // CHECK: [[PERMUTECAST_4:%.+]] = VPU.PermuteCast([[AVGPOOL_0]])
  // CHECK: [[AVGPOOL_1:%.+]] = VPU.NCE.AveragePool([[SOFTMAX_0]])
  // CHECK: [[CONCAT:%.+]] = VPU.Concat([[AVGPOOL_1]], [[PERMUTECAST_4]])
  // CHECK: return [[CONCAT:%.+]] : tensor<1x192x32x32xf16>
}
