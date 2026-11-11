//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --apply-tiling --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU6010

// -----

// Tiling along H (non-reduced axis): reduce output tiles have distinct H offsets.
// Reconstruction uses ConcatOp along H for both the primary output and the reduce output.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @ConvReduceMaxTiledOnH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x8x8xf16, {order = #NHWC}>
func.func @ConvReduceMaxTiledOnH(%arg0: tensor<1x16x8x8xf16, {order = #NHWC}>)
    -> (tensor<1x32x8x8xf16, {order = #NHWC}>, tensor<1x1x8x8xf16, {order = #NHWC}>) {
    %weights = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %conv, %red_max = VPU.NCE.Convolution(%arg0, %weights)
        rawFilterShape [32, 16, 1, 1] {
            axes_value = [1],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            resultSegmentSizes = array<i32: 1, 1, 0, 0>,
            strides = [1, 1],
            tilingStrategy = [1, 1, 2, 1]
        } : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<32x16x1x1xf16, {order = #NHWC}>
          -> tensor<1x32x8x8xf16, {order = #NHWC}>, tensor<1x1x8x8xf16, {order = #NHWC}>
    return %conv, %red_max : tensor<1x32x8x8xf16, {order = #NHWC}>, tensor<1x1x8x8xf16, {order = #NHWC}>

// CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>

// Tile 0: rows [0, 4)
// CHECK:           [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 4, 8]
// CHECK:           [[CONV0:%.+]], [[RMAX0:%.+]] = VPU.NCE.Convolution([[SLICE0]], [[WEIGHTS]])
// CHECK-SAME:          axes_value = [1]
// CHECK-SAME:          -> tensor<1x32x4x8xf16, {order = #NHWC}>, tensor<1x1x4x8xf16, {order = #NHWC}>

// Tile 1: rows [4, 8)
// CHECK:           [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 4, 0] [1, 16, 4, 8]
// CHECK:           [[CONV1:%.+]], [[RMAX1:%.+]] = VPU.NCE.Convolution([[SLICE1]], [[WEIGHTS]])
// CHECK-SAME:          axes_value = [1]
// CHECK-SAME:          -> tensor<1x32x4x8xf16, {order = #NHWC}>, tensor<1x1x4x8xf16, {order = #NHWC}>

// Primary output is concatenated along H
// CHECK:           [[OUT:%.+]] = VPU.Concat([[CONV0]], [[CONV1]])
// CHECK-SAME:          -> tensor<1x32x8x8xf16, {order = #NHWC}>

// Reduce output is also concatenated along H (not element-wise)
// CHECK:           [[RED:%.+]] = VPU.Concat([[RMAX0]], [[RMAX1]])
// CHECK-SAME:          -> tensor<1x1x8x8xf16, {order = #NHWC}>

// CHECK-NOT:       VPU.Maximum
// CHECK:           return [[OUT]], [[RED]]
}

// -----

// MaxPool with both reduce_xy_max and reduce_xy_min, tiled on H (non-reduced axis).
// Both reduce outputs have distinct H tile offsets; reconstruction uses ConcatOp for both.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @MaxPoolBothReducesTiledOnH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x8x4xf16, {order = #NHWC}>
func.func @MaxPoolBothReducesTiledOnH(%arg0: tensor<1x32x8x4xf16, {order = #NHWC}>)
    -> (tensor<1x32x8x4xf16, {order = #NHWC}>, tensor<1x1x8x4xf16, {order = #NHWC}>, tensor<1x1x8x4xf16, {order = #NHWC}>) {
    %pool, %red_max, %red_min = VPU.NCE.MaxPool(%arg0) {
            axes_value = [1],
            kernel_size = [1, 1],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            resultSegmentSizes = array<i32: 1, 1, 1, 0>,
            strides = [1, 1],
            tilingStrategy = [1, 1, 2, 1]
        } : tensor<1x32x8x4xf16, {order = #NHWC}>
          -> tensor<1x32x8x4xf16, {order = #NHWC}>, tensor<1x1x8x4xf16, {order = #NHWC}>, tensor<1x1x8x4xf16, {order = #NHWC}>
    return %pool, %red_max, %red_min
        : tensor<1x32x8x4xf16, {order = #NHWC}>, tensor<1x1x8x4xf16, {order = #NHWC}>, tensor<1x1x8x4xf16, {order = #NHWC}>

// Tile 0: rows [0, 4)
// CHECK:           [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 4, 4]
// CHECK:           [[POOL0:%.+]], [[RMAX0:%.+]], [[RMIN0:%.+]] = VPU.NCE.MaxPool([[SLICE0]])
// CHECK-SAME:          axes_value = [1]
// CHECK-SAME:          -> tensor<1x32x4x4xf16, {order = #NHWC}>, tensor<1x1x4x4xf16, {order = #NHWC}>, tensor<1x1x4x4xf16, {order = #NHWC}>

// Tile 1: rows [4, 8)
// CHECK:           [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 4, 0] [1, 32, 4, 4]
// CHECK:           [[POOL1:%.+]], [[RMAX1:%.+]], [[RMIN1:%.+]] = VPU.NCE.MaxPool([[SLICE1]])
// CHECK-SAME:          axes_value = [1]
// CHECK-SAME:          -> tensor<1x32x4x4xf16, {order = #NHWC}>, tensor<1x1x4x4xf16, {order = #NHWC}>, tensor<1x1x4x4xf16, {order = #NHWC}>

// All three outputs are concatenated along H (no element-wise ops needed)
// CHECK:           [[OUT:%.+]] = VPU.Concat([[POOL0]], [[POOL1]])
// CHECK-SAME:          -> tensor<1x32x8x4xf16, {order = #NHWC}>
// CHECK:           [[REDMAX:%.+]] = VPU.Concat([[RMAX0]], [[RMAX1]])
// CHECK-SAME:          -> tensor<1x1x8x4xf16, {order = #NHWC}>
// CHECK:           [[REDMIN:%.+]] = VPU.Concat([[RMIN0]], [[RMIN1]])
// CHECK-SAME:          -> tensor<1x1x8x4xf16, {order = #NHWC}>

// CHECK-NOT:       VPU.Maximum
// CHECK-NOT:       VPU.Minimum
// CHECK:           return [[OUT]], [[REDMAX]], [[REDMIN]]
}

// -----

// NCE.MatMul (5D GNHWC, axes_value=[2]): reduces along DimsGroups5D::Act::C (logical dim 2).
// Tiling on H (logical dim 3, tilingStrategy=[1,1,1,2,1]) produces distinct H offsets in every
// tile, so VPU.Concat reconstructs both the primary and the reduce output along H.
// This exercises getReduceOutputTiling for 5D layout and the multi-result tiling contract.

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: func.func @MatMulReduceMaxTiledOnH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<3x1x32x16x16xf16, {order = #GNHWC}>
func.func @MatMulReduceMaxTiledOnH(%arg0: tensor<3x1x32x16x16xf16, {order = #GNHWC}>)
    -> (tensor<3x1x32x16x16xf16, {order = #GNHWC}>, tensor<3x1x1x16x16xf16, {order = #GNHWC}>) {
    %weights = const.Declare tensor<3x32x32x1x1xf16, {order = #GNHWC}> = dense<1.0> : tensor<3x32x32x1x1xf16>, [#const.Reorder<#GNHWC>]
    %mm, %red_max = VPU.NCE.MatMul(%arg0, %weights) rawFilterShape [3, 32, 32, 1, 1] {
            axes_value = [2],
            operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            resultSegmentSizes = array<i32: 1, 1, 0, 0>,
            strides = [1, 1],
            tilingStrategy = [1, 1, 1, 2, 1]
        } -> tensor<3x1x32x16x16xf16, {order = #GNHWC}>, tensor<3x1x1x16x16xf16, {order = #GNHWC}>
    return %mm, %red_max : tensor<3x1x32x16x16xf16, {order = #GNHWC}>, tensor<3x1x1x16x16xf16, {order = #GNHWC}>

// CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<3x32x32x1x1xf16, {order = #GNHWC}>

// Tile 0: H rows [0, 8)
// CHECK:           [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0, 0] [3, 1, 32, 8, 16]
// CHECK:           [[MM0:%.+]], [[RMAX0:%.+]] = VPU.NCE.MatMul([[SLICE0]], [[WEIGHTS]])
// CHECK-SAME:          axes_value = [2]
// CHECK-SAME:          -> tensor<3x1x32x8x16xf16, {order = #GNHWC}>, tensor<3x1x1x8x16xf16, {order = #GNHWC}>

// Tile 1: H rows [8, 16)
// CHECK:           [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 8, 0] [3, 1, 32, 8, 16]
// CHECK:           [[MM1:%.+]], [[RMAX1:%.+]] = VPU.NCE.MatMul([[SLICE1]], [[WEIGHTS]])
// CHECK-SAME:          axes_value = [2]
// CHECK-SAME:          -> tensor<3x1x32x8x16xf16, {order = #GNHWC}>, tensor<3x1x1x8x16xf16, {order = #GNHWC}>

// Primary output concatenated along H
// CHECK:           [[OUT:%.+]] = VPU.Concat([[MM0]], [[MM1]])
// CHECK-SAME:          -> tensor<3x1x32x16x16xf16, {order = #GNHWC}>

// Reduce output concatenated along H (C=1 in both tiles; H offsets differ, not C offsets)
// CHECK:           [[RED:%.+]] = VPU.Concat([[RMAX0]], [[RMAX1]])
// CHECK-SAME:          -> tensor<3x1x1x16x16xf16, {order = #GNHWC}>

// CHECK-NOT:       VPU.Maximum
// CHECK:           return [[OUT]], [[RED]]
}

// -----

// NCE.Eltwise with reduce_xy_max, tiled on H (non-reduced axis).
// Reconstruction uses ConcatOp along H for both primary and reduce output.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @EltwiseReduceMaxTiledOnH
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<1x32x8x8xf16, {order = #NHWC}>,
// CHECK-SAME:    [[INPUT1:%.+]]: tensor<1x32x8x8xf16, {order = #NHWC}>
func.func @EltwiseReduceMaxTiledOnH(%arg0: tensor<1x32x8x8xf16, {order = #NHWC}>,
                                    %arg1: tensor<1x32x8x8xf16, {order = #NHWC}>)
    -> (tensor<1x32x8x8xf16, {order = #NHWC}>, tensor<1x1x8x8xf16, {order = #NHWC}>) {
    %eltwise, %red_max = VPU.NCE.Eltwise(%arg0, %arg1) {
            axes_value = [1],
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEStub<>,
            resultSegmentSizes = array<i32: 1, 1, 0, 0>,
            tilingStrategy = [1, 1, 2, 1]
        } : tensor<1x32x8x8xf16, {order = #NHWC}>, tensor<1x32x8x8xf16, {order = #NHWC}>
          -> tensor<1x32x8x8xf16, {order = #NHWC}>, tensor<1x1x8x8xf16, {order = #NHWC}>
    return %eltwise, %red_max : tensor<1x32x8x8xf16, {order = #NHWC}>, tensor<1x1x8x8xf16, {order = #NHWC}>

// Tile 0: rows [0, 4)
// CHECK:           [[SLICE0_0:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 0] [1, 32, 4, 8]
// CHECK:           [[SLICE0_1:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 32, 4, 8]
// CHECK:           [[ELT0:%.+]], [[RMAX0:%.+]] = VPU.NCE.Eltwise([[SLICE0_0]], [[SLICE0_1]])
// CHECK-SAME:          axes_value = [1]
// CHECK-SAME:          -> tensor<1x32x4x8xf16, {order = #NHWC}>, tensor<1x1x4x8xf16, {order = #NHWC}>

// Tile 1: rows [4, 8)
// CHECK:           [[SLICE1_0:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 4, 0] [1, 32, 4, 8]
// CHECK:           [[SLICE1_1:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 4, 0] [1, 32, 4, 8]
// CHECK:           [[ELT1:%.+]], [[RMAX1:%.+]] = VPU.NCE.Eltwise([[SLICE1_0]], [[SLICE1_1]])
// CHECK-SAME:          axes_value = [1]
// CHECK-SAME:          -> tensor<1x32x4x8xf16, {order = #NHWC}>, tensor<1x1x4x8xf16, {order = #NHWC}>

// Primary output concatenated along H
// CHECK:           [[OUT:%.+]] = VPU.Concat([[ELT0]], [[ELT1]])
// CHECK-SAME:          -> tensor<1x32x8x8xf16, {order = #NHWC}>

// Reduce output concatenated along H
// CHECK:           [[RED:%.+]] = VPU.Concat([[RMAX0]], [[RMAX1]])
// CHECK-SAME:          -> tensor<1x1x8x8xf16, {order = #NHWC}>

// CHECK-NOT:       VPU.Maximum
// CHECK:           return [[OUT]], [[RED]]
}

// -----

// NCE.DepthConvolution with reduce_xy_max, tiled on H (non-reduced axis).
// Reconstruction uses ConcatOp along H for both primary and reduce output.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @DepthConvReduceMaxTiledOnH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x8x8xf16, {order = #NHWC}>
func.func @DepthConvReduceMaxTiledOnH(%arg0: tensor<1x32x8x8xf16, {order = #NHWC}>)
    -> (tensor<1x32x8x8xf16, {order = #NHWC}>, tensor<1x1x8x8xf16, {order = #NHWC}>) {
    %filter = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %dwconv, %red_max = VPU.NCE.DepthConvolution(%arg0, %filter) rawFilterShape [32, 1, 1, 1] {
            axes_value = [1],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            resultSegmentSizes = array<i32: 1, 1, 0, 0>,
            strides = [1, 1],
            tilingStrategy = [1, 1, 2, 1]
        } -> tensor<1x32x8x8xf16, {order = #NHWC}>, tensor<1x1x8x8xf16, {order = #NHWC}>
    return %dwconv, %red_max : tensor<1x32x8x8xf16, {order = #NHWC}>, tensor<1x1x8x8xf16, {order = #NHWC}>

// CHECK-DAG:       [[FILTER:%.+]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>

// Tile 0: rows [0, 4)
// CHECK:           [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 4, 8]
// CHECK:           [[DW0:%.+]], [[RMAX0:%.+]] = VPU.NCE.DepthConvolution([[SLICE0]], [[FILTER]])
// CHECK-SAME:          axes_value = [1]
// CHECK-SAME:          -> tensor<1x32x4x8xf16, {order = #NHWC}>, tensor<1x1x4x8xf16, {order = #NHWC}>

// Tile 1: rows [4, 8)
// CHECK:           [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 4, 0] [1, 32, 4, 8]
// CHECK:           [[DW1:%.+]], [[RMAX1:%.+]] = VPU.NCE.DepthConvolution([[SLICE1]], [[FILTER]])
// CHECK-SAME:          axes_value = [1]
// CHECK-SAME:          -> tensor<1x32x4x8xf16, {order = #NHWC}>, tensor<1x1x4x8xf16, {order = #NHWC}>

// Primary output concatenated along H
// CHECK:           [[OUT:%.+]] = VPU.Concat([[DW0]], [[DW1]])
// CHECK-SAME:          -> tensor<1x32x8x8xf16, {order = #NHWC}>

// Reduce output concatenated along H
// CHECK:           [[RED:%.+]] = VPU.Concat([[RMAX0]], [[RMAX1]])
// CHECK-SAME:          -> tensor<1x1x8x8xf16, {order = #NHWC}>

// CHECK-NOT:       VPU.Maximum
// CHECK:           return [[OUT]], [[RED]]
}
