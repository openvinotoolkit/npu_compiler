//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --tiling-strategy-assignment="tiling-mode=ISOLATED" %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

module @Test {

config.Resources 6 of @NCE {
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @NCEMatMulSOGAndGTile
func.func @NCEMatMulSOGAndGTile(%arg0: tensor<64x8x64x32xf16>, %arg1: tensor<64x8x64x32xf16>) -> tensor<512x1x64x64x1xf16, {order = #GNHWC}> {
  %cst_0 = const.Declare tensor<512x64x1x1x4xsi32> = dense<10> : tensor<512x64x1x1x4xsi32>
  %0 = VPU.ShapeCast {shape = [1, 512, 64, 32]} inputs(%arg0 : tensor<64x8x64x32xf16>) -> tensor<1x512x64x32xf16>
  %1 = VPU.ShapeCast {shape = [1, 512, 64, 32]} inputs(%arg1 : tensor<64x8x64x32xf16>) -> tensor<1x512x64x32xf16>
  %2 = VPU.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [512, 64, 32, 1, 1]} : tensor<1x512x64x32xf16> -> tensor<512x64x32x1x1xf16>
  %3 = VPU.PermuteCast(%2) {dst_order = #GNHWC, mem_perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>} : tensor<512x64x32x1x1xf16> -> tensor<512x1x32x64x1xf16, {order = #GNHWC}>
  %4 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [512, 64, 32, 1, 1]} : tensor<1x512x64x32xf16> -> tensor<512x64x32x1x1xf16>
  %5 = VPU.PermuteCast(%4) {dst_order = #GNHWC, mem_perm = #GNHWC} : tensor<512x64x32x1x1xf16> -> tensor<512x64x32x1x1xf16, {order = #GNHWC}>
  %6 = VPU.AffineReshape(%3) {dim_mapping = [[0], [1], [2], [3, 4], [4]], shape_value = [512, 1, 32, 16, 4]} : tensor<512x1x32x64x1xf16, {order = #GNHWC}> -> tensor<512x1x32x16x4xf16, {order = #GNHWC}>
  %7 = VPU.NCE.MatMul(%6, %5, %cst_0) rawFilterShape [512, 64, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
     strides = [1, 1], tilingStrategy = [2, 1, 1, 1, 1]} -> tensor<512x1x64x16x4xf16, {order = #GNHWC}>
  %8 = VPU.AffineReshape(%7) {dim_mapping = [[0], [1], [2], [3], [3, 4]], shape_value = [512, 1, 64, 64, 1]} : tensor<512x1x64x16x4xf16, {order = #GNHWC}> -> tensor<512x1x64x64x1xf16, {order = #GNHWC}>
  return %8 : tensor<512x1x64x64x1xf16, {order = #GNHWC}>

  // CHECK:         VPU.NCE.MatMul
  // CHECK-SAME:    tilingStrategy = [2, 1, 1, 1, 1]
}
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

module @Test {

config.Resources 6 of @NCE {
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @NCEMatMulSOGAndHTile
func.func @NCEMatMulSOGAndHTile(%arg0: tensor<6x1x512x512xf16>, %arg1: tensor<6x1x512x512xf16>) -> tensor<6x1x512x512x1xf16, {order = #GNHWC}> {
  %cst = const.Declare tensor<6x512x1x1x4xsi32> = dense<10> : tensor<6x512x1x1x4xsi32>
  %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 6, 512, 512]} : tensor<6x1x512x512xf16> -> tensor<1x6x512x512xf16>
  %1 = VPU.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 6, 512, 512]} : tensor<6x1x512x512xf16> -> tensor<1x6x512x512xf16>
  %2 = VPU.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [6, 512, 512, 1, 1]} : tensor<1x6x512x512xf16> -> tensor<6x512x512x1x1xf16>
  %3 = VPU.PermuteCast(%2) {dst_order = #GNHWC, mem_perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>} : tensor<6x512x512x1x1xf16> -> tensor<6x1x512x512x1xf16, {order = #GNHWC}>
  %4 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [6, 512, 512, 1, 1]} : tensor<1x6x512x512xf16> -> tensor<6x512x512x1x1xf16>
  %5 = VPU.PermuteCast(%4) {dst_order = #GNHWC, mem_perm = #GNHWC} : tensor<6x512x512x1x1xf16> -> tensor<6x512x512x1x1xf16, {order = #GNHWC}>
  %6 = VPU.AffineReshape(%3) {dim_mapping = [[0], [1], [2], [3, 4], [4]], shape_value = [6, 1, 512, 128, 4]} : tensor<6x1x512x512x1xf16, {order = #GNHWC}> -> tensor<6x1x512x128x4xf16, {order = #GNHWC}>
  %7 = VPU.NCE.MatMul(%6, %5, %cst) rawFilterShape [6, 512, 512, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1]} -> tensor<6x1x512x128x4xf16, {order = #GNHWC}>
  %8 = VPU.AffineReshape(%7) {dim_mapping = [[0], [1], [2], [3], [3, 4]], shape_value = [6, 1, 512, 512, 1]} : tensor<6x1x512x128x4xf16, {order = #GNHWC}> -> tensor<6x1x512x512x1xf16, {order = #GNHWC}>
  return %8 : tensor<6x1x512x512x1xf16, {order = #GNHWC}>

  // CHECK:         VPU.NCE.MatMul
  // CHECK-SAME:    tilingStrategy = [1, 1, 1, 2, 1]
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test attributes {config.compilationMode = #config.compilation_mode<HostCompile>}  {

config.Resources 3 of @NCE {
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @DepthToSpaceDynamic2Dim
func.func @DepthToSpaceDynamic2Dim(%arg0: tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 256, 256]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 512, 512]> : tensor<4xsi64>, order = #NHWC}> {
  %1 = VPU.DepthToSpace(%arg0) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, padded_channels = #IE.ChannelPadding<input = 0 : i64, output = 13 : i64>} : tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 256, 256]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 512, 512]> : tensor<4xsi64>, order = #NHWC}>
  return %1 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 512, 512]> : tensor<4xsi64>, order = #NHWC}>

  // CHECK:         VPU.DepthToSpace
  // CHECK-SAME:    tilingStrategy = [1, 1, 2, 2]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @TilingForDWConvSEP {
    config.Resources 4 of @NCE at 6.000000e+02 MHz

// CHECK-LABEL: @DWConvWithSEPSOK
func.func @DWConvWithSEPSOK(%arg0: tensor<1x288x1x1xf16, {order = #NHWC}>) -> tensor<1x288x2x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<288x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<288x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %sparsity_map = const.Declare tensor<1x288x2x2xi1> = dense<1> : tensor<1x288x2x2xi1>

    %storage_element = VPU.StorageElementTable {
        dataElemType = f16,
        seDepth = 18, seSize = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
        dataShape = [1, 288, 1, 1],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 288, 2, 2]>
    } -> tensor<1x18x2x2xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 288, 2, 2]>
    } -> !VPU.SparseTensor<data=tensor<1x288x1x1xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x288x2x2xi1>,
                           storage_element_table=tensor<1x18x2x2xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 288, 2, 2]>>

    %interpolate = VPU.NCE.DepthConvolution(%input, %weights) rawFilterShape [288, 1, 1, 1] {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        
        strides = [1, 1]
    } -> tensor<1x288x2x2xf16, {order = #NHWC}>

    return %interpolate : tensor<1x288x2x2xf16, {order = #NHWC}>

    // To satisfy DW.Conv + SEP requirements for workload channels, the op is
    // tiled into 2 slices, each with 144 channels; each individual op will then
    // be multiclustered with [64, 32, 32, 16] channels/cluster

    // CHECK:       VPU.NCE.DepthConvolution
    // CHECK-SAME:     tilingStrategy = [1, 2, 1, 1]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DWConvWithSEPNoMC
func.func @DWConvWithSEPNoMC(%arg0: tensor<1x288x1x1xf16, {order = #NHWC}>) -> tensor<1x288x2x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<288x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<288x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %sparsity_map = const.Declare tensor<1x288x2x2xi1> = dense<1> : tensor<1x288x2x2xi1>

    %storage_element = VPU.StorageElementTable {
        dataElemType = f16,
        seDepth = 18, seSize = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
        dataShape = [1, 288, 1, 1],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 288, 2, 2]>
    } -> tensor<1x18x2x2xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 288, 2, 2]>
    } -> !VPU.SparseTensor<data=tensor<1x288x1x1xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x288x2x2xi1>,
                           storage_element_table=tensor<1x18x2x2xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 288, 2, 2]>>

    %interpolate = VPU.NCE.DepthConvolution(%input, %weights) rawFilterShape [288, 1, 1, 1] {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        
        strides = [1, 1]
    } -> tensor<1x288x2x2xf16, {order = #NHWC}>

    return %interpolate : tensor<1x288x2x2xf16, {order = #NHWC}>

    // To satisfy DW.Conv + SEP requirements for workload channels, the op is
    // tiled into 5 slices on channels, with division:
    // [64, 64, 64, 64, 32]

    // CHECK:       VPU.NCE.DepthConvolution
    // CHECK-SAME:     tilingStrategy = [1, 5, 1, 1]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {

config.Resources 3 of @NCE {
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @SplitNCEMaxPoolWithBigC
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x5120x32x4xf16, {order = #NHWC}>)
func.func @SplitNCEMaxPoolWithBigC(%arg0: tensor<1x5120x32x4xf16, {order = #NHWC}>) -> tensor<1x5120x32x4xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
    } -> tensor<1x5120x32x4xf16, {order = #NHWC}>

    return %0 : tensor<1x5120x32x4xf16, {order = #NHWC}>

    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[ARG_0]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:      } -> tensor<1x5120x32x4xf16, {order = #NHWC}>

    // CHECK:       return [[MAXPOOL]] : tensor<1x5120x32x4xf16, {order = #NHWC}>
}

}
