//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --scf-vertical-fusion="vf-merge-configuration=GREEDY" --resolve-shaped-type-result-dims --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// Verify that registering the SCF Expand tiling interface for single-dim
// end-padded `VPU.Expand` lets SCF Vertical Fusion pull the Expand *inside* the
// per-tile `scf.for` body produced for the consumer NCE chain. This is the key
// HostCompile / NPU40XX / NPU5010 enabler needed to remove the residual Expand
// barriers in dynamic networks.
//
// Strategy:
//   output tile [..., dim_pad_full, ..., dim_tile_t, ...]
//   -> input tile [..., dim_pad_orig, ..., dim_tile_t, ...]
//   pads_end on the padded dim is preserved (Expand re-pads inside the tile body).
//   The padded dim must remain untiled (full extent, zero offset).

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ChannelOnlyExpandFusedIntoSCFLoop
func.func @ChannelOnlyExpandFusedIntoSCFLoop(%arg0: tensor<1x3x540x960xf16, {order = #NHWC}>)
        -> tensor<1x16x540x960xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst1 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %expand = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}
        : tensor<1x3x540x960xf16, {order = #NHWC}> -> tensor<1x16x540x960xf16, {order = #NHWC}>

    %0 = VPU.NCE.Convolution(%expand, %cst0) rawFilterShape [16, 16, 3, 3] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            
            strides = [1, 1],
            tilingStrategy = [1, 1, 1, 22]
        } : tensor<1x16x540x960xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>
              -> tensor<1x16x540x960xf16, {order = #NHWC}>

    %1 = VPU.NCE.Convolution(%0, %cst1) rawFilterShape [16, 16, 3, 3] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            
            strides = [1, 1],
            tilingStrategy = [1, 1, 1, 22]
        } : tensor<1x16x540x960xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>
              -> tensor<1x16x540x960xf16, {order = #NHWC}>

    return %1 : tensor<1x16x540x960xf16, {order = #NHWC}>

    // The channel-only `VPU.Expand` must end up *inside* the `scf.for` body
    // produced by SCF VF. The Expand keeps `pads_end = [0, 13, 0, 0]` because the
    // C dim is materialized in full per tile; only H/W are sliced.

    // CHECK: scf.for
    // CHECK:   VPU.Expand
    // CHECK-SAME: pads_end = [0, 13, 0, 0]
    // CHECK:   VPU.NCE.Convolution
    // CHECK:   VPU.NCE.Convolution
    // CHECK:   tensor.insert_slice
    // CHECK:   scf.yield
}

// -----

// Negative case: Expand uses a non-zero `pads_begin`, which is rejected by the
// single-dim end-padding helper regardless of the tile axis. The Expand must NOT
// be pulled inside the loop and must remain outside.

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SpatialPaddingExpandStaysOutsideSCFLoop
func.func @SpatialPaddingExpandStaysOutsideSCFLoop(%arg0: tensor<1x16x538x960xf16, {order = #NHWC}>)
        -> tensor<1x16x540x960xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst1 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %expand = VPU.Expand(%arg0) {pads_begin = [0, 0, 1, 0], pads_end = [0, 0, 1, 0]}
        : tensor<1x16x538x960xf16, {order = #NHWC}> -> tensor<1x16x540x960xf16, {order = #NHWC}>

    %0 = VPU.NCE.Convolution(%expand, %cst0) rawFilterShape [16, 16, 3, 3] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            
            strides = [1, 1],
            tilingStrategy = [1, 1, 1, 22]
        } : tensor<1x16x540x960xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>
              -> tensor<1x16x540x960xf16, {order = #NHWC}>

    %1 = VPU.NCE.Convolution(%0, %cst1) rawFilterShape [16, 16, 3, 3] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            
            strides = [1, 1],
            tilingStrategy = [1, 1, 1, 22]
        } : tensor<1x16x540x960xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>
              -> tensor<1x16x540x960xf16, {order = #NHWC}>

    return %1 : tensor<1x16x540x960xf16, {order = #NHWC}>

    // The H-padded-with-begin Expand has `pads_begin = [0, 0, 1, 0]` which fails the
    // single-dim end-padding helper (`getSinglePaddedExpandDim` requires `pads_begin == 0`).
    // It must therefore stay above the `scf.for` produced for the Conv chain rather than
    // be pulled into the loop body.

    // CHECK-NOT: scf.for
    // CHECK: VPU.Expand
    // CHECK-SAME: pads_begin = [0, 0, 1, 0]
    // CHECK-SAME: pads_end = [0, 0, 1, 0]
    // CHECK-NOT: VPU.Expand
}

// -----

// Positive case: Expand pads only on H (`pads_end = [0, 0, 2, 0]`,
// `pads_begin == 0`). The consumer chain is tiled on W (`tilingStrategy = [1, 1, 1, 22]`),
// i.e. tile axis != padded dim. The SCF Expand tiling model must accept this Expand and
// pull it inside the `scf.for` body. Inside the body the Expand keeps its original
// `pads_end = [0, 0, 2, 0]` because H is materialized in full per tile.

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @HOnlyExpandFusedIntoSCFLoopWhenTiledOnW
func.func @HOnlyExpandFusedIntoSCFLoopWhenTiledOnW(%arg0: tensor<1x16x538x960xf16, {order = #NHWC}>)
        -> tensor<1x16x540x960xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst1 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %expand = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 2, 0]}
        : tensor<1x16x538x960xf16, {order = #NHWC}> -> tensor<1x16x540x960xf16, {order = #NHWC}>

    %0 = VPU.NCE.Convolution(%expand, %cst0) rawFilterShape [16, 16, 3, 3] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            
            strides = [1, 1],
            tilingStrategy = [1, 1, 1, 22]
        } : tensor<1x16x540x960xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>
              -> tensor<1x16x540x960xf16, {order = #NHWC}>

    %1 = VPU.NCE.Convolution(%0, %cst1) rawFilterShape [16, 16, 3, 3] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            
            strides = [1, 1],
            tilingStrategy = [1, 1, 1, 22]
        } : tensor<1x16x540x960xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>
              -> tensor<1x16x540x960xf16, {order = #NHWC}>

    return %1 : tensor<1x16x540x960xf16, {order = #NHWC}>

    // CHECK: scf.for
    // CHECK:   VPU.Expand
    // CHECK-SAME: pads_end = [0, 0, 2, 0]
    // CHECK:   VPU.NCE.Convolution
    // CHECK:   VPU.NCE.Convolution
    // CHECK:   tensor.insert_slice
    // CHECK:   scf.yield
}

// -----

// Positive case: Expand pads only C, while the consumer chain already has a
// multi-dimensional tiling strategy on dynamic non-padded H and W. This must
// not block SCF VF from pulling the Expand into the generated loop body; C
// remains full in the tiled Expand and only non-padded dimensions may be sliced.

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!dynInputC3HW = tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
!dynExpandedC16HW = tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-LABEL: @ChannelOnlyExpandWithMultiDimNonPaddedTilingFusedIntoSCFLoop
// CHECK-SAME:    [[ARG0:%.+]]: tensor<1x3x?x?xf16
func.func @ChannelOnlyExpandWithMultiDimNonPaddedTilingFusedIntoSCFLoop(%arg0: !dynInputC3HW) -> !dynExpandedC16HW {
    %cst0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst1 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %expand = VPU.Expand(%arg0) {
            pads_begin = [0, 0, 0, 0],
            pads_end = [0, 13, 0, 0],
            tilingStrategy = [1, 1, 2, 2]
        } : !dynInputC3HW -> !dynExpandedC16HW

    %0 = VPU.NCE.Convolution(%expand, %cst0) rawFilterShape [16, 16, 3, 3] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            
            strides = [1, 1],
            tilingStrategy = [1, 1, 2, 2]
        } : !dynExpandedC16HW, tensor<16x16x3x3xf16, {order = #NHWC}> -> !dynExpandedC16HW

    %1 = VPU.NCE.Convolution(%0, %cst1) rawFilterShape [16, 16, 3, 3] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            
            strides = [1, 1],
            tilingStrategy = [1, 1, 2, 2]
        } : !dynExpandedC16HW, tensor<16x16x3x3xf16, {order = #NHWC}> -> !dynExpandedC16HW

    return %1 : !dynExpandedC16HW

    // CHECK-DAG: [[C2:%.+]] = arith.constant 2 : index
    // CHECK-DAG: [[C3:%.+]] = arith.constant 3 : index
    // CHECK: tensor.dim [[ARG0]], [[C2]]
    // CHECK: tensor.dim [[ARG0]], [[C3]]
    // CHECK-NOT: VPU.Expand
    // CHECK: scf.for
    // CHECK:   scf.for
    // CHECK:     VPU.Expand
    // CHECK-SAME: pads_end = [0, 13, 0, 0]
    // CHECK:     VPU.NCE.Convolution
    // CHECK:     VPU.NCE.Convolution
    // CHECK:     tensor.insert_slice
    // CHECK:     scf.yield
}

// -----

// Auto-retarget case: the user-supplied `tilingStrategy = [1, 1, 22, 1]` would
// tile on H, but H is the Expand-padded dim, so the SCF Expand tiling model rejects
// the H tile (`isSupportedOutTile` returns false because the tile cuts into the
// padded dim). The VF driver then picks a compatible axis from `getAllowedDims`
// (here W) and pulls the Expand into the per-tile body just like the W-tiled
// positive case above.

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @HOnlyExpandRetargetsWhenTiledOnH
func.func @HOnlyExpandRetargetsWhenTiledOnH(%arg0: tensor<1x16x538x960xf16, {order = #NHWC}>)
        -> tensor<1x16x540x960xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst1 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %expand = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 2, 0]}
        : tensor<1x16x538x960xf16, {order = #NHWC}> -> tensor<1x16x540x960xf16, {order = #NHWC}>

    %0 = VPU.NCE.Convolution(%expand, %cst0) rawFilterShape [16, 16, 3, 3] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            
            strides = [1, 1],
            tilingStrategy = [1, 1, 22, 1]
        } : tensor<1x16x540x960xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>
              -> tensor<1x16x540x960xf16, {order = #NHWC}>

    %1 = VPU.NCE.Convolution(%0, %cst1) rawFilterShape [16, 16, 3, 3] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            
            strides = [1, 1],
            tilingStrategy = [1, 1, 22, 1]
        } : tensor<1x16x540x960xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>
              -> tensor<1x16x540x960xf16, {order = #NHWC}>

    return %1 : tensor<1x16x540x960xf16, {order = #NHWC}>

    // The loop's upper bound is W (960), and inside the body H stays full on both
    // the input slice (538) and the output slice (540) while W is the dynamic dim
    // (`?`). Together these confirm VF retargeted from H to W.

    // CHECK-DAG: [[W_UB:%.+]] = arith.constant 960 : index
    // CHECK: scf.for {{%.+}} = {{%.+}} to [[W_UB]]
    // CHECK:   tensor.extract_slice
    // CHECK-SAME: to tensor<1x16x538x?xf16
    // CHECK:   VPU.Expand
    // CHECK-SAME: pads_end = [0, 0, 2, 0]
    // CHECK:   VPU.NCE.Convolution
    // CHECK:   VPU.NCE.Convolution
    // CHECK:   tensor.insert_slice
    // CHECK-SAME: tensor<1x16x540x?xf16
    // CHECK:   scf.yield
}

// -----

// Negative case: Expand pads *two* dimensions at the end (C and W). The single-dim
// end-padding helper `getSinglePaddedExpandDim` rejects any `pads_end` with more
// than one non-zero entry. Regardless of the tile axis chosen by the consumer, the
// Expand must stay above the `scf.for` produced for the Conv chain.

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MultiDimEndPaddingExpandStaysOutsideSCFLoop
func.func @MultiDimEndPaddingExpandStaysOutsideSCFLoop(%arg0: tensor<1x3x540x958xf16, {order = #NHWC}>)
        -> tensor<1x16x540x960xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst1 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %expand = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 2]}
        : tensor<1x3x540x958xf16, {order = #NHWC}> -> tensor<1x16x540x960xf16, {order = #NHWC}>

    %0 = VPU.NCE.Convolution(%expand, %cst0) rawFilterShape [16, 16, 3, 3] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            
            strides = [1, 1],
            tilingStrategy = [1, 1, 1, 22]
        } : tensor<1x16x540x960xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>
              -> tensor<1x16x540x960xf16, {order = #NHWC}>

    %1 = VPU.NCE.Convolution(%0, %cst1) rawFilterShape [16, 16, 3, 3] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            
            strides = [1, 1],
            tilingStrategy = [1, 1, 1, 22]
        } : tensor<1x16x540x960xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>
              -> tensor<1x16x540x960xf16, {order = #NHWC}>

    return %1 : tensor<1x16x540x960xf16, {order = #NHWC}>

    // CHECK-NOT: scf.for
    // CHECK: VPU.Expand
    // CHECK-SAME: pads_end = [0, 13, 0, 2]
    // CHECK-NOT: VPU.Expand
}

// -----

// Negative case: Expand pads C at the end but also pads a *non-padded* dim at the
// beginning (`pads_begin = [0, 0, 1, 0]`). The helper rejects any non-zero
// `pads_begin` entry, so the Expand stays above the loop even though the tile axis
// (W) is different from both the end-padded and begin-padded dims.

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PadsBeginOnNonPaddedDimStaysOutsideSCFLoop
func.func @PadsBeginOnNonPaddedDimStaysOutsideSCFLoop(%arg0: tensor<1x3x539x960xf16, {order = #NHWC}>)
        -> tensor<1x16x540x960xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst1 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %expand = VPU.Expand(%arg0) {pads_begin = [0, 0, 1, 0], pads_end = [0, 13, 0, 0]}
        : tensor<1x3x539x960xf16, {order = #NHWC}> -> tensor<1x16x540x960xf16, {order = #NHWC}>

    %0 = VPU.NCE.Convolution(%expand, %cst0) rawFilterShape [16, 16, 3, 3] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            
            strides = [1, 1],
            tilingStrategy = [1, 1, 1, 22]
        } : tensor<1x16x540x960xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>
              -> tensor<1x16x540x960xf16, {order = #NHWC}>

    %1 = VPU.NCE.Convolution(%0, %cst1) rawFilterShape [16, 16, 3, 3] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            
            strides = [1, 1],
            tilingStrategy = [1, 1, 1, 22]
        } : tensor<1x16x540x960xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>
              -> tensor<1x16x540x960xf16, {order = #NHWC}>

    return %1 : tensor<1x16x540x960xf16, {order = #NHWC}>

    // CHECK-NOT: scf.for
    // CHECK: VPU.Expand
    // CHECK-SAME: pads_begin = [0, 0, 1, 0]
    // CHECK-SAME: pads_end = [0, 13, 0, 0]
    // CHECK-NOT: VPU.Expand
}

// -----

// Positive case: only the *non-padded* dim (W) is dynamic on the input.
// `getSinglePaddedExpandDim` requires the padded dim (C) to be statically shaped,
// which it is here (input C=3 -> output C=16). The SCF Expand tiling model accepts the
// Expand because the tile axis (W) differs from the padded dim. VF tiles the
// consumer chain on the dynamic W using `tensor.dim` to derive the loop bound,
// and pulls the Expand inside the per-tile body.

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!dynInputC3 = tensor<1x3x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
!dynExpandedC16 = tensor<1x16x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-LABEL: @ChannelOnlyExpandWithDynamicWFusedIntoSCFLoop
// CHECK-SAME:    [[ARG0:%.+]]: tensor<1x3x540x?xf16
func.func @ChannelOnlyExpandWithDynamicWFusedIntoSCFLoop(%arg0: !dynInputC3) -> !dynExpandedC16 {
    %cst0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst1 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %expand = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}
        : !dynInputC3 -> !dynExpandedC16

    %0 = VPU.NCE.Convolution(%expand, %cst0) rawFilterShape [16, 16, 3, 3] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            
            strides = [1, 1],
            tilingStrategy = [1, 1, 1, 22]
        } : !dynExpandedC16, tensor<16x16x3x3xf16, {order = #NHWC}> -> !dynExpandedC16

    %1 = VPU.NCE.Convolution(%0, %cst1) rawFilterShape [16, 16, 3, 3] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            
            strides = [1, 1],
            tilingStrategy = [1, 1, 1, 22]
        } : !dynExpandedC16, tensor<16x16x3x3xf16, {order = #NHWC}> -> !dynExpandedC16

    return %1 : !dynExpandedC16

    // The loop bound is read from the input dynamic W. Inside the body, the input
    // slice keeps the full unpadded C (3) and full H (540), and the output slice
    // keeps the full padded C (16) and full H (540) - only W is dynamic,
    // confirming the W tile axis.

    // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[C3:%.+]] = arith.constant 3 : index
    // CHECK: tensor.dim [[ARG0]], [[C3]]
    // CHECK-NOT: VPU.Expand
    // CHECK: scf.for {{%.+}} = [[C0]] to {{%.+}} step
    // CHECK:   tensor.extract_slice [[ARG0]]
    // CHECK-SAME: to tensor<1x3x540x?xf16
    // CHECK:   VPU.Expand
    // CHECK-SAME: pads_end = [0, 13, 0, 0]
    // CHECK:   VPU.NCE.Convolution
    // CHECK:   VPU.NCE.Convolution
    // CHECK:   tensor.insert_slice
    // CHECK-SAME: tensor<1x16x540x?xf16
    // CHECK:   scf.yield
}

// -----

// Negative case: Expand pads C (single end-padded dim) but the *input* padded-dim
// extent is dynamic. `getSinglePaddedExpandDim` requires the padded dim to be
// statically shaped on the input — otherwise the per-tile body would need a
// `tensor.dim` on the un-materialized input and the SCF Expand tiling contract
// ("padded values produced by Expand inside the tile") no longer holds at
// compile time. Expand must stay above the loop.

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!dynInput = tensor<1x?x540x960xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
!dynExpanded = tensor<1x?x540x960xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-LABEL: @DynamicPaddedDimExpandStaysOutsideSCFLoop
func.func @DynamicPaddedDimExpandStaysOutsideSCFLoop(%arg0: !dynInput) -> !dynExpanded {
    %expand = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}
        : !dynInput -> !dynExpanded

    return %expand : !dynExpanded

    // The dynamic C on the input makes Expand unsupported for SCF VF regardless
    // of any downstream tile axis, so the Expand is preserved verbatim.
    // CHECK-NOT: scf.for
    // CHECK: VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}
}
