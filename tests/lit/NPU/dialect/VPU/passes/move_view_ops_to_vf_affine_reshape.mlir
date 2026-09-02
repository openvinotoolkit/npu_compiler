//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --move-view-ops-to-vf="workload-management-mode=FWLM_V1_PAGES" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// AffineReshape between two VFs with tiling on C (dim 1).
// dim_mapping [[0], [1], [2, 3], [3]] splits H into H and W, but C (dim 1) is a simple 1:1 mapping.
// Since onlySupportPartialTilingDims is true (dims 2,3 not simple), v2 rewriter handles this.
// Parent VF has compatible tiling on C -> AffineReshape is moved into consumer VF.

// CHECK-LABEL: @MoveAffineReshapeBetweenVFs
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x128x256x1xf16, {order = #NHWC}>
func.func @MoveAffineReshapeBetweenVFs(%arg0: tensor<1x128x256x1xf16, {order = #NHWC}>) -> tensor<1x128x64x4xf16, {order = #NHWC}> {
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x128x256x1xf16, {order = #NHWC}>,
                             %arg0 as %arg2: tensor<1x128x256x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x128x256x1xf16, {order = #NHWC}> {
      %inner = VPU.NCE.Eltwise(%arg1, %arg2)
          {resultSegmentSizes = array<i32: 1, 0, 0, 0>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
          -> tensor<1x128x256x1xf16, {order = #NHWC}>
      VPU.Yield %inner
    }
    // AffineReshape: 1x128x256x1 -> 1x128x64x4 (H splits into H and W, C=128 preserved)
    // dim_mapping: [[0], [1], [2, 3], [3]]
    // C (dim 1): 128 -> 128, simple 1:1 mapping -> supported for tiling
    // H (dim 2): splits into output dims [2, 3] -> NOT simple
    %1 = VPU.AffineReshape(%0) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 128, 64, 4]}
        : tensor<1x128x256x1xf16, {order = #NHWC}> -> tensor<1x128x64x4xf16, {order = #NHWC}>
    %2 = VPU.VerticalFusion (%1 as %arg1: tensor<1x128x64x4xf16, {order = #NHWC}>,
                             %1 as %arg2: tensor<1x128x64x4xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x128x64x4xf16, {order = #NHWC}> {
      %inner = VPU.NCE.Eltwise(%arg1, %arg2)
          {resultSegmentSizes = array<i32: 1, 0, 0, 0>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
          -> tensor<1x128x64x4xf16, {order = #NHWC}>
      VPU.Yield %inner
    }
    return %2 : tensor<1x128x64x4xf16, {order = #NHWC}>

    //CHECK:  [[VF0:%.+]] = VPU.VerticalFusion ([[INPUT]]
    //CHECK:  [[VF1:%.+]] = VPU.VerticalFusion ([[VF0]] as [[ARG1:%[^:]+]]: tensor<1x128x256x1xf16, {order = #NHWC}>)
    //CHECK:  [[RESHAPE:%.+]] = VPU.AffineReshape([[ARG1]])
    //CHECK:  [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[RESHAPE]], [[RESHAPE]])
    //CHECK:  VPU.Yield [[ELTWISE]]
    //CHECK:  return [[VF1]] : tensor<1x128x64x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Same AffineReshape pattern but parent VF has no tiling (tilingStrategy=[1,1,1,1]).
// For a regular activation input, the v2 rewriter keeps the AffineReshape outside the consumer VF because there are no
// parent tiling dims to prove compatible with the restricted AffineReshape mapping.

// CHECK-LABEL: @NoMoveAffineReshapeWithNoTilingParent
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x128x256x1xf16, {order = #NHWC}>
func.func @NoMoveAffineReshapeWithNoTilingParent(%arg0: tensor<1x128x256x1xf16, {order = #NHWC}>) -> tensor<1x128x64x4xf16, {order = #NHWC}> {
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x128x256x1xf16, {order = #NHWC}>,
                             %arg0 as %arg2: tensor<1x128x256x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x128x256x1xf16, {order = #NHWC}> {
      %inner = VPU.NCE.Eltwise(%arg1, %arg2)
          {resultSegmentSizes = array<i32: 1, 0, 0, 0>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
          -> tensor<1x128x256x1xf16, {order = #NHWC}>
      VPU.Yield %inner
    }
    %1 = VPU.AffineReshape(%0) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 128, 64, 4]}
        : tensor<1x128x256x1xf16, {order = #NHWC}> -> tensor<1x128x64x4xf16, {order = #NHWC}>
    %2 = VPU.VerticalFusion (%1 as %arg1: tensor<1x128x64x4xf16, {order = #NHWC}>,
                             %1 as %arg2: tensor<1x128x64x4xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x128x64x4xf16, {order = #NHWC}> {
      %inner = VPU.NCE.Eltwise(%arg1, %arg2)
          {resultSegmentSizes = array<i32: 1, 0, 0, 0>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
          -> tensor<1x128x64x4xf16, {order = #NHWC}>
      VPU.Yield %inner
    }
    return %2 : tensor<1x128x64x4xf16, {order = #NHWC}>

    //CHECK:  [[VF0:%.+]] = VPU.VerticalFusion ([[INPUT]]
    //CHECK:  [[RESHAPE:%.+]] = VPU.AffineReshape([[VF0]])
    //CHECK:  [[VF1:%.+]] = VPU.VerticalFusion ([[RESHAPE]] as [[ARG1:%[^:]+]]: tensor<1x128x64x4xf16, {order = #NHWC}>, [[RESHAPE]] as [[ARG2:%[^:]+]]: tensor<1x128x64x4xf16, {order = #NHWC}>)
    //CHECK:  [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[ARG1]], [[ARG2]])
    //CHECK:  VPU.Yield [[ELTWISE]]
    //CHECK:  return [[VF1]] : tensor<1x128x64x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// AffineReshape between two VFs with tiling on H (dim 2).
// dim_mapping [[0], [1], [2, 3], [3]]: input H=256 splits to output H=64 and W=4.
// Output dim 2 is the "split outer dim" — the outermost output dim of the split.
// With split/merge tiling support, tiling on H propagates through the split:
//   output tile H=32 -> input tile H=32*4=128 (ratio = 256/64 = 4).
// Parent VF tiles on H -> v2 rewriter moves AffineReshape into consumer VF.

// CHECK-LABEL: @MoveAffineReshapeWithSplitOuterDimTiling
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x128x256x1xf16, {order = #NHWC}>
func.func @MoveAffineReshapeWithSplitOuterDimTiling(%arg0: tensor<1x128x256x1xf16, {order = #NHWC}>) -> tensor<1x128x64x4xf16, {order = #NHWC}> {
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x128x256x1xf16, {order = #NHWC}>,
                             %arg0 as %arg2: tensor<1x128x256x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x128x256x1xf16, {order = #NHWC}> {
      %inner = VPU.NCE.Eltwise(%arg1, %arg2)
          {resultSegmentSizes = array<i32: 1, 0, 0, 0>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
          -> tensor<1x128x256x1xf16, {order = #NHWC}>
      VPU.Yield %inner
    }
    // AffineReshape: 1x128x256x1 -> 1x128x64x4 (H splits into H and W)
    // Tiling on output dim 2 (H=64): split outer dim, ratio=4
    %1 = VPU.AffineReshape(%0) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 128, 64, 4]}
        : tensor<1x128x256x1xf16, {order = #NHWC}> -> tensor<1x128x64x4xf16, {order = #NHWC}>
    %2 = VPU.VerticalFusion (%1 as %arg1: tensor<1x128x64x4xf16, {order = #NHWC}>,
                             %1 as %arg2: tensor<1x128x64x4xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x128x64x4xf16, {order = #NHWC}> {
      %inner = VPU.NCE.Eltwise(%arg1, %arg2)
          {resultSegmentSizes = array<i32: 1, 0, 0, 0>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
          -> tensor<1x128x64x4xf16, {order = #NHWC}>
      VPU.Yield %inner
    }
    return %2 : tensor<1x128x64x4xf16, {order = #NHWC}>

    //CHECK:  [[VF0:%.+]] = VPU.VerticalFusion ([[INPUT]]
    //CHECK:  [[VF1:%.+]] = VPU.VerticalFusion ([[VF0]] as [[ARG1:%[^:]+]]: tensor<1x128x256x1xf16, {order = #NHWC}>)
    //CHECK:  [[RESHAPE:%.+]] = VPU.AffineReshape([[ARG1]])
    //CHECK:  [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[RESHAPE]], [[RESHAPE]])
    //CHECK:  VPU.Yield [[ELTWISE]]
    //CHECK:  return [[VF1]] : tensor<1x128x64x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// AffineReshape between two VFs with multi-dim tiling on both C (dim 1) and H (dim 2).
// dim_mapping [[0], [1], [2, 3], [3]]: input H=256 splits to output H=64 and W=4, C=128 is 1:1.
// C (dim 1) is a simple 1:1 mapping -> supported.
// H (dim 2) is the split outer dim -> supported.
// Both dims can be tiled simultaneously.
// Parent VF tiles on C and H -> v2 rewriter moves AffineReshape into consumer VF.

// CHECK-LABEL: @MoveAffineReshapeWithMultiDimTiling
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x128x256x1xf16, {order = #NHWC}>
func.func @MoveAffineReshapeWithMultiDimTiling(%arg0: tensor<1x128x256x1xf16, {order = #NHWC}>) -> tensor<1x128x64x4xf16, {order = #NHWC}> {
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x128x256x1xf16, {order = #NHWC}>,
                             %arg0 as %arg2: tensor<1x128x256x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 2, 2, 1]} -> tensor<1x128x256x1xf16, {order = #NHWC}> {
      %inner = VPU.NCE.Eltwise(%arg1, %arg2)
          {resultSegmentSizes = array<i32: 1, 0, 0, 0>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
          -> tensor<1x128x256x1xf16, {order = #NHWC}>
      VPU.Yield %inner
    }
    // AffineReshape: 1x128x256x1 -> 1x128x64x4 (H splits into H and W, C=128 preserved)
    // Multi-dim tiling: C=2 (simple) + H=2 (split outer)
    // Input tiling: C=2 (direct), H=2 (scaled by ratio=4 during tile inference)
    %1 = VPU.AffineReshape(%0) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 128, 64, 4]}
        : tensor<1x128x256x1xf16, {order = #NHWC}> -> tensor<1x128x64x4xf16, {order = #NHWC}>
    %2 = VPU.VerticalFusion (%1 as %arg1: tensor<1x128x64x4xf16, {order = #NHWC}>,
                             %1 as %arg2: tensor<1x128x64x4xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 2, 2, 1]} -> tensor<1x128x64x4xf16, {order = #NHWC}> {
      %inner = VPU.NCE.Eltwise(%arg1, %arg2)
          {resultSegmentSizes = array<i32: 1, 0, 0, 0>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
          -> tensor<1x128x64x4xf16, {order = #NHWC}>
      VPU.Yield %inner
    }
    return %2 : tensor<1x128x64x4xf16, {order = #NHWC}>

    //CHECK:  [[VF0:%.+]] = VPU.VerticalFusion ([[INPUT]]
    //CHECK:  [[VF1:%.+]] = VPU.VerticalFusion ([[VF0]] as [[ARG1:%[^:]+]]: tensor<1x128x256x1xf16, {order = #NHWC}>)
    //CHECK:  [[RESHAPE:%.+]] = VPU.AffineReshape([[ARG1]])
    //CHECK:  [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[RESHAPE]], [[RESHAPE]])
    //CHECK:  VPU.Yield [[ELTWISE]]
    //CHECK:  return [[VF1]] : tensor<1x128x64x4xf16, {order = #NHWC}>
}

// -----

// AffineReshape between two VFs with tiling on H (dim 2) which is a "split inner with outer=1" pattern.
// dim_mapping [[0], [1, 2], [3], [3]]: input C=320 splits to output C=1 and H=320,
// and input H=128, W=2 merge into output W=256.
// Output H (dim 2) is the inner split dim, but outer C (dim 1) = 1.
// When outer=1, tiling on inner dim is equivalent to direct tiling:
//   inputOffset = outerOffset * innerSize + innerOffset = 0 * 320 + innerOffset = innerOffset.
// Parent VF tiles on C [1, 2, 1, 1] (maps to output H).
// backInferTilingStrategy([1, 1, 2, 1]) -> [1, 2, 1, 1] matches parent -> move into consumer VF.
// Uses NCHW layout with SoftMax because dim_mapping [[0],[1,2],[3],[3]]
// infers NCHW output from NCHW input.

// CHECK-LABEL: @MoveAffineReshapeWithSplitInnerOuterOne
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x320x128x2xf16>
func.func @MoveAffineReshapeWithSplitInnerOuterOne(%arg0: tensor<1x320x128x2xf16>) -> tensor<1x1x320x256xf16> {
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x320x128x2xf16>)
        attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x320x128x2xf16> {
      %inner = VPU.SoftMax(%arg1) {axisInd = 3}
          : tensor<1x320x128x2xf16> -> tensor<1x320x128x2xf16>
      VPU.Yield %inner
    }
    // AffineReshape: 1x320x128x2 -> 1x1x320x256
    // dim_mapping [[0], [1, 2], [3], [3]]: C=320 splits to C'=1 and H'=320, H*W merge to W'=256
    // Tiling on output H (dim 2 = 320): split inner with outer=1 (C'=1), direct transfer
    %1 = VPU.AffineReshape(%0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 1, 320, 256]}
        : tensor<1x320x128x2xf16> -> tensor<1x1x320x256xf16>
    %2 = VPU.VerticalFusion (%1 as %arg1: tensor<1x1x320x256xf16>)
        attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x1x320x256xf16> {
      %inner = VPU.SoftMax(%arg1) {axisInd = 3}
          : tensor<1x1x320x256xf16> -> tensor<1x1x320x256xf16>
      VPU.Yield %inner
    }
    return %2 : tensor<1x1x320x256xf16>

    //CHECK:  [[VF0:%.+]] = VPU.VerticalFusion ([[INPUT]]
    //CHECK:  [[VF1:%.+]] = VPU.VerticalFusion ([[VF0]] as [[ARG1:%[^:]+]]: tensor<1x320x128x2xf16>)
    //CHECK:  [[RESHAPE:%.+]] = VPU.AffineReshape([[ARG1]])
    //CHECK:  [[SOFTMAX:%.+]] = VPU.SoftMax([[RESHAPE]])
    //CHECK:  VPU.Yield [[SOFTMAX]]
    //CHECK:  return [[VF1]] : tensor<1x1x320x256xf16>
}

// -----

// 256x2048x16x1x1 -> 1x256x2048x16 with dim_mapping [[0, 1], [2], [3], [3], [3]].
// Input dim 0 splits to output dims 0 and 1, but output dim 0 has size 1 so VF strategy propagation
// should prefer output dim 1 as the meaningful tiling dim.
// Parent VF tiles on input dim 0. Consumer VF tiles on output dim 1.

// CHECK-LABEL: @MoveAffineReshapeRankChangingSplitOuterOne
// CHECK-SAME:      [[INPUT:%.+]]: tensor<256x2048x16x1x1xf16>
func.func @MoveAffineReshapeRankChangingSplitOuterOne(%arg0: tensor<256x2048x16x1x1xf16>) -> tensor<1x256x2048x16xf16> {
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<256x2048x16x1x1xf16>)
        attributes {tilingStrategy = [2, 1, 1, 1, 1]} -> tensor<256x2048x16x1x1xf16> {
      %inner = VPU.SoftMax(%arg1) {axisInd = 2}
          : tensor<256x2048x16x1x1xf16> -> tensor<256x2048x16x1x1xf16>
      VPU.Yield %inner
    }
    %1 = VPU.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3], [3], [3]], shape_value = [1, 256, 2048, 16]}
        : tensor<256x2048x16x1x1xf16> -> tensor<1x256x2048x16xf16>
    %2 = VPU.VerticalFusion (%1 as %arg1: tensor<1x256x2048x16xf16>)
        attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x256x2048x16xf16> {
      %inner = VPU.SoftMax(%arg1) {axisInd = 2}
          : tensor<1x256x2048x16xf16> -> tensor<1x256x2048x16xf16>
      VPU.Yield %inner
    }
    return %2 : tensor<1x256x2048x16xf16>

    // CHECK: [[VF0:%.+]] = VPU.VerticalFusion ([[INPUT]]
    // CHECK: [[VF1:%.+]] = VPU.VerticalFusion ([[VF0]] as [[ARG1:%[^:]+]]: tensor<256x2048x16x1x1xf16>
    // CHECK: [[RESHAPE:%.+]] = VPU.AffineReshape([[ARG1]])
    // CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[RESHAPE]])
    // CHECK: VPU.Yield [[SOFTMAX]]
    // CHECK: return [[VF1]] : tensor<1x256x2048x16xf16>
}

// -----

// 1x256x2048x16 -> 256x2048x16x1x1 with dim_mapping [[0], [0], [1], [2, 3, 4]].
// This is the reverse of the 5D->4D case above: lower rank (4D) reshaped to higher rank (5D).
// Input dims 0,1 merge to output dim 0 (N=1*C=256 -> 256).
// Input dim 2 is simple 1:1 to output dim 1 (2048).
// Input dim 3 splits to output dims 2,3,4 (16 -> 16,1,1).
// Parent VF tiles on input C (dim 1). inferTilingStrategy maps it to output merge dim 0.

// CHECK-LABEL: @MoveAffineReshapeRankChanging4Dto5DMerge
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x256x2048x16xf16>
func.func @MoveAffineReshapeRankChanging4Dto5DMerge(%arg0: tensor<1x256x2048x16xf16>) -> tensor<256x2048x16x1x1xf16> {
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x256x2048x16xf16>)
        attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x256x2048x16xf16> {
      %inner = VPU.SoftMax(%arg1) {axisInd = 2}
          : tensor<1x256x2048x16xf16> -> tensor<1x256x2048x16xf16>
      VPU.Yield %inner
    }
    %1 = VPU.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [256, 2048, 16, 1, 1]}
        : tensor<1x256x2048x16xf16> -> tensor<256x2048x16x1x1xf16>
    %2 = VPU.VerticalFusion (%1 as %arg1: tensor<256x2048x16x1x1xf16>)
        attributes {tilingStrategy = [2, 1, 1, 1, 1]} -> tensor<256x2048x16x1x1xf16> {
      %inner = VPU.SoftMax(%arg1) {axisInd = 2}
          : tensor<256x2048x16x1x1xf16> -> tensor<256x2048x16x1x1xf16>
      VPU.Yield %inner
    }
    return %2 : tensor<256x2048x16x1x1xf16>

    // CHECK: [[VF0:%.+]] = VPU.VerticalFusion ([[INPUT]]
    // CHECK: [[VF1:%.+]] = VPU.VerticalFusion ([[VF0]] as [[ARG1:%[^:]+]]: tensor<1x256x2048x16xf16>
    // CHECK: [[RESHAPE:%.+]] = VPU.AffineReshape([[ARG1]])
    // CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[RESHAPE]])
    // CHECK: VPU.Yield [[SOFTMAX]]
    // CHECK: return [[VF1]] : tensor<256x2048x16x1x1xf16>
}

// -----

// Merge with trailing ones: 256x1x16x512x4 -> 256x1x16x2048x1.
// dim_mapping [[0], [1], [2], [3], [3, 4]]:
//   in_d3(512) and in_d4(4) merge into out_d3(2048), in_d4 also fans out to out_d4(=1).
// Old isMergeDim rejected this because in_d4 maps to two output dims (fan-out).
// New isMergeDim allows it because the extra output dim (d4) has size 1.
// Tiling on out_d3 (2048): targetDimIdx=3 (in_d3=512), otherProduct=4 (in_d4).
//   output tile d3=1024 -> input tile d3=1024/4=256, d4 stays 4.
// Parent VF tiles on d3 [1,1,1,2,1] -> backInfer matches -> move into consumer VF.

#order5d = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: @MoveAffineReshapeMergeWithTrailingOnes
// CHECK-SAME:      [[INPUT:%.+]]: tensor<256x1x16x512x4xf16, {order = #GNHWC}>
func.func @MoveAffineReshapeMergeWithTrailingOnes(%arg0: tensor<256x1x16x512x4xf16, {order = #order5d}>) -> tensor<256x1x16x2048x1xf16, {order = #order5d}> {
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<256x1x16x512x4xf16, {order = #order5d}>)
        attributes {tilingStrategy = [1, 1, 1, 2, 1]} -> tensor<256x1x16x512x4xf16, {order = #order5d}> {
      %inner = VPU.SoftMax(%arg1) {axisInd = 2}
          : tensor<256x1x16x512x4xf16, {order = #order5d}> -> tensor<256x1x16x512x4xf16, {order = #order5d}>
      VPU.Yield %inner
    }
    // AffineReshape: 256x1x16x512x4 -> 256x1x16x2048x1
    // in_d3(512) + in_d4(4) merge to out_d3(2048), in_d4 fans to out_d4(=1)
    %1 = VPU.AffineReshape(%0) {dim_mapping = [[0], [1], [2], [3], [3, 4]], shape_value = [256, 1, 16, 2048, 1]}
        : tensor<256x1x16x512x4xf16, {order = #order5d}> -> tensor<256x1x16x2048x1xf16, {order = #order5d}>
    %2 = VPU.VerticalFusion (%1 as %arg1: tensor<256x1x16x2048x1xf16, {order = #order5d}>)
        attributes {tilingStrategy = [1, 1, 1, 2, 1]} -> tensor<256x1x16x2048x1xf16, {order = #order5d}> {
      %inner = VPU.SoftMax(%arg1) {axisInd = 2}
          : tensor<256x1x16x2048x1xf16, {order = #order5d}> -> tensor<256x1x16x2048x1xf16, {order = #order5d}>
      VPU.Yield %inner
    }
    return %2 : tensor<256x1x16x2048x1xf16, {order = #order5d}>

    // CHECK: [[VF0:%.+]] = VPU.VerticalFusion ([[INPUT]]
    // CHECK: [[VF1:%.+]] = VPU.VerticalFusion ([[VF0]] as [[ARG1:%[^:]+]]: tensor<256x1x16x512x4xf16, {order = #GNHWC}>)
    // CHECK: [[RESHAPE:%.+]] = VPU.AffineReshape([[ARG1]])
    // CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[RESHAPE]])
    // CHECK: VPU.Yield [[SOFTMAX]]
    // CHECK: return [[VF1]] : tensor<256x1x16x2048x1xf16, {order = #GNHWC}>
}
