//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --merge-vertical-fusion-subgraphs="tiling-mode=ISOLATED workload-management-mode=FWLM_V1_PAGES" %s | FileCheck %s
// REQUIRES: platform-NPU5010

// Subgraph extracted from human-pose-estimation-0001 (around %conv4_3_CPM2Fres):
//
//   %al  = Conv -> Elu
//          |          \
//          |           Conv -> Elu -> DepthConv(3x3) -> Elu -> Conv -> Elu = %inc
//          |                                                              /
//          +------ NCE.Eltwise(%al, %inc) ---------------------------------
//
// During pattern-based VF merge (FWLM v2 rewriter) the long %inc-chain is
// greedily merged into a single VF that also absorbs %al. When the eltwise VF
// is then evaluated for merging, the closest parent inside the merged VF is a
// VPU.Elu (SW op). Without the NCEOpInterface guard in
// VFScheduling::getInternalSliceCopyCost the chain walk pushes the SW op into
// `chain` and the subsequent `mlir::cast<VPU::NCEOpInterface>(parent)` aborts.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @BuildSubgraphEltwiseWithSwParent
func.func @BuildSubgraphEltwiseWithSwParent(%arg0: tensor<1x128x36x64xf16, {order = #NHWC}>) -> tensor<1x128x36x64xf16, {order = #NHWC}> {
    %cst_w_al  = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<128x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_w_red = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<128x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_w_dw  = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}>  = dense<1.0> : tensor<128x16x1x1xf16>,  [#const.Reorder<#NHWC>]
    %cst_w_inc = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<128x128x1x1xf16>, [#const.Reorder<#NHWC>]

    // %al = Conv -> Elu  (left branch root)
    %conv_al = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x128x36x64xf16, {order = #NHWC}>, %cst_w_al as %arg2: tensor<128x128x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x128x36x64xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2)  rawFilterShape [128, 128, 1, 1]
        {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         ppe = #VPU.PPEStub<>, strides = [1, 1]}
        : tensor<1x128x36x64xf16, {order = #NHWC}>, tensor<128x128x1x1xf16, {order = #NHWC}>
        -> tensor<1x128x36x64xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    %al = VPU.VerticalFusion (%conv_al as %arg1: tensor<1x128x36x64xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x128x36x64xf16, {order = #NHWC}> {
      %3 = VPU.Elu(%arg1) {x = 1.000000e+00 : f64} : tensor<1x128x36x64xf16, {order = #NHWC}> -> tensor<1x128x36x64xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    // Right branch: Conv -> Elu -> DepthConv(3x3) -> Elu -> Conv -> Elu = %inc
    %conv_red = VPU.VerticalFusion (%al as %arg1: tensor<1x128x36x64xf16, {order = #NHWC}>, %cst_w_red as %arg2: tensor<128x128x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x128x36x64xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2)  rawFilterShape [128, 128, 1, 1]
        {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         ppe = #VPU.PPEStub<>,strides = [1, 1]}
        : tensor<1x128x36x64xf16, {order = #NHWC}>, tensor<128x128x1x1xf16, {order = #NHWC}>
        -> tensor<1x128x36x64xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    %red = VPU.VerticalFusion (%conv_red as %arg1: tensor<1x128x36x64xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x128x36x64xf16, {order = #NHWC}> {
      %3 = VPU.Elu(%arg1) {x = 1.000000e+00 : f64} : tensor<1x128x36x64xf16, {order = #NHWC}> -> tensor<1x128x36x64xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    // 3x3 DepthConv -- KY/KX>1 satisfies isSliceOpNeeded inside chain walk.
    %dw = VPU.VerticalFusion (%red as %arg1: tensor<1x128x36x64xf16, {order = #NHWC}>, %cst_w_dw as %arg2: tensor<128x16x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x128x36x64xf16, {order = #NHWC}> {
      %3 = VPU.NCE.DepthConvolution(%arg1, %arg2)  rawFilterShape [128, 1, 3, 3]
        {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         ppe = #VPU.PPEStub<>, strides = [1, 1]}
        -> tensor<1x128x36x64xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    %relu_dw = VPU.VerticalFusion (%dw as %arg1: tensor<1x128x36x64xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x128x36x64xf16, {order = #NHWC}> {
      %3 = VPU.Elu(%arg1) {x = 1.000000e+00 : f64} : tensor<1x128x36x64xf16, {order = #NHWC}> -> tensor<1x128x36x64xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    %conv_inc = VPU.VerticalFusion (%relu_dw as %arg1: tensor<1x128x36x64xf16, {order = #NHWC}>, %cst_w_inc as %arg2: tensor<128x128x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x128x36x64xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [128, 128, 1, 1] 
        {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         ppe = #VPU.PPEStub<>, strides = [1, 1]}
        : tensor<1x128x36x64xf16, {order = #NHWC}>, tensor<128x128x1x1xf16, {order = #NHWC}>
        -> tensor<1x128x36x64xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    %inc = VPU.VerticalFusion (%conv_inc as %arg1: tensor<1x128x36x64xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x128x36x64xf16, {order = #NHWC}> {
      %3 = VPU.Elu(%arg1) {x = 1.000000e+00 : f64} : tensor<1x128x36x64xf16, {order = #NHWC}> -> tensor<1x128x36x64xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    // NCE.Eltwise(%al, %inc) -- closest parent is %inc (SW Elu).
    %res = VPU.VerticalFusion (%al as %arg1: tensor<1x128x36x64xf16, {order = #NHWC}>, %inc as %arg2: tensor<1x128x36x64xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x128x36x64xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Eltwise(%arg1, %arg2)
        {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
        -> tensor<1x128x36x64xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    return %res : tensor<1x128x36x64xf16, {order = #NHWC}>

    //CHECK:      VPU.NCE.Eltwise
    //CHECK:      return
}
