//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --move-view-ops-to-vf %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func.func @MoveAffineReshapeToRMSVF(%arg0: tensor<1x3072x1x1xf16, {order = #NHWC}>, %arg1: tensor<1x3072x1x1xf16, {order = #NHWC}>)
      -> (tensor<1x1x1x3072xf16>, tensor<1x1x1x3072xf16>) {
  %cst = const.Declare tensor<1x1x1x3072xf16> = dense<1.00776029> : tensor<3072xf32>, [#const.Reshape<[1, 1, 1, 3072]>, #const.CastElemType<f16>]

  %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x3072x1x1xf16, {order = #NHWC}>, %arg1 as %arg3: tensor<1x3072x1x1xf16, {order = #NHWC}>)
          attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x3072x1x1xf16, {order = #NHWC}> {
    %1 = VPU.NCE.Eltwise(%arg2, %arg3) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
      mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
      op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>
    } -> tensor<1x3072x1x1xf16, {order = #NHWC}>
    VPU.Yield %1
  }

  %2 = VPU.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NWCH}
      : tensor<1x3072x1x1xf16, {order = #NHWC}> -> tensor<1x3072x1x1xf16>
  %3 = VPU.AffineReshape(%2) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 3072]}
      : tensor<1x3072x1x1xf16> -> tensor<1x1x1x3072xf16>

  %4 = VPU.VerticalFusion (%3 as %arg2: tensor<1x1x1x3072xf16>, %cst as %arg3: tensor<1x1x1x3072xf16>)
      attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x1x1x3072xf16> {
    %5 = VPU.RMS(%arg2, %arg3) {eps = 9.9999997171806853E-10 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
        : tensor<1x1x1x3072xf16>, tensor<1x1x1x3072xf16> -> tensor<1x1x1x3072xf16>
    VPU.Yield %5
  }

  return %3, %4 : tensor<1x1x1x3072xf16>, tensor<1x1x1x3072xf16>

  // CHECK: [[VF0:%.+]] = VPU.VerticalFusion
  // CHECK:        VPU.NCE.Eltwise

  // CHECK: [[PERMUTE_CAST:%.+]] = VPU.PermuteCast([[VF0]])
  // CHECK: [[AFFINE_RESHAPE:%.+]] = VPU.AffineReshape([[PERMUTE_CAST]])

  // E-217971 AffineReshape must be merged into the VF block, even if it
  // will get unrolled later on, otherwise performance regression is observed.
  // Once root cause of regression is found, this could be cleaned up, with
  // AffineReshape being prevented from merging.

  // CHECK: [[VF1:%.+]] = VPU.VerticalFusion ([[PERMUTE_CAST]]
  // CHECK:        VPU.AffineReshape
  // CHECK:        VPU.RMS

  // CHECK: return [[AFFINE_RESHAPE]], [[VF1]]
}
