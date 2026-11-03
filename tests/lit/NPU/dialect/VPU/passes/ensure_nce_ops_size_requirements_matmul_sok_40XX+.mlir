//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% num-of-dpu-groups=4" --ensure-nce-ops-size-requirements --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// SOK-compatible NCEMatMulOp with no multiCluster strategy assigned yet. Output channels (16384)
// exceed VPU_DIMENSION_LIMIT (8192) on a single tile, but are covered once segmented across the
// 4 available clusters (16384 / 4 = 4096 <= 8192), so the channel split must be deferred to
// strategy assignment and the op must stay untouched here.
// CHECK-LABEL:   @NoStrategyNCEMatMulSOKCompatibleNotSplit
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x1x32x1x1xf16, {order = #GNHWC}>
func.func @NoStrategyNCEMatMulSOKCompatibleNotSplit(%arg0: tensor<1x1x32x1x1xf16, {order = #GNHWC}>) -> tensor<1x1x16384x1x1xf16, {order = #GNHWC}> {
  %weights = const.Declare tensor<1x16384x32x1x1xf16, {order = #GNHWC}> = dense<1.000000e+00> : tensor<1x16384x32x1x1xf16>, [#const.Reorder<#GNHWC>]

  %0 = VPU.NCE.MatMul(%arg0, %weights) rawFilterShape [1, 16384, 32, 1, 1] {
          mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
          strides = [1, 1],
          resultSegmentSizes = array<i32: 1, 0, 0, 0>
  } -> tensor<1x1x16384x1x1xf16, {order = #GNHWC}>

  return %0 : tensor<1x1x16384x1x1xf16, {order = #GNHWC}>

  // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.MatMul([[INPUT]], {{%.+}}) rawFilterShape [1, 16384, 32, 1, 1]
  // CHECK-SAME:          -> tensor<1x1x16384x1x1xf16, {order = #GNHWC}>
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// NCEMatMulOp already assigned the SplitOverKernel strategy. With 4 clusters the output channel
// threshold is extended to VPU_DIMENSION_LIMIT * numClusters = 8192 * 4 = 32768, so the 24576
// output channels fit in a single tile and the op must not be split.
// CHECK-LABEL:   @SOKStrategyNCEMatMulNotSplit
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x1x32x1x1xf16, {order = #GNHWC}>
func.func @SOKStrategyNCEMatMulNotSplit(%arg0: tensor<1x1x32x1x1xf16, {order = #GNHWC}>) -> tensor<1x1x24576x1x1xf16, {order = #GNHWC}> {
  %weights = const.Declare tensor<1x24576x32x1x1xf16, {order = #GNHWC}> = dense<1.000000e+00> : tensor<1x24576x32x1x1xf16>, [#const.Reorder<#GNHWC>]

  %0 = VPU.NCE.MatMul(%arg0, %weights) rawFilterShape [1, 24576, 32, 1, 1] {
          mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
          strides = [1, 1],
          resultSegmentSizes = array<i32: 1, 0, 0, 0>
  } -> tensor<1x1x24576x1x1xf16, {order = #GNHWC}>

  return %0 : tensor<1x1x24576x1x1xf16, {order = #GNHWC}>

  // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.MatMul([[INPUT]], {{%.+}}) rawFilterShape [1, 24576, 32, 1, 1]
  // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
  // CHECK-SAME:          -> tensor<1x1x24576x1x1xf16, {order = #GNHWC}>
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// NCEMatMulOp with the SplitOverKernel strategy assigned, but with 40000 output channels, which
// exceeds the extended threshold VPU_DIMENSION_LIMIT * numClusters = 8192 * 4 = 32768. The op
// must still be split (into 2 tiles of 20000 channels each) and the results concatenated back.
// CHECK-LABEL:   @SOKStrategyNCEMatMulExceedsChannelsStillSplit
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x1x32x1x1xf16, {order = #GNHWC}>
func.func @SOKStrategyNCEMatMulExceedsChannelsStillSplit(%arg0: tensor<1x1x32x1x1xf16, {order = #GNHWC}>) -> tensor<1x1x40000x1x1xf16, {order = #GNHWC}> {
  %weights = const.Declare tensor<1x40000x32x1x1xf16, {order = #GNHWC}> = dense<1.000000e+00> : tensor<1x40000x32x1x1xf16>, [#const.Reorder<#GNHWC>]

  %0 = VPU.NCE.MatMul(%arg0, %weights) rawFilterShape [1, 40000, 32, 1, 1] {
          mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
          strides = [1, 1],
          resultSegmentSizes = array<i32: 1, 0, 0, 0>
  } -> tensor<1x1x40000x1x1xf16, {order = #GNHWC}>

  return %0 : tensor<1x1x40000x1x1xf16, {order = #GNHWC}>

  // CHECK-DAG:   [[WEIGHTS_TILE1:%.+]] = const.Declare tensor<1x20000x32x1x1xf16, {order = #GNHWC}>
  // CHECK-SAME:          #const.SubView<[0, 20000, 0, 0, 0], [1, 20000, 32, 1, 1]>
  // CHECK-DAG:   [[WEIGHTS_TILE0:%.+]] = const.Declare tensor<1x20000x32x1x1xf16, {order = #GNHWC}>
  // CHECK-SAME:          #const.SubView<[0, 0, 0, 0, 0], [1, 20000, 32, 1, 1]>

  // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.MatMul([[INPUT]], [[WEIGHTS_TILE0]]) rawFilterShape [1, 20000, 32, 1, 1]
  // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
  // CHECK-SAME:          -> tensor<1x1x20000x1x1xf16, {order = #GNHWC}>

  // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.MatMul([[INPUT]], [[WEIGHTS_TILE1]]) rawFilterShape [1, 20000, 32, 1, 1]
  // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
  // CHECK-SAME:          -> tensor<1x1x20000x1x1xf16, {order = #GNHWC}>

  // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
  // CHECK-SAME:          -> tensor<1x1x40000x1x1xf16, {order = #GNHWC}>
}
