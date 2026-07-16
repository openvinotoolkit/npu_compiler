//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// XFAIL: *
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --ensure-nce-ops-size-requirements="enable-output-ensurance=false" --mlir-print-elementsattrs-with-hex-if-larger=-1 --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1, d3)>
func.func @IllegalNCEMatMul(%arg0: tensor<8x32x9216x1x1xf16>, %arg1: tensor<8x32x9216x1x1xf16>) -> tensor<1x8x32x32xf16> {
  %cst = const.Declare tensor<8x32x1x1x4xsi32> = dense<1> : tensor<8x32x1x1x4xsi32>
  %0 = VPU.PermuteCast(%arg0) {dst_order = #GNHWC, mem_perm = #map} : tensor<8x32x9216x1x1xf16> -> tensor<8x1x9216x32x1xf16, {order = #GNHWC}>
  %1 = VPU.PermuteCast(%arg1) {dst_order = #GNHWC, mem_perm = #GNHWC} : tensor<8x32x9216x1x1xf16> -> tensor<8x32x9216x1x1xf16, {order = #GNHWC}>
  %2 = VPU.AffineReshape(%0) {dim_mapping = [[0], [1], [2], [3, 4], [4]], shape_value = [8, 1, 9216, 8, 4]} : tensor<8x1x9216x32x1xf16, {order = #GNHWC}> -> tensor<8x1x9216x8x4xf16, {order = #GNHWC}>
  %3 = VPU.NCE.MatMul(%2, %1, %cst) rawFilterShape [8, 32, 9216, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1]} -> tensor<8x1x32x8x4xf16, {order = #GNHWC}>
  %4 = VPU.AffineReshape(%3) {dim_mapping = [[0], [1], [2], [3], [3, 4]], shape_value = [8, 1, 32, 32, 1]} : tensor<8x1x32x8x4xf16, {order = #GNHWC}> -> tensor<8x1x32x32x1xf16, {order = #GNHWC}>
  %5 = VPU.PermuteCast(%4) {dst_order = #NCDHW, mem_perm = #map1} : tensor<8x1x32x32x1xf16, {order = #GNHWC}> -> tensor<8x32x32x1x1xf16>
  %6 = VPU.AffineReshape(%5) {dim_mapping = [[0, 1], [2], [3], [3], [3]], shape_value = [1, 8, 32, 32]} : tensor<8x32x32x1x1xf16> -> tensor<1x8x32x32xf16>
  return %6 : tensor<1x8x32x32xf16>
}
