//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --multi-cluster-strategy-assignment %s | FileCheck %s
// REQUIRES: platform-NPU5010

// Exercises getRequiredCMX on a non-tileable GenericSwLayerOp.

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @GenericSwLayerPrefetchParent
module @GenericSwLayerPrefetchParent {

module @VPU.SW {
  func.func @generated_0(%arg0: tensor<1x2x2x48xf16>) -> tensor<1x2x2x48xf16> {
    %0 = tensor.empty() : tensor<1x2x2x48xf16>
    %1 = linalg.softmax dimension(3) ins(%arg0 : tensor<1x2x2x48xf16>) outs(%0 : tensor<1x2x2x48xf16>) -> tensor<1x2x2x48xf16>
    return %1 : tensor<1x2x2x48xf16>
  }
}

// CHECK-LABEL: @main
// CHECK: VPU.GenericSwLayer
// CHECK: VPU.NCE.MaxPool
// CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
func.func @main(%arg0: tensor<1x2x48x2xf16>) -> tensor<1x2x48x2xf16> {
    %0 = VPU.MemPermute(%arg0) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x2x48x2xf16> -> tensor<1x2x48x2xf16, {order = #NCWH}>
    %1 = VPU.GenericSwLayer(%0 : tensor<1x2x48x2xf16, {order = #NCWH}>) @VPU.SW::@generated_0 -> tensor<1x2x48x2xf16, {order = #NCWH}>
    %2 = VPU.PermuteCast(%1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x2x48x2xf16, {order = #NCWH}> -> tensor<1x48x2x2xf16, {order = #NHWC}>
    %3 = VPU.NCE.MaxPool(%2) {kernel_size = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} -> tensor<1x48x2x2xf16, {order = #NHCW}>
    %4 = VPU.PermuteCast(%3) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x48x2x2xf16, {order = #NHCW}> -> tensor<1x2x48x2xf16>
    return %4 : tensor<1x2x48x2xf16>
}

}
