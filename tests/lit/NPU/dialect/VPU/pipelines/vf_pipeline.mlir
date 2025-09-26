//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --vertical-fusion="vf-outlining-instance-threshold=0 vf-outlining-instance-threshold=0" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX


module @Do_not_Canocalize_VF_op_with_outline {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input"  : tensor<1x64x300x300xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x64x300x300xf32>
  }
  func.func @main(%arg0: tensor<1x64x300x300xf32>) -> tensor<1x64x300x300xf32> {
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x64x300x300xf32>) attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x300x300xf32> {
      %0 = VPU.Convert(%arg1) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x64x300x300xf32> -> tensor<1x64x300x300xf16>
      %1 = VPU.Swish(%0) {beta_value = 1.000000e+00 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x64x300x300xf16> -> tensor<1x64x300x300xf16>
      %2 = VPU.Swish(%1) {beta_value = 1.000000e+00 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x64x300x300xf16> -> tensor<1x64x300x300xf16>
      %3 = VPU.Convert(%2) {dstElemType = f32, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x64x300x300xf16> -> tensor<1x64x300x300xf32>
      VPU.Yield %3
    }
    return %0 : tensor<1x64x300x300xf32>
  }

    // CHECK-LABEL: @Do_not_Canocalize_VF_op_with_outline
    // CHECK: DataInfo "input" : tensor<1x64x300x300xf32>
    // CHECK: DataInfo "output" : tensor<1x64x300x300xf32>
    // CHECK:  func.func private @main_vf1([[INPUT:%.+]]: tensor<1x64x300x300xf32>) -> tensor<1x64x300x300xf32> {
    // CHECK:    [[CONVERT1:%.+]] = VPU.Convert([[INPUT:%.+]]) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x64x300x300xf32> -> tensor<1x64x300x300xf16>
    // CHECK:    [[SWH1:%.+]] = VPU.Swish([[CONVERT1]]) {beta_value = 1.000000e+00 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x64x300x300xf16> -> tensor<1x64x300x300xf16>
    // CHECK:    [[SWH2:%.+]] = VPU.Swish([[SWH1]]) {beta_value = 1.000000e+00 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x64x300x300xf16> -> tensor<1x64x300x300xf16>
    // CHECK:    [[CONVERT2:%.+]] = VPU.Convert([[SWH2]]) {dstElemType = f32, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x64x300x300xf16> -> tensor<1x64x300x300xf32>
    // CHECK:    return [[CONVERT2]] : tensor<1x64x300x300xf32>
    // CHECK:  }
    // CHECK:  func.func @main([[INPUT0:%.+]]: tensor<1x64x300x300xf32>) -> tensor<1x64x300x300xf32> {
    // CHECK:    [[FUNC:%.+]] = call @main_vf1([[INPUT0]]) : (tensor<1x64x300x300xf32>) -> tensor<1x64x300x300xf32>
    // CHECK:    return [[FUNC]] : tensor<1x64x300x300xf32>
    // CHECK:  }
}
