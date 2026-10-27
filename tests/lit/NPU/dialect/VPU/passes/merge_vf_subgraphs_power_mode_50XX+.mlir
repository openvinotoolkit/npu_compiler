//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --cmx-stack-frames-reserve-mem --cmx-metadata-reserve-mem --merge-vertical-fusion-subgraphs="enable-vertical-fusion-pipelining=true workload-management-mode=FWLM_V1_PAGES vf-merge-cost-mode=power" %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!q = !quant.uniform<u8:f16, 0.0033894703668706557:146>

module @executors {
  config.Resources 3 of @NCE at 1.700000e+03 MHz {
      config.MemoryResource 1571840 bytes of @CMX_NN {config.bandwidth = 32 : i64, config.derateFactor = 1.000000e+00 : f64}
  }

// CHECK-LABEL: @MergeVFWithPowerMode
func.func @MergeVFWithPowerMode(%arg0: tensor<1x32x800x1280x!q, {order = #NHWC}>) -> tensor<1x32x800x1280x!q, {order = #NHWC}> {
    %weights3x3 = const.Declare tensor<32x32x3x3x!q, {order = #NHWC}> = dense<1> : tensor<32x32x3x3xui8>, [#const.CastElemType<!q>, #const.Reorder<#NHWC>]
    %weights1x1 = const.Declare tensor<32x32x1x1x!q, {order = #NHWC}> = dense<1> : tensor<32x32x1x1xui8>, [#const.CastElemType<!q>, #const.Reorder<#NHWC>]

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x32x800x1280x!q, {order = #NHWC}>, %weights3x3 as %arg2: tensor<32x32x3x3x!q, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 17, 4]} -> tensor<1x32x800x1280x!q, {order = #NHWC}> {
        %output = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [32, 32, 3, 3] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x32x800x1280x!q, {order = #NHWC}>, tensor<32x32x3x3x!q, {order = #NHWC}> -> tensor<1x32x800x1280x!q, {order = #NHWC}>
        VPU.Yield %output
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x800x1280x!q, {order = #NHWC}>, %weights3x3 as %arg2: tensor<32x32x3x3x!q, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 17, 4]} -> tensor<1x32x800x1280x!q, {order = #NHWC}> {
        %output = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [32, 32, 3, 3] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x32x800x1280x!q, {order = #NHWC}>, tensor<32x32x3x3x!q, {order = #NHWC}> -> tensor<1x32x800x1280x!q, {order = #NHWC}>
        VPU.Yield %output
    }
    %2 = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x800x1280x!q, {order = #NHWC}>, %1 as %arg2: tensor<1x32x800x1280x!q, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 16]} -> tensor<1x32x800x1280x!q, {order = #NHWC}> {
        %output = VPU.NCE.Eltwise(%arg1, %arg2) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>} -> tensor<1x32x800x1280x!q, {order = #NHWC}>
        VPU.Yield %output
    }
    %3 = VPU.VerticalFusion (%2 as %arg1: tensor<1x32x800x1280x!q, {order = #NHWC}>, %weights3x3 as %arg2: tensor<32x32x3x3x!q, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 17, 4]} -> tensor<1x32x800x1280x!q, {order = #NHWC}> {
        %output = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [32, 32, 3, 3] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x32x800x1280x!q, {order = #NHWC}>, tensor<32x32x3x3x!q, {order = #NHWC}> -> tensor<1x32x800x1280x!q, {order = #NHWC}>
        VPU.Yield %output
    }
    %4 = VPU.VerticalFusion (%2 as %arg1: tensor<1x32x800x1280x!q, {order = #NHWC}>, %3 as %arg2: tensor<1x32x800x1280x!q, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 16]} -> tensor<1x32x800x1280x!q, {order = #NHWC}> {
        %output = VPU.NCE.Eltwise(%arg1, %arg2) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>} -> tensor<1x32x800x1280x!q, {order = #NHWC}>
        VPU.Yield %output
    }
    %5 = VPU.VerticalFusion (%4 as %arg1: tensor<1x32x800x1280x!q, {order = #NHWC}>, %weights3x3 as %arg2: tensor<32x32x3x3x!q, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 17, 4]} -> tensor<1x32x800x1280x!q, {order = #NHWC}> {
        %output = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [32, 32, 3, 3] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x32x800x1280x!q, {order = #NHWC}>, tensor<32x32x3x3x!q, {order = #NHWC}> -> tensor<1x32x800x1280x!q, {order = #NHWC}>
        VPU.Yield %output
    }
    %6 = VPU.VerticalFusion (%4 as %arg1: tensor<1x32x800x1280x!q, {order = #NHWC}>, %5 as %arg2: tensor<1x32x800x1280x!q, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 16]} -> tensor<1x32x800x1280x!q, {order = #NHWC}> {
        %output = VPU.NCE.Eltwise(%arg1, %arg2) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>} -> tensor<1x32x800x1280x!q, {order = #NHWC}>
        VPU.Yield %output
    }
    %7 = VPU.VerticalFusion (%6 as %arg1: tensor<1x32x800x1280x!q, {order = #NHWC}>, %weights3x3 as %arg2: tensor<32x32x3x3x!q, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 17, 4]} -> tensor<1x32x800x1280x!q, {order = #NHWC}> {
        %output = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [32, 32, 3, 3] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x32x800x1280x!q, {order = #NHWC}>, tensor<32x32x3x3x!q, {order = #NHWC}> -> tensor<1x32x800x1280x!q, {order = #NHWC}>
        VPU.Yield %output
    }
    %8 = VPU.VerticalFusion (%6 as %arg1: tensor<1x32x800x1280x!q, {order = #NHWC}>, %7 as %arg2: tensor<1x32x800x1280x!q, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 16]} -> tensor<1x32x800x1280x!q, {order = #NHWC}> {
        %output = VPU.NCE.Eltwise(%arg1, %arg2) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>} -> tensor<1x32x800x1280x!q, {order = #NHWC}>
        VPU.Yield %output
    }
    %9 = VPU.VerticalFusion (%8 as %arg1: tensor<1x32x800x1280x!q, {order = #NHWC}>, %weights3x3 as %arg2: tensor<32x32x3x3x!q, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 17, 4]} -> tensor<1x32x800x1280x!q, {order = #NHWC}> {
        %output = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [32, 32, 3, 3] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x32x800x1280x!q, {order = #NHWC}>, tensor<32x32x3x3x!q, {order = #NHWC}> -> tensor<1x32x800x1280x!q, {order = #NHWC}>
        VPU.Yield %output
    }
    %10 = VPU.VerticalFusion (%8 as %arg1: tensor<1x32x800x1280x!q, {order = #NHWC}>, %9 as %arg2: tensor<1x32x800x1280x!q, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 16]} -> tensor<1x32x800x1280x!q, {order = #NHWC}> {
        %output = VPU.NCE.Eltwise(%arg1, %arg2) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>} -> tensor<1x32x800x1280x!q, {order = #NHWC}>
        VPU.Yield %output
    }
    %11 = VPU.VerticalFusion (%10 as %arg1: tensor<1x32x800x1280x!q, {order = #NHWC}>, %weights3x3 as %arg2: tensor<32x32x3x3x!q, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 17, 4]} -> tensor<1x32x800x1280x!q, {order = #NHWC}> {
        %output = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [32, 32, 3, 3] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x32x800x1280x!q, {order = #NHWC}>, tensor<32x32x3x3x!q, {order = #NHWC}> -> tensor<1x32x800x1280x!q, {order = #NHWC}>
        VPU.Yield %output
    }
    %12 = VPU.VerticalFusion (%10 as %arg1: tensor<1x32x800x1280x!q, {order = #NHWC}>, %11 as %arg2: tensor<1x32x800x1280x!q, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 16]} -> tensor<1x32x800x1280x!q, {order = #NHWC}> {
        %output = VPU.NCE.Eltwise(%arg1, %arg2) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>} -> tensor<1x32x800x1280x!q, {order = #NHWC}>
        VPU.Yield %output
    }
    %13 = VPU.VerticalFusion (%12 as %arg1: tensor<1x32x800x1280x!q, {order = #NHWC}>, %weights3x3 as %arg2: tensor<32x32x3x3x!q, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 17, 4]} -> tensor<1x32x800x1280x!q, {order = #NHWC}> {
        %output = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [32, 32, 3, 3] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x32x800x1280x!q, {order = #NHWC}>, tensor<32x32x3x3x!q, {order = #NHWC}> -> tensor<1x32x800x1280x!q, {order = #NHWC}>
        VPU.Yield %output
    }
    %14 = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x800x1280x!q, {order = #NHWC}>, %weights3x3 as %arg2: tensor<32x32x3x3x!q, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 17, 4]} -> tensor<1x32x800x1280x!q, {order = #NHWC}> {
        %output = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [32, 32, 3, 3] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x32x800x1280x!q, {order = #NHWC}>, tensor<32x32x3x3x!q, {order = #NHWC}> -> tensor<1x32x800x1280x!q, {order = #NHWC}>
        VPU.Yield %output
    }
    %15 = VPU.VerticalFusion (%14 as %arg1: tensor<1x32x800x1280x!q, {order = #NHWC}>, %weights1x1 as %arg2: tensor<32x32x1x1x!q, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 28]} -> tensor<1x32x800x1280x!q, {order = #NHWC}> {
        %output = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [32, 32, 1, 1] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x32x800x1280x!q, {order = #NHWC}>, tensor<32x32x1x1x!q, {order = #NHWC}> -> tensor<1x32x800x1280x!q, {order = #NHWC}>
        VPU.Yield %output
    }
    %16 = VPU.VerticalFusion (%13 as %arg1: tensor<1x32x800x1280x!q, {order = #NHWC}>, %weights3x3 as %arg2: tensor<32x32x3x3x!q, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 17, 4]} -> tensor<1x32x800x1280x!q, {order = #NHWC}> {
        %output = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [32, 32, 3, 3] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x32x800x1280x!q, {order = #NHWC}>, tensor<32x32x3x3x!q, {order = #NHWC}> -> tensor<1x32x800x1280x!q, {order = #NHWC}>
        VPU.Yield %output
    }
    %17 = VPU.VerticalFusion (%16 as %arg1: tensor<1x32x800x1280x!q, {order = #NHWC}>, %15 as %arg2: tensor<1x32x800x1280x!q, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 16]} -> tensor<1x32x800x1280x!q, {order = #NHWC}> {
        %output = VPU.NCE.Eltwise(%arg1, %arg2) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>} -> tensor<1x32x800x1280x!q, {order = #NHWC}>
        VPU.Yield %output
    }
    return %17 : tensor<1x32x800x1280x!q, {order = #NHWC}>

    // CHECK:       %[[VF:.*]] = VPU.VerticalFusion
    // CHECK-SAME:  scenario = #VPU.vf_scenario<FULL_PREFETCHING>
    // CHECK-SAME:  tilingStrategy = [1, 1, 17, 4]
    // CHECK:       VPU.NCE.Convolution
    // CHECK:       VPU.NCE.Convolution
    // CHECK:       VPU.NCE.Convolution
    // CHECK:       VPU.NCE.Convolution
    // CHECK:       VPU.NCE.Eltwise
    // CHECK:       VPU.NCE.Convolution
    // CHECK:       VPU.NCE.Eltwise
    // CHECK:       VPU.NCE.Convolution
    // CHECK:       VPU.NCE.Eltwise
    // CHECK:       VPU.NCE.Convolution
    // CHECK:       VPU.NCE.Eltwise
    // CHECK:       VPU.NCE.Convolution
    // CHECK:       VPU.NCE.Eltwise
    // CHECK:       VPU.NCE.Convolution
    // CHECK:       VPU.NCE.Eltwise
    // CHECK:       VPU.NCE.Convolution
    // CHECK:       VPU.NCE.Convolution
    // CHECK:       VPU.NCE.Eltwise
    // CHECK:       VPU.Yield
    // CHECK:       return %[[VF]]
}
}
