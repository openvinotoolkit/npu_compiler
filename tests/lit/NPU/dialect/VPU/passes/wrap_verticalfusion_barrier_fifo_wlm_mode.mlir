//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --wrap-in-vertical-fusion="workload-management-mode=PWLM_V1_BARRIER_FIFO"  %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>


// CHECK-LABEL: @WrapNCETiledTaskWith2dTiling
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x32x256x256xf16, {order = #NHWC}>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<32x32x3x3xf16, {order = #NHWC}>
func.func @WrapNCETiledTaskWith2dTiling(%arg0: tensor<1x32x256x256xf16, {order = #NHWC}>, %weights: tensor<32x32x3x3xf16, {order = #NHWC}>) -> tensor<1x32x256x256xf16, {order = #NHWC}> {
       %0 = VPU.NCE.Convolution(%arg0, %weights)
                {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                ppe = #VPU.PPEStub<>,
                rawFilterShape = [32, 32, 3, 3],
                strides = [1, 1],
                tilingStrategy = [1, 1, 2, 2]} : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>
    return %0 : tensor<1x32x256x256xf16, {order = #NHWC}>

    //CHECK:  VPU.VerticalFusion ([[INPUT0]] as [[ARG0:%.+]]: tensor<1x32x256x256xf16, {order = #NHWC}>, [[WEIGHTS]] as [[ARG1:%.+]]: tensor<32x32x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:  attributes {tilingStrategy = [1, 1, 2, 2]} -> tensor<1x32x256x256xf16, {order = #NHWC}> {
    //CHECK:  VPU.NCE.Convolution([[ARG0]], [[ARG1]])
    //CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    //CHECK-SAME:   pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:  rawFilterShape = [32, 32, 3, 3], strides = [1, 1]}
    //CHECK-SAME:  -> tensor<1x32x256x256xf16, {order = #NHWC}>
    //CHECK:    VPU.Yield
}
