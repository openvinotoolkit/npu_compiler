//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --multi-cluster-strategy-assignment %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CMXConcatWithSplitOverHeight
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x4x8xf16, {order = #NHWC}>, [[ARG_1:%[^:]+]]: tensor<1x16x4x8xf16, {order = #NHWC}>)
func.func @CMXConcatWithSplitOverHeight(%arg0: tensor<1x16x4x8xf16, {order = #NHWC}>, %arg1: tensor<1x16x4x8xf16, {order = #NHWC}>) -> tensor<1x16x2x4xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_0 = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) rawFilterShape [32, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,  strides = [2, 2]} : tensor<1x16x4x8xf16, {order = #NHWC}>, tensor<32x16x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x2x4xf16, {order = #NHWC}>

    %cst1 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %1 = VPU.NCE.Convolution(%arg1, %cst_1, %cst1) rawFilterShape [32, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,  strides = [2, 2]} : tensor<1x16x4x8xf16, {order = #NHWC}>, tensor<32x16x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x2x4xf16, {order = #NHWC}>

    %2 = VPU.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]]} : tensor<1x32x2x4xf16, {order = #NHWC}>, tensor<1x32x2x4xf16, {order = #NHWC}> -> tensor<1x64x2x4xf16, {order = #NHWC}>

    %cst2 = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %cst_2 = const.Declare tensor<16x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %3 = VPU.NCE.Convolution(%2, %cst_2, %cst2) rawFilterShape [16, 64, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,  strides = [1, 1]} : tensor<1x64x2x4xf16, {order = #NHWC}>, tensor<16x64x3x3xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x2x4xf16, {order = #NHWC}>

    return %3 : tensor<1x16x2x4xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE0:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS0:%.+]] = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK:        [[VAL0:%.+]] = VPU.NCE.Convolution([[ARG_0]], [[WEIGHTS0]], [[WEIGHTSTABLE0]])
    //CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

    //CHECK:        [[WEIGHTSTABLE1:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS1:%.+]] = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK:        [[VAL1:%.+]] = VPU.NCE.Convolution([[ARG_1]], [[WEIGHTS1]], [[WEIGHTSTABLE1]])
    //CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

    //CHECK:        [[VAL2:%.+]] = VPU.Concat([[VAL0]], [[VAL1]])

    //CHECK:        [[WEIGHTSTABLE2:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    //CHECK:        [[WEIGHTS2:%.+]] = const.Declare tensor<16x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x64x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK:        [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL2]], [[WEIGHTS2]], [[WEIGHTSTABLE2]])
    //CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CMXConcatWithClustering
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x4x4xf16, {order = #NHWC}>, [[ARG_1:%[^:]+]]: tensor<1x16x4x4xf16, {order = #NHWC}>)
func.func @CMXConcatWithClustering(%arg0: tensor<1x16x4x4xf16, {order = #NHWC}>, %arg1: tensor<1x16x4x4xf16, {order = #NHWC}>) -> tensor<1x32x2x2xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %cst_0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) rawFilterShape [16, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,  strides = [1, 1]} : tensor<1x16x4x4xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x4x4xf16, {order = #NHWC}>

    %cst1 = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %cst_1 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %1 = VPU.NCE.Convolution(%arg1, %cst_1, %cst1) rawFilterShape [16, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,  strides = [1, 1]} : tensor<1x16x4x4xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x4x4xf16, {order = #NHWC}>

    %2 = VPU.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x4x4xf16, {order = #NHWC}>, tensor<1x16x4x4xf16, {order = #NHWC}> -> tensor<1x32x4x4xf16, {order = #NHWC}>

    %cst2 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_2 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %3 = VPU.NCE.Convolution(%2, %cst_2, %cst2) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,  strides = [1, 1]} : tensor<1x32x4x4xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x2x2xf16, {order = #NHWC}>

    return %3 : tensor<1x32x2x2xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE0:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    //CHECK:        [[WEIGHTS0:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.+]] = VPU.NCE.Convolution([[ARG_0]], [[WEIGHTS0]], [[WEIGHTSTABLE0]])
    //CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>

    //CHECK:        [[WEIGHTSTABLE1:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    //CHECK:        [[WEIGHTS1:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK:        [[VAL1:%.+]] = VPU.NCE.Convolution([[ARG_1]], [[WEIGHTS1]], [[WEIGHTSTABLE1]])
    //CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>

    //CHECK:        [[VAL2:%.+]] = VPU.Concat([[VAL0]], [[VAL1]])

    //CHECK:        [[WEIGHTSTABLE2:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS2:%.+]] = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK:        [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL2]], [[WEIGHTS2]], [[WEIGHTSTABLE2]])
    //CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>

    //CHECK:        return [[VAL3]] : tensor<1x32x2x2xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CMXConcatWithClusteringWithHKSwitchConvs
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x8x8xf16, {order = #NHWC}>, [[ARG_1:%[^:]+]]: tensor<1x16x8x8xf16, {order = #NHWC}>)
func.func @CMXConcatWithClusteringWithHKSwitchConvs(%arg0: tensor<1x16x8x8xf16, {order = #NHWC}>, %arg1: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x32x4x4xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %cst_0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) rawFilterShape [16, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,  strides = [1, 1]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x8x8xf16, {order = #NHWC}>

    %cst1 = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %cst_1 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %1 = VPU.NCE.Convolution(%arg1, %cst_1, %cst1) rawFilterShape [16, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,  strides = [1, 1]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x8x8xf16, {order = #NHWC}>

    %2 = VPU.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}> -> tensor<1x32x8x8xf16, {order = #NHWC}>

    %cst2 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_2 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %3 = VPU.NCE.Convolution(%2, %cst_2, %cst2) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,  strides = [2, 2]} : tensor<1x32x8x8xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x4x4xf16, {order = #NHWC}>

    return %3 : tensor<1x32x4x4xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE0:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    //CHECK:        [[WEIGHTS0:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.+]] = VPU.NCE.Convolution([[ARG_0]], [[WEIGHTS0]], [[WEIGHTSTABLE0]])
    //CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>

    //CHECK:        [[WEIGHTSTABLE1:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    //CHECK:        [[WEIGHTS1:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK:        [[VAL1:%.+]] = VPU.NCE.Convolution([[ARG_1]], [[WEIGHTS1]], [[WEIGHTSTABLE1]])
    //CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>

    //CHECK:        [[VAL2:%.+]] = VPU.Concat([[VAL0]], [[VAL1]])

    //CHECK:        [[WEIGHTSTABLE2:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS2:%.+]] = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK:        [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL2]], [[WEIGHTS2]], [[WEIGHTSTABLE2]])
    //CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>

    //CHECK:        return [[VAL3]] : tensor<1x32x4x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CMXConcatWithMultipleUsers
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x8x8xf16, {order = #NHWC}>, [[ARG_1:%[^:]+]]: tensor<1x16x8x8xf16, {order = #NHWC}>)
func.func @CMXConcatWithMultipleUsers(%arg0: tensor<1x16x8x8xf16, {order = #NHWC}>, %arg1: tensor<1x16x8x8xf16, {order = #NHWC}>) -> (tensor<1x32x4x4xf16, {order = #NHWC}>, tensor<1x48x8x8xf16, {order = #NHWC}>) {
    %cst = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %cst_0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) rawFilterShape [16, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,  strides = [1, 1]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x8x8xf16, {order = #NHWC}>

    %cst1 = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %cst_1 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %1 = VPU.NCE.Convolution(%arg1, %cst_1, %cst1) rawFilterShape [16, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,  strides = [1, 1]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x8x8xf16, {order = #NHWC}>

    %2 = VPU.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}> -> tensor<1x32x8x8xf16, {order = #NHWC}>

    %cst2 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_2 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %3 = VPU.NCE.Convolution(%2, %cst_2, %cst2) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,  strides = [2, 2]} : tensor<1x32x8x8xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x4x4xf16, {order = #NHWC}>

    %cst3 = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    %cst_3 = const.Declare tensor<48x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %4 = VPU.NCE.Convolution(%2, %cst_3, %cst3) rawFilterShape [48, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,  strides = [1, 1]} : tensor<1x32x8x8xf16, {order = #NHWC}>, tensor<48x32x3x3xf16, {order = #NHWC}>, tensor<48x1x1x4xsi32> -> tensor<1x48x8x8xf16, {order = #NHWC}>

    return %3, %4 : tensor<1x32x4x4xf16, {order = #NHWC}>, tensor<1x48x8x8xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE0:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    //CHECK:        [[WEIGHTS0:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK:        [[VAL0:%.+]] = VPU.NCE.Convolution([[ARG_0]], [[WEIGHTS0]], [[WEIGHTSTABLE0]])
    //CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>

    //CHECK:        [[WEIGHTSTABLE1:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    //CHECK:        [[WEIGHTS1:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK:        [[VAL1:%.+]] = VPU.NCE.Convolution([[ARG_1]], [[WEIGHTS1]], [[WEIGHTSTABLE1]])
    //CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>

    //CHECK:        [[VAL2:%.+]] = VPU.Concat([[VAL0]], [[VAL1]])

    //CHECK:        [[WEIGHTSTABLE2:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS2:%.+]] = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK:        [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL2]], [[WEIGHTS2]], [[WEIGHTSTABLE2]])
    //CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>

    //CHECK:        [[WEIGHTSTABLE3:%.+]] = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    //CHECK:        [[WEIGHTS3:%.+]] = const.Declare tensor<48x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x32x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK:        [[VAL4:%.+]] = VPU.NCE.Convolution([[VAL2]], [[WEIGHTS3]], [[WEIGHTSTABLE3]])
    //CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CMXConcatWithShortcut
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x8x8xf16, {order = #NHWC}>)
func.func @CMXConcatWithShortcut(%arg0: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x32x4x4xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %cst_0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) rawFilterShape [16, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,  strides = [1, 1]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x8x8xf16, {order = #NHWC}>

    %cst1 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %1 = VPU.NCE.Convolution(%0, %cst_1, %cst1) rawFilterShape [32, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,  strides = [1, 1]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<32x16x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x8x8xf16, {order = #NHWC}>

    %2 = VPU.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x32x8x8xf16, {order = #NHWC}> -> tensor<1x48x8x8xf16, {order = #NHWC}>

    %cst2 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_2 = const.Declare tensor<32x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x48x3x3xf16>, [#const.Reorder<#NHWC>]
    %3 = VPU.NCE.Convolution(%2, %cst_2, %cst2) rawFilterShape [32, 48, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,  strides = [2, 2]} : tensor<1x48x8x8xf16, {order = #NHWC}>, tensor<32x48x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x4x4xf16, {order = #NHWC}>

    return %3 : tensor<1x32x4x4xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE0:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    //CHECK:        [[WEIGHTS0:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK:        [[VAL0:%.+]] = VPU.NCE.Convolution([[ARG_0]], [[WEIGHTS0]], [[WEIGHTSTABLE0]])
    //CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

    //CHECK:        [[WEIGHTSTABLE1:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS1:%.+]] = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK:        [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], [[WEIGHTS1]], [[WEIGHTSTABLE1]])
    //CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

    //CHECK:        [[VAL2:%.+]] = VPU.Concat([[VAL0]], [[VAL1]])

    //CHECK:        [[WEIGHTSTABLE2:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS2:%.+]] = const.Declare tensor<32x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x48x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK:        [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL2]], [[WEIGHTS2]], [[WEIGHTSTABLE2]])
    //CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NDHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4, d1)>
#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2, d3)>

// CHECK-LABEL: @CMXConcatWithoutSplitOverHeight
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x132x120xf16, {order = #NHWC}>, [[ARG_1:%[^:]+]]: tensor<1x16x132x120xf16, {order = #NHWC}>)
func.func @CMXConcatWithoutSplitOverHeight(%arg0: tensor<1x16x132x120xf16, {order = #NHWC}>, %arg1: tensor<1x16x132x120xf16, {order = #NHWC}>) -> tensor<1x240x64x33xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.562500e-02> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 16 : i64>, #const.Reshape<[16, 1, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[16, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_0, %cst_1) rawFilterShape [16, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,  strides = [1, 1]} -> tensor<1x16x132x120xf16, {order = #NHWC}>
    %1 = VPU.ShapeCast {shape = [1, 33, 120, 64]} inputs(%0 : tensor<1x16x132x120xf16, {order = #NHWC}>) -> tensor<1x33x120x64xf16, {order = #NHWC}>
    %2 = VPU.AffineReshape(%1) {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 33, 120, 1, 64]} : tensor<1x33x120x64xf16, {order = #NHWC}> -> tensor<1x33x120x1x64xf16, {order = #NDHWC}>

    %3 = VPU.NCE.DepthConvolution(%arg1, %cst_0, %cst_1) rawFilterShape [16, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,  strides = [1, 1]} -> tensor<1x16x132x120xf16, {order = #NHWC}>
    %4 = VPU.ShapeCast {shape = [1, 33, 120, 64]} inputs(%3 : tensor<1x16x132x120xf16, {order = #NHWC}>) -> tensor<1x33x120x64xf16, {order = #NHWC}>
    %5 = VPU.AffineReshape(%4) {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 33, 120, 1, 64]} : tensor<1x33x120x64xf16, {order = #NHWC}> -> tensor<1x33x120x1x64xf16, {order = #NDHWC}>

    %6 = VPU.Concat(%2, %5) {static_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 1, 0]]} : tensor<1x33x120x1x64xf16, {order = #NDHWC}>, tensor<1x33x120x1x64xf16, {order = #NDHWC}> -> tensor<1x33x120x2x64xf16, {order = #NDHWC}>

    %7 = VPU.PermuteCast(%6) {dst_order = #NCDHW, mem_perm = #NCDHW} : tensor<1x33x120x2x64xf16, {order = #NDHWC}> -> tensor<1x120x2x64x33xf16>
    %8 = VPU.AffineReshape(%7) {dim_mapping = [[0], [0], [0], [0], [1, 2, 3]], shape_value = [15360, 33, 1, 1]} : tensor<1x120x2x64x33xf16> -> tensor<15360x33x1x1xf16>
    %9 = VPU.MemPermute(%8) {dst_order = #NCHW, mem_perm = #map} : tensor<15360x33x1x1xf16> -> tensor<33x15360x1x1xf16>
    %10 = VPU.Reshape(%9) {shape_value = [1, 33, 120, 2, 64]} : tensor<33x15360x1x1xf16> -> tensor<1x33x120x2x64xf16>
    %11 = VPU.PermuteCast(%10) {dst_order = #map1, mem_perm = #NCDHW} : tensor<1x33x120x2x64xf16> -> tensor<1x120x2x64x33xf16, {order = #map1}>
    %12 = VPU.AffineReshape(%11) {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 240, 64, 33]} : tensor<1x120x2x64x33xf16, {order = #map1}> -> tensor<1x240x64x33xf16, {order = #NWCH}>
    %13 = VPU.MemPermute(%12) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x240x64x33xf16, {order = #NWCH}> -> tensor<1x240x64x33xf16, {order = #NHWC}>

    %cst_3 = const.Declare tensor<240x240x1x1xf16, {order = #NHWC}> = dense<1.562500e-02> : tensor<240x240x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_4 = const.Declare tensor<240x1x1x4xsi32> = dense<10> : tensor<240x1x1x4xsi32>
    %14 = VPU.NCE.Convolution(%13, %cst_3, %cst_4) rawFilterShape [240, 240, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,  strides = [1, 1]} : tensor<1x240x64x33xf16, {order = #NHWC}>, tensor<240x240x1x1xf16, {order = #NHWC}>, tensor<240x1x1x4xsi32> -> tensor<1x240x64x33xf16, {order = #NHWC}>

    return %14 : tensor<1x240x64x33xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTS0:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.562500e-02> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 16 : i64>, #const.Reshape<[16, 1, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[16, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    //CHECK:        [[WEIGHTSTABLE0:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    //CHECK:        [[VAL0:%.+]] = VPU.NCE.DepthConvolution([[ARG_0]], [[WEIGHTS0]], [[WEIGHTSTABLE0]])
    //CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    //CHECK:        [[SHAPECAST0:%.+]] = VPU.ShapeCast {shape = [1, 33, 120, 64]} inputs([[VAL0]] : tensor<1x16x132x120xf16, {order = #NHWC}>) -> tensor<1x33x120x64xf16, {order = #NHWC}>
    //CHECK:        [[AFFINERESHAPE0:%.+]] = VPU.AffineReshape([[SHAPECAST0]])

    //CHECK:        [[VAL1:%.+]] = VPU.NCE.DepthConvolution([[ARG_1]], [[WEIGHTS0]], [[WEIGHTSTABLE0]])
    //CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    //CHECK:        [[SHAPECAST1:%.+]] = VPU.ShapeCast {shape = [1, 33, 120, 64]} inputs([[VAL1]] : tensor<1x16x132x120xf16, {order = #NHWC}>) -> tensor<1x33x120x64xf16, {order = #NHWC}>
    //CHECK:        [[AFFINERESHAPE1:%.+]] = VPU.AffineReshape([[SHAPECAST1]])

    //CHECK:        [[VAL2:%.+]] = VPU.Concat([[AFFINERESHAPE0]], [[AFFINERESHAPE1]])

    //CHECK:        [[PERMUTECAST0:%.+]] = VPU.PermuteCast([[VAL2]]) {dst_order = #NCDHW, mem_perm = #NCDHW} : tensor<1x33x120x2x64xf16, {order = #NDHWC}> -> tensor<1x120x2x64x33xf16>
    //CHECK:        [[AFFINERESHAPE2:%.+]]  = VPU.AffineReshape([[PERMUTECAST0]])
    //CHECK:        [[MEMPERMUTE0:%.+]] = VPU.MemPermute([[AFFINERESHAPE2]]) {dst_order = #NCHW, mem_perm = #map} : tensor<15360x33x1x1xf16> -> tensor<33x15360x1x1xf16>
    //CHECK:        [[RESHAPE0:%.+]] = VPU.Reshape([[MEMPERMUTE0]]) {shape_value = [1, 33, 120, 2, 64]} : tensor<33x15360x1x1xf16> -> tensor<1x33x120x2x64xf16>
    //CHECK:        [[PERMUTECAST1:%.+]] = VPU.PermuteCast([[RESHAPE0]]) {dst_order = #map1, mem_perm = #NCDHW} : tensor<1x33x120x2x64xf16> -> tensor<1x120x2x64x33xf16, {order = #map1}>
    //CHECK:        [[AFFINERESHAPE3:%.+]] = VPU.AffineReshape([[PERMUTECAST1]])
    //CHECK:        [[MEMPERMUTE1:%.+]] = VPU.MemPermute([[AFFINERESHAPE3]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x240x64x33xf16, {order = #NWCH}> -> tensor<1x240x64x33xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTS2:%.+]] = const.Declare tensor<240x240x1x1xf16, {order = #NHWC}> = dense<1.562500e-02> : tensor<240x240x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    //CHECK:        [[WEIGHTSTABLE2:%.+]] = const.Declare tensor<240x1x1x4xsi32> = dense<10> : tensor<240x1x1x4xsi32>
    //CHECK:        [[VAL3:%.+]] = VPU.NCE.Convolution([[MEMPERMUTE1]], [[WEIGHTS2]], [[WEIGHTSTABLE2]])
    //CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
}
