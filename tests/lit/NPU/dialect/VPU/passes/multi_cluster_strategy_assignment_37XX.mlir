//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --multi-cluster-strategy-assignment %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
!qElemType = !quant.uniform<u8:f16, 0.0017310915969488189:127>

// CHECK-LABEL: @AdaptSWAlignmentToAvoidSpilling
func.func @AdaptSWAlignmentToAvoidSpilling(%arg0: tensor<1x256x16x32xf16, {order = #NHWC}>) -> tensor<1x256x16x32xf16, {order = #NHWC}> {
    %cst_1 = const.Declare tensor<256x256x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x256x3x3xf16, {order = #NHWC}>
    %cst_2 = const.Declare tensor<256x256x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x256x3x3xf16, {order = #NHWC}>
    %cst_3 = const.Declare tensor<256x256x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x256x3x3xf16, {order = #NHWC}>
    %cst_18 = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    %cst_37 = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    %cst_38 = const.Declare tensor<256x16x1x1xf16, {order = #NHWC}> = dense<10.0>: tensor<1x1x1x256xf16>, [#const.Transpose<#NWCH>, #const.Reshape<[256, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[256, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %cst_39 = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    %cst_40 = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    %cst_41 = const.Declare tensor<256x16x1x1xf16, {order = #NHWC}> = dense<10.0> : tensor<1x1x1x256xf16>, [#const.Transpose<#NWCH>, #const.Reshape<[256, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[256, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %cst_42 = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    %cst_43 = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    %cst_44 = const.Declare tensor<256x16x1x1xf16, {order = #NHWC}> = dense<10.0> : tensor<1x1x1x256xf16>, [#const.Transpose<#NWCH>, #const.Reshape<[256, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[256, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %cst_45 = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>

    %12 = VPU.MVN(%arg0) {across_channels = false, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<1x256x16x32xf16, {order = #NHWC}> -> tensor<1x256x16x32xf16, {order = #NHWC}>
    %13 = VPU.NCE.DepthConvolution(%12, %cst_44, %cst_45) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [256, 1, 1, 1], strides = [1, 1]} -> tensor<1x256x16x32x!qElemType, {order = #NHWC}>
    %14 = VPU.NCE.Convolution(%13, %cst_1, %cst_43) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [256, 256, 3, 3], strides = [1, 1]} -> tensor<1x256x16x32xf16, {order = #NHWC}>
    %15 = VPU.MVN(%14) {across_channels = false, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<1x256x16x32xf16, {order = #NHWC}> -> tensor<1x256x16x32xf16, {order = #NHWC}>
    %16 = VPU.NCE.DepthConvolution(%15, %cst_41, %cst_42) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [256, 1, 1, 1], strides = [1, 1]} -> tensor<1x256x16x32x!qElemType, {order = #NHWC}>
    %17 = VPU.NCE.Convolution(%16, %cst_2, %cst_40) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [256, 256, 3, 3], strides = [1, 1]} -> tensor<1x256x16x32xf16, {order = #NHWC}>
    %18 = VPU.MVN(%17) {across_channels = false, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<1x256x16x32xf16, {order = #NHWC}> -> tensor<1x256x16x32xf16, {order = #NHWC}>
    %19 = VPU.NCE.DepthConvolution(%18, %cst_38, %cst_39) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [256, 1, 1, 1], strides = [1, 1]} -> tensor<1x256x16x32x!qElemType, {order = #NHWC}>
    %20 = VPU.ShapeCast {shape = [1, 16, 32, 256]} inputs(%13 : tensor<1x256x16x32x!qElemType, {order = #NHWC}>) -> tensor<1x16x32x256x!qElemType, {order = #NHWC}>
    %21 = VPU.ShapeCast {shape = [1, 16, 32, 256]} inputs(%19 : tensor<1x256x16x32x!qElemType, {order = #NHWC}>) -> tensor<1x16x32x256x!qElemType, {order = #NHWC}>
    %22 = VPU.NCE.Eltwise(%20, %21) {ppe = #VPU.PPEStub<>, op_type = #VPU.eltwise_type<ADD>} -> tensor<1x16x32x256x!qElemType, {order = #NHWC}>
    %23 = VPU.ShapeCast {shape = [1, 256, 16, 32]} inputs(%22 : tensor<1x16x32x256x!qElemType, {order = #NHWC}>) -> tensor<1x256x16x32x!qElemType, {order = #NHWC}>
    %24 = VPU.NCE.Convolution(%23, %cst_3, %cst_37) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [256, 256, 3, 3], strides = [1, 1]} -> tensor<1x256x16x32xf16, {order = #NHWC}>
    return %24 : tensor<1x256x16x32xf16, {order = #NHWC}>

    //CHECK-COUNT-13:   const.Declare
    //CHECK:            [[VAL0:%.+]] = VPU.MVN(%arg0)
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    //CHECK:            [[VAL1:%.+]] = VPU.NCE.DepthConvolution([[VAL0]],
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    //CHECK:            [[VAL2:%.+]] = VPU.NCE.Convolution([[VAL1]],
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    //CHECK:            [[VAL3:%.+]] = VPU.MVN([[VAL2]])
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    //CHECK:            [[VAL4:%.+]] = VPU.NCE.DepthConvolution([[VAL3]]
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    //CHECK:            [[VAL5:%.+]] = VPU.NCE.Convolution([[VAL4]]
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    //CHECK:            [[VAL6:%.+]] = VPU.MVN([[VAL5]]
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    //CHECK:            [[VAL7:%.+]] = VPU.NCE.DepthConvolution([[VAL6]]
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    //CHECK:            [[VAL8:%.+]] = VPU.ShapeCast {shape = [1, 16, 32, 256]} inputs([[VAL1]]
    //CHECK:            [[VAL9:%.+]] = VPU.ShapeCast {shape = [1, 16, 32, 256]} inputs([[VAL7]]
    //CHECK:            [[VAL10:%.+]] = VPU.NCE.Eltwise([[VAL8]], [[VAL9]])
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    //CHECK:            [[VAL11:%.+]] = VPU.ShapeCast {shape = [1, 256, 16, 32]} inputs([[VAL10]]
    //CHECK:            [[VAL12:%.+]] = VPU.NCE.Convolution([[VAL11]],
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    //CHECK:            return [[VAL12]] : tensor<1x256x16x32xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.15407909318512561:127>

// CHECK-LABEL: @NoSpillingWhenSOHAlignmentCanBeAdjustedCompatible
func.func @NoSpillingWhenSOHAlignmentCanBeAdjustedCompatible(%arg0: tensor<1x32x60x60x!qElemType, {order = #NHWC}>) -> tensor<1x32x15x15x!qElemType, {order = #NHWC}>  {
    %cst = const.Declare tensor<16x32x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x3x3xf32, {order = #NHWC}>
    %cst_0 = const.Declare tensor<64x16x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x3x3xf32, {order = #NHWC}>
    %cst_1 = const.Declare tensor<32x64x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x64x3x3xf32, {order = #NHWC}>
    %cst_2 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_3 = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %cst_4 = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<10.0> : tensor<1x64x1x1xf32>, [#const.CastElemType<f16>, #const.Reshape<[64, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[64, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %cst_5 = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %cst_6 = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %cst, %cst_6) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, rawFilterShape = [16, 32, 3, 3], strides = [2, 2]} -> tensor<1x16x30x30x!qElemType, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst_0, %cst_5) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [64, 16, 3, 3], strides = [1, 1]} -> tensor<1x64x30x30xf16, {order = #NHWC}>
    %2 = VPU.NCE.DepthConvolution(%1, %cst_4, %cst_3) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [64, 1, 1, 1], strides = [1, 1]} -> tensor<1x64x30x30x!qElemType, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %cst_1, %cst_2) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, rawFilterShape = [32, 64, 3, 3], strides = [2, 2]} -> tensor<1x32x15x15x!qElemType, {order = #NHWC}>

    return %3 : tensor<1x32x15x15x!qElemType, {order = #NHWC}>

    //CHECK-COUNT-7:    const.Declare
    //CHECK:            [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0,
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    //CHECK:            [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]],
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    //CHECK:            [[VAL2:%.+]] = VPU.NCE.DepthConvolution([[VAL1]],
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    //CHECK:            [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL2]],
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    //CHECK:            return [[VAL3]] : tensor<1x32x15x15x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.15407909318512561:127>

// CHECK-LABEL: @ConsiderDualPortDMAForSOHSpilling
func.func @ConsiderDualPortDMAForSOHSpilling(%arg0: tensor<1x16x135x240x!qElemType, {order = #NHWC}>) -> tensor<1x32x135x240x!qElemType, {order = #NHWC}> {

    %cst = const.Declare tensor<128x32x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x32x3x3xf32, {order = #NHWC}>
    %cst_0 = const.Declare tensor<32x128x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x128x3x3xf32, {order = #NHWC}>
    %cst_1 = const.Declare tensor<32x16x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf32, {order = #NHWC}>
    %cst_2 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_3 = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    %cst_4 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %cst_1, %cst_4) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} -> tensor<1x32x135x240x!qElemType, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst, %cst_3) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [128, 32, 3, 3], strides = [1, 1]} -> tensor<1x128x135x240x!qElemType, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%1, %cst_0, %cst_2) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 128, 3, 3], strides = [1, 1]} -> tensor<1x32x135x240x!qElemType, {order = #NHWC}>


    return %2 : tensor<1x32x135x240x!qElemType, {order = #NHWC}>

    //CHECK-COUNT-5:    const.Declare
    //CHECK:            [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0,
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    //CHECK:            [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]],
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    //CHECK:            [[VAL2:%.+]] = VPU.NCE.Convolution([[VAL1]],
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    //CHECK:            return [[VAL2]] : tensor<1x32x135x240x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedSOH
func.func @ConvAssignedSOH(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, %cst_0, %cst)
    //CHECK-SAME:    {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]}
    //CHECK-SAME:      -> tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x80x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedSOK
func.func @ConvAssignedSOK(%arg0: tensor<1x128x1x1xf16, {order = #NHWC}>) -> tensor<1x1024x1x1xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1024x1x1x4xsi32> = dense<10> : tensor<1024x1x1x4xsi32>
    %cst_0 = const.Declare tensor<1024x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [1024, 128, 1, 1], strides = [1, 1]} -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x1024x1x1xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<1024x1x1x4xsi32> = dense<10> : tensor<1024x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<1024x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x128x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]])
    //CHECK-SAME:   {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [1024, 128, 1, 1], strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x1024x1x1xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x1024x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedSOK
func.func @ConvAssignedSOK(%arg0: tensor<1x64x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1x1xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    %cst_0 = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [48, 64, 3, 3], strides = [1, 1]} -> tensor<1x48x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x48x1x1xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]])
    //CHECK-SAME:   {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [48, 64, 3, 3], strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x48x1x1xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x48x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedSOB
// CHECK-SAME:      [[INPUT:%.*]]: tensor<2x16x96x96xf16, {order = #NHWC}>
func.func @ConvAssignedSOB(%arg0: tensor<2x16x96x96xf16, {order = #NHWC}>) -> tensor<2x16x96x94xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %cst_0 = const.Declare tensor<16x16x3x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x5xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 5], strides = [1, 1]} -> tensor<2x16x96x94xf16, {order = #NHWC}>
    return %0 : tensor<2x16x96x94xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<16x16x3x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x5xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTSTABLE]])
    //CHECK-SAME:   {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [16, 16, 3, 5], strides = [1, 1]}
    //CHECK-SAME:   -> tensor<2x16x96x94xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<2x16x96x94xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvAssignedSOK
func.func @DepthConvAssignedSOK(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x1x1x3x3xf16>, [#const.Reshape<[32, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[32, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x1x1x3x3xf16>, [#const.Reshape<[32, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[32, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.DepthConvolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]])
    //CHECK-SAME:   {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]}
    //CHECK:        -> tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DepthConvAssignedSOK
func.func @DepthConvAssignedSOK(%arg0: tensor<1x128x1x1xf16, {order = #NHWC}>) -> tensor<1x128x1x1xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    %cst_1 = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x1x1x3x3xf16>, [#const.Reshape<[128, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[128, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [128, 1, 3, 3], strides = [1, 1]} -> tensor<1x128x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x128x1x1xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<128x1x1x3x3xf16>, [#const.Reshape<[128, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[128, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.DepthConvolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]])
    //CHECK-SAME:   {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 1, 3, 3], strides = [1, 1]}
    //CHECK:        -> tensor<1x128x1x1xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x128x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DepthConvAssignedSOK
func.func @DepthConvAssignedSOK(%arg0: tensor<1x32x1x1xf16, {order = #NHWC}>) -> tensor<1x32x1x1xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x1x1x3x3xf16>, [#const.Reshape<[32, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[32, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x32x1x1xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x1x1x3x3xf16>, [#const.Reshape<[32, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[32, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.DepthConvolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]])
    //CHECK-SAME:   {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x32x1x1xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x32x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolAssignedSOH
func.func @MaxPoolAssignedSOH(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.MaxPool(%arg0)
    //CHECK-SAME:   {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolAssignedSOK
func.func @MaxPoolAssignedSOK(%arg0: tensor<1x32x1x1xf16, {order = #NHWC}>) -> tensor<1x32x1x1xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x32x1x1xf16, {order = #NHWC}>

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.MaxPool(%arg0)
    //CHECK-SAME:   {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x32x1x1xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x32x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  func.func @MaxPoolAssignedSOB
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<2x32x112x112xf16, {order = #NHWC}>)
func.func @MaxPoolAssignedSOB(%input: tensor<2x32x112x112xf16, {order = #NHWC}>) -> tensor<2x32x112x112xf16, {order = #NHWC}> {
    %maxpool = VPU.NCE.MaxPool(%input) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        kernel_size = [1, 1]
    } -> tensor<2x32x112x112xf16, {order = #NHWC}>
    return %maxpool : tensor<2x32x112x112xf16, {order = #NHWC}>

    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } -> tensor<2x32x112x112xf16, {order = #NHWC}>

    // CHECK:       return [[MAXPOOL]] : tensor<2x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  func.func @AvgPoolAssignedSOB
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<2x32x112x112xf16, {order = #NHWC}>)
func.func @AvgPoolAssignedSOB(%input: tensor<2x32x112x112xf16, {order = #NHWC}>) -> tensor<2x32x112x112xf16, {order = #NHWC}> {
    %avgpool = VPU.NCE.AveragePool(%input) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        kernel_size = [1, 1]
    } -> tensor<2x32x112x112xf16, {order = #NHWC}>
    return %avgpool : tensor<2x32x112x112xf16, {order = #NHWC}>

    // CHECK:       [[AVGPOOL:%.+]] = VPU.NCE.AveragePool([[INPUT]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } -> tensor<2x32x112x112xf16, {order = #NHWC}>

    // CHECK:       return [[AVGPOOL]] : tensor<2x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SparseConvAssignedSOH
func.func @SparseConvAssignedSOH(%arg0 : tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1 : tensor<1x64x28x28xi1, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {

    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1)
        -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    %weights = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<80x1x1x640xi1> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<64> : tensor<80xi64>, alignment = 16 : i64>, is_weights}
        -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
                             sparsity_map=tensor<80x1x1x640xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<64> : tensor<80xi64>, alignment = 16 : i64>>

    %weights_table = const.Declare tensor<80x1x1x4xsi32> = dense<1> : tensor<80x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%input_sparse, %weights_sparse, %weights_table) {
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 01 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [80, 64, 3, 3],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
        }

    return %0 : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                                  sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>

    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, %arg1)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<80x1x1x640xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]])
    // CHECK-SAME:      {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<64> : tensor<80xi64>, alignment = 16 : i64>}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<80x1x1x640xi1>, is_weights, #VPU.SparsityCompression

    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<80x1x1x4xsi32> = dense<1> : tensor<80x1x1x4xsi32>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS_SPARSE]], [[CST_WEIGHTS_TABLE]])
    // CHECK-SAME:          {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEStub<>,
    // CHECK-SAME:          rawFilterShape = [80, 64, 3, 3],
    // CHECK-SAME:          strides = [1, 1]}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {
    // CHECK:         VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <VECTOR_FP16>
    // CHECK:       }

    // CHECK:       return [[OUT]] : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                                sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SparseConvAssignedSOK
func.func @SparseConvAssignedSOK(%arg0 : tensor<1x128x1x1xf16, {order = #NHWC}>, %arg1 : tensor<1x128x1x1xi1, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>> {

    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1)
        -> !VPU.SparseTensor<data=tensor<1x128x1x1xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x128x1x1xi1, {order = #NHWC}>>

    %weights = const.Declare tensor<1008x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1008x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<1008x1x1x128xi1> = dense<1.000000e+00> : tensor<1008x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<64> : tensor<1008xi64>, alignment = 16 : i64>, is_weights}
        -> !VPU.SparseTensor<data=tensor<1008x128x1x1xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1008x1x1x128xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<64> : tensor<1008xi64>, alignment = 16 : i64>>

    %weights_table = const.Declare tensor<1008x1x1x4xsi32> = dense<1> : tensor<1008x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%input_sparse, %weights_sparse, %weights_table) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [1008, 128, 1, 1],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
        }

    return %0 : !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
                                  sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>>

    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, %arg1)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x128x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x128x1x1xi1, {order = #NHWC}>>

    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<1008x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1008x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<1008x1x1x128xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1008x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]])
    // CHECK-SAME:       {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<64> : tensor<1008xi64>, alignment = 16 : i64>}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1008x128x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1008x1x1x128xi1>, is_weights, #VPU.SparsityCompression

    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<1008x1x1x4xsi32> = dense<1> : tensor<1008x1x1x4xsi32>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS_SPARSE]], [[CST_WEIGHTS_TABLE]])
    // CHECK-SAME:          {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEStub<>,
    // CHECK-SAME:          rawFilterShape = [1008, 128, 1, 1],
    // CHECK-SAME:          strides = [1, 1]}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>> {
    // CHECK:         VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <VECTOR_FP16>
    // CHECK:       }

    // CHECK:       return [[OUT]] : !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                                     sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>>
}

// -----

// Weights contain only sparse values, so the weight splits would have only sparse values as well
// SOK is avoided in case any split has empty weights

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SparseConvNotAssignedSOK
func.func @SparseConvNotAssignedSOK(%arg0 : tensor<1x128x1x1xf16, {order = #NHWC}>, %arg1 : tensor<1x128x1x1xi1, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>> {

    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1)
        -> !VPU.SparseTensor<data=tensor<1x128x1x1xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x128x1x1xi1, {order = #NHWC}>>

    %weights = const.Declare tensor<1008x128x1x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1008x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<1008x1x1x128xi1> = dense<0.000000e+00> : tensor<1008x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {
            sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<0> : tensor<1008xi64>, alignment = 16 : i64>, is_weights
        } -> !VPU.SparseTensor<data=tensor<1008x128x1x1xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1008x1x1x128xi1>,
                               is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<0> : tensor<1008xi64>, alignment = 16 : i64>>

    %weights_table = const.Declare tensor<1008x1x1x4xsi32> = dense<1> : tensor<1008x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%input_sparse, %weights_sparse, %weights_table) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [1008, 128, 1, 1],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
        }

    return %0 : !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
                                  sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>>

    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, %arg1)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x128x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x128x1x1xi1, {order = #NHWC}>>

    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<1008x128x1x1xf16, {order = #NHWC}> = dense<0.000000e+00>
    // CHECK-SAME:      : tensor<1008x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<1008x1x1x128xi1> = dense<0.000000e+00>
    // CHECK-SAME:      : tensor<1008x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]]) {
    // CHECK-SAME:          is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<0> : tensor<1008xi64>, alignment = 16 : i64>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1008x128x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                             sparsity_map=tensor<1008x1x1x128xi1>,
    // CHECK-SAME:                             is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<0> : tensor<1008xi64>, alignment = 16 : i64>>

    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<1008x1x1x4xsi32> = dense<1> : tensor<1008x1x1x4xsi32>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS_SPARSE]], [[CST_WEIGHTS_TABLE]])
    // CHECK-SAME:          {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEStub<>,
    // CHECK-SAME:          rawFilterShape = [1008, 128, 1, 1],
    // CHECK-SAME:          strides = [1, 1]}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>> {
    // CHECK:         VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <VECTOR_FP16>
    // CHECK:       }

    // CHECK:       return [[OUT]] : !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                                     sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @TanhAssignedClustering
func.func @TanhAssignedClustering(%arg0: tensor<1x4x1x512xf16, {order = #NCHW}>) -> tensor<1x4x1x512xf16, {order = #NCHW}> {

    %1 = VPU.Tanh(%arg0) : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>

    return %1 : tensor<1x4x1x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultTANH:%.*]] = VPU.Tanh(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultTANH]] : tensor<1x4x1x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @TanhAssignedSOHForEqualCost
func.func @TanhAssignedSOHForEqualCost(%arg0: tensor<1x4x32x512xf16, {order = #NCHW}>) -> tensor<1x4x32x512xf16, {order = #NCHW}> {

    %1 = VPU.Tanh(%arg0) : tensor<1x4x32x512xf16, {order = #NCHW}> -> tensor<1x4x32x512xf16, {order = #NCHW}>

    return %1 : tensor<1x4x32x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultTANH:%.*]] = VPU.Tanh(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4x32x512xf16, {order = #NCHW}> -> tensor<1x4x32x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultTANH]] : tensor<1x4x32x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MVNAssignedClustering
func.func @MVNAssignedClustering(%arg0: tensor<1x1x1x512xf16, {order = #NCHW}>) -> tensor<1x1x1x512xf16, {order = #NCHW}> {

    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, normalize_variance = true} : tensor<1x1x1x512xf16, {order = #NCHW}> -> tensor<1x1x1x512xf16, {order = #NCHW}>

    return %0 : tensor<1x1x1x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultMVN:%.*]] = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true} : tensor<1x1x1x512xf16, {order = #NCHW}> -> tensor<1x1x1x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultMVN]] : tensor<1x1x1x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SoftMaxAssignedClustering
func.func @SoftMaxAssignedClustering(%arg0: tensor<1x1x1x512xf16, {order = #NCHW}>) -> tensor<1x1x1x512xf16, {order = #NCHW}> {

    %1 = VPU.SoftMax(%arg0) {axisInd = 3} : tensor<1x1x1x512xf16, {order = #NCHW}> -> tensor<1x1x1x512xf16, {order = #NCHW}>

    return %1 : tensor<1x1x1x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultSoftMax:%.*]] = VPU.SoftMax(%arg0) {axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x512xf16, {order = #NCHW}> -> tensor<1x1x1x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultSoftMax]] : tensor<1x1x1x512xf16, {order = #NCHW}>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SoftMaxAssignedSplitOverKernel
func.func @SoftMaxAssignedSplitOverKernel(%arg0: tensor<1x4x1x512xf16, {order = #NCHW}>) -> tensor<1x4x1x512xf16, {order = #NCHW}> {
    %1 = VPU.SoftMax(%arg0) {axisInd = 3} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>

    return %1 : tensor<1x4x1x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultSoftMax:%.*]] = VPU.SoftMax(%arg0) {axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultSoftMax]] : tensor<1x4x1x512xf16, {order = #NCHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SoftMaxAssignedSplitOverHeight
func.func @SoftMaxAssignedSplitOverHeight(%arg0: tensor<1x4x2x512xf16, {order = #NHWC}>) -> tensor<1x4x2x512xf16, {order = #NHWC}> {
    %1 = VPU.SoftMax(%arg0) {axisInd = 3} : tensor<1x4x2x512xf16, {order = #NHWC}> -> tensor<1x4x2x512xf16, {order = #NHWC}>

    return %1 : tensor<1x4x2x512xf16, {order = #NHWC}>

    //CHECK:   [[ResultSoftMax:%.*]] = VPU.SoftMax(%arg0) {axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4x2x512xf16, {order = #NHWC}> -> tensor<1x4x2x512xf16, {order = #NHWC}>
    //CHECK:   return [[ResultSoftMax]] : tensor<1x4x2x512xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SoftMaxAssignedSplitOverWidth
func.func @SoftMaxAssignedSplitOverWidth(%arg0: tensor<1x1x2x512xf16, {order = #NCHW}>) -> tensor<1x1x2x512xf16, {order = #NCHW}> {
    %1 = VPU.SoftMax(%arg0) {axisInd = 2} : tensor<1x1x2x512xf16, {order = #NCHW}> -> tensor<1x1x2x512xf16, {order = #NCHW}>

    return %1 : tensor<1x1x2x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultSoftMax:%.*]] = VPU.SoftMax(%arg0) {axisInd = 2 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>} : tensor<1x1x2x512xf16, {order = #NCHW}> -> tensor<1x1x2x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultSoftMax]] : tensor<1x1x2x512xf16, {order = #NCHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NHWCSoftMaxAssignedSplitOverHeight
func.func @NHWCSoftMaxAssignedSplitOverHeight(%arg0: tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x8x4x76xf16, {order = #NHWC}> {
    %1 = VPU.SoftMax(%arg0) {axisInd = 3} : tensor<1x8x4x76xf16, {order = #NHWC}> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    return %1 : tensor<1x8x4x76xf16, {order = #NHWC}>

    //CHECK:   [[ResultSoftMax:%.*]] = VPU.SoftMax(%arg0) {axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x8x4x76xf16, {order = #NHWC}> -> tensor<1x8x4x76xf16, {order = #NHWC}>
    //CHECK:   return [[ResultSoftMax]] : tensor<1x8x4x76xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MVNAssignedSplitOverKernel
func.func @MVNAssignedSplitOverKernel(%arg0: tensor<1x4x1x512xf16, {order = #NCHW}>) -> tensor<1x4x1x512xf16, {order = #NCHW}> {

    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, normalize_variance = true} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>

    return %0 : tensor<1x4x1x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultMVN:%.*]] = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultMVN]] : tensor<1x4x1x512xf16, {order = #NCHW}>
}

// -----

// CHECK-LABEL: @Mvn6AssignedSplitOverKernel
func.func @Mvn6AssignedSplitOverKernel(%arg0: tensor<1x32x15x64xf16>) -> tensor<1x32x15x64xf16> {
    %0 = VPU.MVN6(%arg0) {axes = [2], eps = 9.9999997473787516E-6 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>} : tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16>
    return %0 : tensor<1x32x15x64xf16>

    //CHECK:   [[MVN6:%.+]] = VPU.MVN6({{[^:]+}}) {
    //CHECK-SAME:     axes = [2],
    //CHECK-SAME:     eps = 9.9999997473787516E-6 : f64,
    //CHECK-SAME:     eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>,
    //CHECK-SAME:     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    //CHECK-SAME:     normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>}
    //CHECK-SAME:     : tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16>
    //CHECK:   return [[MVN6]] : tensor<1x32x15x64xf16>
}

// -----

// CHECK-LABEL: @Mvn6AssignedSplitOverHeight
func.func @Mvn6AssignedSplitOverHeight(%arg0: tensor<1x32x15x64xf16>) -> tensor<1x32x15x64xf16> {
    %0 = VPU.MVN6(%arg0) {axes = [1, 3], eps = 9.9999997473787516E-6 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>} : tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16>
    return %0 : tensor<1x32x15x64xf16>

    //CHECK:   [[MVN6:%.+]] = VPU.MVN6({{[^:]+}}) {
    //CHECK-SAME:     axes = [1, 3],
    //CHECK-SAME:     eps = 9.9999997473787516E-6 : f64,
    //CHECK-SAME:     eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>,
    //CHECK-SAME:     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    //CHECK-SAME:     normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>}
    //CHECK-SAME:     : tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16>
    //CHECK:   return [[MVN6]] : tensor<1x32x15x64xf16>
}

// -----

// CHECK-LABEL: @Mvn6AssignedClustering
func.func @Mvn6AssignedClustering(%arg0: tensor<1x32x15x1xf16>) -> tensor<1x32x15x1xf16> {
    %0 = VPU.MVN6(%arg0) {axes = [1, 2], eps = 9.9999997473787516E-6 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>} : tensor<1x32x15x1xf16> -> tensor<1x32x15x1xf16>
    return %0 : tensor<1x32x15x1xf16>

    //CHECK:   [[MVN6:%.+]] = VPU.MVN6({{[^:]+}}) {
    //CHECK-SAME:     axes = [1, 2],
    //CHECK-SAME:     eps = 9.9999997473787516E-6 : f64,
    //CHECK-SAME:     eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>,
    //CHECK-SAME:     multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    //CHECK-SAME:     normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>}
    //CHECK-SAME:     : tensor<1x32x15x1xf16> -> tensor<1x32x15x1xf16>
    //CHECK:   return [[MVN6]] : tensor<1x32x15x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TanhAfterConvAssignedClustering
func.func @TanhAfterConvAssignedClustering(%arg0: tensor<1x4x1x512xf16, {order = #NCHW}>) -> tensor<1x1024x1x1xf16, {order = #NHWC}> {

    %39 = const.Declare tensor<1x1024x1x1xf16> = dense<10.1> : tensor<1x1024x1x1xf16>
    %40 = const.Declare tensor<1x1024x1x1xf16, {order = #NHWC}> = dense<10.1> : tensor<1x1024x1x1xf16, {order = #NHWC}>

    %cst_7 = const.Declare tensor<1024x1024x1x1xf16, {order = #NHWC}> = dense<10.1> : tensor<1x1024x1024xf16>, [#const.Reshape<[1024, 1024]>, #const.Reshape<[1024, 1024, 1, 1]>, #const.Reorder<#NHWC>]
    %cst_27 = const.Declare tensor<1024x1x1x4xsi32> = dense<10> : tensor<1024x1x1x4xsi32>

    %42 = VPU.NCE.Convolution(%40, %cst_7, %cst_27) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [1024, 1024, 1, 1], strides = [1, 1]} -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    %45 = VPU.Tanh(%42) : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1024x1x1xf16, {order = #NHWC}>

    return %45 : tensor<1x1024x1x1xf16, {order = #NHWC}>

    //CHECK:   [[ResultConvolution:%.*]] = VPU.NCE.Convolution([[cst0:%.*]], [[cst1:%.*]], [[cst2:%.*]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [1024, 1024, 1, 1], strides = [1, 1]} -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    //CHECK:   [[ResultTanh:%.*]] = VPU.Tanh([[ResultConvolution]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    //CHECK:   return [[ResultTanh]] : tensor<1x1024x1x1xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @InterpolateHalfPixelAssignedSOHOverlapped
func.func @InterpolateHalfPixelAssignedSOHOverlapped(%arg0: tensor<1x1x96x160xf16>) -> tensor<1x1x192x320xf16> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 320]} : tensor<1x1x96x160xf16> -> tensor<1x1x192x320xf16>
    return %0 : tensor<1x1x192x320xf16>
    //CHECK:   [[INTERPOLATE:%.*]] = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 320]} : tensor<1x1x96x160xf16> -> tensor<1x1x192x320xf16>
    //CHECK:   return [[INTERPOLATE]] : tensor<1x1x192x320xf16>
}

// -----

// CHECK-LABEL: @InterpolateAlignCornersAssignedSOHOverlapped
func.func @InterpolateAlignCornersAssignedSOHOverlapped(%arg0: tensor<1x1x96x160xf16>) -> tensor<1x1x192x320xf16> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 320]} : tensor<1x1x96x160xf16> -> tensor<1x1x192x320xf16>
    return %0 : tensor<1x1x192x320xf16>
    //CHECK:   [[INTERPOLATE:%.*]] = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 320]} : tensor<1x1x96x160xf16> -> tensor<1x1x192x320xf16>
    //CHECK:   return [[INTERPOLATE]] : tensor<1x1x192x320xf16>
}

// -----

// CHECK-LABEL: @InterpolateAlignCornersAssignedClustering
func.func @InterpolateAlignCornersAssignedClustering(%arg0: tensor<1x1x1x160xf16>) -> tensor<1x1x1x320xf16> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [1.000000e+00, 2.000000e+00], sizes_attr = [1, 320]} : tensor<1x1x1x160xf16> -> tensor<1x1x1x320xf16>
    return %0 : tensor<1x1x1x320xf16>
    //CHECK:   [[INTERPOLATE:%.*]] = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [1.000000e+00, 2.000000e+00], sizes_attr = [1, 320]} : tensor<1x1x1x160xf16> -> tensor<1x1x1x320xf16>
    //CHECK:   return [[INTERPOLATE]] : tensor<1x1x1x320xf16>
}

// -----

// CHECK-LABEL: @InterpolatePytorchHalfPixelAssignedSOHOverlapped
func.func @InterpolatePytorchHalfPixelAssignedSOHOverlapped(%arg0: tensor<1x1x96x160xf16>) -> tensor<1x1x192x320xf16> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <PYTORCH_HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 320]} : tensor<1x1x96x160xf16> -> tensor<1x1x192x320xf16>
    return %0 : tensor<1x1x192x320xf16>
    //CHECK:   [[INTERPOLATE:%.*]] = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 320]} : tensor<1x1x96x160xf16> -> tensor<1x1x192x320xf16>
    //CHECK:   return [[INTERPOLATE]] : tensor<1x1x192x320xf16>
}

// -----

// CHECK-LABEL: @InterpolatePytorchHalfPixelAssignedClustering
func.func @InterpolatePytorchHalfPixelAssignedClustering(%arg0: tensor<1x1x1x160xf16>) -> tensor<1x1x1x320xf16> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <PYTORCH_HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [1.000000e+00, 2.000000e+00], sizes_attr = [1, 320]} : tensor<1x1x1x160xf16> -> tensor<1x1x1x320xf16>
    return %0 : tensor<1x1x1x320xf16>
    //CHECK:   [[INTERPOLATE:%.*]] = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [1.000000e+00, 2.000000e+00], sizes_attr = [1, 320]} : tensor<1x1x1x160xf16> -> tensor<1x1x1x320xf16>
    //CHECK:   return [[INTERPOLATE]] : tensor<1x1x1x320xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwishAssignedSplitOverHeight
func.func @SwishAssignedSplitOverHeight(%arg0: tensor<1x1x64x64xf16, {order = #NHWC}>) -> tensor<1x1x64x64xf16, {order = #NHWC}> {

    %0 = VPU.Swish(%arg0) : tensor<1x1x64x64xf16, {order = #NHWC}> -> tensor<1x1x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x1x64x64xf16, {order = #NHWC}>

    //CHECK:      [[ResultSwish:%.*]] = VPU.Swish(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
    //CHECK-SAME: tensor<1x1x64x64xf16, {order = #NHWC}> -> tensor<1x1x64x64xf16, {order = #NHWC}>
    //CHECK:      return [[ResultSwish]] : tensor<1x1x64x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwishAssignedSplitOverKernel
func.func @SwishAssignedSplitOverKernel(%arg0: tensor<1x64x1x64xf16, {order = #NHWC}>) -> tensor<1x64x1x64xf16, {order = #NHWC}> {

    %0 = VPU.Swish(%arg0) : tensor<1x64x1x64xf16, {order = #NHWC}> -> tensor<1x64x1x64xf16, {order = #NHWC}>

    return %0 : tensor<1x64x1x64xf16, {order = #NHWC}>

    //CHECK:      [[ResultSwish:%.*]] = VPU.Swish(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} :
    //CHECK-SAME: tensor<1x64x1x64xf16, {order = #NHWC}> -> tensor<1x64x1x64xf16, {order = #NHWC}>
    //CHECK:      return [[ResultSwish]] : tensor<1x64x1x64xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @GeluAssignedSplitOverHeight
func.func @GeluAssignedSplitOverHeight(%arg0: tensor<1x1x4x512xf16, {order = #NCHW}>) -> tensor<1x1x4x512xf16, {order = #NCHW}> {

    %0 = VPU.Gelu(%arg0) : tensor<1x1x4x512xf16, {order = #NCHW}> -> tensor<1x1x4x512xf16, {order = #NCHW}>

    return %0 : tensor<1x1x4x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultGelu:%.*]] = VPU.Gelu(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x4x512xf16, {order = #NCHW}> -> tensor<1x1x4x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultGelu]] : tensor<1x1x4x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @GeluAssignedSplitOverKernel
func.func @GeluAssignedSplitOverKernel(%arg0: tensor<1x4x1x512xf16, {order = #NCHW}>) -> tensor<1x4x1x512xf16, {order = #NCHW}> {

    %0 = VPU.Gelu(%arg0) : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>

    return %0 : tensor<1x4x1x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultGelu:%.*]] = VPU.Gelu(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultGelu]] : tensor<1x4x1x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @HardSigmoidAssignedSplitOverHeight
func.func @HardSigmoidAssignedSplitOverHeight(%arg0: tensor<1x1x4x512xf16, {order = #NCHW}>) -> tensor<1x1x4x512xf16, {order = #NCHW}> {

    %0 = VPU.HardSigmoid(%arg0) {alpha_value = 0.1666259765625 : f64, beta_value = 5.000000e-01 : f64} : tensor<1x1x4x512xf16, {order = #NCHW}> -> tensor<1x1x4x512xf16, {order = #NCHW}>

    return %0 : tensor<1x1x4x512xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.HardSigmoid(%arg0) {alpha_value = 0.1666259765625 : f64, beta_value = 5.000000e-01 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x4x512xf16, {order = #NCHW}> -> tensor<1x1x4x512xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x1x4x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @HardSigmoidAssignedSplitOverKernel
func.func @HardSigmoidAssignedSplitOverKernel(%arg0: tensor<1x4x1x512xf16, {order = #NCHW}>) -> tensor<1x4x1x512xf16, {order = #NCHW}> {

    %0 = VPU.HardSigmoid(%arg0) {alpha_value = 0.1666259765625 : f64, beta_value = 5.000000e-01 : f64} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>

    return %0 : tensor<1x4x1x512xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.HardSigmoid(%arg0) {alpha_value = 0.1666259765625 : f64, beta_value = 5.000000e-01 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x4x1x512xf16, {order = #NCHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CompressConvolutionAssignedSOHOverlapped
func.func @CompressConvolutionAssignedSOHOverlapped(%arg0: tensor<1x3x224x224xf16, {order = #NHWC}>) -> tensor<1x64x112x112xf16, {order = #NHWC}> {
    %weight_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %filter = const.Declare tensor<64x1x1x160xf16, {order = #NHWC}> = dense<1.0> : tensor<64x3x7x7xf16>, [#const.CastElemType<ui8>,
            #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>, #const.SubView<[0, 0, 0, 0], [64, 3, 7, 7]>,
            #const.Reshape<[64, 1, 1, 147]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 13]>]

    %expand = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]}
            : tensor<1x3x224x224xf16, {order = #NHWC}> -> tensor<1x4x224x224xf16, {order = #NHWC}>

    %compress_conv = VPU.NCE.CompressConvolution(%expand, %filter, %weight_table)
            {
                cm_sp_pattern = 15 : i64,
                pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
                ppe = #VPU.PPEStub<>,
                rawFilterShape = [64, 4, 7, 7], strides = [2, 2]
            } -> tensor<1x64x112x112xf16, {order = #NHWC}>

    return %compress_conv : tensor<1x64x112x112xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    //CHECK:        [[FILTER:%.+]] = const.Declare tensor<64x1x1x160xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x3x7x7xf16>
    //CHECK:        [[EXPAND:%.+]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} :
    //CHECK-SAME:           tensor<1x3x224x224xf16, {order = #NHWC}> -> tensor<1x4x224x224xf16, {order = #NHWC}>

    //CHECK:        [[VAL0:%.+]] = VPU.NCE.CompressConvolution([[EXPAND]], [[FILTER]], [[WEIGHTS_TABLE]])
    //CHECK-SAME:           {cm_sp_pattern = 15 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
    //CHECK-SAME:           pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>
    //CHECK-SAME:           ppe = #VPU.PPEStub<>,
    //CHECK-SAME:           rawFilterShape = [64, 4, 7, 7], strides = [2, 2]}
    //CHECK-SAME:           -> tensor<1x64x112x112xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x64x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CompressConvolutionAssignedSOB
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<2x3x224x224xf16, {order = #NHWC}>)
func.func @CompressConvolutionAssignedSOB(%arg0: tensor<2x3x224x224xf16, {order = #NHWC}>) -> tensor<2x64x112x112xf16, {order = #NHWC}> {
    %weight_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %filter = const.Declare tensor<64x1x1x160xf16, {order = #NHWC}> = dense<1.0> : tensor<64x3x7x7xf16>, [#const.CastElemType<ui8>,
            #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>, #const.SubView<[0, 0, 0, 0], [64, 3, 7, 7]>,
            #const.Reshape<[64, 1, 1, 147]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 13]>]

    %expand = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]}
            : tensor<2x3x224x224xf16, {order = #NHWC}> -> tensor<2x4x224x224xf16, {order = #NHWC}>

    %compress_conv = VPU.NCE.CompressConvolution(%expand, %filter, %weight_table) {
        cm_sp_pattern = 7 : i64,
        pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [64, 3, 7, 7], strides = [2, 2]
    } -> tensor<2x64x112x112xf16, {order = #NHWC}>

    return %compress_conv : tensor<2x64x112x112xf16, {order = #NHWC}>

    // CHECK:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    // CHECK:   [[FILTER:%.+]] = const.Declare tensor<64x1x1x160xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x3x7x7xf16>
    // CHECK:   [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} :
    // CHECK-SAME:  tensor<2x3x224x224xf16, {order = #NHWC}> -> tensor<2x4x224x224xf16, {order = #NHWC}>

    // CHECK:   [[VAL0:%.+]] = VPU.NCE.CompressConvolution([[EXPAND]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:   {cm_sp_pattern = 7 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,

    // CHECK:    return [[VAL0]] : tensor<2x64x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CompressConvolutionSOHOverlappedRollback
func.func @CompressConvolutionSOHOverlappedRollback(%arg0: tensor<1x4x60x60xf16, {order = #NHWC}>) -> tensor<1x16x30x30xf16, {order = #NHWC}> {
    %weight_table = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %filter = const.Declare tensor<32x4x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x4x3x3xf16>, [#const.Reorder<#NHWC>]

    %depth_conv_weight_table = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %depth_conv_filter = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x32x1x1xf32>, [#const.CastElemType<f16>, #const.Reshape<[32, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.Reorder<#NHWC>, #const.Reshape<[32, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]

    %conv_weight_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %conv_filter = const.Declare tensor<16x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<16x32x3x3xf16>, [#const.Reorder<#NHWC>]

    %compress_conv = VPU.NCE.CompressConvolution(%arg0, %filter, %weight_table)
        {
            cm_sp_pattern = 7 : i64,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [32, 3, 3, 3], strides = [1, 1]
        } -> tensor<1x32x60x60xf16, {order = #NHWC}>

    %depth_conv = VPU.NCE.DepthConvolution(%compress_conv, %depth_conv_filter, %depth_conv_weight_table)
        {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [32, 1, 1, 1], strides = [1, 1]
        } -> tensor<1x32x60x60xf16, {order = #NHWC}>

    %conv = VPU.NCE.Convolution(%depth_conv, %conv_filter, %conv_weight_table)
        {
            pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 32, 3, 3], strides = [2, 2]
        } -> tensor<1x16x30x30xf16, {order = #NHWC}>


    return %conv : tensor<1x16x30x30xf16, {order = #NHWC}>

    //CHECK:        [[COMPRESS_CONV_WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    //CHECK:        [[COMPRESS_CONV_FILTER:%.+]] = const.Declare tensor<32x4x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x4x3x3xf16>
    //CHECK:        [[DEPTH_CONV_WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    //CHECK:        [[DEPTH_CONV_FILTER:%.+]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x32x1x1xf32>
    //CHECK:        [[CONV_WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    //CHECK:        [[CONV_FILTER:%.+]] = const.Declare tensor<16x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.+]] = VPU.NCE.CompressConvolution(%arg0, [[COMPRESS_CONV_FILTER]], [[COMPRESS_CONV_WEIGHTS_TABLE]])
    //CHECK-SAME:           {cm_sp_pattern = 7 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,

    //CHECK:        [[VAL1:%.+]] = VPU.NCE.DepthConvolution([[VAL0]], [[DEPTH_CONV_FILTER]], [[DEPTH_CONV_WEIGHTS_TABLE]])
    //CHECK-SAME:           {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

    //CHECK:        [[VAL2:%.+]] = VPU.NCE.Convolution([[VAL1]], [[CONV_FILTER]], [[CONV_WEIGHTS_TABLE]])
    //CHECK-SAME:           {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

    //CHECK:        return [[VAL2]] : tensor<1x16x30x30xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MultiplyAssignedSplitOverHeight
func.func @MultiplyAssignedSplitOverHeight(%arg0: tensor<1x32x44x44xf16, {order = #NHWC}>,
            %arg1: tensor<1x1x44x44xf16, {order = #NHWC}>) -> tensor<1x32x44x44xf16, {order = #NHWC}> {

    %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
                tensor<1x32x44x44xf16, {order = #NHWC}>,
                tensor<1x1x44x44xf16, {order = #NHWC}> -> tensor<1x32x44x44xf16, {order = #NHWC}>

    return %0 : tensor<1x32x44x44xf16, {order = #NHWC}>

    //CHECK:      [[MULTIPLY:%.*]] = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x44x44xf16, {order = #NHWC}>, tensor<1x1x44x44xf16, {order = #NHWC}> -> tensor<1x32x44x44xf16, {order = #NHWC}>
    //CHECK:      return [[MULTIPLY]] : tensor<1x32x44x44xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MultiplyAssignedSplitOverKernel
func.func @MultiplyAssignedSplitOverKernel(%arg0: tensor<1x32x1x44xf16, {order = #NHWC}>,
            %arg1: tensor<1x1x1x44xf16, {order = #NHWC}>) -> tensor<1x32x1x44xf16, {order = #NHWC}> {

    %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
                tensor<1x32x1x44xf16, {order = #NHWC}>,
                tensor<1x1x1x44xf16, {order = #NHWC}> -> tensor<1x32x1x44xf16, {order = #NHWC}>

    return %0 : tensor<1x32x1x44xf16, {order = #NHWC}>

    //CHECK:      [[MULTIPLY:%.*]] = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x44xf16, {order = #NHWC}>, tensor<1x1x1x44xf16, {order = #NHWC}> -> tensor<1x32x1x44xf16, {order = #NHWC}>
    //CHECK:      return [[MULTIPLY]] : tensor<1x32x1x44xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MultiplyAssignedClustering
func.func @MultiplyAssignedClustering(%arg0: tensor<1x1x1x44xf16, {order = #NHWC}>,
            %arg1: tensor<1x1x1x1xf16, {order = #NHWC}>) -> tensor<1x1x1x44xf16, {order = #NHWC}> {

    %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
                tensor<1x1x1x44xf16, {order = #NHWC}>,
                tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x44xf16, {order = #NHWC}>

    return %0 : tensor<1x1x1x44xf16, {order = #NHWC}>

    //CHECK:      [[MULTIPLY:%.*]] = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x44xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x44xf16, {order = #NHWC}>
    //CHECK:      return [[MULTIPLY]] : tensor<1x1x1x44xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @HSwishAssignedSplitOverHeight
func.func @HSwishAssignedSplitOverHeight(%arg0: tensor<1x1x4x512xf16, {order = #NCHW}>) -> tensor<1x1x4x512xf16, {order = #NCHW}> {

    %0 = VPU.HSwish(%arg0) : tensor<1x1x4x512xf16, {order = #NCHW}> -> tensor<1x1x4x512xf16, {order = #NCHW}>

    return %0 : tensor<1x1x4x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultHSwish:%.*]] = VPU.HSwish(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x4x512xf16, {order = #NCHW}> -> tensor<1x1x4x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultHSwish]] : tensor<1x1x4x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @HSwishAssignedSplitOverKernel
func.func @HSwishAssignedSplitOverKernel(%arg0: tensor<1x4x1x512xf16, {order = #NCHW}>) -> tensor<1x4x1x512xf16, {order = #NCHW}> {

    %0 = VPU.HSwish(%arg0) : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>

    return %0 : tensor<1x4x1x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultHSwish:%.*]] = VPU.HSwish(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultHSwish]] : tensor<1x4x1x512xf16, {order = #NCHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateNearestAssignedSOH
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x64x10x10xf16, {order = #NHWC}>)
func.func @InterpolateNearestAssignedSOH(%input_data: tensor<1x64x10x10xf16, {order = #NHWC}>) -> tensor<1x64x20x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x64x20x20xi1> = dense<1> : tensor<1x64x20x20xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 64, dataShape = [1, 64, 10, 10],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 20, 20]>
    } -> tensor<1x1x20x20xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%input_data, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            nearest_mode = <FLOOR>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            mode = <NEAREST>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 20, 20]>
    } -> !VPU.SparseTensor<data=tensor<1x64x10x10xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x64x20x20xi1>,
                           storage_element_table=tensor<1x1x20x20xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 20, 20]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [64, 64, 1, 1],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        scales_attr = [2, 2]
    } -> tensor<1x64x20x20xf16, {order = #NHWC}>

    return %interpolate : tensor<1x64x20x20xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x64x20x20xi1> = dense<true> : tensor<1x64x20x20xi1>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      rawFilterShape = [64, 64, 1, 1],
    // CHECK-SAME:      scales_attr = [2, 2]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x64x20x20xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x64x20x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateNearestAssignedSOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x256x5x10xf16, {order = #NHWC}>)
func.func @InterpolateNearestAssignedSOK(%input_data: tensor<1x256x5x10xf16, {order = #NHWC}>) -> tensor<1x256x10x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<256x256x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x256x10x20xi1> = dense<1> : tensor<1x256x10x20xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 256, dataShape = [1, 256, 5, 10],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 256, 10, 20]>
    } -> tensor<1x1x10x20xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%input_data, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            nearest_mode = <FLOOR>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            mode = <NEAREST>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 256, 10, 20]>
    } -> !VPU.SparseTensor<data=tensor<1x256x5x10xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x256x10x20xi1>,
                           storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 256, 10, 20]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [256, 256, 1, 1],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        scales_attr = [2, 2]
    } -> tensor<1x256x10x20xf16, {order = #NHWC}>

    return %interpolate : tensor<1x256x10x20xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<256x256x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x256x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x256x10x20xi1> = dense<true> : tensor<1x256x10x20xi1>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      rawFilterShape = [256, 256, 1, 1],
    // CHECK-SAME:      scales_attr = [2, 2]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x256x10x20xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x256x10x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateBilinearAssignedSOH
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x64x20x20xf16, {order = #NHWC}>)
func.func @InterpolateBilinearAssignedSOH(%input_data: tensor<1x64x20x20xf16, {order = #NHWC}>) -> tensor<1x64x40x40xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x64x2x2xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x2x2xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x64x41x41xi1> = dense<1> : tensor<1x64x41x41xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 64, dataShape = [1, 64, 10, 10],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], offsets = [0, 0, 0, 0], sizes = [1, 64, 41, 41]>
    } -> tensor<1x1x41x41xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%input_data, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 41, 41]>
    } -> !VPU.SparseTensor<data=tensor<1x64x20x20xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x64x41x41xi1>,
                           storage_element_table=tensor<1x1x41x41xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], offsets = [0, 0, 0, 0], sizes = [1, 64, 41, 41]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [64, 64, 2, 2],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [2, 2]
    } -> tensor<1x64x40x40xf16, {order = #NHWC}>

    return %interpolate : tensor<1x64x40x40xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<64x64x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x2x2xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x64x41x41xi1> = dense<true> : tensor<1x64x41x41xi1>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      rawFilterShape = [64, 64, 2, 2],
    // CHECK-SAME:      scales_attr = [2, 2],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x64x40x40xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x64x40x40xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateBilinearAssignedSOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x256x5x10xf16, {order = #NHWC}>)
func.func @InterpolateBilinearAssignedSOK(%input_data: tensor<1x256x5x10xf16, {order = #NHWC}>) -> tensor<1x256x10x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x256x2x2xf16, {order = #NHWC}> = dense<1.0> : tensor<256x256x2x2xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x256x11x21xi1> = dense<1> : tensor<1x256x11x21xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 64, dataShape = [1, 64, 5, 10],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], offsets = [0, 0, 0, 0], sizes = [1, 64, 11, 21]>
    } -> tensor<1x1x11x21xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%input_data, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 256, 11, 21]>
    } -> !VPU.SparseTensor<data=tensor<1x256x5x10xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x256x11x21xi1>,
                           storage_element_table=tensor<1x1x11x21xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], offsets = [0, 0, 0, 0], sizes = [1, 256, 11, 21]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [256, 256, 2, 2],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [2, 2]
    } -> tensor<1x256x10x20xf16, {order = #NHWC}>

    return %interpolate : tensor<1x256x10x20xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<256x256x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x256x2x2xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x256x11x21xi1> = dense<true> : tensor<1x256x11x21xi1>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      rawFilterShape = [256, 256, 2, 2],
    // CHECK-SAME:      scales_attr = [2, 2]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x256x10x20xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x256x10x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @SparseTensorDataAlignmentCheckSOK
func.func @SparseTensorDataAlignmentCheckSOK(%arg0: tensor<1x64x5x5xf16, {order = #NHWC}>) -> tensor<1x64x10x10xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x64x4x4xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x4x4xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x64x22x22xi1> = dense<1> : tensor<1x64x22x22xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 64, dataShape = [1, 64, 5, 5],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                                    scale = [1.0, 1.0, 2.0, 2.0], offsets = [0, 0, 0, 0], sizes = [1, 64, 22, 22]>
    } -> tensor<1x1x22x22xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
            scale = [1.0, 1.0, 4.0, 4.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 22, 22]>
    } -> !VPU.SparseTensor<data=tensor<1x64x5x5xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x64x22x22xi1>,
                           storage_element_table=tensor<1x1x22x22xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                                              scale = [1.0, 1.0, 4.0, 4.0], offsets = [0, 0, 0, 0], sizes = [1, 64, 22, 22]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [64, 64, 4, 4],
        strides = [2, 2],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [4, 4]
    } -> tensor<1x64x10x10xf16, {order = #NHWC}>

    return %interpolate : tensor<1x64x10x10xf16, {order = #NHWC}>

    // CHECK:   [[WEIGHTS:%.+]] = const.Declare tensor<64x64x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x4x4xf16>, [#const.Reorder<#NHWC>]
    // CHECK:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    // CHECK:   [[INPUT_SM:%.+]] = const.Declare tensor<1x64x22x22xi1> = dense<true> : tensor<1x64x22x22xi1>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      rawFilterShape = [64, 64, 4, 4],
    // CHECK-SAME:      scales_attr = [4, 4], strides = [2, 2]}
    // CHECK-SAME:      -> tensor<1x64x10x10xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x64x10x10xf16, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @SparseTensorDataAlignmentCheckSOH
func.func @SparseTensorDataAlignmentCheckSOH(%arg0: tensor<1x16x5x5xf16, {order = #NHWC}>) -> tensor<1x16x10x10xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x4x4xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x4x4xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x16x22x22xi1> = dense<1> : tensor<1x16x22x22xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 16, dataShape = [1, 16, 5, 5],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                                    scale = [1.0, 1.0, 2.0, 2.0], offsets = [0, 0, 0, 0], sizes = [1, 16, 22, 22]>
    } -> tensor<1x1x22x22xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
            scale = [1.0, 1.0, 4.0, 4.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 16, 22, 22]>
    } -> !VPU.SparseTensor<data=tensor<1x16x5x5xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x16x22x22xi1>,
                           storage_element_table=tensor<1x1x22x22xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                                              scale = [1.0, 1.0, 4.0, 4.0], offsets = [0, 0, 0, 0], sizes = [1, 16, 22, 22]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [16, 16, 4, 4],
        strides = [2, 2],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [4, 4]
    } -> tensor<1x16x10x10xf16, {order = #NHWC}>

    return %interpolate : tensor<1x16x10x10xf16, {order = #NHWC}>

    // CHECK:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x4x4xf16>, [#const.Reorder<#NHWC>]
    // CHECK:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK:   [[INPUT_SM:%.+]] = const.Declare tensor<1x16x22x22xi1> = dense<true> : tensor<1x16x22x22xi1>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      rawFilterShape = [16, 16, 4, 4],
    // CHECK-SAME:      scales_attr = [4, 4], strides = [2, 2]}
    // CHECK-SAME:      -> tensor<1x16x10x10xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x16x10x10xf16, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// Check a bug in calculating tiled sep tensor data, details in E#104795
// CHECK-LABEL: @SEPInterpolateAvoidTillingBug
func.func @SEPInterpolateAvoidTillingBug(%arg0: tensor<1x16x160x160xf16, {order = #NHWC}>) -> tensor<1x16x320x320xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x4x4xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x4x4xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x16x642x642xi1> = dense<1> : tensor<1x16x642x642xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 16, dataShape = [1, 16, 5, 5],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                                    scale = [1.0, 1.0, 2.0, 2.0], offsets = [0, 0, 0, 0], sizes = [1, 16, 642, 642]>
    } -> tensor<1x1x642x642xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
            scale = [1.0, 1.0, 4.0, 4.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 16, 642, 642]>
    } -> !VPU.SparseTensor<data=tensor<1x16x160x160xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x16x642x642xi1>,
                           storage_element_table=tensor<1x1x642x642xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                                              scale = [1.0, 1.0, 4.0, 4.0], offsets = [0, 0, 0, 0], sizes = [1, 16, 642, 642]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [16, 16, 4, 4],
        strides = [2, 2],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [4, 4]
    } -> tensor<1x16x320x320xf16, {order = #NHWC}>

    return %interpolate : tensor<1x16x320x320xf16, {order = #NHWC}>

    // CHECK:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x4x4xf16>, [#const.Reorder<#NHWC>]
    // CHECK:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK:   [[INPUT_SM:%.+]] = const.Declare tensor<1x16x642x642xi1> = dense<true> : tensor<1x16x642x642xi1>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      rawFilterShape = [16, 16, 4, 4],
    // CHECK-SAME:      scales_attr = [4, 4], strides = [2, 2]}
    // CHECK-SAME:      -> tensor<1x16x320x320xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x16x320x320xf16, {order = #NHWC}>

}

// -----

// CHECK-LABEL:   func.func @ConvertOpAssignedMCStratF32To16
func.func @ConvertOpAssignedMCStratF32To16(%arg0: tensor<1x48x160x80xf32>) -> tensor<1x48x160x80xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
    return %0 : tensor<1x48x160x80xf16>
    // CHECK:       [[CONVERT:%.+]] = VPU.Convert(%arg0) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
    // CHECK:       return [[CONVERT]] : tensor<1x48x160x80xf16>
}

// -----

// CHECK-LABEL:   func.func @ConvertOpAssignedMCStratF16To32
func.func @ConvertOpAssignedMCStratF16To32(%arg0: tensor<1x48x160x80xf16>) -> tensor<1x48x160x80xf32> {
    %0 = VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x48x160x80xf16> -> tensor<1x48x160x80xf32>
    return %0 : tensor<1x48x160x80xf32>
    // CHECK:       [[CONVERT:%.+]] = VPU.Convert(%arg0) {dstElemType = f32, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x48x160x80xf16> -> tensor<1x48x160x80xf32>
    // CHECK:       return [[CONVERT]] : tensor<1x48x160x80xf32>
}

// -----

// CHECK-LABEL:   func.func @ConvertOpAssignedMCStratClustering
func.func @ConvertOpAssignedMCStratClustering(%arg0: tensor<2x48x160x80xf16>) -> tensor<2x48x160x80xf32> {
    %0 = VPU.Convert(%arg0) {dstElemType = f32} : tensor<2x48x160x80xf16> -> tensor<2x48x160x80xf32>
    return %0 : tensor<2x48x160x80xf32>
    // CHECK:       [[CONVERT:%.+]] = VPU.Convert(%arg0) {dstElemType = f32, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<2x48x160x80xf16> -> tensor<2x48x160x80xf32>
    // CHECK:       return [[CONVERT]] : tensor<2x48x160x80xf32>
}

// -----

// CHECK-LABEL:   func.func @ConvertOpAssignedMCStratSOK
func.func @ConvertOpAssignedMCStratSOK(%arg0: tensor<1x48x3x3xf16>) -> tensor<1x48x3x3xf32> {
    %0 = VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x48x3x3xf16> -> tensor<1x48x3x3xf32>
    return %0 : tensor<1x48x3x3xf32>
    // CHECK:       [[CONVERT:%.+]] = VPU.Convert(%arg0) {dstElemType = f32, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x48x3x3xf16> -> tensor<1x48x3x3xf32>
    // CHECK:       return [[CONVERT]] : tensor<1x48x3x3xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AssignSOHForLayerWithLargeActivation
func.func @AssignSOHForLayerWithLargeActivation(%arg0: tensor<1x64x250x250xf16, {order = #NHWC}>) -> tensor<1x32x250x250xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_0 = const.Declare tensor<32x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 64, 3, 3], strides = [1, 1]} -> tensor<1x32x250x250xf16, {order = #NHWC}>
    return %0 : tensor<1x32x250x250xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]])
    //CHECK-SAME:        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    //CHECK-SAME:        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    //CHECK-SAME:        ppe = #VPU.PPEStub<>,
    //CHECK-SAME:        rawFilterShape = [32, 64, 3, 3], strides = [1, 1]
    //CHECK-SAME:      -> tensor<1x32x250x250xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x32x250x250xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @AssignSOHForLayerWithTopK
func.func @AssignSOHForLayerWithTopK(%arg0: tensor<1x151x513x513xf16>) -> tensor<1x1x513x513xsi32> {
    %output_values, %target_shape = VPU.TopK(%arg0)
        {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, operandSegmentSizes = array<i32: 1, 0, 0>, sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<1x151x513x513xf16> -> tensor<1x1x513x513xf16>, tensor<1x1x513x513xsi32>
    return %target_shape : tensor<1x1x513x513xsi32>

    //CHECK:        [[OUTPUT_VALUES:%.+]], [[TARGET_SHAPE:%.+]] = VPU.TopK(%arg0)
    //CHECK-SAME:        axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>,
    //CHECK-SAME:        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, operandSegmentSizes = array<i32: 1, 0, 0>, sort = #IE.topk_sort_type<SORT_INDICES>}
    //CHECK-SAME:        : tensor<1x151x513x513xf16> -> tensor<1x1x513x513xf16>, tensor<1x1x513x513xsi32>
    //CHECK:        return [[TARGET_SHAPE]] : tensor<1x1x513x513xsi32>
}

// -----

// CHECK-LABEL: @AssignSOKForLayerWithTopK
func.func @AssignSOKForLayerWithTopK(%arg0: tensor<1x151x1x513xf16>) -> tensor<1x151x1x1xsi32> {
    %output_values, %target_shape = VPU.TopK(%arg0)
        {axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, operandSegmentSizes = array<i32: 1, 0, 0>, sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<1x151x1x513xf16> -> tensor<1x151x1x1xf16>, tensor<1x151x1x1xsi32>
    return %target_shape : tensor<1x151x1x1xsi32>

    //CHECK:        [[OUTPUT_VALUES:%.+]], [[TARGET_SHAPE:%.+]] = VPU.TopK(%arg0)
    //CHECK-SAME:        axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>,
    //CHECK-SAME:        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, operandSegmentSizes = array<i32: 1, 0, 0>, sort = #IE.topk_sort_type<SORT_INDICES>}
    //CHECK-SAME:        : tensor<1x151x1x513xf16> -> tensor<1x151x1x1xf16>, tensor<1x151x1x1xsi32>
    //CHECK:        return [[TARGET_SHAPE]] : tensor<1x151x1x1xsi32>
}

// CHECK-LABEL: @AssignClusteringForLayerWithNormalizeL2
func.func @AssignClusteringForLayerWithNormalizeL2(%arg0: tensor<1x151x513x513xf16>) -> tensor<1x151x513x513xf16> {
    %output_values = VPU.NormalizeL2(%arg0) {axes_value = [1,2,3], eps = 9.9999999392252903E-9 : f64, eps_mode = #IE.eps_mode<ADD>} : tensor<1x151x513x513xf16> -> tensor<1x151x513x513xf16>
    return %output_values : tensor<1x151x513x513xf16>

    //CHECK:        [[OUTPUT_VALUES:%.+]] = VPU.NormalizeL2(%arg0)
    //CHECK-SAME:        axes_value = [1, 2, 3], eps = 9.9999999392252903E-9 : f64, eps_mode = #IE.eps_mode<ADD>,
    //CHECK-SAME:        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
    //CHECK-SAME:        : tensor<1x151x513x513xf16> -> tensor<1x151x513x513xf16>
    //CHECK:        return [[OUTPUT_VALUES]] : tensor<1x151x513x513xf16>
}

// CHECK-LABEL: @AssignClustering2ForLayerWithNormalizeL2
func.func @AssignClustering2ForLayerWithNormalizeL2(%arg0: tensor<1x1x513x513xf16>) -> tensor<1x1x513x513xf16> {
    %output_values = VPU.NormalizeL2(%arg0) {axes_value = [2,3], eps = 9.9999999392252903E-9 : f64, eps_mode = #IE.eps_mode<ADD>} : tensor<1x1x513x513xf16> -> tensor<1x1x513x513xf16>
    return %output_values : tensor<1x1x513x513xf16>

    //CHECK:        [[OUTPUT_VALUES:%.+]] = VPU.NormalizeL2(%arg0)
    //CHECK-SAME:        axes_value = [2, 3], eps = 9.9999999392252903E-9 : f64, eps_mode = #IE.eps_mode<ADD>,
    //CHECK-SAME:        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
    //CHECK-SAME:        : tensor<1x1x513x513xf16> -> tensor<1x1x513x513xf16>
    //CHECK:        return [[OUTPUT_VALUES]] : tensor<1x1x513x513xf16>
}

// CHECK-LABEL: @AssignSOHForLayerWithNormalizeL2
func.func @AssignSOHForLayerWithNormalizeL2(%arg0: tensor<1x1x513x513xf16>) -> tensor<1x1x513x513xf16> {
    %output_values = VPU.NormalizeL2(%arg0) {axes_value = [3], eps = 9.9999999392252903E-9 : f64, eps_mode = #IE.eps_mode<ADD>} : tensor<1x1x513x513xf16> -> tensor<1x1x513x513xf16>
    return %output_values : tensor<1x1x513x513xf16>

    //CHECK:        [[OUTPUT_VALUES:%.+]] = VPU.NormalizeL2(%arg0)
    //CHECK-SAME:        axes_value = [3], eps = 9.9999999392252903E-9 : f64, eps_mode = #IE.eps_mode<ADD>,
    //CHECK-SAME:        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:        : tensor<1x1x513x513xf16> -> tensor<1x1x513x513xf16>
    //CHECK:        return [[OUTPUT_VALUES]] : tensor<1x1x513x513xf16>
}

// -----

// CHECK-LABEL: @AssignSOKForLayerWithNormalizeL2
func.func @AssignSOKForLayerWithNormalizeL2(%arg0: tensor<1x151x513x513xf16>) -> tensor<1x151x513x513xf16> {
    %output_values = VPU.NormalizeL2(%arg0) {axes_value = [2,3], eps = 9.9999999392252903E-9 : f64, eps_mode = #IE.eps_mode<MAX>} : tensor<1x151x513x513xf16> -> tensor<1x151x513x513xf16>
    return %output_values : tensor<1x151x513x513xf16>

    //CHECK:        [[OUTPUT_VALUES:%.+]] = VPU.NormalizeL2(%arg0)
    //CHECK-SAME:        axes_value = [2, 3], eps = 9.9999999392252903E-9 : f64, eps_mode = #IE.eps_mode<MAX>,
    //CHECK-SAME:        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    //CHECK-SAME:        : tensor<1x151x513x513xf16> -> tensor<1x151x513x513xf16>
    //CHECK:        return [[OUTPUT_VALUES]] : tensor<1x151x513x513xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoAssignHKSwitchForLayerWithTiling
func.func @NoAssignHKSwitchForLayerWithTiling(%arg0: tensor<1x128x64x128xf16, {order = #NHWC}>) -> tensor<1x64x64x128xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<64x128x4x4xf16, {order = #NHWC}> = dense<1.0> : tensor<64x128x4x4xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst_1)
             {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 2 : i64, right = 1 : i64, top = 2 : i64, bottom = 1 : i64>,
              rawFilterShape = [64, 128, 4, 4], strides = [1, 1]} -> tensor<1x64x64x128xf16, {order = #NHWC}>

    return %0 : tensor<1x64x64x128xf16, {order = #NHWC}>

    //CHECK:             VPU.NCE.Convolution
    //CHECK-SAME:        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    //CHECK-NOT:         multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SigmoidAssignedSplitOverKernel
func.func @SigmoidAssignedSplitOverKernel(%arg0: tensor<1x16x1x513xf16, {order = #NCHW}>) -> tensor<1x16x1x513xf16, {order = #NCHW}> {

    %0 = VPU.Sigmoid(%arg0) : tensor<1x16x1x513xf16, {order = #NCHW}> -> tensor<1x16x1x513xf16, {order = #NCHW}>

    return %0 : tensor<1x16x1x513xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Sigmoid(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x16x1x513xf16, {order = #NCHW}> -> tensor<1x16x1x513xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x16x1x513xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SigmoidAssignedSplitOverHeight
func.func @SigmoidAssignedSplitOverHeight(%arg0: tensor<1x1x16x513xf16, {order = #NCHW}>) -> tensor<1x1x16x513xf16, {order = #NCHW}> {

    %0 = VPU.Sigmoid(%arg0) : tensor<1x1x16x513xf16, {order = #NCHW}> -> tensor<1x1x16x513xf16, {order = #NCHW}>

    return %0 : tensor<1x1x16x513xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Sigmoid(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x16x513xf16, {order = #NCHW}> -> tensor<1x1x16x513xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x1x16x513xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @LogAssignedSplitOverKernel
func.func @LogAssignedSplitOverKernel(%arg0: tensor<1x16x1x513xf16, {order = #NCHW}>) -> tensor<1x16x1x513xf16, {order = #NCHW}> {

    %0 = VPU.Log(%arg0) : tensor<1x16x1x513xf16, {order = #NCHW}> -> tensor<1x16x1x513xf16, {order = #NCHW}>

    return %0 : tensor<1x16x1x513xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Log(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x16x1x513xf16, {order = #NCHW}> -> tensor<1x16x1x513xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x16x1x513xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @LogAssignedSplitOverHeight
func.func @LogAssignedSplitOverHeight(%arg0: tensor<1x1x16x513xf16, {order = #NCHW}>) -> tensor<1x1x16x513xf16, {order = #NCHW}> {

    %0 = VPU.Log(%arg0) : tensor<1x1x16x513xf16, {order = #NCHW}> -> tensor<1x1x16x513xf16, {order = #NCHW}>

    return %0 : tensor<1x1x16x513xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Log(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x16x513xf16, {order = #NCHW}> -> tensor<1x1x16x513xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x1x16x513xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SqrtAssignedSplitOverKernel
func.func @SqrtAssignedSplitOverKernel(%arg0: tensor<1x16x1x513xf16, {order = #NCHW}>) -> tensor<1x16x1x513xf16, {order = #NCHW}> {

    %0 = VPU.Sqrt(%arg0) : tensor<1x16x1x513xf16, {order = #NCHW}> -> tensor<1x16x1x513xf16, {order = #NCHW}>

    return %0 : tensor<1x16x1x513xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Sqrt(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x16x1x513xf16, {order = #NCHW}> -> tensor<1x16x1x513xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x16x1x513xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SqrtAssignedSplitOverHeight
func.func @SqrtAssignedSplitOverHeight(%arg0: tensor<1x1x16x513xf16, {order = #NCHW}>) -> tensor<1x1x16x513xf16, {order = #NCHW}> {

    %0 = VPU.Sqrt(%arg0) : tensor<1x1x16x513xf16, {order = #NCHW}> -> tensor<1x1x16x513xf16, {order = #NCHW}>

    return %0 : tensor<1x1x16x513xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Sqrt(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x16x513xf16, {order = #NCHW}> -> tensor<1x1x16x513xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x1x16x513xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @LogAssignedSplitOverKernel
func.func @LogAssignedSplitOverKernel(%arg0: tensor<1x16x1x513xf32, {order = #NCHW}>) -> tensor<1x16x1x513xf32, {order = #NCHW}> {

    %0 = VPU.Log(%arg0) : tensor<1x16x1x513xf32, {order = #NCHW}> -> tensor<1x16x1x513xf32, {order = #NCHW}>

    return %0 : tensor<1x16x1x513xf32, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Log(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x16x1x513xf32, {order = #NCHW}> -> tensor<1x16x1x513xf32, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x16x1x513xf32, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @LogAssignedSplitOverHeight
func.func @LogAssignedSplitOverHeight(%arg0: tensor<1x1x16x513xf32, {order = #NCHW}>) -> tensor<1x1x16x513xf32, {order = #NCHW}> {

    %0 = VPU.Log(%arg0) : tensor<1x1x16x513xf32, {order = #NCHW}> -> tensor<1x1x16x513xf32, {order = #NCHW}>

    return %0 : tensor<1x1x16x513xf32, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Log(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x16x513xf32, {order = #NCHW}> -> tensor<1x1x16x513xf32, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x1x16x513xf32, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SqrtAssignedSplitOverKernel
func.func @SqrtAssignedSplitOverKernel(%arg0: tensor<1x16x1x513xf32, {order = #NCHW}>) -> tensor<1x16x1x513xf32, {order = #NCHW}> {

    %0 = VPU.Sqrt(%arg0) : tensor<1x16x1x513xf32, {order = #NCHW}> -> tensor<1x16x1x513xf32, {order = #NCHW}>

    return %0 : tensor<1x16x1x513xf32, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Sqrt(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x16x1x513xf32, {order = #NCHW}> -> tensor<1x16x1x513xf32, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x16x1x513xf32, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SqrtAssignedSplitOverHeight
func.func @SqrtAssignedSplitOverHeight(%arg0: tensor<1x1x16x513xf32, {order = #NCHW}>) -> tensor<1x1x16x513xf32, {order = #NCHW}> {

    %0 = VPU.Sqrt(%arg0) : tensor<1x1x16x513xf32, {order = #NCHW}> -> tensor<1x1x16x513xf32, {order = #NCHW}>

    return %0 : tensor<1x1x16x513xf32, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Sqrt(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x16x513xf32, {order = #NCHW}> -> tensor<1x1x16x513xf32, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x1x16x513xf32, {order = #NCHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthToSpaceAssignedSplitOverHeight
func.func @DepthToSpaceAssignedSplitOverHeight(%arg0: tensor<1x128x180x270xf16, {order = #NHWC}>) -> tensor<1x8x720x1080xf16, {order = #NHWC}> {

    %0 = VPU.DepthToSpace(%arg0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x128x180x270xf16, {order = #NHWC}> -> tensor<1x8x720x1080xf16, {order = #NHWC}>

    return %0 : tensor<1x8x720x1080xf16, {order = #NHWC}>

    //CHECK:   [[Result:%.*]] = VPU.DepthToSpace(%arg0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x128x180x270xf16, {order = #NHWC}> -> tensor<1x8x720x1080xf16, {order = #NHWC}>
    //CHECK:   return [[Result]] : tensor<1x8x720x1080xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthToSpaceAssignedSplitOverWidth
func.func @DepthToSpaceAssignedSplitOverWidth(%arg0: tensor<1x128x1x270xf16, {order = #NHWC}>) -> tensor<1x8x4x1080xf16, {order = #NHWC}> {

    %0 = VPU.DepthToSpace(%arg0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x128x1x270xf16, {order = #NHWC}> -> tensor<1x8x4x1080xf16, {order = #NHWC}>

    return %0 : tensor<1x8x4x1080xf16, {order = #NHWC}>

    //CHECK:   [[Result:%.*]] = VPU.DepthToSpace(%arg0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>} : tensor<1x128x1x270xf16, {order = #NHWC}> -> tensor<1x8x4x1080xf16, {order = #NHWC}>
    //CHECK:   return [[Result]] : tensor<1x8x4x1080xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MinimumAssignedSplitOverHeight
func.func @MinimumAssignedSplitOverHeight(%arg0: tensor<1x1x44x44xf16, {order = #NCHW}>,
            %arg1: tensor<1x1x44x44xf16, {order = #NCHW}>) -> tensor<1x1x44x44xf16, {order = #NCHW}> {

    %0 = VPU.Minimum(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
                tensor<1x1x44x44xf16, {order = #NCHW}>,
                tensor<1x1x44x44xf16, {order = #NCHW}> -> tensor<1x1x44x44xf16, {order = #NCHW}>

    return %0 : tensor<1x1x44x44xf16, {order = #NCHW}>

    //CHECK:      [[MINIMUM:%.*]] = VPU.Minimum({{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x44x44xf16, {order = #NCHW}>, tensor<1x1x44x44xf16, {order = #NCHW}> -> tensor<1x1x44x44xf16, {order = #NCHW}>
    //CHECK:      return [[MINIMUM]] : tensor<1x1x44x44xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MinimumAssignedSplitOverKernel
func.func @MinimumAssignedSplitOverKernel(%arg0: tensor<1x44x1x44xf16, {order = #NCHW}>,
            %arg1: tensor<1x44x1x44xf16, {order = #NCHW}>) -> tensor<1x44x1x44xf16, {order = #NCHW}> {

    %0 = VPU.Minimum(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
                tensor<1x44x1x44xf16, {order = #NCHW}>,
                tensor<1x44x1x44xf16, {order = #NCHW}> -> tensor<1x44x1x44xf16, {order = #NCHW}>

    return %0 : tensor<1x44x1x44xf16, {order = #NCHW}>

    //CHECK:      [[MINIMUM:%.*]] = VPU.Minimum({{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x44x1x44xf16, {order = #NCHW}>, tensor<1x44x1x44xf16, {order = #NCHW}> -> tensor<1x44x1x44xf16, {order = #NCHW}>
    //CHECK:      return [[MINIMUM]] : tensor<1x44x1x44xf16, {order = #NCHW}>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MinimumAssignedClustering
func.func @MinimumAssignedClustering(%arg0: tensor<1x1x1x1xf16, {order = #NCHW}>,
            %arg1: tensor<1x1x1x1xf16, {order = #NCHW}>) -> tensor<1x1x1x1xf16, {order = #NCHW}> {

    %0 = VPU.Minimum(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
                tensor<1x1x1x1xf16, {order = #NCHW}>,
                tensor<1x1x1x1xf16, {order = #NCHW}> -> tensor<1x1x1x1xf16, {order = #NCHW}>

    return %0 : tensor<1x1x1x1xf16, {order = #NCHW}>

    //CHECK:      [[MINIMUM:%.*]] = VPU.Minimum({{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x1xf16, {order = #NCHW}>, tensor<1x1x1x1xf16, {order = #NCHW}> -> tensor<1x1x1x1xf16, {order = #NCHW}>
    //CHECK:      return [[MINIMUM]] : tensor<1x1x1x1xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MaximumAssignedSplitOverHeight
func.func @MaximumAssignedSplitOverHeight(%arg0: tensor<1x1x44x44xf16, {order = #NCHW}>,
            %arg1: tensor<1x1x44x44xf16, {order = #NCHW}>) -> tensor<1x1x44x44xf16, {order = #NCHW}> {

    %0 = VPU.Maximum(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
                tensor<1x1x44x44xf16, {order = #NCHW}>,
                tensor<1x1x44x44xf16, {order = #NCHW}> -> tensor<1x1x44x44xf16, {order = #NCHW}>

    return %0 : tensor<1x1x44x44xf16, {order = #NCHW}>

    //CHECK:      [[MAXIMUM:%.*]] = VPU.Maximum(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x44x44xf16, {order = #NCHW}>, tensor<1x1x44x44xf16, {order = #NCHW}> -> tensor<1x1x44x44xf16, {order = #NCHW}>
    //CHECK:      return [[MAXIMUM]] : tensor<1x1x44x44xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MaximumAssignedSplitOverKernel
func.func @MaximumAssignedSplitOverKernel(%arg0: tensor<1x44x1x44xf16, {order = #NCHW}>,
            %arg1: tensor<1x44x1x44xf16, {order = #NCHW}>) -> tensor<1x44x1x44xf16, {order = #NCHW}> {

    %0 = VPU.Maximum(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
                tensor<1x44x1x44xf16, {order = #NCHW}>,
                tensor<1x44x1x44xf16, {order = #NCHW}> -> tensor<1x44x1x44xf16, {order = #NCHW}>

    return %0 : tensor<1x44x1x44xf16, {order = #NCHW}>

    //CHECK:      [[MAXIMUM:%.*]] = VPU.Maximum(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x44x1x44xf16, {order = #NCHW}>, tensor<1x44x1x44xf16, {order = #NCHW}> -> tensor<1x44x1x44xf16, {order = #NCHW}>
    //CHECK:      return [[MAXIMUM]] : tensor<1x44x1x44xf16, {order = #NCHW}>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MaximumAssignedClustering
func.func @MaximumAssignedClustering(%arg0: tensor<1x1x1x1xf16, {order = #NCHW}>,
            %arg1: tensor<1x1x1x1xf16, {order = #NCHW}>) -> tensor<1x1x1x1xf16, {order = #NCHW}> {

    %0 = VPU.Maximum(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
                tensor<1x1x1x1xf16, {order = #NCHW}>,
                tensor<1x1x1x1xf16, {order = #NCHW}> -> tensor<1x1x1x1xf16, {order = #NCHW}>

    return %0 : tensor<1x1x1x1xf16, {order = #NCHW}>

    //CHECK:      [[MAXIMUM:%.*]] = VPU.Maximum(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x1xf16, {order = #NCHW}>, tensor<1x1x1x1xf16, {order = #NCHW}> -> tensor<1x1x1x1xf16, {order = #NCHW}>
    //CHECK:      return [[MAXIMUM]] : tensor<1x1x1x1xf16, {order = #NCHW}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SOKCONVConsiderDMAPrefetching
func.func @SOKCONVConsiderDMAPrefetching(%arg0: tensor<1x960x65x65xf16, {order = #NHWC}>) -> tensor<1x320x65x65xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<320x960x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<320x960x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<320x1x1x4xsi32> = dense<0> : tensor<320x1x1x4xsi32>
    %cst_1 = const.Declare tensor<960x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<960x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<960x1x1x4xsi32> = dense<0> : tensor<960x1x1x4xsi32>
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_2) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [960, 1, 3, 3], strides = [1, 1]} -> tensor<1x960x65x65xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst, %cst_0) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [320, 960, 1, 1], strides = [1, 1]} -> tensor<1x320x65x65xf16, {order = #NHWC}>
    return %1 : tensor<1x320x65x65xf16, {order = #NHWC}>

    // Get MC strategy considering the tiling strategy
    // Calculate the cost more accurate by considering DMA prefetching
    // CHECK:       VPU.NCE.DepthConvolution
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

    // CHECK:       VPU.NCE.Convolution
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ParentPermuteOpWithHeightEqualOneNoStrategy
func.func @ParentPermuteOpWithHeightEqualOneNoStrategy(%arg0: tensor<1x16x1x112xf16>, %arg1: tensor<1x16x32x112xf16>) -> tensor<1x16x33x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, ppe = #VPU.PPEStub<>} -> tensor<1x16x1x112xf16, {order = #NHWC}>
    %1 = VPU.NCE.AveragePool(%0) {kernel_size = [1, 1], ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1]} -> tensor<1x16x1x112xf16, {order = #NHWC}>

    %2 = VPU.NCE.Permute(%arg1) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, ppe = #VPU.PPEStub<>} -> tensor<1x16x32x112xf16, {order = #NHWC}>
    %3 = VPU.NCE.AveragePool(%2) {kernel_size = [1, 1], ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1]} -> tensor<1x16x32x112xf16, {order = #NHWC}>

    %4 = VPU.Concat(%1, %3) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0]]} : tensor<1x16x1x112xf16, {order = #NHWC}>, tensor<1x16x32x112xf16, {order = #NHWC}> -> tensor<1x16x33x112xf16, {order = #NHWC}>

    return %4 : tensor<1x16x33x112xf16, {order = #NHWC}>

    //CHECK: [[Permute_1:%.*]] = VPU.NCE.Permute(%arg0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, ppe = #VPU.PPEStub<>} -> tensor<1x16x1x112xf16, {order = #NHWC}>
    //CHECK: [[AVGPool_1:%.*]] = VPU.NCE.AveragePool([[Permute_1]]) {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x16x1x112xf16, {order = #NHWC}>

    //CHECK: [[Permute_2:%.*]] = VPU.NCE.Permute(%arg1) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEStub<>} -> tensor<1x16x32x112xf16, {order = #NHWC}>
    //CHECK: [[AVGPool_2:%.*]] = VPU.NCE.AveragePool([[Permute_2]]) {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x16x32x112xf16, {order = #NHWC}>
    //CHECK: [[Result:%.*]] = VPU.Concat([[AVGPool_1]], [[AVGPool_2]]) {
    //CHECK-SAME{LITERAL}:            static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0]]
    //CHECK:  } : tensor<1x16x1x112xf16, {order = #NHWC}>, tensor<1x16x32x112xf16, {order = #NHWC}> -> tensor<1x16x33x112xf16, {order = #NHWC}>

    //CHECK: return [[Result]] : tensor<1x16x33x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ChildPermuteOpWithHeightEqualOneNoStrategy
func.func @ChildPermuteOpWithHeightEqualOneNoStrategy(%arg0: tensor<1x512x1x16xf16>, %arg1: tensor<1x512x1x32xf16>) -> tensor<1x512x1x48xf16, {order = #NHWC}> {
    %0 = VPU.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 16]]} : tensor<1x512x1x16xf16>, tensor<1x512x1x32xf16> -> tensor<1x512x1x48xf16>
    %1 = VPU.NCE.Permute(%0) {dstElemType = f16, dstOrder = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, expandedChannels = 512 : i64, ppe = #VPU.PPEStub<>} -> tensor<1x512x1x48xf16, {order = #NHWC}>
    return %1 : tensor<1x512x1x48xf16, {order = #NHWC}>


    //CHECK: [[Concat:%.*]] = VPU.Concat(%arg0, %arg1) {
    //CHECK-SAME{LITERAL}:            static_offsets = [[0, 0, 0, 0], [0, 0, 0, 16]]
    //CHECK:  } : tensor<1x512x1x16xf16>, tensor<1x512x1x32xf16> -> tensor<1x512x1x48xf16>

    //CHECK: [[Permute:%.*]] = VPU.NCE.Permute([[Concat]]) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 512 : i64, ppe = #VPU.PPEStub<>} -> tensor<1x512x1x48xf16, {order = #NHWC}>
    //CHECK: return [[Permute]] : tensor<1x512x1x48xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedStrategySOHCostMax
func.func @ConvAssignedStrategySOHCostMax(%arg0: tensor<1x384x8x8xf16, {order = #NHWC}>) -> tensor<1x112x8x8xf16, {order = #NHWC}> {
    %weights_sm = const.Declare tensor<112x1x1x18816xi1> = dense<1.000000e+00> : tensor<112x384x7x7xf16, {order = #NHWC}>, [#const.GetSparsityMap]
    %weights = const.Declare tensor<112x384x7x7xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<112x384x7x7xf16, {order = #NHWC}>, [#const.Sparsify<false>]
    %weights_table = const.Declare tensor<112x1x1x4xsi32> = dense<1> : tensor<112x1x1x4xsi32>
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {
            sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<64> : tensor<112xi64>, alignment = 16 : i64>, is_weights}
            -> !VPU.SparseTensor<data=tensor<112x384x7x7xf16, {order = #NHWC}>,
            sparsity_map=tensor<112x1x1x18816xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<64>
            : tensor<112xi64>, alignment = 16 : i64>>
    %conv = VPU.NCE.Convolution(%arg0, %weights_sparse, %weights_table) {
            pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [112, 384, 7, 7], strides = [1, 1]}
            -> tensor<1x112x8x8xf16, {order = #NHWC}>

    return %conv : tensor<1x112x8x8xf16, {order = #NHWC}>

    //CHECK:    [[WEIGHTS_SM:%.+]] = const.Declare tensor<112x1x1x18816xi1> = dense<1.000000e+00> : tensor<112x384x7x7xf16, {order = #NHWC}>, [#const.GetSparsityMap]
    //CHECK:    [[WEIGHTS:%.+]] = const.Declare tensor<112x384x7x7xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<112x384x7x7xf16, {order = #NHWC}>, [#const.Sparsify<false>]
    //CHECK:    [[WEIGHTS_TBL:%.+]] = const.Declare tensor<112x1x1x4xsi32> = dense<1> : tensor<112x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM]]) {
    //CHECK-SAME:    is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<64> : tensor<112xi64>, alignment = 16 : i64>}
    //CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<112x384x7x7xf16, {order = #NHWC}>, sparsity_map=tensor<112x1x1x18816xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<64> : tensor<112xi64>, alignment = 16 : i64>>
    //CHECK:    [[CONV:%.+]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS_SPARSE]], [[WEIGHTS_TBL]]) {
    //CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>, ppe = #VPU.PPEStub<>,
    //CHECK-SAME:    rawFilterShape = [112, 384, 7, 7], strides = [1, 1]}
    //CHECK-SAME:    -> tensor<1x112x8x8xf16, {order = #NHWC}>
    //CHECK:    return [[CONV]] : tensor<1x112x8x8xf16, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8<0:254>:f16, 0.0078740157480314959:127>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @SOHOPermuteOutputOverlapped
func.func @SOHOPermuteOutputOverlapped(%arg0: tensor<1x4x64x64x!qElemType, {order = #NCHW}>) -> tensor<1x64x8x8x!qElemType, {order = #NHWC}> {
    %cst = const.Declare tensor<32x16x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<64x32x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %cst_3 = const.Declare tensor<16x1x1x48x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x1x1x48xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %cst_4 = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %0 = VPU.NCE.Permute(%arg0) {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64, ppe = #VPU.PPEStub<>} -> tensor<1x4x64x64x!qElemType, {order = #NHWC}>
    %1 = VPU.NCE.CompressConvolution(%0, %cst_3, %cst_4) {cm_sp_pattern = 1 : i64, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, rawFilterShape = [16, 1, 3, 3], strides = [2, 2]} -> tensor<1x16x32x32x!qElemType, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%1, %cst, %cst_0) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, rawFilterShape = [32, 16, 3, 3], strides = [2, 2]} -> tensor<1x32x16x16x!qElemType, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %cst_1, %cst_2) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, rawFilterShape = [64, 32, 3, 3], strides = [2, 2]} -> tensor<1x64x8x8x!qElemType, {order = #NHWC}>
    return %3 : tensor<1x64x8x8x!qElemType, {order = #NHWC}>

    // There is no output spilling after SOHO Permute op if the next op is SOHO Conv/CompressConv
    // CHECK:       VPU.NCE.Permute
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>
    // CHECK:       VPU.NCE.CompressConvolution
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>
    // CHECK:       VPU.NCE.Convolution
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: func.func @PermuteAssignedSOHOverlapped
func.func @PermuteAssignedSOHOverlapped(%arg0: tensor<1x3x256x224xf16>) -> tensor<1x4x256x224x!qElemType, {order = #NHWC}> {
    %PERMUTE = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x4x256x224x!qElemType, {order = #NHWC}>

    // CHECK:   multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>

    return %PERMUTE : tensor<1x4x256x224x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @PermuteFP16AssignedSOHOverlapped
func.func @PermuteFP16AssignedSOHOverlapped(%arg0: tensor<1x3x256x224xf16>) -> tensor<1x4x256x224xf16, {order = #NHWC}> {
    %PERMUTE = VPU.NCE.Permute(%arg0) {
        dstElemType = f16,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x4x256x224xf16, {order = #NHWC}>

    // CHECK:   multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>

    return %PERMUTE : tensor<1x4x256x224xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ClampAssignedSplitOverKernel
func.func @ClampAssignedSplitOverKernel(%arg0: tensor<1x16x1x60xf16, {order = #NHWC}>) -> tensor<1x16x1x60xf16, {order = #NHWC}> {

    %0 = VPU.Clamp(%arg0) {max = 1.000000e+00 : f64, min = -1.000000e+00 : f64} :
        tensor<1x16x1x60xf16, {order = #NHWC}> -> tensor<1x16x1x60xf16, {order = #NHWC}>

    return %0 : tensor<1x16x1x60xf16, {order = #NHWC}>

    //CHECK:   [[Result:%.*]] = VPU.Clamp(%arg0) {
    //CHECK-SAME:       max = 1.000000e+00 : f64, min = -1.000000e+00 : f64,
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} :
    //CHECK-SAME:       tensor<1x16x1x60xf16, {order = #NHWC}> -> tensor<1x16x1x60xf16, {order = #NHWC}>
    //CHECK:   return [[Result]] : tensor<1x16x1x60xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ClampAssignedSplitOverHeight
func.func @ClampAssignedSplitOverHeight(%arg0: tensor<1x16x128x60xf16, {order = #NHWC}>) -> tensor<1x16x128x60xf16, {order = #NHWC}> {

    %0 = VPU.Clamp(%arg0) {max = 1.000000e+00 : f64, min = -1.000000e+00 : f64} :
        tensor<1x16x128x60xf16, {order = #NHWC}> -> tensor<1x16x128x60xf16, {order = #NHWC}>

    return %0 : tensor<1x16x128x60xf16, {order = #NHWC}>

    //CHECK:   [[Result:%.*]] = VPU.Clamp(%arg0) {
    //CHECK-SAME:       max = 1.000000e+00 : f64, min = -1.000000e+00 : f64,
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
    //CHECK-SAME:       tensor<1x16x128x60xf16, {order = #NHWC}> -> tensor<1x16x128x60xf16, {order = #NHWC}>
    //CHECK:   return [[Result]] : tensor<1x16x128x60xf16, {order = #NHWC}>

}

// -----

// CHECK-LABEL: @AbsAssignedSplitOverKernel
func.func @AbsAssignedSplitOverKernel(%arg0: tensor<1x16x1x60xf16>) -> tensor<1x16x1x60xf16> {

    %0 = VPU.Abs(%arg0) : tensor<1x16x1x60xf16> -> tensor<1x16x1x60xf16>

    return %0 : tensor<1x16x1x60xf16>

    //CHECK:   [[ABS:%.+]] = VPU.Abs(%arg0) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    //CHECK-SAME:       tensor<1x16x1x60xf16> -> tensor<1x16x1x60xf16>
    //CHECK:   return [[ABS]] : tensor<1x16x1x60xf16>
}

// -----

// CHECK-LABEL: @AbsAssignedSplitOverHeight
func.func @AbsAssignedSplitOverHeight(%arg0: tensor<1x1x128x60xf16>) -> tensor<1x1x128x60xf16> {

    %0 = VPU.Abs(%arg0) : tensor<1x1x128x60xf16> -> tensor<1x1x128x60xf16>

    return %0 : tensor<1x1x128x60xf16>

    //CHECK:   [[ABS:%.+]] = VPU.Abs(%arg0) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:       tensor<1x1x128x60xf16> -> tensor<1x1x128x60xf16>
    //CHECK:   return [[ABS]] : tensor<1x1x128x60xf16>
}

// -----

// CHECK-LABEL: @DivideAssignedSplitOverKernel
func.func @DivideAssignedSplitOverKernel(%arg0: tensor<1x106x1x256xf16>, %arg1: tensor<1x106x1x1xf16>) -> tensor<1x106x1x256xf16> {

    %0 = VPU.Divide(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xf16>

    return %0 : tensor<1x106x1x256xf16>

    //CHECK:   [[DIVIDE:%.+]] = VPU.Divide(%arg0, %arg1) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    //CHECK-SAME:       tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xf16>
    //CHECK:   return [[DIVIDE]] : tensor<1x106x1x256xf16>
}

// -----

// CHECK-LABEL: @DivideAssignedSplitOverHeight
func.func @DivideAssignedSplitOverHeight(%arg0: tensor<1x16x256x256xf16>, %arg1: tensor<1x16x1x1xf16>) -> tensor<1x16x256x256xf16> {

    %0 = VPU.Divide(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x256xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x256x256xf16>

    return %0 : tensor<1x16x256x256xf16>

    //CHECK:   [[DIVIDE:%.+]] = VPU.Divide(%arg0, %arg1) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:       tensor<1x16x256x256xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x256x256xf16>
    //CHECK:   return [[DIVIDE]] : tensor<1x16x256x256xf16>
}

// -----

// CHECK-LABEL: @PowerAssignedSplitOverKernel
func.func @PowerAssignedSplitOverKernel(%arg0: tensor<1x106x1x256xf16>, %arg1: tensor<1x106x1x1xf16>) -> tensor<1x106x1x256xf16> {

    %0 = VPU.Power(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xf16>

    return %0 : tensor<1x106x1x256xf16>

    //CHECK:   [[POWER:%.+]] = VPU.Power(%arg0, %arg1) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    //CHECK-SAME:       tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xf16>
    //CHECK:   return [[POWER]] : tensor<1x106x1x256xf16>
}

// -----

// CHECK-LABEL: @PowerAssignedSplitOverHeight
func.func @PowerAssignedSplitOverHeight(%arg0: tensor<1x16x256x256xf16>, %arg1: tensor<1x16x1x1xf16>) -> tensor<1x16x256x256xf16> {

    %0 = VPU.Power(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x256xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x256x256xf16>

    return %0 : tensor<1x16x256x256xf16>

    //CHECK:   [[POWER:%.+]] = VPU.Power(%arg0, %arg1) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:       tensor<1x16x256x256xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x256x256xf16>
    //CHECK:   return [[POWER]] : tensor<1x16x256x256xf16>
}

// -----

// CHECK-LABEL: @PowerAssignedClustering
func.func @PowerAssignedClustering(%arg0: tensor<2x106x1x256xf16>, %arg1: tensor<2x106x1x1xf16>) -> tensor<2x106x1x256xf16> {

    %0 = VPU.Power(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xf16>

    return %0 : tensor<2x106x1x256xf16>

    //CHECK:   [[POWER:%.+]] = VPU.Power(%arg0, %arg1) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
    //CHECK-SAME:       tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xf16>
    //CHECK:   return [[POWER]] : tensor<2x106x1x256xf16>
}

// -----

// CHECK-LABEL:   @GreaterAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<1x106x1x1xf16>
func.func @GreaterAssignedSplitOverKernel(%arg0: tensor<1x106x1x256xf16>, %arg1: tensor<1x106x1x1xf16>) -> tensor<1x106x1x256xi8> {

    %0 = VPU.Greater(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xi8>

    return %0 : tensor<1x106x1x256xi8>

    //CHECK:   [[GREATER:%.+]] = VPU.Greater([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    //CHECK-SAME:       tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xi8>
    //CHECK:   return [[GREATER]] : tensor<1x106x1x256xi8>
}

// -----

// CHECK-LABEL:   @GreaterAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x256x256xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x1xf16>
func.func @GreaterAssignedSplitOverHeight(%arg0: tensor<1x1x256x256xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x1x256x256xi8> {

    %0 = VPU.Greater(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xi8>

    return %0 : tensor<1x1x256x256xi8>

    //CHECK:   [[GREATER:%.+]] = VPU.Greater([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:       tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xi8>
    //CHECK:   return [[GREATER]] : tensor<1x1x256x256xi8>
}

// -----

// CHECK-LABEL:   @GreaterAssignedClustering
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<2x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<2x106x1x1xf16>
func.func @GreaterAssignedClustering(%arg0: tensor<2x106x1x256xf16>, %arg1: tensor<2x106x1x1xf16>) -> tensor<2x106x1x256xi8> {

    %0 = VPU.Greater(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xi8>

    return %0 : tensor<2x106x1x256xi8>

    //CHECK:   [[GREATER:%.+]] = VPU.Greater(%arg0, %arg1) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
    //CHECK-SAME:       tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xi8>
    //CHECK:   return [[GREATER]] : tensor<2x106x1x256xi8>
}

// -----

// CHECK-LABEL:   @LessAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<1x106x1x1xf16>
func.func @LessAssignedSplitOverKernel(%arg0: tensor<1x106x1x256xf16>, %arg1: tensor<1x106x1x1xf16>) -> tensor<1x106x1x256xi8> {

    %0 = VPU.Less(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xi8>

    return %0 : tensor<1x106x1x256xi8>

    //CHECK:   [[LESS:%.+]] = VPU.Less([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    //CHECK-SAME:       tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xi8>
    //CHECK:   return [[LESS]] : tensor<1x106x1x256xi8>
}

// -----

// CHECK-LABEL:   @LessAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x256x256xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x1xf16>
func.func @LessAssignedSplitOverHeight(%arg0: tensor<1x1x256x256xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x1x256x256xi8> {

    %0 = VPU.Less(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xi8>

    return %0 : tensor<1x1x256x256xi8>

    //CHECK:   [[LESS:%.+]] = VPU.Less([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:       tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xi8>
    //CHECK:   return [[LESS]] : tensor<1x1x256x256xi8>
}

// -----

// CHECK-LABEL:   @LessAssignedClustering
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<2x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<2x106x1x1xf16>
func.func @LessAssignedClustering(%arg0: tensor<2x106x1x256xf16>, %arg1: tensor<2x106x1x1xf16>) -> tensor<2x106x1x256xi8> {

    %0 = VPU.Less(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xi8>

    return %0 : tensor<2x106x1x256xi8>

    //CHECK:   [[LESS:%.+]] = VPU.Less(%arg0, %arg1) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
    //CHECK-SAME:       tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xi8>
    //CHECK:   return [[LESS]] : tensor<2x106x1x256xi8>
}

// -----

// CHECK-LABEL:   @EqualAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<1x106x1x1xf16>
func.func @EqualAssignedSplitOverKernel(%arg0: tensor<1x106x1x256xf16>, %arg1: tensor<1x106x1x1xf16>) -> tensor<1x106x1x256xi8> {

    %0 = VPU.Equal(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xi8>

    return %0 : tensor<1x106x1x256xi8>

    //CHECK:   [[EQUAL:%.+]] = VPU.Equal([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    //CHECK-SAME:       tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xi8>
    //CHECK:   return [[EQUAL]] : tensor<1x106x1x256xi8>
}

// -----

// CHECK-LABEL:   @EqualAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x256x256xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x1xf16>
func.func @EqualAssignedSplitOverHeight(%arg0: tensor<1x1x256x256xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x1x256x256xi8> {

    %0 = VPU.Equal(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xi8>

    return %0 : tensor<1x1x256x256xi8>

    //CHECK:   [[EQUAL:%.+]] = VPU.Equal([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:       tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xi8>
    //CHECK:   return [[EQUAL]] : tensor<1x1x256x256xi8>
}

// -----

// CHECK-LABEL:   @EqualAssignedClustering
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<2x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<2x106x1x1xf16>
func.func @EqualAssignedClustering(%arg0: tensor<2x106x1x256xf16>, %arg1: tensor<2x106x1x1xf16>) -> tensor<2x106x1x256xi8> {

    %0 = VPU.Equal(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xi8>

    return %0 : tensor<2x106x1x256xi8>

    //CHECK:   [[EQUAL:%.+]] = VPU.Equal([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
    //CHECK-SAME:       tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xi8>
    //CHECK:   return [[EQUAL]] : tensor<2x106x1x256xi8>
}

// -----

// CHECK-LABEL:   @SubtractAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<1x106x1x1xf16>
func.func @SubtractAssignedSplitOverKernel(%arg0: tensor<1x106x1x256xf16>, %arg1: tensor<1x106x1x1xf16>) -> tensor<1x106x1x256xf16> {

    %0 = VPU.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xf16>

    return %0 : tensor<1x106x1x256xf16>

    //CHECK:   [[SUBTRACT:%.+]] = VPU.Subtract([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    //CHECK-SAME:       tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xf16>
    //CHECK:   return [[SUBTRACT]] : tensor<1x106x1x256xf16>
}

// -----

// CHECK-LABEL:   @SubtractAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x256x256xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x1xf16>
func.func @SubtractAssignedSplitOverHeight(%arg0: tensor<1x1x256x256xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x1x256x256xf16> {

    %0 = VPU.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xf16>

    return %0 : tensor<1x1x256x256xf16>

    //CHECK:   [[SUBTRACT:%.+]] = VPU.Subtract([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:       tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xf16>
    //CHECK:   return [[SUBTRACT]] : tensor<1x1x256x256xf16>
}

// -----

// CHECK-LABEL:   @SubtractAssignedClustering
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<2x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<2x106x1x1xf16>
func.func @SubtractAssignedClustering(%arg0: tensor<2x106x1x256xf16>, %arg1: tensor<2x106x1x1xf16>) -> tensor<2x106x1x256xf16> {

    %0 = VPU.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xf16>

    return %0 : tensor<2x106x1x256xf16>

    //CHECK:   [[SUBTRACT:%.+]] = VPU.Subtract(%arg0, %arg1) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
    //CHECK-SAME:       tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xf16>
    //CHECK:   return [[SUBTRACT]] : tensor<2x106x1x256xf16>
}

// -----

// CHECK-LABEL:   @AddAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<1x106x1x1xf16>
func.func @AddAssignedSplitOverKernel(%arg0: tensor<1x106x1x256xf16>, %arg1: tensor<1x106x1x1xf16>) -> tensor<1x106x1x256xf16> {

    %0 = VPU.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xf16>

    return %0 : tensor<1x106x1x256xf16>

    //CHECK:   [[ADD:%.+]] = VPU.Add([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    //CHECK-SAME:       tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xf16>
    //CHECK:   return [[ADD]] : tensor<1x106x1x256xf16>
}

// -----

// CHECK-LABEL:   @AddAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x256x256xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x1xf16>
func.func @AddAssignedSplitOverHeight(%arg0: tensor<1x1x256x256xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x1x256x256xf16> {

    %0 = VPU.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xf16>

    return %0 : tensor<1x1x256x256xf16>

    //CHECK:   [[ADD:%.+]] = VPU.Add([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:       tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xf16>
    //CHECK:   return [[ADD]] : tensor<1x1x256x256xf16>
}

// -----

// CHECK-LABEL:   @AddAssignedClustering
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<2x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<2x106x1x1xf16>
func.func @AddAssignedClustering(%arg0: tensor<2x106x1x256xf16>, %arg1: tensor<2x106x1x1xf16>) -> tensor<2x106x1x256xf16> {

    %0 = VPU.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xf16>

    return %0 : tensor<2x106x1x256xf16>

    //CHECK:   [[ADD:%.+]] = VPU.Add(%arg0, %arg1) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
    //CHECK-SAME:       tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xf16>
    //CHECK:   return [[ADD]] : tensor<2x106x1x256xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SOHEltwiseAlignment
func.func @SOHEltwiseAlignment(%arg0: tensor<1x64x100x170xf16, {order = #NHWC}>, %arg1: tensor<1x64x100x170xf16, {order = #NHWC}>) -> tensor<1x64x100x170xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<16x1x1x4xsi32> =  dense<1> : tensor<16x1x1x4xsi32>
    %cst_1 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x1x3x3xf16>, [#const.Reshape<[16, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<16x1x1x4xsi32> =  dense<1> : tensor<16x1x1x4xsi32>
    %cst_3 = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_4 = const.Declare tensor<64x1x1x4xsi32> =  dense<1> : tensor<64x1x1x4xsi32>
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {is_inplace = true, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x64x100x170xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst, %cst_0) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [16, 64, 1, 1], strides = [1, 1]} -> tensor<1x16x100x170xf16, {order = #NHWC}>
    %2 = VPU.NCE.DepthConvolution(%1, %cst_1, %cst_2) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 1, 3, 3], strides = [1, 1]} -> tensor<1x16x100x170xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %cst_3, %cst_4) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [64, 16, 1, 1], strides = [1, 1]} -> tensor<1x64x100x170xf16, {order = #NHWC}>
    %4 = VPU.NCE.Eltwise(%0, %3) {is_inplace = true, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x64x100x170xf16, {order = #NHWC}>
    return %4: tensor<1x64x100x170xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<16x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x64x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:       [[CST_0:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK-DAG:       [[CST_1:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x1x3x3xf16>, [#const.Reshape<[16, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:       [[CST_2:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK-DAG:       [[CST_3:%.+]] = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:       [[CST_4:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    // CHECK:           [[ADD_0:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1) {
    // CHECK-SAME:          is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:          -> tensor<1x64x100x170xf16, {order = #NHWC}>
    // CHECK:           [[CONV_0:%.+]] = VPU.NCE.Convolution([[ADD_0]], [[CST]], [[CST_0]]) {
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:          rawFilterShape = [16, 64, 1, 1], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x16x100x170xf16, {order = #NHWC}>
    // CHECK:           [[DWCONV_0:%.+]] = VPU.NCE.DepthConvolution([[CONV_0]], [[CST_1]], [[CST_2]]) {
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:          rawFilterShape = [16, 1, 3, 3], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x16x100x170xf16, {order = #NHWC}>
    // CHECK:           [[CONV_1:%.+]] = VPU.NCE.Convolution([[DWCONV_0]], [[CST_3]], [[CST_4]]) {
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:          rawFilterShape = [64, 16, 1, 1], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x64x100x170xf16, {order = #NHWC}>
    // CHECK:           [[ADD_1:%.+]] = VPU.NCE.Eltwise([[ADD_0]], [[CONV_1]]) {
    // CHECK-SAME:          is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:          -> tensor<1x64x100x170xf16, {order = #NHWC}>
    //CHECK:        return [[ADD_1]] : tensor<1x64x100x170xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SOKConvolutionEltwise
func.func @SOKConvolutionEltwise(%arg0: tensor<1x1024x4x4x!quant.uniform<u8:f16, 0.015109125305624568:128>, {order = #NHWC}>, %arg1: tensor<1x1024x4x4x!quant.uniform<u8:f16, 0.015109125305624568:128>, {order = #NHWC}>) -> tensor<1x2048x2x2xf16> {
    %cst = const.Declare tensor<2048x1024x1x1x!quant.uniform<u8<0:254>:f16, 0.011682062637148879:127>, {order = #NHWC}> = dense<1> : tensor<2048x1024x1x1xsi8>, [#const.CastElemType<!quant.uniform<i8<-127:127>:f16, 0.011682062637148879>>, #const.CastElemType<si8>, #const.CastElemType<i32>, #const.Add<1.270000e+02 : f64>, #const.CastElemType<ui8>, #const.CastElemType<!quant.uniform<u8<0:254>:f16, 0.011682062637148879:127>>, #const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<2048x1x1x4xsi32> = dense<1> : tensor<2048x1x1x4xsi32>
    %cst_1 = const.Declare tensor<1024x1024x1x1x!quant.uniform<u8<0:254>:f16, 0.0036145465111169289:127>, {order = #NHWC}> = dense<1> : tensor<1024x1024x1x1xsi8>, [#const.CastElemType<!quant.uniform<i8<-127:127>:f16, 0.0036145465111169289>>, #const.CastElemType<si8>, #const.CastElemType<i32>, #const.Add<1.270000e+02 : f64>, #const.CastElemType<ui8>, #const.CastElemType<!quant.uniform<u8<0:254>:f16, 0.0036145465111169289:127>>, #const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<1024x1x1x4xsi32> = dense<1> : tensor<1024x1x1x4xsi32>
    %cst_3 = const.Declare tensor<2048x1024x3x3x!quant.uniform<u8<0:254>:f16, 0.0093559721323448839:127>, {order = #NHWC}> = dense<1> : tensor<2048x1024x3x3xsi8>, [#const.CastElemType<!quant.uniform<i8<-127:127>:f16, 0.0093559721323448839>>, #const.CastElemType<si8>, #const.CastElemType<i32>, #const.Add<1.270000e+02 : f64>, #const.CastElemType<ui8>, #const.CastElemType<!quant.uniform<u8<0:254>:f16, 0.0093559721323448839:127>>, #const.Reorder<#NHWC>]
    %cst_4 = const.Declare tensor<2048x1x1x4xsi32> = dense<1> : tensor<2048x1x1x4xsi32>

    %0 = VPU.NCE.Eltwise(%arg0, %arg1)  {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x1024x4x4x!quant.uniform<u8:f16, 0.0091846662409165325>, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst, %cst_0) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [2048, 1024, 1, 1], strides = [2, 2]} -> tensor<1x2048x2x2x!quant.uniform<u8:f16, 0.017686840132171033:128>, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%0, %cst_1, %cst_2) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [1024, 1024, 1, 1], strides = [1, 1]} -> tensor<1x1024x4x4x!quant.uniform<u8:f16, 0.0040072758992513021>, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %cst_3, %cst_4) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, rawFilterShape = [2048, 1024, 3, 3], strides = [2, 2]} -> tensor<1x2048x2x2x!quant.uniform<u8:f16, 0.0040072758992513021>, {order = #NHWC}>
    %4 = VPU.NCE.Eltwise(%1, %3) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x2048x2x2xf16>
    return %4: tensor<1x2048x2x2xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<2048x1024x1x1x!qElemType1, {order = #NHWC}> = dense<1> : tensor<2048x1024x1x1xsi8>, [#const.CastElemType<!qElemType2>, #const.CastElemType<si8>, #const.CastElemType<i32>, #const.Add<1.270000e+02 : f64>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    // CHECK-DAG:       [[CST_0:%.+]] = const.Declare tensor<2048x1x1x4xsi32> = dense<1> : tensor<2048x1x1x4xsi32>
    // CHECK-DAG:       [[CST_1:%.+]] = const.Declare tensor<1024x1024x1x1x!qElemType3, {order = #NHWC}> = dense<1> : tensor<1024x1024x1x1xsi8>, [#const.CastElemType<!qElemType4>, #const.CastElemType<si8>, #const.CastElemType<i32>, #const.Add<1.270000e+02 : f64>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType3>, #const.Reorder<#NHWC>]
    // CHECK-DAG:       [[CST_2:%.+]] = const.Declare tensor<1024x1x1x4xsi32> = dense<1> : tensor<1024x1x1x4xsi32>
    // CHECK-DAG:       [[CST_3:%.+]] = const.Declare tensor<2048x1024x3x3x!qElemType5, {order = #NHWC}> = dense<1> : tensor<2048x1024x3x3xsi8>, [#const.CastElemType<!qElemType6>, #const.CastElemType<si8>, #const.CastElemType<i32>, #const.Add<1.270000e+02 : f64>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType5>, #const.Reorder<#NHWC>]
    // CHECK-DAG:       [[CST_4:%.+]] = const.Declare tensor<2048x1x1x4xsi32> = dense<1> : tensor<2048x1x1x4xsi32>
    // CHECK:           [[ADD_0:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1) {
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:          -> tensor<1x1024x4x4x!qElemType7, {order = #NHWC}>
    // CHECK:           [[CONV_0:%.+]] = VPU.NCE.Convolution([[ADD_0]], [[CST]], [[CST_0]]) {
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:          rawFilterShape = [2048, 1024, 1, 1], strides = [2, 2]}
    // CHECK-SAME:          -> tensor<1x2048x2x2x!qElemType8, {order = #NHWC}>
    // CHECK:           [[CONV_1:%.+]] = VPU.NCE.Convolution([[ADD_0]], [[CST_1]], [[CST_2]]) {
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:          rawFilterShape = [1024, 1024, 1, 1], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x1024x4x4x!qElemType9, {order = #NHWC}>
    // CHECK:           [[CONV_2:%.+]] = VPU.NCE.Convolution([[CONV_1]], [[CST_3]], [[CST_4]]) {
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:          rawFilterShape = [2048, 1024, 3, 3], strides = [2, 2]}
    // CHECK-SAME:          -> tensor<1x2048x2x2x!qElemType9, {order = #NHWC}>
    // CHECK:           [[ADD_1:%.+]] = VPU.NCE.Eltwise([[CONV_0]], [[CONV_2]]) {
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:          -> tensor<1x2048x2x2xf16>
    // CHECK:        return [[ADD_1]] : tensor<1x2048x2x2xf16>
}

// -----

// CHECK-LABEL: func.func @PReluAssignedSplitOverHeight
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x1x128x128xf16>)
func.func @PReluAssignedSplitOverHeight(%arg0: tensor<1x1x128x128xf16>) -> tensor<1x1x128x128xf16> {

    %cst = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    %0 = VPU.PRelu(%arg0, %cst) : tensor<1x1x128x128xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x128x128xf16>

    return %0 : tensor<1x1x128x128xf16>

    //CHECK-DAG:    [[SLOPE:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    //CHECK:        [[PRELU:%.+]] = VPU.PRelu([[INPUT_DATA]], [[SLOPE]]) {
    //CHECK-SAME:               multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:             : tensor<1x1x128x128xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x128x128xf16>
    //CHECK:        return [[PRELU]] : tensor<1x1x128x128xf16>
}

// -----

// CHECK-LABEL: func.func @PReluAssignedSplitOverKernel
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x128x1x1xf16>)
func.func @PReluAssignedSplitOverKernel(%arg0: tensor<1x128x1x1xf16>) -> tensor<1x128x1x1xf16> {

    %cst = const.Declare tensor<1x128x1x1xf16> = dense<-1.000000e+01> : tensor<1x128x1x1xf16>
    %0 = VPU.PRelu(%arg0, %cst) : tensor<1x128x1x1xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x1x1xf16>

    return %0 : tensor<1x128x1x1xf16>

    //CHECK-DAG:    [[SLOPE:%.+]] = const.Declare tensor<1x128x1x1xf16> = dense<-1.000000e+01> : tensor<1x128x1x1xf16>
    //CHECK:        [[PRELU:%.+]] = VPU.PRelu([[INPUT_DATA]], [[SLOPE]]) {
    //CHECK-SAME:               multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    //CHECK-SAME:             : tensor<1x128x1x1xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x1x1xf16>
    //CHECK:        return [[PRELU]] : tensor<1x128x1x1xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipSubGraphOptimizationForINFCost
func.func @SkipSubGraphOptimizationForINFCost(%input_data: tensor<1x128x72x72xf16, {order = #NHWC}>, %input_data2: tensor<1x64x144x144xf16, {order = #NHWC}>) -> tensor<1x64x144x144xf16, {order = #NHWC}> {
    // conv
    %convWeights = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x128x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %convWeightsTable = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input_data, %convWeights, %convWeightsTable) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [64, 128, 1, 1], strides = [1, 1]
    } -> tensor<1x64x72x72xf16, {order = #NHWC}>

    // transposedConv
    %sparsityMap = const.Declare tensor<1x64x147x147xi1, {order = #NHWC}> = dense<1> : tensor<1x64x147x147xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    %storageElement = VPU.StorageElementTable {
        dataElemType = f16,
        dataShape = [1, 64, 72, 72],
        seAttr = #VPU.SEUpsampling<
            factors = [1, 1],
            padding = [2, 2, 2, 2]>,
        seDepth = 1 : i64,
        seSize = 64 : i64
    } -> tensor<1x1x147x147xi32, {order = #NHWC}>
    %input = VPU.GroupSparseTensor(%conv, %sparsityMap, %storageElement) {
        seAttr = #VPU.SEUpsampling<
            factors = [1, 1],
            padding = [2, 2, 2, 2]>
    } -> !VPU.SparseTensor<
            data=tensor<1x64x72x72xf16, {order = #NHWC}>,
            sparsity_map=tensor<1x64x147x147xi1, {order = #NHWC}>,
            storage_element_table=tensor<1x1x147x147xi32, {order = #NHWC}>,
            #VPU.SEUpsampling<
                factors = [1, 1],
                padding = [2, 2, 2, 2]>>

    %weightsCst = const.Declare tensor<64x64x4x4xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x4x4xf16, {order = #NHWC}>, [#const.Sparsify<false>]
    %weightsSparsityMap = const.Declare tensor<64x1x1x1024xi1> = dense<1.0> : tensor<64x64x4x4xf16, {order = #NHWC}>, [#const.GetSparsityMap]
    %weights = VPU.GroupSparseTensor(%weightsCst, %weightsSparsityMap) {
        sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<16> : tensor<64xi64>, alignment = 16 : i64>, is_weights}
        -> !VPU.SparseTensor<data=tensor<64x64x4x4xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x1024xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<16> : tensor<64xi64>, alignment = 16 : i64>>

    %weightsTable = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %transposedConv = VPU.NCE.Convolution(%input, %weights, %weightsTable) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [64, 64, 4, 4], strides = [1, 1]
    } -> tensor<1x64x144x144xf16, {order = #NHWC}>

    // add
    %add = VPU.NCE.Eltwise(%transposedConv, %input_data2) {
        is_inplace = true,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x64x144x144xf16, {order = #NHWC}>


    return %add : tensor<1x64x144x144xf16, {order = #NHWC}>

    // CHECK:        [[CONV_WEIGHTS:%.+]] = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:        [[CONV_WEIGHTS_TBL:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    // CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution(%arg0, [[CONV_WEIGHTS]], [[CONV_WEIGHTS_TBL]])
    // CHECK-SAME:        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:        ppe = #VPU.PPEStub<>,
    // CHECK-SAME:        rawFilterShape = [64, 128, 1, 1], strides = [1, 1]} -> tensor<1x64x72x72xf16, {order = #NHWC}>

    // CHECK:        [[SPARSITY_MAP:%.+]] = const.Declare tensor<1x64x147x147xi1, {order = #NHWC}> = dense<1> : tensor<1x64x147x147xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:        [[SE_TBL:%.+]] = VPU.StorageElementTable
    // CHECK-SAME:        {dataElemType = f16,
    // CHECK-SAME:        dataShape = [1, 64, 72, 72],
    // CHECK-SAME:        seAttr = #VPU.SEUpsampling<
    // CHECK-SAME:            factors = [1, 1],
    // CHECK-SAME:            padding = [2, 2, 2, 2]>,
    // CHECK-SAME:            seDepth = 1 : i64, seSize = 64 : i64} -> tensor<1x1x147x147xi32, {order = #NHWC}>
    // CHECK:        [[SPARSE_INPUT:%.+]] = VPU.GroupSparseTensor([[CONV]], [[SPARSITY_MAP]], [[SE_TBL]])
    // CHECK-SAME:        {seAttr = #VPU.SEUpsampling<
    // CHECK-SAME:            factors = [1, 1],
    // CHECK-SAME:            padding = [2, 2, 2, 2]>}
    // CHECK-SAME:        -> !VPU.SparseTensor<
    // CHECK-SAME:            data=tensor<1x64x72x72xf16, {order = #NHWC}>,
    // CHECK-SAME:            sparsity_map=tensor<1x64x147x147xi1, {order = #NHWC}>,
    // CHECK-SAME:            storage_element_table=tensor<1x1x147x147xi32, {order = #NHWC}>,
    // CHECK-SAME:        #VPU.SEUpsampling<
    // CHECK-SAME:            factors = [1, 1],
    // CHECK-SAME:            padding = [2, 2, 2, 2]>>
    // CHECK:        [[DECONV_WEIGHTS_CST:%.+]] = const.Declare tensor<64x64x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x4x4xf16, {order = #NHWC}>, [#const.Sparsify<false>]
    // CHECK:        [[DECONV_WEIGHTS_SM:%.+]] = const.Declare tensor<64x1x1x1024xi1> = dense<1.000000e+00> : tensor<64x64x4x4xf16, {order = #NHWC}>, [#const.GetSparsityMap]
    // CHECK:        [[DECONV_WEIGHTS:%.+]] = VPU.GroupSparseTensor([[DECONV_WEIGHTS_CST]], [[DECONV_WEIGHTS_SM]]) {
    // CHECK-SAME:        is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<16> : tensor<64xi64>, alignment = 16 : i64>}
    // CHECK-SAME:        -> !VPU.SparseTensor<data=tensor<64x64x4x4xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x1024xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<16> : tensor<64xi64>, alignment = 16 : i64>>
    // CHECK:        [[DECONV_WEIGHTS_TBL:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    // CHECK:        [[DECONV:%.+]] = VPU.NCE.Convolution([[SPARSE_INPUT]], [[DECONV_WEIGHTS]], [[DECONV_WEIGHTS_TBL]])
    // CHECK-SAME:        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:        ppe = #VPU.PPEStub<>,
    // CHECK-SAME:        rawFilterShape = [64, 64, 4, 4], strides = [1, 1]} -> tensor<1x64x144x144xf16, {order = #NHWC}>

    // CHECK:        [[ADD:%.+]] = VPU.NCE.Eltwise([[DECONV]], %arg1)
    // CHECK-SAME:        {is_inplace = true,
    // CHECK-SAME:        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:        op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:        ppe = #VPU.PPEStub<>
    // CHECK-SAME:        -> tensor<1x64x144x144xf16, {order = #NHWC}>

    // CHECK:        return [[ADD]] : tensor<1x64x144x144xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @FloorAssignedSplitOverKernel
func.func @FloorAssignedSplitOverKernel(%arg0: tensor<1x16x1x513xf16, {order = #NCHW}>) -> tensor<1x16x1x513xf16, {order = #NCHW}> {

    %0 = VPU.Floor(%arg0) : tensor<1x16x1x513xf16, {order = #NCHW}> -> tensor<1x16x1x513xf16, {order = #NCHW}>

    return %0 : tensor<1x16x1x513xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Floor({{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x16x1x513xf16, {order = #NCHW}> -> tensor<1x16x1x513xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x16x1x513xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @FloorAssignedSplitOverHeight
func.func @FloorAssignedSplitOverHeight(%arg0: tensor<1x16x16x512xf16, {order = #NCHW}>) -> tensor<1x16x16x512xf16, {order = #NCHW}> {

    %0 = VPU.Floor(%arg0) : tensor<1x16x16x512xf16, {order = #NCHW}> -> tensor<1x16x16x512xf16, {order = #NCHW}>

    return %0 : tensor<1x16x16x512xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Floor({{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x16x16x512xf16, {order = #NCHW}> -> tensor<1x16x16x512xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x16x16x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @FloorAssignedClustering
func.func @FloorAssignedClustering(%arg0: tensor<1x1x1x513xf16, {order = #NCHW}>) -> tensor<1x1x1x513xf16, {order = #NCHW}> {

    %0 = VPU.Floor(%arg0) : tensor<1x1x1x513xf16, {order = #NCHW}> -> tensor<1x1x1x513xf16, {order = #NCHW}>

    return %0 : tensor<1x1x1x513xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Floor({{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x513xf16, {order = #NCHW}> -> tensor<1x1x1x513xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x1x1x513xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @CeilingAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x16x1x513xf16, {order = #NCHW}>
func.func @CeilingAssignedSplitOverKernel(%arg0: tensor<1x16x1x513xf16, {order = #NCHW}>) -> tensor<1x16x1x513xf16, {order = #NCHW}> {

    %0 = VPU.Ceiling(%arg0) : tensor<1x16x1x513xf16, {order = #NCHW}> -> tensor<1x16x1x513xf16, {order = #NCHW}>

    return %0 : tensor<1x16x1x513xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.+]] = VPU.Ceiling([[INPUT_0]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x16x1x513xf16, {order = #NCHW}> -> tensor<1x16x1x513xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x16x1x513xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @CeilingAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x16x512xf16, {order = #NCHW}>
func.func @CeilingAssignedSplitOverHeight(%arg0: tensor<1x1x16x512xf16, {order = #NCHW}>) -> tensor<1x1x16x512xf16, {order = #NCHW}> {

    %0 = VPU.Ceiling(%arg0) : tensor<1x1x16x512xf16, {order = #NCHW}> -> tensor<1x1x16x512xf16, {order = #NCHW}>

    return %0 : tensor<1x1x16x512xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.+]] = VPU.Ceiling([[INPUT_0]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x16x512xf16, {order = #NCHW}> -> tensor<1x1x16x512xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x1x16x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @CeilingAssignedClustering
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x1x513xf16, {order = #NCHW}>
func.func @CeilingAssignedClustering(%arg0: tensor<1x1x1x513xf16, {order = #NCHW}>) -> tensor<1x1x1x513xf16, {order = #NCHW}> {

    %0 = VPU.Ceiling(%arg0) : tensor<1x1x1x513xf16, {order = #NCHW}> -> tensor<1x1x1x513xf16, {order = #NCHW}>

    return %0 : tensor<1x1x1x513xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.+]] = VPU.Ceiling([[INPUT_0]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x513xf16, {order = #NCHW}> -> tensor<1x1x1x513xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x1x1x513xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @RoundAssignedSplitOverKernel
func.func @RoundAssignedSplitOverKernel(%arg0: tensor<1x16x1x513xf16, {order = #NCHW}>) -> tensor<1x16x1x513xf16, {order = #NCHW}> {
    %0 = VPU.Round(%arg0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x16x1x513xf16, {order = #NCHW}> -> tensor<1x16x1x513xf16, {order = #NCHW}>

    return %0 : tensor<1x16x1x513xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Round({{[^:]+}}) {mode = #IE.round_mode<HALF_TO_EVEN>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x16x1x513xf16, {order = #NCHW}> -> tensor<1x16x1x513xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x16x1x513xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @RoundAssignedSplitOverHeight
func.func @RoundAssignedSplitOverHeight(%arg0: tensor<1x1x16x512xf16, {order = #NCHW}>) -> tensor<1x1x16x512xf16, {order = #NCHW}> {

    %0 = VPU.Round(%arg0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x1x16x512xf16, {order = #NCHW}> -> tensor<1x1x16x512xf16, {order = #NCHW}>

    return %0 : tensor<1x1x16x512xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Round({{[^:]+}}) {mode = #IE.round_mode<HALF_TO_EVEN>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x16x512xf16, {order = #NCHW}> -> tensor<1x1x16x512xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x1x16x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @RoundAssignedClustering
func.func @RoundAssignedClustering(%arg0: tensor<1x1x1x513xf16, {order = #NCHW}>) -> tensor<1x1x1x513xf16, {order = #NCHW}> {

    %0 = VPU.Round(%arg0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x1x1x513xf16, {order = #NCHW}> -> tensor<1x1x1x513xf16, {order = #NCHW}>

    return %0 : tensor<1x1x1x513xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Round({{[^:]+}}) {mode = #IE.round_mode<HALF_TO_EVEN>, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x513xf16, {order = #NCHW}> -> tensor<1x1x1x513xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x1x1x513xf16, {order = #NCHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AssignedAccordingToParentStrategySOK
func.func @AssignedAccordingToParentStrategySOK(%arg0: tensor<1x128x2x2xf16, {order = #NHWC}>) -> tensor<1x1024x2x2xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1024x1x1x4xsi32> = dense<10> : tensor<1024x1x1x4xsi32>
    %cst_0 = const.Declare tensor<1024x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [1024, 128, 1, 1], strides = [1, 1]} -> tensor<1x1024x2x2xf16, {order = #NHWC}>
    %1 = VPU.Gelu(%0) : tensor<1x1024x2x2xf16, {order = #NHWC}> -> tensor<1x1024x2x2xf16, {order = #NHWC}>
    return %1 : tensor<1x1024x2x2xf16, {order = #NHWC}>

    //CHECK-DAG:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<1024x1x1x4xsi32> = dense<10> : tensor<1024x1x1x4xsi32>
    //CHECK-DAG:        [[WEIGHTS:%.*]] = const.Declare tensor<1024x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x128x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK:            [[VAL0:%.*]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]])
    //CHECK-SAME:                   {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [1024, 128, 1, 1], strides = [1, 1]} -> tensor<1x1024x2x2xf16, {order = #NHWC}>
    //CHECK:            [[GELU:%.*]] = VPU.Gelu([[VAL0]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x1024x2x2xf16, {order = #NHWC}> -> tensor<1x1024x2x2xf16, {order = #NHWC}>
    //CHECK:            return [[GELU]] : tensor<1x1024x2x2xf16, {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AssignedAccordingToParentStrategySOH
func.func @AssignedAccordingToParentStrategySOH(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} -> tensor<1x80x28x28xf16, {order = #NHWC}>
    %1 = VPU.Gelu(%0) : tensor<1x80x28x28xf16, {order = #NHWC}> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %1 : tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK-DAG:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    //CHECK-DAG:        [[WEIGHTS:%.*]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK:            [[VAL0:%.*]] = VPU.NCE.Convolution(%arg0, %cst_0, %cst)
    //CHECK-SAME:                {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} -> tensor<1x80x28x28xf16, {order = #NHWC}>
    //CHECK:            [[GELU:%.*]] = VPU.Gelu([[VAL0]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x80x28x28xf16, {order = #NHWC}> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    //CHECK:            return [[GELU]] : tensor<1x80x28x28xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @ReduceL1SplitOverKernel(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1024x1x1xf16> {
func.func @ReduceL1SplitOverKernel(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1024x1x1xf16> {
  %0 = VPU.ReduceL1(%arg0) {axes_value = [2, 3], keep_dims} : tensor<1x1024x7x7xf16> -> tensor<1x1024x1x1xf16>
  return %0 : tensor<1x1024x1x1xf16>

// CHECK:       %[[VAL_1:.*]] = VPU.ReduceL1(%[[VAL_0]]) {axes_value = [2, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
// CHECK-SAME:    : tensor<1x1024x7x7xf16> -> tensor<1x1024x1x1xf16>
// CHECK:       return %[[VAL_1]] : tensor<1x1024x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @ReduceL2SplitOverKernel(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1024x1x1xf16> {
func.func @ReduceL2SplitOverKernel(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1024x1x1xf16> {
  %0 = VPU.ReduceL2(%arg0) {axes_value = [2, 3], keep_dims} : tensor<1x1024x7x7xf16> -> tensor<1x1024x1x1xf16>
  return %0 : tensor<1x1024x1x1xf16>

// CHECK:       %[[VAL_1:.*]] = VPU.ReduceL2(%[[VAL_0]]) {axes_value = [2, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
// CHECK-SAME:    : tensor<1x1024x7x7xf16> -> tensor<1x1024x1x1xf16>
// CHECK:       return %[[VAL_1]] : tensor<1x1024x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @ReduceLogicalAndClustering(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x1x1xf16> {
func.func @ReduceLogicalAndClustering(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x1x1xf16> {
  %0 = VPU.ReduceLogicalAnd(%arg0) {axes_value = [1, 2, 3], keep_dims} : tensor<1x1024x7x7xf16> -> tensor<1x1x1x1xf16>
  return %0 : tensor<1x1x1x1xf16>

// CHECK:       %[[VAL_1:.*]] = VPU.ReduceLogicalAnd(%[[VAL_0]]) {axes_value = [1, 2, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
// CHECK-SAME:    : tensor<1x1024x7x7xf16> -> tensor<1x1x1x1xf16>
// CHECK:       return %[[VAL_1]] : tensor<1x1x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @ReduceLogicalOrClustering(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x1x1xf16> {
func.func @ReduceLogicalOrClustering(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x1x1xf16> {
  %0 = VPU.ReduceLogicalOr(%arg0) {axes_value = [1, 2, 3], keep_dims} : tensor<1x1024x7x7xf16> -> tensor<1x1x1x1xf16>
  return %0 : tensor<1x1x1x1xf16>

// CHECK:       %[[VAL_1:.*]] = VPU.ReduceLogicalOr(%[[VAL_0]]) {axes_value = [1, 2, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
// CHECK-SAME:    : tensor<1x1024x7x7xf16> -> tensor<1x1x1x1xf16>
// CHECK:       return %[[VAL_1]] : tensor<1x1x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @ReduceMaxSplitOverHeight(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
func.func @ReduceMaxSplitOverHeight(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
  %0 = VPU.ReduceMax(%arg0) {axes_value = [1, 3], keep_dims} : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
  return %0 : tensor<1x1x7x1xf16>

// CHECK:       %[[VAL_1:.*]] = VPU.ReduceMax(%[[VAL_0]]) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
// CHECK-SAME:    : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
// CHECK:       return %[[VAL_1]] : tensor<1x1x7x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @ReduceMeanSplitOverHeight(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
func.func @ReduceMeanSplitOverHeight(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
  %0 = VPU.ReduceMean(%arg0) {axes_value = [1, 3], keep_dims} : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
  return %0 : tensor<1x1x7x1xf16>

// CHECK:       %[[VAL_1:.*]] = VPU.ReduceMean(%[[VAL_0]]) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
// CHECK-SAME:    : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
// CHECK:       return %[[VAL_1]] : tensor<1x1x7x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @ReduceProdSplitOverHeight(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
func.func @ReduceProdSplitOverHeight(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
  %0 = VPU.ReduceProd(%arg0) {axes_value = [1, 3], keep_dims} : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
  return %0 : tensor<1x1x7x1xf16>

// CHECK:       %[[VAL_1:.*]] = VPU.ReduceProd(%[[VAL_0]]) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
// CHECK-SAME:    : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
// CHECK:       return %[[VAL_1]] : tensor<1x1x7x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @ReduceSumSplitOverHeight(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
func.func @ReduceSumSplitOverHeight(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
  %0 = VPU.ReduceSum(%arg0) {axes_value = [1, 3], keep_dims} : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
  return %0 : tensor<1x1x7x1xf16>

// CHECK:       %[[VAL_1:.*]] = VPU.ReduceSum(%[[VAL_0]]) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
// CHECK-SAME:    : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
// CHECK:       return %[[VAL_1]] : tensor<1x1x7x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @ReduceSum5D(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<1x1024x7x7x3xf16>) -> tensor<1x1x7x1x3xf16> {
func.func @ReduceSum5D(%arg0: tensor<1x1024x7x7x3xf16>) -> tensor<1x1x7x1x3xf16> {
  %0 = VPU.ReduceSum(%arg0) {axes_value = [1, 3], keep_dims} : tensor<1x1024x7x7x3xf16> -> tensor<1x1x7x1x3xf16>
  return %0 : tensor<1x1x7x1x3xf16>

// CHECK:       %[[VAL_1:.*]] = VPU.ReduceSum(%[[VAL_0]]) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
// CHECK-SAME:    : tensor<1x1024x7x7x3xf16> -> tensor<1x1x7x1x3xf16>
// CHECK:       return %[[VAL_1]] : tensor<1x1x7x1x3xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PropagateDistributedPermuteCast
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x144x16x48xf16, {order = #NHWC}>)
func.func @PropagateDistributedPermuteCast(%arg0: tensor<1x144x16x48xf16, {order = #NHWC}>) -> tensor<1x128x144x16xf16, {order = #NHWC}>{
    %eltwise = VPU.NCE.Eltwise(%arg0, %arg0) {
        op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
            -> tensor<1x144x16x48xf16, {order = #NCHW}>
    %permute_cast = VPU.PermuteCast(%eltwise) {
        dst_order = #NHWC, mem_perm = #NCHW } : tensor<1x144x16x48xf16, {order = #NCHW}>
            -> tensor<1x48x144x16xf16, {order = #NHWC}>
    %wt = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
    %weights = const.Declare tensor<128x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x48x3x3xf16, {order = #NHWC}>
    %conv = VPU.NCE.Convolution(%permute_cast, %weights, %wt) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [128, 48, 3, 3], strides = [1, 1]}
            -> tensor<1x128x144x16xf16, {order = #NHWC}>
    return %conv : tensor<1x128x144x16xf16, {order = #NHWC}>
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>
    // CHECK:       [[PERMUTE_CAST:%.+]] = VPU.PermuteCast([[ELTWISE]])
    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PropagateDistributedLayoutCast
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x48x144x16xf16, {order = #NHWC}>)
func.func @PropagateDistributedLayoutCast(%arg0: tensor<1x48x144x16xf16, {order = #NHWC}>) -> tensor<1x128x144x16xf16, {order = #NHWC}>{
    %eltwise = VPU.NCE.Eltwise(%arg0, %arg0) {
        op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
            -> tensor<1x48x144x16xf16, {order = #NHWC}>
    %layout_cast = VPU.LayoutCast(%eltwise) {dst_order = #NHWC} : tensor<1x48x144x16xf16, {order = #NHWC}> -> tensor<1x48x144x16xf16, {order = #NHWC}>
    %wt = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
    %weights = const.Declare tensor<128x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x48x3x3xf16, {order = #NHWC}>
    %conv = VPU.NCE.Convolution(%layout_cast, %weights, %wt) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [128, 48, 3, 3], strides = [1, 1]}
            -> tensor<1x128x144x16xf16, {order = #NHWC}>
    return %conv : tensor<1x128x144x16xf16, {order = #NHWC}>
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>
    // CHECK:       [[LAYOUT_CAST:%.+]] = VPU.LayoutCast([[ELTWISE]])
    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AssignedAccordingToParentStrategyWithRollback
func.func @AssignedAccordingToParentStrategyWithRollback(%arg0: tensor<1x64x52x52xf16>) -> tensor<1x64x52x52xf16> {
    %cst = const.Declare tensor<64x1x1x4xsi32> = dense<10>  : tensor<64x1x1x4xsi32>
    %cst_0 = const.Declare tensor<64x16x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<1.000000e+00> : tensor<64x1x1x3x3xf32>, [#const.Reshape<[64, 1, 3, 3]>, #const.CastElemType<f16>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>>, #const.Reshape<[64, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
    %0 = VPU.Swish(%arg0) {beta_value = 1.000000e+00 : f64} : tensor<1x64x52x52xf16> -> tensor<1x64x52x52xf16>
    %1 = VPU.ShapeCast {shape = [1, 64, 169, 16]} inputs(%0 : tensor<1x64x52x52xf16>) -> tensor<1x64x169x16xf16>
    %2 = VPU.NCE.Permute(%1) {dstElemType = f16, dstOrder = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, expandedChannels = 64 : i64, ppe = #VPU.PPEStub<>} -> tensor<1x64x169x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    %3 = VPU.ShapeCast {shape = [1, 64, 52, 52]} inputs(%2 : tensor<1x64x169x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>) -> tensor<1x64x52x52xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    %4 = VPU.NCE.DepthConvolution(%3, %cst_0, %cst) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [64, 1, 3, 3], strides = [1, 1]} -> tensor<1x64x52x52xf16>
    %5 = VPU.Swish(%4) {beta_value = 1.000000e+00 : f64} : tensor<1x64x52x52xf16> -> tensor<1x64x52x52xf16>
    return %5 : tensor<1x64x52x52xf16>

    //CHECK-DAG:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    //CHECK-DAG:        [[WEIGHTS:%.*]] = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x1x1x3x3xf32>, [#const.Reshape<[64, 1, 3, 3]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[64, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]
    //CHECK:            [[SWISH_1:%.*]] = VPU.Swish(%arg0) {beta_value = 1.000000e+00 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x64x52x52xf16> -> tensor<1x64x52x52xf16>
    //CHECK:            [[SHAPECAST_1:%.+]] = VPU.ShapeCast {shape = [1, 64, 169, 16]} inputs([[SWISH_1]]
    //CHECK:            [[PERMUTE:%.*]] = VPU.NCE.Permute([[SHAPECAST_1]]) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 64 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEStub<>} -> tensor<1x64x169x16xf16, {order = #NHWC}>
    //CHECK:            [[SHAPECAST_2:%.+]] = VPU.ShapeCast {shape = [1, 64, 52, 52]} inputs([[PERMUTE]]
    //CHECK:            [[DEPTHCONV:%.*]] = VPU.NCE.DepthConvolution([[SHAPECAST_2]], [[WEIGHTS]], [[WEIGHTSTABLE]])
    //CHECK-SAME:                   {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 1, 3, 3], strides = [1, 1]} -> tensor<1x64x52x52xf16>
    //CHECK:            [[SWISH_2:%.*]] = VPU.Swish([[DEPTHCONV]]) {beta_value = 1.000000e+00 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x52x52xf16> -> tensor<1x64x52x52xf16>
    //CHECK:            return [[SWISH_2]] : tensor<1x64x52x52xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MVN1NormalizeAssignedSplitOverKernel
func.func @MVN1NormalizeAssignedSplitOverKernel(%arg0: tensor<1x3x20x35971xf16>, %arg1: tensor<1x3x1x1xf16, {order = #NHWC}>) -> tensor<1x3x20x35971xf16> {
    %0 = VPU.MVN1Normalize(%arg0, %arg1) {across_channels = false, normalize_variance = false} : tensor<1x3x20x35971xf16>, tensor<1x3x1x1xf16, {order = #NHWC}> -> tensor<1x3x20x35971xf16>
    return %0: tensor<1x3x20x35971xf16>

    //CHECK:        [[VAL0:%.*]] = VPU.MVN1Normalize(%arg0, %arg1)
    // CHECK-SAME:      {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = false} :
    // CHECK-SAME:      tensor<1x3x20x35971xf16>, tensor<1x3x1x1xf16, {order = #NHWC}> -> tensor<1x3x20x35971xf16>

    //CHECK:        return [[VAL0]] : tensor<1x3x20x35971xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MVN1MeanVarAssignedClustering
func.func @MVN1MeanVarAssignedClustering(%arg0: tensor<1x1x1x2xf32, {order = #NHWC}>) -> tensor<1x1x1x2xf16, {order = #NHWC}> {
    %0 = VPU.MVN1MeanVar(%arg0) {
            across_channels = true,
            eps = 1.000000e-09 : f64,
            normalize_variance = true,
            orig_shape = [1, 1, 1, 515971],
            output_type = f16} :
        tensor<1x1x1x2xf32, {order = #NHWC}> -> tensor<1x1x1x2xf16, {order = #NHWC}>
    return %0: tensor<1x1x1x2xf16, {order = #NHWC}>

    //CHECK:        [[VAL0:%.*]] = VPU.MVN1MeanVar(%arg0)
    // CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>

    //CHECK:        return [[VAL0]] : tensor<1x1x1x2xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MVN1SumAssignedSplitOverKernel
func.func @MVN1SumAssignedSplitOverKernel(%arg0: tensor<1x12x20x3997xf16>) -> tensor<1x12x1x2xf32, {order = #NHWC}> {
    %0 = VPU.MVN1SumOp(%arg0) {
            across_channels = false,
            normalize_variance = true,
            output_height = 1 : i64} :
        tensor<1x12x20x3997xf16> -> tensor<1x12x1x2xf32, {order = #NHWC}>
    return %0: tensor<1x12x1x2xf32, {order = #NHWC}>

    //CHECK:        [[VAL0:%.*]] = VPU.MVN1SumOp(%arg0)
    // CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>

    //CHECK:        return [[VAL0]] : tensor<1x12x1x2xf32, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.TileResource 4 of @NCE at 1.700000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}
// CHECK-LABEL: @MVN1SumAssignedSOH
func.func @MVN1SumAssignedSOH(%arg0: tensor<1x3x3997x1xf16, {order = #NHWC}>) -> tensor<1x3x4x2xf32, {order = #NHWC}> {
    %0 = VPU.MVN1SumOp(%arg0) {
            across_channels = false,
            normalize_variance = true,
            output_height = 4 : i64} :
        tensor<1x3x3997x1xf16, {order = #NHWC}> -> tensor<1x3x4x2xf32, {order = #NHWC}>
    return %0: tensor<1x3x4x2xf32, {order = #NHWC}>

    //CHECK:        [[VAL0:%.*]] = VPU.MVN1SumOp(%arg0)
    // CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

    //CHECK:        return [[VAL0]] : tensor<1x3x4x2xf32, {order = #NHWC}>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @LogSoftmaxAssignedClustering
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x1x1x512xf16, {order = #NCHW}>
func.func @LogSoftmaxAssignedClustering(%arg0: tensor<1x1x1x512xf16, {order = #NCHW}>) -> tensor<1x1x1x512xf16, {order = #NCHW}> {

    %1 = VPU.LogSoftmax(%arg0) {axisInd = 3} : tensor<1x1x1x512xf16, {order = #NCHW}> -> tensor<1x1x1x512xf16, {order = #NCHW}>

    return %1 : tensor<1x1x1x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultLogSoftmax:%.+]] = VPU.LogSoftmax([[INPUT]]) {axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x512xf16, {order = #NCHW}> -> tensor<1x1x1x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultLogSoftmax]] : tensor<1x1x1x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @LogSoftmaxAssignedSplitOverKernel
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x4x1x512xf16, {order = #NCHW}>
func.func @LogSoftmaxAssignedSplitOverKernel(%arg0: tensor<1x4x1x512xf16, {order = #NCHW}>) -> tensor<1x4x1x512xf16, {order = #NCHW}> {
    %1 = VPU.LogSoftmax(%arg0) {axisInd = 3} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>

    return %1 : tensor<1x4x1x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultLogSoftmax:%.+]] = VPU.LogSoftmax([[INPUT]]) {axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultLogSoftmax]] : tensor<1x4x1x512xf16, {order = #NCHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @NHWCLogSoftmaxAssignedSplitOverHeight
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x8x4x76xf16, {order = #NHWC}>
func.func @NHWCLogSoftmaxAssignedSplitOverHeight(%arg0: tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x8x4x76xf16, {order = #NHWC}> {
    %1 = VPU.LogSoftmax(%arg0) {axisInd = 3} : tensor<1x8x4x76xf16, {order = #NHWC}> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    return %1 : tensor<1x8x4x76xf16, {order = #NHWC}>

    //CHECK:   [[ResultLogSoftmax:%.+]] = VPU.LogSoftmax([[INPUT]]) {axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x8x4x76xf16, {order = #NHWC}> -> tensor<1x8x4x76xf16, {order = #NHWC}>
    //CHECK:   return [[ResultLogSoftmax]] : tensor<1x8x4x76xf16, {order = #NHWC}>

}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: @GroupedMatMulAssignedSOG
func.func @GroupedMatMulAssignedSOG(%arg0:  tensor<4x1x32x64x1xf16, {order =#GNHWC}>, %arg1: tensor<4x64x32x1x1xf16, {order = #GNHWC}>) -> tensor<4x1x64x64x1xf16, {order = #GNHWC}> {
    %cst = const.Declare tensor<4x64x1x1x4xsi32> = dense<1> : tensor<4x64x1x1x4xsi32>

    %0 = VPU.NCE.MatMul(%arg0, %arg1, %cst) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        rawFilterShape = [4, 1, 64, 32, 1], strides = [1, 1]
    } -> tensor<4x1x64x64x1xf16, {order = #GNHWC}>

    return %0 : tensor<4x1x64x64x1xf16, {order = #GNHWC}>

    // CHECK:        [[CONST:%.+]] = const.Declare
    // CHECK:        [[MATMUL:%.+]] = VPU.NCE.MatMul
    // CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>
    // CHECK:        return [[MATMUL]] : tensor<4x1x64x64x1xf16, {order = #GNHWC}>
}


// -----

// CHECK-LABEL: func.func @SelectOpAssignedSplitOverWidth
// CHECK-SAME:  [[INPUT0:%.+]]: tensor<1x1x1x8192xf16>, [[INPUT1:%.+]]: tensor<1x1x1x8192xf16>, [[INPUT2:%.+]]: tensor<1x1x1x8192xf16>
func.func @SelectOpAssignedSplitOverWidth(%arg0: tensor<1x1x1x8192xf16>, %arg1: tensor<1x1x1x8192xf16>, %arg2: tensor<1x1x1x8192xf16>) -> tensor<1x1x1x8192xf16> {
    %0 = VPU.Select(%arg0, %arg1, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x8192xf16>, tensor<1x1x1x8192xf16>, tensor<1x1x1x8192xf16> -> tensor<1x1x1x8192xf16>
    return %0 : tensor<1x1x1x8192xf16>

    // CHECK:       [[SELECT:%.+]] = VPU.Select([[INPUT0]], [[INPUT1]], [[INPUT2]])
    // CHECK-SAME:          {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>}
    // CHECK:       return [[SELECT]]  : tensor<1x1x1x8192xf16>

}
