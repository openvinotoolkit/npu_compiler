//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --make-ops-with-distributed-tensor --make-distributed-copies %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToDistributedOpSOH
func.func @ConvToDistributedOpSOH(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} : tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<80x64x3x3xf16, {order = #NHWC}>, tensor<80x1x1x4xsi32> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.+]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[WEIGHTSTABLE_CMX:%.+]] = VPU.Copy([[WEIGHTSTABLE]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Convolution
    //CHECK-SAME:             [[INPUT_CMX]]
    //CHECK-SAME:             [[WEIGHTS_CMX]]
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]]) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [80, 64, 3, 3], strides = [1, 1]
    //CHECK-SAME:             }
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]
    // CHECK-SAME:                   -> tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK:        return [[OUT]] : tensor<1x80x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToDistributedOpSOK
func.func @ConvToDistributedOpSOK(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>) -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %cst_0 = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [64, 128, 1, 1], strides = [1, 1]} : tensor<1x128x28x28xf16, {order = #NHWC}>, tensor<64x128x1x1xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<64x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    //CHECK:        [[WEIGHTSTABLE_CMX:%.+]] = VPU.Copy([[WEIGHTSTABLE]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX]],
    //CHECK-SAME:             [[WEIGHTS_CMX]],
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]]) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:                 rawFilterShape = [64, 128, 1, 1], strides = [1, 1]
    //CHECK-SAME:             }
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToDistributedOpSOB
// CHECK-SAME:      [[INPUT:%.+]]: tensor<2x16x96x96xf16, {order = #NHWC}>
func.func @ConvToDistributedOpSOB(%arg0: tensor<2x16x96x96xf16, {order = #NHWC}>) -> tensor<2x16x96x94xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %cst_0 = const.Declare tensor<16x16x3x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x5xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 5], strides = [1, 1]} : tensor<2x16x96x96xf16, {order = #NHWC}>, tensor<16x16x3x5xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<2x16x96x94xf16, {order = #NHWC}>
    return %0 : tensor<2x16x96x94xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<16x16x3x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x5xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<2x16x96x96xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<16x16x3x5xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[WEIGHTSTABLE_CMX:%.+]] = VPU.Copy([[WEIGHTSTABLE]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:           [[INPUT_CMX]],
    //CHECK-SAME:           [[WEIGHTS_CMX]],
    //CHECK-SAME:           [[WEIGHTSTABLE_CMX]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<2x16x96x94xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<2x16x96x94xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToDistributedOpClustering
func.func @ConvToDistributedOpClustering(%arg0: tensor<1x64x14x14xf16, {order = #NHWC}>) -> tensor<1x48x14x14xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    %cst_0 = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [48, 64, 3, 3], strides = [1, 1]} : tensor<1x64x14x14xf16, {order = #NHWC}>, tensor<48x64x3x3xf16, {order = #NHWC}>, tensor<48x1x1x4xsi32> -> tensor<1x48x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x48x14x14xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.+]] = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x64x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<48x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[WEIGHTSTABLE_CMX:%.+]] = VPU.Copy([[WEIGHTSTABLE]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<48x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX]]
    //CHECK-SAME:             [[WEIGHTS_CMX]],
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x48x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x48x14x14xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvToDistributedOpSOH
func.func @DepthConvToDistributedOpSOH(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[WEIGHTSTABLE_CMX:%.+]] = VPU.Copy([[WEIGHTSTABLE]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.DepthConvolution(
    //CHECK-SAME:             [[INPUT_CMX]]
    //CHECK-SAME:             [[WEIGHTS_CMX]],
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----


#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvToDistributedOpSOHWithAlign
func.func @DepthConvToDistributedOpSOHWithAlign(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[WEIGHTSTABLE_CMX:%.+]] = VPU.Copy([[WEIGHTSTABLE]]

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.DepthConvolution(
    //CHECK-SAME:             [[INPUT_CMX]]
    //CHECK-SAME:             [[WEIGHTS_CMX]],
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DepthConvToDistributedOpSOK
func.func @DepthConvToDistributedOpSOK(%arg0: tensor<1x128x56x56xf16, {order = #NHWC}>) -> tensor<1x128x56x56xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    %cst_1 = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [128, 1, 3, 3], strides = [1, 1]} -> tensor<1x128x56x56xf16, {order = #NHWC}>
    return %0 : tensor<1x128x56x56xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.+]] = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<128x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy
    //CHECK-SAME:   !VPU.DistributedTensor<1x128x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<128x16x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    //CHECK:        [[WEIGHTSTABLE_CMX:%.+]] = VPU.Copy([[WEIGHTSTABLE]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<128x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.DepthConvolution(
    //CHECK-SAME:             [[INPUT_CMX]]
    //CHECK-SAME:             [[WEIGHTS_CMX]]
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x128x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x128x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DepthConvToDistributedOpClustering
func.func @DepthConvToDistributedOpClustering(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[WEIGHTSTABLE_CMX:%.+]] = VPU.Copy([[WEIGHTSTABLE]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.DepthConvolution(
    //CHECK-SAME:             [[INPUT_CMX]]
    //CHECK-SAME:             [[WEIGHTS_CMX]]
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolToDistributedOpSOH
func.func @MaxPoolToDistributedOpSOH(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.MaxPool([[INPUT_CMX]]) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]}
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolToDistributedOpSOHWithAlign
func.func @MaxPoolToDistributedOpSOHWithAlign(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.MaxPool(
    //CHECK-SAME:             [[INPUT_CMX]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolToDistributedOpClustering
func.func @MaxPoolToDistributedOpClustering(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.MaxPool(
    //CHECK-SAME:             [[INPUT_CMX]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  func.func @MaxPoolToDistributedOpSOB
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<2x32x112x112xf16, {order = #NHWC}>)
func.func @MaxPoolToDistributedOpSOB(%input: tensor<2x32x112x112xf16, {order = #NHWC}>) -> tensor<2x32x112x112xf16, {order = #NHWC}> {
    %maxpool = VPU.NCE.MaxPool(%input) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        kernel_size = [1, 1]
    } -> tensor<2x32x112x112xf16, {order = #NHWC}>
    return %maxpool : tensor<2x32x112x112xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]

    // CHECK:       [[OUT_CMX:%.+]] = VPU.NCE.MaxPool([[INPUT_CMX]]) {
    // CHECK-SAME:          kernel_size = [1, 1],
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEStub<>,
    // CHECK-SAME:          strides = [1, 1]}
    // CHECK-SAME:  -> !VPU.DistributedTensor<2x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>

    // CHECK:       [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    // CHECK:       return [[OUT]] : tensor<2x32x112x112xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @ReduceL1SplitOverKernel(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1024x1x1xf16> {
func.func @ReduceL1SplitOverKernel(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1024x1x1xf16> {
  %0 = VPU.ReduceL1(%arg0) {axes_value = [2, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x1024x7x7xf16> -> tensor<1x1024x1x1xf16>
  return %0 : tensor<1x1024x1x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.Copy(%[[VAL_0]]
    // CHECK:   %[[VAL_4:.*]] = VPU.ReduceL1(%[[VAL_1]]) {axes_value = [2, 3], keep_dims}
    // CHECK-SAME:              -> !VPU.DistributedTensor<1x1024x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_7:.*]] = VPU.Copy(%[[VAL_4]]

    // CHECK:   return %[[VAL_7]] : tensor<1x1024x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @ReduceL2SplitOverKernel(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1024x1x1xf16> {
func.func @ReduceL2SplitOverKernel(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1024x1x1xf16> {
  %0 = VPU.ReduceL2(%arg0) {axes_value = [2, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x1024x7x7xf16> -> tensor<1x1024x1x1xf16>
  return %0 : tensor<1x1024x1x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.Copy(%[[VAL_0]]
    // CHECK-SAME:              > !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_4:.*]] = VPU.ReduceL2(%[[VAL_1]]) {axes_value = [2, 3], keep_dims}
    // CHECK-SAME:               -> !VPU.DistributedTensor<1x1024x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_7:.*]] = VPU.Copy(%[[VAL_4]]
    // CHECK:   return %[[VAL_7]] : tensor<1x1024x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @ReduceLogicalAndClustering(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x1x1xf16> {
func.func @ReduceLogicalAndClustering(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x1x1xf16> {
  %0 = VPU.ReduceLogicalAnd(%arg0) {axes_value = [1, 2, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1024x7x7xf16> -> tensor<1x1x1x1xf16>
  return %0 : tensor<1x1x1x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.Copy(%[[VAL_0]]
    // CHECK-SAME:              -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_4:.*]] = VPU.ReduceLogicalAnd(%[[VAL_1]]
    // CHECK-SAME:               -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_7:.*]] = VPU.Copy(%[[VAL_4]]
    // CHECK:   return %[[VAL_7]] : tensor<1x1x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @ReduceLogicalOrClustering(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x1x1xf16> {
func.func @ReduceLogicalOrClustering(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x1x1xf16> {
  %0 = VPU.ReduceLogicalOr(%arg0) {axes_value = [1, 2, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1024x7x7xf16> -> tensor<1x1x1x1xf16>
  return %0 : tensor<1x1x1x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.Copy(%[[VAL_0]]
    // CHECK-SAME:              -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_4:.*]] = VPU.ReduceLogicalOr(%[[VAL_1]]
    // CHECK-SAME:               -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_7:.*]] = VPU.Copy(%[[VAL_4]]
    // CHECK:   return %[[VAL_7]] : tensor<1x1x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @ReduceMaxSplitOverHeight(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
func.func @ReduceMaxSplitOverHeight(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
  %0 = VPU.ReduceMax(%arg0) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
  return %0 : tensor<1x1x7x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.Copy(%[[VAL_0]]
    // CHECK-SAME:               -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_4:.*]] = VPU.ReduceMax(%[[VAL_1]]
    // CHECK-SAME:               -> !VPU.DistributedTensor<1x1x7x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_7:.*]] = VPU.Copy(%[[VAL_4]]
    // CHECK:   return %[[VAL_7]] : tensor<1x1x7x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @ReduceMeanSplitOverHeight(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
func.func @ReduceMeanSplitOverHeight(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
  %0 = VPU.ReduceMean(%arg0) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
  return %0 : tensor<1x1x7x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.Copy(%[[VAL_0]]
    // CHECK-SAME:               -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_4:.*]] = VPU.ReduceMean(%[[VAL_1]]
    // CHECK-SAME:               -> !VPU.DistributedTensor<1x1x7x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_7:.*]] = VPU.Copy(%[[VAL_4]]
    // CHECK:   return %[[VAL_7]] : tensor<1x1x7x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @ReduceProdSplitOverHeight(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
func.func @ReduceProdSplitOverHeight(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
  %0 = VPU.ReduceProd(%arg0) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
  return %0 : tensor<1x1x7x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.Copy(%[[VAL_0]]
    // CHECK-SAME:               -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_4:.*]] = VPU.ReduceProd(%[[VAL_1]]
    // CHECK-SAME:               -> !VPU.DistributedTensor<1x1x7x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_7:.*]] = VPU.Copy(%[[VAL_4]]
    // CHECK:   return %[[VAL_7]] : tensor<1x1x7x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @ReduceSumSplitOverHeight(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
func.func @ReduceSumSplitOverHeight(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
  %0 = VPU.ReduceSum(%arg0) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
  return %0 : tensor<1x1x7x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.Copy(%[[VAL_0]]
    // CHECK-SAME:               -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_4:.*]] = VPU.ReduceSum(%[[VAL_1]]
    // CHECK-SAME:               -> !VPU.DistributedTensor<1x1x7x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_7:.*]] = VPU.Copy(%[[VAL_4]]
    // CHECK:   return %[[VAL_7]] : tensor<1x1x7x1xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddToDistributedOpSOH
func.func @EltwiseAddToDistributedOpSOH(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>, %arg1: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) { multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, op_type = #VPU.eltwise_type<ADD> } :
         tensor<1x32x112x112xf16, {order = #NHWC}>, tensor<1x32x112x112xf16, {order = #NHWC}>
         -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0: tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[INPUT0_CMX:%.+]] = VPU.Copy
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1_CMX:%.+]] = VPU.Copy

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Eltwise([[INPUT0_CMX]], [[INPUT1_CMX]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]
    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddToDistributedOpClustering
func.func @EltwiseAddToDistributedOpClustering(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>, %arg1: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) { multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, ppe = #VPU.PPEStub<>, op_type = #VPU.eltwise_type<ADD> } :
         tensor<1x32x14x14xf16, {order = #NHWC}>, tensor<1x32x14x14xf16, {order = #NHWC}>
         -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0: tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[INPUT0_CMX:%.+]] = VPU.Copy
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1_CMX:%.+]] = VPU.Copy(%arg1
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Eltwise([[INPUT0_CMX]], [[INPUT1_CMX]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AvgPoolToDistributedOpSOH
func.func @AvgPoolToDistributedOpSOH(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [3, 3]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.AveragePool([[INPUT_CMX]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AvgPoolToDistributedOpSOHWithAlign
func.func @AvgPoolToDistributedOpSOHWithAlign(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [3, 3]
         } -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.AveragePool([[INPUT_CMX]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  func.func @AvgPoolToDistributedOpSOB
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<2x32x112x112xf16, {order = #NHWC}>)
func.func @AvgPoolToDistributedOpSOB(%input: tensor<2x32x112x112xf16, {order = #NHWC}>) -> tensor<2x32x112x112xf16, {order = #NHWC}> {
    %avgpool = VPU.NCE.AveragePool(%input) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        kernel_size = [1, 1]
    } -> tensor<2x32x112x112xf16, {order = #NHWC}>
    return %avgpool : tensor<2x32x112x112xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<2x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>

    // CHECK:       [[OUT_CMX:%.+]] = VPU.NCE.AveragePool([[INPUT_CMX]])
    // CHECK-SAME:  -> !VPU.DistributedTensor<2x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>

    // CHECK:       [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    // CHECK:       return [[OUT]] : tensor<2x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SparseConvToDistributedOpSOH
func.func @SparseConvToDistributedOpSOH(%arg0 : tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1 : tensor<1x64x28x28xi1, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {

    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1)
        -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    %weights = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<80x1x1x640xi1> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
                             sparsity_map=tensor<80x1x1x640xi1>, is_weights>

    %weights_table = const.Declare tensor<80x1x1x4xsi32> = dense<1> : tensor<80x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%input_sparse, %weights_sparse, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 01 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [80, 64, 3, 3],
            strides = [1, 1]
        } : !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>, !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<80x1x1x640xi1>, is_weights>, tensor<80x1x1x4xsi32> -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
        }

    return %0 : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                                  sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>

    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, %arg1)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<80x1x1x640xi1> = dense<1.000000e+00>
    // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]]) {is_weights}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<80x1x1x640xi1>, is_weights>

    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<80x1x1x4xsi32>

    // CHECK:       [[INPUT_SPARSE_CMX:%.+]] = VPU.Copy
    // CHECK-SAME:      ([[INPUT_SPARSE]]
    // CHECK-SAME:      -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                           sparsity_map=!VPU.DistributedTensor<1x64x28x28xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>>

    // CHECK:       [[WEIGHTS_SPARSE_CMX:%.+]] = VPU.Copy
    // CHECK-SAME:      ([[WEIGHTS_SPARSE]]
    // CHECK-SAME:      -> !VPU.SparseTensor<data=!VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
    // CHECK-SAME:                           sparsity_map=!VPU.DistributedTensor<80x1x1x640xi1, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
    // CHECK-SAME:                           is_weights>

    // CHECK:       [[WEIGHTS_TABLE_CMX:%.+]] = VPU.Copy
    // CHECK-SAME:      ([[CST_WEIGHTS_TABLE]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[OUT_CMX:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE_CMX]], [[WEIGHTS_SPARSE_CMX]], [[WEIGHTS_TABLE_CMX]])
    // CHECK-SAME:      -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                           sparsity_map=!VPU.DistributedTensor<1x80x28x28xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>>

    // CHECK:       [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>

    // CHECK:       return [[OUT]] : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                                     sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSOHAlignmentForEltwiseCase1
func.func @OptimizeSOHAlignmentForEltwiseCase1(%arg0: tensor<1x16x22x22xf16, {order = #NHWC}>, %arg1: tensor<1x16x22x22xf16, {order = #NHWC}>) -> tensor<1x16x22x22xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %cst_0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} : tensor<1x16x22x22xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x22x22xf16, {order = #NHWC}>
    %1 = VPU.NCE.Eltwise(%arg0, %arg1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, op_type = #VPU.eltwise_type<ADD>} -> tensor<1x16x22x22xf16, {order = #NHWC}>
    %2 = VPU.NCE.Eltwise(%0, %1) {ppe = #VPU.PPEStub<>, op_type = #VPU.eltwise_type<ADD>} -> tensor<1x16x22x22xf16, {order = #NHWC}>
    return %2 : tensor<1x16x22x22xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX_0:%.+]] = VPU.Copy
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[WEIGHTSTABLE_CMX:%.+]] = VPU.Copy([[WEIGHTSTABLE]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[OUT_CMX_0:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX_0]],
    //CHECK-SAME:             [[WEIGHTS_CMX]],
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUT_0:%.+]] = VPU.Copy([[OUT_CMX_0]]

    //CHECK:        [[INPUT0_CMX_1:%.+]] = VPU.Copy
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>

    //CHECK:        [[INPUT1_CMX_1:%.+]] = VPU.Copy
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>

    //CHECK:        [[OUT_CMX_1:%.+]] = VPU.NCE.Eltwise([[INPUT0_CMX_1]], [[INPUT1_CMX_1]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>

    //CHECK:        [[OUT_1:%.+]] = VPU.Copy([[OUT_CMX_1]]

    //CHECK:        [[OUT_2:%.+]] = VPU.NCE.Eltwise([[OUT_0]], [[OUT_1]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x16x22x22xf16, {order = #NHWC}>

    //CHECK:        return [[OUT_2]] : tensor<1x16x22x22xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSOHAlignmentForEltwiseCase2
func.func @OptimizeSOHAlignmentForEltwiseCase2(%arg0: tensor<1x16x22x22xf16, {order = #NHWC}>, %arg1: tensor<1x16x22x22xf16, {order = #NHWC}>) -> tensor<1x16x22x22xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, op_type = #VPU.eltwise_type<ADD>} -> tensor<1x16x22x22xf16, {order = #NHWC}>
    %1 = VPU.NCE.Eltwise(%0, %arg1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, op_type = #VPU.eltwise_type<ADD>} -> tensor<1x16x22x22xf16, {order = #NHWC}>
    %cst = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %cst_0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %2 = VPU.NCE.Convolution(%1, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} : tensor<1x16x22x22xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x22x22xf16, {order = #NHWC}>
    return %2 : tensor<1x16x22x22xf16, {order = #NHWC}>

    //CHECK:        [[INPUT0_CMX_0:%.+]] = VPU.Copy

    //CHECK:        [[INPUT1_CMX_0:%.+]] = VPU.Copy(%arg1
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>

    //CHECK:        [[OUT_CMX_0:%.+]] = VPU.NCE.Eltwise([[INPUT0_CMX_0]], [[INPUT1_CMX_0]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>

    //CHECK:        [[OUT_0:%.+]] = VPU.Copy([[OUT_CMX_0]]

    //CHECK:        [[INPUT0_CMX_1:%.+]] = VPU.Copy([[OUT_0]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>

    //CHECK:        [[INPUT1_CMX_1:%.+]] = VPU.Copy(%arg1
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>

    //CHECK:        [[OUT_CMX_1:%.+]] = VPU.NCE.Eltwise([[INPUT0_CMX_1]], [[INPUT1_CMX_1]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>

    //CHECK:        [[OUT_1:%.+]] = VPU.Copy([[OUT_CMX_1]]

    //CHECK:        [[WEIGHTSTABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX_2:%.+]] = VPU.Copy([[OUT_1]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[WEIGHTSTABLE_CMX:%.+]] = VPU.Copy([[WEIGHTSTABLE]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[OUT_CMX_2:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX_2]],
    //CHECK-SAME:             [[WEIGHTS_CMX]],
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUT_2:%.+]] = VPU.Copy([[OUT_CMX_2]]

    //CHECK:        return [[OUT_2]] : tensor<1x16x22x22xf16, {order = #NHWC}>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @MVNToDistributedOpDuplicateBuffer
func.func @MVNToDistributedOpDuplicateBuffer(%arg0: tensor<1x4x512x1xf16, {order = #NCWH}>) -> tensor<1x4x512x1xf16, {order = #NCWH}> {

    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true} : tensor<1x4x512x1xf16, {order = #NCWH}> -> tensor<1x4x512x1xf16, {order = #NCWH}>

    return %0: tensor<1x4x512x1xf16, {order = #NCWH}>

    //CHECK: [[ClusterCopy:%.+]] = VPU.Copy
    //CHECK-SAME:                 -> !VPU.DistributedTensor<1x4x512x1xf16, #NCWH, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK: [[RClusterMVN:%.+]] = VPU.MVN([[ClusterCopy]])
    //CHECK-SAME:                  -> !VPU.DistributedTensor<1x4x512x1xf16, #NCWH, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK: [[OUT:%.+]] = VPU.Copy([[RClusterMVN]]

    //CHECK: return [[OUT]] : tensor<1x4x512x1xf16, {order = #NCWH}>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @MVNToDistributedOpSegmentedBuffer
func.func @MVNToDistributedOpSegmentedBuffer(%arg0: tensor<1x4x512x1xf16, {order = #NCWH}>) -> tensor<1x4x512x1xf16, {order = #NCWH}> {

    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x4x512x1xf16, {order = #NCWH}> -> tensor<1x4x512x1xf16, {order = #NCWH}>

    return %0: tensor<1x4x512x1xf16, {order = #NCWH}>

    //CHECK: [[ClusterCopy:%.+]] = VPU.Copy
    //CHECK-SAME:                  -> !VPU.DistributedTensor<1x4x512x1xf16, #NCWH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK: [[RClusterMVN:%.+]] = VPU.MVN([[ClusterCopy]])
    //CHECK-SAME:                  -> !VPU.DistributedTensor<1x4x512x1xf16, #NCWH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK: [[OUT:%.+]] = VPU.Copy([[RClusterMVN]]

    //CHECK: return [[OUT]] : tensor<1x4x512x1xf16, {order = #NCWH}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @MVN6SOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x32x15x64xf16>)
func.func @MVN6SOK(%arg0: tensor<1x32x15x64xf16>) -> tensor<1x32x15x64xf16> {
    %0 = VPU.MVN6(%arg0) {axes = [2], eps = 1.000000e-02 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>} : tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16>
    return %0 : tensor<1x32x15x64xf16>

    //CHECK:        [[INPUT:%.+]] = VPU.Copy([[INPUT_DATA]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x15x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64

    //CHECK:        [[MVN:%.+]] = VPU.MVN6([[INPUT]])
    //CHECK-SAME:                   -> !VPU.DistributedTensor<1x32x15x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[MVN]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x15x64xf16>
}

// -----

// CHECK-LABEL: func.func @MVN6SOH
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x32x15x64xf16>)
func.func @MVN6SOH(%arg0: tensor<1x32x15x64xf16>) -> tensor<1x32x15x64xf16> {
    %0 = VPU.MVN6(%arg0) {axes = [1, 3], eps = 1.000000e-02 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>} : tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16>
    return %0 : tensor<1x32x15x64xf16>

    //CHECK:        [[INPUT:%.+]] = VPU.Copy([[INPUT_DATA]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x15x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64

    //CHECK:        [[MVN:%.+]] = VPU.MVN6([[INPUT]])
    //CHECK-SAME:                   -> !VPU.DistributedTensor<1x32x15x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[MVN]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x15x64xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @PadSwSOH
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x20x50xf16>)
func.func @PadSwSOH(%arg0: tensor<1x16x20x50xf16>) -> tensor<1x18x20x60xf16> {
    %0 = VPU.Pad(%arg0) {mode = #IE.pad_mode<EDGE>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 2, 0, 10]} : tensor<1x16x20x50xf16> -> tensor<1x18x20x60xf16>
    return %0 : tensor<1x18x20x60xf16>

    //CHECK:        [[INPUT:%.+]] = VPU.Copy([[INPUT_DATA]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x20x50xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[PAD:%.+]] = VPU.Pad([[INPUT]])
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x18x20x60xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[PAD]]
    //CHECK-SAME:                    -> tensor<1x18x20x60xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @PadSwSOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x20x50xf16>)
func.func @PadSwSOK(%arg0: tensor<1x16x20x50xf16>) -> tensor<1x16x22x52xf16> {
    %0 = VPU.Pad(%arg0) {mode = #IE.pad_mode<EDGE>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 0, 2, 2]} : tensor<1x16x20x50xf16> -> tensor<1x16x22x52xf16>
    return %0 : tensor<1x16x22x52xf16>

    //CHECK:        [[INPUT:%.+]] = VPU.Copy([[INPUT_DATA]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x20x50xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[PAD:%.+]] = VPU.Pad([[INPUT]])
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x22x52xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[PAD]]
    //CHECK-SAME:                    -> tensor<1x16x22x52xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UnrollSOKConvOutputSegmented
func.func @UnrollSOKConvOutputSegmented(%input: tensor<1x64x64x64xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [64, 64, 1, 1], strides = [1, 1]}
                : tensor<1x64x64x64xf16, {order = #NHWC}>, tensor<64x64x1x1xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x64x64xf16, {order = #NHWC}>
    %mvn = VPU.MVN(%conv) {
            across_channels = false, eps = 1.0E-4 : f64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true
            } : tensor<1x64x64x64xf16, {order = #NHWC}>
                -> tensor<1x64x64x64xf16, {order = #NHWC}>

    return %mvn : tensor<1x64x64x64xf16, {order = #NHWC}>

    // (DUP) CONV (SEG) -> (SEG) MVN (SEG)

    //CHECK:        [[CONV_IN:%.+]] = VPU.Copy
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[SOK_CONV:%.+]] = VPU.NCE.Convolution([[CONV_IN]],
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[MVN_IN:%.+]] = VPU.MVN
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[SOK_MVN:%.+]] = VPU.Copy([[MVN_IN]])
    //CHECK-SAME:       -> tensor<1x64x64x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UnrollSOKConvOutputSegmentedWithSlice
func.func @UnrollSOKConvOutputSegmentedWithSlice(%input: tensor<1x80x1x3000xf16, {order = #NHWC}>) -> tensor<1x384x1x1500xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<384x80x1x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<384x80x1x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<384x1x1x4xsi32> = dense<10> : tensor<384x1x1x4xsi32>

    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [384, 80, 1, 3], strides = [1, 1]}
               : tensor<1x80x1x3000xf16, {order = #NHWC}>, tensor<384x80x1x3xf16, {order = #NHWC}>, tensor<384x1x1x4xsi32> -> tensor<1x384x1x3000xf16, {order = #NHWC}>
    %slice = VPU.Slice %conv [0, 0, 0, 0] [1, 384, 1, 1500] : tensor<1x384x1x3000xf16, {order = #NHWC}> to tensor<1x384x1x1500xf16, {order = #NHWC}>
    %gelu =  VPU.Gelu(%slice) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x384x1x1500xf16, {order = #NHWC}> -> tensor<1x384x1x1500xf16, {order = #NHWC}>

    return %gelu : tensor<1x384x1x1500xf16, {order = #NHWC}>

    // (DUP) CONV (SEG) -> SLICE -> (SEG) GELU (SEG)

    // CHECK:     [[CONV_IN:%.+]] = VPU.Copy
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x80x1x3000xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:     [[SOK_CONV:%.+]] = VPU.NCE.Convolution([[CONV_IN]],
    // CHECK-SAME:   -> !VPU.DistributedTensor<1x384x1x3000xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK:     [[CONV_OUT:%.+]]  = VPU.Copy([[SOK_CONV]]
    // CHECK-SAME:   -> tensor<1x384x1x3000xf16, {order = #NHWC}>

    // CHECK:     [[SLICE:%.+]] = VPU.Slice [[CONV_OUT]] [0, 0, 0, 0] [1, 384, 1, 1500] : tensor<1x384x1x3000xf16, {order = #NHWC}> to tensor<1x384x1x1500xf16, {order = #NHWC}>

    // CHECK:     [[GELU_IN:%.+]] = VPU.Copy([[SLICE]]
    // CHECK-SAME: -> !VPU.DistributedTensor<1x384x1x1500xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK:     [[SOK_GELU:%.+]] = VPU.Gelu([[GELU_IN]]
    // CHECK-SAME: -> !VPU.DistributedTensor<1x384x1x1500xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UnrollSOKDWConvInputOutputSegmented
func.func @UnrollSOKDWConvInputOutputSegmented(%input: tensor<1x64x64x64xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>

    %mvn = VPU.MVN(%input) {across_channels = false, eps = 1.0E-4 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true}
            : tensor<1x64x64x64xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16, {order = #NHWC}>
    %dwconv = VPU.NCE.DepthConvolution(%mvn, %weights, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [64, 1, 1, 1], strides = [1, 1]}
                -> tensor<1x64x64x64xf16, {order = #NHWC}>

    return %dwconv : tensor<1x64x64x64xf16, {order = #NHWC}>

    // (SEG) MVN (SEG) -> (SEG) DWCONV (SEG)

    //CHECK:        [[MVN_IN:%.+]] = VPU.Copy
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[SOK_MVN:%.+]] = VPU.MVN([[MVN_IN]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    //CHECK:        VPU.Copy
    //CHECK-SAME:       -> tensor<1x64x64x64xf16, {order = #NHWC}>

    //CHECK:        [[DWCONV_IN:%.+]] = VPU.Copy
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[SOK_DWCONV:%.+]] = VPU.NCE.DepthConvolution
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UnrollSOKDWConvInputOutputDuplicated
func.func @UnrollSOKDWConvInputOutputDuplicated(%input: tensor<1x1x320x1xf16>) -> tensor<1x320x1x1xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<320x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x320xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 1, 1, 320]>, #const.Reshape<[1, 320, 1, 1]>, #const.Reshape<[320, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[320, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<320x1x1x4xsi32> = dense<10> : tensor<320x1x1x4xsi32>

    %mvn = VPU.MVN(%input) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true}
            : tensor<1x1x320x1xf16> -> tensor<1x1x320x1xf16>

    %reshape = VPU.AffineReshape(%mvn) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 320, 1, 1]}
            : tensor<1x1x320x1xf16> -> tensor<1x320x1x1xf16>

    %cast = VPU.PermuteCast(%reshape) {dst_order = #NHWC, mem_perm = #NHWC}
            : tensor<1x320x1x1xf16> -> tensor<1x320x1x1xf16, {order = #NHWC}>

    %dwconv = VPU.NCE.DepthConvolution(%cast, %weights, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [320, 1, 1, 1], strides = [1, 1]}
                -> tensor<1x320x1x1xf16, {order = #NHWC}>

    %activation = VPU.Sigmoid(%dwconv) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
            : tensor<1x320x1x1xf16, {order = #NHWC}> -> tensor<1x320x1x1xf16, {order = #NHWC}>

    return %activation : tensor<1x320x1x1xf16, {order = #NHWC}>

    // (DUP) MVN (DUP) -> (DUP) DWCONV (DUP | SEG)

    //CHECK:        [[MVN_COPY_IN:%.+]] = VPU.Copy
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x1x320x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[MVN:%.+]] = VPU.MVN([[MVN_COPY_IN]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x1x320x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[MVN_COPY_OUT:%.+]] = VPU.Copy([[MVN]]
    //CHECK-SAME:       -> tensor<1x1x320x1xf16>

    //CHECK:        [[RESHAPE:%.+]] = VPU.AffineReshape([[MVN_COPY_OUT]])
    //CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 320, 1, 1]} : tensor<1x1x320x1xf16> -> tensor<1x320x1x1xf16>
    //CHECK:        [[CAST:%.+]] = VPU.PermuteCast([[RESHAPE]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x320x1x1xf16> -> tensor<1x320x1x1xf16, {order = #NHWC}>

    //CHECK:        [[DWCONV_INPUT_COPY_IN:%.+]] = VPU.Copy([[CAST]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[DWCONV_WEIGHTS_COPY_IN:%.+]] = VPU.Copy
    //CHECK-SAME:       -> !VPU.DistributedTensor<320x16x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    //CHECK:        [[DWCONV_WEIGHTS_TABLE_COPY_IN:%.+]] = VPU.Copy
    //CHECK-SAME:       -> !VPU.DistributedTensor<320x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    //CHECK:        [[DWCONV:%.+]] = VPU.NCE.DepthConvolution([[DWCONV_INPUT_COPY_IN]],
    //CHECK-SAME:                                           [[DWCONV_WEIGHTS_COPY_IN]],
    //CHECK-SAME:                                           [[DWCONV_WEIGHTS_TABLE_COPY_IN]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[DWCONV_COPY_OUT:%.+]] = VPU.Copy([[DWCONV]]
    //CHECK-SAME:       -> tensor<1x320x1x1xf16, {order = #NHWC}>

    //CHECK:        [[SIGMOID_COPY_IN:%.+]] = VPU.Copy([[DWCONV_COPY_OUT]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[SIGMOID:%.+]] = VPU.Sigmoid([[SIGMOID_COPY_IN]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[SIGMOID_COPY_OUT:%.+]] = VPU.Copy([[SIGMOID]]
    //CHECK-SAME:       -> tensor<1x320x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @InterpolateToDistributedOpDuplicatedBuffer
func.func @InterpolateToDistributedOpDuplicatedBuffer(%arg0: tensor<1x1x96x160xf16>) -> tensor<1x1x192x320xf16> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3], initial_input_dims_attr = [1, 1, 96, 160], initial_output_dims_attr = [1, 1, 192, 320], multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 320], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x1x96x160xf16> -> tensor<1x1x192x320xf16>
    return %0 : tensor<1x1x192x320xf16>

    //CHECK:  [[ClusterCopy:%.+]] = VPU.Copy
    //CHECK-SAME:    !VPU.DistributedTensor<1x1x96x160xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:  [[ClusterInterpolate:%.+]] = VPU.Interpolate([[ClusterCopy]])
    //CHECK-SAME:                -> !VPU.DistributedTensor<1x1x192x320xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:  [[OUT:%.+]] = VPU.Copy([[ClusterInterpolate]]
    //CHECK: return [[OUT]] : tensor<1x1x192x320xf16>
}


// -----

// CHECK-LABEL: @InterpolateToDistributedOpSegmentedBuffer

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @InterpolateToDistributedOpSegmentedBuffer(%arg0: tensor<1x1x96x160xf16>) -> tensor<1x1x192x320xf16> {

    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3], initial_input_dims_attr = [1, 1, 96, 160], initial_output_dims_attr = [1, 1, 192, 320], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 320], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x1x96x160xf16> -> tensor<1x1x192x320xf16>
    return %0 : tensor<1x1x192x320xf16>

    //CHECK:  [[ClusterCopy:%.+]] = VPU.Copy
    //CHECK-SAME{LITERAL}: !VPU.DistributedTensor<1x1x96x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]], memory_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]}>
    //CHECK:  [[ClusterInterpolate:%.+]] = VPU.Interpolate([[ClusterCopy]])
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x1x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:  [[OUT:%.+]] = VPU.Copy([[ClusterInterpolate]]
    //CHECK: return [[OUT]] : tensor<1x1x192x320xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AssignInputAlignmentToOutputForEltwise
func.func @AssignInputAlignmentToOutputForEltwise(%arg0: tensor<1x1024x14x14xf16, {order = #NHWC}>, %arg1: tensor<1x1024x14x14xf16, {order = #NHWC}>) -> tensor<1x1024x14x14xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1024x1x1x4xsi32> = dense<10> : tensor<1024x1x1x4xsi32>
    %cst_0 = const.Declare tensor<1024x1024x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x1024x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            rawFilterShape = [1024, 1024, 3, 3], strides = [1, 1]} : tensor<1x1024x14x14xf16, {order = #NHWC}>, tensor<1024x1024x3x3xf16, {order = #NHWC}>, tensor<1024x1x1x4xsi32> -> tensor<1x1024x14x14xf16, {order = #NHWC}>
    %1 = VPU.NCE.Eltwise(%arg0, %0) {multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>, ppe = #VPU.PPEStub<>, op_type = #VPU.eltwise_type<ADD>} -> tensor<1x1024x14x14xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%1, %cst_0, %cst) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            rawFilterShape = [1024, 1024, 3, 3], strides = [1, 1]} : tensor<1x1024x14x14xf16, {order = #NHWC}>, tensor<1024x1024x3x3xf16, {order = #NHWC}>, tensor<1024x1x1x4xsi32> -> tensor<1x1024x14x14xf16, {order = #NHWC}>
    return %2 : tensor<1x1024x14x14xf16, {order = #NHWC}>

    // The Eltwise's input and output alignment should be equal
    //CHECK:        [[WEIGHTSTABLE:%.+]] = const.Declare tensor<1024x1x1x4xsi32> = dense<10> : tensor<1024x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<1024x1024x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x1024x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[COPY_IN:%.+]] = VPU.Copy
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1024x14x14xf16, #NHWC, @CMX_NN, {
    //CHECK-SAME:       mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>
    //CHECK:        [[COPY_W:%.+]] = VPU.Copy
    //CHECK-SAME:   -> !VPU.DistributedTensor<1024x1024x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:        [[COPY_WT:%.+]] = VPU.Copy
    //CHECK-SAME:   -> !VPU.DistributedTensor<1024x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution([[COPY_IN]],
    //CHECK-SAME:       [[COPY_W]],
    //CHECK-SAME:       [[COPY_WT]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1024x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[COPY_TO_ELT_DDR:%.+]] = VPU.Copy([[CONV]]
    //CHECK-SAME:   -> tensor<1x1024x14x14xf16, {order = #NHWC}>
    //CHECK:        [[COPY_TO_ELT_CMX:%.+]] = VPU.Copy
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1024x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>
    //CHECK:        [[COPT_TO_ELT_IN:%.+]] = VPU.Copy([[COPY_TO_ELT_DDR]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1024x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>

    //CHECK:        [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[COPY_TO_ELT_CMX]],
    //CHECK-SAME:       [[COPT_TO_ELT_IN]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1024x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>

    //CHECK:        [[COPY_ELT_OUT:%.+]] = VPU.Copy([[ELTWISE]])
    //CHECK-SAME:   -> tensor<1x1024x14x14xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @InterpolateToDistributedOpOVERLAPPED

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @InterpolateToDistributedOpOVERLAPPED(%arg0: tensor<1x32x20x168xf16, {order = #NHWC}>) -> tensor<1x32x38x336xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SCALES>, coord_mode = <PYTORCH_HALF_PIXEL>,
                                nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0],
                                cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
                                initial_input_dims_attr = [1, 32, 95, 168], initial_input_offset_attr = [0, 0, 75, 0],
                                initial_output_dims_attr = [1, 32, 190, 336], initial_output_offset_attr = [0, 0, 152, 0],
                                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.900000e+00, 2.000000e+00], sizes_attr = [1, 32, 190, 336],
                                tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} :
                                tensor<1x32x20x168xf16, {order = #NHWC}> -> tensor<1x32x38x336xf16, {order = #NHWC}>
    return %0 : tensor<1x32x38x336xf16, {order = #NHWC}>

    //CHECK:               [[CLUSTERCOPYIN:%.+]] = VPU.Copy
    //CHECK-SAME{LITERAL}: -> !VPU.DistributedTensor<1x32x20x168xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 32, 11, 168], [1, 32, 10, 168]], compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0]], memory_shapes = [[1, 32, 11, 168], [1, 32, 10, 168]], memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0]]}>

    //CHECK:                [[INTERPOLATE:%.+]] = VPU.Interpolate([[CLUSTERCOPYIN]])
    //CHECK-SAME:                -> !VPU.DistributedTensor<1x32x38x336xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:                [[CLUSTERCOPYOUT:%.+]] = VPU.Copy([[INTERPOLATE]]
    //CHECK-SAME:                -> tensor<1x32x38x336xf16, {order = #NHWC}>

    //CHECK:                return [[CLUSTERCOPYOUT]] : tensor<1x32x38x336xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @ConvertToDistributedOpSOH
func.func @ConvertToDistributedOpSOH(%arg0: tensor<1x48x160x80xf32>) -> tensor<1x48x160x80xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
    return %0 : tensor<1x48x160x80xf16>
    // CHECK:   [[TILING1:%.+]] = VPU.Copy
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x48x160x80xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:   [[TILING2:%.+]] = VPU.Convert([[TILING1]])
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x48x160x80xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:   [[TILING3:%.+]] = VPU.Copy([[TILING2]]

    // CHECK:   return [[TILING3]] : tensor<1x48x160x80xf16>
}


// -----

// CHECK-LABEL:   func.func @ConvertToDistributedOpSOK
func.func @ConvertToDistributedOpSOK(%arg0: tensor<1x48x160x80xf32>) -> tensor<1x48x160x80xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
    return %0 : tensor<1x48x160x80xf16>
    // CHECK:   [[TILING1:%.+]] = VPU.Copy
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x48x160x80xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:   [[TILING2:%.+]] = VPU.Convert([[TILING1]])
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x48x160x80xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:   [[TILING3:%.+]] = VPU.Copy([[TILING2]]

    // CHECK:   return [[TILING3]] : tensor<1x48x160x80xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @ConvertToDistributedOpClustering
func.func @ConvertToDistributedOpClustering(%arg0: tensor<1x48x160x80xf32>) -> tensor<1x48x160x80xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
    return %0 : tensor<1x48x160x80xf16>
    // CHECK:   [[TILING1:%.+]] = VPU.Copy
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x48x160x80xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:   [[TILING2:%.+]] = VPU.Convert([[TILING1]])
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x48x160x80xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:   [[TILING3:%.+]] = VPU.Copy([[TILING2]]

    // CHECK:   return [[TILING3]] : tensor<1x48x160x80xf16>
}

// -----

// CHECK-LABEL: @InterpolateAlignCornersToDistributedOpOVERLAPPED
func.func @InterpolateAlignCornersToDistributedOpOVERLAPPED(%arg0: tensor<1x1x257x257xf16>) -> tensor<1x1x17x17xf16> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ALIGN_CORNERS>,
            nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
            axes_attr = [0, 1, 2, 3], initial_input_dims_attr = [1, 1, 257, 257], initial_output_dims_attr = [1, 1, 17, 17],
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            scales_attr = [1.0000100135803223, 1.0000100135803223, 0.066157855093479156, 0.066157855093479156],
            sizes_attr = [1, 1, 17, 17], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]}
            : tensor<1x1x257x257xf16> -> tensor<1x1x17x17xf16>
    return %0 : tensor<1x1x17x17xf16>

    //CHECK:        [[COPY_IN:%.+]] = VPU.Copy
    //CHECK-SAME{LITERAL}:  mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 129, 257], [1, 1, 113, 257]], compute_offsets = [[0, 0, 0, 0], [0, 0, 144, 0]], memory_shapes = [[1, 1, 129, 257], [1, 1, 113, 257]], memory_offsets = [[0, 0, 0, 0], [0, 0, 144, 0]]}>

    //CHECK:        [[INTERPOLATE:%.+]] = VPU.Interpolate([[COPY_IN]])
    //CHECK:                -> !VPU.DistributedTensor<1x1x17x17xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[COPY_OUT:%.+]] = VPU.Copy([[INTERPOLATE]]
    //CHECK:        return [[COPY_OUT]] : tensor<1x1x17x17xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CompressConvWrapping
func.func @CompressConvWrapping(%arg0: tensor<1x4x224x224xf16, {order = #NHWC}>, %arg1: tensor<64x1x1x160xf16, {order = #NHWC}>, %arg2: tensor<64x1x1x4xsi32>) -> tensor<1x64x112x112xf16, {order = #NHWC}>  {
    %compressConv = VPU.NCE.CompressConvolution(%arg0, %arg1, %arg2) {
                cm_sp_pattern = 15 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
                pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
                ppe = #VPU.PPEStub<>,
                rawFilterShape = [64, 4, 7, 7], strides = [2, 2]}
        -> tensor<1x64x112x112xf16, {order = #NHWC}>

    return %compressConv : tensor<1x64x112x112xf16, {order = #NHWC}>

    //CHECK:        [[IN_CMX:%.+]] = VPU.Copy
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x4x224x224xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7],
    //CHECK-SAME:   pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy
    // CHECK-SAME:      -> !VPU.DistributedTensor<64x1x1x160xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[WEIGHTSTABLE_CMX:%.+]] = VPU.Copy
    // CHECK-SAME:      -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[COMPRESSCONV:%.+]] = VPU.NCE.CompressConvolution(
    //CHECK-SAME:     [[IN_CMX]],
    //CHECK-SAME:     [[WEIGHTS_CMX]],
    //CHECK-SAME:     [[WEIGHTSTABLE_CMX]])
    //CHECK-SAME:     -> !VPU.DistributedTensor<1x64x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUT_CMX:%.+]] = VPU.Copy([[COMPRESSCONV]]
    //CHECK-SAME:     -> tensor<1x64x112x112xf16, {order = #NHWC}>

    //CHECK:        return [[OUT_CMX]] : tensor<1x64x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CompressConvToDistributedOpSOB
// CHECK-SAME:  [[ARG0:%.+]]: tensor<2x4x224x224xf16, {order = #NHWC}>
// CHECK-SAME:  [[ARG1:%.+]]: tensor<64x1x1x160xf16, {order = #NHWC}>
// CHECK-SAME:  [[ARG2:%.+]]: tensor<64x1x1x4xsi32>
func.func @CompressConvToDistributedOpSOB(%arg0: tensor<2x4x224x224xf16, {order = #NHWC}>, %arg1: tensor<64x1x1x160xf16, {order = #NHWC}>, %arg2: tensor<64x1x1x4xsi32>) -> tensor<2x64x112x112xf16, {order = #NHWC}>  {
    %compressConv = VPU.NCE.CompressConvolution(%arg0, %arg1, %arg2) {
                cm_sp_pattern = 15 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,
                pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
                ppe = #VPU.PPEStub<>,
                rawFilterShape = [64, 4, 7, 7], strides = [2, 2]}
        -> tensor<2x64x112x112xf16, {order = #NHWC}>

    return %compressConv : tensor<2x64x112x112xf16, {order = #NHWC}>

    // CHECK:   [[IN_CMX:%.+]] = VPU.Copy([[ARG0]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<2x4x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>

    // CHECK:   [[WEIGHTS_CMX:%.+]] = VPU.Copy([[ARG1]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<64x1x1x160xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:   [[WEIGHTSTABLE_CMX:%.+]] = VPU.Copy([[ARG2]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:   [[COMPRESSCONV:%.+]] = VPU.NCE.CompressConvolution(
    // CHECK-SAME:  [[IN_CMX]],
    // CHECK-SAME:  [[WEIGHTS_CMX]],
    // CHECK-SAME:  [[WEIGHTSTABLE_CMX]])
    // CHECK-SAME:  -> !VPU.DistributedTensor<2x64x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>

    // CHECK:   [[OUT_CMX:%.+]] = VPU.Copy([[COMPRESSCONV]]

    // CHECK:   return [[OUT_CMX]] : tensor<2x64x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TwoSiblingCompressConvWrapping
// CHECK-SAME:     [[ARG0:%.+]]: tensor<1x4x224x224xf16, {order = #NHWC}>
func.func @TwoSiblingCompressConvWrapping(
    %arg0: tensor<1x4x224x224xf16, {order = #NHWC}>, %arg1: tensor<64x1x1x160xf16, {order = #NHWC}>,
    %arg2: tensor<64x1x1x4xsi32>, %arg3: tensor<64x1x1x48xf16, {order = #NHWC}>)
        -> (tensor<1x64x224x224xf16, {order = #NHWC}>, tensor<1x64x224x224xf16, {order = #NHWC}>) {
    %compressConv0 = VPU.NCE.CompressConvolution(%arg0, %arg1, %arg2) {
                cm_sp_pattern = 15 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
                pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
                ppe = #VPU.PPEStub<>,
                rawFilterShape = [64, 4, 7, 7], strides = [1, 1]}
        -> tensor<1x64x224x224xf16, {order = #NHWC}>

    %compressConv1 = VPU.NCE.CompressConvolution(%arg0, %arg3, %arg2) {
                cm_sp_pattern = 15 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                ppe = #VPU.PPEStub<>,
                rawFilterShape = [64, 4, 3, 3], strides = [1, 1]}
        -> tensor<1x64x224x224xf16, {order = #NHWC}>

    return %compressConv0, %compressConv1 : tensor<1x64x224x224xf16, {order = #NHWC}>, tensor<1x64x224x224xf16, {order = #NHWC}>

    //CHECK:        [[IN_CMX0:%.+]] = VPU.Copy([[ARG0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x4x224x224xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7],
    //CHECK-SAME:            pads = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
    //CHECK-SAME:            strides = [1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[COMPRESS0:%.+]] = VPU.NCE.CompressConvolution([[IN_CMX0]]
    //CHECK-SAME:     -> !VPU.DistributedTensor<1x64x224x224xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[IN_CMX1:%.+]] = VPU.Copy([[ARG0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x4x224x224xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3],
    //CHECK-SAME:            pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:            strides = [1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[COMPRESS1:%.+]] = VPU.NCE.CompressConvolution([[IN_CMX1]]
    //CHECK-SAME:     -> !VPU.DistributedTensor<1x64x224x224xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ChainOpsMultipleConsumersToNCEClusteringSOHOverlappedImproperSplitForOutputShape
// Between the two conv siblings, even though one has the bigger kernel, the inferred output shapes per cluster
// don't fully satisfy H >= 1 for each tile.
func.func @ChainOpsMultipleConsumersToNCEClusteringSOHOverlappedImproperSplitForOutputShape(%arg0: tensor<1x128x8x8xf16, {order = #NHWC}>)
    -> (tensor<1x96x1x1xf16, {order = #NHWC}>, tensor<1x96x8x8xf16, {order = #NHWC}>) {
    %cst = const.Declare tensor<96x1x1x4xsi32> = dense<10> : tensor<96x1x1x4xsi32>
    %cst_0 = const.Declare tensor<96x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<96x96x8x8xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x8x8xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<96x96x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [96, 128, 1, 1], strides = [1, 1]} : tensor<1x128x8x8xf16, {order = #NHWC}>, tensor<96x128x1x1xf16, {order = #NHWC}>, tensor<96x1x1x4xsi32> -> tensor<1x96x8x8xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst_1, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [96, 96, 8, 8], strides = [8, 8]} : tensor<1x96x8x8xf16, {order = #NHWC}>, tensor<96x96x8x8xf16, {order = #NHWC}>, tensor<96x1x1x4xsi32> -> tensor<1x96x1x1xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%0, %cst_2, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [96, 96, 1, 1], strides = [1, 1]} : tensor<1x96x8x8xf16, {order = #NHWC}>, tensor<96x96x1x1xf16, {order = #NHWC}>, tensor<96x1x1x4xsi32> -> tensor<1x96x8x8xf16, {order = #NHWC}>
    return %1, %2 : tensor<1x96x1x1xf16, {order = #NHWC}>, tensor<1x96x8x8xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTSTABLE:%.+]] = const.Declare tensor<96x1x1x4xsi32> = dense<10> : tensor<96x1x1x4xsi32>
    //CHECK-DAG:    [[WEIGHTS_0:%.+]] = const.Declare tensor<96x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_1:%.+]] = const.Declare tensor<96x96x8x8xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x8x8xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_2:%.+]] = const.Declare tensor<96x96x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x1x1xf16>, [#const.Reorder<#NHWC>]

    // Conv producer

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x128x8x8xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[WEIGHTS_0_CMX:%.+]] = VPU.Copy([[WEIGHTS_0]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<96x128x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[WEIGHTSTABLE_CMX:%.+]] = VPU.Copy([[WEIGHTSTABLE]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[OUT_0_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX]],
    //CHECK-SAME:             [[WEIGHTS_0_CMX]],
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x96x8x8xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUT_0:%.+]] = VPU.Copy([[OUT_0_CMX]]

    // First conv comsumer
    //
    // There's no way to satisfy both siblings with a single overlap config, thus default to it's own overlap config
    //

    //CHECK:        [[OUT_0_COPYBACK:%.+]] = VPU.Copy([[OUT_0]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x96x8x8xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[WEIGHTS_1_CMX:%.+]] = VPU.Copy([[WEIGHTS_1]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<96x96x8x8xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    //CHECK:        [[WEIGHTSTABLE_1_CMX:%.+]] = VPU.Copy([[WEIGHTSTABLE]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    //CHECK:        [[OUT_1_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[OUT_0_COPYBACK]],
    //CHECK-SAME:             [[WEIGHTS_1_CMX]],
    //CHECK-SAME:             [[WEIGHTSTABLE_1_CMX]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x96x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[OUT_1:%.+]] = VPU.Copy([[OUT_1_CMX]]

    // Second conv comsumer

    //CHECK:        [[OUT_0_COPYBACK_1:%.+]] = VPU.Copy([[OUT_0]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x96x8x8xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[WEIGHTS_2_CMX:%.+]] = VPU.Copy([[WEIGHTS_2]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<96x96x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[WEIGHTSTABLE_2_CMX:%.+]] = VPU.Copy([[WEIGHTSTABLE]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[OUT_2_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[OUT_0_COPYBACK_1]],
    //CHECK-SAME:             [[WEIGHTS_2_CMX]],
    //CHECK-SAME:             [[WEIGHTSTABLE_2_CMX]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x96x8x8xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUT_2:%.+]] = VPU.Copy([[OUT_2_CMX]]

    //CHECK:        return [[OUT_1]], [[OUT_2]] : tensor<1x96x1x1xf16, {order = #NHWC}>, tensor<1x96x8x8xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MultiplySWSOHTileNotAtBroadcastAxis
func.func @MultiplySWSOHTileNotAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16, {order = #NHWC}>,
                %arg1: tensor<1x1x44x44xf16, {order = #NHWC}>) -> tensor<1x32x44x44xf16, {order = #NHWC}> {
    %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16, {order = #NHWC}>,
                tensor<1x1x44x44xf16, {order = #NHWC}> -> tensor<1x32x44x44xf16, {order = #NHWC}>

    return %0 : tensor<1x32x44x44xf16, {order = #NHWC}>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x44x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[MULTIPLY:%.+]] = VPU.Multiply([[INPUT0]], [[INPUT1]])
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x32x44x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[MULTIPLY]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MultiplySWSOHTileAtBroadcastAxis
func.func @MultiplySWSOHTileAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16, {order = #NHWC}>,
                %arg1: tensor<1x1x1x44xf16, {order = #NHWC}>) -> tensor<1x32x44x44xf16, {order = #NHWC}> {
    %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16, {order = #NHWC}>,
                tensor<1x1x1x44xf16, {order = #NHWC}> -> tensor<1x32x44x44xf16, {order = #NHWC}>

    return %0 : tensor<1x32x44x44xf16, {order = #NHWC}>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[MULTIPLY:%.+]] = VPU.Multiply([[INPUT0]], [[INPUT1]])
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x32x44x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[MULTIPLY]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MultiplySWSOKTileNotAtBroadcastAxis
func.func @MultiplySWSOKTileNotAtBroadcastAxis(%arg0: tensor<1x32x1x44xf16, {order = #NHWC}>,
                %arg1: tensor<1x32x1x1xf16, {order = #NHWC}>) -> tensor<1x32x1x44xf16, {order = #NHWC}> {
    %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} :
                tensor<1x32x1x44xf16, {order = #NHWC}>,
                tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x1x44xf16, {order = #NHWC}>

    return %0 : tensor<1x32x1x44xf16, {order = #NHWC}>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[MULTIPLY:%.+]] = VPU.Multiply([[INPUT0]], [[INPUT1]])
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x32x1x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[MULTIPLY]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x1x44xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MultiplySWSOKTileAtBroadcastAxis
func.func @MultiplySWSOKTileAtBroadcastAxis(%arg0: tensor<1x32x1x44xf16, {order = #NHWC}>,
                %arg1: tensor<1x1x1x44xf16, {order = #NHWC}>) -> tensor<1x32x1x44xf16, {order = #NHWC}> {
    %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} :
                tensor<1x32x1x44xf16, {order = #NHWC}>,
                tensor<1x1x1x44xf16, {order = #NHWC}> -> tensor<1x32x1x44xf16, {order = #NHWC}>

    return %0 : tensor<1x32x1x44xf16, {order = #NHWC}>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[MULTIPLY:%.+]] = VPU.Multiply([[INPUT0]], [[INPUT1]])
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x32x1x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[MULTIPLY]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x1x44xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MultiplySWSetAlignmentPerInputType
func.func @MultiplySWSetAlignmentPerInputType(%arg0: tensor<1x128x8x8xf16, {order = #NHWC}>, %arg1: tensor<1x1x1x128xf16>) -> tensor<1x32x1x128xf16> {
    %cst = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %cst_0 = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [64, 128, 1, 1], strides = [1, 1]} : tensor<1x128x8x8xf16, {order = #NHWC}>, tensor<64x128x1x1xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x8x8xf16, {order = #NHWC}>

    %1 = VPU.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x64xf16>
    %2 = VPU.Reshape(%1) {shape_value = [1, 32, 1, 128]} : tensor<1x8x8x64xf16> -> tensor<1x32x1x128xf16>

    %3 = VPU.Multiply(%2, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x128xf16>, tensor<1x1x1x128xf16> -> tensor<1x32x1x128xf16>

    return %3 : tensor<1x32x1x128xf16>

    // CHECK:        [[RESHAPE:%.+]] = VPU.Reshape
    // CHECK-SAME:                    {shape_value = [1, 32, 1, 128]} : tensor<1x8x8x64xf16> -> tensor<1x32x1x128xf16>

    // CHECK:        [[MULTIPLY_INPUT1:%.+]] = VPU.Copy([[RESHAPE:%.+]])
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK:        [[MULTIPLY_INPUT2:%.+]] = VPU.Copy
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x1x1x128xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[MULTIPLY:%.+]] = VPU.Multiply([[MULTIPLY_INPUT1]], [[MULTIPLY_INPUT2]])
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[MULTIPLY]]

    // CHECK:        return [[OUTPUT]] : tensor<1x32x1x128xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEInterpolateToDistributedOpClustering
func.func @NCEInterpolateToDistributedOpClustering(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x2x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x16x2x2xi1> = dense<1> : tensor<1x16x2x2xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = f16, seDepth = 1, seSize = [16], dataShape = [1, 16, 1, 1],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>
    } -> tensor<1x1x2x2xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 16, 2, 2]>
    } -> !VPU.SparseTensor<data=tensor<1x16x1x1xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x16x2x2xi1>,
                           storage_element_table=tensor<1x1x2x2xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [16, 16, 1, 1],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x16x2x2xf16, {order = #NHWC}>

    return %interpolate : tensor<1x16x2x2xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[WEIGHTSTABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // CHECK-DAG:    [[INPUT_SM:%.+]] = const.Declare tensor<1x16x2x2xi1> = dense<true> : tensor<1x16x2x2xi1>
    // CHECK:        [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 1, 1],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>,
    // CHECK-SAME:       seDepth = 1 : i64, seSize = [16]}
    // CHECK-SAME:       -> tensor<1x1x2x2xi32, {order = #NHWC}>
    // CHECK:        [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])
    // CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT_SPARSE]]
    // CHECK-SAME:           -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
    // CHECK-SAME:                                sparsity_map=!VPU.DistributedTensor<1x16x2x2xi1, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
    // CHECK-SAME:                                storage_element_table=!VPU.DistributedTensor<1x1x2x2xi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
    // CHECK-SAME:                                #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>>

    // CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    // CHECK-SAME:   -> !VPU.DistributedTensor<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[WEIGHTSTABLE_CMX:%.+]] = VPU.Copy([[WEIGHTSTABLE]]
    // CHECK-SAME:   -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Interpolate(
    // CHECK-SAME:             [[INPUT_CMX]],
    // CHECK-SAME:             [[WEIGHTS_CMX]],
    // CHECK-SAME:             [[WEIGHTSTABLE_CMX]])
    // CHECK-SAME:   -> !VPU.DistributedTensor<1x16x2x2xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    // CHECK:        return [[OUT]] : tensor<1x16x2x2xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEInterpolateToDistributedOpSOH
func.func @NCEInterpolateToDistributedOpSOH(%arg0: tensor<1x64x5x10xf16, {order = #NHWC}>) -> tensor<1x64x10x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x64x10x20xi1> = dense<1> : tensor<1x64x10x20xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = f16, seDepth = 1, seSize = [64], dataShape = [1, 64, 5, 10],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>
    } -> tensor<1x1x10x20xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 10, 20]>
    } -> !VPU.SparseTensor<data=tensor<1x64x5x10xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x64x10x20xi1>,
                           storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [64, 64, 1, 1],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x64x10x20xf16, {order = #NHWC}>

    return %interpolate : tensor<1x64x10x20xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[WEIGHTSTABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    // CHECK-DAG:    [[INPUT_SM:%.+]] = const.Declare tensor<1x64x10x20xi1> = dense<true> : tensor<1x64x10x20xi1>
    // CHECK:        [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 64, 5, 10],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>,
    // CHECK-SAME:       seDepth = 1 : i64, seSize = [64]}
    // CHECK-SAME:       -> tensor<1x1x10x20xi32, {order = #NHWC}>

    // CHECK:        [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])
    // CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT_SPARSE]]
    // CHECK-SAME:           -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x64x5x10xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                sparsity_map=!VPU.DistributedTensor<1x64x10x20xi1, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>,
    // CHECK-SAME:                                storage_element_table=!VPU.DistributedTensor<1x1x10x20xi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>,
    // CHECK-SAME:                                #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>>

    // CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    // CHECK-SAME:   -> !VPU.DistributedTensor<64x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[WEIGHTSTABLE_CMX:%.+]] = VPU.Copy([[WEIGHTSTABLE]]
    // CHECK-SAME:   -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Interpolate(
    // CHECK-SAME:             [[INPUT_CMX]],
    // CHECK-SAME:             [[WEIGHTS_CMX]],
    // CHECK-SAME:             [[WEIGHTSTABLE_CMX]])
    // CHECK-SAME:   -> !VPU.DistributedTensor<1x64x10x20xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    // CHECK:        return [[OUT]] : tensor<1x64x10x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEInterpolateToDistributedOpSOK
func.func @NCEInterpolateToDistributedOpSOK(%arg0: tensor<1x64x5x10xf16, {order = #NHWC}>) -> tensor<1x64x10x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x64x10x20xi1> = dense<1> : tensor<1x64x10x20xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = f16, seDepth = 1, seSize = [64], dataShape = [1, 64, 5, 10],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>
    } -> tensor<1x1x10x20xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 10, 20]>
    } -> !VPU.SparseTensor<data=tensor<1x64x5x10xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x64x10x20xi1>,
                           storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [64, 64, 1, 1],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x64x10x20xf16, {order = #NHWC}>

    return %interpolate : tensor<1x64x10x20xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[WEIGHTSTABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    // CHECK-DAG:    [[INPUT_SM:%.+]] = const.Declare tensor<1x64x10x20xi1> = dense<true> : tensor<1x64x10x20xi1>
    // CHECK:        [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 64, 5, 10],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>,
    // CHECK-SAME:       seDepth = 1 : i64, seSize = [64]}
    // CHECK-SAME:       -> tensor<1x1x10x20xi32, {order = #NHWC}>
    // CHECK:        [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])
    // CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT_SPARSE]]
    // CHECK-SAME:           -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x64x5x10xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
    // CHECK-SAME:                                sparsity_map=!VPU.DistributedTensor<1x64x10x20xi1, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
    // CHECK-SAME:                                storage_element_table=!VPU.DistributedTensor<1x1x10x20xi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
    // CHECK-SAME:                                #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>>

    // CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    // CHECK-SAME:   -> !VPU.DistributedTensor<64x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // CHECK:        [[WEIGHTSTABLE_CMX:%.+]] = VPU.Copy([[WEIGHTSTABLE]]
    // CHECK-SAME:   -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Interpolate(
    // CHECK-SAME:             [[INPUT_CMX]],
    // CHECK-SAME:             [[WEIGHTS_CMX]],
    // CHECK-SAME:             [[WEIGHTSTABLE_CMX]])
    // CHECK-SAME:   -> !VPU.DistributedTensor<1x64x10x20xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    // CHECK:        return [[OUT]] : tensor<1x64x10x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddToDistributedOpSOHDuplicatedIn
func.func @EltwiseAddToDistributedOpSOHDuplicatedIn(%arg0: tensor<1x3x52x104xf16, {order = #NHWC}>) -> tensor<1x16x39x26xf16, {order = #NHWC}> {
    %0 = VPU.ShapeCast {shape = [1, 16, 39, 26]} inputs(%arg0 : tensor<1x3x52x104xf16, {order = #NHWC}>) -> tensor<1x16x39x26xf16, {order = #NHWC}>
    %1 = VPU.NCE.Eltwise(%0, %0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, op_type = #VPU.eltwise_type<ADD> } -> tensor<1x16x39x26xf16, {order = #NHWC}>


    return %1: tensor<1x16x39x26xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_SHAPE_CAST:%.+]] = VPU.ShapeCast {shape = [1, 16, 39, 26]} inputs(%arg0 : tensor<1x3x52x104xf16, {order = #NHWC}>) -> tensor<1x16x39x26xf16, {order = #NHWC}>
    //CHECK:        [[INPUT0_CMX:%.+]] = VPU.Copy([[INPUT_SHAPE_CAST]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x39x26xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Eltwise([[INPUT0_CMX]], [[INPUT0_CMX]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x39x26xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x16x39x26xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TopKSWTilingSOH
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x31x103x513xf16, {order = #NHWC}>)
func.func @TopKSWTilingSOH(%arg0: tensor<1x31x103x513xf16, {order = #NHWC}>) -> tensor<1x1x103x513xsi32, {order = #NHWC}> {
    %aux = const.Declare tensor<1x1x1x496xui8> = dense<0> : tensor<1x1x1x496xui8>
    %output_values, %target_shape = VPU.TopK(%arg0, %aux) {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<1x31x103x513xf16, {order = #NHWC}>, tensor<1x1x1x496xui8> -> tensor<1x1x103x513xf16, {order = #NHWC}>, tensor<1x1x103x513xsi32, {order = #NHWC}>

    return %target_shape : tensor<1x1x103x513xsi32, {order = #NHWC}>

    //CHECK:        [[AUX:%.+]] = const.Declare tensor<1x1x1x496xui8> = dense<0> : tensor<1x1x1x496xui8>

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x31x103x513xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:        [[AUX_CMX:%.+]] = VPU.Copy([[AUX]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1x1x496xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]], [[TARGET:%.+]] = VPU.TopK([[INPUT_CMX]], [[AUX_CMX]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1x103x513xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    //CHECK-SAME:   !VPU.DistributedTensor<1x1x103x513xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT_VALUES:%.+]] = VPU.Copy([[OUTPUT]]
    //CHECK-SAME:   -> tensor<1x1x103x513xf16, {order = #NHWC}>

    //CHECK:        [[TARGET_SHAPE:%.+]] = VPU.Copy([[TARGET]]
    //CHECK-SAME:   -> tensor<1x1x103x513xsi32, {order = #NHWC}>

    //CHECK:        return [[TARGET_SHAPE]] : tensor<1x1x103x513xsi32, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TopKSWTilingSOK
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x103x513x31xf16, {order = #NHWC}>)
func.func @TopKSWTilingSOK(%arg0: tensor<1x103x513x31xf16, {order = #NHWC}>) -> tensor<1x103x513x1xsi32, {order = #NHWC}> {
    %aux = const.Declare tensor<1x1x1x496xui8> = dense<0> : tensor<1x1x1x496xui8>
    %output_values, %target_shape = VPU.TopK(%arg0, %aux) {axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<1x103x513x31xf16, {order = #NHWC}>, tensor<1x1x1x496xui8> -> tensor<1x103x513x1xf16, {order = #NHWC}>, tensor<1x103x513x1xsi32, {order = #NHWC}>

    return %target_shape : tensor<1x103x513x1xsi32, {order = #NHWC}>

    //CHECK:        [[AUX:%.+]] = const.Declare tensor<1x1x1x496xui8> = dense<0> : tensor<1x1x1x496xui8>

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x103x513x31xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    //CHECK:        [[AUX_CMX:%.+]] = VPU.Copy([[AUX]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1x1x496xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]], [[TARGET:%.+]] = VPU.TopK([[INPUT_CMX]], [[AUX_CMX]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x103x513x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>,
    //CHECK-SAME:   !VPU.DistributedTensor<1x103x513x1xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT_VALUES:%.+]] = VPU.Copy([[OUTPUT]]
    //CHECK-SAME:    -> tensor<1x103x513x1xf16, {order = #NHWC}>

    //CHECK:        [[TARGET_SHAPE:%.+]] = VPU.Copy([[TARGET]]
    //CHECK-SAME:    -> tensor<1x103x513x1xsi32, {order = #NHWC}>

    //CHECK:        return [[TARGET_SHAPE]] : tensor<1x103x513x1xsi32, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SliceConvConcatGeluSOK

func.func @SliceConvConcatGeluSOK(%arg0: tensor<1x80x1x3008xf16, {order = #NHWC}>) -> tensor<1x512x1x3000xf16, {order = #NHWC}> {
    %weights_0 = const.Declare tensor<256x80x1x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x80x1x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table_0 = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    %weights_1 = const.Declare tensor<256x80x1x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x80x1x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table_1 = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>

    %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 80, 1, 3000] : tensor<1x80x1x3008xf16, {order = #NHWC}> to tensor<1x80x1x3000xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %weights_0, %weights_table_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [256, 80, 1, 3], strides = [1, 1]} : tensor<1x80x1x3000xf16, {order = #NHWC}>, tensor<256x80x1x3xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x1x3000xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%0, %weights_1, %weights_table_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [256, 80, 1, 3], strides = [1, 1]} : tensor<1x80x1x3000xf16, {order = #NHWC}>, tensor<256x80x1x3xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x1x3000xf16, {order = #NHWC}>

    %3 = VPU.Concat(%1, %2) {static_offsets = [[0, 0, 0, 0], [0, 256, 0, 0]]} : tensor<1x256x1x3000xf16, {order = #NHWC}>, tensor<1x256x1x3000xf16, {order = #NHWC}> -> tensor<1x512x1x3000xf16, {order = #NHWC}>

    %4 = VPU.Slice %3 [0, 0, 0, 0] [1, 512, 1, 1500] : tensor<1x512x1x3000xf16, {order = #NHWC}> to tensor<1x512x1x1500xf16, {order = #NHWC}>
    %5 = VPU.Gelu(%4) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x512x1x1500xf16, {order = #NHWC}> -> tensor<1x512x1x1500xf16, {order = #NHWC}>

    %6 = VPU.Slice %3 [0, 0, 0, 1500] [1, 512, 1, 1500] : tensor<1x512x1x3000xf16, {order = #NHWC}> to tensor<1x512x1x1500xf16, {order = #NHWC}>
    %7 = VPU.Gelu(%6) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x512x1x1500xf16, {order = #NHWC}> -> tensor<1x512x1x1500xf16, {order = #NHWC}>

    %8 = VPU.Concat(%5, %7) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1500]]} : tensor<1x512x1x1500xf16, {order = #NHWC}>, tensor<1x512x1x1500xf16, {order = #NHWC}> -> tensor<1x512x1x3000xf16, {order = #NHWC}>
    return %8 : tensor<1x512x1x3000xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS_0:%.+]] = const.Declare tensor<256x80x1x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x80x1x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTSTABLE_0:%.+]] = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    // CHECK-DAG:   [[WEIGHTS_1:%.+]] = const.Declare tensor<256x80x1x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x80x1x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTSTABLE_1:%.+]] = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>

    // CHECK: [[CONV_INPUT:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 80, 1, 3000] : tensor<1x80x1x3008xf16, {order = #NHWC}> to tensor<1x80x1x3000xf16, {order = #NHWC}>
    // CHECK: [[CONV0_INPUT_CMX:%.+]] = VPU.Copy([[CONV_INPUT]]
    // CHECK-SAME: -> !VPU.DistributedTensor<1x80x1x3000xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK: [[CONV0_WEIGHT_CMX:%.+]] = VPU.Copy([[WEIGHTS_0]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<256x80x1x3xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // CHECK: [[CONV0_WEIGHTTABLE_CMX:%.+]] = VPU.Copy([[WEIGHTSTABLE_0]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<256x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution([[CONV0_INPUT_CMX]], [[CONV0_WEIGHT_CMX]], [[CONV0_WEIGHTTABLE_CMX]])
    // CHECK-SAME: -> !VPU.DistributedTensor<1x256x1x3000xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK: [[CONV0_OUTPUT:%.+]] = VPU.Copy([[CONV0]]

    // CHECK: [[CONV1_INPUT_CMX:%.+]] = VPU.Copy([[CONV_INPUT]]
    // CHECK: [[CONV1_WEIGHT_CMX:%.+]] = VPU.Copy([[WEIGHTS_1]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<256x80x1x3xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // CHECK: [[CONV1_WEIGHTTABLE_CMX:%.+]] = VPU.Copy([[WEIGHTSTABLE_1]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<256x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution([[CONV1_INPUT_CMX]], [[CONV1_WEIGHT_CMX]], [[CONV1_WEIGHTTABLE_CMX]])
    // CHECK-SAME: -> !VPU.DistributedTensor<1x256x1x3000xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK: [[CONV1_OUTPUT:%.+]] = VPU.Copy([[CONV1]]

    // CHECK: [[CONV_CONCAT:%.+]] = VPU.Concat([[CONV0_OUTPUT]], [[CONV1_OUTPUT]]) {static_offsets = [
    // CHECK-SAME:    [0, 0, 0, 0], [0, 256, 0, 0]
    // CHECK-SAME:  ]} :
    // CHECK-SAME: tensor<1x256x1x3000xf16, {order = #NHWC}>,
    // CHECK-SAME: tensor<1x256x1x3000xf16, {order = #NHWC}> -> tensor<1x512x1x3000xf16, {order = #NHWC}>

    // CHECK: [[GELU_0_SLICE:%.+]] = VPU.Slice [[CONV_CONCAT]] [0, 0, 0, 0] [1, 512, 1, 1500] :
    // CHECK-SAME:                  tensor<1x512x1x3000xf16, {order = #NHWC}> to tensor<1x512x1x1500xf16, {order = #NHWC}>

    // CHECK: [[GELU_0_INPUT:%.+]] = VPU.Copy([[GELU_0_SLICE]]
    // CHECK-SAME:   -> !VPU.DistributedTensor<1x512x1x1500xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK: [[GELU_0:%.+]] = VPU.Gelu([[GELU_0_INPUT]])
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x512x1x1500xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK: [[GELU_0_OUTPUT:%.+]] = VPU.Copy([[GELU_0]]

    // CHECK: [[GELU_1_SLICE:%.+]] = VPU.Slice [[CONV_CONCAT]] [0, 0, 0, 1500] [1, 512, 1, 1500] :
    // CHECK-SAME:                  tensor<1x512x1x3000xf16, {order = #NHWC}> to tensor<1x512x1x1500xf16, {order = #NHWC}>
    // CHECK: [[GELU_1_INPUT:%.+]] = VPU.Copy([[GELU_1_SLICE]]
    // CHECK-SAME:   -> !VPU.DistributedTensor<1x512x1x1500xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK: [[GELU_1:%.+]] = VPU.Gelu([[GELU_1_INPUT]])
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x512x1x1500xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK: [[GELU_1_OUTPUT:%.+]] = VPU.Copy([[GELU_1]]

    // CHECK: [[GELU_CONCAT:%.+]] = VPU.Concat([[GELU_0_OUTPUT]], [[GELU_1_OUTPUT]]) {static_offsets = [
    // CHECK-SAME:     [0, 0, 0, 0], [0, 0, 0, 1500]
    // CHECK:  ]} : tensor<1x512x1x1500xf16, {order = #NHWC}>, tensor<1x512x1x1500xf16, {order = #NHWC}> -> tensor<1x512x1x3000xf16, {order = #NHWC}>

    // CHECK: return [[GELU_CONCAT]] : tensor<1x512x1x3000xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthToSpaceToDistributedOpSOH
func.func @DepthToSpaceToDistributedOpSOH(%arg0: tensor<1x128x12x270xf16, {order = #NHWC}>) -> tensor<1x8x48x1080xf16, {order = #NHWC}> {
    %0 = VPU.DepthToSpace(%arg0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x128x12x270xf16, {order = #NHWC}> -> tensor<1x8x48x1080xf16, {order = #NHWC}>

    return %0 : tensor<1x8x48x1080xf16, {order = #NHWC}>

    //CHECK:        [[INPUT:%.+]] = VPU.Copy
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[CLUSTER_TILING_D2S:%.+]] = VPU.DepthToSpace([[INPUT]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[CLUSTER_TILING_D2S]]
    //CHECK-SAME:   -> tensor<1x8x48x1080xf16, {order = #NHWC}>

    //CHECK:        return [[OUTPUT]] : tensor<1x8x48x1080xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthToSpaceToDistributedOpSOW
func.func @DepthToSpaceToDistributedOpSOW(%arg0: tensor<1x128x1x270xf16, {order = #NHWC}>) -> tensor<1x8x4x1080xf16, {order = #NHWC}> {
    %0 = VPU.DepthToSpace(%arg0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>} : tensor<1x128x1x270xf16, {order = #NHWC}> -> tensor<1x8x4x1080xf16, {order = #NHWC}>

    return %0 : tensor<1x8x4x1080xf16, {order = #NHWC}>

    //CHECK:                 [[INPUT:%.+]] = VPU.Copy
    //CHECK-SAME:            -> !VPU.DistributedTensor<1x128x1x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>

    //CHECK:                 [[CLUSTER_TILING_D2S:%.+]] = VPU.DepthToSpace([[INPUT]])
    //CHECK-SAME{LITERAL}:   -> !VPU.DistributedTensor<1x8x4x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64, alignment = [1, 1, 1, 4]}>

    //CHECK:                 [[OUTPUT:%.+]] = VPU.Copy([[CLUSTER_TILING_D2S]]
    //CHECK-SAME:            -> tensor<1x8x4x1080xf16, {order = #NHWC}>

    //CHECK:                 return [[OUTPUT]] : tensor<1x8x4x1080xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

func.func @NCEPermute3x224x224(%arg0: tensor<1x3x256x224xf16>) -> tensor<1x4x256x224x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x4x256x224x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x4x256x224x!qElemType, {order = #NHWC}>

    // CHECK:       [[COPY_INPUT:%.+]] = VPU.Copy
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x3x256x224xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[COPY_INPUT]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x4x256x224x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[COPY_OUTPUT:%.+]] = VPU.Copy
    // CHECK-SAME:      -> tensor<1x4x256x224x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

func.func @NCEPermute3x224x224(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x16x112x112xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<16x1x1x48x!qElemType, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x1x1x48xf16>, [
            #const.CastElemType<ui8>,
            #const.CastElemType<!qElemType>,
            #const.Reorder<#NHWC>
        ]

    %WEIGHT_TABLE = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    %1 = VPU.NCE.CompressConvolution(%0, %WEIGHTS, %WEIGHT_TABLE) {
        cm_sp_pattern = 7 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [16, 4, 3, 3],
        strides = [2, 2]
    } -> tensor<1x16x112x112xf16, {order = #NHWC}>

    return %1 : tensor<1x16x112x112xf16, {order = #NHWC}>

    // CHECK:       [[COPY_INPUT:%.+]] = VPU.Copy
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x3x224x224xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:      {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3],
    // CHECK-SAME:      pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:      strides = [2, 2], num_clusters = 2 : i64}>

    // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[COPY_INPUT]])
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x4x224x224x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3],
    // CHECK-SAME:      pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:      strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>

    // CHECK:      -> !VPU.DistributedTensor<1x4x224x224x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3],
    // CHECK-SAME:      pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:      strides = [2, 2], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL:   @NCEPermuteWith2CompressConvConsumers
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x3x224x224xf16>
func.func @NCEPermuteWith2CompressConvConsumers(%arg0: tensor<1x3x224x224xf16>)
        -> (tensor<1x16x224x224xf16, {order = #NHWC}>, tensor<1x16x224x224xf16, {order = #NHWC}>) {
    %WEIGHTS0 = const.Declare tensor<16x1x1x48x!qElemType, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x1x1x48xf16>, [
            #const.CastElemType<ui8>,
            #const.CastElemType<!qElemType>,
            #const.Reorder<#NHWC>
        ]

    %WEIGHTS1 = const.Declare tensor<16x1x1x160x!qElemType, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x1x1x160xf16>, [
            #const.CastElemType<ui8>,
            #const.CastElemType<!qElemType>,
            #const.Reorder<#NHWC>
        ]

    %WEIGHT_TABLE = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    %1 = VPU.NCE.CompressConvolution(%0, %WEIGHTS0, %WEIGHT_TABLE) {
        cm_sp_pattern = 7 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [16, 4, 3, 3],
        strides = [1, 1]
    } -> tensor<1x16x224x224xf16, {order = #NHWC}>

    %2 = VPU.NCE.CompressConvolution(%0, %WEIGHTS1, %WEIGHT_TABLE) {
        cm_sp_pattern = 7 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [16, 4, 7, 7],
        strides = [1, 1]
    } -> tensor<1x16x224x224xf16, {order = #NHWC}>

    return %1, %2 : tensor<1x16x224x224xf16, {order = #NHWC}>, tensor<1x16x224x224xf16, {order = #NHWC}>

    // CHECK:       [[COPY_INPUT:%.+]] = VPU.Copy([[INPUT]])
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x3x224x224xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[COPY_INPUT]])
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x4x224x224x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:      -> !VPU.DistributedTensor<1x4x224x224x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3],
    // CHECK-SAME:      pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:      strides = [1, 1], num_clusters = 2 : i64}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DivideSWSOHTileNotAtBroadcastAxis
func.func @DivideSWSOHTileNotAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16>,
                %arg1: tensor<1x1x44x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Divide(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16>,
                tensor<1x1x44x44xf16> -> tensor<1x32x44x44xf16>

    return %0 : tensor<1x32x44x44xf16>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[DIVIDE:%.+]] = VPU.Divide([[INPUT0]], [[INPUT1]])
    //CHECK-SAME:                   -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[DIVIDE]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DivideSWSOHTileAtBroadcastAxis
func.func @DivideSWSOHTileAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16>,
                %arg1: tensor<1x1x1x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Divide(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16>,
                tensor<1x1x1x44xf16> -> tensor<1x32x44x44xf16>

    return %0 : tensor<1x32x44x44xf16>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy(%arg0
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy(%arg1
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[DIVIDE:%.+]] = VPU.Divide([[INPUT0]], [[INPUT1]])
    //CHECK-SAME:                    -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[DIVIDE]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PowerSWSOKTileNotAtBroadcastAxis
func.func @PowerSWSOKTileNotAtBroadcastAxis(%arg0: tensor<1x32x1x44xf16, {order = #NHWC}>,
                %arg1: tensor<1x32x1x1xf16, {order = #NHWC}>) -> tensor<1x32x1x44xf16, {order = #NHWC}> {
    %0 = VPU.Power(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} :
                tensor<1x32x1x44xf16, {order = #NHWC}>,
                tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x1x44xf16, {order = #NHWC}>

    return %0 : tensor<1x32x1x44xf16, {order = #NHWC}>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[POWER:%.+]] = VPU.Power([[INPUT0]], [[INPUT1]])
    //CHECK-SAME:                    -> !VPU.DistributedTensor<1x32x1x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[POWER]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x1x44xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PowerSWSOKTileAtBroadcastAxis
func.func @PowerSWSOKTileAtBroadcastAxis(%arg0: tensor<1x32x1x44xf16, {order = #NHWC}>,
                %arg1: tensor<1x1x1x44xf16, {order = #NHWC}>) -> tensor<1x32x1x44xf16, {order = #NHWC}> {
    %0 = VPU.Power(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} :
                tensor<1x32x1x44xf16, {order = #NHWC}>,
                tensor<1x1x1x44xf16, {order = #NHWC}> -> tensor<1x32x1x44xf16, {order = #NHWC}>

    return %0 : tensor<1x32x1x44xf16, {order = #NHWC}>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[POWER:%.+]] = VPU.Power([[INPUT0]], [[INPUT1]])
    //CHECK-SAME:                -> !VPU.DistributedTensor<1x32x1x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[POWER]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x1x44xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @GreaterSWSOHTileNotAtBroadcastAxis
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x32x44x44xf16>, [[INPUT_1:%.+]]: tensor<1x1x44x44xf16>
func.func @GreaterSWSOHTileNotAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16>,
                %arg1: tensor<1x1x44x44xf16>) -> tensor<1x32x44x44xi8> {
    %0 = VPU.Greater(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16>,
                tensor<1x1x44x44xf16> -> tensor<1x32x44x44xi8>

    return %0 : tensor<1x32x44x44xi8>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy([[INPUT_0]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy([[INPUT_1]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[GREATER:%.+]] = VPU.Greater([[INPUT0]], [[INPUT1]])
    //CHECK-SAME:            -> !VPU.DistributedTensor<1x32x44x44xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[GREATER]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xi8>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @GreaterSWSOHTileAtBroadcastAxis
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x32x44x44xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x44xf16>
func.func @GreaterSWSOHTileAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16>,
                %arg1: tensor<1x1x1x44xf16>) -> tensor<1x32x44x44xi8> {
    %0 = VPU.Greater(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16>,
                tensor<1x1x1x44xf16> -> tensor<1x32x44x44xi8>

    return %0 : tensor<1x32x44x44xi8>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy([[INPUT_0]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy([[INPUT_1]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[GREATER:%.+]] = VPU.Greater([[INPUT0]], [[INPUT1]])
    //CHECK-SAME:            -> !VPU.DistributedTensor<1x32x44x44xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[GREATER]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xi8>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @LessSWSOHTileNotAtBroadcastAxis
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x32x44x44xf16>, [[INPUT_1:%.+]]: tensor<1x1x44x44xf16>
func.func @LessSWSOHTileNotAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16>,
                %arg1: tensor<1x1x44x44xf16>) -> tensor<1x32x44x44xi8> {
    %0 = VPU.Less(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16>,
                tensor<1x1x44x44xf16> -> tensor<1x32x44x44xi8>

    return %0 : tensor<1x32x44x44xi8>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy([[INPUT_0]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy([[INPUT_1]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[LESS:%.+]] = VPU.Less([[INPUT0]], [[INPUT1]])
    //CHECK-SAME:                -> !VPU.DistributedTensor<1x32x44x44xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[LESS]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xi8>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @LessSWSOHTileAtBroadcastAxis
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x32x44x44xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x44xf16>
func.func @LessSWSOHTileAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16>,
                %arg1: tensor<1x1x1x44xf16>) -> tensor<1x32x44x44xi8> {
    %0 = VPU.Less(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16>,
                tensor<1x1x1x44xf16> -> tensor<1x32x44x44xi8>

    return %0 : tensor<1x32x44x44xi8>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy([[INPUT_0]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy([[INPUT_1]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[LESS:%.+]] = VPU.Less([[INPUT0]], [[INPUT1]])
    //CHECK-SAME:            -> !VPU.DistributedTensor<1x32x44x44xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[LESS]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xi8>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @EqualSWSOHTileNotAtBroadcastAxis
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x32x44x44xf16>, [[INPUT_1:%.+]]: tensor<1x1x44x44xf16>
func.func @EqualSWSOHTileNotAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16>,
                %arg1: tensor<1x1x44x44xf16>) -> tensor<1x32x44x44xi8> {
    %0 = VPU.Equal(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16>,
                tensor<1x1x44x44xf16> -> tensor<1x32x44x44xi8>

    return %0 : tensor<1x32x44x44xi8>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy([[INPUT_0]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy([[INPUT_1]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[EQUAL:%.+]] = VPU.Equal([[INPUT0]], [[INPUT1]])
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x32x44x44xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[EQUAL]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xi8>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @EqualSWSOHTileAtBroadcastAxis
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x32x44x44xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x44xf16>
func.func @EqualSWSOHTileAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16>,
                %arg1: tensor<1x1x1x44xf16>) -> tensor<1x32x44x44xi8> {
    %0 = VPU.Equal(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16>,
                tensor<1x1x1x44xf16> -> tensor<1x32x44x44xi8>

    return %0 : tensor<1x32x44x44xi8>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy([[INPUT_0]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy([[INPUT_1]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[EQUAL:%.+]] = VPU.Equal([[INPUT0]], [[INPUT1]])
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x32x44x44xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[EQUAL]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xi8>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @SubtractSWSOHTileNotAtBroadcastAxis
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x32x44x44xf16>, [[INPUT_1:%.+]]: tensor<1x1x44x44xf16>
func.func @SubtractSWSOHTileNotAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16>,
                %arg1: tensor<1x1x44x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16>,
                tensor<1x1x44x44xf16> -> tensor<1x32x44x44xf16>

    return %0 : tensor<1x32x44x44xf16>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy([[INPUT_0]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy([[INPUT_1]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[SUBTRACT:%.+]] = VPU.Subtract([[INPUT0]], [[INPUT1]])
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[SUBTRACT]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @SubtractSWSOHTileAtBroadcastAxis
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x32x44x44xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x44xf16>
func.func @SubtractSWSOHTileAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16>,
                %arg1: tensor<1x1x1x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16>,
                tensor<1x1x1x44xf16> -> tensor<1x32x44x44xf16>

    return %0 : tensor<1x32x44x44xf16>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy([[INPUT_0]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy([[INPUT_1]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[SUBTRACT:%.+]] = VPU.Subtract([[INPUT0]], [[INPUT1]])
    //CHECK-SAME:         -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[SUBTRACT]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @AddSWSOHTileNotAtBroadcastAxis
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x32x44x44xf16>, [[INPUT_1:%.+]]: tensor<1x1x44x44xf16>
func.func @AddSWSOHTileNotAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16>,
                %arg1: tensor<1x1x44x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16>,
                tensor<1x1x44x44xf16> -> tensor<1x32x44x44xf16>

    return %0 : tensor<1x32x44x44xf16>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy([[INPUT_0]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy([[INPUT_1]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[ADD:%.+]] = VPU.Add([[INPUT0]], [[INPUT1]])
    //CHECK-SAME:         -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[ADD]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @AddSWSOHTileAtBroadcastAxis
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x32x44x44xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x44xf16>
func.func @AddSWSOHTileAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16>,
                %arg1: tensor<1x1x1x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16>,
                tensor<1x1x1x44xf16> -> tensor<1x32x44x44xf16>

    return %0 : tensor<1x32x44x44xf16>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy([[INPUT_0]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy([[INPUT_1]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[ADD:%.+]] = VPU.Add([[INPUT0]], [[INPUT1]])
    //CHECK-SAME:         -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[ADD]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UnrollSOKAveragePoolInputOutputDuplicated
func.func @UnrollSOKAveragePoolInputOutputDuplicated(%input: tensor<1x1x320x1xf16>) -> tensor<1x320x1x1xf16, {order = #NHWC}> {
    %mvn = VPU.MVN(%input) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true}
            : tensor<1x1x320x1xf16> -> tensor<1x1x320x1xf16>

    %reshape = VPU.AffineReshape(%mvn) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 320, 1, 1]}
            : tensor<1x1x320x1xf16> -> tensor<1x320x1x1xf16>

    %cast = VPU.PermuteCast(%reshape) {dst_order = #NHWC, mem_perm = #NHWC}
            : tensor<1x320x1x1xf16> -> tensor<1x320x1x1xf16, {order = #NHWC}>

    %averagePool = VPU.NCE.AveragePool(%cast) {kernel_size = [1, 1],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x320x1x1xf16, {order = #NHWC}>

    %activation = VPU.Sigmoid(%averagePool) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
            : tensor<1x320x1x1xf16, {order = #NHWC}> -> tensor<1x320x1x1xf16, {order = #NHWC}>

    return %activation : tensor<1x320x1x1xf16, {order = #NHWC}>

    // (DUP) MVN (DUP) -> (DUP) AveragePool (DUP | SEG)

    //CHECK:        [[MVN_COPY_IN:%.+]] = VPU.Copy
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x1x320x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[MVN:%.+]] = VPU.MVN([[MVN_COPY_IN]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x1x320x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[MVN_COPY_OUT:%.+]] = VPU.Copy([[MVN]]
    //CHECK-SAME:       -> tensor<1x1x320x1xf16>

    //CHECK:        [[RESHAPE:%.+]] = VPU.AffineReshape([[MVN_COPY_OUT]])
    //CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 320, 1, 1]} : tensor<1x1x320x1xf16> -> tensor<1x320x1x1xf16>

    //CHECK:        [[CAST:%.+]] = VPU.PermuteCast([[RESHAPE]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x320x1x1xf16> -> tensor<1x320x1x1xf16, {order = #NHWC}>

    //CHECK:        [[AVERAGEPOOL_INPUT_COPY_IN:%.+]] = VPU.Copy([[CAST]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[AVERAGEPOOL:%.+]]  = VPU.NCE.AveragePool([[AVERAGEPOOL_INPUT_COPY_IN]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[AVERAGEPOOL_COPY_OUT:%.+]] = VPU.Copy([[AVERAGEPOOL]])
    //CHECK-SAME:       -> tensor<1x320x1x1xf16, {order = #NHWC}>

    //CHECK:        [[SIGMOID_COPY_IN:%.+]] = VPU.Copy([[AVERAGEPOOL_COPY_OUT]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[SIGMOID:%.+]] = VPU.Sigmoid([[SIGMOID_COPY_IN]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[SIGMOID_COPY_OUT:%.+]] = VPU.Copy([[SIGMOID]])
    //CHECK-SAME:       -> tensor<1x320x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MultiDepthConv
func.func @MultiDepthConv(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x96x112x112xf16, {order = #NHWC}> {
    %wt_1 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %weight_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %dwconv_1 = VPU.NCE.DepthConvolution(%arg0, %weight_1, %wt_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x112x112xf16, {order = #NHWC}>

    %wt_2 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %weight_2 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %dwconv_2 = VPU.NCE.DepthConvolution(%arg0, %weight_2, %wt_2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x112x112xf16, {order = #NHWC}>


    %wt_3 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %weight_3 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %dwconv_3 = VPU.NCE.DepthConvolution(%arg0, %weight_3, %wt_3) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x112x112xf16, {order = #NHWC}>

    %concat = VPU.Concat(%dwconv_1, %dwconv_2, %dwconv_3) {static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0]]} : tensor<1x32x112x112xf16, {order = #NHWC}>, tensor<1x32x112x112xf16, {order = #NHWC}>, tensor<1x32x112x112xf16, {order = #NHWC}> -> tensor<1x96x112x112xf16, {order = #NHWC}>
    return %concat: tensor<1x96x112x112xf16, {order = #NHWC}>

    // CHECK: [[TILING_COPY_1:%.+]] = VPU.Copy
    // CHECK: [[TILING_COPY_2:%.+]] = VPU.Copy
    // CHECK: [[TILING_COPY_3:%.+]] = VPU.Copy
    // CHECK: [[DWCONV_1:%.+]] = VPU.NCE.DepthConvolution
    // CHECK: [[TILING_COPY_OUT_1:%.+]] = VPU.Copy

    // CHECK: [[TILING_COPY_4:%.+]] = VPU.Copy
    // CHECK: [[TILING_COPY_5:%.+]] = VPU.Copy
    // CHECK: [[TILING_COPY_6:%.+]] = VPU.Copy
    // CHECK: [[DWCONV_2:%.+]] = VPU.NCE.DepthConvolution
    // CHECK: [[TILING_COPY_OUT_2:%.+]] = VPU.Copy

    // CHECK: [[TILING_COPY_7:%.+]] = VPU.Copy
    // CHECK: [[TILING_COPY_8:%.+]] = VPU.Copy
    // CHECK: [[TILING_COPY_9:%.+]] = VPU.Copy
    // CHECK: [[DWCONV_3:%.+]] = VPU.NCE.DepthConvolution
    // CHECK: [[TILING_COPY_OUT_3:%.+]] = VPU.Copy

    // CHECK: [[CONCAT:%.+]] = VPU.Concat([[TILING_COPY_OUT_1]], [[TILING_COPY_OUT_2]], [[TILING_COPY_OUT_3]])
    // CHECK:    {static_offsets = [
    // CHECK-SAME:   [0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0]
    // CHECK-SAME:   ]} : tensor<1x32x112x112xf16, {order = #NHWC}>, tensor<1x32x112x112xf16, {order = #NHWC}>, tensor<1x32x112x112xf16, {order = #NHWC}>
    // CHECK-SAME:    -> tensor<1x96x112x112xf16, {order = #NHWC}>
    // CHECK:   return [[CONCAT]] : tensor<1x96x112x112xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: func.func @PReluSWSOH
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x8x128x128xf16>)
func.func @PReluSWSOH(%arg0: tensor<1x8x128x128xf16>) -> tensor<1x8x128x128xf16> {

    %cst = const.Declare tensor<1x8x1x1xf16> = dense<-1.000000e+01> : tensor<1x8x1x1xf16>
    %0 = VPU.PRelu(%arg0, %cst) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
          : tensor<1x8x128x128xf16>, tensor<1x8x1x1xf16> -> tensor<1x8x128x128xf16>

    return %0 : tensor<1x8x128x128xf16>

    //CHECK-DAG:    [[SLOPE:%.+]] = const.Declare tensor<1x8x1x1xf16> = dense<-1.000000e+01> : tensor<1x8x1x1xf16>
    //CHECK:        [[INPUT0:%.+]] = VPU.Copy([[INPUT_DATA]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x8x128x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy([[SLOPE]])
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x8x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[PRELU:%.+]] = VPU.PRelu([[INPUT0]], [[INPUT1]])
    //CHECK-SAME:         -> !VPU.DistributedTensor<1x8x128x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[PRELU]]

    //CHECK:        return [[OUTPUT]] : tensor<1x8x128x128xf16>
}

// -----

// CHECK-LABEL: func.func @PReluSWSOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x128x1x1xf16>)
func.func @PReluSWSOK(%arg0: tensor<1x128x1x1xf16>) -> tensor<1x128x1x1xf16> {

    %cst = const.Declare tensor<1x128x1x1xf16> = dense<-1.000000e+01> : tensor<1x128x1x1xf16>
    %0 = VPU.PRelu(%arg0, %cst) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
          : tensor<1x128x1x1xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x1x1xf16>

    return %0 : tensor<1x128x1x1xf16>

    //CHECK-DAG:    [[SLOPE:%.+]] = const.Declare tensor<1x128x1x1xf16> = dense<-1.000000e+01> : tensor<1x128x1x1xf16>
    //CHECK:        [[INPUT0:%.+]] = VPU.Copy([[INPUT_DATA]])
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x128x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy([[SLOPE]])
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x128x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[PRELU:%.+]] = VPU.PRelu([[INPUT0]], [[INPUT1]])
    //CHECK-SAME:         -> !VPU.DistributedTensor<1x128x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[PRELU]]

    //CHECK:        return [[OUTPUT]] : tensor<1x128x1x1xf16>
}

// -----

// CHECK-LABEL: func.func @PReluSWClustering
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x8x128x128xf16>)
func.func @PReluSWClustering(%arg0: tensor<1x8x128x128xf16>) -> tensor<1x8x128x128xf16> {

    %cst = const.Declare tensor<1x8x1x1xf16> = dense<-1.000000e+01> : tensor<1x8x1x1xf16>
    %0 = VPU.PRelu(%arg0, %cst) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
          : tensor<1x8x128x128xf16>, tensor<1x8x1x1xf16> -> tensor<1x8x128x128xf16>

    return %0 : tensor<1x8x128x128xf16>

    //CHECK-DAG:    [[SLOPE:%.+]] = const.Declare tensor<1x8x1x1xf16> = dense<-1.000000e+01> : tensor<1x8x1x1xf16>
    //CHECK:        [[INPUT0:%.+]] = VPU.Copy([[INPUT_DATA]])
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x8x128x128xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy([[SLOPE]])
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x8x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[PRELU:%.+]] = VPU.PRelu([[INPUT0]], [[INPUT1]])
    //CHECK-SAME:         -> !VPU.DistributedTensor<1x8x128x128xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[PRELU]]

    //CHECK:        return [[OUTPUT]] : tensor<1x8x128x128xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertOpINT64toFP64Clustering
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x1x1x128xsi64>, [[OUTPUT_DATA:%.+]]: tensor<1x1x1x128xf64>)
func.func @ConvertOpINT64toFP64Clustering(%arg0: tensor<1x1x1x128xsi64>, %arg1: tensor<1x1x1x128xf64>) -> tensor<1x1x1x128xf64> {

    %0 = VPU.Convert(%arg0) {
            dstElemType = f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
          : tensor<1x1x1x128xsi64> -> tensor<1x1x1x128xf64>

    return %0 : tensor<1x1x1x128xf64>

    //CHECK:        [[INPUT:%.+]] = VPU.Copy([[INPUT_DATA]])
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x128xsi64, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[CONVERT:%.+]] = VPU.Convert([[INPUT]])
    //CHECK:                            -> !VPU.DistributedTensor<1x1x1x128xf64, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[CONVERT]]

    //CHECK:        return [[OUTPUT]] : tensor<1x1x1x128xf64>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @FloorSWSOH
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x16x512xf16>)
func.func @FloorSWSOH(%arg0: tensor<1x16x16x512xf16>) -> tensor<1x16x16x512xf16> {

    %0 = VPU.Floor(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
          : tensor<1x16x16x512xf16> -> tensor<1x16x16x512xf16>

    return %0 : tensor<1x16x16x512xf16>

    //CHECK:        [[INPUT:%.+]] = VPU.Copy([[INPUT_DATA]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[FLOOR:%.+]] = VPU.Floor([[INPUT]]
    //CHECK:                            -> !VPU.DistributedTensor<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[FLOOR]]

    //CHECK:        return [[OUTPUT]] : tensor<1x16x16x512xf16>
}

// -----

// CHECK-LABEL: func.func @FloorSWSOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x1x513xf16>)
func.func @FloorSWSOK(%arg0: tensor<1x16x1x513xf16>) -> tensor<1x16x1x513xf16> {

    %0 = VPU.Floor(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
          : tensor<1x16x1x513xf16> -> tensor<1x16x1x513xf16>

    return %0 : tensor<1x16x1x513xf16>

    //CHECK:        [[INPUT:%.+]] = VPU.Copy([[INPUT_DATA]])
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x1x513xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[FLOOR:%.+]] = VPU.Floor([[INPUT]])
    //CHECK:                            -> !VPU.DistributedTensor<1x16x1x513xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[FLOOR]]

    //CHECK:        return [[OUTPUT]] : tensor<1x16x1x513xf16>
}

// -----

// CHECK-LABEL: func.func @FloorSWClustering
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x1x1x513xf16>)
func.func @FloorSWClustering(%arg0: tensor<1x1x1x513xf16>) -> tensor<1x1x1x513xf16> {

    %0 = VPU.Floor(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
          : tensor<1x1x1x513xf16> -> tensor<1x1x1x513xf16>

    return %0 : tensor<1x1x1x513xf16>

    //CHECK:        [[INPUT:%.+]] = VPU.Copy([[INPUT_DATA]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x513xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[FLOOR:%.+]] = VPU.Floor([[INPUT]])
    //CHECK:                            -> !VPU.DistributedTensor<1x1x1x513xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[FLOOR]]

    //CHECK:        return [[OUTPUT]] : tensor<1x1x1x513xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @RoundSWSOH
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x16x512xf16>)
func.func @RoundSWSOH(%arg0: tensor<1x16x16x512xf16>) -> tensor<1x16x16x512xf16> {
    %0 = VPU.Round(%arg0) {
            mode = #IE.round_mode<HALF_TO_EVEN>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
          : tensor<1x16x16x512xf16> -> tensor<1x16x16x512xf16>

    return %0 : tensor<1x16x16x512xf16>

    //CHECK:        [[INPUT:%.+]] = VPU.Copy([[INPUT_DATA]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[ROUND:%.+]] = VPU.Round([[INPUT]])
    //CHECK:                            -> !VPU.DistributedTensor<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[ROUND]]

    //CHECK:        return [[OUTPUT]] : tensor<1x16x16x512xf16>
}

// -----

// CHECK-LABEL: func.func @RoundSWSOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x1x513xf16>)
func.func @RoundSWSOK(%arg0: tensor<1x16x1x513xf16>) -> tensor<1x16x1x513xf16> {

    %0 = VPU.Round(%arg0) {
            mode = #IE.round_mode<HALF_TO_EVEN>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
          : tensor<1x16x1x513xf16> -> tensor<1x16x1x513xf16>

    return %0 : tensor<1x16x1x513xf16>

    //CHECK:        [[INPUT:%.+]] = VPU.Copy([[INPUT_DATA]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x1x513xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[ROUND:%.+]] = VPU.Round([[INPUT]])
    //CHECK:                            -> !VPU.DistributedTensor<1x16x1x513xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[ROUND]]

    //CHECK:        return [[OUTPUT]] : tensor<1x16x1x513xf16>
}

// -----

// CHECK-LABEL: func.func @RoundSWClustering
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x1x1x513xf16>)
func.func @RoundSWClustering(%arg0: tensor<1x1x1x513xf16>) -> tensor<1x1x1x513xf16> {

    %0 = VPU.Round(%arg0) {
            mode = #IE.round_mode<HALF_TO_EVEN>, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
          : tensor<1x1x1x513xf16> -> tensor<1x1x1x513xf16>

    return %0 : tensor<1x1x1x513xf16>

    //CHECK:        [[INPUT:%.+]] = VPU.Copy([[INPUT_DATA]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x513xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[ROUND:%.+]] = VPU.Round([[INPUT]])
    //CHECK:                            -> !VPU.DistributedTensor<1x1x1x513xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[ROUND]]

    //CHECK:        return [[OUTPUT]] : tensor<1x1x1x513xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AccumulateClustering
// CHECK-SAME: ([[LHS:%arg[0-9]]]: tensor<1x64x16x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS:%arg[0-9]]]: tensor<1x64x16x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[LHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
func.func @AccumulateClustering(
    %LHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %RHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %LHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>,
    %RHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>
) -> tensor<1x64x16x1xf16, {order = #NHWC}> {
    %ACCUMULATE = VPU.Accumulate(%LHS, %RHS, %LHS_SCALES, %RHS_SCALES) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
    } : tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>
            -> tensor<1x64x16x1xf16, {order = #NHWC}>

    // CHECK:   [[COPY_LHS:%.+]] = VPU.Copy([[LHS]])
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_RHS:%.+]] = VPU.Copy([[RHS]])
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_LHS_SCALES:%.+]] = VPU.Copy([[LHS_SCALES]])
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_RHS_SCALES:%.+]] = VPU.Copy([[RHS_SCALES]])
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }>

    // CHECK:   [[ACCUMULATE:%.+]] = VPU.Accumulate(
    // CHECK-SAME:      [[COPY_LHS]],
    // CHECK-SAME:      [[COPY_RHS]],
    // CHECK-SAME:      [[COPY_LHS_SCALES]],
    // CHECK-SAME:      [[COPY_RHS_SCALES]])
    // CHECK-SAME:   -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_OUT:%.+]] = VPU.Copy([[ACCUMULATE]])
    // CHECK-SAME:  -> tensor<1x64x16x1xf16, {order = #NHWC}>

    return %ACCUMULATE : tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK:   return [[COPY_OUT]] : tensor<1x64x16x1xf16, {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AccumulateSplitOverHeight
// CHECK-SAME: ([[LHS:%arg[0-9]]]: tensor<1x64x16x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS:%arg[0-9]]]: tensor<1x64x16x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[LHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
func.func @AccumulateSplitOverHeight(
    %LHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %RHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %LHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>,
    %RHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>
) -> tensor<1x64x16x1xf16, {order = #NHWC}> {
    %ACCUMULATE = VPU.Accumulate(%LHS, %RHS, %LHS_SCALES, %RHS_SCALES) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    } : tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>
            -> tensor<1x64x16x1xf16, {order = #NHWC}>

    // CHECK:   [[COPY_LHS:%.+]] = VPU.Copy([[LHS]])
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_RHS:%.+]] = VPU.Copy([[RHS]])
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_LHS_SCALES:%.+]] = VPU.Copy([[LHS_SCALES]])
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_RHS_SCALES:%.+]] = VPU.Copy([[RHS_SCALES]])
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }>

    // CHECK:   [[ACCUMULATE:%.+]] = VPU.Accumulate(
    // CHECK-SAME:      [[COPY_LHS]],
    // CHECK-SAME:      [[COPY_RHS]],
    // CHECK-SAME:      [[COPY_LHS_SCALES]],
    // CHECK-SAME:      [[COPY_RHS_SCALES]])
    // CHECK-SAME:   -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_OUT:%.+]] = VPU.Copy([[ACCUMULATE]])
    // CHECK-SAME:  -> tensor<1x64x16x1xf16, {order = #NHWC}>

    return %ACCUMULATE : tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK:   return [[COPY_OUT]] : tensor<1x64x16x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AccumulateSplitOverKernel
// CHECK-SAME: ([[LHS:%arg[0-9]]]: tensor<1x64x16x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS:%arg[0-9]]]: tensor<1x64x16x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[LHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
func.func @AccumulateSplitOverKernel(
    %LHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %RHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %LHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>,
    %RHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>
) -> tensor<1x64x16x1xf16, {order = #NHWC}> {
    %ACCUMULATE = VPU.Accumulate(%LHS, %RHS, %LHS_SCALES, %RHS_SCALES) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    } : tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>
            -> tensor<1x64x16x1xf16, {order = #NHWC}>

    // CHECK:   [[COPY_LHS:%.+]] = VPU.Copy([[LHS]])
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 2, 1, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_RHS:%.+]] = VPU.Copy([[RHS]])
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 2, 1, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_LHS_SCALES:%.+]] = VPU.Copy([[LHS_SCALES]])
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 2, 1, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_RHS_SCALES:%.+]] = VPU.Copy([[RHS_SCALES]])
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 2, 1, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }>

    // CHECK:   [[ACCUMULATE:%.+]] = VPU.Accumulate(
    // CHECK-SAME:      [[COPY_LHS]],
    // CHECK-SAME:      [[COPY_RHS]],
    // CHECK-SAME:      [[COPY_LHS_SCALES]],
    // CHECK-SAME:      [[COPY_RHS_SCALES]])
    // CHECK-SAME:   -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 2, 1, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_OUT:%.+]] = VPU.Copy([[ACCUMULATE]])
    // CHECK-SAME:  -> tensor<1x64x16x1xf16, {order = #NHWC}>

    return %ACCUMULATE : tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK:   return [[COPY_OUT]] : tensor<1x64x16x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AccumulateSplitOverWidth
// CHECK-SAME: ([[LHS:%arg[0-9]]]: tensor<1x64x16x32xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS:%arg[0-9]]]: tensor<1x64x16x32xf16, {order = #NHWC}>,
// CHECK-SAME:  [[LHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
func.func @AccumulateSplitOverWidth(
    %LHS: tensor<1x64x16x32xf16, {order = #NHWC}>,
    %RHS: tensor<1x64x16x32xf16, {order = #NHWC}>,
    %LHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>,
    %RHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>
) -> tensor<1x64x16x32xf16, {order = #NHWC}> {
    %ACCUMULATE = VPU.Accumulate(%LHS, %RHS, %LHS_SCALES, %RHS_SCALES) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>
    } : tensor<1x64x16x32xf16, {order = #NHWC}>,
        tensor<1x64x16x32xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>
            -> tensor<1x64x16x32xf16, {order = #NHWC}>

    // CHECK:   [[COPY_LHS:%.+]] = VPU.Copy([[LHS]])
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x32xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, 2],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_RHS:%.+]] = VPU.Copy([[RHS]])
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x32xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, 2],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_LHS_SCALES:%.+]] = VPU.Copy([[LHS_SCALES]])
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_RHS_SCALES:%.+]] = VPU.Copy([[RHS_SCALES]])
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }>

    // CHECK:   [[ACCUMULATE:%.+]] = VPU.Accumulate(
    // CHECK-SAME:      [[COPY_LHS]],
    // CHECK-SAME:      [[COPY_RHS]],
    // CHECK-SAME:      [[COPY_LHS_SCALES]],
    // CHECK-SAME:      [[COPY_RHS_SCALES]])
    // CHECK-SAME:   -> !VPU.DistributedTensor<1x64x16x32xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, 2],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_OUT:%.+]] = VPU.Copy([[ACCUMULATE]])
    // CHECK-SAME:  -> tensor<1x64x16x32xf16, {order = #NHWC}>

    return %ACCUMULATE : tensor<1x64x16x32xf16, {order = #NHWC}>
    // CHECK:   return [[COPY_OUT]] : tensor<1x64x16x32xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PoolingSplitOverWidth
// CHECK-SAME: ([[DATA:%arg[0-9]]]: tensor<1x16x32x64xf16, {order = #NHWC}>)
func.func @PoolingSplitOverWidth(
    %DATA: tensor<1x16x32x64xf16, {order = #NHWC}>
) -> tensor<1x16x32x64xf16, {order = #NHWC}> {
    %POOL = VPU.NCE.MaxPool(%DATA) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        kernel_size = [3, 3]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>
            -> tensor<1x16x32x64xf16, {order = #NHWC}>

    // CHECK:   [[COPY_INPUT:%.+]] = VPU.Copy([[DATA]])
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x16x32x64xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "OVERLAPPED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, 2],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }>

    // CHECK:   [[POOL:%.+]] = VPU.NCE.MaxPool(
    // CHECK-SAME:      [[COPY_INPUT]]
    // CHECK-SAME:   -> !VPU.DistributedTensor<1x16x32x64xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, 2],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_OUT:%.+]] = VPU.Copy([[POOL]])
    // CHECK-SAME:  -> tensor<1x16x32x64xf16, {order = #NHWC}>

    return %POOL : tensor<1x16x32x64xf16, {order = #NHWC}>
    // CHECK:   return [[COPY_OUT]] : tensor<1x16x32x64xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @FakeQuantizeSWSOH
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x3x384x640xf16>)
func.func @FakeQuantizeSWSOH(%arg0: tensor<1x3x384x640xf16>) -> tensor<1x3x384x640xf16> {
    %inLow = const.Declare tensor<1x3x1x1xf16> = dense<-1.000000e+01> : tensor<1x3x1x1xf16>
    %inHigh = const.Declare tensor<1x3x1x1xf16> = dense<1.000000e+01> : tensor<1x3x1x1xf16>
    %outLow = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    %outHigh = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>

    %fq = VPU.FakeQuantize(%arg0, %inLow, %inHigh, %outLow, %outHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x3x384x640xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x384x640xf16>
    return %fq : tensor<1x3x384x640xf16>

    //CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x3x1x1xf16> = dense<-1.000000e+01> : tensor<1x3x1x1xf16>
    //CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x3x1x1xf16> = dense<1.000000e+01> : tensor<1x3x1x1xf16>
    //CHECK-DAG: [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    //CHECK-DAG: [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>

    //CHECK: [[INPUT_COPY:%.+]] = VPU.Copy([[INPUT_DATA]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x3x384x640xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK: [[IN_LOW_COPY:%.+]] = VPU.Copy([[IN_LOW]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK: [[IN_HIGH_COPY:%.+]]  = VPU.Copy([[IN_HIGH]])
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK: [[OUT_LOW_COPY:%.+]] = VPU.Copy([[OUT_LOW]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK: [[OUT_HIGH_COPY:%.+]] = VPU.Copy([[OUT_HIGH]]
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK: [[FQ_CLUSTER:%.+]] = VPU.FakeQuantize([[INPUT_COPY]], [[IN_LOW_COPY]], [[IN_HIGH_COPY]], [[OUT_LOW_COPY]], [[OUT_HIGH_COPY]])
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x3x384x640xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK: [[OUTPUT:%.+]] = VPU.Copy([[FQ_CLUSTER]]
    //CHECK: return [[OUTPUT]] : tensor<1x3x384x640xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @FakeQuantizeSWSOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x128x1x512xf16>)
func.func @FakeQuantizeSWSOK(%arg0: tensor<1x128x1x512xf16>) -> tensor<1x128x1x512xf16> {
    %inLow = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    %inHigh = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>
    %outLow = const.Declare tensor<1x128x1x1xf16> = dense<-1.000000e+01> : tensor<1x128x1x1xf16>
    %outHigh = const.Declare tensor<1x128x1x1xf16> = dense<1.000000e+01> : tensor<1x128x1x1xf16>

    %fq = VPU.FakeQuantize(%arg0, %inLow, %inHigh, %outLow, %outHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x128x1x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x128x1x1xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x1x512xf16>
    return %fq : tensor<1x128x1x512xf16>

    //CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    //CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>
    //CHECK-DAG: [[OUT_LOW:%.+]] = const.Declare tensor<1x128x1x1xf16> = dense<-1.000000e+01> : tensor<1x128x1x1xf16>
    //CHECK-DAG: [[OUT_HIGH:%.+]] = const.Declare tensor<1x128x1x1xf16> = dense<1.000000e+01> : tensor<1x128x1x1xf16>

    //CHECK: [[INPUT_COPY:%.+]] = VPU.Copy([[INPUT_DATA]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x1x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK: [[IN_LOW_COPY:%.+]] = VPU.Copy([[IN_LOW]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK: [[IN_HIGH_COPY:%.+]]  = VPU.Copy([[IN_HIGH]]
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK: [[OUT_LOW_COPY:%.+]] = VPU.Copy([[OUT_LOW]]
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x128x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK: [[OUT_HIGH_COPY:%.+]] = VPU.Copy([[OUT_HIGH]]
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x128x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK: [[FQ_CLUSTER:%.+]] = VPU.FakeQuantize([[INPUT_COPY]], [[IN_LOW_COPY]], [[IN_HIGH_COPY]], [[OUT_LOW_COPY]], [[OUT_HIGH_COPY]])
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x128x1x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK: [[OUTPUT:%.+]] = VPU.Copy([[FQ_CLUSTER]]
    //CHECK: return [[OUTPUT]] : tensor<1x128x1x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @FakeQuantizeSWClustering
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x1x1x512xf16>)
func.func @FakeQuantizeSWClustering(%arg0: tensor<1x1x1x512xf16>) -> tensor<1x1x1x512xf16> {
    %inLow = const.Declare tensor<1x1x1x512xf16> = dense<-1.000000e+01> : tensor<1x1x1x512xf16>
    %inHigh = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+01> : tensor<1x1x1x512xf16>
    %outLow = const.Declare tensor<1x1x1x512xf16> = dense<-1.000000e+01> : tensor<1x1x1x512xf16>
    %outHigh = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+01> : tensor<1x1x1x512xf16>

    %fq = VPU.FakeQuantize(%arg0, %inLow, %inHigh, %outLow, %outHigh) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>, levels = 256 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16> -> tensor<1x1x1x512xf16>
    return %fq : tensor<1x1x1x512xf16>

    //CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x512xf16> = dense<-1.000000e+01> : tensor<1x1x1x512xf16>
    //CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+01> : tensor<1x1x1x512xf16>
    //CHECK-DAG: [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x512xf16> = dense<-1.000000e+01> : tensor<1x1x1x512xf16>
    //CHECK-DAG: [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+01> : tensor<1x1x1x512xf16>

    //CHECK: [[INPUT_COPY:%.+]] = VPU.Copy([[INPUT_DATA]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK: [[IN_LOW_COPY:%.+]] = VPU.Copy([[IN_LOW]]
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK: [[IN_HIGH_COPY:%.+]] = VPU.Copy([[IN_HIGH]]
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK: [[OUT_LOW_COPY:%.+]] = VPU.Copy([[OUT_LOW]]
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK: [[OUT_HIGH_COPY:%.+]] = VPU.Copy([[OUT_HIGH]]
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK: [[FQ_CLUSTER:%.+]] = VPU.FakeQuantize([[INPUT_COPY]], [[IN_LOW_COPY]], [[IN_HIGH_COPY]], [[OUT_LOW_COPY]], [[OUT_HIGH_COPY]])
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK: [[OUTPUT:%.+]] = VPU.Copy([[FQ_CLUSTER]]
    //CHECK: return [[OUTPUT]] : tensor<1x1x1x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @SelectSWSplitOverHeight
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x10x40x40xf16>, [[INPUT1:%.+]]: tensor<1x10x40x40xf16>)
func.func @SelectSWSplitOverHeight(%arg0: tensor<1x10x40x40xf16>, %arg1: tensor<1x10x40x40xf16>) -> tensor<1x10x40x40xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<f16>, [#const.Reshape<[1]>, #const.Reshape<[1, 1, 1, 1]>]
    %0 = VPU.Select(%arg0, %cst, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
        } : tensor<1x10x40x40xf16>, tensor<1x1x1x1xf16>, tensor<1x10x40x40xf16> -> tensor<1x10x40x40xf16>
    return %0 : tensor<1x10x40x40xf16>

    //CHECK-DAG:    [[INPUT0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<f16>, [#const.Reshape<[1]>, #const.Reshape<[1, 1, 1, 1]>]

    //CHECK:        [[INPUT_COPY:%.+]] = VPU.Copy([[INPUT]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x10x40x40xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[INPUT0_COPY:%.+]] = VPU.Copy([[INPUT0]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1_COPY:%.+]] = VPU.Copy([[INPUT1]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x10x40x40xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[SELECT:%.+]] = VPU.Select(
    //CHECK-SAME:           [[INPUT_COPY]],
    //CHECK-SAME:           [[INPUT0_COPY]],
    //CHECK-SAME:           [[INPUT1_COPY]])
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x10x40x40xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[SELECT]]

    //CHECK: return [[OUTPUT]] : tensor<1x10x40x40xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @SelectSWSplitOverKernel
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x1x40xf16>, [[INPUT1:%.+]]: tensor<1x16x1x40xf16>)
func.func @SelectSWSplitOverKernel(%arg0: tensor<1x16x1x40xf16>, %arg1: tensor<1x16x1x40xf16>) -> tensor<1x16x1x40xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<f16>, [#const.Reshape<[1]>, #const.Reshape<[1, 1, 1, 1]>]
    %0 = VPU.Select(%arg0, %cst, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
        } : tensor<1x16x1x40xf16>, tensor<1x1x1x1xf16>, tensor<1x16x1x40xf16> -> tensor<1x16x1x40xf16>
    return %0 : tensor<1x16x1x40xf16>

    //CHECK-DAG:    [[INPUT0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<f16>, [#const.Reshape<[1]>, #const.Reshape<[1, 1, 1, 1]>]

    //CHECK:        [[INPUT_COPY:%.+]] = VPU.Copy([[INPUT]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x1x40xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[INPUT0_COPY:%.+]] = VPU.Copy([[INPUT0]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1_COPY:%.+]] = VPU.Copy([[INPUT1]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x1x40xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[SELECT:%.+]] = VPU.Select(
    //CHECK-SAME:           [[INPUT_COPY]],
    //CHECK-SAME:           [[INPUT0_COPY]],
    //CHECK-SAME:           [[INPUT1_COPY]])
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x1x40xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[SELECT]]

    //CHECK: return [[OUTPUT]] : tensor<1x16x1x40xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @LSTMGatesSWSOHTile
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x100x2048xf16>, [[INPUT_1:%.+]]: tensor<1x1x100x512xf16>
func.func @LSTMGatesSWSOHTile(%arg0: tensor<1x1x100x2048xf16>, %arg1: tensor<1x1x100x512xf16>) -> (tensor<1x1x100x512xf16>, tensor<1x1x100x512xf16>) {
    %0, %1 = VPU.LSTMGates(%arg0, %arg1) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
                } : tensor<1x1x100x2048xf16>, tensor<1x1x100x512xf16> -> tensor<1x1x100x512xf16>, tensor<1x1x100x512xf16>

    return %0, %1 : tensor<1x1x100x512xf16>, tensor<1x1x100x512xf16>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy([[INPUT_0]])
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x100x2048xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy([[INPUT_1]])
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x100x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[LSTMGATES_0:%.+]], [[LSTMGATES_1:%.+]] = VPU.LSTMGates(
    //CHECK-SAME:         [[INPUT0]],
    //CHECK-SAME:         [[INPUT1]])
    //CHECK-SAME:          -> !VPU.DistributedTensor<1x1x100x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPU.DistributedTensor<1x1x100x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT0:%.+]] = VPU.Copy([[LSTMGATES_0]]

    //CHECK:        [[OUTPUT1:%.+]] = VPU.Copy([[LSTMGATES_1]]

    //CHECK:        return [[OUTPUT0]], [[OUTPUT1]] : tensor<1x1x100x512xf16>, tensor<1x1x100x512xf16>
}

// -----

// CHECK-LABEL: @AndClustering
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x16x32x32xf16>, [[INPUT1:%.+]]: tensor<1x1x32x32xf16>)
func.func @AndClustering(%arg0: tensor<1x16x32x32xf16>, %arg1: tensor<1x1x32x32xf16>) -> tensor<1x16x32x32xf16> {
    %0 = VPU.And(%arg0, %arg1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
    } : tensor<1x16x32x32xf16>, tensor<1x1x32x32xf16> -> tensor<1x16x32x32xf16>

    return %0 : tensor<1x16x32x32xf16>

    // CHECK:               [[INPUT0_COPY:%.+]] = VPU.Copy([[INPUT0]])
    // CHECK-SAME:               -> !VPU.DistributedTensor<1x16x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:               [[INPUT1_COPY:%.+]] = VPU.Copy([[INPUT1]]
    // CHECK-SAME:               -> !VPU.DistributedTensor<1x1x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:               [[AND:%.+]] = VPU.And([[INPUT0_COPY]], [[INPUT1_COPY]])
    // CHECK-SAME:               -> !VPU.DistributedTensor<1x16x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:               [[RES:%.+]] = VPU.Copy([[AND]]

    // CHECK:               return [[RES]] : tensor<1x16x32x32xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @LogSoftmaxSWSOH
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x4x2x512xf16, {order = #NHWC}>)
func.func @LogSoftmaxSWSOH(%arg0: tensor<1x4x2x512xf16, {order = #NHWC}>) -> tensor<1x4x2x512xf16, {order = #NHWC}> {
    %0 = VPU.LogSoftmax(%arg0) {
            axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
          : tensor<1x4x2x512xf16, {order = #NHWC}> -> tensor<1x4x2x512xf16, {order = #NHWC}>

    return %0 : tensor<1x4x2x512xf16, {order = #NHWC}>

    //CHECK:        [[INPUT:%.+]] = VPU.Copy([[INPUT_DATA]])
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x4x2x512xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[LOG_SOFTMAX:%.+]] = VPU.LogSoftmax([[INPUT]])
    //CHECK:                            -> !VPU.DistributedTensor<1x4x2x512xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[LOG_SOFTMAX]]

    //CHECK:        return [[OUTPUT]] : tensor<1x4x2x512xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @AndSplitOverHeight
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x16x32x32xf16>, [[INPUT1:%.+]]: tensor<1x1x32x32xf16>)
func.func @AndSplitOverHeight(%arg0: tensor<1x16x32x32xf16>, %arg1: tensor<1x1x32x32xf16>) -> tensor<1x16x32x32xf16> {
    %0 = VPU.And(%arg0, %arg1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    } : tensor<1x16x32x32xf16>, tensor<1x1x32x32xf16> -> tensor<1x16x32x32xf16>

    return %0 : tensor<1x16x32x32xf16>

    // CHECK:               [[INPUT0_COPY:%.+]] = VPU.Copy([[INPUT0]])
    // CHECK-SAME:               -> !VPU.DistributedTensor<1x16x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:               [[INPUT1_COPY:%.+]] = VPU.Copy([[INPUT1]]
    // CHECK-SAME:               -> !VPU.DistributedTensor<1x1x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:               [[AND:%.+]] = VPU.And([[INPUT0_COPY]], [[INPUT1_COPY]])
    // CHECK-SAME:               -> !VPU.DistributedTensor<1x16x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:               [[RES:%.+]] = VPU.Copy([[AND]]

    // CHECK:               return [[RES]] : tensor<1x16x32x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @LogSoftmaxSWSOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x4x1x512xf16, {order = #NCHW}>)
func.func @LogSoftmaxSWSOK(%arg0: tensor<1x4x1x512xf16, {order = #NCHW}>) -> tensor<1x4x1x512xf16, {order = #NCHW}> {
    %0 = VPU.LogSoftmax(%arg0) {
            axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
          : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>

    return %0 : tensor<1x4x1x512xf16, {order = #NCHW}>

    //CHECK:        [[INPUT:%.+]] = VPU.Copy([[INPUT_DATA]])
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x4x1x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[LOG_SOFTMAX:%.+]] = VPU.LogSoftmax([[INPUT]])
    //CHECK:                            -> !VPU.DistributedTensor<1x4x1x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[LOG_SOFTMAX]]

    //CHECK:        return [[OUTPUT]] : tensor<1x4x1x512xf16, {order = #NCHW}>
}

// -----

// CHECK-LABEL: @AndSplitOverKernel
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x64x32x32xf16>, [[INPUT1:%.+]]: tensor<1x1x32x32xf16>)
func.func @AndSplitOverKernel(%arg0: tensor<1x64x32x32xf16>, %arg1: tensor<1x1x32x32xf16>) -> tensor<1x64x32x32xf16> {
    %0 = VPU.And(%arg0, %arg1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    } : tensor<1x64x32x32xf16>, tensor<1x1x32x32xf16> -> tensor<1x64x32x32xf16>

    return %0 : tensor<1x64x32x32xf16>

    // CHECK:               [[INPUT0_COPY:%.+]] = VPU.Copy([[INPUT0]]
    // CHECK-SAME:               -> !VPU.DistributedTensor<1x64x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:               [[INPUT1_COPY:%.+]] = VPU.Copy([[INPUT1]])
    // CHECK-SAME:               -> !VPU.DistributedTensor<1x1x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:               [[AND:%.+]] = VPU.And([[INPUT0_COPY]], [[INPUT1_COPY]])
    // CHECK-SAME:               -> !VPU.DistributedTensor<1x64x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:               [[RES:%.+]] = VPU.Copy([[AND]]

    // CHECK:               return [[RES]] : tensor<1x64x32x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @SinSWWithSOH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x44x44xf16>
func.func @SinSWWithSOH(%arg0: tensor<1x32x44x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Sin(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16>
    return %0 : tensor<1x32x44x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:        [[SIN:%.+]] = VPU.Sin([[IN]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[SIN]]
    // CHECK:        return [[OUT]] : tensor<1x32x44x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @SinSWWithSOK
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x1x44xf16>
func.func @SinSWWithSOK(%arg0: tensor<1x32x1x44xf16>) -> tensor<1x32x1x44xf16> {
    %0 = VPU.Sin(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x44xf16> -> tensor<1x32x1x44xf16>
    return %0 : tensor<1x32x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:        [[SIN:%.+]] = VPU.Sin([[IN]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[SIN]]
    // CHECK:        return [[OUT]] : tensor<1x32x1x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @SinSWWithClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x44xf16>
func.func @SinSWWithClustering(%arg0: tensor<1x1x1x44xf16>) -> tensor<1x1x1x44xf16> {
    %0 = VPU.Sin(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16>
    return %0 : tensor<1x1x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[SIN:%.+]] = VPU.Sin([[IN]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[SIN]]
    // CHECK:        return [[OUT]] : tensor<1x1x1x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @CosSWWithSOH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x44x44xf16>
func.func @CosSWWithSOH(%arg0: tensor<1x32x44x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Cos(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16>
    return %0 : tensor<1x32x44x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:        [[COS:%.+]] = VPU.Cos([[IN]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[COS]]
    // CHECK:        return [[OUT]] : tensor<1x32x44x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @CosSWWithSOK
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x1x44xf16>
func.func @CosSWWithSOK(%arg0: tensor<1x32x1x44xf16>) -> tensor<1x32x1x44xf16> {
    %0 = VPU.Cos(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x44xf16> -> tensor<1x32x1x44xf16>
    return %0 : tensor<1x32x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:        [[COS:%.+]] = VPU.Cos([[IN]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[COS]]
    // CHECK:        return [[OUT]] : tensor<1x32x1x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @CosSWWithClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x44xf16>
func.func @CosSWWithClustering(%arg0: tensor<1x1x1x44xf16>) -> tensor<1x1x1x44xf16> {
    %0 = VPU.Cos(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16>
    return %0 : tensor<1x1x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[COS:%.+]] = VPU.Cos([[IN]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[COS]]
    // CHECK:        return [[OUT]] : tensor<1x1x1x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @ExpSWWithSOH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x44x44xf16>
func.func @ExpSWWithSOH(%arg0: tensor<1x32x44x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Exp(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16>
    return %0 : tensor<1x32x44x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:        [[EXP:%.+]] = VPU.Exp([[IN]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[EXP]]
    // CHECK:        return [[OUT]] : tensor<1x32x44x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @ExpSWWithSOK
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x1x44xf16>
func.func @ExpSWWithSOK(%arg0: tensor<1x32x1x44xf16>) -> tensor<1x32x1x44xf16> {
    %0 = VPU.Exp(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x44xf16> -> tensor<1x32x1x44xf16>
    return %0 : tensor<1x32x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:        [[EXP:%.+]] = VPU.Exp([[IN]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[EXP]]
    // CHECK:        return [[OUT]] : tensor<1x32x1x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @ExpSWWithClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x44xf16>
func.func @ExpSWWithClustering(%arg0: tensor<1x1x1x44xf16>) -> tensor<1x1x1x44xf16> {
    %0 = VPU.Exp(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16>
    return %0 : tensor<1x1x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[EXP:%.+]] = VPU.Exp([[IN]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[EXP]]
    // CHECK:        return [[OUT]] : tensor<1x1x1x44xf16>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @LogSoftmaxSWClustering
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x1x1x512xf16, {order = #NCHW}>)
func.func @LogSoftmaxSWClustering(%arg0: tensor<1x1x1x512xf16, {order = #NCHW}>) -> tensor<1x1x1x512xf16, {order = #NCHW}> {
    %0 = VPU.LogSoftmax(%arg0) {
            axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
          : tensor<1x1x1x512xf16, {order = #NCHW}> -> tensor<1x1x1x512xf16, {order = #NCHW}>

    return %0 : tensor<1x1x1x512xf16, {order = #NCHW}>

    //CHECK:        [[INPUT:%.+]] = VPU.Copy([[INPUT_DATA]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[LOG_SOFTMAX:%.+]] = VPU.LogSoftmax([[INPUT]]
    //CHECK:                            -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[LOG_SOFTMAX]]

    //CHECK:        return [[OUTPUT]] : tensor<1x1x1x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @MishSWWithSOH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x44x44xf16>
func.func @MishSWWithSOH(%arg0: tensor<1x32x44x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Mish(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16>
    return %0 : tensor<1x32x44x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:        [[MISH:%.+]] = VPU.Mish([[IN]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[MISH]]
    // CHECK:        return [[OUT]] : tensor<1x32x44x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @MishSWWithSOK
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x1x44xf16>
func.func @MishSWWithSOK(%arg0: tensor<1x32x1x44xf16>) -> tensor<1x32x1x44xf16> {
    %0 = VPU.Mish(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x44xf16> -> tensor<1x32x1x44xf16>
    return %0 : tensor<1x32x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:        [[MISH:%.+]] = VPU.Mish([[IN]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[MISH]]
    // CHECK:        return [[OUT]] : tensor<1x32x1x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @MishSWWithClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x44xf16>
func.func @MishSWWithClustering(%arg0: tensor<1x1x1x44xf16>) -> tensor<1x1x1x44xf16> {
    %0 = VPU.Mish(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16>
    return %0 : tensor<1x1x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[MISH:%.+]] = VPU.Mish([[IN]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[MISH]]
    // CHECK:        return [[OUT]] : tensor<1x1x1x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @NegativeSWWithSOH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x44x44xf16>
func.func @NegativeSWWithSOH(%arg0: tensor<1x32x44x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Negative(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16>
    return %0 : tensor<1x32x44x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:        [[NEGATIVE:%.+]] = VPU.Negative([[IN]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[NEGATIVE]]
    // CHECK:        return [[OUT]] : tensor<1x32x44x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @NegativeSWWithSOK
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x1x44xf16>
func.func @NegativeSWWithSOK(%arg0: tensor<1x32x1x44xf16>) -> tensor<1x32x1x44xf16> {
    %0 = VPU.Negative(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x44xf16> -> tensor<1x32x1x44xf16>
    return %0 : tensor<1x32x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:        [[NEGATIVE:%.+]] = VPU.Negative([[IN]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[NEGATIVE]]
    // CHECK:        return [[OUT]] : tensor<1x32x1x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @NegativeSWWithClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x44xf16>
func.func @NegativeSWWithClustering(%arg0: tensor<1x1x1x44xf16>) -> tensor<1x1x1x44xf16> {
    %0 = VPU.Negative(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16>
    return %0 : tensor<1x1x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[NEGATIVE:%.+]] = VPU.Negative([[IN]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[NEGATIVE]]
    // CHECK:        return [[OUT]] : tensor<1x1x1x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @LogicalNotSWWithSOH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x44x44xf16>
func.func @LogicalNotSWWithSOH(%arg0: tensor<1x32x44x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.LogicalNot(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16>
    return %0 : tensor<1x32x44x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:        [[LOGICALNOT:%.+]] = VPU.LogicalNot([[IN]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[LOGICALNOT]]
    // CHECK:        return [[OUT]] : tensor<1x32x44x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @LogicalNotSWWithSOK
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x1x44xf16>
func.func @LogicalNotSWWithSOK(%arg0: tensor<1x32x1x44xf16>) -> tensor<1x32x1x44xf16> {
    %0 = VPU.LogicalNot(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x44xf16> -> tensor<1x32x1x44xf16>
    return %0 : tensor<1x32x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:        [[LOGICALNOT:%.+]] = VPU.LogicalNot([[IN]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[LOGICALNOT]]
    // CHECK:        return [[OUT]] : tensor<1x32x1x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @LogicalNotSWWithClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x44xf16>
func.func @LogicalNotSWWithClustering(%arg0: tensor<1x1x1x44xf16>) -> tensor<1x1x1x44xf16> {
    %0 = VPU.LogicalNot(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16>
    return %0 : tensor<1x1x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[LOGICALNOT:%.+]] = VPU.LogicalNot([[IN]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[LOGICALNOT]]
    // CHECK:        return [[OUT]] : tensor<1x1x1x44xf16>
}

// -----

// CHECK-LABEL:   @GatherDMA
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x128256x2048xf16>,
// CHECK-SAME:    [[INDICES:%.+]]: tensor<1x1x1024x1xi64>
func.func @GatherDMA(%input: tensor<1x1x128256x2048xf16>, %indices: tensor<1x1x1024x1xi64>) -> tensor<1x1x1024x2048xf16> {

    %gatherDMA = VPU.GatherDMA(%input, %indices) {axis_value = 2 : i64, batch_dims = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>} :
                tensor<1x1x128256x2048xf16>, tensor<1x1x1024x1xi64> -> tensor<1x1x1024x2048xf16>
    return %gatherDMA : tensor<1x1x1024x2048xf16>

    // CHECK:        [[INDICES_COPY:%.+]] = VPU.Copy([[INDICES]]) {out_mem_space = @CMX_NN} : tensor<1x1x1024x1xi64>
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x1x1024x1xi64, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[GATHER_DMA:%.+]] = VPU.GatherDMA([[INPUT]], [[INDICES_COPY]]) {axis_value = 2 : i64, batch_dims = 1 : i64} :
    // CHECK-SAME:      tensor<1x1x128256x2048xf16>, !VPU.DistributedTensor<1x1x1024x1xi64, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x1x1024x2048xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[GATHER_DMA]]) :
    // CHECK-SAME:      !VPU.DistributedTensor<1x1x1024x2048xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>
    // CHECK-SAME:      -> tensor<1x1x1024x2048xf16>

    // CHECK:        return [[OUT]] : tensor<1x1x1024x2048xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @ReLUSWWithSOH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x44x44xf16>
func.func @ReLUSWWithSOH(%arg0: tensor<1x32x44x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.ReLU(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16>
    return %0 : tensor<1x32x44x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:        [[RELU:%.+]] = VPU.ReLU([[IN]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[RELU]]
    // CHECK:        return [[OUT]] : tensor<1x32x44x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @ReLUSWWithSOK
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x1x44xf16>
func.func @ReLUSWWithSOK(%arg0: tensor<1x32x1x44xf16>) -> tensor<1x32x1x44xf16> {
    %0 = VPU.ReLU(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x44xf16> -> tensor<1x32x1x44xf16>
    return %0 : tensor<1x32x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:        [[RELU:%.+]] = VPU.ReLU([[IN]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[RELU]]
    // CHECK:        return [[OUT]] : tensor<1x32x1x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @ReLUSWWithClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x44xf16>
func.func @ReLUSWWithClustering(%arg0: tensor<1x1x1x44xf16>) -> tensor<1x1x1x44xf16> {
    %0 = VPU.ReLU(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16>
    return %0 : tensor<1x1x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[RELU:%.+]] = VPU.ReLU([[IN]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[RELU]]
    // CHECK:        return [[OUT]] : tensor<1x1x1x44xf16>
}
