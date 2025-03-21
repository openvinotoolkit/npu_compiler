//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --make-ops-with-distributed-tensor --make-distributed-copies --wrap-distributed-ops-in-nceclustertiling %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCEClusterTilingSOH
func.func @ConvToNCEClusterTilingSOH(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x64x28x28xf16, {order = #NHWC}> -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<80x64x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<80x64x3x3xf16, {order = #NHWC}> -> tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<80x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<80x1x1x4xsi32> -> tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [80, 64, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x80x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCEClusterTilingSOK
func.func @ConvToNCEClusterTilingSOK(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>) -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %cst_0 = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [64, 128, 1, 1], strides = [1, 1]} -> tensor<1x64x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x128x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x128x28x28xf16, {order = #NHWC}> -> tensor<1x128x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<64x128x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<64x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<64x128x1x1xf16, {order = #NHWC}> -> tensor<64x128x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<64x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<64x1x1x4xsi32> -> tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x128x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<64x128x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:                 rawFilterShape = [64, 128, 1, 1], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x64x28x28xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCEClusterTilingSOB
// CHECK-SAME:      [[INPUT:%.*]]: tensor<2x16x96x96xf16, {order = #NHWC}>
func.func @ConvToNCEClusterTilingSOB(%arg0: tensor<2x16x96x96xf16, {order = #NHWC}>) -> tensor<2x16x96x94xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %cst_0 = const.Declare tensor<16x16x3x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x5xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 5], strides = [1, 1]} -> tensor<2x16x96x94xf16, {order = #NHWC}>
    return %0 : tensor<2x16x96x94xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<16x16x3x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x5xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as %arg1: tensor<2x16x96x96xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<2x16x96x96xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<2x16x96x96xf16, {order = #NHWC}> -> tensor<2x16x96x96xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<16x16x3x5xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<16x16x3x5xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<16x16x3x5xf16, {order = #NHWC}> -> tensor<16x16x3x5xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32> -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:           [[INPUT_CMX]] as %arg1: tensor<2x16x96x96xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:           [[WEIGHTS_CMX]] as %arg2: tensor<16x16x3x5xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:           [[WEIGHTSTABLE_CMX]] as %arg3: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<2x16x96x94xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:               pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:               rawFilterShape = [16, 16, 3, 5], strides = [1, 1]
    //CHECK-SAME:           } -> tensor<2x16x96x94xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<2x16x96x94xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<2x16x96x94xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<2x16x96x94xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<2x16x96x94xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<2x16x96x94xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCEClusterTilingClustering
func.func @ConvToNCEClusterTilingClustering(%arg0: tensor<1x64x14x14xf16, {order = #NHWC}>) -> tensor<1x48x14x14xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    %cst_0 = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [48, 64, 3, 3], strides = [1, 1]} -> tensor<1x48x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x48x14x14xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x64x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x64x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x64x14x14xf16, {order = #NHWC}> -> tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<48x64x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<48x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<48x64x3x3xf16, {order = #NHWC}> -> tensor<48x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<48x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<48x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<48x1x1x4xsi32> -> tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<48x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x48x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [48, 64, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x48x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x48x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x48x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy(%arg1) : tensor<1x48x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x48x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x48x14x14xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvToNCEClusterTilingSOH
func.func @DepthConvToNCEClusterTilingSOH(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x112x112xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x112x112xf16, {order = #NHWC}> -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<32x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x16x1x1xf16, {order = #NHWC}> -> tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<32x1x1x4xsi32>) -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32> -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES4:%.*]] = VPU.NCE.DepthConvolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:               pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,
    //CHECK-SAME:               rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    //CHECK:            [[RES5:%.*]] = VPU.Copy(%arg1) : tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES5:%.*]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----


#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvToNCEClusterTilingSOHWithAlign
func.func @DepthConvToNCEClusterTilingSOHWithAlign(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x14x14xf16, {order = #NHWC}> -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<32x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x16x1x1xf16, {order = #NHWC}> -> tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<32x1x1x4xsi32>) -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32> -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES4:%.*]] = VPU.NCE.DepthConvolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:               pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,
    //CHECK-SAME:               rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES5:%.*]] = VPU.Copy(%arg1) : tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES5:%.*]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DepthConvToNCEClusterTilingSOK
func.func @DepthConvToNCEClusterTilingSOK(%arg0: tensor<1x128x56x56xf16, {order = #NHWC}>) -> tensor<1x128x56x56xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    %cst_1 = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [128, 1, 3, 3], strides = [1, 1]} -> tensor<1x128x56x56xf16, {order = #NHWC}>
    return %0 : tensor<1x128x56x56xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<128x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x128x56x56xf16, {order = #NHWC}>)
    //CHECK-SAME:   !VPU.DistributedTensor<1x128x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x128x56x56xf16, {order = #NHWC}> -> tensor<1x128x56x56xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<128x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<128x16x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<128x16x1x1xf16, {order = #NHWC}> -> tensor<128x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<128x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<128x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<128x1x1x4xsi32> -> tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x128x56x56xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<128x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x128x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES4:%.*]] = VPU.NCE.DepthConvolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [128, 1, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x128x56x56xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x128x56x56xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x128x56x56xf16, {order = #NHWC}> {
    //CHECK:            [[RES5:%.*]] = VPU.Copy(%arg1) : tensor<1x128x56x56xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x128x56x56xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES5]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x128x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DepthConvToNCEClusterTilingClustering
func.func @DepthConvToNCEClusterTilingClustering(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x14x14xf16, {order = #NHWC}> -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<32x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x16x1x1xf16, {order = #NHWC}> -> tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<32x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32> -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES4:%.*]] = VPU.NCE.DepthConvolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [32, 1, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES5:%.*]] = VPU.Copy(%arg1) : tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES5]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolToNCEClusterTilingSOH
func.func @MaxPoolToNCEClusterTilingSOH(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x112x112xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:        [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x112x112xf16, {order = #NHWC}> -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        VPU.Yield [[RES0]]

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.MaxPool(%arg1) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolToNCEClusterTilingSOHWithAlign
func.func @MaxPoolToNCEClusterTilingSOHWithAlign(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:        [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x14x14xf16, {order = #NHWC}> -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        VPU.Yield [[RES0]]

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.MaxPool(%arg1) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolToNCEClusterTilingClustering
func.func @MaxPoolToNCEClusterTilingClustering(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:        [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x14x14xf16, {order = #NHWC}> -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        VPU.Yield [[RES0]]

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.MaxPool(%arg1) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  func.func @MaxPoolToNCEClusterTilingSOB
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<2x32x112x112xf16, {order = #NHWC}>)
func.func @MaxPoolToNCEClusterTilingSOB(%input: tensor<2x32x112x112xf16, {order = #NHWC}>) -> tensor<2x32x112x112xf16, {order = #NHWC}> {
    %maxpool = VPU.NCE.MaxPool(%input) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        kernel_size = [1, 1]
    } -> tensor<2x32x112x112xf16, {order = #NHWC}>
    return %maxpool : tensor<2x32x112x112xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_CMX:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[INPUT_ARG:%.+]]: tensor<2x32x112x112xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<2x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}> {
    // CHECK:           [[RES0:%.+]] = VPU.Copy([[INPUT_ARG]]) {out_mem_space = @CMX_NN} : tensor<2x32x112x112xf16, {order = #NHWC}> -> tensor<2x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:           VPU.Yield [[RES0]]
    // CHECK:       }

    // CHECK:       [[OUT_CMX:%.+]] = VPU.NCE.ClusterTiling ([[INPUT_CMX]] as [[INPUT_CMX_ARG:%.+]]: tensor<2x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<2x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}> {
    // CHECK:           [[RES1:%.+]] = VPU.NCE.MaxPool([[INPUT_CMX_ARG]]) {
    // CHECK-SAME:          kernel_size = [1, 1],
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEStub<>,
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      } -> tensor<2x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:           VPU.Yield [[RES1]]
    // CHECK:       }

    // CHECK:       [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[OUT_CMX_ARG:%.+]]: tensor<2x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<2x32x112x112xf16, {order = #NHWC}> {
    // CHECK:           [[RES2:%.+]] = VPU.Copy([[OUT_CMX_ARG]]) : tensor<2x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<2x32x112x112xf16, {order = #NHWC}>
    // CHECK:           VPU.Yield [[RES2]]
    // CHECK:       }

    // CHECK:       return [[OUT]] : tensor<2x32x112x112xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @ReduceL1SplitOverKernel(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1024x1x1xf16> {
func.func @ReduceL1SplitOverKernel(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1024x1x1xf16> {
  %0 = VPU.ReduceL1(%arg0) {axes_value = [2, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x1024x7x7xf16> -> tensor<1x1024x1x1xf16>
  return %0 : tensor<1x1024x1x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.NCE.ClusterTiling (%[[VAL_0]] as %[[VAL_2:.*]]: tensor<1x1024x7x7xf16>) -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    // CHECK:     %[[VAL_3:.*]] = VPU.Copy(%[[VAL_2]]) {out_mem_space = @CMX_NN} : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_4:.*]] = VPU.NCE.ClusterTiling (%[[VAL_1]] as %[[VAL_5:.*]]: tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1024x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    // CHECK:     %[[VAL_6:.*]] = VPU.ReduceL1(%[[VAL_5]]) {axes_value = [2, 3], keep_dims} : tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1024x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_7:.*]] = VPU.NCE.ClusterTiling (%[[VAL_4]] as %[[VAL_8:.*]]: tensor<1x1024x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1024x1x1xf16> {
    // CHECK:     %[[VAL_9:.*]] = VPU.Copy(%[[VAL_8]]) : tensor<1x1024x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1024x1x1xf16>
    // CHECK:   return %[[VAL_7]] : tensor<1x1024x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @ReduceL2SplitOverKernel(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1024x1x1xf16> {
func.func @ReduceL2SplitOverKernel(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1024x1x1xf16> {
  %0 = VPU.ReduceL2(%arg0) {axes_value = [2, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x1024x7x7xf16> -> tensor<1x1024x1x1xf16>
  return %0 : tensor<1x1024x1x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.NCE.ClusterTiling (%[[VAL_0]] as %[[VAL_2:.*]]: tensor<1x1024x7x7xf16>) -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    // CHECK:     %[[VAL_3:.*]] = VPU.Copy(%[[VAL_2]]) {out_mem_space = @CMX_NN} : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_4:.*]] = VPU.NCE.ClusterTiling (%[[VAL_1]] as %[[VAL_5:.*]]: tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1024x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    // CHECK:     %[[VAL_6:.*]] = VPU.ReduceL2(%[[VAL_5]]) {axes_value = [2, 3], keep_dims} : tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1024x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_7:.*]] = VPU.NCE.ClusterTiling (%[[VAL_4]] as %[[VAL_8:.*]]: tensor<1x1024x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1024x1x1xf16> {
    // CHECK:     %[[VAL_9:.*]] = VPU.Copy(%[[VAL_8]]) : tensor<1x1024x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1024x1x1xf16>
    // CHECK:   return %[[VAL_7]] : tensor<1x1024x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @ReduceLogicalAndClustering(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x1x1xf16> {
func.func @ReduceLogicalAndClustering(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x1x1xf16> {
  %0 = VPU.ReduceLogicalAnd(%arg0) {axes_value = [1, 2, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1024x7x7xf16> -> tensor<1x1x1x1xf16>
  return %0 : tensor<1x1x1x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.NCE.ClusterTiling (%[[VAL_0]] as %[[VAL_2:.*]]: tensor<1x1024x7x7xf16>) -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:     %[[VAL_3:.*]] = VPU.Copy(%[[VAL_2]]) {out_mem_space = @CMX_NN} : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_4:.*]] = VPU.NCE.ClusterTiling (%[[VAL_1]] as %[[VAL_5:.*]]: tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:     %[[VAL_6:.*]] = VPU.ReduceLogicalAnd(%[[VAL_5]]) {axes_value = [1, 2, 3], keep_dims} : tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_7:.*]] = VPU.NCE.ClusterTiling (%[[VAL_4]] as %[[VAL_8:.*]]: tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x1x1xf16> {
    // CHECK:     %[[VAL_9:.*]] = VPU.Copy(%[[VAL_8]]) : tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x1x1xf16>
    // CHECK:   return %[[VAL_7]] : tensor<1x1x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @ReduceLogicalOrClustering(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x1x1xf16> {
func.func @ReduceLogicalOrClustering(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x1x1xf16> {
  %0 = VPU.ReduceLogicalOr(%arg0) {axes_value = [1, 2, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1024x7x7xf16> -> tensor<1x1x1x1xf16>
  return %0 : tensor<1x1x1x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.NCE.ClusterTiling (%[[VAL_0]] as %[[VAL_2:.*]]: tensor<1x1024x7x7xf16>) -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:     %[[VAL_3:.*]] = VPU.Copy(%[[VAL_2]]) {out_mem_space = @CMX_NN} : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_4:.*]] = VPU.NCE.ClusterTiling (%[[VAL_1]] as %[[VAL_5:.*]]: tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:     %[[VAL_6:.*]] = VPU.ReduceLogicalOr(%[[VAL_5]]) {axes_value = [1, 2, 3], keep_dims} : tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_7:.*]] = VPU.NCE.ClusterTiling (%[[VAL_4]] as %[[VAL_8:.*]]: tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x1x1xf16> {
    // CHECK:     %[[VAL_9:.*]] = VPU.Copy(%[[VAL_8]]) : tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x1x1xf16>
    // CHECK:   return %[[VAL_7]] : tensor<1x1x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @ReduceMaxSplitOverHeight(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
func.func @ReduceMaxSplitOverHeight(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
  %0 = VPU.ReduceMax(%arg0) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
  return %0 : tensor<1x1x7x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.NCE.ClusterTiling (%[[VAL_0]] as %[[VAL_2:.*]]: tensor<1x1024x7x7xf16>) -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:     %[[VAL_3:.*]] = VPU.Copy(%[[VAL_2]]) {out_mem_space = @CMX_NN} : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_4:.*]] = VPU.NCE.ClusterTiling (%[[VAL_1]] as %[[VAL_5:.*]]: tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1x7x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:     %[[VAL_6:.*]] = VPU.ReduceMax(%[[VAL_5]]) {axes_value = [1, 3], keep_dims} : tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x7x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_7:.*]] = VPU.NCE.ClusterTiling (%[[VAL_4]] as %[[VAL_8:.*]]: tensor<1x1x7x1xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x7x1xf16> {
    // CHECK:     %[[VAL_9:.*]] = VPU.Copy(%[[VAL_8]]) : tensor<1x1x7x1xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x7x1xf16>
    // CHECK:   return %[[VAL_7]] : tensor<1x1x7x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @ReduceMeanSplitOverHeight(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
func.func @ReduceMeanSplitOverHeight(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
  %0 = VPU.ReduceMean(%arg0) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
  return %0 : tensor<1x1x7x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.NCE.ClusterTiling (%[[VAL_0]] as %[[VAL_2:.*]]: tensor<1x1024x7x7xf16>) -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:     %[[VAL_3:.*]] = VPU.Copy(%[[VAL_2]]) {out_mem_space = @CMX_NN} : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_4:.*]] = VPU.NCE.ClusterTiling (%[[VAL_1]] as %[[VAL_5:.*]]: tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1x7x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:     %[[VAL_6:.*]] = VPU.ReduceMean(%[[VAL_5]]) {axes_value = [1, 3], keep_dims} : tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x7x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_7:.*]] = VPU.NCE.ClusterTiling (%[[VAL_4]] as %[[VAL_8:.*]]: tensor<1x1x7x1xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x7x1xf16> {
    // CHECK:     %[[VAL_9:.*]] = VPU.Copy(%[[VAL_8]]) : tensor<1x1x7x1xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x7x1xf16>
    // CHECK:   return %[[VAL_7]] : tensor<1x1x7x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @ReduceProdSplitOverHeight(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
func.func @ReduceProdSplitOverHeight(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
  %0 = VPU.ReduceProd(%arg0) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
  return %0 : tensor<1x1x7x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.NCE.ClusterTiling (%[[VAL_0]] as %[[VAL_2:.*]]: tensor<1x1024x7x7xf16>) -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:     %[[VAL_3:.*]] = VPU.Copy(%[[VAL_2]]) {out_mem_space = @CMX_NN} : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_4:.*]] = VPU.NCE.ClusterTiling (%[[VAL_1]] as %[[VAL_5:.*]]: tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1x7x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:     %[[VAL_6:.*]] = VPU.ReduceProd(%[[VAL_5]]) {axes_value = [1, 3], keep_dims} : tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x7x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_7:.*]] = VPU.NCE.ClusterTiling (%[[VAL_4]] as %[[VAL_8:.*]]: tensor<1x1x7x1xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x7x1xf16> {
    // CHECK:     %[[VAL_9:.*]] = VPU.Copy(%[[VAL_8]]) : tensor<1x1x7x1xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x7x1xf16>
    // CHECK:   return %[[VAL_7]] : tensor<1x1x7x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @ReduceSumSplitOverHeight(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
func.func @ReduceSumSplitOverHeight(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
  %0 = VPU.ReduceSum(%arg0) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
  return %0 : tensor<1x1x7x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.NCE.ClusterTiling (%[[VAL_0]] as %[[VAL_2:.*]]: tensor<1x1024x7x7xf16>) -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:     %[[VAL_3:.*]] = VPU.Copy(%[[VAL_2]]) {out_mem_space = @CMX_NN} : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_4:.*]] = VPU.NCE.ClusterTiling (%[[VAL_1]] as %[[VAL_5:.*]]: tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1x7x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:     %[[VAL_6:.*]] = VPU.ReduceSum(%[[VAL_5]]) {axes_value = [1, 3], keep_dims} : tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x7x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_7:.*]] = VPU.NCE.ClusterTiling (%[[VAL_4]] as %[[VAL_8:.*]]: tensor<1x1x7x1xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x7x1xf16> {
    // CHECK:     %[[VAL_9:.*]] = VPU.Copy(%[[VAL_8]]) : tensor<1x1x7x1xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x7x1xf16>
    // CHECK:   return %[[VAL_7]] : tensor<1x1x7x1xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddToNCEClusterTilingSOH
func.func @EltwiseAddToNCEClusterTilingSOH(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>, %arg1: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) { multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, op_type = #VPU.eltwise_type<ADD> } :
         tensor<1x32x112x112xf16, {order = #NHWC}>, tensor<1x32x112x112xf16, {order = #NHWC}>
         -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0: tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[INPUT0_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg2: tensor<1x32x112x112xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x112x112xf16, {order = #NHWC}> -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[INPUT1_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg1 as %arg2: tensor<1x32x112x112xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x112x112xf16, {order = #NHWC}> -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0_CMX]] as %arg2: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[INPUT1_CMX]] as %arg3: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Eltwise(%arg2, %arg3) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg2: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy(%arg2) : tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }
    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddToNCEClusterTilingClustering
func.func @EltwiseAddToNCEClusterTilingClustering(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>, %arg1: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) { multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, ppe = #VPU.PPEStub<>, op_type = #VPU.eltwise_type<ADD> } :
         tensor<1x32x14x14xf16, {order = #NHWC}>, tensor<1x32x14x14xf16, {order = #NHWC}>
         -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0: tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[INPUT0_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg2: tensor<1x32x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x14x14xf16, {order = #NHWC}> -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[INPUT1_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg1 as %arg2: tensor<1x32x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x14x14xf16, {order = #NHWC}> -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0_CMX]] as %arg2: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[INPUT1_CMX]] as %arg3: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Eltwise(%arg2, %arg3) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg2: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy(%arg2) : tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AvgPoolToNCEClusterTilingSOH
func.func @AvgPoolToNCEClusterTilingSOH(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [3, 3]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x112x112xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:        [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x112x112xf16, {order = #NHWC}> -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        VPU.Yield [[RES0]]

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.AveragePool(%arg1) {kernel_size = [3, 3], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AvgPoolToNCEClusterTilingSOHWithAlign
func.func @AvgPoolToNCEClusterTilingSOHWithAlign(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [3, 3]
         } -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:        [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x14x14xf16, {order = #NHWC}> -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        VPU.Yield [[RES0]]

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.AveragePool(%arg1) {kernel_size = [3, 3], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  func.func @AvgPoolToNCEClusterTilingSOB
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<2x32x112x112xf16, {order = #NHWC}>)
func.func @AvgPoolToNCEClusterTilingSOB(%input: tensor<2x32x112x112xf16, {order = #NHWC}>) -> tensor<2x32x112x112xf16, {order = #NHWC}> {
    %avgpool = VPU.NCE.AveragePool(%input) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        kernel_size = [1, 1]
    } -> tensor<2x32x112x112xf16, {order = #NHWC}>
    return %avgpool : tensor<2x32x112x112xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_CMX:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[INPUT_ARG:%.+]]: tensor<2x32x112x112xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<2x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}> {
    // CHECK:           [[RES0:%.+]] = VPU.Copy([[INPUT_ARG]]) {out_mem_space = @CMX_NN} : tensor<2x32x112x112xf16, {order = #NHWC}> -> tensor<2x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:           VPU.Yield [[RES0]]
    // CHECK:       }

    // CHECK:       [[OUT_CMX:%.+]] = VPU.NCE.ClusterTiling ([[INPUT_CMX]] as [[INPUT_CMX_ARG:%.+]]: tensor<2x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<2x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}> {
    // CHECK:           [[RES1:%.+]] = VPU.NCE.AveragePool([[INPUT_CMX_ARG]]) {
    // CHECK-SAME:          kernel_size = [1, 1],
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEStub<>,
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      } -> tensor<2x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:           VPU.Yield [[RES1]]
    // CHECK:       }

    // CHECK:       [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[OUT_CMX_ARG:%.+]]: tensor<2x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<2x32x112x112xf16, {order = #NHWC}> {
    // CHECK:           [[RES2:%.+]] = VPU.Copy([[OUT_CMX_ARG]]) : tensor<2x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<2x32x112x112xf16, {order = #NHWC}>
    // CHECK:           VPU.Yield [[RES2]]
    // CHECK:       }

    // CHECK:       return [[OUT]] : tensor<2x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SparseConvToNCEClusterTilingSOH
func.func @SparseConvToNCEClusterTilingSOH(%arg0 : tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1 : tensor<1x64x28x28xi1, {order = #NHWC}>)
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
    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<80x1x1x640xi1> = dense<1.000000e+00>
    // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]]) {is_weights}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<80x1x1x640xi1>, is_weights>

    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<80x1x1x4xsi32>

    // CHECK:       [[INPUT_SPARSE_CMX:%.+]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      ([[INPUT_SPARSE]] as [[INPUT_SPARSE_ARG:%.+]]: !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                                                                       sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                           sparsity_map=!VPU.DistributedTensor<1x64x28x28xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>> {
    // CHECK:         [[VAR0:%.+]] = VPU.Copy([[INPUT_SPARSE_ARG]]) {out_mem_space = @CMX_NN}
    // CHECK:         VPU.Yield [[VAR0]]
    // CHECK:       }

    // CHECK:       [[WEIGHTS_SPARSE_CMX:%.+]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      ([[WEIGHTS_SPARSE]] as [[WEIGHTS_SPARSE_ARG:%.+]]: !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                                                                           sparsity_map=tensor<80x1x1x640xi1>, is_weights>)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=!VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
    // CHECK-SAME:                           sparsity_map=!VPU.DistributedTensor<80x1x1x640xi1, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
    // CHECK-SAME:                           is_weights> {
    // CHECK:         [[VAR1:%.+]] = VPU.Copy([[WEIGHTS_SPARSE_ARG]]) {out_mem_space = @CMX_NN}
    // CHECK:         VPU.Yield [[VAR1]]
    // CHECK:       }

    // CHECK:       [[WEIGHTS_TABLE_CMX:%.+]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      ([[CST_WEIGHTS_TABLE]] as [[WEIGHTS_TABLE_ARG:%.+]]: tensor<80x1x1x4xsi32>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:         [[VAR2:%.+]] = VPU.Copy([[WEIGHTS_TABLE_ARG]]) {out_mem_space = @CMX_NN}
    // CHECK:         VPU.Yield [[VAR2]]
    // CHECK:       }

    // CHECK:       [[OUT_CMX:%.+]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:      [[INPUT_SPARSE_CMX]] as [[INPUT_SPARSE_CMX_ARG:[^:]+]]:
    // CHECK-SAME:          !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                            sparsity_map=tensor<1x64x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>,
    // CHECK-SAME:      [[WEIGHTS_SPARSE_CMX]] as [[WEIGHTS_SPARSE_CMX_ARG:[^:]+]]:
    // CHECK-SAME:          !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                            sparsity_map=tensor<80x1x1x640xi1, {mem_space = @CMX_NN, order = #NCHW}>, is_weights>,
    // CHECK-SAME:      [[WEIGHTS_TABLE_CMX]] as [[WEIGHTS_TABLE_CMX_ARG:[^:]+]]: tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                           sparsity_map=!VPU.DistributedTensor<1x80x28x28xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>> {
    // CHECK:         [[VAR3:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE_CMX_ARG]], [[WEIGHTS_SPARSE_CMX_ARG]], [[WEIGHTS_TABLE_CMX_ARG]])
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x80x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>
    // CHECK:         VPU.Yield [[VAR3]]
    // CHECK:       }

    // CHECK:       [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[OUT_CMX_ARG:[^:]+]]:
    // CHECK-SAME:      !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                        sparsity_map=tensor<1x80x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {
    // CHECK:         [[VAR4:%.+]] = VPU.Copy([[OUT_CMX_ARG]])
    // CHECK:         VPU.Yield [[VAR4]]
    // CHECK:       }

    // CHECK:       return [[OUT]] : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                                     sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSOHAlignmentForEltwiseCase1
func.func @OptimizeSOHAlignmentForEltwiseCase1(%arg0: tensor<1x16x22x22xf16, {order = #NHWC}>, %arg1: tensor<1x16x22x22xf16, {order = #NHWC}>) -> tensor<1x16x22x22xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %cst_0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} -> tensor<1x16x22x22xf16, {order = #NHWC}>
    %1 = VPU.NCE.Eltwise(%arg0, %arg1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, op_type = #VPU.eltwise_type<ADD>} -> tensor<1x16x22x22xf16, {order = #NHWC}>
    %2 = VPU.NCE.Eltwise(%0, %1) {ppe = #VPU.PPEStub<>, op_type = #VPU.eltwise_type<ADD>} -> tensor<1x16x22x22xf16, {order = #NHWC}>
    return %2 : tensor<1x16x22x22xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX_0:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg2: tensor<1x16x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x16x22x22xf16, {order = #NHWC}> -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg2: tensor<16x16x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<16x16x3x3xf16, {order = #NHWC}> -> tensor<16x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg2: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32> -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX_0:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX_0]] as %arg2: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg3: tensor<16x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg4: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg2, %arg3, %arg4) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [16, 16, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_0:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX_0]] as %arg2: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x22x22xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg2) : tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x22x22xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[INPUT0_CMX_1:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg2: tensor<1x16x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x16x22x22xf16, {order = #NHWC}> -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[INPUT1_CMX_1:%.*]] = VPU.NCE.ClusterTiling (%arg1 as %arg2: tensor<1x16x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x16x22x22xf16, {order = #NHWC}> -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX_1:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0_CMX_1]] as %arg2: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[INPUT1_CMX_1]] as %arg3: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Eltwise(%arg2, %arg3) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX_1]] as %arg2: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x22x22xf16, {order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy(%arg2) : tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x22x22xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_2:%.*]] = VPU.NCE.Eltwise([[OUT_0]], [[OUT_1]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x16x22x22xf16, {order = #NHWC}>

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
    %2 = VPU.NCE.Convolution(%1, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} -> tensor<1x16x22x22xf16, {order = #NHWC}>
    return %2 : tensor<1x16x22x22xf16, {order = #NHWC}>

    //CHECK:        [[INPUT0_CMX_0:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg2: tensor<1x16x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x16x22x22xf16, {order = #NHWC}> -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[INPUT1_CMX_0:%.*]] = VPU.NCE.ClusterTiling (%arg1 as %arg2: tensor<1x16x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x16x22x22xf16, {order = #NHWC}> -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX_0:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0_CMX_0]] as %arg2: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[INPUT1_CMX_0]] as %arg3: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Eltwise(%arg2, %arg3) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_0:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX_0]] as %arg2: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x22x22xf16, {order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy(%arg2) : tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x22x22xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[INPUT0_CMX_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0]] as %arg2: tensor<1x16x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x16x22x22xf16, {order = #NHWC}> -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[INPUT1_CMX_1:%.*]] = VPU.NCE.ClusterTiling (%arg1 as %arg2: tensor<1x16x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x16x22x22xf16, {order = #NHWC}> -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX_1:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0_CMX_1]] as %arg2: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[INPUT1_CMX_1]] as %arg3: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Eltwise(%arg2, %arg3) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX_1]] as %arg2: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x22x22xf16, {order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy(%arg2) : tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x22x22xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX_2:%.*]] = VPU.NCE.ClusterTiling ([[OUT_1]] as %arg2: tensor<1x16x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x16x22x22xf16, {order = #NHWC}> -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg2: tensor<16x16x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<16x16x3x3xf16, {order = #NHWC}> -> tensor<16x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg2: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32> -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX_2:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX_2]] as %arg2: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg3: tensor<16x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg4: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg2, %arg3, %arg4) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [16, 16, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_2:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX_2]] as %arg2: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x22x22xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg2) : tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x22x22xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT_2]] : tensor<1x16x22x22xf16, {order = #NHWC}>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @MVNToNCEClusterTilingDuplicateBuffer
func.func @MVNToNCEClusterTilingDuplicateBuffer(%arg0: tensor<1x4x512x1xf16, {order = #NCWH}>) -> tensor<1x4x512x1xf16, {order = #NCWH}> {

    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true} : tensor<1x4x512x1xf16, {order = #NCWH}> -> tensor<1x4x512x1xf16, {order = #NCWH}>

    return %0: tensor<1x4x512x1xf16, {order = #NCWH}>

    //CHECK: [[ClusterCopy:%.*]] = VPU.NCE.ClusterTiling (%arg0 as [[ARG1:%.*]]: tensor<1x4x512x1xf16, {order = #NCWH}>) -> !VPU.DistributedTensor<1x4x512x1xf16, #NCWH, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK: [[RCopy:%*]] = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x4x512x1xf16, {order = #NCWH}> -> tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>
    //CHECK: VPU.Yield [[RCopy]]

    //CHECK: [[RClusterMVN:%.*]] = VPU.NCE.ClusterTiling ([[ClusterCopy]] as [[ARG2:%.*]]: tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>) -> !VPU.DistributedTensor<1x4x512x1xf16, #NCWH, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK: [[RMVN:%*]] = VPU.MVN([[ARG2]]) {across_channels = false, eps = 1.0013580322265625E-5 : f64, normalize_variance = true} : tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}> -> tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>
    //CHECK: VPU.Yield [[RMVN]]

    //CHECK: [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[RClusterMVN]] as [[ARG3:%.*]]: tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>) -> tensor<1x4x512x1xf16, {order = #NCWH}> {
    //CHECK: [[RCopy1:%*]] =  VPU.Copy([[ARG3]]) : tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}> -> tensor<1x4x512x1xf16, {order = #NCWH}>
    //CHECK: VPU.Yield [[RCopy1]]

    //CHECK: return [[OUT]] : tensor<1x4x512x1xf16, {order = #NCWH}>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @MVNToNCEClusterTilingSegmentedBuffer
func.func @MVNToNCEClusterTilingSegmentedBuffer(%arg0: tensor<1x4x512x1xf16, {order = #NCWH}>) -> tensor<1x4x512x1xf16, {order = #NCWH}> {

    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x4x512x1xf16, {order = #NCWH}> -> tensor<1x4x512x1xf16, {order = #NCWH}>

    return %0: tensor<1x4x512x1xf16, {order = #NCWH}>

    //CHECK: [[ClusterCopy:%.*]] = VPU.NCE.ClusterTiling (%arg0 as [[ARG1:%.*]]: tensor<1x4x512x1xf16, {order = #NCWH}>) -> !VPU.DistributedTensor<1x4x512x1xf16, #NCWH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK: [[RCopy:%*]] = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x4x512x1xf16, {order = #NCWH}> -> tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>
    //CHECK: VPU.Yield [[RCopy]]

    //CHECK: [[RClusterMVN:%.*]] = VPU.NCE.ClusterTiling ([[ClusterCopy]] as [[ARG2:%.*]]: tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>) -> !VPU.DistributedTensor<1x4x512x1xf16, #NCWH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK: [[RMVN:%*]] = VPU.MVN([[ARG2]]) {across_channels = false, eps = 1.0013580322265625E-5 : f64, normalize_variance = true} : tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}> -> tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>
    //CHECK: VPU.Yield [[RMVN]]

    //CHECK: [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[RClusterMVN]] as [[ARG3:%.*]]: tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>) -> tensor<1x4x512x1xf16, {order = #NCWH}> {
    //CHECK: [[RCopy1:%*]] =  VPU.Copy([[ARG3]]) : tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}> -> tensor<1x4x512x1xf16, {order = #NCWH}>
    //CHECK: VPU.Yield [[RCopy1]]

    //CHECK: return [[OUT]] : tensor<1x4x512x1xf16, {order = #NCWH}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @MVN6SOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x32x15x64xf16>)
func.func @MVN6SOK(%arg0: tensor<1x32x15x64xf16>) -> tensor<1x32x15x64xf16> {
    %0 = VPU.MVN6(%arg0) {axes = [2], eps = 1.000000e-02 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>} : tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16>
    return %0 : tensor<1x32x15x64xf16>

    //CHECK:        [[INPUT:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG1:%.+]]: tensor<1x32x15x64xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x15x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64
    //CHECK:         [[INNER_COPY:%.*]] = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[MVN:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG2:%.+]]: tensor<1x32x15x64xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:                   -> !VPU.DistributedTensor<1x32x15x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64
    //CHECK:         [[INNER_MVN:%.*]] = VPU.MVN6([[ARG2]]) {axes = [2], eps = 1.000000e-02 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>} : tensor<1x32x15x64xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x15x64xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[MVN]] as [[ARG3:%.+]]: tensor<1x32x15x64xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x15x64xf16>
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG3]]) : tensor<1x32x15x64xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x15x64xf16>

    //CHECK:        return [[OUTPUT]] : tensor<1x32x15x64xf16>
}

// -----

// CHECK-LABEL: func.func @MVN6SOH
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x32x15x64xf16>)
func.func @MVN6SOH(%arg0: tensor<1x32x15x64xf16>) -> tensor<1x32x15x64xf16> {
    %0 = VPU.MVN6(%arg0) {axes = [1, 3], eps = 1.000000e-02 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>} : tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16>
    return %0 : tensor<1x32x15x64xf16>

    //CHECK:        [[INPUT:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG1:%.+]]: tensor<1x32x15x64xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x15x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    //CHECK:         [[INNER_COPY:%.*]] = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[MVN:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG2:%.+]]: tensor<1x32x15x64xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:                   -> !VPU.DistributedTensor<1x32x15x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    //CHECK:         [[INNER_MVN:%.*]] = VPU.MVN6([[ARG2]]) {axes = [1, 3], eps = 1.000000e-02 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>} : tensor<1x32x15x64xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x15x64xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[MVN]] as [[ARG3:%.+]]: tensor<1x32x15x64xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x15x64xf16>
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG3]]) : tensor<1x32x15x64xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x15x64xf16>

    //CHECK:        return [[OUTPUT]] : tensor<1x32x15x64xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @PadSwSOH
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x20x50xf16>)
func.func @PadSwSOH(%arg0: tensor<1x16x20x50xf16>) -> tensor<1x18x20x60xf16> {
    %0 = VPU.Pad(%arg0) {mode = #IE.pad_mode<EDGE>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 2, 0, 10]} : tensor<1x16x20x50xf16> -> tensor<1x18x20x60xf16>
    return %0 : tensor<1x18x20x60xf16>

    //CHECK:        [[INPUT:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG1:%.+]]: tensor<1x16x20x50xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x20x50xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:         [[INNER_COPY:%.*]] = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x16x20x50xf16> -> tensor<1x16x20x50xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:        }

    //CHECK:        [[PAD:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG2:%.+]]: tensor<1x16x20x50xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x18x20x60xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_PAD:%.*]] = VPU.Pad([[ARG2]]) {mode = #IE.pad_mode<EDGE>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 2, 0, 10]} : tensor<1x16x20x50xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x18x20x60xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:        }

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[PAD]] as [[ARG3:%.+]]: tensor<1x18x20x60xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x18x20x60xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG3]]) : tensor<1x18x20x60xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x18x20x60xf16>
    //CHECK:        }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @PadSwSOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x20x50xf16>)
func.func @PadSwSOK(%arg0: tensor<1x16x20x50xf16>) -> tensor<1x16x22x52xf16> {
    %0 = VPU.Pad(%arg0) {mode = #IE.pad_mode<EDGE>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 0, 2, 2]} : tensor<1x16x20x50xf16> -> tensor<1x16x22x52xf16>
    return %0 : tensor<1x16x22x52xf16>

    //CHECK:        [[INPUT:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG1:%.+]]: tensor<1x16x20x50xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x20x50xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:         [[INNER_COPY:%.*]] = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x16x20x50xf16> -> tensor<1x16x20x50xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:        }

    //CHECK:        [[PAD:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG2:%.+]]: tensor<1x16x20x50xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x22x52xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_PAD:%.*]] = VPU.Pad([[ARG2]]) {mode = #IE.pad_mode<EDGE>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 0, 2, 2]} : tensor<1x16x20x50xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x22x52xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:        }

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[PAD]] as [[ARG3:%.+]]: tensor<1x16x22x52xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x16x22x52xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG3]]) : tensor<1x16x22x52xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x22x52xf16>
    //CHECK:        }
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
                -> tensor<1x64x64x64xf16, {order = #NHWC}>
    %mvn = VPU.MVN(%conv) {
            across_channels = false, eps = 1.0E-4 : f64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true
            } : tensor<1x64x64x64xf16, {order = #NHWC}>
                -> tensor<1x64x64x64xf16, {order = #NHWC}>

    return %mvn : tensor<1x64x64x64xf16, {order = #NHWC}>

    // (DUP) CONV (SEG) -> (SEG) MVN (SEG)

    //CHECK:        [[CONV_IN:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x64x64x64xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {

    //CHECK:        [[SOK_CONV:%.*]] = VPU.NCE.ClusterTiling ([[CONV_IN]] as %arg1: tensor<1x64x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>, %1 as %arg2: tensor<64x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, %2 as %arg3: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[MVN_IN:%.*]] = VPU.NCE.ClusterTiling (%4 as %arg1: tensor<1x64x64x64xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {

    //CHECK:        [[SOK_MVN:%.*]] = VPU.NCE.ClusterTiling (%5 as %arg1: tensor<1x64x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
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
               -> tensor<1x384x1x3000xf16, {order = #NHWC}>
    %slice = VPU.Slice %conv [0, 0, 0, 0] [1, 384, 1, 1500] : tensor<1x384x1x3000xf16, {order = #NHWC}> to tensor<1x384x1x1500xf16, {order = #NHWC}>
    %gelu =  VPU.Gelu(%slice) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x384x1x1500xf16, {order = #NHWC}> -> tensor<1x384x1x1500xf16, {order = #NHWC}>

    return %gelu : tensor<1x384x1x1500xf16, {order = #NHWC}>

    // (DUP) CONV (SEG) -> SLICE -> (SEG) GELU (SEG)

    // CHECK:     [[CONV_IN:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x80x1x3000xf16, {order = #NHWC}>)
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x80x1x3000xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:          [[CONV_IN_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x80x1x3000xf16, {order = #NHWC}> -> tensor<1x80x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:     [[SOK_CONV:%.*]] = VPU.NCE.ClusterTiling ([[CONV_IN]] as %arg1: tensor<1x80x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}>, %1 as %arg2: tensor<384x80x1x3xf16, {mem_space = @CMX_NN, order = #NHWC}>, %2 as %arg3: tensor<384x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:   -> !VPU.DistributedTensor<1x384x1x3000xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:        [[SOK_CONV_INNER:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [384, 80, 1, 3], strides = [1, 1]} -> tensor<1x384x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:     [[CONV_OUT:%.*]]  = VPU.NCE.ClusterTiling ([[SOK_CONV]] as %arg1: tensor<1x384x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x384x1x3000xf16, {order = #NHWC}> {
    // CHECK:       [[CONV_OUT_INNER:%.*]]  = VPU.Copy(%arg1) : tensor<1x384x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x384x1x3000xf16, {order = #NHWC}>

    // CHECK:     [[SLICE:%.*]] = VPU.Slice [[CONV_OUT]] [0, 0, 0, 0] [1, 384, 1, 1500] : tensor<1x384x1x3000xf16, {order = #NHWC}> to tensor<1x384x1x1500xf16, {order = #NHWC}>

    // CHECK:     [[GELU_IN:%.*]] = VPU.NCE.ClusterTiling ([[SLICE]] as %arg1: tensor<1x384x1x1500xf16, {order = #NHWC}>)
    // CHECK-SAME: -> !VPU.DistributedTensor<1x384x1x1500xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:        [[GELU_IN_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x384x1x1500xf16, {order = #NHWC}> -> tensor<1x384x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:     [[SOK_GELU:%.*]] = VPU.NCE.ClusterTiling ([[GELU_IN]] as %arg1: tensor<1x384x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME: -> !VPU.DistributedTensor<1x384x1x1500xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:       [[SOK_GELU_INNER:%.*]] = VPU.Gelu(%arg1) : tensor<1x384x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x384x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}>
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

    //CHECK:        [[MVN_IN:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x64x64x64xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {

    //CHECK:        [[SOK_MVN:%.*]] = VPU.NCE.ClusterTiling ([[MVN_IN]] as %arg1: tensor<1x64x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {

    //CHECK:        [[DWCONV_IN:%.*]] = VPU.NCE.ClusterTiling (%2 as %arg1: tensor<1x64x64x64xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {

    //CHECK:        [[SOK_DWCONV:%.*]] = VPU.NCE.ClusterTiling (%3 as %arg1: tensor<1x64x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>, %4 as %arg2: tensor<64x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, %5 as %arg3: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
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

    //CHECK:        [[MVN_COPY_IN:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x1x320x1xf16>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x1x320x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:                    VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x1x320x1xf16>
    //CHECK-SAME:                   -> tensor<1x1x320x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[MVN:%.*]] = VPU.NCE.ClusterTiling ([[MVN_COPY_IN]] as %arg1: tensor<1x1x320x1xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x1x320x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:                    VPU.MVN(%arg1) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x1x320x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-SAME:                   -> tensor<1x1x320x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[MVN_COPY_OUT:%.*]] = VPU.NCE.ClusterTiling ([[MVN]] as %arg1: tensor<1x1x320x1xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> tensor<1x1x320x1xf16> {
    //CHECK:                    VPU.Copy(%arg1) : tensor<1x1x320x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-SAME:                   -> tensor<1x1x320x1xf16>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[RESHAPE:%.*]] = VPU.AffineReshape([[MVN_COPY_OUT]])
    //CHECK-SAME{{LITERAL}}:    {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 320, 1, 1]} : tensor<1x1x320x1xf16> -> tensor<1x320x1x1xf16>
    //CHECK:        [[CAST:%.*]] = VPU.PermuteCast([[RESHAPE]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x320x1x1xf16> -> tensor<1x320x1x1xf16, {order = #NHWC}>

    //CHECK:        [[DWCONV_INPUT_COPY_IN:%.*]] = VPU.NCE.ClusterTiling ([[CAST]] as %arg1: tensor<1x320x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:                    VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x320x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:                   -> tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[DWCONV_WEIGHTS_COPY_IN:%.*]] = VPU.NCE.ClusterTiling (%cst as %arg1: tensor<320x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<320x16x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:                    VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<320x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:                   -> tensor<320x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[DWCONV_WEIGHTS_TABLE_COPY_IN:%.*]] = VPU.NCE.ClusterTiling (%cst_0 as %arg1: tensor<320x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<320x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:                    VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<320x1x1x4xsi32>
    //CHECK-SAME:                   -> tensor<320x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[DWCONV:%.*]] = VPU.NCE.ClusterTiling ([[DWCONV_INPUT_COPY_IN]] as %arg1: tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                                           [[DWCONV_WEIGHTS_COPY_IN]] as %arg2: tensor<320x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                                           [[DWCONV_WEIGHTS_TABLE_COPY_IN]] as %arg3: tensor<320x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:                    VPU.NCE.DepthConvolution(%arg1, %arg2, %arg3) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [320, 1, 1, 1], strides = [1, 1]}
    //CHECK-SAME:                   -> tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[DWCONV_COPY_OUT:%.*]] = VPU.NCE.ClusterTiling ([[DWCONV]] as %arg1: tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x320x1x1xf16, {order = #NHWC}> {
    //CHECK:                    VPU.Copy(%arg1) : tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                        -> tensor<1x320x1x1xf16, {order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[SIGMOID_COPY_IN:%.*]] = VPU.NCE.ClusterTiling ([[DWCONV_COPY_OUT]] as %arg1: tensor<1x320x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:                    VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x320x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:                   -> tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[SIGMOID:%.*]] = VPU.NCE.ClusterTiling ([[SIGMOID_COPY_IN]] as %arg1: tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:                    VPU.Sigmoid(%arg1) : tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:                   -> tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[SIGMOID_COPY_OUT:%.*]] = VPU.NCE.ClusterTiling ([[SIGMOID]] as %arg1: tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x320x1x1xf16, {order = #NHWC}> {
    //CHECK:                    VPU.Copy(%arg1) : tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                        -> tensor<1x320x1x1xf16, {order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @InterpolateToNCEClusterTilingDuplicatedBuffer
func.func @InterpolateToNCEClusterTilingDuplicatedBuffer(%arg0: tensor<1x1x96x160xf16>) -> tensor<1x1x192x320xf16> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3], initial_input_dims_attr = [1, 1, 96, 160], initial_output_dims_attr = [1, 1, 192, 320], multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 320], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x1x96x160xf16> -> tensor<1x1x192x320xf16>
    return %0 : tensor<1x1x192x320xf16>

    //CHECK:  [[ClusterCopy:%.*]] = VPU.NCE.ClusterTiling (%arg0 as [[ARG1:%.*]]: tensor<1x1x96x160xf16>)
    //CHECK-SAME:    !VPU.DistributedTensor<1x1x96x160xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:                VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x1x96x160xf16> -> tensor<1x1x96x160xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:  }
    //CHECK:  [[ClusterInterpolate:%.*]] = VPU.NCE.ClusterTiling ([[ClusterCopy]] as [[ARG2:%.*]]: tensor<1x1x96x160xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1x192x320xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:                VPU.Interpolate([[ARG2]]) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], initial_input_dims_attr = [1, 1, 96, 160], initial_output_dims_attr = [1, 1, 192, 320], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 320], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x1x96x160xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x192x320xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:  }
    //CHECK:  [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[ClusterInterpolate]] as [[ARG3:%.*]]: tensor<1x1x192x320xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x192x320xf16> {
    //CHECK:                VPU.Copy([[ARG3]]) : tensor<1x1x192x320xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x192x320xf16>
    //CHECK:  }
    //CHECK: return [[OUT]] : tensor<1x1x192x320xf16>
}


// -----

// CHECK-LABEL: @InterpolateToNCEClusterTilingSegmentedBuffer

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @InterpolateToNCEClusterTilingSegmentedBuffer(%arg0: tensor<1x1x96x160xf16>) -> tensor<1x1x192x320xf16> {

    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3], initial_input_dims_attr = [1, 1, 96, 160], initial_output_dims_attr = [1, 1, 192, 320], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 320], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x1x96x160xf16> -> tensor<1x1x192x320xf16>
    return %0 : tensor<1x1x192x320xf16>

    //CHECK:  [[ClusterCopy:%.*]] = VPU.NCE.ClusterTiling (%arg0 as [[ARG1:%.*]]: tensor<1x1x96x160xf16>)
    //CHECK-SAME{LITERAL}: !VPU.DistributedTensor<1x1x96x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]], memory_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]}> {
    //CHECK:                VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x1x96x160xf16> -> tensor<1x1x96x160xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:  }
    //CHECK:  [[ClusterInterpolate:%.*]] = VPU.NCE.ClusterTiling ([[ClusterCopy]] as [[ARG2:%.*]]: tensor<1x1x96x160xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:                VPU.Interpolate([[ARG2]]) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], initial_input_dims_attr = [1, 1, 96, 160], initial_output_dims_attr = [1, 1, 192, 320], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 320], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x1x96x160xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x192x320xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:  }
    //CHECK:  [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[ClusterInterpolate]] as [[ARG3:%.*]]: tensor<1x1x192x320xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x192x320xf16> {
    //CHECK:                VPU.Copy([[ARG3]]) : tensor<1x1x192x320xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x192x320xf16>
    //CHECK:  }
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
            rawFilterShape = [1024, 1024, 3, 3], strides = [1, 1]} -> tensor<1x1024x14x14xf16, {order = #NHWC}>
    %1 = VPU.NCE.Eltwise(%arg0, %0) {multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>, ppe = #VPU.PPEStub<>, op_type = #VPU.eltwise_type<ADD>} -> tensor<1x1024x14x14xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%1, %cst_0, %cst) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            rawFilterShape = [1024, 1024, 3, 3], strides = [1, 1]} -> tensor<1x1024x14x14xf16, {order = #NHWC}>
    return %2 : tensor<1x1024x14x14xf16, {order = #NHWC}>

    // The Eltwise's input and output alignment should be equal
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<1024x1x1x4xsi32> = dense<10> : tensor<1024x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<1024x1024x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x1024x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[COPY_IN:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg2: tensor<1x1024x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1024x14x14xf16, #NHWC, @CMX_NN, {
    //CHECK-SAME:       mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>
    //CHECK:        [[COPY_W:%.*]] = VPU.NCE.ClusterTiling (%cst_0 as %arg2: tensor<1024x1024x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1024x1024x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:        [[COPY_WT:%.*]] = VPU.NCE.ClusterTiling (%cst as %arg2: tensor<1024x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1024x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[CONV:%.*]] = VPU.NCE.ClusterTiling ([[COPY_IN]] as %arg2: tensor<1x1024x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:       [[COPY_W]] as %arg3: tensor<1024x1024x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:       [[COPY_WT]] as %arg4: tensor<1024x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1024x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[COPY_TO_ELT_DDR:%.*]] = VPU.NCE.ClusterTiling ([[CONV]] as %arg2: tensor<1x1024x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> tensor<1x1024x14x14xf16, {order = #NHWC}>
    //CHECK:        [[COPY_TO_ELT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg2: tensor<1x1024x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1024x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>
    //CHECK:        [[COPT_TO_ELT_IN:%.*]] = VPU.NCE.ClusterTiling ([[COPY_TO_ELT_DDR]] as %arg2: tensor<1x1024x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1024x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>

    //CHECK:        [[ELTWISE:%.*]] = VPU.NCE.ClusterTiling ([[COPY_TO_ELT_CMX]] as %arg2: tensor<1x1024x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:       [[COPT_TO_ELT_IN]] as %arg3: tensor<1x1024x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1024x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>

    //CHECK:        [[COPY_ELT_OUT:%.*]] = VPU.NCE.ClusterTiling ([[ELTWISE]] as %arg2: tensor<1x1024x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> tensor<1x1024x14x14xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @InterpolateToNCEClusterTilingOVERLAPPED

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @InterpolateToNCEClusterTilingOVERLAPPED(%arg0: tensor<1x32x20x168xf16, {order = #NHWC}>) -> tensor<1x32x38x336xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SCALES>, coord_mode = <PYTORCH_HALF_PIXEL>,
                                nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0],
                                cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
                                initial_input_dims_attr = [1, 32, 95, 168], initial_input_offset_attr = [0, 0, 75, 0],
                                initial_output_dims_attr = [1, 32, 190, 336], initial_output_offset_attr = [0, 0, 152, 0],
                                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.900000e+00, 2.000000e+00], sizes_attr = [1, 32, 190, 336],
                                tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} :
                                tensor<1x32x20x168xf16, {order = #NHWC}> -> tensor<1x32x38x336xf16, {order = #NHWC}>
    return %0 : tensor<1x32x38x336xf16, {order = #NHWC}>

    //CHECK:               [[CLUSTERCOPYIN:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x20x168xf16, {order = #NHWC}>)
    //CHECK-SAME{LITERAL}: -> !VPU.DistributedTensor<1x32x20x168xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 32, 11, 168], [1, 32, 10, 168]], compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0]], memory_shapes = [[1, 32, 11, 168], [1, 32, 10, 168]], memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0]]}> {
    //CHECK:                    VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x20x168xf16, {order = #NHWC}> -> tensor<1x32x20x168xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:               }

    //CHECK:                [[INTERPOLATE:%.*]] = VPU.NCE.ClusterTiling ([[CLUSTERCOPYIN]] as %arg1: tensor<1x32x20x168xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> !VPU.DistributedTensor<1x32x38x336xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:                }

    //CHECK:                [[CLUSTERCOPYOUT:%.*]] = VPU.NCE.ClusterTiling ([[INTERPOLATE]] as %arg1: tensor<1x32x38x336xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x38x336xf16, {order = #NHWC}> {
    //CHECK:                    VPU.Copy(%arg1) : tensor<1x32x38x336xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x38x336xf16, {order = #NHWC}>
    //CHECK:                }

    //CHECK:                return [[CLUSTERCOPYOUT]] : tensor<1x32x38x336xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @ConvertToNCEClusterTilingSOH
func.func @ConvertToNCEClusterTilingSOH(%arg0: tensor<1x48x160x80xf32>) -> tensor<1x48x160x80xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
    return %0 : tensor<1x48x160x80xf16>
    // CHECK:   [[TILING1:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x48x160x80xf32>) -> !VPU.DistributedTensor<1x48x160x80xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   [[COPY1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x48x160x80xf32>
    // CHECK:   VPU.Yield [[COPY1]]

    // CHECK:   [[TILING2:%.*]] = VPU.NCE.ClusterTiling ([[TILING1]] as %arg1: tensor<1x48x160x80xf32, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x48x160x80xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   [[CONVERT:%.*]] = VPU.Convert(%arg1) {dstElemType = f16} : tensor<1x48x160x80xf32, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   VPU.Yield [[CONVERT]]

    // CHECK:   [[TILING3:%.*]] = VPU.NCE.ClusterTiling ([[TILING2]] as %arg1: tensor<1x48x160x80xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK:   [[COPY2:%.*]] = VPU.Copy(%arg1) : tensor<1x48x160x80xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   VPU.Yield [[COPY2]]

    // CHECK:   return [[TILING3]] : tensor<1x48x160x80xf16>
}


// -----

// CHECK-LABEL:   func.func @ConvertToNCEClusterTilingSOK
func.func @ConvertToNCEClusterTilingSOK(%arg0: tensor<1x48x160x80xf32>) -> tensor<1x48x160x80xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
    return %0 : tensor<1x48x160x80xf16>
    // CHECK:   [[TILING1:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x48x160x80xf32>) -> !VPU.DistributedTensor<1x48x160x80xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:   [[COPY1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x48x160x80xf32>
    // CHECK:   VPU.Yield [[COPY1]]

    // CHECK:   [[TILING2:%.*]] = VPU.NCE.ClusterTiling ([[TILING1]] as %arg1: tensor<1x48x160x80xf32, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x48x160x80xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:   [[CONVERT:%.*]] = VPU.Convert(%arg1) {dstElemType = f16} : tensor<1x48x160x80xf32, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   VPU.Yield [[CONVERT]]

    // CHECK:   [[TILING3:%.*]] = VPU.NCE.ClusterTiling ([[TILING2]] as %arg1: tensor<1x48x160x80xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK:   [[COPY2:%.*]] = VPU.Copy(%arg1) : tensor<1x48x160x80xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   VPU.Yield [[COPY2]]

    // CHECK:   return [[TILING3]] : tensor<1x48x160x80xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @ConvertToNCEClusterTilingClustering
func.func @ConvertToNCEClusterTilingClustering(%arg0: tensor<1x48x160x80xf32>) -> tensor<1x48x160x80xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
    return %0 : tensor<1x48x160x80xf16>
    // CHECK:   [[TILING1:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x48x160x80xf32>) -> !VPU.DistributedTensor<1x48x160x80xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:   [[COPY1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x48x160x80xf32>
    // CHECK:   VPU.Yield [[COPY1]]

    // CHECK:   [[TILING2:%.*]] = VPU.NCE.ClusterTiling ([[TILING1]] as %arg1: tensor<1x48x160x80xf32, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x48x160x80xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:   [[CONVERT:%.*]] = VPU.Convert(%arg1) {dstElemType = f16} : tensor<1x48x160x80xf32, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   VPU.Yield [[CONVERT]]

    // CHECK:   [[TILING3:%.*]] = VPU.NCE.ClusterTiling ([[TILING2]] as %arg1: tensor<1x48x160x80xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK:   [[COPY2:%.*]] = VPU.Copy(%arg1) : tensor<1x48x160x80xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   VPU.Yield [[COPY2]]

    // CHECK:   return [[TILING3]] : tensor<1x48x160x80xf16>
}

// -----

// CHECK-LABEL: @InterpolateAlignCornersToNCEClusterTilingOVERLAPPED
func.func @InterpolateAlignCornersToNCEClusterTilingOVERLAPPED(%arg0: tensor<1x1x257x257xf16>) -> tensor<1x1x17x17xf16> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ALIGN_CORNERS>,
            nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
            axes_attr = [0, 1, 2, 3], initial_input_dims_attr = [1, 1, 257, 257], initial_output_dims_attr = [1, 1, 17, 17],
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            scales_attr = [1.0000100135803223, 1.0000100135803223, 0.066157855093479156, 0.066157855093479156],
            sizes_attr = [1, 1, 17, 17], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]}
            : tensor<1x1x257x257xf16> -> tensor<1x1x17x17xf16>
    return %0 : tensor<1x1x17x17xf16>

    //CHECK:        [[COPY_IN:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x1x257x257xf16>)
    //CHECK-SAME{LITERAL}:  mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 129, 257], [1, 1, 113, 257]], compute_offsets = [[0, 0, 0, 0], [0, 0, 144, 0]], memory_shapes = [[1, 1, 129, 257], [1, 1, 113, 257]], memory_offsets = [[0, 0, 0, 0], [0, 0, 144, 0]]}>

    //CHECK:        [[INTERPOLATE:%.*]] = VPU.NCE.ClusterTiling ([[COPY_IN]] as %arg1: tensor<1x1x257x257xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK:                -> !VPU.DistributedTensor<1x1x17x17xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {

    //CHECK:        [[COPY_OUT:%.*]] = VPU.NCE.ClusterTiling ([[INTERPOLATE]] as %arg1: tensor<1x1x17x17xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x17x17xf16>
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

    //CHECK:        [[IN_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg3: tensor<1x4x224x224xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x4x224x224xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7],
    //CHECK-SAME:   pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}> {
    //CHECK:            %5 = VPU.Copy(%arg3) {out_mem_space = @CMX_NN} : tensor<1x4x224x224xf16, {order = #NHWC}> -> tensor<1x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield %5
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg1 as %arg3: tensor<64x1x1x160xf16, {order = #NHWC}>) -> !VPU.DistributedTensor<64x1x1x160xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            %5 = VPU.Copy(%arg3) {out_mem_space = @CMX_NN} : tensor<64x1x1x160xf16, {order = #NHWC}> -> tensor<64x1x1x160xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield %5
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg2 as %arg3: tensor<64x1x1x4xsi32>) -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            %5 = VPU.Copy(%arg3) {out_mem_space = @CMX_NN} : tensor<64x1x1x4xsi32> -> tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield %5
    //CHECK:        }

    //CHECK:        [[COMPRESSCONV:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:     [[IN_CMX]] as %arg3: tensor<1x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:     [[WEIGHTS_CMX]] as %arg4: tensor<64x1x1x160xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:     [[WEIGHTSTABLE_CMX]] as %arg5: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:     -> !VPU.DistributedTensor<1x64x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            %5 = VPU.NCE.CompressConvolution(%arg3, %arg4, %arg5) {cm_sp_pattern = 15 : i64, pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
    //CHECK-SAME:       ppe = #VPU.PPEStub<>, rawFilterShape = [64, 4, 7, 7], strides = [2, 2]} -> tensor<1x64x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield %5
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[COMPRESSCONV]] as %arg3: tensor<1x64x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x64x112x112xf16, {order = #NHWC}> {
    //CHECK:            %5 = VPU.Copy(%arg3) : tensor<1x64x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x64x112x112xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield %5
    //CHECK:        }

    //CHECK:        return [[OUT_CMX]] : tensor<1x64x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CompressConvToNCEClusterTilingSOB
// CHECK-SAME:  [[ARG0:%.+]]: tensor<2x4x224x224xf16, {order = #NHWC}>
// CHECK-SAME:  [[ARG1:%.+]]: tensor<64x1x1x160xf16, {order = #NHWC}>
// CHECK-SAME:  [[ARG2:%.+]]: tensor<64x1x1x4xsi32>
func.func @CompressConvToNCEClusterTilingSOB(%arg0: tensor<2x4x224x224xf16, {order = #NHWC}>, %arg1: tensor<64x1x1x160xf16, {order = #NHWC}>, %arg2: tensor<64x1x1x4xsi32>) -> tensor<2x64x112x112xf16, {order = #NHWC}>  {
    %compressConv = VPU.NCE.CompressConvolution(%arg0, %arg1, %arg2) {
                cm_sp_pattern = 15 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,
                pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
                ppe = #VPU.PPEStub<>,
                rawFilterShape = [64, 4, 7, 7], strides = [2, 2]}
        -> tensor<2x64x112x112xf16, {order = #NHWC}>

    return %compressConv : tensor<2x64x112x112xf16, {order = #NHWC}>

    // CHECK:   [[IN_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ARG0]] as [[ARG3:%.+]]: tensor<2x4x224x224xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<2x4x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}> {
    // CHECK:       [[RES0:%.+]] = VPU.Copy([[ARG3]]) {out_mem_space = @CMX_NN} : tensor<2x4x224x224xf16, {order = #NHWC}> -> tensor<2x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[RES0]]
    // CHECK:   }

    // CHECK:   [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ARG1]] as [[ARG3:%.+]]: tensor<64x1x1x160xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<64x1x1x160xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:       [[RES0:%.+]] = VPU.Copy([[ARG3]]) {out_mem_space = @CMX_NN} : tensor<64x1x1x160xf16, {order = #NHWC}> -> tensor<64x1x1x160xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[RES0]]
    // CHECK:   }

    // CHECK:   [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ARG2]] as [[ARG3:%.+]]: tensor<64x1x1x4xsi32>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:       [[RES0:%.+]] = VPU.Copy([[ARG3]]) {out_mem_space = @CMX_NN} : tensor<64x1x1x4xsi32> -> tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:       VPU.Yield [[RES0]]
    // CHECK:   }

    // CHECK:   [[COMPRESSCONV:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:  [[IN_CMX]] as [[ARG3:%.+]]: tensor<2x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:  [[WEIGHTS_CMX]] as [[ARG4:%.+]]: tensor<64x1x1x160xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:  [[WEIGHTSTABLE_CMX]] as [[ARG5:%.+]]: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<2x64x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}> {
    // CHECK:       [[RES0:%.+]] = VPU.NCE.CompressConvolution([[ARG3]], [[ARG4]], [[ARG5]]) {cm_sp_pattern = 15 : i64, pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
    // CHECK-SAME:  ppe = #VPU.PPEStub<>, rawFilterShape = [64, 4, 7, 7], strides = [2, 2]} -> tensor<2x64x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[RES0]]
    // CHECK:   }

    // CHECK:   [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[COMPRESSCONV]] as [[ARG3:%.+]]: tensor<2x64x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<2x64x112x112xf16, {order = #NHWC}> {
    // CHECK:       [[RES0:%.+]] = VPU.Copy([[ARG3]]) : tensor<2x64x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<2x64x112x112xf16, {order = #NHWC}>
    // CHECK:       VPU.Yield [[RES0]]
    // CHECK:   }

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

    //CHECK:        [[IN_CMX0:%.*]] = VPU.NCE.ClusterTiling ([[ARG0]] as [[IN_ARG0:[^:]+]]: tensor<1x4x224x224xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x4x224x224xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7],
    //CHECK-SAME:            pads = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
    //CHECK-SAME:            strides = [1, 1], num_clusters = 2 : i64}> {
    //CHECK:            VPU.Copy([[IN_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x4x224x224xf16, {order = #NHWC}>
    //CHECK-SAME:          -> tensor<1x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[COMPRESS0:%.*]] = VPU.NCE.ClusterTiling ([[IN_CMX0]] as [[IN_ARG1:[^:]+]]: tensor<1x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:     -> !VPU.DistributedTensor<1x64x224x224xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            VPU.NCE.CompressConvolution([[IN_ARG1]]
    //CHECK-SAME:           {cm_sp_pattern = 15 : i64,
    //CHECK-SAME:            pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>
    //CHECK-SAME:            rawFilterShape = [64, 4, 7, 7], strides = [1, 1]}
    //CHECK-SAME:          -> tensor<1x64x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[IN_CMX1:%.*]] = VPU.NCE.ClusterTiling ([[ARG0]] as [[IN_ARG2:[^:]+]]: tensor<1x4x224x224xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x4x224x224xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3],
    //CHECK-SAME:            pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:            strides = [1, 1], num_clusters = 2 : i64}> {
    //CHECK:            VPU.Copy([[IN_ARG2]]) {out_mem_space = @CMX_NN} : tensor<1x4x224x224xf16, {order = #NHWC}>
    //CHECK-SAME:          -> tensor<1x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[COMPRESS1:%.*]] = VPU.NCE.ClusterTiling ([[IN_CMX1]] as [[IN_ARG3:[^:]+]]: tensor<1x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:     -> !VPU.DistributedTensor<1x64x224x224xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            VPU.NCE.CompressConvolution([[IN_ARG3]]
    //CHECK-SAME:           {cm_sp_pattern = 15 : i64,
    //CHECK-SAME:            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    //CHECK-SAME:            rawFilterShape = [64, 4, 3, 3], strides = [1, 1]}
    //CHECK-SAME:          -> tensor<1x64x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

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
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [96, 128, 1, 1], strides = [1, 1]} -> tensor<1x96x8x8xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst_1, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [96, 96, 8, 8], strides = [8, 8]} -> tensor<1x96x1x1xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%0, %cst_2, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [96, 96, 1, 1], strides = [1, 1]} -> tensor<1x96x8x8xf16, {order = #NHWC}>
    return %1, %2 : tensor<1x96x1x1xf16, {order = #NHWC}>, tensor<1x96x8x8xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTSTABLE:%.*]] = const.Declare tensor<96x1x1x4xsi32> = dense<10> : tensor<96x1x1x4xsi32>
    //CHECK-DAG:    [[WEIGHTS_0:%.*]] = const.Declare tensor<96x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_1:%.*]] = const.Declare tensor<96x96x8x8xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x8x8xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_2:%.*]] = const.Declare tensor<96x96x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x1x1xf16>, [#const.Reorder<#NHWC>]

    // Conv producer

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x128x8x8xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x128x8x8xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x128x8x8xf16, {order = #NHWC}> -> tensor<1x128x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_0_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_0]] as %arg1: tensor<96x128x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<96x128x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x128x1x1xf16, {order = #NHWC}> -> tensor<96x128x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<96x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32> -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_0_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x128x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_0_CMX]] as %arg2: tensor<96x128x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x96x8x8xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:                 rawFilterShape = [96, 128, 1, 1], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x96x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_0:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0_CMX]] as %arg1: tensor<1x96x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x96x8x8xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x96x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x96x8x8xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    // First conv comsumer
    //
    // There's no way to satisfy both siblings with a single overlap config, thus default to it's own overlap config
    //

    //CHECK:        [[OUT_0_COPYBACK:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0]] as %arg1: tensor<1x96x8x8xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x96x8x8xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x96x8x8xf16, {order = #NHWC}> -> tensor<1x96x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_1_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_1]] as %arg1: tensor<96x96x8x8xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<96x96x8x8xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x96x8x8xf16, {order = #NHWC}> -> tensor<96x96x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_1_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<96x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32> -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_1_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[OUT_0_COPYBACK]] as %arg1: tensor<1x96x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_1_CMX]] as %arg2: tensor<96x96x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_1_CMX]] as %arg3: tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x96x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:                 rawFilterShape = [96, 96, 8, 8], strides = [8, 8]
    //CHECK-SAME:             } -> tensor<1x96x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_1_CMX]] as %arg1: tensor<1x96x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x96x1x1xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x96x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x96x1x1xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    // Second conv comsumer

    //CHECK:        [[OUT_0_COPYBACK_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0]] as %arg1: tensor<1x96x8x8xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x96x8x8xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x96x8x8xf16, {order = #NHWC}> -> tensor<1x96x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_2_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_2]] as %arg1: tensor<96x96x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<96x96x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x96x1x1xf16, {order = #NHWC}> -> tensor<96x96x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_2_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<96x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32> -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_2_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[OUT_0_COPYBACK_1]] as %arg1: tensor<1x96x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_2_CMX]] as %arg2: tensor<96x96x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_2_CMX]] as %arg3: tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x96x8x8xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:                 rawFilterShape = [96, 96, 1, 1], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x96x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_2:%.*]] = VPU.NCE.ClusterTiling ([[OUT_2_CMX]] as %arg1: tensor<1x96x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x96x8x8xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x96x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x96x8x8xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

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

    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg2: tensor<1x32x44x44xf16, {order = #NHWC}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x44x44xf16, {order = #NHWC}> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling (%arg1 as %arg2: tensor<1x1x44x44xf16, {order = #NHWC}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x44x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x1x44x44xf16, {order = #NHWC}> -> tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[MULTIPLY:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK:                                                  [[INPUT1]] as %arg3: tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> !VPU.DistributedTensor<1x32x44x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_MULTI:%.*]] = VPU.Multiply(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[MULTIPLY]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x44x44xf16, {order = #NHWC}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x44x44xf16, {order = #NHWC}>

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

    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg2: tensor<1x32x44x44xf16, {order = #NHWC}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x44x44xf16, {order = #NHWC}> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling (%arg1 as %arg2: tensor<1x1x1x44xf16, {order = #NHWC}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x1x1x44xf16, {order = #NHWC}> -> tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[MULTIPLY:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK:                                                  [[INPUT1]] as %arg3: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> !VPU.DistributedTensor<1x32x44x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_MULTI:%.*]] = VPU.Multiply(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[MULTIPLY]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x44x44xf16, {order = #NHWC}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x44x44xf16, {order = #NHWC}>

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

    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg2: tensor<1x32x1x44xf16, {order = #NHWC}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x1x44xf16, {order = #NHWC}> -> tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling (%arg1 as %arg2: tensor<1x32x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[MULTIPLY:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as %arg2: tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK:                                                  [[INPUT1]] as %arg3: tensor<1x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> !VPU.DistributedTensor<1x32x1x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_MULTI:%.*]] = VPU.Multiply(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[MULTIPLY]] as %arg2: tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x1x44xf16, {order = #NHWC}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) : tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x1x44xf16, {order = #NHWC}>

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

    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg2: tensor<1x32x1x44xf16, {order = #NHWC}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x1x44xf16, {order = #NHWC}> -> tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling (%arg1 as %arg2: tensor<1x1x1x44xf16, {order = #NHWC}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x1x1x44xf16, {order = #NHWC}> -> tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[MULTIPLY:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as %arg2: tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK:                                                  [[INPUT1]] as %arg3: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> !VPU.DistributedTensor<1x32x1x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_MULTI:%.*]] = VPU.Multiply(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[MULTIPLY]] as %arg2: tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x1x44xf16, {order = #NHWC}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) : tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x1x44xf16, {order = #NHWC}>

    //CHECK:        return [[OUTPUT]] : tensor<1x32x1x44xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MultiplySWSetAlignmentPerInputType
func.func @MultiplySWSetAlignmentPerInputType(%arg0: tensor<1x128x8x8xf16, {order = #NHWC}>, %arg1: tensor<1x1x1x128xf16>) -> tensor<1x32x1x128xf16> {
    %cst = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %cst_0 = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [64, 128, 1, 1], strides = [1, 1]} -> tensor<1x64x8x8xf16, {order = #NHWC}>

    %1 = VPU.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x64xf16>
    %2 = VPU.Reshape(%1) {shape_value = [1, 32, 1, 128]} : tensor<1x8x8x64xf16> -> tensor<1x32x1x128xf16>

    %3 = VPU.Multiply(%2, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x128xf16>, tensor<1x1x1x128xf16> -> tensor<1x32x1x128xf16>

    return %3 : tensor<1x32x1x128xf16>

    // CHECK:        [[MULTIPLY_INPUT1:%.*]] = VPU.NCE.ClusterTiling ([[RESHAPE:%.*]] as %arg2: tensor<1x32x1x128xf16>)
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:            [[RES0:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x1x128xf16> -> tensor<1x32x1x128xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:            VPU.Yield [[RES0]]
    // CHECK:        }

    // CHECK:        [[MULTIPLY_INPUT2:%.*]] = VPU.NCE.ClusterTiling (%arg1 as %arg2: tensor<1x1x1x128xf16>)
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x1x1x128xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:            [[RES0:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x1x1x128xf16> -> tensor<1x1x1x128xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:            VPU.Yield [[RES0]]
    // CHECK:        }

    // CHECK:        [[MULTIPLY:%.*]] = VPU.NCE.ClusterTiling ([[MULTIPLY_INPUT1]] as %arg2: tensor<1x32x1x128xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[MULTIPLY_INPUT2]] as %arg3: tensor<1x1x1x128xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:            [[RES0:%.*]] = VPU.Multiply(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1x128xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x128xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x1x128xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:            VPU.Yield [[RES0]]
    // CHECK:        }

    // CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[MULTIPLY]] as %arg2: tensor<1x32x1x128xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x1x128xf16> {
    // CHECK:            [[RES0:%.*]] = VPU.Copy(%arg2) : tensor<1x32x1x128xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x1x128xf16>
    // CHECK:            VPU.Yield [[RES0]]
    // CHECK:        }

    // CHECK:        return [[OUTPUT]] : tensor<1x32x1x128xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEInterpolateToNCEClusterTilingClustering
func.func @NCEInterpolateToNCEClusterTilingClustering(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x2x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x16x2x2xi1> = dense<1> : tensor<1x16x2x2xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 16, dataShape = [1, 16, 1, 1],
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

    // CHECK-DAG:    [[WEIGHTS:%.*]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[WEIGHTSTABLE:%.*]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // CHECK-DAG:    [[INPUT_SM:%.*]] = const.Declare tensor<1x16x2x2xi1> = dense<true> : tensor<1x16x2x2xi1>
    // CHECK:        [[INPUT_SE:%.*]] = VPU.StorageElementTable {dataElemType = i32, dataShape = [1, 16, 1, 1],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>,
    // CHECK-SAME:       seDepth = 1 : i64, seSize = 16 : i64}
    // CHECK-SAME:       -> tensor<1x1x2x2xi32, {order = #NHWC}>
    // CHECK:        [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])
    // CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_SPARSE]] as [[INNER_ARG0:[^:]+]]:
    // CHECK-SAME:           !VPU.SparseTensor<data=tensor<1x16x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                             sparsity_map=tensor<1x16x2x2xi1>,
    // CHECK-SAME:                             storage_element_table=tensor<1x1x2x2xi32, {order = #NHWC}>,
    // CHECK-SAME:                             #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>>)
    // CHECK-SAME:           -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
    // CHECK-SAME:                                sparsity_map=!VPU.DistributedTensor<1x16x2x2xi1, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:                                storage_element_table=!VPU.DistributedTensor<1x1x2x2xi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>,
    // CHECK-SAME:                                #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>> {
    // CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES0]]
    // CHECK:        }

    // CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[INNER_ARG0:[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:   -> !VPU.DistributedTensor<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    // CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES1]]
    // CHECK:        }

    // CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[INNER_ARG0:[^:]+]]: tensor<16x1x1x4xsi32>)
    // CHECK-SAME:   -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    // CHECK:            [[RES2:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES2]]
    // CHECK:        }

    // CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG0:[^:]+]]: !VPU.SparseTensor<data=tensor<1x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      sparsity_map=tensor<1x16x2x2xi1, {mem_space = @CMX_NN, order = #NCHW}>,
    // CHECK-SAME:                                                                      storage_element_table=tensor<1x1x2x2xi32, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                                                         scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                                                                                         nearest_mode = <FLOOR>,
    // CHECK-SAME:                                                                                         offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>>,
    // CHECK-SAME:             [[WEIGHTS_CMX]] as [[INNER_ARG1:[^:]+]]: tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[INNER_ARG2:[^:]+]]: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:   -> !VPU.DistributedTensor<1x16x2x2xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:            [[RES3:%.*]] = VPU.NCE.Interpolate([[INNER_ARG0]], [[INNER_ARG1]], [[INNER_ARG2]])
    // CHECK-SAME:           mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:           rawFilterShape = [16, 16, 1, 1],
    // CHECK-SAME:           scales_attr = [2, 2],
    // CHECK-SAME:           strides = [1, 1]
    // CHECK-SAME:           } -> tensor<1x16x2x2xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:            VPU.Yield [[RES3]]
    // CHECK:        }

    // CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG0:[^:]+]]: tensor<1x16x2x2xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x2x2xf16, {order = #NHWC}> {
    // CHECK:            [[RES4:%.*]] = VPU.Copy([[INNER_ARG0]])
    // CHECK:            VPU.Yield [[RES4]]
    // CHECK:        }

    // CHECK:        return [[OUT]] : tensor<1x16x2x2xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEInterpolateToNCEClusterTilingSOH
func.func @NCEInterpolateToNCEClusterTilingSOH(%arg0: tensor<1x64x5x10xf16, {order = #NHWC}>) -> tensor<1x64x10x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x64x10x20xi1> = dense<1> : tensor<1x64x10x20xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 64, dataShape = [1, 64, 5, 10],
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

    // CHECK-DAG:    [[WEIGHTS:%.*]] = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[WEIGHTSTABLE:%.*]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    // CHECK-DAG:    [[INPUT_SM:%.*]] = const.Declare tensor<1x64x10x20xi1> = dense<true> : tensor<1x64x10x20xi1>
    // CHECK:        [[INPUT_SE:%.*]] = VPU.StorageElementTable {dataElemType = i32, dataShape = [1, 64, 5, 10],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>,
    // CHECK-SAME:       seDepth = 1 : i64, seSize = 64 : i64}
    // CHECK-SAME:       -> tensor<1x1x10x20xi32, {order = #NHWC}>

    // CHECK:        [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])
    // CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_SPARSE]] as [[INNER_ARG0:[^:]+]]:
    // CHECK-SAME:           !VPU.SparseTensor<data=tensor<1x64x5x10xf16, {order = #NHWC}>,
    // CHECK-SAME:                             sparsity_map=tensor<1x64x10x20xi1>,
    // CHECK-SAME:                             storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
    // CHECK-SAME:                             #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>>)
    // CHECK-SAME:           -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x64x5x10xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                sparsity_map=!VPU.DistributedTensor<1x64x10x20xi1, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>,
    // CHECK-SAME:                                storage_element_table=!VPU.DistributedTensor<1x1x10x20xi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>,
    // CHECK-SAME:                                #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>> {
    // CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES0]]
    // CHECK:        }

    // CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[INNER_ARG0:[^:]+]]: tensor<64x64x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:   -> !VPU.DistributedTensor<64x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES1]]
    // CHECK:        }

    // CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[INNER_ARG0:[^:]+]]: tensor<64x1x1x4xsi32>)
    // CHECK-SAME:   -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:            [[RES2:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES2]]
    // CHECK:        }

    // CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG0:[^:]+]]: !VPU.SparseTensor<data=tensor<1x64x5x10xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      sparsity_map=tensor<1x64x10x20xi1, {mem_space = @CMX_NN, order = #NCHW}>,
    // CHECK-SAME:                                                                      storage_element_table=tensor<1x1x10x20xi32, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                                                         scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECKSAME:                                                                                          nearest_mode = <FLOOR>,
    // CHECK-SAME:                                                                                         offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>>,
    // CHECK-SAME:             [[WEIGHTS_CMX]] as [[INNER_ARG1:[^:]+]]: tensor<64x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[INNER_ARG2:[^:]+]]: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:   -> !VPU.DistributedTensor<1x64x10x20xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:            [[RES3:%.*]] = VPU.NCE.Interpolate([[INNER_ARG0]], [[INNER_ARG1]], [[INNER_ARG2]])
    // CHECK-SAME:           mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:           rawFilterShape = [64, 64, 1, 1],
    // CHECK-SAME:           scales_attr = [2, 2]
    // CHECK-SAME:           strides = [1, 1]
    // CHECK-SAME:           } -> tensor<1x64x10x20xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:            VPU.Yield [[RES3]]
    // CHECK:        }

    // CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG0:[^:]+]]: tensor<1x64x10x20xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x64x10x20xf16, {order = #NHWC}> {
    // CHECK:            [[RES4:%.*]] = VPU.Copy([[INNER_ARG0]])
    // CHECK:            VPU.Yield [[RES4]]
    // CHECK:        }

    // CHECK:        return [[OUT]] : tensor<1x64x10x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEInterpolateToNCEClusterTilingSOK
func.func @NCEInterpolateToNCEClusterTilingSOK(%arg0: tensor<1x64x5x10xf16, {order = #NHWC}>) -> tensor<1x64x10x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x64x10x20xi1> = dense<1> : tensor<1x64x10x20xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 64, dataShape = [1, 64, 5, 10],
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

    // CHECK-DAG:    [[WEIGHTS:%.*]] = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[WEIGHTSTABLE:%.*]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    // CHECK-DAG:    [[INPUT_SM:%.*]] = const.Declare tensor<1x64x10x20xi1> = dense<true> : tensor<1x64x10x20xi1>
    // CHECK:        [[INPUT_SE:%.*]] = VPU.StorageElementTable {dataElemType = i32, dataShape = [1, 64, 5, 10],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>,
    // CHECK-SAME:       seDepth = 1 : i64, seSize = 64 : i64}
    // CHECK-SAME:       -> tensor<1x1x10x20xi32, {order = #NHWC}>
    // CHECK:        [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])
    // CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_SPARSE]] as [[INNER_ARG0:[^:]+]]:
    // CHECK-SAME:           !VPU.SparseTensor<data=tensor<1x64x5x10xf16, {order = #NHWC}>,
    // CHECK-SAME:                             sparsity_map=tensor<1x64x10x20xi1>,
    // CHECK-SAME:                             storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
    // CHECK-SAME:                             #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>>)
    // CHECK-SAME:           -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x64x5x10xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
    // CHECK-SAME:                                sparsity_map=!VPU.DistributedTensor<1x64x10x20xi1, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:                                storage_element_table=!VPU.DistributedTensor<1x1x10x20xi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>,
    // CHECK-SAME:                                #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>> {
    // CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES0]]
    // CHECK:        }

    // CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[INNER_ARG0:[^:]+]]: tensor<64x64x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:   -> !VPU.DistributedTensor<64x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    // CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES1]]
    // CHECK:        }

    // CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[INNER_ARG0:[^:]+]]: tensor<64x1x1x4xsi32>)
    // CHECK-SAME:   -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    // CHECK:            [[RES2:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES2]]
    // CHECK:        }

    // CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG0:[^:]+]]: !VPU.SparseTensor<data=tensor<1x64x5x10xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      sparsity_map=tensor<1x64x10x20xi1, {mem_space = @CMX_NN, order = #NCHW}>,
    // CHECK-SAME:                                                                      storage_element_table=tensor<1x1x10x20xi32, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                                                         scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECKSAME:                                                                                          nearest_mode = <FLOOR>,
    // CHECK-SAME:                                                                                         offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>>,
    // CHECK-SAME:             [[WEIGHTS_CMX]] as [[INNER_ARG1:[^:]+]]: tensor<64x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[INNER_ARG2:[^:]+]]: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:   -> !VPU.DistributedTensor<1x64x10x20xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:            [[RES3:%.*]] = VPU.NCE.Interpolate([[INNER_ARG0]], [[INNER_ARG1]], [[INNER_ARG2]])
    // CHECK-SAME:           mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:           rawFilterShape = [64, 64, 1, 1],
    // CHECK-SAME:           scales_attr = [2, 2]
    // CHECK-SAME:           strides = [1, 1]
    // CHECK-SAME:           } -> tensor<1x64x10x20xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:            VPU.Yield [[RES3]]
    // CHECK:        }

    // CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG0:[^:]+]]: tensor<1x64x10x20xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x64x10x20xf16, {order = #NHWC}> {
    // CHECK:            [[RES4:%.*]] = VPU.Copy([[INNER_ARG0]])
    // CHECK:            VPU.Yield [[RES4]]
    // CHECK:        }

    // CHECK:        return [[OUT]] : tensor<1x64x10x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddToNCEClusterTilingSOHDuplicatedIn
func.func @EltwiseAddToNCEClusterTilingSOHDuplicatedIn(%arg0: tensor<1x3x52x104xf16, {order = #NHWC}>) -> tensor<1x16x39x26xf16, {order = #NHWC}> {
    %0 = VPU.ShapeCast {shape = [1, 16, 39, 26]} inputs(%arg0 : tensor<1x3x52x104xf16, {order = #NHWC}>) -> tensor<1x16x39x26xf16, {order = #NHWC}>
    %1 = VPU.NCE.Eltwise(%0, %0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, op_type = #VPU.eltwise_type<ADD> } -> tensor<1x16x39x26xf16, {order = #NHWC}>


    return %1: tensor<1x16x39x26xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_SHAPE_CAST:%.*]] = VPU.ShapeCast {shape = [1, 16, 39, 26]} inputs(%arg0 : tensor<1x3x52x104xf16, {order = #NHWC}>) -> tensor<1x16x39x26xf16, {order = #NHWC}>
    //CHECK:        [[INPUT0_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_SHAPE_CAST]] as %arg1: tensor<1x16x39x26xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x39x26xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x16x39x26xf16, {order = #NHWC}> -> tensor<1x16x39x26xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:          VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0_CMX]] as %arg1: tensor<1x16x39x26xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x39x26xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[RES1:%.*]] = VPU.NCE.Eltwise(%arg1, %arg1) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x16x39x26xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:          VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x16x39x26xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x39x26xf16, {order = #NHWC}> {
    //CHECK:          [[RES2:%.*]] = VPU.Copy(%arg1) : tensor<1x16x39x26xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x39x26xf16, {order = #NHWC}>
    //CHECK:          VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x16x39x26xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TopKSWTilingSOH
func.func @TopKSWTilingSOH(%arg0: tensor<1x31x103x513xf16, {order = #NHWC}>) -> tensor<1x1x103x513xsi32, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8, {order = #NHWC}> = dense<0> : tensor<1x1x1x16xui8>, [#const.Reorder<#NHWC>]
    %output_values, %target_shape = VPU.TopK(%arg0, %cst) {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, operandSegmentSizes = array<i32: 1, 0, 1>, sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<1x31x103x513xf16, {order = #NHWC}>, tensor<1x1x1x16xui8, {order = #NHWC}> -> tensor<1x1x103x513xf16, {order = #NHWC}>, tensor<1x1x103x513xsi32, {order = #NHWC}>

    return %target_shape : tensor<1x1x103x513xsi32, {order = #NHWC}>

    //CHECK:        [[CST:%.*]] = const.Declare tensor<1x1x1x16xui8, {order = #NHWC}> = dense<0> : tensor<1x1x1x16xui8>, [#const.Reorder<#NHWC>]
    //CHECK:        [[INPUT:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x31x103x513xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x31x103x513xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[INNER_COPY:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x31x103x513xf16, {order = #NHWC}> -> tensor<1x31x103x513xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[INNER_COPY]]
    //CHECK:        }

    //CHECK:        [[AUX_BUFFER:%.*]] = VPU.NCE.ClusterTiling ([[CST]] as %arg1: tensor<1x1x1x16xui8, {order = #NHWC}>)
    //CHECK:         -> !VPU.DistributedTensor<1x1x1x16xui8, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[AUX_BUFFER_INNER_COPY:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x1x1x16xui8, {order = #NHWC}> -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[AUX_BUFFER_INNER_COPY]]
    //CHECK:        }

    //CHECK:        [[OUTPUTS:%.*]]:2 = VPU.NCE.ClusterTiling ([[INPUT]] as %arg1: tensor<1x31x103x513xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[AUX_BUFFER]] as %arg2: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> (!VPU.DistributedTensor<1x1x103x513xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    //CHECK-SMAE:   !VPU.DistributedTensor<1x1x103x513xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    //CHECK:            [[OUTPUT:%.*]], [[TARGET:%.*]] = VPU.TopK(%arg1, %arg2) {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, operandSegmentSizes = array<i32: 1, 0, 1>, sort = #IE.topk_sort_type<SORT_INDICES>}
    //CHECK-SMAE:       : tensor<1x31x103x513xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x1x103x513xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x1x103x513xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[OUTPUT]], [[TARGET]]
    //CHECK:        }

    //CHECK:        [[OUTPUT_VALUES:%.*]] = VPU.NCE.ClusterTiling ([[OUTPUTS]]#0 as %arg1: tensor<1x1x103x513xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x1x103x513xf16, {order = #NHWC}> {
    //CHECK:            [[INNER_COPY:%.*]] = VPU.Copy(%arg1) : tensor<1x1x103x513xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x1x103x513xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[INNER_COPY]]
    //CHECK:        }

    //CHECK:        [[TARGET_SHAPE:%.*]] = VPU.NCE.ClusterTiling ([[OUTPUTS]]#1 as %arg1: tensor<1x1x103x513xsi32, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x1x103x513xsi32, {order = #NHWC}> {
    //CHECK:            [[INNER_COPY:%.*]] = VPU.Copy(%arg1) : tensor<1x1x103x513xsi32, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x1x103x513xsi32, {order = #NHWC}>
    //CHECK:            VPU.Yield [[INNER_COPY]]
    //CHECK:        }

    //CHECK:        return [[TARGET_SHAPE]] : tensor<1x1x103x513xsi32, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TopKSWTilingSOK
func.func @TopKSWTilingSOK(%arg0: tensor<1x103x513x31xf16, {order = #NHWC}>) -> tensor<1x103x513x1xsi32, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x496xui8, {order = #NHWC}> = dense<0> : tensor<1x1x1x496xui8>, [#const.Reorder<#NHWC>]
    %output_values, %target_shape = VPU.TopK(%arg0, %cst) {axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, operandSegmentSizes = array<i32: 1, 0, 1>, sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<1x103x513x31xf16, {order = #NHWC}>, tensor<1x1x1x496xui8, {order = #NHWC}> -> tensor<1x103x513x1xf16, {order = #NHWC}>, tensor<1x103x513x1xsi32, {order = #NHWC}>

    return %target_shape : tensor<1x103x513x1xsi32, {order = #NHWC}>

    //CHECK:        [[CST:%.*]] = const.Declare tensor<1x1x1x496xui8, {order = #NHWC}> = dense<0> : tensor<1x1x1x496xui8>, [#const.Reorder<#NHWC>]
    //CHECK:        [[INPUT:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x103x513x31xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x103x513x31xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[INNER_COPY:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x103x513x31xf16, {order = #NHWC}> -> tensor<1x103x513x31xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[INNER_COPY]]
    //CHECK:        }

    //CHECK:        [[AUX_BUFFER:%.*]] = VPU.NCE.ClusterTiling ([[CST]] as %arg1: tensor<1x1x1x496xui8, {order = #NHWC}>)
    //CHECK:         -> !VPU.DistributedTensor<1x1x1x496xui8, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[AUX_BUFFER_INNER_COPY:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x1x1x496xui8, {order = #NHWC}> -> tensor<1x1x1x496xui8, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[AUX_BUFFER_INNER_COPY]]
    //CHECK:        }

    //CHECK:        [[OUTPUTS:%.*]]:2 = VPU.NCE.ClusterTiling ([[INPUT]] as %arg1: tensor<1x103x513x31xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[AUX_BUFFER]] as %arg2: tensor<1x1x1x496xui8, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> (!VPU.DistributedTensor<1x103x513x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>,
    //CHECK-SMAE:   !VPU.DistributedTensor<1x103x513x1xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) {
    //CHECK:            [[OUTPUT:%.*]], [[TARGET:%.*]] = VPU.TopK(%arg1, %arg2) {axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, operandSegmentSizes = array<i32: 1, 0, 1>, sort = #IE.topk_sort_type<SORT_INDICES>}
    //CHECK-SMAE:       : tensor<1x103x513x31xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x1x1x496xui8, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x103x513x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x103x513x1xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[OUTPUT]], [[TARGET]]
    //CHECK:        }

    //CHECK:        [[OUTPUT_VALUES:%.*]] = VPU.NCE.ClusterTiling ([[OUTPUTS]]#0 as %arg1: tensor<1x103x513x1xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x103x513x1xf16, {order = #NHWC}> {
    //CHECK:            [[INNER_COPY:%.*]] = VPU.Copy(%arg1) : tensor<1x103x513x1xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x103x513x1xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[INNER_COPY]]
    //CHECK:        }

    //CHECK:        [[TARGET_SHAPE:%.*]] = VPU.NCE.ClusterTiling ([[OUTPUTS]]#1 as %arg1: tensor<1x103x513x1xsi32, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x103x513x1xsi32, {order = #NHWC}> {
    //CHECK:            [[INNER_COPY:%.*]] = VPU.Copy(%arg1) : tensor<1x103x513x1xsi32, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x103x513x1xsi32, {order = #NHWC}>
    //CHECK:            VPU.Yield [[INNER_COPY]]
    //CHECK:        }

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
    %1 = VPU.NCE.Convolution(%0, %weights_0, %weights_table_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [256, 80, 1, 3], strides = [1, 1]} -> tensor<1x256x1x3000xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%0, %weights_1, %weights_table_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [256, 80, 1, 3], strides = [1, 1]} -> tensor<1x256x1x3000xf16, {order = #NHWC}>

    %3 = VPU.Concat(%1, %2) {static_offsets = [[0, 0, 0, 0], [0, 256, 0, 0]]} : tensor<1x256x1x3000xf16, {order = #NHWC}>, tensor<1x256x1x3000xf16, {order = #NHWC}> -> tensor<1x512x1x3000xf16, {order = #NHWC}>

    %4 = VPU.Slice %3 [0, 0, 0, 0] [1, 512, 1, 1500] : tensor<1x512x1x3000xf16, {order = #NHWC}> to tensor<1x512x1x1500xf16, {order = #NHWC}>
    %5 = VPU.Gelu(%4) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x512x1x1500xf16, {order = #NHWC}> -> tensor<1x512x1x1500xf16, {order = #NHWC}>

    %6 = VPU.Slice %3 [0, 0, 0, 1500] [1, 512, 1, 1500] : tensor<1x512x1x3000xf16, {order = #NHWC}> to tensor<1x512x1x1500xf16, {order = #NHWC}>
    %7 = VPU.Gelu(%6) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x512x1x1500xf16, {order = #NHWC}> -> tensor<1x512x1x1500xf16, {order = #NHWC}>

    %8 = VPU.Concat(%5, %7) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1500]]} : tensor<1x512x1x1500xf16, {order = #NHWC}>, tensor<1x512x1x1500xf16, {order = #NHWC}> -> tensor<1x512x1x3000xf16, {order = #NHWC}>
    return %8 : tensor<1x512x1x3000xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS_0:%.*]] = const.Declare tensor<256x80x1x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x80x1x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTSTABLE_0:%.*]] = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    // CHECK-DAG:   [[WEIGHTS_1:%.*]] = const.Declare tensor<256x80x1x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x80x1x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTSTABLE_1:%.*]] = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>

    // CHECK: [[CONV_INPUT:%.*]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 80, 1, 3000] : tensor<1x80x1x3008xf16, {order = #NHWC}> to tensor<1x80x1x3000xf16, {order = #NHWC}>
    // CHECK: [[CONV0_INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[CONV_INPUT]] as %arg1: tensor<1x80x1x3000xf16, {order = #NHWC}>)
    // CHECK-SAME: -> !VPU.DistributedTensor<1x80x1x3000xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:   [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x80x1x3000xf16, {order = #NHWC}> -> tensor<1x80x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:   VPU.Yield [[RES0]]

    // CHECK: [[CONV0_WEIGHT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_0]] as %arg1: tensor<256x80x1x3xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<256x80x1x3xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    // CHECK:   [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<256x80x1x3xf16, {order = #NHWC}> -> tensor<256x80x1x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:   VPU.Yield [[RES1]]

    // CHECK: [[CONV0_WEIGHTTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE_0]] as %arg1: tensor<256x1x1x4xsi32>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<256x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    // CHECK:   [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<256x1x1x4xsi32> -> tensor<256x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   VPU.Yield [[RES2]]

    // CHECK: [[CONV0:%.*]] = VPU.NCE.ClusterTiling ([[CONV0_INPUT_CMX]] as %arg1: tensor<1x80x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK:               [[CONV0_WEIGHT_CMX]] as %arg2: tensor<256x80x1x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK:               [[CONV0_WEIGHTTABLE_CMX]] as %arg3: tensor<256x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME: -> !VPU.DistributedTensor<1x256x1x3000xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:   [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK:   VPU.Yield [[RES3]]

    // CHECK: [[CONV0_OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[CONV0]] as %arg1: tensor<1x256x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x256x1x3000xf16, {order = #NHWC}> {
    // CHECK:   [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x256x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x256x1x3000xf16, {order = #NHWC}>
    // CHECK:   VPU.Yield [[RES4]]

    // CHECK: [[CONV1_INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[CONV_INPUT]] as %arg1: tensor<1x80x1x3000xf16, {order = #NHWC}>)
    // CHECK-SAME:   -> !VPU.DistributedTensor<1x80x1x3000xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:   [[RES5:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x80x1x3000xf16, {order = #NHWC}> -> tensor<1x80x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:   VPU.Yield [[RES5]]
    // CHECK: [[CONV1_WEIGHT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_1]] as %arg1: tensor<256x80x1x3xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<256x80x1x3xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    // CHECK:   [[RES6:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<256x80x1x3xf16, {order = #NHWC}> -> tensor<256x80x1x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:   VPU.Yield [[RES6]]

    // CHECK: [[CONV1_WEIGHTTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE_1]] as %arg1: tensor<256x1x1x4xsi32>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<256x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    // CHECK:   [[RES7:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<256x1x1x4xsi32> -> tensor<256x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   VPU.Yield [[RES7]]
    // CHECK:  }

    // CHECK: [[CONV1:%.*]] = VPU.NCE.ClusterTiling ([[CONV1_INPUT_CMX]] as %arg1: tensor<1x80x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK:               [[CONV1_WEIGHT_CMX]] as %arg2: tensor<256x80x1x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK:               [[CONV1_WEIGHTTABLE_CMX]] as %arg3: tensor<256x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME: -> !VPU.DistributedTensor<1x256x1x3000xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:   [[RES8:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK:   VPU.Yield [[RES8]]

    // CHECK: [[CONV1_OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[CONV1]] as %arg1: tensor<1x256x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x256x1x3000xf16, {order = #NHWC}> {
    // CHECK:   [[RES9:%.*]] = VPU.Copy(%arg1) : tensor<1x256x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x256x1x3000xf16, {order = #NHWC}>
    // CHECK:   VPU.Yield [[RES9]]

    // CHECK: [[CONV_CONCAT:%.*]] = VPU.Concat([[CONV0_OUTPUT]], [[CONV1_OUTPUT]]) {static_offsets = [
    // CHECK-SAME:    [0, 0, 0, 0], [0, 256, 0, 0]
    // CHECK-SAME:  ]} :
    // CHECK-SAME: tensor<1x256x1x3000xf16, {order = #NHWC}>,
    // CHECK-SAME: tensor<1x256x1x3000xf16, {order = #NHWC}> -> tensor<1x512x1x3000xf16, {order = #NHWC}>

    // CHECK: [[GELU_0_SLICE:%.*]] = VPU.Slice [[CONV_CONCAT]] [0, 0, 0, 0] [1, 512, 1, 1500] :
    // CHECK-SAME:                  tensor<1x512x1x3000xf16, {order = #NHWC}> to tensor<1x512x1x1500xf16, {order = #NHWC}>

    // CHECK: [[GELU_0_INPUT:%.*]] = VPU.NCE.ClusterTiling ([[GELU_0_SLICE]] as %arg1: tensor<1x512x1x1500xf16, {order = #NHWC}>)
    // CHECK-SAME:   -> !VPU.DistributedTensor<1x512x1x1500xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:   [[RES10:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x512x1x1500xf16, {order = #NHWC}> -> tensor<1x512x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:   VPU.Yield [[RES10]]

    // CHECK: [[GELU_0:%.*]] = VPU.NCE.ClusterTiling ([[GELU_0_INPUT]] as %arg1: tensor<1x512x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x512x1x1500xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:   [[RES11:%.*]] = VPU.Gelu(%arg1) : tensor<1x512x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x512x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:   VPU.Yield [[RES11]]

    // CHECK: [[GELU_0_OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[GELU_0]] as %arg1: tensor<1x512x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x512x1x1500xf16, {order = #NHWC}> {
    // CHECK:   [[RES12:%.*]] = VPU.Copy(%arg1) : tensor<1x512x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x512x1x1500xf16, {order = #NHWC}>
    // CHECK:   VPU.Yield [[RES12]]

    // CHECK: [[GELU_1_SLICE:%.*]] = VPU.Slice [[CONV_CONCAT]] [0, 0, 0, 1500] [1, 512, 1, 1500] :
    // CHECK-SAME:                  tensor<1x512x1x3000xf16, {order = #NHWC}> to tensor<1x512x1x1500xf16, {order = #NHWC}>
    // CHECK: [[GELU_1_INPUT:%.*]] = VPU.NCE.ClusterTiling ([[GELU_1_SLICE]] as %arg1: tensor<1x512x1x1500xf16, {order = #NHWC}>)
    // CHECK-SAME:   -> !VPU.DistributedTensor<1x512x1x1500xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:   [[RES13:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x512x1x1500xf16, {order = #NHWC}> -> tensor<1x512x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:   VPU.Yield [[RES13]]

    // CHECK: [[GELU_1:%.*]] = VPU.NCE.ClusterTiling ([[GELU_1_INPUT]] as %arg1: tensor<1x512x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x512x1x1500xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:   [[RES14:%.*]] = VPU.Gelu(%arg1) : tensor<1x512x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x512x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:   VPU.Yield [[RES14]]

    // CHECK: [[GELU_1_OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[GELU_1]] as %arg1: tensor<1x512x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x512x1x1500xf16, {order = #NHWC}> {
    // CHECK:   [[RES15:%.*]] = VPU.Copy(%arg1) : tensor<1x512x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x512x1x1500xf16, {order = #NHWC}>
    // CHECK:   VPU.Yield [[RES15]]

    // CHECK: [[GELU_CONCAT:%.*]] = VPU.Concat([[GELU_0_OUTPUT]], [[GELU_1_OUTPUT]]) {static_offsets = [
    // CHECK-SAME:     [0, 0, 0, 0], [0, 0, 0, 1500]
    // CHECK:  ]} : tensor<1x512x1x1500xf16, {order = #NHWC}>, tensor<1x512x1x1500xf16, {order = #NHWC}> -> tensor<1x512x1x3000xf16, {order = #NHWC}>

    // CHECK: return [[GELU_CONCAT]] : tensor<1x512x1x3000xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthToSpaceToNCEClusterTilingSOH
func.func @DepthToSpaceToNCEClusterTilingSOH(%arg0: tensor<1x128x12x270xf16, {order = #NHWC}>) -> tensor<1x8x48x1080xf16, {order = #NHWC}> {
    %0 = VPU.DepthToSpace(%arg0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x128x12x270xf16, {order = #NHWC}> -> tensor<1x8x48x1080xf16, {order = #NHWC}>

    return %0 : tensor<1x8x48x1080xf16, {order = #NHWC}>

    //CHECK:        [[INPUT:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x128x12x270xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[INNER_COPY:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x128x12x270xf16, {order = #NHWC}> -> tensor<1x128x12x270xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[INNER_COPY]]
    //CHECK:        }

    //CHECK:        [[CLUSTER_TILING_D2S:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as %arg1: tensor<1x128x12x270xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}> {
    //CHECK:            [[INNER_D2S:%.*]] = VPU.DepthToSpace(%arg1) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x128x12x270xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x8x48x1080xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[INNER_D2S]]
    //CHECK:        }

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[CLUSTER_TILING_D2S]] as %arg1: tensor<1x8x48x1080xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> tensor<1x8x48x1080xf16, {order = #NHWC}> {
    //CHECK:            [[INNER_COPY:%.*]] = VPU.Copy(%arg1) : tensor<1x8x48x1080xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x8x48x1080xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[INNER_COPY]]
    //CHECK:        }

    //CHECK:        return [[OUTPUT]] : tensor<1x8x48x1080xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthToSpaceToNCEClusterTilingSOW
func.func @DepthToSpaceToNCEClusterTilingSOW(%arg0: tensor<1x128x1x270xf16, {order = #NHWC}>) -> tensor<1x8x4x1080xf16, {order = #NHWC}> {
    %0 = VPU.DepthToSpace(%arg0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>} : tensor<1x128x1x270xf16, {order = #NHWC}> -> tensor<1x8x4x1080xf16, {order = #NHWC}>

    return %0 : tensor<1x8x4x1080xf16, {order = #NHWC}>

    //CHECK:                 [[INPUT:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x128x1x270xf16, {order = #NHWC}>)
    //CHECK-SAME:            -> !VPU.DistributedTensor<1x128x1x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}> {
    //CHECK:                     [[INNER_COPY:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x128x1x270xf16, {order = #NHWC}> -> tensor<1x128x1x270xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                     VPU.Yield [[INNER_COPY]]
    //CHECK:                 }

    //CHECK:                 [[CLUSTER_TILING_D2S:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as %arg1: tensor<1x128x1x270xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME{LITERAL}:   -> !VPU.DistributedTensor<1x8x4x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64, alignment = [1, 1, 1, 4]}> {
    //CHECK:                     [[INNER_D2S:%.*]] = VPU.DepthToSpace(%arg1) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x128x1x270xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x8x4x1080xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                     VPU.Yield [[INNER_D2S]]
    //CHECK:                 }

    //CHECK:                 [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[CLUSTER_TILING_D2S]] as %arg1: tensor<1x8x4x1080xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:            -> tensor<1x8x4x1080xf16, {order = #NHWC}> {
    //CHECK:                     [[INNER_COPY:%.*]] = VPU.Copy(%arg1) : tensor<1x8x4x1080xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x8x4x1080xf16, {order = #NHWC}>
    //CHECK:                     VPU.Yield [[INNER_COPY]]
    //CHECK:                 }

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

    // CHECK:       [[COPY_INPUT:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x3x256x224xf16>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x3x256x224xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[NCE_PERMUTE:%.*]] = VPU.NCE.ClusterTiling ([[COPY_INPUT]] as %arg1: tensor<1x3x256x224xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x4x256x224x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {

    // CHECK:       VPU.NCE.Permute
    // CHECK-SAME:      dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64
    // CHECK-SAME:      -> tensor<1x4x256x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[COPY_OUTPUT:%.*]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      -> tensor<1x4x256x224x!qElemType, {order = #NHWC}> {
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

    // CHECK:       [[COPY_INPUT:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x3x224x224xf16>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x3x224x224xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:      {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3],
    // CHECK-SAME:      pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:      strides = [2, 2], num_clusters = 2 : i64}> {

    // CHECK:       [[NCE_PERMUTE:%.*]] = VPU.NCE.ClusterTiling ([[COPY_INPUT]] as %arg1: tensor<1x3x224x224xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x4x224x224x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3],
    // CHECK-SAME:      pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:      strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}> {

    // CHECK:       VPU.NCE.Permute
    // CHECK-SAME:      {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64
    // CHECK-SAME:      -> tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:      -> !VPU.DistributedTensor<1x4x224x224x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3],
    // CHECK-SAME:      pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:      strides = [2, 2], num_clusters = 2 : i64}> {
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

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

    // CHECK:       [[COPY_INPUT:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x3x224x224xf16>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x3x224x224xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {

    // CHECK:       [[NCE_PERMUTE:%.*]] = VPU.NCE.ClusterTiling ([[COPY_INPUT]] as %arg1: tensor<1x3x224x224xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x4x224x224x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {

    // CHECK:       VPU.NCE.Permute
    // CHECK-SAME:      {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:      -> tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:      -> !VPU.DistributedTensor<1x4x224x224x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3],
    // CHECK-SAME:      pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:      strides = [1, 1], num_clusters = 2 : i64}> {
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

    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg2: tensor<1x32x44x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling (%arg1 as %arg2: tensor<1x1x44x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x1x44x44xf16> -> tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[DIVIDE:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK:                                                [[INPUT1]] as %arg3: tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_DIVIDE:%.*]] = VPU.Divide(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[DIVIDE]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x44x44xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xf16>

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

    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg2: tensor<1x32x44x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling (%arg1 as %arg2: tensor<1x1x1x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[DIVIDE:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK:                                                [[INPUT1]] as %arg3: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_DIVIDE:%.*]] = VPU.Divide(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[DIVIDE]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x44x44xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xf16>

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

    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg2: tensor<1x32x1x44xf16, {order = #NHWC}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x1x44xf16, {order = #NHWC}> -> tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling (%arg1 as %arg2: tensor<1x32x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[POWER:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as %arg2: tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK:                                                  [[INPUT1]] as %arg3: tensor<1x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> !VPU.DistributedTensor<1x32x1x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_POWER:%.*]] = VPU.Power(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[POWER]] as %arg2: tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x1x44xf16, {order = #NHWC}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) : tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x1x44xf16, {order = #NHWC}>

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

    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg2: tensor<1x32x1x44xf16, {order = #NHWC}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x1x44xf16, {order = #NHWC}> -> tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling (%arg1 as %arg2: tensor<1x1x1x44xf16, {order = #NHWC}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x1x1x44xf16, {order = #NHWC}> -> tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[POWER:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as %arg2: tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK:                                                  [[INPUT1]] as %arg3: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> !VPU.DistributedTensor<1x32x1x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_POWER:%.*]] = VPU.Power(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[POWER]] as %arg2: tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x1x44xf16, {order = #NHWC}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) : tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x1x44xf16, {order = #NHWC}>

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

    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_0]] as %arg2: tensor<1x32x44x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_1]] as %arg2: tensor<1x1x44x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x1x44x44xf16> -> tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[GREATER:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK:                                                [[INPUT1]] as %arg3: tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x32x44x44xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_GREATER:%.*]] = VPU.Greater(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[GREATER]] as %arg2: tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x44x44xi8> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) : tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xi8>

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

    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_0]] as %arg2: tensor<1x32x44x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_1]] as %arg2: tensor<1x1x1x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[GREATER:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK:                                                [[INPUT1]] as %arg3: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x32x44x44xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_GREATER:%.*]] = VPU.Greater(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[GREATER]] as %arg2: tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x44x44xi8> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) : tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xi8>

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

    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_0]] as %arg2: tensor<1x32x44x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_1]] as %arg2: tensor<1x1x44x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x1x44x44xf16> -> tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[LESS:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK:                                                [[INPUT1]] as %arg3: tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x32x44x44xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_LESS:%.*]] = VPU.Less(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[LESS]] as %arg2: tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x44x44xi8> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) : tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xi8>

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

    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_0]] as %arg2: tensor<1x32x44x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_1]] as %arg2: tensor<1x1x1x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[LESS:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK:                                                [[INPUT1]] as %arg3: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x32x44x44xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_LESS:%.*]] = VPU.Less(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[LESS]] as %arg2: tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x44x44xi8> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) : tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xi8>

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

    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_0]] as {{[^:]+}}: tensor<1x32x44x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy({{[^:]+}}) {out_mem_space = @CMX_NN} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_1]] as {{[^:]+}}: tensor<1x1x44x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy({{[^:]+}}) {out_mem_space = @CMX_NN} : tensor<1x1x44x44xf16> -> tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[EQUAL:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as {{[^:]+}}: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK:                                                [[INPUT1]] as {{[^:]+}}: tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x32x44x44xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_EQUAL:%.*]] = VPU.Equal({{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[EQUAL]] as {{[^:]+}}: tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x44x44xi8> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy({{[^:]+}}) : tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xi8>

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

    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_0]] as {{[^:]+}}: tensor<1x32x44x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy({{[^:]+}}) {out_mem_space = @CMX_NN} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_1]] as {{[^:]+}}: tensor<1x1x1x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy({{[^:]+}}) {out_mem_space = @CMX_NN} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[EQUAL:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as {{[^:]+}}: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK:                                                [[INPUT1]] as {{[^:]+}}: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x32x44x44xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_EQUAL:%.*]] = VPU.Equal({{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[EQUAL]] as {{[^:]+}}: tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x44x44xi8> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy({{[^:]+}}) : tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xi8>

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

    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_0]] as %arg2: tensor<1x32x44x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_1]] as %arg2: tensor<1x1x44x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x1x44x44xf16> -> tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[SUBTRACT:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK:                                                [[INPUT1]] as %arg3: tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_SUBTRACT:%.*]] = VPU.Subtract(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[SUBTRACT]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x44x44xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xf16>

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

    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_0]] as %arg2: tensor<1x32x44x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_1]] as %arg2: tensor<1x1x1x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[SUBTRACT:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK:                                                [[INPUT1]] as %arg3: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_SUBTRACT:%.*]] = VPU.Subtract(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[SUBTRACT]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x44x44xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xf16>

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

    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_0]] as %arg2: tensor<1x32x44x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_1]] as %arg2: tensor<1x1x44x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x1x44x44xf16> -> tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[ADD:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK:                                                [[INPUT1]] as %arg3: tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_ADD:%.*]] = VPU.Add(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[ADD]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x44x44xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xf16>

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

    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_0]] as %arg2: tensor<1x32x44x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_1]] as %arg2: tensor<1x1x1x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[ADD:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK:                                                [[INPUT1]] as %arg3: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_ADD:%.*]] = VPU.Add(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[ADD]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x44x44xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xf16>

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

    //CHECK:        [[MVN_COPY_IN:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x1x320x1xf16>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x1x320x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:                    VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x1x320x1xf16>
    //CHECK-SAME:                    -> tensor<1x1x320x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[MVN:%.*]] = VPU.NCE.ClusterTiling ([[MVN_COPY_IN]] as %arg1: tensor<1x1x320x1xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x1x320x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:                    VPU.MVN(%arg1) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x1x320x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-SAME:                   -> tensor<1x1x320x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[MVN_COPY_OUT:%.*]] = VPU.NCE.ClusterTiling ([[MVN]] as %arg1: tensor<1x1x320x1xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> tensor<1x1x320x1xf16> {
    //CHECK:                    VPU.Copy(%arg1) : tensor<1x1x320x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-SAME:                   -> tensor<1x1x320x1xf16>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[RESHAPE:%.*]] = VPU.AffineReshape([[MVN_COPY_OUT]])
    //CHECK-SAME{{LITERAL}}:    {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 320, 1, 1]} : tensor<1x1x320x1xf16> -> tensor<1x320x1x1xf16>

    //CHECK:        [[CAST:%.*]] = VPU.PermuteCast([[RESHAPE]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x320x1x1xf16> -> tensor<1x320x1x1xf16, {order = #NHWC}>

    //CHECK:        [[AVERAGEPOOL_INPUT_COPY_IN:%.*]] = VPU.NCE.ClusterTiling ([[CAST]] as %arg1: tensor<1x320x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:                    VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x320x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:                   -> tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[AVERAGEPOOL:%.*]]  = VPU.NCE.ClusterTiling ([[AVERAGEPOOL_INPUT_COPY_IN]] as %arg1: tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:                    VPU.NCE.AveragePool(%arg1) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]}
    //CHECK-SAME:                    -> tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[AVERAGEPOOL_COPY_OUT:%.*]] = VPU.NCE.ClusterTiling ([[AVERAGEPOOL]] as %arg1: tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x320x1x1xf16, {order = #NHWC}> {
    //CHECK:                    VPU.Copy(%arg1) : tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:                  -> tensor<1x320x1x1xf16, {order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[SIGMOID_COPY_IN:%.*]] = VPU.NCE.ClusterTiling ([[AVERAGEPOOL_COPY_OUT]] as %arg1: tensor<1x320x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:                    VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x320x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:                   -> tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[SIGMOID:%.*]] = VPU.NCE.ClusterTiling ([[SIGMOID_COPY_IN]] as %arg1: tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:                    VPU.Sigmoid(%arg1) : tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:                   -> tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[SIGMOID_COPY_OUT:%.*]] = VPU.NCE.ClusterTiling ([[SIGMOID]] as %arg1: tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x320x1x1xf16, {order = #NHWC}> {
    //CHECK:                    VPU.Copy(%arg1) : tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                        -> tensor<1x320x1x1xf16, {order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }
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

    // CHECK: [[TILING_COPY_1:%.*]] = VPU.NCE.ClusterTiling
    // CHECK: [[TILING_COPY_2:%.*]] = VPU.NCE.ClusterTiling
    // CHECK: [[TILING_COPY_3:%.*]] = VPU.NCE.ClusterTiling
    // CHECK: [[DWCONV_1:%.*]] = VPU.NCE.ClusterTiling
    // CHECK: [[TILING_COPY_OUT_1:%.*]] = VPU.NCE.ClusterTiling

    // CHECK: [[TILING_COPY_4:%.*]] = VPU.NCE.ClusterTiling
    // CHECK: [[TILING_COPY_5:%.*]] = VPU.NCE.ClusterTiling
    // CHECK: [[TILING_COPY_6:%.*]] = VPU.NCE.ClusterTiling
    // CHECK: [[DWCONV_2:%.*]] = VPU.NCE.ClusterTiling
    // CHECK: [[TILING_COPY_OUT_2:%.*]] = VPU.NCE.ClusterTiling

    // CHECK: [[TILING_COPY_7:%.*]] = VPU.NCE.ClusterTiling
    // CHECK: [[TILING_COPY_8:%.*]] = VPU.NCE.ClusterTiling
    // CHECK: [[TILING_COPY_9:%.*]] = VPU.NCE.ClusterTiling
    // CHECK: [[DWCONV_3:%.*]] = VPU.NCE.ClusterTiling
    // CHECK: [[TILING_COPY_OUT_3:%.*]] = VPU.NCE.ClusterTiling

    // CHECK: [[CONCAT:%.*]] = VPU.Concat([[TILING_COPY_OUT_1]], [[TILING_COPY_OUT_2]], [[TILING_COPY_OUT_3]])
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
    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG1:%.+]]: tensor<1x8x128x128xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x8x128x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x8x128x128xf16> -> tensor<1x8x128x128xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling ([[SLOPE]] as [[ARG2:%.+]]: tensor<1x8x1x1xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x8x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG2]]) {out_mem_space = @CMX_NN} : tensor<1x8x1x1xf16> -> tensor<1x8x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[PRELU:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as [[ARG3:%.+]]: tensor<1x8x128x128xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK:                                               [[INPUT1]] as [[ARG4:%.+]]: tensor<1x8x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x8x128x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_POWER:%.*]] = VPU.PRelu([[ARG3]], [[ARG4]]) : tensor<1x8x128x128xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x8x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x8x128x128xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[PRELU]] as [[ARG5:%.+]]: tensor<1x8x128x128xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x8x128x128xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG5]]) : tensor<1x8x128x128xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x8x128x128xf16>

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
    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG1:%.+]]: tensor<1x128x1x1xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x128x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x128x1x1xf16> -> tensor<1x128x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling ([[SLOPE]] as [[ARG2:%.+]]: tensor<1x128x1x1xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x128x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG2]]) {out_mem_space = @CMX_NN} : tensor<1x128x1x1xf16> -> tensor<1x128x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[PRELU:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as [[ARG3:%.+]]: tensor<1x128x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK:                                               [[INPUT1]] as [[ARG4:%.+]]: tensor<1x128x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x128x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_POWER:%.*]] = VPU.PRelu([[ARG3]], [[ARG4]]) : tensor<1x128x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x128x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x128x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[PRELU]] as [[ARG5:%.+]]: tensor<1x128x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x128x1x1xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG5]]) : tensor<1x128x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x128x1x1xf16>

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
    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG1:%.+]]: tensor<1x8x128x128xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x8x128x128xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x8x128x128xf16> -> tensor<1x8x128x128xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling ([[SLOPE]] as [[ARG2:%.+]]: tensor<1x8x1x1xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x8x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG2]]) {out_mem_space = @CMX_NN} : tensor<1x8x1x1xf16> -> tensor<1x8x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[PRELU:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as [[ARG3:%.+]]: tensor<1x8x128x128xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK:                                               [[INPUT1]] as [[ARG4:%.+]]: tensor<1x8x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x8x128x128xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_POWER:%.*]] = VPU.PRelu([[ARG3]], [[ARG4]]) : tensor<1x8x128x128xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x8x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x8x128x128xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[PRELU]] as [[ARG5:%.+]]: tensor<1x8x128x128xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x8x128x128xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG5]]) : tensor<1x8x128x128xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x8x128x128xf16>

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

    //CHECK:        [[INPUT:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG0:%.+]]: tensor<1x1x1x128xsi64>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x128xsi64, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x128xsi64> -> tensor<1x1x1x128xsi64, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[CONVERT:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG1:%.+]]: tensor<1x1x1x128xsi64, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK:                            -> !VPU.DistributedTensor<1x1x1x128xf64, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_CONVERT:%.*]] = VPU.Convert([[ARG1]]) {dstElemType = f64} : tensor<1x1x1x128xsi64, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:                            -> tensor<1x1x1x128xf64, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[CONVERT]] as [[ARG2:%.+]]: tensor<1x1x1x128xf64, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x1x128xf64> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG2]]) : tensor<1x1x1x128xf64, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x1x128xf64>

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

    //CHECK:        [[INPUT:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG0:%.+]]: tensor<1x16x16x512xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x16x16x512xf16> -> tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[FLOOR:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG1:%.+]]: tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK:                            -> !VPU.DistributedTensor<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_FLOOR:%.*]] = VPU.Floor([[ARG1]]) : tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:                            -> tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[FLOOR]] as [[ARG2:%.+]]: tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x16x16x512xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG2]]) : tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x16x512xf16>

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

    //CHECK:        [[INPUT:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG0:%.+]]: tensor<1x16x1x513xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x1x513xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x16x1x513xf16> -> tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[FLOOR:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG1:%.+]]: tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK:                            -> !VPU.DistributedTensor<1x16x1x513xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_FLOOR:%.*]] = VPU.Floor([[ARG1]]) : tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:                            -> tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[FLOOR]] as [[ARG2:%.+]]: tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x16x1x513xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG2]]) : tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x1x513xf16>

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

    //CHECK:        [[INPUT:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG0:%.+]]: tensor<1x1x1x513xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x513xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x513xf16> -> tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[FLOOR:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG1:%.+]]: tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK:                            -> !VPU.DistributedTensor<1x1x1x513xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_FLOOR:%.*]] = VPU.Floor([[ARG1]]) : tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:                            -> tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[FLOOR]] as [[ARG2:%.+]]: tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x1x513xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG2]]) : tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x1x513xf16>

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

    //CHECK:        [[INPUT:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG0:%.+]]: tensor<1x16x16x512xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x16x16x512xf16> -> tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[ROUND:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG1:%.+]]: tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK:                            -> !VPU.DistributedTensor<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_ROUND:%.*]] = VPU.Round([[ARG1]]) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:                            -> tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[ROUND]] as [[ARG2:%.+]]: tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x16x16x512xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG2]]) : tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x16x512xf16>

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

    //CHECK:        [[INPUT:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG0:%.+]]: tensor<1x16x1x513xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x1x513xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x16x1x513xf16> -> tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[ROUND:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG1:%.+]]: tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK:                            -> !VPU.DistributedTensor<1x16x1x513xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_ROUND:%.*]] = VPU.Round([[ARG1]]) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:                            -> tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[ROUND]] as [[ARG2:%.+]]: tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x16x1x513xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG2]]) : tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x1x513xf16>

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

    //CHECK:        [[INPUT:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG0:%.+]]: tensor<1x1x1x513xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x513xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x513xf16> -> tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[ROUND:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG1:%.+]]: tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK:                            -> !VPU.DistributedTensor<1x1x1x513xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_ROUND:%.*]] = VPU.Round([[ARG1]]) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:                            -> tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[ROUND]] as [[ARG2:%.+]]: tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x1x513xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG2]]) : tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x1x513xf16>

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

    // CHECK:   [[COPY_LHS:%.*]] = VPU.NCE.ClusterTiling ([[LHS]] as [[LHS_COPY_ARG:%.*]]: tensor<1x64x16x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[LHS_COPY_RES:%.*]] = VPU.Copy([[LHS_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[LHS_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_RHS:%.*]] = VPU.NCE.ClusterTiling ([[RHS]] as [[RHS_COPY_ARG:%.*]]: tensor<1x64x16x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[RHS_COPY_RES:%.*]] = VPU.Copy([[RHS_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[RHS_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_LHS_SCALES:%.*]] = VPU.NCE.ClusterTiling ([[LHS_SCALES]] as [[LHS_SCALES_COPY_ARG:%.*]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[LHS_SCALES_COPY_RES:%.*]] = VPU.Copy([[LHS_SCALES_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[LHS_SCALES_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_RHS_SCALES:%.*]] = VPU.NCE.ClusterTiling ([[RHS_SCALES]] as [[RHS_SCALES_COPY_ARG:%.*]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[RHS_SCALES_COPY_RES:%.*]] = VPU.Copy([[RHS_SCALES_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[RHS_SCALES_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[ACCUMULATE:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:      [[COPY_LHS]] as [[LHS_ARG:%arg[0-9]]]:
    // CHECK-SAME:      [[COPY_RHS]] as [[RHS_ARG:%arg[0-9]]]:
    // CHECK-SAME:      [[COPY_LHS_SCALES]] as [[LHS_SCALES_ARG:%arg[0-9]]]:
    // CHECK-SAME:      [[COPY_RHS_SCALES]] as [[RHS_SCALES_ARG:%arg[0-9]]]:
    // CHECK-SAME:  ) -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[ACCUMULATE_RES:%.*]] = VPU.Accumulate([[LHS_ARG]], [[RHS_ARG]], [[LHS_SCALES_ARG]], [[RHS_SCALES_ARG]])
    // CHECK:       VPU.Yield [[ACCUMULATE_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_OUT:%.*]] = VPU.NCE.ClusterTiling ([[ACCUMULATE]] as [[OUT_COPY_ARG:%.*]]: tensor<1x64x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:  -> tensor<1x64x16x1xf16, {order = #NHWC}> {
    // CHECK:       [[OUT_COPY_RES:%.*]] = VPU.Copy([[OUT_COPY_ARG]]) :
    // CHECK-SAME:      tensor<1x64x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK:       VPU.Yield [[OUT_COPY_RES]]
    // CHECK:   }

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

    // CHECK:   [[COPY_LHS:%.*]] = VPU.NCE.ClusterTiling ([[LHS]] as [[LHS_COPY_ARG:%.*]]: tensor<1x64x16x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[LHS_COPY_RES:%.*]] = VPU.Copy([[LHS_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[LHS_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_RHS:%.*]] = VPU.NCE.ClusterTiling ([[RHS]] as [[RHS_COPY_ARG:%.*]]: tensor<1x64x16x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[RHS_COPY_RES:%.*]] = VPU.Copy([[RHS_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[RHS_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_LHS_SCALES:%.*]] = VPU.NCE.ClusterTiling ([[LHS_SCALES]] as [[LHS_SCALES_COPY_ARG:%.*]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[LHS_SCALES_COPY_RES:%.*]] = VPU.Copy([[LHS_SCALES_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[LHS_SCALES_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_RHS_SCALES:%.*]] = VPU.NCE.ClusterTiling ([[RHS_SCALES]] as [[RHS_SCALES_COPY_ARG:%.*]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[RHS_SCALES_COPY_RES:%.*]] = VPU.Copy([[RHS_SCALES_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[RHS_SCALES_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[ACCUMULATE:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:      [[COPY_LHS]] as [[LHS_ARG:%arg[0-9]]]:
    // CHECK-SAME:      [[COPY_RHS]] as [[RHS_ARG:%arg[0-9]]]:
    // CHECK-SAME:      [[COPY_LHS_SCALES]] as [[LHS_SCALES_ARG:%arg[0-9]]]:
    // CHECK-SAME:      [[COPY_RHS_SCALES]] as [[RHS_SCALES_ARG:%arg[0-9]]]:
    // CHECK-SAME:  ) -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[ACCUMULATE_RES:%.*]] = VPU.Accumulate([[LHS_ARG]], [[RHS_ARG]], [[LHS_SCALES_ARG]], [[RHS_SCALES_ARG]])
    // CHECK:       VPU.Yield [[ACCUMULATE_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_OUT:%.*]] = VPU.NCE.ClusterTiling ([[ACCUMULATE]] as [[OUT_COPY_ARG:%.*]]: tensor<1x64x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:  -> tensor<1x64x16x1xf16, {order = #NHWC}> {
    // CHECK:       [[OUT_COPY_RES:%.*]] = VPU.Copy([[OUT_COPY_ARG]]) :
    // CHECK-SAME:      tensor<1x64x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK:       VPU.Yield [[OUT_COPY_RES]]
    // CHECK:   }

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

    // CHECK:   [[COPY_LHS:%.*]] = VPU.NCE.ClusterTiling ([[LHS]] as [[LHS_COPY_ARG:%.*]]: tensor<1x64x16x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 2, 1, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[LHS_COPY_RES:%.*]] = VPU.Copy([[LHS_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[LHS_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_RHS:%.*]] = VPU.NCE.ClusterTiling ([[RHS]] as [[RHS_COPY_ARG:%.*]]: tensor<1x64x16x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 2, 1, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[RHS_COPY_RES:%.*]] = VPU.Copy([[RHS_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[RHS_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_LHS_SCALES:%.*]] = VPU.NCE.ClusterTiling ([[LHS_SCALES]] as [[LHS_SCALES_COPY_ARG:%.*]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 2, 1, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[LHS_SCALES_COPY_RES:%.*]] = VPU.Copy([[LHS_SCALES_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[LHS_SCALES_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_RHS_SCALES:%.*]] = VPU.NCE.ClusterTiling ([[RHS_SCALES]] as [[RHS_SCALES_COPY_ARG:%.*]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 2, 1, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[RHS_SCALES_COPY_RES:%.*]] = VPU.Copy([[RHS_SCALES_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[RHS_SCALES_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[ACCUMULATE:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:      [[COPY_LHS]] as [[LHS_ARG:%arg[0-9]]]:
    // CHECK-SAME:      [[COPY_RHS]] as [[RHS_ARG:%arg[0-9]]]:
    // CHECK-SAME:      [[COPY_LHS_SCALES]] as [[LHS_SCALES_ARG:%arg[0-9]]]:
    // CHECK-SAME:      [[COPY_RHS_SCALES]] as [[RHS_SCALES_ARG:%arg[0-9]]]:
    // CHECK-SAME:  ) -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 2, 1, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[ACCUMULATE_RES:%.*]] = VPU.Accumulate([[LHS_ARG]], [[RHS_ARG]], [[LHS_SCALES_ARG]], [[RHS_SCALES_ARG]])
    // CHECK:       VPU.Yield [[ACCUMULATE_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_OUT:%.*]] = VPU.NCE.ClusterTiling ([[ACCUMULATE]] as [[OUT_COPY_ARG:%.*]]: tensor<1x64x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:  -> tensor<1x64x16x1xf16, {order = #NHWC}> {
    // CHECK:       [[OUT_COPY_RES:%.*]] = VPU.Copy([[OUT_COPY_ARG]]) :
    // CHECK-SAME:      tensor<1x64x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK:       VPU.Yield [[OUT_COPY_RES]]
    // CHECK:   }

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

    // CHECK:   [[COPY_LHS:%.*]] = VPU.NCE.ClusterTiling ([[LHS]] as [[LHS_COPY_ARG:%.*]]: tensor<1x64x16x32xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x32xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, 2],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[LHS_COPY_RES:%.*]] = VPU.Copy([[LHS_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x16x32xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[LHS_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_RHS:%.*]] = VPU.NCE.ClusterTiling ([[RHS]] as [[RHS_COPY_ARG:%.*]]: tensor<1x64x16x32xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x32xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, 2],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[RHS_COPY_RES:%.*]] = VPU.Copy([[RHS_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x16x32xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[RHS_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_LHS_SCALES:%.*]] = VPU.NCE.ClusterTiling ([[LHS_SCALES]] as [[LHS_SCALES_COPY_ARG:%.*]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[LHS_SCALES_COPY_RES:%.*]] = VPU.Copy([[LHS_SCALES_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[LHS_SCALES_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_RHS_SCALES:%.*]] = VPU.NCE.ClusterTiling ([[RHS_SCALES]] as [[RHS_SCALES_COPY_ARG:%.*]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[RHS_SCALES_COPY_RES:%.*]] = VPU.Copy([[RHS_SCALES_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[RHS_SCALES_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[ACCUMULATE:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:      [[COPY_LHS]] as [[LHS_ARG:%arg[0-9]]]:
    // CHECK-SAME:      [[COPY_RHS]] as [[RHS_ARG:%arg[0-9]]]:
    // CHECK-SAME:      [[COPY_LHS_SCALES]] as [[LHS_SCALES_ARG:%arg[0-9]]]:
    // CHECK-SAME:      [[COPY_RHS_SCALES]] as [[RHS_SCALES_ARG:%arg[0-9]]]:
    // CHECK-SAME:  ) -> !VPU.DistributedTensor<1x64x16x32xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, 2],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[ACCUMULATE_RES:%.*]] = VPU.Accumulate([[LHS_ARG]], [[RHS_ARG]], [[LHS_SCALES_ARG]], [[RHS_SCALES_ARG]])
    // CHECK:       VPU.Yield [[ACCUMULATE_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_OUT:%.*]] = VPU.NCE.ClusterTiling ([[ACCUMULATE]] as [[OUT_COPY_ARG:%.*]]: tensor<1x64x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:  -> tensor<1x64x16x32xf16, {order = #NHWC}> {
    // CHECK:       [[OUT_COPY_RES:%.*]] = VPU.Copy([[OUT_COPY_ARG]]) :
    // CHECK-SAME:      tensor<1x64x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x64x16x32xf16, {order = #NHWC}>
    // CHECK:       VPU.Yield [[OUT_COPY_RES]]
    // CHECK:   }

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

    // CHECK:   [[COPY_INPUT:%.*]] = VPU.NCE.ClusterTiling ([[DATA]] as [[INPUT_COPY_ARG:%.*]]: tensor<1x16x32x64xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x16x32x64xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "OVERLAPPED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, 2],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[INPUT_COPY_RES:%.*]] = VPU.Copy([[INPUT_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x16x32x64xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[INPUT_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[POOL:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:      [[COPY_INPUT]] as [[INPUT_ARG:%arg[0-9]]]: tensor<1x16x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:  ) -> !VPU.DistributedTensor<1x16x32x64xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, 2],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[POOL_RES:%.*]] = VPU.NCE.MaxPool([[INPUT_ARG]])
    // CHECK:       VPU.Yield [[POOL_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_OUT:%.*]] = VPU.NCE.ClusterTiling ([[POOL]] as [[OUT_COPY_ARG:%.*]]: tensor<1x16x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:  -> tensor<1x16x32x64xf16, {order = #NHWC}> {
    // CHECK:       [[OUT_COPY_RES:%.*]] = VPU.Copy([[OUT_COPY_ARG]]) :
    // CHECK-SAME:      tensor<1x16x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x16x32x64xf16, {order = #NHWC}>
    // CHECK:       VPU.Yield [[OUT_COPY_RES]]
    // CHECK:   }

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

    //CHECK: [[INPUT_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG1:[^:]+]]: tensor<1x3x384x640xf16>) -> !VPU.DistributedTensor<1x3x384x640xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK: VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x3x384x640xf16> -> tensor<1x3x384x640xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[IN_LOW_COPY:%.+]] = VPU.NCE.ClusterTiling ([[IN_LOW]] as [[ARG1:[^:]+]]: tensor<1x3x1x1xf16>) -> !VPU.DistributedTensor<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:    VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x3x1x1xf16> -> tensor<1x3x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[IN_HIGH_COPY:%.+]]  = VPU.NCE.ClusterTiling ([[IN_HIGH]] as [[ARG1:[^:]+]]: tensor<1x3x1x1xf16>) -> !VPU.DistributedTensor<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:    VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x3x1x1xf16> -> tensor<1x3x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[OUT_LOW_COPY:%.+]] = VPU.NCE.ClusterTiling ([[OUT_LOW]] as [[ARG1:[^:]+]]: tensor<1x1x1x1xf16>) -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:    VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[OUT_HIGH_COPY:%.+]] = VPU.NCE.ClusterTiling ([[OUT_HIGH]] as [[ARG1:[^:]+]]: tensor<1x1x1x1xf16>) -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:    VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[FQ_CLUSTER:%.+]] = VPU.NCE.ClusterTiling ([[INPUT_COPY]] as  [[ARG1:[^:]+]]: tensor<1x3x384x640xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[IN_LOW_COPY]] as  [[ARG2:[^:]+]]: tensor<1x3x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[IN_HIGH_COPY]] as  [[ARG3:[^:]+]]: tensor<1x3x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[OUT_LOW_COPY]] as  [[ARG4:[^:]+]]: tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[OUT_HIGH_COPY]] as  [[ARG5:[^:]+]]: tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x3x384x640xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:   VPU.FakeQuantize([[ARG1]], [[ARG2]], [[ARG3]], [[ARG4]], [[ARG5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x384x640xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x3x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x3x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x3x384x640xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[OUTPUT:%.+]] = VPU.NCE.ClusterTiling ([[FQ_CLUSTER]] as [[ARG1:[^:]+]]: tensor<1x3x384x640xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x3x384x640xf16> {
    //CHECK:     VPU.Copy([[ARG1]]) : tensor<1x3x384x640xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x3x384x640xf16>
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

    //CHECK: [[INPUT_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG1:[^:]+]]: tensor<1x128x1x512xf16>) -> !VPU.DistributedTensor<1x128x1x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK: VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x128x1x512xf16> -> tensor<1x128x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[IN_LOW_COPY:%.+]] = VPU.NCE.ClusterTiling ([[IN_LOW]] as [[ARG1:[^:]+]]: tensor<1x1x1x1xf16>) -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:    VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[IN_HIGH_COPY:%.+]]  = VPU.NCE.ClusterTiling ([[IN_HIGH]] as [[ARG1:[^:]+]]: tensor<1x1x1x1xf16>) -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:    VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[OUT_LOW_COPY:%.+]] = VPU.NCE.ClusterTiling ([[OUT_LOW]] as [[ARG1:[^:]+]]: tensor<1x128x1x1xf16>) -> !VPU.DistributedTensor<1x128x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:    VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x128x1x1xf16> -> tensor<1x128x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[OUT_HIGH_COPY:%.+]] = VPU.NCE.ClusterTiling ([[OUT_HIGH]] as [[ARG1:[^:]+]]: tensor<1x128x1x1xf16>) -> !VPU.DistributedTensor<1x128x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:    VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x128x1x1xf16> -> tensor<1x128x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[FQ_CLUSTER:%.+]] = VPU.NCE.ClusterTiling ([[INPUT_COPY]] as [[ARG1:[^:]+]]: tensor<1x128x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[IN_LOW_COPY]] as [[ARG2:[^:]+]]: tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[IN_HIGH_COPY]] as [[ARG3:[^:]+]]: tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[OUT_LOW_COPY]] as [[ARG4:[^:]+]]: tensor<1x128x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[OUT_HIGH_COPY]] as [[ARG5:[^:]+]]: tensor<1x128x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x128x1x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:   VPU.FakeQuantize([[ARG1]], [[ARG2]], [[ARG3]], [[ARG4]], [[ARG5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x128x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x128x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x128x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x128x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[OUTPUT:%.+]] = VPU.NCE.ClusterTiling ([[FQ_CLUSTER]] as [[ARG1:[^:]+]]: tensor<1x128x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x128x1x512xf16> {
    //CHECK:     VPU.Copy([[ARG1]]) : tensor<1x128x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x128x1x512xf16>
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

    //CHECK: [[INPUT_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG1:[^:]+]]: tensor<1x1x1x512xf16>) -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK: VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x512xf16> -> tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[IN_LOW_COPY:%.+]] = VPU.NCE.ClusterTiling ([[IN_LOW]] as [[ARG1:[^:]+]]: tensor<1x1x1x512xf16>) -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK: VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x512xf16> -> tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[IN_HIGH_COPY:%.+]] = VPU.NCE.ClusterTiling ([[IN_HIGH]] as [[ARG1:[^:]+]]: tensor<1x1x1x512xf16>) -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK: VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x512xf16> -> tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[OUT_LOW_COPY:%.+]] = VPU.NCE.ClusterTiling ([[OUT_LOW]] as [[ARG1:[^:]+]]: tensor<1x1x1x512xf16>) -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK: VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x512xf16> -> tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[OUT_HIGH_COPY:%.+]] = VPU.NCE.ClusterTiling ([[OUT_HIGH]] as [[ARG1:[^:]+]]: tensor<1x1x1x512xf16>) -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK: VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x512xf16> -> tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[FQ_CLUSTER:%.+]] = VPU.NCE.ClusterTiling ([[INPUT_COPY]] as  [[ARG1:[^:]+]]: tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[IN_LOW_COPY]] as  [[ARG2:[^:]+]]: tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[IN_HIGH_COPY]] as  [[ARG3:[^:]+]]: tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[OUT_LOW_COPY]] as  [[ARG4:[^:]+]]: tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[OUT_HIGH_COPY]] as  [[ARG5:[^:]+]]: tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:   VPU.FakeQuantize([[ARG1]], [[ARG2]], [[ARG3]], [[ARG4]], [[ARG5]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>, levels = 256 : i64} : tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[OUTPUT:%.+]] = VPU.NCE.ClusterTiling ([[FQ_CLUSTER]] as [[ARG1:[^:]+]]: tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x1x512xf16> {
    //CHECK:     VPU.Copy([[ARG1]]) : tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x1x512xf16>
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

    //CHECK:        [[INPUT_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[INPUT_ARG:%arg[0-9]]]: tensor<1x10x40x40xf16>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x10x40x40xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            VPU.Copy([[INPUT_ARG]]) {out_mem_space = @CMX_NN} : tensor<1x10x40x40xf16> -> tensor<1x10x40x40xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT0_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT0]] as [[INPUT0_ARG:%arg[0-9]]]: tensor<1x1x1x1xf16>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            VPU.Copy([[INPUT0_ARG]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT1_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT1]] as [[INPUT1_ARG:%arg[0-9]]]: tensor<1x10x40x40xf16>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x10x40x40xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            VPU.Copy([[INPUT1_ARG]]) {out_mem_space = @CMX_NN} : tensor<1x10x40x40xf16> -> tensor<1x10x40x40xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[SELECT:%.+]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:           [[INPUT_COPY]] as [[INPUT_ARG:%arg[0-9]]]: tensor<1x10x40x40xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK-SAME:           [[INPUT0_COPY]] as [[INPUT0_ARG:%arg[0-9]]]: tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK-SAME:           [[INPUT1_COPY]] as [[INPUT1_ARG:%arg[0-9]]]: tensor<1x10x40x40xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-SAME:       ) -> !VPU.DistributedTensor<1x10x40x40xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:        VPU.Select([[INPUT_ARG]], [[INPUT0_ARG]], [[INPUT1_ARG]]) {
    //CHECK-SAME:       auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x40x40xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x10x40x40xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x10x40x40xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.NCE.ClusterTiling ([[SELECT]] as [[OUTPUT_ARG:%arg[0-9]]]: tensor<1x10x40x40xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x10x40x40xf16> {
    //CHECK:            VPU.Copy([[OUTPUT_ARG]]) : tensor<1x10x40x40xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x10x40x40xf16>

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

    //CHECK:        [[INPUT_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[INPUT_ARG:%arg[0-9]]]: tensor<1x16x1x40xf16>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x1x40xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:            VPU.Copy([[INPUT_ARG]]) {out_mem_space = @CMX_NN} : tensor<1x16x1x40xf16> -> tensor<1x16x1x40xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT0_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT0]] as [[INPUT0_ARG:%arg[0-9]]]: tensor<1x1x1x1xf16>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            VPU.Copy([[INPUT0_ARG]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT1_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT1]] as [[INPUT1_ARG:%arg[0-9]]]: tensor<1x16x1x40xf16>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x1x40xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:            VPU.Copy([[INPUT1_ARG]]) {out_mem_space = @CMX_NN} : tensor<1x16x1x40xf16> -> tensor<1x16x1x40xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[SELECT:%.+]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:           [[INPUT_COPY]] as [[INPUT_ARG:%arg[0-9]]]: tensor<1x16x1x40xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK-SAME:           [[INPUT0_COPY]] as [[INPUT0_ARG:%arg[0-9]]]: tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK-SAME:           [[INPUT1_COPY]] as [[INPUT1_ARG:%arg[0-9]]]: tensor<1x16x1x40xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-SAME:       ) -> !VPU.DistributedTensor<1x16x1x40xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:        VPU.Select([[INPUT_ARG]], [[INPUT0_ARG]], [[INPUT1_ARG]]) {
    //CHECK-SAME:       auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x40xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x16x1x40xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x1x40xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.NCE.ClusterTiling ([[SELECT]] as [[OUTPUT_ARG:%arg[0-9]]]: tensor<1x16x1x40xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x16x1x40xf16> {
    //CHECK:            VPU.Copy([[OUTPUT_ARG]]) : tensor<1x16x1x40xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x1x40xf16>

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

    //CHECK:        [[INPUT0:%.+]] = VPU.NCE.ClusterTiling ([[INPUT_0]] as %arg2: tensor<1x1x100x2048xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x100x2048xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.+]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x1x100x2048xf16> -> tensor<1x1x100x2048xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT1:%.+]] = VPU.NCE.ClusterTiling ([[INPUT_1]] as %arg2: tensor<1x1x100x512xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x100x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.+]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x1x100x512xf16> -> tensor<1x1x100x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[LSTMGATES:%.+]]:2 = VPU.NCE.ClusterTiling (
    //CHECK-SAME:         [[INPUT0]] as %arg2: tensor<1x1x100x2048xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK-SAME:         [[INPUT1]] as %arg3: tensor<1x1x100x512xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-SAME:         ) -> (!VPU.DistributedTensor<1x1x100x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPU.DistributedTensor<1x1x100x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    //CHECK:          [[INNER_LSTMGATES_0:%.+]], [[INNER_LSTMGATES_1:%.+]] = VPU.LSTMGates(%arg2, %arg3) : tensor<1x1x100x2048xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x100x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x100x512xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x100x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT0:%.+]] = VPU.NCE.ClusterTiling ([[LSTMGATES]]#0 as %arg2: tensor<1x1x100x512xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x100x512xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) : tensor<1x1x100x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x100x512xf16>

    //CHECK:        [[OUTPUT1:%.+]] = VPU.NCE.ClusterTiling ([[LSTMGATES]]#1 as %arg2: tensor<1x1x100x512xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x100x512xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) : tensor<1x1x100x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x100x512xf16>

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

    // CHECK:               [[INPUT0_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT0]] as [[INPUT0_COPY_ARG:%arg[0-9]]]: tensor<1x16x32x32xf16>) -> !VPU.DistributedTensor<1x16x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK-SAME:              {
    // CHECK:                       [[INPUT0_COPY_RES:%.+]] = VPU.Copy([[INPUT0_COPY_ARG]]) {out_mem_space = @CMX_NN} : tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:                       VPU.Yield [[INPUT0_COPY_RES]]
    // CHECK:                   }

    // CHECK:               [[INPUT1_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT1]] as [[INPUT1_COPY_ARG:%arg[0-9]]]: tensor<1x1x32x32xf16>) -> !VPU.DistributedTensor<1x1x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK-SAME:              {
    // CHECK:                       [[INPUT1_COPY_RES:%.+]] = VPU.Copy([[INPUT1_COPY_ARG]]) {out_mem_space = @CMX_NN} : tensor<1x1x32x32xf16> -> tensor<1x1x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:                       VPU.Yield [[INPUT1_COPY_RES]]
    // CHECK:                   }

    // CHECK:               [[AND:%.+]] = VPU.NCE.ClusterTiling ([[INPUT0_COPY]] as [[AND_LHS:%arg[0-9]]]: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[INPUT1_COPY]] as [[AND_RHS:%arg[0-9]]]: tensor<1x1x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x16x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK-SAME:              {
    // CHECK:                       [[AND_OUT:%.+]] = VPU.And([[AND_LHS]], [[AND_RHS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:                       VPU.Yield [[AND_OUT]]
    // CHECK:                   }

    // CHECK:               [[RES:%.+]] = VPU.NCE.ClusterTiling ([[AND]] as [[AND_COPY_ARG:%arg[0-9]]]: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x16x32x32xf16> {
    // CHECK:                   [[COPY_OUT:%.+]] = VPU.Copy([[AND_COPY_ARG]]) : tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x32x32xf16>
    // CHECK:                   VPU.Yield [[COPY_OUT]]
    // CHECK:               }

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

    //CHECK:        [[INPUT:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG0:%.+]]: tensor<1x4x2x512xf16, {order = #NHWC}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x4x2x512xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x4x2x512xf16, {order = #NHWC}> -> tensor<1x4x2x512xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[LOG_SOFTMAX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG1:%.+]]: tensor<1x4x2x512xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK:                            -> !VPU.DistributedTensor<1x4x2x512xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_LOG_SOFTMAX:%.*]] = VPU.LogSoftmax([[ARG1]]) {axisInd = 3 : i64} : tensor<1x4x2x512xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x4x2x512xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[LOG_SOFTMAX]] as [[ARG1:%.+]]: tensor<1x4x2x512xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x4x2x512xf16, {order = #NHWC}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG1]]) : tensor<1x4x2x512xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x4x2x512xf16, {order = #NHWC}>

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

    // CHECK:               [[INPUT0_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT0]] as [[INPUT0_COPY_ARG:%arg[0-9]]]: tensor<1x16x32x32xf16>) -> !VPU.DistributedTensor<1x16x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:              {
    // CHECK:                       [[INPUT0_COPY_RES:%.+]] = VPU.Copy([[INPUT0_COPY_ARG]]) {out_mem_space = @CMX_NN} : tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:                       VPU.Yield [[INPUT0_COPY_RES]]
    // CHECK:                   }

    // CHECK:               [[INPUT1_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT1]] as [[INPUT1_COPY_ARG:%arg[0-9]]]: tensor<1x1x32x32xf16>) -> !VPU.DistributedTensor<1x1x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:              {
    // CHECK:                       [[INPUT1_COPY_RES:%.+]] = VPU.Copy([[INPUT1_COPY_ARG]]) {out_mem_space = @CMX_NN} : tensor<1x1x32x32xf16> -> tensor<1x1x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:                       VPU.Yield [[INPUT1_COPY_RES]]
    // CHECK:                   }

    // CHECK:               [[AND:%.+]] = VPU.NCE.ClusterTiling ([[INPUT0_COPY]] as [[AND_LHS:%arg[0-9]]]: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[INPUT1_COPY]] as [[AND_RHS:%arg[0-9]]]: tensor<1x1x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x16x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:              {
    // CHECK:                       [[AND_OUT:%.+]] = VPU.And([[AND_LHS]], [[AND_RHS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:                       VPU.Yield [[AND_OUT]]
    // CHECK:                   }

    // CHECK:               [[RES:%.+]] = VPU.NCE.ClusterTiling ([[AND]] as [[AND_COPY_ARG:%arg[0-9]]]: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x16x32x32xf16> {
    // CHECK:                   [[COPY_OUT:%.+]] = VPU.Copy([[AND_COPY_ARG]]) : tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x32x32xf16>
    // CHECK:                   VPU.Yield [[COPY_OUT]]
    // CHECK:               }

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

    //CHECK:        [[INPUT:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG0:%.+]]: tensor<1x4x1x512xf16, {order = #NCHW}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x4x1x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[LOG_SOFTMAX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG1:%.+]]: tensor<1x4x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK:                            -> !VPU.DistributedTensor<1x4x1x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_LOG_SOFTMAX:%.*]] = VPU.LogSoftmax([[ARG1]]) {axisInd = 3 : i64} : tensor<1x4x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x4x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[LOG_SOFTMAX]] as [[ARG1:%.+]]: tensor<1x4x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x4x1x512xf16, {order = #NCHW}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG1]]) : tensor<1x4x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>

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

    // CHECK:               [[INPUT0_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT0]] as [[INPUT0_COPY_ARG:%arg[0-9]]]: tensor<1x64x32x32xf16>) -> !VPU.DistributedTensor<1x64x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:              {
    // CHECK:                       [[INPUT0_COPY_RES:%.+]] = VPU.Copy([[INPUT0_COPY_ARG]]) {out_mem_space = @CMX_NN} : tensor<1x64x32x32xf16> -> tensor<1x64x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:                       VPU.Yield [[INPUT0_COPY_RES]]
    // CHECK:                   }

    // CHECK:               [[INPUT1_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT1]] as [[INPUT1_COPY_ARG:%arg[0-9]]]: tensor<1x1x32x32xf16>) -> !VPU.DistributedTensor<1x1x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK-SAME:              {
    // CHECK:                       [[INPUT1_COPY_RES:%.+]] = VPU.Copy([[INPUT1_COPY_ARG]]) {out_mem_space = @CMX_NN} : tensor<1x1x32x32xf16> -> tensor<1x1x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:                       VPU.Yield [[INPUT1_COPY_RES]]
    // CHECK:                   }

    // CHECK:               [[AND:%.+]] = VPU.NCE.ClusterTiling ([[INPUT0_COPY]] as [[AND_LHS:%arg[0-9]]]: tensor<1x64x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[INPUT1_COPY]] as [[AND_RHS:%arg[0-9]]]: tensor<1x1x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x64x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:              {
    // CHECK:                       [[AND_OUT:%.+]] = VPU.And([[AND_LHS]], [[AND_RHS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x64x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:                       VPU.Yield [[AND_OUT]]
    // CHECK:                   }

    // CHECK:               [[RES:%.+]] = VPU.NCE.ClusterTiling ([[AND]] as [[AND_COPY_ARG:%arg[0-9]]]: tensor<1x64x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x64x32x32xf16> {
    // CHECK:                   [[COPY_OUT:%.+]] = VPU.Copy([[AND_COPY_ARG]]) : tensor<1x64x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x64x32x32xf16>
    // CHECK:                   VPU.Yield [[COPY_OUT]]
    // CHECK:               }

    // CHECK:               return [[RES]] : tensor<1x64x32x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @SinSWWithSOH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x44x44xf16>
func.func @SinSWWithSOH(%arg0: tensor<1x32x44x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Sin(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16>
    return %0 : tensor<1x32x44x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as {{[^:]+}}: tensor<1x32x44x44xf16>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:        [[SIN:%.+]] = VPU.NCE.ClusterTiling ([[IN]] as {{[^:]+}}: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[SIN]] as {{[^:]+}}: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x44x44xf16>
    // CHECK:        return [[OUT]] : tensor<1x32x44x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @SinSWWithSOK
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x1x44xf16>
func.func @SinSWWithSOK(%arg0: tensor<1x32x1x44xf16>) -> tensor<1x32x1x44xf16> {
    %0 = VPU.Sin(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x44xf16> -> tensor<1x32x1x44xf16>
    return %0 : tensor<1x32x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as {{[^:]+}}: tensor<1x32x1x44xf16>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:        [[SIN:%.+]] = VPU.NCE.ClusterTiling ([[IN]] as {{[^:]+}}: tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[SIN]] as {{[^:]+}}: tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x1x44xf16>
    // CHECK:        return [[OUT]] : tensor<1x32x1x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @SinSWWithClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x44xf16>
func.func @SinSWWithClustering(%arg0: tensor<1x1x1x44xf16>) -> tensor<1x1x1x44xf16> {
    %0 = VPU.Sin(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16>
    return %0 : tensor<1x1x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as {{[^:]+}}: tensor<1x1x1x44xf16>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[SIN:%.+]] = VPU.NCE.ClusterTiling ([[IN]] as {{[^:]+}}: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[SIN]] as {{[^:]+}}: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x1x44xf16>
    // CHECK:        return [[OUT]] : tensor<1x1x1x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @CosSWWithSOH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x44x44xf16>
func.func @CosSWWithSOH(%arg0: tensor<1x32x44x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Cos(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16>
    return %0 : tensor<1x32x44x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as {{[^:]+}}: tensor<1x32x44x44xf16>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:        [[COS:%.+]] = VPU.NCE.ClusterTiling ([[IN]] as {{[^:]+}}: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[COS]] as {{[^:]+}}: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x44x44xf16>
    // CHECK:        return [[OUT]] : tensor<1x32x44x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @CosSWWithSOK
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x1x44xf16>
func.func @CosSWWithSOK(%arg0: tensor<1x32x1x44xf16>) -> tensor<1x32x1x44xf16> {
    %0 = VPU.Cos(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x44xf16> -> tensor<1x32x1x44xf16>
    return %0 : tensor<1x32x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as {{[^:]+}}: tensor<1x32x1x44xf16>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:        [[COS:%.+]] = VPU.NCE.ClusterTiling ([[IN]] as {{[^:]+}}: tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[COS]] as {{[^:]+}}: tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x1x44xf16>
    // CHECK:        return [[OUT]] : tensor<1x32x1x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @CosSWWithClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x44xf16>
func.func @CosSWWithClustering(%arg0: tensor<1x1x1x44xf16>) -> tensor<1x1x1x44xf16> {
    %0 = VPU.Cos(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16>
    return %0 : tensor<1x1x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as {{[^:]+}}: tensor<1x1x1x44xf16>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[COS:%.+]] = VPU.NCE.ClusterTiling ([[IN]] as {{[^:]+}}: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[COS]] as {{[^:]+}}: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x1x44xf16>
    // CHECK:        return [[OUT]] : tensor<1x1x1x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @ExpSWWithSOH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x44x44xf16>
func.func @ExpSWWithSOH(%arg0: tensor<1x32x44x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Exp(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16>
    return %0 : tensor<1x32x44x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as {{[^:]+}}: tensor<1x32x44x44xf16>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:        [[EXP:%.+]] = VPU.NCE.ClusterTiling ([[IN]] as {{[^:]+}}: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[EXP]] as {{[^:]+}}: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x44x44xf16>
    // CHECK:        return [[OUT]] : tensor<1x32x44x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @ExpSWWithSOK
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x1x44xf16>
func.func @ExpSWWithSOK(%arg0: tensor<1x32x1x44xf16>) -> tensor<1x32x1x44xf16> {
    %0 = VPU.Exp(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x44xf16> -> tensor<1x32x1x44xf16>
    return %0 : tensor<1x32x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as {{[^:]+}}: tensor<1x32x1x44xf16>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:        [[EXP:%.+]] = VPU.NCE.ClusterTiling ([[IN]] as {{[^:]+}}: tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[EXP]] as {{[^:]+}}: tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x1x44xf16>
    // CHECK:        return [[OUT]] : tensor<1x32x1x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @ExpSWWithClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x44xf16>
func.func @ExpSWWithClustering(%arg0: tensor<1x1x1x44xf16>) -> tensor<1x1x1x44xf16> {
    %0 = VPU.Exp(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16>
    return %0 : tensor<1x1x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as {{[^:]+}}: tensor<1x1x1x44xf16>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[EXP:%.+]] = VPU.NCE.ClusterTiling ([[IN]] as {{[^:]+}}: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[EXP]] as {{[^:]+}}: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x1x44xf16>
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

    //CHECK:        [[INPUT:%.+]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG0:%.+]]: tensor<1x1x1x512xf16, {order = #NCHW}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_COPY:%.+]] = VPU.Copy([[ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x512xf16, {order = #NCHW}> -> tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[LOG_SOFTMAX:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG1:%.+]]: tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK:                            -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:          [[INNER_LOG_SOFTMAX:%.+]] = VPU.LogSoftmax([[ARG1]]) {axisInd = 3 : i64} : tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.NCE.ClusterTiling ([[LOG_SOFTMAX]] as [[ARG1:%.+]]: tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x1x512xf16, {order = #NCHW}> {
    //CHECK:          [[INNER_COPY:%.+]] = VPU.Copy([[ARG1]]) : tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x1x512xf16, {order = #NCHW}>

    //CHECK:        return [[OUTPUT]] : tensor<1x1x1x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @MishSWWithSOH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x44x44xf16>
func.func @MishSWWithSOH(%arg0: tensor<1x32x44x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Mish(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16>
    return %0 : tensor<1x32x44x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as {{[^:]+}}: tensor<1x32x44x44xf16>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:        [[MISH:%.+]] = VPU.NCE.ClusterTiling ([[IN]] as {{[^:]+}}: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[MISH]] as {{[^:]+}}: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x44x44xf16>
    // CHECK:        return [[OUT]] : tensor<1x32x44x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @MishSWWithSOK
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x1x44xf16>
func.func @MishSWWithSOK(%arg0: tensor<1x32x1x44xf16>) -> tensor<1x32x1x44xf16> {
    %0 = VPU.Mish(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x44xf16> -> tensor<1x32x1x44xf16>
    return %0 : tensor<1x32x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as {{[^:]+}}: tensor<1x32x1x44xf16>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:        [[MISH:%.+]] = VPU.NCE.ClusterTiling ([[IN]] as {{[^:]+}}: tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[MISH]] as {{[^:]+}}: tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x1x44xf16>
    // CHECK:        return [[OUT]] : tensor<1x32x1x44xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @MishSWWithClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x44xf16>
func.func @MishSWWithClustering(%arg0: tensor<1x1x1x44xf16>) -> tensor<1x1x1x44xf16> {
    %0 = VPU.Mish(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16>
    return %0 : tensor<1x1x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as {{[^:]+}}: tensor<1x1x1x44xf16>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[MISH:%.+]] = VPU.NCE.ClusterTiling ([[IN]] as {{[^:]+}}: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[MISH]] as {{[^:]+}}: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x1x44xf16>
    // CHECK:        return [[OUT]] : tensor<1x1x1x44xf16>
}
