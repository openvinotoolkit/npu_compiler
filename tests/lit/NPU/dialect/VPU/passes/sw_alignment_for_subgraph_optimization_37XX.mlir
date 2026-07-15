//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --make-ops-with-distributed-tensor --make-distributed-copies %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MVNInputAlignForSegmentedInput
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @MVNInputAlignForSegmentedInput(%input: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x64x64xf16, {order = #NWHC}> {
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
             strides = [1, 1]}
                : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x64x64xf16, {order = #NHWC}>
    %permute_cast = VPU.PermuteCast(%conv) {
            dst_order = #NWHC,
            mem_perm = #NCHW}
            : tensor<1x32x64x64xf16, {order = #NHWC}>
                -> tensor<1x32x64x64xf16, {order = #NWHC}>
    %mvn = VPU.MVN(%permute_cast) {
            across_channels = false, eps = 1.0E-4 : f64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true
            } : tensor<1x32x64x64xf16, {order = #NWHC}>
                -> tensor<1x32x64x64xf16, {order = #NWHC}>

    return %mvn : tensor<1x32x64x64xf16, {order = #NWHC}>

    //CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[INPUT_COPY:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[WEIGHTS_COPY:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x32x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    //CHECK:        [[WT_COPY:%.+]] = VPU.Copy([[WEIGHTS_TABLE]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    //CHECK:        [[CLUSTERED_CONV:%.+]] = VPU.NCE.Convolution([[INPUT_COPY]], [[WEIGHTS_COPY]], [[WT_COPY]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[CONV_OUT_COPY:%.+]] = VPU.Copy([[CLUSTERED_CONV]]
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NHWC}>

    //CHECK:        [[PERMUTE:%.+]] = VPU.PermuteCast([[CONV_OUT_COPY]]) {dst_order = #NWHC, mem_perm = #NCHW} : tensor<1x32x64x64xf16, {order = #NHWC}> -> tensor<1x32x64x64xf16, {order = #NWHC}>

    //CHECK:        [[CLUSTERED_COPY_TO_MVN:%.+]] = VPU.Copy([[PERMUTE]] as {{%[^:]+}}: tensor<1x32x64x64xf16, {order = #NWHC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[CLUSTERED_MVN:%.+]] = VPU.MVN([[CLUSTERED_COPY_TO_MVN]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[MVN_OUT_COPY:%.+]] =  VPU.Copy([[CLUSTERED_MVN]]
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NWHC}>

    //CHECK:        return [[MVN_OUT_COPY]] : tensor<1x32x64x64xf16, {order = #NWHC}>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SOKMVNInputAlignForSegmentedInput
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @SOKMVNInputAlignForSegmentedInput(%input: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x64x64xf16, {order = #NWHC}> {
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
             strides = [1, 1]}
                : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x64x64xf16, {order = #NHWC}>
    %permute_cast = VPU.PermuteCast(%conv) {
            dst_order = #NWHC,
            mem_perm = #NCHW}
            : tensor<1x32x64x64xf16, {order = #NHWC}>
                -> tensor<1x32x64x64xf16, {order = #NWHC}>
    %mvn = VPU.MVN(%permute_cast) {
            across_channels = false, eps = 1.0E-4 : f64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true
            } : tensor<1x32x64x64xf16, {order = #NWHC}>
                -> tensor<1x32x64x64xf16, {order = #NWHC}>

    return %mvn : tensor<1x32x64x64xf16, {order = #NWHC}>

    //CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[INPUT_COPY:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[WEIGHTS_COPY:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x32x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    //CHECK:        [[WT_COPY:%.+]] = VPU.Copy([[WEIGHTS_TABLE]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    //CHECK:        [[CLUSTERED_CONV:%.+]] = VPU.NCE.Convolution([[INPUT_COPY]], [[WEIGHTS_COPY]], [[WT_COPY]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[CONV_OUT_COPY:%.+]] = VPU.Copy([[CLUSTERED_CONV]]
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NHWC}>

    //CHECK:        [[PERMUTE:%.+]] = VPU.PermuteCast([[CONV_OUT_COPY]]) {dst_order = #NWHC, mem_perm = #NCHW} : tensor<1x32x64x64xf16, {order = #NHWC}> -> tensor<1x32x64x64xf16, {order = #NWHC}>

    //CHECK:        [[CLUSTERED_COPY_TO_MVN:%.+]] = VPU.Copy([[PERMUTE]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NWHC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[CLUSTERED_MVN:%.+]] = VPU.MVN([[CLUSTERED_COPY_TO_MVN]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NWHC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[MVN_OUT_COPY:%.+]] =  VPU.Copy([[CLUSTERED_MVN]]
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NWHC}>

    //CHECK:        return [[MVN_OUT_COPY]] : tensor<1x32x64x64xf16, {order = #NWHC}>
}


// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @TanhInputAlignForSegmentedInput
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @TanhInputAlignForSegmentedInput(%input: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
             strides = [1, 1]}
                : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x64x64xf16, {order = #NHWC}>
    %tanh = VPU.Tanh(%conv) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x32x64x64xf16, {order = #NHWC}> -> tensor<1x32x64x64xf16, {order = #NHWC}>

    return %tanh : tensor<1x32x64x64xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[INPUT_COPY:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[WEIGHTS_COPY:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x32x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    //CHECK:        [[WT_COPY:%.+]] = VPU.Copy([[WEIGHTS_TABLE]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    //CHECK:        [[CLUSTERED_CONV:%.+]] = VPU.NCE.Convolution([[INPUT_COPY]], [[WEIGHTS_COPY]], [[WT_COPY]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[CONV_OUT_COPY:%.+]] = VPU.Copy([[CLUSTERED_CONV]]
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NHWC}>

    //CHECK:        [[CLUSTERED_COPY_TO_TANH:%.+]] = VPU.Copy([[CONV_OUT_COPY]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[CLUSTERED_TANH:%.+]] = VPU.Tanh([[CLUSTERED_COPY_TO_TANH]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[TANH_OUT_COPY:%.+]] = VPU.Copy([[CLUSTERED_TANH]]
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NHWC}>

    //CHECK:        return [[TANH_OUT_COPY]] : tensor<1x32x64x64xf16, {order = #NHWC}>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SOKTanhInputAlignForSegmentedInput
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @SOKTanhInputAlignForSegmentedInput(%input: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
             strides = [1, 1]}
                : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x64x64xf16, {order = #NHWC}>
    %tanh = VPU.Tanh(%conv) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x64x64xf16, {order = #NHWC}> -> tensor<1x32x64x64xf16, {order = #NHWC}>

    return %tanh : tensor<1x32x64x64xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[INPUT_COPY:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[WEIGHTS_COPY:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x32x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    //CHECK:        [[WT_COPY:%.+]] = VPU.Copy([[WEIGHTS_TABLE]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    //CHECK:        [[CLUSTERED_CONV:%.+]] = VPU.NCE.Convolution([[INPUT_COPY]], [[WEIGHTS_COPY]], [[WT_COPY]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[CONV_OUT_COPY:%.+]] = VPU.Copy([[CLUSTERED_CONV]]
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NHWC}>

    //CHECK:        [[CLUSTERED_COPY_TO_TANH:%.+]] = VPU.Copy([[CONV_OUT_COPY]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[CLUSTERED_TANH:%.+]] = VPU.Tanh([[CLUSTERED_COPY_TO_TANH]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[TANH_OUT_COPY:%.+]] = VPU.Copy([[CLUSTERED_TANH]]
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NHWC}>

    //CHECK:        return [[TANH_OUT_COPY]] : tensor<1x32x64x64xf16, {order = #NHWC}>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MVNOutputAlignForSegmentedOutput
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NWHC}>
func.func @MVNOutputAlignForSegmentedOutput(%input: tensor<1x32x64x64xf16, {order = #NWHC}>) -> tensor<1x32x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    %mvn = VPU.MVN(%input) {across_channels = false, eps = 1.0E-4 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true}
            : tensor<1x32x64x64xf16, {order = #NWHC}> -> tensor<1x32x64x64xf16, {order = #NWHC}>
    %permute_cast = VPU.PermuteCast(%mvn) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x32x64x64xf16, {order = #NWHC}> -> tensor<1x32x64x64xf16, {order = #NHWC}>
    %dwconv = VPU.NCE.DepthConvolution(%permute_cast, %weights, %weights_table) rawFilterShape [32, 1, 1, 1] {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
             strides = [1, 1]}
                -> tensor<1x32x64x64xf16, {order = #NHWC}>

    return %dwconv : tensor<1x32x64x64xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[INPUT_COPY:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NWHC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[MVN:%.+]] = VPU.MVN([[INPUT_COPY]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NWHC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[MVN_OUT_COPY:%.+]] = VPU.Copy([[MVN]]
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NWHC}>

    //CHECK:        [[PERMUTE:%.+]] = VPU.PermuteCast([[MVN_OUT_COPY]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x32x64x64xf16, {order = #NWHC}> -> tensor<1x32x64x64xf16, {order = #NHWC}>

    //CHECK:        [[CONV_IN_COPY:%.+]] = VPU.Copy([[PERMUTE]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[WEIGHTS_COPY:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    //CHECK:        [[WT_COPY:%.+]] = VPU.Copy([[WEIGHTS_TABLE]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    //CHECK:        [[DWCONV:%.+]] = VPU.NCE.DepthConvolution([[CONV_IN_COPY]], [[WEIGHTS_COPY]], [[WT_COPY]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[DWCONV_OUT_COPY:%.+]] = VPU.Copy([[DWCONV]]
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NHWC}>

    //CHECK:        return [[DWCONV_OUT_COPY]] : tensor<1x32x64x64xf16, {order = #NHWC}>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DoNotAlignForSOHInput
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @DoNotAlignForSOHInput(%input: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x64x64xf16, {order = #NWHC}> {
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
             strides = [1, 1]}
                : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x64x64xf16, {order = #NHWC}>
    %permute_cast = VPU.PermuteCast(%conv) {
            dst_order = #NWHC,
            mem_perm = #NCHW}
            : tensor<1x32x64x64xf16, {order = #NHWC}>
                -> tensor<1x32x64x64xf16, {order = #NWHC}>
    %mvn = VPU.MVN(%permute_cast) {
            across_channels = false, eps = 1.0E-4 : f64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true
            } : tensor<1x32x64x64xf16, {order = #NWHC}>
                -> tensor<1x32x64x64xf16, {order = #NWHC}>

    return %mvn : tensor<1x32x64x64xf16, {order = #NWHC}>

    // Don't add alignment when the input is SOH
    // since SOH and channel split cannot be compatible
    //CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[INPUT_COPY:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[WEIGHTS_COPY:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[WT_COPY:%.+]] = VPU.Copy([[WEIGHTS_TABLE]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[CLUSTERED_CONV:%.+]] = VPU.NCE.Convolution([[INPUT_COPY]], [[WEIGHTS_COPY]], [[WT_COPY]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[CONV_OUT_COPY:%.+]] = VPU.Copy([[CLUSTERED_CONV]]
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NHWC}>

    //CHECK:        [[PERMUTE:%.+]] = VPU.PermuteCast([[CONV_OUT_COPY]]) {dst_order = #NWHC, mem_perm = #NCHW} : tensor<1x32x64x64xf16, {order = #NHWC}> -> tensor<1x32x64x64xf16, {order = #NWHC}>

    //CHECK:        [[CLUSTERED_COPY_TO_MVN:%.+]] = VPU.Copy([[PERMUTE]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NWHC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[CLUSTERED_MVN:%.+]] = VPU.MVN([[CLUSTERED_COPY_TO_MVN]])
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NWHC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[MVN_OUT_COPY:%.+]] =  VPU.Copy([[CLUSTERED_MVN]]
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NWHC}>

    //CHECK:        return [[MVN_OUT_COPY]] : tensor<1x32x64x64xf16, {order = #NWHC}>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DoNotAlignForUnalignedChannel
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @DoNotAlignForUnalignedChannel(%input: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x3x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
             strides = [1, 1]}
                : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x64x64xf16, {order = #NHWC}>
    %slice = VPU.Slice %conv [0, 0, 0, 0] [1, 3, 64, 64] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x3x64x64xf16, {order = #NHWC}>
    %tanh = VPU.Tanh(%slice) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x3x64x64xf16, {order = #NHWC}> -> tensor<1x3x64x64xf16, {order = #NHWC}>

    return %tanh : tensor<1x3x64x64xf16, {order = #NHWC}>

    // Don't add alignment when the channel needs extra expansion to align
    // to avoid regression caused by redundant alignment
    //CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[INPUT_COPY:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[WEIGHTS_COPY:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x32x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    //CHECK:        [[WT_COPY:%.+]] = VPU.Copy([[WEIGHTS_TABLE]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    //CHECK:        [[CLUSTERED_CONV:%.+]] = VPU.NCE.Convolution([[INPUT_COPY]], [[WEIGHTS_COPY]], [[WT_COPY]])
    //CHECK-SAME:   ->  !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[CONV_OUT_COPY:%.+]] = VPU.Copy([[CLUSTERED_CONV]]
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NHWC}>

    //CHECK:        [[SLICE:%.+]] = VPU.Slice [[CONV_OUT_COPY]] [0, 0, 0, 0] [1, 3, 64, 64] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x3x64x64xf16, {order = #NHWC}>

    //CHECK:        [[TANH_IN_COPY:%.+]] = VPU.Copy([[SLICE]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x3x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[TANH:%.+]] = VPU.Tanh([[TANH_IN_COPY]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x3x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[TANH_OUT_COPY:%.+]] = VPU.Copy([[TANH]]
    //CHECK-SAME:   -> tensor<1x3x64x64xf16, {order = #NHWC}>

    //CHECK:        return [[TANH_OUT_COPY]] : tensor<1x3x64x64xf16, {order = #NHWC}>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SWOpHasAlignedInputChannelReq
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x384x1x1xf16, {order = #NHWC}>
func.func @SWOpHasAlignedInputChannelReq(%input: tensor<1x384x1x1xf16, {order = #NHWC}>) -> tensor<1x32x1x1xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x384x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x384x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) rawFilterShape [32, 384, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
             strides = [1, 1]}
                : tensor<1x384x1x1xf16, {order = #NHWC}>, tensor<32x384x1x1xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x1x1xf16, {order = #NHWC}>
    %swish = VPU.Swish(%conv) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x1x1xf16, {order = #NHWC}>

    return %swish : tensor<1x32x1x1xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<32x384x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x384x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[INPUT_COPY:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x384x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[WEIGHTS_COPY:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:   !VPU.DistributedTensor<32x384x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    //CHECK:        [[WT_COPY:%.+]] = VPU.Copy([[WEIGHTS_TABLE]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    //CHECK:        [[CLUSTERED_CONV:%.+]] = VPU.NCE.Convolution([[INPUT_COPY]], [[WEIGHTS_COPY]], [[WT_COPY]])
    //CHECK-SAME:   ->  !VPU.DistributedTensor<1x32x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[CONV_OUT_COPY:%.+]] = VPU.Copy([[CLUSTERED_CONV]]
    //CHECK-SAME:   -> tensor<1x32x1x1xf16, {order = #NHWC}>

    //CHECK:        [[SWISH_IN_COPY:%.+]] = VPU.Copy([[CONV_OUT_COPY]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[SWISH:%.+]] = VPU.Swish([[SWISH_IN_COPY]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[SWISH_OUT_COPY:%.+]] = VPU.Copy([[SWISH]]
    //CHECK-SAME:   -> tensor<1x32x1x1xf16, {order = #NHWC}>

    //CHECK:        return [[SWISH_OUT_COPY]] : tensor<1x32x1x1xf16, {order = #NHWC}>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SWOpHasAlignedOutputChannelReq
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x384x1x1xf16, {order = #NHWC}>
func.func @SWOpHasAlignedOutputChannelReq(%input: tensor<1x384x1x1xf16, {order = #NHWC}>) -> tensor<1x32x1x1xf16, {order = #NHWC}> {
    %swish = VPU.Swish(%input) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x384x1x1xf16, {order = #NHWC}> -> tensor<1x384x1x1xf16, {order = #NHWC}>
    %weights = const.Declare tensor<32x384x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x384x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%swish, %weights, %weights_table) rawFilterShape [32, 384, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
             strides = [1, 1]}
                -> tensor<1x32x1x1xf16, {order = #NHWC}>

    return %conv : tensor<1x32x1x1xf16, {order = #NHWC}>

    //CHECK:        [[SWISH_IN_COPY:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x384x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[SWISH:%.+]] = VPU.Swish([[SWISH_IN_COPY]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x384x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[SWISH_OUT_COPY:%.+]] = VPU.Copy([[SWISH]]
    //CHECK-SAME:   -> tensor<1x384x1x1xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<32x384x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x384x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[INPUT_COPY:%.+]] = VPU.Copy([[SWISH_OUT_COPY]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x384x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[WEIGHTS_COPY:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:   !VPU.DistributedTensor<32x384x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    //CHECK:        [[WT_COPY:%.+]] = VPU.Copy([[WEIGHTS_TABLE]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    //CHECK:        [[CLUSTERED_CONV:%.+]] = VPU.NCE.Convolution([[INPUT_COPY]], [[WEIGHTS_COPY]], [[WT_COPY]])
    //CHECK-SAME:   ->  !VPU.DistributedTensor<1x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[CONV_OUT_COPY:%.+]] = VPU.Copy([[CLUSTERED_CONV]]
    //CHECK-SAME:   -> tensor<1x32x1x1xf16, {order = #NHWC}>

    //CHECK:        return [[CONV_OUT_COPY]] : tensor<1x32x1x1xf16, {order = #NHWC}>
}
