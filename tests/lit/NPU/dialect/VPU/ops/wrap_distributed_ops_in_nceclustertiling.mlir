//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --wrap-distributed-ops-in-nceclustertiling %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPU.DistributedTensor<
    1x64x28x28xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x80x28x28xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!WeightsDistributed = !VPU.DistributedTensor<
    80x64x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!WeightsTableDistributed = !VPU.DistributedTensor<
    80x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

// CHECK:        func.func @CheckConv([[INPUT:%.+]]: !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
// CHECK-SAME:    [[WEIGHTS:%.+]]: !VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
// CHECK-SAME:    [[WT:%.+]]: !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
// CHECK-SAME:     -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {

func.func @CheckConv(%input: !InputDistributed, %weights: !WeightsDistributed,
                     %wt: !WeightsTableDistributed) -> !OutputDistributed {

    %convOut = VPU.NCE.Convolution(%input, %weights, %wt) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                                                          ppe = #VPU.PPEStub<>,
                                                          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                                                          rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} -> !OutputDistributed
    return %convOut : !OutputDistributed
}
//CHECK:            [[NCE_CLUSTER_TILING:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[INNER_ARG0:%.+]]: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
//CHECK-SAME:                                   [[WEIGHTS]] as [[INNER_ARG1:%.+]]: tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
//CHECK-SAME:                                   [[WT]] as [[INNER_ARG2:%.+]]: tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
//CHECK-SAME:                                   -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
//CHECK:                 [[INNER_OP_OUT:%.+]] = VPU.NCE.Convolution([[INNER_ARG0]], [[INNER_ARG1]], [[INNER_ARG2]])
//CHECK-SAME:                            {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
//CHECK-SAME:                            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
//CHECK-SAME:                            ppe = #VPU.PPEStub<>,
//CHECK-SAME:                            rawFilterShape = [80, 64, 3, 3], strides = [1, 1]}
//CHECK-SAME:                            -> tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
//CHECK:                 VPU.Yield [[INNER_OP_OUT]]
//CHECK:            }
//CHECK:      return [[NCE_CLUSTER_TILING]] : !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPU.DistributedTensor<
    1x64x28x28xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x64x28x28xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

//CHECK:      func.func @CheckCopy([[INPUT:%.+]]: !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
//CHECK-SAME:   -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {

func.func @CheckCopy(%input: !InputDistributed) -> !OutputDistributed {

    %copyOut = VPU.Copy(%input) {out_mem_space = @CMX_NN} : !OutputDistributed -> !OutputDistributed
    return %copyOut : !OutputDistributed

}
//CHECK:    [[NCE_CLUSTER_TILING:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[INNER_ARG0:%.+]]: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>)
//CHECK-SAME:                         -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {

//CHECK:            [[INNER_OP_OUT:%.+]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
//CHECK-SAME:                            -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
//CHECK:            VPU.Yield [[INNER_OP_OUT]]
//CHECK:    }
//CHECK:    return [[NCE_CLUSTER_TILING]] : !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPU.DistributedTensor<
    1x32x14x14xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!WeightsDistributed = !VPU.DistributedTensor<
    32x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!WeightsTableDistributed = !VPU.DistributedTensor<
    32x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x32x14x14xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_clusters = 2 : i64,
    num_tiles = [1, 1, 2, 1],
    alignment = [1, 16, 1, 1]
}>

// CHECK:       func.func @CheckDepthConv([[INPUT:%.+]]: !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
// CHECK-SAME:                                                                  {mode = "DUPLICATED", num_clusters = 2 : i64}>,
// CHECK-SAME:                                  [[WEIGHTS:%.+]]: !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
// CHECK-SAME:                                  [[WT:%.+]]: !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)

func.func @CheckDepthConv(%input: !InputDistributed, %weights: !WeightsDistributed,
                          %wt: !WeightsTableDistributed) -> !OutputDistributed {

    %depthConvOut= VPU.NCE.DepthConvolution(%input, %weights, %wt) { pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                                                                ppe = #VPU.PPEStub<>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> !OutputDistributed
    return %depthConvOut : !OutputDistributed
}

//CHECK:        [[NCE_CLUSTER_TILING:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[INNER_ARG0:%.+]]: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>,
//CHECK-SAME:                           [[WEIGHTS]] as [[INNER_ARG1:%.+]]: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
//CHECK-SAME:                           [[WT]] as [[INNER_ARG2:%.+]]: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
//CHECK-SAME:                            -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
//CHECK:                [[INNER_OP_OUT:%.+]] = VPU.NCE.DepthConvolution([[INNER_ARG0]], [[INNER_ARG1]], [[INNER_ARG2]]) {
//CHECK-SAME:                             pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
//CHECK-SAME:                             ppe = #VPU.PPEStub<>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]}
//CHECK-SAME:                             -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
//CHECK:                VPU.Yield [[INNER_OP_OUT]]
//CHECK:        }
//CHECK:    return [[NCE_CLUSTER_TILING]] : !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPU.DistributedTensor<
    1x4x224x224xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_clusters = 2 : i64,
    num_tiles = [1, 1, 2, 1],
    kernel = [7, 7],
    pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
    strides = [2, 2]
}>

!WeightsDistributed = !VPU.DistributedTensor<
    64x1x1x160xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!WeightsTableDistributed = !VPU.DistributedTensor<
    64x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x64x112x112xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_clusters = 2 : i64,
    num_tiles = [1, 1, 2, 1]
}>

// CHECK:       func.func @CheckCompressConv([[INPUT:%.+]]: !VPU.DistributedTensor<1x4x224x224xf16, #NHWC, @CMX_NN,
// CHECK-SAME:                                              {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7],
// CHECK-SAME:                                              pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
// CHECK-SAME:                                              strides = [2, 2], num_clusters = 2 : i64}>,
// CHECK-SAME:                                  [[WEIGHTS:%.+]]: !VPU.DistributedTensor<64x1x1x160xf16, #NHWC, @CMX_NN,
// CHECK-SAME:                                              {mode = "DUPLICATED", num_clusters = 2 : i64}>,
// CHECK-SAME:                                  [[WT:%.+]]: !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN,
// CHECK-SAME:                                              {mode = "DUPLICATED", num_clusters = 2 : i64}>)

func.func @CheckCompressConv(%input: !InputDistributed, %weights: !WeightsDistributed,
                            %wt: !WeightsTableDistributed) -> !OutputDistributed {

    %compressConvOut= VPU.NCE.CompressConvolution(%input, %weights, %wt)  {cm_sp_pattern = 15 : i64,
                                                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
                                                pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
                                                ppe = #VPU.PPEStub<>,
                                                rawFilterShape = [64, 4, 7, 7], strides = [2, 2]} -> !OutputDistributed
    return %compressConvOut : !OutputDistributed
}

//CHECK:       [[NCE_CLUSTER_TILING:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[INNER_ARG0:%.+]]: tensor<1x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>,
//CHECK-SAME:                              [[WEIGHTS]] as [[INNER_ARG1:%.+]]: tensor<64x1x1x160xf16, {mem_space = @CMX_NN, order = #NHWC}>,
//CHECK-SAME:                              [[WT]] as [[INNER_ARG2:%.+]]: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
//CHECK-SAME:                               -> !VPU.DistributedTensor<1x64x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
//CHECK:            [[INNER_OP_OUT:%.+]] = VPU.NCE.CompressConvolution([[INNER_ARG0]], [[INNER_ARG1]], [[INNER_ARG2]]) {cm_sp_pattern = 15 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
//CHECK-SAME:                   pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
//CHECK-SAME:                   ppe = #VPU.PPEStub<>,
//CHECK-SAME:                   rawFilterShape = [64, 4, 7, 7], strides = [2, 2]}
//CHECK-SAME:                   -> tensor<1x64x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
//CHECK:            VPU.Yield [[INNER_OP_OUT]]
//CHECK:        }
//CHECK:    return [[NCE_CLUSTER_TILING]] : !VPU.DistributedTensor<1x64x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPU.DistributedTensor<
    1x320x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x320x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

// CHECK:       func.func @CheckAvgPool([[INPUT:%.+]]: !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN,
//CHECK-SAME:                           {mode = "DUPLICATED", num_clusters = 2 : i64}>)

func.func @CheckAvgPool(%input: !InputDistributed) -> !OutputDistributed {

    %avgPoolOut= VPU.NCE.AveragePool(%input)  {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                                        ppe = #VPU.PPEStub<>,
                                        strides = [1, 1]} -> !OutputDistributed
    return %avgPoolOut : !OutputDistributed
}

//CHECK:        [[NCE_CLUSTER_TILING:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[INNER_ARG0:%.+]]: tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
//CHECK-SAME:                               -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
//CHECK:                 [[INNER_OP_OUT:%.+]] = VPU.NCE.AveragePool([[INNER_ARG0]])
//CHECK-SAME:                   {kernel_size = [1, 1],
//CHECK-SAME:                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]}
//CHECK-SAME:                    -> tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
//CHECK:                VPU.Yield [[INNER_OP_OUT]]
//CHECK:        }
//CHECK:    return [[NCE_CLUSTER_TILING]] : !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPU.DistributedTensor<
    1x32x112x112xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x32x112x112xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK:       func.func @CheckMaxPool([[INPUT:%.+]]: !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
//CHECK-SAME:                           {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)

func.func @CheckMaxPool(%input: !InputDistributed) -> !OutputDistributed {

    %maxPoolOut= VPU.NCE.MaxPool(%input) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                                            ppe = #VPU.PPEStub<>, strides = [1, 1]} -> !OutputDistributed
    return %maxPoolOut : !OutputDistributed
}

//CHECK:        [[NCE_CLUSTER_TILING:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[INNER_ARG0:%.+]]: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>)
//CHECK-SAME:                               -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
//CHECK:             [[INNER_OP_OUT:%.+]] = VPU.NCE.MaxPool([[INNER_ARG0]]) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]}
//CHECK-SAME:                             -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
//CHECK:             VPU.Yield [[INNER_OP_OUT]]
//CHECK:        }
//CHECK:    return [[NCE_CLUSTER_TILING]] : !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed0 = !VPU.DistributedTensor<
    1x32x112x112xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!InputDistributed1 = !VPU.DistributedTensor<
    1x32x112x112xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x32x112x112xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK:       func.func @CheckEltwise([[INPUT0:%.+]]: !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
//CHECK-SAME:               [[INPUT1:%.+]]: !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)

func.func @CheckEltwise(%input0: !InputDistributed0, %input1: !InputDistributed1) -> !OutputDistributed {

    %eltwiseOut= VPU.NCE.Eltwise(%input0, %input1) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> !OutputDistributed
    return %eltwiseOut : !OutputDistributed
}

//CHECK:        [[NCE_CLUSTER_TILING:%.+]] = VPU.NCE.ClusterTiling ([[INPUT0]] as [[INNER_ARG0:%.+]]: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>,
//CHECK-SAME:                               [[INPUT1]] as [[INNER_ARG1:%.+]]: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>)
//CHECK-SAME:                                -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
//CHECK:             [[INNER_OP_OUT:%.+]] = VPU.NCE.Eltwise([[INNER_ARG0]], [[INNER_ARG1]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
//CHECK-SAME:                       -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
//CHECK:             VPU.Yield [[INNER_OP_OUT]]
//CHECK:        }
//CHECK:    return [[NCE_CLUSTER_TILING]] : !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

!InputDistributed = !VPU.DistributedTensor<
    1x3x256x224xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x4x256x224x!qElemType, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK:       func.func @CheckPermute([[INPUT:%.+]]: !VPU.DistributedTensor<1x3x256x224xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)

func.func @CheckPermute(%input: !InputDistributed) -> !OutputDistributed {

    %permuteOut= VPU.NCE.Permute(%input) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        ppe = #VPU.PPEStub<>
    } -> !OutputDistributed
    return %permuteOut : !OutputDistributed
}

//CHECK:        [[NCE_CLUSTER_TILING:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[INNER_ARG0:%.+]]: tensor<1x3x256x224xf16, {mem_space = @CMX_NN, order = #NCHW}>)
//CHECK-SAME:                               -> !VPU.DistributedTensor<1x4x256x224x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
//CHECK:             [[INNER_OP_OUT:%.+]] = VPU.NCE.Permute([[INNER_ARG0]]) {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64,
//CHECK-SAME:                                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
//CHECK-SAME:                                    ppe = #VPU.PPEStub<>}
//CHECK-SAME:                                    -> tensor<1x4x256x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>
//CHECK:             VPU.Yield [[INNER_OP_OUT]]
//CHECK:        }
//CHECK:    return [[NCE_CLUSTER_TILING]] : !VPU.DistributedTensor<1x4x256x224x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPU.DistributedTensor<
    1x320x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x320x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

// CHECK:       func.func @CheckSigmoid([[INPUT:%.+]]: !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN,
//CHECK-SAME:                                               {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)

func.func @CheckSigmoid(%input: !InputDistributed) -> !OutputDistributed {

    %sigmoidOut = VPU.Sigmoid(%input) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
                        : !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> -> !OutputDistributed
    return %sigmoidOut : !OutputDistributed
}

//CHECK:        [[NCE_CLUSTER_TILING:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[INNER_ARG0:%.+]]: tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
//CHECK-SAME:                               -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
//CHECK:            [[INNER_OP_OUT:%.+]] = VPU.Sigmoid([[INNER_ARG0]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
//CHECK-SAME:                               : tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
//CHECK:            VPU.Yield [[INNER_OP_OUT]]
//CHECK:    }
//CHECK:    return [[NCE_CLUSTER_TILING]] : !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!WeightsType = tensor<
    128x256x3x3xf16, {
    mem_space = @CMX_NN,
    order = #NHWC
}>

!WeightsTableType = tensor<
    128x1x1x4xsi32, {
    mem_space = @CMX_NN,
    order = #NCHW
}>

!SparseInput = !VPU.SparseTensor<
    data=tensor<1x256x28x28xf16, {
    mem_space = @CMX_NN,
    order = #NHWC}>,
    sparsity_map=tensor<1x256x28x28xi1, {
    mem_space = @CMX_NN,
    order = #NHWC}>
>

!SparseConvOutputDist = !VPU.SparseTensor<
    data=!VPU.DistributedTensor<1x128x14x14xf16,
     #NHWC,
     @CMX_NN, {
        mode = "DUPLICATED|SEGMENTED",
        num_tiles = [1, 2, 1, 1],
        num_clusters = 2 : i64,
        alignment = [1, 16, 1, 1]}>,
    sparsity_map=!VPU.DistributedTensor<1x128x14x14xi1,
     #NHWC,
     @CMX_NN, {
        mode = "DUPLICATED|SEGMENTED",
        num_tiles = [1, 2, 1, 1],
        num_clusters = 2 : i64,
        alignment = [1, 16, 1, 1]}>
>



// CHECK:      func.func @SparseConvolution([[INPUT:%.+]]: !VPU.SparseTensor<data=tensor<1x256x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x256x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>,
// CHECK-SAME:                         [[WEIGHTS_TABLE:%.+]]: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
// CHECK-SAME:                         [[WEIGHTS:%.+]]: tensor<128x256x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>)
// CHECK-SAME:         -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPU.DistributedTensor<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>> {
func.func @SparseConvolution(%input: !SparseInput,
                        %weightsTable: !WeightsTableType,
                        %weights: !WeightsType)
        -> !SparseConvOutputDist {

    %sparseConv = VPU.NCE.Convolution(%input, %weights, %weightsTable)
                    {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
                    ppe = #VPU.PPEStub<>,
                    rawFilterShape = [128, 256, 3, 3], strides = [2, 2]}
                        -> !SparseConvOutputDist


    return %sparseConv : !SparseConvOutputDist

}

//CHECK:         [[NCE_CLUSTER_TILING:%.+]] = VPU.NCE.ClusterTiling
//CHECK-SAME:                ([[INPUT]] as [[INNER_ARG0:%.+]]: !VPU.SparseTensor<data=tensor<1x256x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x256x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>,
//CHECK-SAME:                [[WEIGHTS]] as [[INNER_ARG1:%.+]]: tensor<128x256x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
//CHECK-SAME:                [[WEIGHTS_TABLE]] as [[INNER_ARG2:%.+]]: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
//CHECK-SAME:                 -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPU.DistributedTensor<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>> {

//CHECK:               [[INNER_OP_OUT:%.+]] = VPU.NCE.Convolution([[INNER_ARG0]], [[INNER_ARG1]], [[INNER_ARG2]]) {
//CHECK-SAME:                    pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
//CHECK-SAME:                    ppe = #VPU.PPEStub<>,
//CHECK-SAME:                    rawFilterShape = [128, 256, 3, 3], strides = [2, 2]}
//CHECK-SAME:                     -> !VPU.SparseTensor<data=tensor<1x128x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x128x14x14xi1, {mem_space = @CMX_NN, order = #NHWC}>>
//CHECK:      VPU.Yield [[INNER_OP_OUT]]
//CHECK:    return [[NCE_CLUSTER_TILING]] : !VPU.SparseTensor<data=!VPU.DistributedTensor<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPU.DistributedTensor<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
