//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --remove-output-sparse-to-avoid-suboptimal-dpu-workloads %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!ActConv0Distributed = !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments}>
!OutConv0Distributed = !VPU.SparseTensor<data=!VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>, sparsity_map=!VPU.DistributedTensor<1x128x28x28xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>>

!WeightsDistributed = !VPU.DistributedTensor<128x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments}>
!WtableDistribution = !VPU.DistributedTensor<128x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments}>

!SparseType = !VPU.SparseTensor<data=tensor<1x128x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x128x28x28xi1, {order = #NHWC}>>

!ActConv1Distributed = !VPU.SparseTensor<data=!VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments}>, sparsity_map=!VPU.DistributedTensor<1x128x28x28xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments}>>
!OutConv1Distributed = !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>

// CHECK-LABEL: @RemoveOutputSparseForConvSOK
func.func @RemoveOutputSparseForConvSOK(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>) -> tensor<1x128x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    %cst_0 = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    %cst_2 = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x128x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.UnrolledType(%arg0 : tensor<1x128x28x28xf16, {order = #NHWC}>) -> !ActConv0Distributed
    %1 = VPU.UnrolledType(%cst_0: tensor<128x128x1x1xf16, {order = #NHWC}>) -> !WeightsDistributed
    %2 = VPU.UnrolledType(%cst : tensor<128x1x1x4xsi32>) -> !WtableDistribution

    %3 = VPU.NCE.Convolution(%0, %1, %2) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [128, 128, 1, 1], strides = [1, 1]} : !ActConv0Distributed, !WeightsDistributed, !WtableDistribution -> !OutConv0Distributed

    %4 = VPU.UnrolledType(%3: !OutConv0Distributed) -> !SparseType

    %5 = VPU.UnrolledType(%4 : !SparseType) -> !ActConv1Distributed
    %6 = VPU.UnrolledType(%cst_2 : tensor<128x128x1x1xf16, {order = #NHWC}>) -> !WeightsDistributed
    %7 = VPU.UnrolledType(%cst_1 : tensor<128x1x1x4xsi32>) -> !WtableDistribution

    %8 = VPU.NCE.Convolution(%5, %6, %7) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [128, 128, 1, 1], strides = [1, 1]} : !ActConv1Distributed, !WeightsDistributed, !WtableDistribution -> !OutConv1Distributed

    %9 = VPU.UnrolledType(%8 : !OutConv1Distributed) -> tensor<1x128x28x28xf16, {order = #NHWC}>
    return %9 : tensor<1x128x28x28xf16, {order = #NHWC}>

    //CHECK:    [[WEIGHTSTABLE_0:%.+]] = const.Declare tensor<128x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_0:%.+]] = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTSTABLE_1:%.+]] = const.Declare tensor<128x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_1:%.+]] = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}>

    //CHECK:    [[UNROLLED_INPUT_0:%.+]] = VPU.UnrolledType([[ARG0:%.+]] : tensor<1x128x28x28xf16, {order = #NHWC}>) -> !VPU.DistributedTensor
    //CHECK:    [[UNROLLED_WEIGHTS_0:%.+]] = VPU.UnrolledType([[WEIGHTS_0]] : tensor<128x128x1x1xf16, {order = #NHWC}>) -> !VPU.DistributedTensor
    //CHECK:    [[UNROLLED_WEIGHTSTABLE_0:%.+]] = VPU.UnrolledType([[WEIGHTSTABLE_0]] : tensor<128x1x1x4xsi32>) -> !VPU.DistributedTensor<128x1x1x4xsi32

    //CHECK:        [[CONV0:%.+]] = VPU.NCE.Convolution([[UNROLLED_INPUT_0]], [[UNROLLED_WEIGHTS_0]], [[UNROLLED_WEIGHTSTABLE_0]])
    //CHECK-SAME:  ->  !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED"

    //CHECK:     [[UNROLLED_OUTPUT_0:%.+]] = VPU.UnrolledType([[CONV0]]
    //CHECK-SAME: -> tensor<1x128x28x28xf16, {order = #NHWC}>

    //CHECK:    [[UNROLLED_INPUT_1:%.+]] = VPU.UnrolledType([[UNROLLED_OUTPUT_0]] : tensor<1x128x28x28xf16, {order = #NHWC}>
    //CHECK:    [[UNROLLED_WEIGHTS_1:%.+]] = VPU.UnrolledType([[WEIGHTS_1]] : tensor<128x128x1x1xf16, {order = #NHWC}>) -> !VPU.DistributedTensor
    //CHECK:    [[UNROLLED_WEIGHTSTABLE_1:%.+]] = VPU.UnrolledType([[WEIGHTSTABLE_1]] : tensor<128x1x1x4xsi32>) -> !VPU.DistributedTensor<128x1x1x4xsi32

    //CHECK:    [[CONV1:%.+]] = VPU.NCE.Convolution([[UNROLLED_INPUT_1]], [[UNROLLED_WEIGHTS_1]], [[UNROLLED_WEIGHTSTABLE_1]])
    //CHECK:        VPU.UnrolledType([[CONV1]]
    //CHECK-SAME: -> tensor<1x128x28x28xf16, {order = #NHWC}>
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!ActConv0Distributed = !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments}>
!OutConv0Distributed = !VPU.SparseTensor<data=!VPU.DistributedTensor<1x256x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPU.DistributedTensor<1x256x28x28xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1]}>>

!WeightsConv0Distributed = !VPU.DistributedTensor<256x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments}>
!WtableConv0Distribution = !VPU.DistributedTensor<256x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments}>

!WeightsConv1Distributed = !VPU.DistributedTensor<128x256x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments}>
!WtableConv1Distribution = !VPU.DistributedTensor<128x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments}>

!SparseType = !VPU.SparseTensor<data=tensor<1x256x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x256x28x28xi1, {order = #NHWC}>>

!ActConv1Distributed = !VPU.SparseTensor<data=!VPU.DistributedTensor<1x256x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments}>, sparsity_map=!VPU.DistributedTensor<1x256x28x28xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments}>>
!OutConv1Distributed = !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>

// CHECK-LABEL: @DoNotRemoveOutputSparseForConvSOK
func.func @DoNotRemoveOutputSparseForConvSOK(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>) -> tensor<1x128x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    %cst_0 = const.Declare tensor<256x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    %cst_2 = const.Declare tensor<128x256x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x256x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.UnrolledType(%arg0 : tensor<1x128x28x28xf16, {order = #NHWC}>) -> !ActConv0Distributed
    %1 = VPU.UnrolledType(%cst_0: tensor<256x128x1x1xf16, {order = #NHWC}>) -> !WeightsConv0Distributed
    %2 = VPU.UnrolledType(%cst : tensor<256x1x1x4xsi32>) -> !WtableConv0Distribution

    %3 = VPU.NCE.Convolution(%0, %1, %2) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [256, 128, 1, 1], strides = [1, 1]} : !ActConv0Distributed, !WeightsConv0Distributed, !WtableConv0Distribution -> !OutConv0Distributed

    %4 = VPU.UnrolledType(%3: !OutConv0Distributed) -> !SparseType

    %5 = VPU.UnrolledType(%4 : !SparseType) -> !ActConv1Distributed
    %6 = VPU.UnrolledType(%cst_2 : tensor<128x256x1x1xf16, {order = #NHWC}>) -> !WeightsConv1Distributed
    %7 = VPU.UnrolledType(%cst_1 : tensor<128x1x1x4xsi32>) -> !WtableConv1Distribution

    %8 = VPU.NCE.Convolution(%5, %6, %7) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [128, 256, 1, 1], strides = [1, 1]} : !ActConv1Distributed, !WeightsConv1Distributed, !WtableConv1Distribution -> !OutConv1Distributed
    %9 = VPU.UnrolledType(%8 : !OutConv1Distributed) -> tensor<1x128x28x28xf16, {order = #NHWC}>
    return %9 : tensor<1x128x28x28xf16, {order = #NHWC}>

    //CHECK:    [[WEIGHTSTABLE_0:%.+]] = const.Declare tensor<256x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_0:%.+]] = const.Declare tensor<256x128x1x1xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTSTABLE_1:%.+]] = const.Declare tensor<128x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_1:%.+]] = const.Declare tensor<128x256x1x1xf16, {order = #NHWC}>

    //CHECK:    [[UNROLLED_INPUT_0:%.+]] = VPU.UnrolledType([[ARG0:%.+]] : tensor<1x128x28x28xf16, {order = #NHWC}>) -> !VPU.DistributedTensor
    //CHECK:    [[UNROLLED_WEIGHTS_0:%.+]] = VPU.UnrolledType([[WEIGHTS_0]] : tensor<256x128x1x1xf16, {order = #NHWC}>) -> !VPU.DistributedTensor
    //CHECK:    [[UNROLLED_WEIGHTSTABLE_0:%.+]] = VPU.UnrolledType([[WEIGHTSTABLE_0]] : tensor<256x1x1x4xsi32>) -> !VPU.DistributedTensor<256x1x1x4xsi32

    //CHECK:        [[CONV0:%.+]] = VPU.NCE.Convolution([[UNROLLED_INPUT_0]], [[UNROLLED_WEIGHTS_0]], [[UNROLLED_WEIGHTSTABLE_0]])
    //CHECK-SAME:   -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x256x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED"

    //CHECK:     [[UNROLLED_OUTPUT_0:%.+]] = VPU.UnrolledType([[CONV0]]
    //CHECK-SAME:   -> !VPU.SparseTensor<data=tensor<1x256x28x28xf16, {order = #NHWC}>

    //CHECK:    [[UNROLLED_INPUT_1:%.+]] = VPU.UnrolledType([[UNROLLED_OUTPUT_0]] : !VPU.SparseTensor<data=tensor<1x256x28x28xf16, {order = #NHWC}>
    //CHECK:    [[UNROLLED_WEIGHTS_1:%.+]] = VPU.UnrolledType([[WEIGHTS_1]] : tensor<128x256x1x1xf16, {order = #NHWC}>) -> !VPU.DistributedTensor
    //CHECK:    [[UNROLLED_WEIGHTSTABLE_1:%.+]] = VPU.UnrolledType([[WEIGHTSTABLE_1]] : tensor<128x1x1x4xsi32>) -> !VPU.DistributedTensor<128x1x1x4xsi32

    //CHECK:    [[CONV1:%.+]] = VPU.NCE.Convolution([[UNROLLED_INPUT_1]], [[UNROLLED_WEIGHTS_1]], [[UNROLLED_WEIGHTSTABLE_1]])
    //CHECK:        VPU.UnrolledType([[CONV1]]
    //CHECK-SAME:   -> tensor<1x128x28x28xf16, {order = #NHWC}>
}
