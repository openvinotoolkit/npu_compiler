//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --correct-NCE-workloads %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  @ConvLargeSparseOutput
// CHECK-SAME:     ([[INPUT_DDR:%.+]]: tensor<1x64x40x40xf16, {order = #NHWC}>)
func.func @ConvLargeSparseOutput(%input_ddr: tensor<1x64x40x40xf16, {order = #NHWC}>) -> !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {order = #NHWC}>> {
    %cst_weights = const.Declare tensor<384x64x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<384x64x4x4xf16>, [#const.Reorder<#NHWC>]
    %cst_weights_table = const.Declare tensor<384x1x1x4xsi32, {order = #NHWC}> = dense<10> : tensor<384x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %input = VPU.Copy(%input_ddr) {out_mem_space = @CMX_NN} : tensor<1x64x40x40xf16, {order = #NHWC}> -> tensor<1x64x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %weights = VPU.Copy(%cst_weights) {out_mem_space = @CMX_NN} : tensor<384x64x4x4xf16, {order = #NHWC}> -> tensor<384x64x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %weights_table = VPU.Copy(%cst_weights_table) {out_mem_space = @CMX_NN} : tensor<384x1x1x4xsi32, {order = #NHWC}> -> tensor<384x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    %conv_out = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [384, 64, 4, 4],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {mem_space = @CMX_NN, order = #NHWC}>> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 384, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
            }

    %output = VPU.Copy(%conv_out) : !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {mem_space = @CMX_NN, order = #NHWC}>>
        ->  !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {order = #NHWC}>>

    return %output : !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {order = #NHWC}>>

    // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<384x64x4x4xf16, {order = #NHWC}>
    // CHECK-DAG:   [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<384x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[INPUT:%.+]] = VPU.Copy([[INPUT_DDR]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x64x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[WEIGHTS:%.+]] = VPU.Copy([[CST_WEIGHTS]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<384x64x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = VPU.Copy([[CST_WEIGHTS_TABLE]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<384x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    // Equal channels per clusters which are a power of two

    // CHECK:       [[CONV_OUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {mem_space = @CMX_NN, order = #NHWC}>> {
    // CHECK:               VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 256, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK:               VPU.DPU.Workload outOffsets [0, 256, 0, 0] outSizes [1, 128, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK:           }

    // CHECK:       [[OUTPUT:%.+]] = VPU.Copy([[CONV_OUT]])
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {order = #NHWC}>>

    // CHECK:       return [[OUTPUT]] : !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {order = #NHWC}>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Input_CMX = !VPU.DistributedTensor<
    1x3x224x224xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!Output_CMX = !VPU.DistributedTensor<
    1x3x224x224xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!Input_DDR = tensor<1x3x224x224xf16, {order = #NCHW}>
!InputStub_CMX = tensor<1x3x224x224xf16, {mem_space = @CMX_NN, order = #NCHW}>
!OutputStub_CMX = tensor<1x3x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @NCEPermuteClustered
func.func @NCEPermuteClustered(%arg0: !Input_DDR) -> !Output_CMX {
    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : !Input_DDR -> !Input_CMX
    %output = VPU.NCE.Permute(%0) {
            dstElemType = f16, dstOrder = #NHWC, expandedChannels = 3 : i64,
            minimumHardwareExecutionCost = 4294967195 : i64,
            ppe = #VPU.PPEStub<>
        } -> !Output_CMX {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 112, 224] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 112, 0] outSizes [1, 3, 112, 224] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
        }

    return %output : !Output_CMX

    // CHECK:       VPU.NCE.Permute
    // CHECK-SAME:      dstElemType = f16, dstOrder = #NHWC, expandedChannels = 3 : i64
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x3x224x224xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:          mode = "SEGMENTED",
    // CHECK-SAME:          num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:          num_clusters = 2 : i64}>

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [0, 0, 0, 0] outSizes [1, 3, 112, 224]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 0

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [0, 0, 0, 0] outSizes [1, 3, 112, 224]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 1
}
