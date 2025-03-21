//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --correct-NCE-workloads %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @ConvLargeSparseOutput
// CHECK-SAME:    ([[INPUT_DDR:%.+]]: tensor<1x64x40x40xf16, {order = #NHWC}>)
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

    // CHECK:       [[CONV_OUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {mem_space = @CMX_NN, order = #NHWC}>> {
    // CHECK:               VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 384, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK:           }

    // CHECK:       [[OUTPUT:%.+]] = VPU.Copy([[CONV_OUT]])
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {order = #NHWC}>>

    // CHECK:       return [[OUTPUT]] : !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {order = #NHWC}>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @OptimizeWorkloadForDepthwiseConv(%arg0: tensor<1x128x56x56xf16, {order = #NHWC}>) -> tensor<1x128x54x54xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<128x1x1x4xsi32, {order = #NHWC}> = dense<10> : tensor<128x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = [@CMX_NN, 0]} : tensor<1x128x56x56xf16, {order = #NHWC}> -> tensor<1x128x56x56xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = [@CMX_NN, 0]} : tensor<128x16x1x1xf16, {order = #NHWC}> -> tensor<128x16x1x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    %2 = VPU.Copy(%wt) {out_mem_space = [@CMX_NN, 0]} : tensor<128x1x1x4xsi32, {order = #NHWC}> -> tensor<128x1x1x4xsi32, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    %4 = VPU.NCE.DepthConvolution(%0, %1, %2) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 1, 3, 3], strides = [1, 1]} -> tensor<1x128x54x54xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> {
      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 128, 28, 28] <left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64> <CUBOID_16x16>
    }

    %5 = VPU.Copy(%4) : tensor<1x128x54x54xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        -> tensor<1x128x54x54xf16, {order = #NHWC}>

    return %5 : tensor<1x128x54x54xf16, {order = #NHWC}>

    // CHECK:       VPU.NCE.DepthConvolution
    // CHECK:         VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 28, 28] <left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64> <CUBOID_16x16>
    // CHECK:         VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 32, 28, 28] <left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64> <CUBOID_16x16>
    // CHECK:         VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 32, 28, 28] <left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64> <CUBOID_16x16>
    // CHECK:         VPU.DPU.Workload outOffsets [0, 96, 0, 0] outSizes [1, 32, 28, 28] <left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64> <CUBOID_16x16>
}


// -----

!qElemType = !quant.uniform<u8:f16, 0.0017310915969488189:127>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @OptimizeWorkloadForDepthwiseConvWithU8(%arg0: tensor<1x128x56x56x!qElemType, {order = #NHWC}>) -> tensor<1x128x54x54x!qElemType, {order = #NHWC}> {
    %cst0 = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<128x1x1x4xsi32, {order = #NHWC}> = dense<10> : tensor<128x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = [@CMX_NN, 0]} : tensor<1x128x56x56x!qElemType, {order = #NHWC}> -> tensor<1x128x56x56x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = [@CMX_NN, 0]} : tensor<128x16x1x1xf16, {order = #NHWC}> -> tensor<128x16x1x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    %2 = VPU.Copy(%wt) {out_mem_space = [@CMX_NN, 0]} : tensor<128x1x1x4xsi32, {order = #NHWC}> -> tensor<128x1x1x4xsi32, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    %4 = VPU.NCE.DepthConvolution(%0, %1, %2) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 1, 3, 3], strides = [1, 1]} -> tensor<1x128x54x54x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}> {
      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 128, 28, 28] <left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64> <CUBOID_16x16>
    }

    %5 = VPU.Copy(%4) : tensor<1x128x54x54x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        -> tensor<1x128x54x54x!qElemType, {order = #NHWC}>

    return %5 : tensor<1x128x54x54x!qElemType, {order = #NHWC}>

    // CHECK:       VPU.NCE.DepthConvolution


    // CHECK:         VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 28, 28] <left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64> <CUBOID_16x16>
    // CHECK-NOT:     VPU.DPU.Workload outOffsets [0, 16, 0, 0]
    // CHECK:         VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 32, 28, 28] <left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64> <CUBOID_16x16>
    // CHECK-NOT:     VPU.DPU.Workload outOffsets [0, 48, 0, 0]
    // CHECK:         VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 32, 28, 28] <left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64> <CUBOID_16x16>
    // CHECK-NOT:     VPU.DPU.Workload outOffsets [0, 80, 0, 0]
    // CHECK:         VPU.DPU.Workload outOffsets [0, 96, 0, 0] outSizes [1, 32, 28, 28] <left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64> <CUBOID_16x16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType0 = !quant.uniform<u8:f16, 1.000000e+00>

!Input_CMX = !VPU.DistributedTensor<
    1x3x224x224xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [1, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments
}>

!Output_CMX = !VPU.DistributedTensor<
    1x4x224x224x!qElemType0, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [7, 7],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [1, 1],
    num_clusters = 2: i64,
    uniform_distributed_segments
}>

!Input_DDR = tensor<1x3x224x224xf16, {order = #NCHW}>
!InputStub_CMX = tensor<1x3x224x224xf16, {mem_space = @CMX_NN, order = #NCHW}>
!OutputStub_CMX = tensor<1x4x224x224x!qElemType0, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @NCEPermuteClustered
func.func @NCEPermuteClustered(%arg0: !Input_DDR) -> !Output_CMX {
    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : !Input_DDR -> !Input_CMX

    %output = VPU.NCE.Permute(%0) {
            dstElemType = !qElemType0, dstOrder = #NHWC,
            expandedChannels = 4 : i64, minimumHardwareExecutionCost = 4294967195 : i64,
            ppe = #VPU.PPEStub<>
        } -> !Output_CMX {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 4, 112, 224] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 112, 0] outSizes [1, 4, 112, 224] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
        }

    return %output : !Output_CMX

    // CHECK:       VPU.NCE.Permute
    // CHECK-SAME:      dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x4x224x224x!qElemType, #NHWC, @CMX_NN, {
    // CHECK-SAME:           mode = "OVERLAPPED",
    // CHECK-SAME:           num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:           kernel = [7, 7],
    // CHECK-SAME:           pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:           strides = [1, 1],
    // CHECK-SAME:           num_clusters = 2 : i64,
    // CHECK-SAME:           uniform_distributed_segments}>

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [0, 0, 0, 0] outSizes [1, 3, 112, 224]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 0

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [0, 0, 112, 0] outSizes [1, 3, 112, 224]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 1
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Input_CMX = !VPU.DistributedTensor<
    1x256x24x42xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

!Weights_CMX = !VPU.DistributedTensor<
    256x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments
}>

!WeightsTable_CMX = !VPU.DistributedTensor<
    256x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments
}>

!Output_CMX = !VPU.DistributedTensor<
    1x256x24x42xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

// CHECK-LABEL: @DepthConvWithL1aOpt
func.func @DepthConvWithL1aOpt(%arg0: !Input_CMX) -> !Output_CMX {
    %cst0 = const.Declare tensor<256x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<256x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<256x1x1x4xsi32, {order = #NCHW}> =
        dense<10> : tensor<256x1x1x4xsi32>

    %0 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<256x16x1x1xf16, {order = #NHWC}> -> !Weights_CMX

    %1 = VPU.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<256x1x1x4xsi32, {order = #NCHW}> -> !WeightsTable_CMX

    %2 = VPU.NCE.DepthConvolution(%arg0, %0, %1) {
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [256, 1, 3, 3],
            strides = [1, 1]} -> !Output_CMX {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 128, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
                VPU.DPU.Workload outOffsets [0, 128, 0, 0] outSizes [1, 128, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
        }

    return %2 : !Output_CMX


    // CHECK:       VPU.NCE.DepthConvolution
    // split workload into size 32 to enable small kernel optimization
    // CHECK:          VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 32, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 32, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 96, 0, 0] outSizes [1, 32, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 128, 0, 0] outSizes [1, 32, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 160, 0, 0] outSizes [1, 32, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 192, 0, 0] outSizes [1, 32, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 224, 0, 0] outSizes [1, 32, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Input_CMX = !VPU.DistributedTensor<
    1x128x24x42xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

!Weights_CMX = !VPU.DistributedTensor<
    128x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[64, 16, 1, 1], [64, 16, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [64, 0, 0, 0]],
    memory_shapes = [[64, 16, 1, 1], [64, 16, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [64, 0, 0, 0]]
}>

!WeightsTable_CMX = !VPU.DistributedTensor<
    128x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[64, 1, 1, 4], [64, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [64, 0, 0, 0]],
    memory_shapes = [[64, 1, 1, 4], [64, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [64, 0, 0, 0]]
}>

!Output_CMX = !VPU.DistributedTensor<
    1x128x24x42xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[1, 64, 24, 42], [1, 64, 24, 42]],
    compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]],
    memory_shapes = [[1, 128, 24, 42], [1, 128, 24, 42]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: @DepthConvWithL1aOpt
func.func @DepthConvWithL1aOpt(%arg0: !Input_CMX) -> !Output_CMX {
    %cst0 = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<128x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<128x1x1x4xsi32, {order = #NCHW}> =
        dense<10> : tensor<128x1x1x4xsi32>

    %0 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<128x16x1x1xf16, {order = #NHWC}> -> !Weights_CMX
    %1 = VPU.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<128x1x1x4xsi32, {order = #NCHW}> -> !WeightsTable_CMX

    %2 = VPU.NCE.DepthConvolution(%arg0, %0, %1) {
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [128, 1, 3, 3],
            strides = [1, 1]} -> !Output_CMX {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
                VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
        }

    return %2 : !Output_CMX


    // CHECK:       VPU.NCE.DepthConvolution
    // CHECK:          VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 24, 42]
    // CHECK-SAME:              <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16>
    // CHECK-SAME:              attributes {cluster_id = 0 : i64}
    // CHECK:          VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 32, 24, 42]
    // CHECK-SAME:              <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16>
    // CHECK-SAME:              attributes {cluster_id = 0 : i64}
    // CHECK:          VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 32, 24, 42]
    // CHECK-SAME:              <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16>
    // CHECK-SAME:              attributes {cluster_id = 1 : i64}
    // CHECK:          VPU.DPU.Workload outOffsets [0, 96, 0, 0] outSizes [1, 32, 24, 42]
    // CHECK-SAME:              <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16>
    // CHECK-SAME:              attributes {cluster_id = 1 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Input_CMX = !VPU.DistributedTensor<
    1x256x24x42xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

!Weights_CMX = !VPU.DistributedTensor<
    256x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments
}>

!WeightsTable_CMX = !VPU.DistributedTensor<
    256x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments
}>

!Output_CMX = !VPU.DistributedTensor<
    1x256x24x42xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

// CHECK-LABEL: @DepthConvWithL1aOpt
func.func @DepthConvWithL1aOpt(%arg0: !Input_CMX) -> !Output_CMX {
    %cst0 = const.Declare tensor<256x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<256x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<256x1x1x4xsi32, {order = #NCHW}> =
        dense<10> : tensor<256x1x1x4xsi32>

    %0 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<256x16x1x1xf16, {order = #NHWC}> -> !Weights_CMX
    %1 = VPU.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<256x1x1x4xsi32, {order = #NCHW}> -> !WeightsTable_CMX

    %2 = VPU.NCE.DepthConvolution(%arg0, %0, %1) {
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [256, 1, 3, 3],
            strides = [1, 1]} -> !Output_CMX {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 128, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
                VPU.DPU.Workload outOffsets [0, 128, 0, 0] outSizes [1, 128, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
        }

    return %2 : !Output_CMX


    // CHECK:       VPU.NCE.DepthConvolution
    // split workload into size 32 to enable small kernel optimization
    // CHECK:          VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 32, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 32, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 96, 0, 0] outSizes [1, 32, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 128, 0, 0] outSizes [1, 32, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 160, 0, 0] outSizes [1, 32, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 192, 0, 0] outSizes [1, 32, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 224, 0, 0] outSizes [1, 32, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Input_CMX = !VPU.DistributedTensor<
    1x2048x24x42xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

!Weights_CMX = !VPU.DistributedTensor<
    2048x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments
}>

!WeightsTable_CMX = !VPU.DistributedTensor<
    2048x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments
}>

!Output_CMX = !VPU.DistributedTensor<
    1x2048x24x42xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

// CHECK-LABEL: @DepthConvWithoutL1aOpt
func.func @DepthConvWithoutL1aOpt(%arg0: !Input_CMX) -> !Output_CMX {
    %cst0 = const.Declare tensor<2048x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<2048x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<2048x1x1x4xsi32, {order = #NCHW}> =
        dense<10> : tensor<2048x1x1x4xsi32>

    %0 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<2048x16x1x1xf16, {order = #NHWC}> -> !Weights_CMX
    %1 = VPU.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<2048x1x1x4xsi32, {order = #NCHW}> -> !WeightsTable_CMX

    %2 = VPU.NCE.DepthConvolution(%arg0, %0, %1) {
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [2048, 1, 3, 3],
            strides = [1, 1]} -> !Output_CMX {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 1024, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
                VPU.DPU.Workload outOffsets [0, 1024, 0, 0] outSizes [1, 1024, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
        }

    return %2 : !Output_CMX


    // CHECK:       VPU.NCE.DepthConvolution
    // Don't split workload into size 32 to enable small kernel optimization
    // CHECK:          VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 128, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 192, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 256, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 320, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 384, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 448, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 512, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 576, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 640, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 704, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 768, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 832, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 896, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 960, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 1024, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 1088, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 1152, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 1216, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 1280, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 1344, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 1408, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 1472, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 1536, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 1600, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 1664, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 1728, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 1792, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 1856, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 1920, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 1984, 0, 0] outSizes [1, 64, 24, 42] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!SparseInputType = !VPU.SparseTensor<
    data=tensor<1x64x16x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>,
    sparsity_map=tensor<1x64x4x4xi1, {mem_space = [@CMX_NN, 0], order = #NHWC}>
>

// CHECK-LABEL: @DepthConvSparseInputWithoutL1aOpt
func.func @DepthConvSparseInputWithoutL1aOpt(
    %ACT: !SparseInputType,
    %FILT: tensor<64x16x1x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>,
    %WEIGHT_TABLE: tensor<64x1x1x4xsi32, {mem_space = [@CMX_NN, 0], order = #NCHW}>
) -> tensor<1x64x16x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> {
    // CHECK:   [[ACT:%.+]]: !VPU.SparseTensor<{{.*}}>, [[FILT:%.+]]: tensor<64x16x1x1xf16{{.*}}>, [[WEIGHT_TABLE:%.+]]: tensor<64x1x1x4xsi32{{.*}}>
    %DWCONV = VPU.NCE.DepthConvolution(%ACT, %FILT, %WEIGHT_TABLE) {
        minimumHardwareExecutionCost = 790 : i64,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEInt<
            mode = <NOOP>,
            clamp_low = -2147483648 : i64,
            clamp_high = 2147483647 : i64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64
        >,
        rawFilterShape = [64, 1, 3, 3],
        strides = [1, 1]
    } -> tensor<1x64x16x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> {
        VPU.DPU.Workload
            outOffsets [0, 0, 0, 0]
            outSizes [1, 64, 16, 16]
            <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
            <CUBOID_16x16>
    }
    // CHECK:   [[DWCONV:%.+]] = VPU.NCE.DepthConvolution([[ACT]], [[FILT]], [[WEIGHT_TABLE]])
    // CHECK-NEXT:  VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [0, 0, 0, 0]
    // CHECK-SAME:      outSizes [1, 64, 16, 16]

    return %DWCONV : tensor<1x64x16x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:   return [[DWCONV]] : tensor<1x64x16x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
}
