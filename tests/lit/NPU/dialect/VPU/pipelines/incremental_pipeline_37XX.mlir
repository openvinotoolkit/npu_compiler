//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --incremental-pipeline %s | FileCheck %s
// REQUIRES: arch-NPU37XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @MaxpoolIncrementalPipelineCheck(%arg0: tensor<1x16x1x4xf16, {order = #NHWC}>) -> tensor<1x16x1x4xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) { kernel_size = [1, 1], pad =  #VPU.Padding<bottom = 0, left = 0, right = 0, top = 0>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x16x1x4xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x4xf16, {order = #NHWC}>
}

    //CHECK: [[OP0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    //CHECK-SAME:         -> !VPU.DistributedTensor<1x16x1x4xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK: [[OP1:%.+]] = VPU.NCE.MaxPool([[OP0]]) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]}
    //CHECK-SAME:         -> !VPU.DistributedTensor<1x16x1x4xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK: [[OUT:%.+]] = VPU.Copy([[OP1]])
    //CHECK-SAME:         : !VPU.DistributedTensor<1x16x1x4xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK-SAME:         -> tensor<1x16x1x4xf16, {order = #NHWC}>

    //CHECK: return [[OUT]] : tensor<1x16x1x4xf16, {order = #NHWC}>


// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @ConvIncrementalPipeline(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad =  #VPU.Padding<bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} : tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<80x64x3x3xf16, {order = #NHWC}>, tensor<80x1x1x4xsi32> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x80x28x28xf16, {order = #NHWC}>


    //CHECK: [[CONST:%.+]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    //CHECK: [[CONST_0:%.+]] =  const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK: [[OP0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK: [[OP1:%.+]] = VPU.Copy([[CONST_0]]) {out_mem_space = @CMX_NN}
    //CHECK-SAME:   -> !VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK: [[OP2:%.+]] = VPU.Copy([[CONST]]) {out_mem_space = @CMX_NN}
    //CHECK-SAME:   -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK: [[OP3:%.+]] = VPU.NCE.Convolution([[OP0]], [[OP1]], [[OP2]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]}
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK: [[OP4:%.+]] = VPU.Copy([[OP3]])
    //CHECK-SAME:   : !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK-SAME:   -> tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK: return [[OP4]] : tensor<1x80x28x28xf16, {order = #NHWC}>

}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @TanHIncrementalPipeline(%arg0: tensor<1x1x16x4xf16, {order = #NHWC}>) -> tensor<1x1x16x4xf16, {order = #NHWC}> {
    %1 = VPU.Tanh(%arg0) : tensor<1x1x16x4xf16, {order = #NHWC}> -> tensor<1x1x16x4xf16, {order = #NHWC}>
    return %1 : tensor<1x1x16x4xf16, {order = #NHWC}>
}

    //CHECK: [[OP0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1x16x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK: [[OP1:%.+]] = VPU.Tanh([[OP0]])
    //CHECK-SAME:   : !VPU.DistributedTensor<1x1x16x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1x16x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK: [[OP2:%.+]] = VPU.Copy([[OP1]])
    //CHECK-SAME:   : !VPU.DistributedTensor<1x1x16x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK-SAME:   -> tensor<1x1x16x4xf16, {order = #NHWC}>
    //CHECK: return [[OP2]] : tensor<1x1x16x4xf16, {order = #NHWC}>

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEPermuteIncrementalPipeline
// CHECK-SAME:        [[INPUT:%.+]]: tensor<1x3x64x64xf16>
func.func @NCEPermuteIncrementalPipeline(%arg0: tensor<1x3x64x64xf16>) -> tensor<1x16x32x32x!qElemType, {order = #NHWC}> {
    %weights = const.Declare tensor<16x1x1x48x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x1x1x48xf16>,
                [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %wt = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %nce_permute = VPU.NCE.Permute(%arg0)
            {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64, ppe = #VPU.PPEStub<>}
    -> tensor<1x4x64x64x!qElemType, {order = #NHWC}>

    %compress_conv = VPU.NCE.CompressConvolution(%nce_permute, %weights, %wt)
            {cm_sp_pattern = 1 : i64,
            pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 1, 3, 3], strides = [2, 2]}
    -> tensor<1x16x32x32x!qElemType, {order = #NHWC}>

    return %compress_conv : tensor<1x16x32x32x!qElemType, {order = #NHWC}>

    //CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<16x1x1x48x!qElemType, {order = #NHWC}>
    //CHECK:        [[WT:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    //CHECK:        [[IN_CMX:%.+]] = VPU.Copy([[INPUT]]) {out_mem_space = @CMX_NN}
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x3x64x64xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3],
    //CHECK-SAME:           pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [2, 2], num_clusters = 2 : i64}>

    //CHECK:        [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[IN_CMX]]) {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64, ppe = #VPU.PPEStub<>}
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x4x64x64x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3],
    //CHECK-SAME:           pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>

    //CHECK:        [[COPY_TO_DDR:%.+]] = VPU.Copy([[NCE_PERMUTE]])
    //CHECK-SAME:           : !VPU.DistributedTensor<1x4x64x64x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3],
    //CHECK-SAME:           pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>
    //CHECK-SAME:           -> tensor<1x4x64x64x!qElemType, {order = #NHWC}>

    //CHECK:        [[COPY_TO_CMX:%.+]] = VPU.Copy([[COPY_TO_DDR]]) {out_mem_space = @CMX_NN}
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x4x64x64x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3],
    //CHECK-SAME:           pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [2, 2], num_clusters = 2 : i64}>

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = @CMX_NN}
    //CHECK-SAME:           -> !VPU.DistributedTensor<16x1x1x48x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[WT_CMX:%.+]] = VPU.Copy([[WT]]) {out_mem_space = @CMX_NN}
    //CHECK-SAME:           -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[COMPRESS_CONV:%.+]] = VPU.NCE.CompressConvolution([[COPY_TO_CMX]], [[WEIGHTS_CMX]], [[WT_CMX]])
    //CHECK-SAME:           {cm_sp_pattern = 1 : i64, pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    //CHECK-SAME:           ppe = #VPU.PPEStub<>, rawFilterShape = [16, 1, 3, 3], strides = [2, 2]}
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x16x32x32x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[COMPRESS_CONV]])
    //CHECK-SAME:           : !VPU.DistributedTensor<1x16x32x32x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK-SAME:           -> tensor<1x16x32x32x!qElemType, {order = #NHWC}>

    //CHECK:    return [[OUT]] : tensor<1x16x32x32x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @ConvTanHIncrementalPipeline(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad =  #VPU.Padding<bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} : tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<80x64x3x3xf16, {order = #NHWC}>, tensor<80x1x1x4xsi32> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    %1 = VPU.Tanh(%0) : tensor<1x80x28x28xf16, {order = #NHWC}> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %1 : tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK: [[CONST:%.+]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    //CHECK: [[CONST_0:%.+]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK: [[OP0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK: [[OP1:%.+]] = VPU.Copy([[CONST_0]]) {out_mem_space = @CMX_NN}
    //CHECK-SAME:       -> !VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK: [[OP2:%.+]] = VPU.Copy([[CONST]]) {out_mem_space = @CMX_NN}
    //CHECK-SAME:       -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK: [[OP3:%.+]] = VPU.NCE.Convolution([[OP0]], [[OP1]], [[OP2]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]}
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK: [[OP4:%.+]] = VPU.Copy([[OP3]])
    //CHECK-SAME:       : !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK-SAME:       -> tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK: [[OP5:%.+]] = VPU.Copy([[OP4]]) {out_mem_space = @CMX_NN}
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK: [[OP6:%.+]] = VPU.Tanh([[OP5]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK: [[OP7:%.+]] = VPU.Copy([[OP6]])
    //CHECK-SAME:       -> tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK: return [[OP7]] : tensor<1x80x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAssignedSOHWithOddWidthAndSmallHeight
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x4x331776xf16, {order = #NHWC}>)
func.func @EltwiseAssignedSOHWithOddWidthAndSmallHeight(%arg0: tensor<1x16x4x331776xf16, {order = #NHWC}>) -> tensor<1x16x4x16186xf16, {order = #NHWC}> {
    %eltwise1_input2 = const.Declare tensor<1x16x4x8093xf16, {order = #NHWC}> = dense<1.0> : tensor<1x16x4x8093xf16>, [#const.Reorder<#NHWC>]
    %eltwise2_input2 = const.Declare tensor<1x16x4x8093xf16, {order = #NHWC}> = dense<1.0> : tensor<1x16x4x8093xf16>, [#const.Reorder<#NHWC>]

    %eltwise1_input1 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 16, 4, 8093] : tensor<1x16x4x331776xf16, {order = #NHWC}> to tensor<1x16x4x8093xf16, {order = #NHWC}>
    %eltwise1 = VPU.NCE.Eltwise(%eltwise1_input1, %eltwise1_input2) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x16x4x8093xf16, {order = #NHWC}>

    %eltwise2_input1 = VPU.Slice %arg0 [0, 0, 0, 8093] [1, 16, 4, 8093] : tensor<1x16x4x331776xf16, {order = #NHWC}> to tensor<1x16x4x8093xf16, {order = #NHWC}>
    %eltwise2 = VPU.NCE.Eltwise(%eltwise2_input1, %eltwise2_input2) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x16x4x8093xf16, {order = #NHWC}>

    %concat = VPU.Concat(%eltwise1, %eltwise2) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 8093]]} : tensor<1x16x4x8093xf16, {order = #NHWC}>, tensor<1x16x4x8093xf16, {order = #NHWC}> -> tensor<1x16x4x16186xf16, {order = #NHWC}>
    return %concat : tensor<1x16x4x16186xf16, {order = #NHWC}>

    //CHECK-DAG: [[ELTWISE_INPUT2:%.+]] = const.Declare tensor<1x16x4x8093xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x16x4x8093xf16>, [#const.Reorder<#NHWC>]
    //CHECK: [[SLICE_0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 16, 4, 8093] : tensor<1x16x4x331776xf16, {order = #NHWC}> to tensor<1x16x4x8093xf16, {order = #NHWC}>
    //CHECK: [[ELTWISE_0_COPY_0:%.+]] = VPU.Copy([[SLICE_0]]) {out_mem_space = @CMX_NN}
    //CHECK-SAME:    -> !VPU.DistributedTensor<1x16x4x8093xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK: [[ELTWISE_0_COPY_1:%.+]] = VPU.Copy([[ELTWISE_INPUT2]]) {out_mem_space = @CMX_NN}
    //CHECK-SAME:    -> !VPU.DistributedTensor<1x16x4x8093xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK: [[ELTWISE_0:%.+]]  = VPU.NCE.Eltwise([[ELTWISE_0_COPY_0]], [[ELTWISE_0_COPY_1]]) {op_type = #VPU.eltwise_type<ADD>
    //CHECK-SAME:    -> !VPU.DistributedTensor<1x16x4x8093xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK: [[ELTWISE_0_COPY_OUT:%.+]] = VPU.Copy([[ELTWISE_0]])
    //CHECK-SAME:    : !VPU.DistributedTensor<1x16x4x8093xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK-SAME:    -> tensor<1x16x4x8093xf16, {order = #NHWC}>

    //CHECK: [[SLICE_1:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 8093] [1, 16, 4, 8093] : tensor<1x16x4x331776xf16, {order = #NHWC}> to tensor<1x16x4x8093xf16, {order = #NHWC}>
    //CHECK: [[ELTWISE_1_COPY_0:%.+]] = VPU.Copy([[SLICE_1]]) {out_mem_space = @CMX_NN}
    //CHECK-SAME:    -> !VPU.DistributedTensor<1x16x4x8093xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK: [[ELTWISE_1_COPY_1:%.+]] = VPU.Copy([[ELTWISE_INPUT2]]) {out_mem_space = @CMX_NN}
    //CHECK-SAME:    -> !VPU.DistributedTensor<1x16x4x8093xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK: [[ELTWISE_1:%.+]]  = VPU.NCE.Eltwise([[ELTWISE_1_COPY_0]], [[ELTWISE_1_COPY_1]]) {op_type = #VPU.eltwise_type<ADD>
    //CHECK-SAME:    -> !VPU.DistributedTensor<1x16x4x8093xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK: [[ELTWISE_1_COPY_OUT:%.+]] = VPU.Copy([[ELTWISE_1]])
    //CHECK-SAME:    : !VPU.DistributedTensor<1x16x4x8093xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK-SAME:    -> tensor<1x16x4x8093xf16, {order = #NHWC}>

    //CHECK: [[CONCAT:%.+]] = VPU.Concat([[ELTWISE_0_COPY_OUT]], [[ELTWISE_1_COPY_OUT]]) {
    //CHECK:     static_offsets = [
    //CHECK-SAME:   [0, 0, 0, 0], [0, 0, 0, 8093]
    //CHECK:     ]} : tensor<1x16x4x8093xf16, {order = #NHWC}>, tensor<1x16x4x8093xf16, {order = #NHWC}> -> tensor<1x16x4x16186xf16, {order = #NHWC}>
    //CHECK: return [[CONCAT]] : tensor<1x16x4x16186xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @InterpolateIncrementalPipeline(
// CHECK-SAME:                    %[[VAL_0:.+]]: tensor<1x2x540x1920xf16>) -> tensor<1x2x256x512xf16> {
func.func @InterpolateIncrementalPipeline(%arg0: tensor<1x2x540x1920xf16>) -> tensor<1x2x256x512xf16> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [1.3333300352096558, 1.3333300352096558], sizes_attr = [256, 512]} : tensor<1x2x540x1920xf16> -> tensor<1x2x256x512xf16>
    return %0 : tensor<1x2x256x512xf16>

    // CHECK:           %[[VAL_1:.+]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 0] [1, 2, 540, 959] : tensor<1x2x540x1920xf16> to tensor<1x2x540x959xf16>
    // CHECK:           %[[VAL_2:.+]] = VPU.Copy(%[[VAL_1]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x2x540x959xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = {{\[\[}}1, 2, 270, 959], [1, 2, 270, 959]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 270, 0]], memory_shapes = {{\[\[}}1, 2, 270, 959], [1, 2, 270, 959]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 270, 0]]}>

    // CHECK-DAG:       %[[COORDINATES_0:.+]] = const.Declare tensor<1x1x1x256xsi32>
    // CHECK-DAG:       %[[COORDINATES_CMX_0:.+]] = VPU.Copy(%[[COORDINATES_0]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x256xsi32> -> !VPU.DistributedTensor<1x1x1x256xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK-DAG:       %[[LAMBDAS_0:.+]] = const.Declare tensor<1x1x1x512xf16>
    // CHECK-DAG:       %[[LAMBDAS_CMX_0:.+]] = VPU.Copy(%[[LAMBDAS_0]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x512xf16> -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:           %[[VAL_3:.+]] = VPU.Interpolate(%[[VAL_2]], %[[COORDINATES_CMX_0]], %[[LAMBDAS_CMX_0]]) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SCALES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], initial_input_dims_attr = [1, 2, 540, 1920], initial_input_offset_attr = [0, 0, 0, 0], initial_output_dims_attr = [1, 2, 256, 512], initial_output_offset_attr = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0, 1, 1>, scales_attr = [0.47407407407407409, 0.26694473409801878], sizes_attr = [256, 512], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]}
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x2x256x256xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:           %[[VAL_4:.+]] = VPU.Copy(%[[VAL_3]])
    // CHECK-SAME:           : !VPU.DistributedTensor<1x2x256x256xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:           -> tensor<1x2x256x256xf16>

    // CHECK:           %[[VAL_10:.+]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 961] [1, 2, 540, 959] : tensor<1x2x540x1920xf16> to tensor<1x2x540x959xf16>
    // CHECK:           %[[VAL_11:.+]] = VPU.Copy(%[[VAL_10]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x2x540x959xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = {{\[\[}}1, 2, 270, 959], [1, 2, 270, 959]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 270, 0]], memory_shapes = {{\[\[}}1, 2, 270, 959], [1, 2, 270, 959]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 270, 0]]}>


    // CHECK-DAG:       %[[COORDINATES_1:.+]] = const.Declare tensor<1x1x1x256xsi32>
    // CHECK-DAG:       %[[COORDINATES_CMX_1:.+]] = VPU.Copy(%[[COORDINATES_1]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x256xsi32> -> !VPU.DistributedTensor<1x1x1x256xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK-DAG:       %[[LAMBDAS_1:.+]] = const.Declare tensor<1x1x1x512xf16>
    // CHECK-DAG:       %[[LAMBDAS_CMX_1:.+]] = VPU.Copy(%[[LAMBDAS_1]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x512xf16> -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:           %[[VAL_12:.+]] = VPU.Interpolate(%[[VAL_11]], %[[COORDINATES_CMX_1]], %[[LAMBDAS_CMX_1]]) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SCALES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], initial_input_dims_attr = [1, 2, 540, 1920], initial_input_offset_attr = [0, 0, 0, 961], initial_output_dims_attr = [1, 2, 256, 512], initial_output_offset_attr = [0, 0, 0, 256], operandSegmentSizes = array<i32: 1, 0, 0, 0, 1, 1>, scales_attr = [0.47407407407407409, 0.26694473409801878], sizes_attr = [256, 512], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]}
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x2x256x256xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:           %[[VAL_13:.+]] = VPU.Copy(%[[VAL_12]])
    // CHECK-SAME:           : !VPU.DistributedTensor<1x2x256x256xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:           -> tensor<1x2x256x256xf16>

    // CHECK:           %[[VAL_31:.+]] = VPU.Concat(%[[VAL_4]], %[[VAL_13]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 256]]} : tensor<1x2x256x256xf16>, tensor<1x2x256x256xf16> -> tensor<1x2x256x512xf16>
    // CHECK:           return %[[VAL_31]] : tensor<1x2x256x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @ReduceSumSOK(
// CHECK-SAME:      %[[VAL_0:.+]]: tensor<1x1024x7x7xf32>) -> tensor<1x1024xf32> {
func.func @ReduceSumSOK(%arg0: tensor<1x1024x7x7xf32>) -> tensor<1x1024xf32> {
  %0 = VPU.ReduceSum(%arg0) {axes_value = [2, 3], keep_dims} : tensor<1x1024x7x7xf32> -> tensor<1x1024x1x1xf32>
  %1 = VPU.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 1024]} : tensor<1x1024x1x1xf32> -> tensor<1x1024xf32>
  return %1 : tensor<1x1024xf32>

// CHECK:   %[[VAL_1:.+]] = VPU.Copy(%[[VAL_0]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:      -> !VPU.DistributedTensor<1x1024x7x7xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

// CHECK:   %[[VAL_2:.+]] = VPU.ReduceSum(%[[VAL_1]]) {axes_value = [2, 3], keep_dims}
// CHECK-SAME:      -> !VPU.DistributedTensor<1x1024x1x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

// CHECK:   %[[VAL_3:.+]] = VPU.Copy(%[[VAL_2]])
// CHECK-SAME:      : !VPU.DistributedTensor<1x1024x1x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
// CHECK-SAME:      -> tensor<1x1024x1x1xf32>

// CHECK:   %[[VAL_10:.+]] = VPU.AffineReshape(%[[VAL_3]]) {dim_mapping = {{\[\[}}0], [1], [1], [1]], shape_value = [1, 1024]} : tensor<1x1024x1x1xf32> -> tensor<1x1024xf32>
// CHECK:   return %[[VAL_10]] : tensor<1x1024xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @ReduceSumSOH(
// CHECK-SAME:      %[[VAL_0:.+]]: tensor<1x16x32x64xf32>) -> tensor<1x1x32x1xf32> {
func.func @ReduceSumSOH(%arg0: tensor<1x16x32x64xf32>) -> tensor<1x1x32x1xf32> {
  %0 = VPU.ReduceSum(%arg0) {axes_value = [1, 3], keep_dims} : tensor<1x16x32x64xf32> -> tensor<1x1x32x1xf32>
  return %0 : tensor<1x1x32x1xf32>

// CHECK:   %[[VAL_1:.+]] = VPU.Copy(%[[VAL_0]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x64xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

// CHECK:   %[[VAL_2:.+]] = VPU.ReduceSum(%[[VAL_1]]) {axes_value = [1, 3], keep_dims}
// CHECK-SAME:      -> !VPU.DistributedTensor<1x1x32x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

// CHECK:   %[[VAL_10:.+]] = VPU.Copy(%[[VAL_2]])
// CHECK-SAME:      : !VPU.DistributedTensor<1x1x32x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK-SAME:      -> tensor<1x1x32x1xf32>

// CHECK:   return %[[VAL_10]] : tensor<1x1x32x1xf32>
}
