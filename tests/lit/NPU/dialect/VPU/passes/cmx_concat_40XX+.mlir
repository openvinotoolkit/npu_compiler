//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% ppe-version=IntPPE" --cmx-concat --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ConvInputDistributed = !VPU.DistributedTensor<
    1x128x16x157xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 9, 157], [1, 128, 9, 157]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]],
    memory_shapes = [[1, 128, 9, 157], [1, 128, 9, 157]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]]
}>

!ConcatInputConvOutputDistributed = !VPU.DistributedTensor<
    1x128x16x157xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 8, 157], [1, 128, 8, 157]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    memory_shapes = [[1, 128, 9, 157], [1, 128, 9, 157]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]]
}>

!ConcatOutputDistributed = !VPU.DistributedTensor<
    1x128x16x158xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 9, 158], [1, 128, 9, 158]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]],
    memory_shapes = [[1, 128, 9, 158], [1, 128, 9, 158]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]]
}>

!SecondConvOutputDistributed = !VPU.DistributedTensor<
    1x128x16x158xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 8, 158], [1, 128, 8, 158]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    memory_shapes = [[1, 128, 8, 158], [1, 128, 8, 158]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]
}>

!ConvInput_DDR = tensor<1x128x16x157xf16, {mem_space = @DDR, order = #NHWC}>

!ConvInputStub_CMX = tensor<1x128x16x157xf16, {mem_space = @CMX_NN, order = #NHWC}>
!ConvOutputStub_CMX = tensor<1x128x16x157xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @ConcatClusterTilingWithExplicitDistributedAttrAndConstantSecondInput
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x128x16x157xf16, {mem_space = @DDR, order = #NHWC}>,
// CHECK-SAME:  [[WEIGHTS0:%.+]]: tensor<128x128x3x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME:  [[WEIGHTS1:%.+]]: tensor<128x128x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME:  [[WTABLE:%.+]]: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
func.func @ConcatClusterTilingWithExplicitDistributedAttrAndConstantSecondInput(%arg0: !ConvInput_DDR,
        %weights0: tensor<128x128x3x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weights1: tensor<128x128x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weightsTable: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
           -> !SecondConvOutputDistributed {

    %constInput = const.Declare tensor<1x128x16x1xf16, {order = #NHWC}> = dense<0.000000e+00> :
        tensor<1x128x16x1xf16>, [#const.Reorder<#NHWC>]

    %inputCMX = VPU.Copy(%arg0) { out_mem_space = @CMX_NN } : !ConvInput_DDR -> !ConvInputDistributed

    %convOutput = VPU.NCE.Convolution(%inputCMX, %weights0, %weightsTable) {
            pad = #VPU.Padding<left = 0, right = 0, top = 1, bottom = 1>,
            ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -128 : i64, clamp_high = 127 : i64,
                lrelu_mult = 1638 : i64, lrelu_shift = 14 : i64, fp_prelu_alpha = 0.0999755859375 : f64>,
            rawFilterShape = [128, 128, 3, 1], strides = [1, 1]}
             -> !ConcatInputConvOutputDistributed

    %convOutDDR = VPU.Copy(%convOutput) : !ConcatInputConvOutputDistributed -> tensor<1x128x16x157xf16, {order = #NHWC}>

    %concatOut = VPU.Concat(%constInput, %convOutDDR) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1]]} :
        tensor<1x128x16x1xf16, {order = #NHWC}>, tensor<1x128x16x157xf16, {order = #NHWC}> -> tensor<1x128x16x158xf16, {order = #NHWC}>

    %concatOutCMX = VPU.Copy(%concatOut) {out_mem_space = @CMX_NN} : tensor<1x128x16x158xf16, {order = #NHWC}>
            -> !ConcatOutputDistributed

    %output = VPU.NCE.Convolution(%concatOutCMX, %weights1, %weightsTable) {
            pad = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
            ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -128 : i64, clamp_high = 127 : i64,
                lrelu_mult = 1638 : i64, lrelu_shift = 14 : i64, fp_prelu_alpha = 0.0999755859375 : f64>,
            rawFilterShape = [128, 128, 3, 3], strides = [1, 1]}
             -> !SecondConvOutputDistributed

    return %output: !SecondConvOutputDistributed

    //CHECK:        [[CST:%.*]] = const.Declare tensor<1x128x16x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x128x16x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[CST_INPUT:%.*]] =  VPU.Copy([[CST]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x16x1xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 128, 9, 1], [1, 128, 9, 1]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 128, 9, 1], [1, 128, 9, 1]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]]}>

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.Copy([[INPUT]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x16x157xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 128, 9, 157], [1, 128, 9, 157]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 128, 9, 157], [1, 128, 9, 157]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]]}>

    //CHECK:        [[CONV_OUTPUT:%.*]] = VPU.NCE.Convolution([[INPUT_CMX]], [[WEIGHTS0]], [[WTABLE]])
    //CHECK-SAME:                -> !VPU.DistributedTensor<1x128x16x157xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                        {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:                compute_shapes = [[1, 128, 8, 157], [1, 128, 8, 157]],
    //CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    //CHECK-SAME{LITERAL}:                memory_shapes = [[1, 128, 9, 157], [1, 128, 9, 157]],
    //CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]]}>
    
    //CHECK:        [[CONCAT_OUTPUT:%.*]] = VPU.Concat([[CST_INPUT]], [[CONV_OUTPUT]])
    //CHECK-SAME{LITERAL}:             {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1]]} :
    //CHECK-SAME:       !VPU.DistributedTensor<1x128x16x1xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 128, 9, 1], [1, 128, 9, 1]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 128, 9, 1], [1, 128, 9, 1]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]]}>
    //CHECK-SAME:       !VPU.DistributedTensor<1x128x16x157xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 128, 8, 157], [1, 128, 8, 157]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 128, 9, 157], [1, 128, 9, 157]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]]}>
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x16x158xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 128, 9, 158], [1, 128, 9, 158]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 128, 9, 158], [1, 128, 9, 158]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]]}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.Convolution([[CONCAT_OUTPUT]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x16x158xf16, #NHWC, @CMX_NN
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 128, 8, 158], [1, 128, 8, 158]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 128, 8, 158], [1, 128, 8, 158]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ConvIn0 = !VPU.DistributedTensor<
    1x64x20x32xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 1, 2],
    num_clusters = 2,
    compute_shapes = [[1, 64, 20, 16], [1, 64, 20, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 16]],
    memory_shapes = [[1, 64, 20, 16], [1, 64, 20, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 16]]
}>

!ConvIn1 = !VPU.DistributedTensor<
    1x16x20x32xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 1, 2],
    num_clusters = 2,
    compute_shapes = [[1, 16, 20, 16], [1, 16, 20, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 16]],
    memory_shapes = [[1, 16, 20, 16], [1, 16, 20, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 16]]
}>

!ConvOut0 = !VPU.DistributedTensor<
    1x64x20x32xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 1, 2],
    num_clusters = 2,
    compute_shapes = [[1, 64, 20, 16], [1, 64, 20, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 16]],
    memory_shapes = [[1, 64, 20, 18], [1, 64, 20, 18]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 14]]
}>

!ConvOut1 = !VPU.DistributedTensor<
    1x16x20x32xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 1, 2],
    num_clusters = 2,
    compute_shapes = [[1, 16, 20, 16], [1, 16, 20, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 16]],
    memory_shapes = [[1, 16, 20, 18], [1, 16, 20, 18]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 14]]
}>

!ConvIn23 = !VPU.DistributedTensor<
    1x80x12x32xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 1, 2],
    num_clusters = 2,
    compute_shapes = [[1, 80, 12, 18], [1, 80, 12, 18]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 14]],
    memory_shapes = [[1, 80, 12, 18], [1, 80, 12, 18]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 14]]
}>

!ConvOut23 = !VPU.DistributedTensor<
    1x64x12x32xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 1, 2],
    num_clusters = 2,
    compute_shapes = [[1, 64, 12, 16], [1, 64, 12, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 16]],
    memory_shapes = [[1, 64, 12, 16], [1, 64, 12, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 16]]
}>

!ConvInOut0_DDR = tensor<1x64x20x32xf16, {mem_space = @DDR, order = #NHWC}>
!ConvInOut1_DDR = tensor<1x16x20x32xf16, {mem_space = @DDR, order = #NHWC}>
!ConvIn23_DDR = tensor<1x80x12x32xf16, {mem_space = @DDR, order = #NHWC}>

!ConvInOutStub0_CMX = tensor<1x64x20x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
!ConvInOutStub1_CMX = tensor<1x16x20x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
!ConvInStub23_CMX = tensor<1x80x12x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @SOWProducersSOWConsumersOfConcatWithExplicitDistributedAttrWithHSlice
// CHECK-SAME:  [[INPUT0:%.+]]: tensor<1x64x20x32xf16, {mem_space = @DDR, order = #NHWC}>,
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<1x16x20x32xf16, {mem_space = @DDR, order = #NHWC}>,
// CHECK-SAME:  [[WEIGHTS0:%.+]]: tensor<64x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME:  [[WEIGHTS1:%.+]]: tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME:  [[WEIGHTS2:%.+]]: tensor<64x80x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME:  [[WEIGHTS3:%.+]]: tensor<64x80x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME:  [[WTABLE0:%.+]]: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
// CHECK-SAME:  [[WTABLE1:%.+]]: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
func.func @SOWProducersSOWConsumersOfConcatWithExplicitDistributedAttrWithHSlice(
        %input0: !ConvInOut0_DDR,
        %input1: !ConvInOut1_DDR,
        %weights0: tensor<64x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weights1: tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weights2: tensor<64x80x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weights3: tensor<64x80x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weightsTable0: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
        %weightsTable1: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
           -> (!ConvOut23, !ConvOut23) {

    %input0CMX = VPU.Copy(%input0) { out_mem_space = @CMX_NN } : !ConvInOut0_DDR
     -> !ConvIn0

    %conv0Output = VPU.NCE.Convolution(%input0CMX, %weights0, %weightsTable0) {
            pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
            rawFilterShape = [64, 64, 1, 1], strides = [1, 1]}
             -> !ConvOut0

    %conv0OutDDR = VPU.Copy(%conv0Output) : !ConvOut0 -> !ConvInOut0_DDR

    %input1CMX = VPU.Copy(%input1) { out_mem_space = @CMX_NN } : !ConvInOut1_DDR -> !ConvIn1

    %conv1Output = VPU.NCE.Convolution(%input1CMX, %weights1, %weightsTable1) {
            pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
            rawFilterShape = [16, 16, 1, 1], strides = [1, 1]}
             -> !ConvOut1

    %conv1OutDDR = VPU.Copy(%conv1Output) : !ConvOut1 -> !ConvInOut1_DDR

    %concatOut = VPU.Concat(%conv0OutDDR, %conv1OutDDR) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} :
        !ConvInOut0_DDR, !ConvInOut1_DDR -> tensor<1x80x20x32xf16, {mem_space = @DDR, order = #NHWC}>

    %slice0 = VPU.Slice %concatOut [0, 0, 0, 0] [1, 80, 12, 32] : tensor<1x80x20x32xf16, {mem_space = @DDR, order = #NHWC}> to !ConvIn23_DDR

    %input2CMX = VPU.Copy(%slice0) {out_mem_space = @CMX_NN} : !ConvIn23_DDR -> !ConvIn23

    %output0 = VPU.NCE.Convolution(%input2CMX, %weights2, %weightsTable0) {
            pad = #VPU.Padding<left = 1, right = 1, top = 1, bottom = 1>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
            rawFilterShape = [64, 80, 3, 3], strides = [1, 1]}
             -> !ConvOut23

    %slice1 = VPU.Slice %concatOut [0, 0, 8, 0] [1, 80, 12, 32] : tensor<1x80x20x32xf16, {mem_space = @DDR, order = #NHWC}> to !ConvIn23_DDR

    %input3CMX = VPU.Copy(%slice1) {out_mem_space = @CMX_NN} : !ConvIn23_DDR -> !ConvIn23

    %output1 = VPU.NCE.Convolution(%input3CMX, %weights3, %weightsTable0) {
            pad = #VPU.Padding<left = 2, right = 2, top = 2, bottom = 2>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
            rawFilterShape = [64, 80, 5, 5], strides = [1, 1]}
             -> !ConvOut23

    return %output0, %output1: !ConvOut23, !ConvOut23

    //CHECK:        [[IN0_CMX:%.*]] =  VPU.Copy([[INPUT0]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x20x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 64, 20, 16], [1, 64, 20, 16]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 16]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 64, 20, 16], [1, 64, 20, 16]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 16]]}>

    //CHECK:        [[CONV0_OUT:%.*]] = VPU.NCE.Convolution([[IN0_CMX]], [[WEIGHTS0]], [[WTABLE0]])
    //CHECK-SAME:                -> !VPU.DistributedTensor<1x64x20x32xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                        {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:                compute_shapes = [[1, 64, 20, 16], [1, 64, 20, 16]],
    //CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 16]],
    //CHECK-SAME{LITERAL}:                memory_shapes = [[1, 64, 20, 18], [1, 64, 20, 18]],
    //CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 14]]}>

    //CHECK:        [[IN1_CMX:%.*]] = VPU.Copy([[INPUT1]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x20x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 16, 20, 16], [1, 16, 20, 16]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 16]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 16, 20, 16], [1, 16, 20, 16]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 16]]}>

    //CHECK:        [[CONV1_OUT:%.*]] = VPU.NCE.Convolution([[IN1_CMX]], [[WEIGHTS1]], [[WTABLE1]])
    //CHECK-SAME:                -> !VPU.DistributedTensor<1x16x20x32xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                        {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:                compute_shapes = [[1, 16, 20, 16], [1, 16, 20, 16]],
    //CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 16]],
    //CHECK-SAME{LITERAL}:                memory_shapes = [[1, 16, 20, 18], [1, 16, 20, 18]],
    //CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 14]]}>

    //CHECK:        [[CONCAT_OUT:%.*]] = VPU.Concat([[CONV0_OUT]], [[CONV1_OUT]])
    //CHECK-SAME{LITERAL}:             {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} :
    //CHECK-SAME:       !VPU.DistributedTensor<1x64x20x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 64, 20, 16], [1, 64, 20, 16]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 16]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 64, 20, 18], [1, 64, 20, 18]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 14]]}>
    //CHECK-SAME:       !VPU.DistributedTensor<1x16x20x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 16, 20, 16], [1, 16, 20, 16]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 16]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 16, 20, 18], [1, 16, 20, 18]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 14]]}>
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x80x20x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 80, 20, 18], [1, 80, 20, 18]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 14]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 80, 20, 18], [1, 80, 20, 18]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 14]]}>

    //CHECK: [[SLICE0:%.*]] = VPU.Slice [[CONCAT_OUT]] [0, 0, 0, 0] [1, 80, 12, 32]
    //CHECK-SAME:           to !VPU.DistributedTensor<1x80x12x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 80, 12, 18], [1, 80, 12, 18]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 14]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 80, 12, 18], [1, 80, 12, 18]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 14]]}>

    //CHECK:        VPU.NCE.Convolution([[SLICE0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x12x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 64, 12, 16], [1, 64, 12, 16]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 16]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 64, 12, 16], [1, 64, 12, 16]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 16]]}>

        //CHECK: [[SLICE1:%.*]] = VPU.Slice [[CONCAT_OUT]] [0, 0, 8, 0] [1, 80, 12, 32]
    //CHECK-SAME:           to !VPU.DistributedTensor<1x80x12x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 80, 12, 18], [1, 80, 12, 18]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 14]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 80, 12, 18], [1, 80, 12, 18]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 14]]}>

    //CHECK:        VPU.NCE.Convolution([[SLICE1]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x12x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 64, 12, 16], [1, 64, 12, 16]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 16]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 64, 12, 16], [1, 64, 12, 16]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 16]]}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ConvIn0 = !VPU.DistributedTensor<
    1x16x8x10xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 16, 4, 10], [1, 16, 4, 10]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]],
    memory_shapes = [[1, 16, 4, 10], [1, 16, 4, 10]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]
}>

!ConvIn1 = !VPU.DistributedTensor<
    1x32x8x10xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 32, 4, 10], [1, 32, 4, 10]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]],
    memory_shapes = [[1, 32, 4, 10], [1, 32, 4, 10]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]
}>

!ConvOut0 = !VPU.DistributedTensor<
    1x16x8x10xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 16, 4, 10], [1, 16, 4, 10]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]],
    memory_shapes = [[1, 16, 5, 10], [1, 16, 5, 10]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]
}>

!ConvOut1 = !VPU.DistributedTensor<
    1x32x8x10xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 32, 4, 10], [1, 32, 4, 10]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]],
    memory_shapes = [[1, 32, 5, 10], [1, 32, 5, 10]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]
}>

!ConvIn23 = !VPU.DistributedTensor<
    1x48x4x10xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 48, 3, 10], [1, 48, 3, 10]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0]],
    memory_shapes = [[1, 48, 3, 10], [1, 48, 3, 10]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0]]
}>

!ConvOut23 = !VPU.DistributedTensor<
    1x48x4x10xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 48, 2, 10], [1, 48, 2, 10]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]],
    memory_shapes = [[1, 48, 2, 10], [1, 48, 2, 10]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]
}>

!ConvInOut0_DDR = tensor<1x16x8x10xf16, {mem_space = @DDR, order = #NHWC}>
!ConvInOut1_DDR = tensor<1x32x8x10xf16, {mem_space = @DDR, order = #NHWC}>
!ConvIn23_DDR = tensor<1x48x4x10xf16, {mem_space = @DDR, order = #NHWC}>

!ConvInOutStub0_CMX = tensor<1x16x8x10xf16, {mem_space = @CMX_NN, order = #NHWC}>
!ConvInOutStub1_CMX = tensor<1x32x8x10xf16, {mem_space = @CMX_NN, order = #NHWC}>
!ConvInStub23_CMX = tensor<1x48x4x10xf16, {mem_space = @CMX_NN, order = #NHWC}>

func.func @SOHProducersSOHConsumersOfConcatWithExplicitDistributedAttrWithHSlice(
        %input0: !ConvInOut0_DDR,
        %input1: !ConvInOut1_DDR,
        %weights0: tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weights1: tensor<32x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weights2: tensor<48x48x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weights3: tensor<48x48x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weightsTable0: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
        %weightsTable1: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
        %weightsTable2: tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
           -> (!ConvOut23, !ConvOut23) {

    %input0CMX = VPU.Copy(%input0) { out_mem_space = @CMX_NN } : !ConvInOut0_DDR -> !ConvIn0

    %conv0Output = VPU.NCE.Convolution(%input0CMX, %weights0, %weightsTable0) {
            pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
            rawFilterShape = [16, 16, 1, 1], strides = [1, 1]}
             -> !ConvOut0

    %conv0OutDDR = VPU.Copy(%conv0Output) : !ConvOut0 -> !ConvInOut0_DDR

    %input1CMX = VPU.Copy(%input1) { out_mem_space = @CMX_NN } : !ConvInOut1_DDR -> !ConvIn1

    %conv1Output = VPU.NCE.Convolution(%input1CMX, %weights1, %weightsTable1) {
            pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
            rawFilterShape = [32, 32, 1, 1], strides = [1, 1]}
             -> !ConvOut1

    %conv1OutDDR = VPU.Copy(%conv1Output) : !ConvOut1 -> !ConvInOut1_DDR

    %concatOut = VPU.Concat(%conv0OutDDR, %conv1OutDDR) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} :
        !ConvInOut0_DDR, !ConvInOut1_DDR -> tensor<1x48x8x10xf16, {mem_space = @DDR, order = #NHWC}>

    %slice0 = VPU.Slice %concatOut [0, 0, 0, 0] [1, 48, 4, 10] : tensor<1x48x8x10xf16, {mem_space = @DDR, order = #NHWC}> to !ConvIn23_DDR

    %input2CMX = VPU.Copy(%slice0) {out_mem_space = @CMX_NN} : !ConvIn23_DDR -> !ConvIn23

    %output0 = VPU.NCE.Convolution(%input2CMX, %weights2, %weightsTable2) {
            pad = #VPU.Padding<left = 1, right = 1, top = 1, bottom = 1>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
            rawFilterShape = [48, 48, 3, 3], strides = [1, 1]}
             -> !ConvOut23

    %slice1 = VPU.Slice %concatOut [0, 0, 0, 0] [1, 48, 4, 10] : tensor<1x48x8x10xf16, {mem_space = @DDR, order = #NHWC}> to !ConvIn23_DDR

    %input3CMX = VPU.Copy(%slice1) {out_mem_space = @CMX_NN} : !ConvIn23_DDR -> !ConvIn23

    %output1 = VPU.NCE.Convolution(%input3CMX, %weights3, %weightsTable2) {
            pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
            rawFilterShape = [48, 48, 1, 1], strides = [1, 1]}
             -> !ConvOut23

    return %output0, %output1: !ConvOut23, !ConvOut23
}

// CHECK:               VPU.Concat
// CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]}
// CHECK-SAME:              : tensor<1x16x8x10xf16, {mem_space = @DDR, order = #NHWC}>, tensor<1x32x8x10xf16, {mem_space = @DDR, order = #NHWC}>
// CHECK-SAME:              -> tensor<1x48x8x10xf16, {mem_space = @DDR, order = #NHWC}>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ConvIn01 = !VPU.DistributedTensor<
    1x16x10x10xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[1, 16, 10, 10], [1, 16, 10, 10]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 16, 10, 10], [1, 16, 10, 10]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConvOut0 = !VPU.DistributedTensor<
    1x32x10x10xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[1, 16, 10, 10], [1, 16, 10, 10]],
    compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]],
    memory_shapes = [[1, 32, 10, 10], [1, 32, 10, 10]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConvOut1 = !VPU.DistributedTensor<
    1x64x10x10xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[1, 32, 10, 10], [1, 32, 10, 10]],
    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]],
    memory_shapes = [[1, 64, 10, 10], [1, 64, 10, 10]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConvIn2 = !VPU.DistributedTensor<
    1x96x5x10xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 96, 3, 10], [1, 96, 2, 10]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]],
    memory_shapes = [[1, 96, 3, 10], [1, 96, 2, 10]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]
}>

!ConvIn3 = !VPU.DistributedTensor<
    1x96x6x10xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 96, 3, 10], [1, 96, 3, 10]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]],
    memory_shapes = [[1, 96, 3, 10], [1, 96, 3, 10]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]
}>

!ConvOut2 = !VPU.DistributedTensor<
    1x96x5x10xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 96, 3, 10], [1, 96, 2, 10]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]],
    memory_shapes = [[1, 96, 3, 10], [1, 96, 2, 10]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]
}>

!ConvOut3 = !VPU.DistributedTensor<
    1x96x6x10xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 96, 3, 10], [1, 96, 3, 10]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]],
    memory_shapes = [[1, 96, 3, 10], [1, 96, 3, 10]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]
}>

!ConvIn01_DDR = tensor<1x16x10x10xf16, {mem_space = @DDR, order = #NHWC}>
!ConvOut0_DDR = tensor<1x32x10x10xf16, {mem_space = @DDR, order = #NHWC}>
!ConvOut1_DDR = tensor<1x64x10x10xf16, {mem_space = @DDR, order = #NHWC}>
!ConvIn2_DDR = tensor<1x96x5x10xf16, {mem_space = @DDR, order = #NHWC}>
!ConvIn3_DDR = tensor<1x96x6x10xf16, {mem_space = @DDR, order = #NHWC}>

!ConvInStub01_CMX = tensor<1x16x10x10xf16, {mem_space = @CMX_NN, order = #NHWC}>
!ConvOutStub0_CMX = tensor<1x32x10x10xf16, {mem_space = @CMX_NN, order = #NHWC}>
!ConvOutStub1_CMX = tensor<1x64x10x10xf16, {mem_space = @CMX_NN, order = #NHWC}>
!ConvInStub2_CMX = tensor<1x96x5x10xf16, {mem_space = @CMX_NN, order = #NHWC}>
!ConvInStub3_CMX = tensor<1x96x6x10xf16, {mem_space = @CMX_NN, order = #NHWC}>

func.func @SOKProducersSOHConsumersOfConcatWithExplicitDistributedAttrWithHSlice(
        %input: !ConvIn01_DDR,
        %weights0: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weights1: tensor<64x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weights2: tensor<96x96x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weightsTable0: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
        %weightsTable1: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
        %weightsTable2: tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
           -> (!ConvOut2, !ConvOut3) {

    %inputCMX = VPU.Copy(%input) { out_mem_space = @CMX_NN } : !ConvIn01_DDR -> !ConvIn01

    %conv0Output = VPU.NCE.Convolution(%inputCMX, %weights0, %weightsTable0) {
            pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
            rawFilterShape = [32, 16, 1, 1], strides = [1, 1]}
             -> !ConvOut0

    %conv0OutDDR = VPU.Copy(%conv0Output) : !ConvOut0 -> !ConvOut0_DDR

    %conv1Output = VPU.NCE.Convolution(%inputCMX, %weights1, %weightsTable1) {
            pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
            rawFilterShape = [64, 16, 1, 1], strides = [1, 1]}
             -> !ConvOut1

    %conv1OutDDR = VPU.Copy(%conv1Output) : !ConvOut1 -> !ConvOut1_DDR

    %concatOut = VPU.Concat(%conv0OutDDR, %conv1OutDDR) {static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]]} :
        !ConvOut0_DDR, !ConvOut1_DDR -> tensor<1x96x10x10xf16, {mem_space = @DDR, order = #NHWC}>

    %slice0 = VPU.Slice %concatOut [0, 0, 0, 0] [1, 96, 5, 10] : tensor<1x96x10x10xf16, {mem_space = @DDR, order = #NHWC}> to !ConvIn2_DDR

    %input2CMX = VPU.Copy(%slice0) {out_mem_space = @CMX_NN} : !ConvIn2_DDR -> !ConvIn2

    %output0 = VPU.NCE.Convolution(%input2CMX, %weights2, %weightsTable2) {
            pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
            rawFilterShape = [96, 96, 1, 1], strides = [1, 1]}
             -> !ConvOut2

    %slice1 = VPU.Slice %concatOut [0, 0, 4, 0] [1, 96, 6, 10] : tensor<1x96x10x10xf16, {mem_space = @DDR, order = #NHWC}> to !ConvIn3_DDR

    %input3CMX = VPU.Copy(%slice1) {out_mem_space = @CMX_NN} : !ConvIn3_DDR -> !ConvIn3

    %output1 = VPU.NCE.Convolution(%input3CMX, %weights2, %weightsTable2) {
            pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
            rawFilterShape = [96, 96, 1, 1], strides = [1, 1]}
             -> !ConvOut3

    return %output0, %output1: !ConvOut2, !ConvOut3

    // CHECK:               VPU.Concat
    // CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]]}
    // CHECK-SAME:              : tensor<1x32x10x10xf16, {mem_space = @DDR, order = #NHWC}>, tensor<1x64x10x10xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK-SAME:              -> tensor<1x96x10x10xf16, {mem_space = @DDR, order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!WeightsType = tensor<
    128x256x3x3xf16,
    {mem_space = @CMX_NN, order = #NHWC}
>

!WeightsTableType = tensor<
    128x1x1x4xsi32,
    {mem_space = @CMX_NN, order = #NCHW}
>

!InDataDistribution = !VPU.DistributedTensor<
    1x256x28x28xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 256, 16, 28], [1, 256, 15, 28]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 13, 0]],
    memory_shapes = [[1, 256, 16, 28], [1, 256, 15, 28]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 13, 0]]
}>

!InSparseMapDistribution = !VPU.DistributedTensor<
    1x256x28x28xi1, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 256, 16, 28], [1, 256, 15, 28]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 13, 0]],
    memory_shapes = [[1, 256, 16, 28], [1, 256, 15, 28]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 13, 0]]
}>

!OutDataDistribution = !VPU.DistributedTensor<
    1x128x14x14xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 7, 14], [1, 128, 7, 14]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]],
    memory_shapes = [[1, 128, 9, 14], [1, 128, 8, 14]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]
}>

!OutSparseMapDistribution = !VPU.DistributedTensor<
    1x128x14x14xi1, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 7, 14], [1, 128, 7, 14]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]],
    memory_shapes = [[1, 128, 9, 14], [1, 128, 8, 14]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]
}>

!SparseDistributedInput = !VPU.SparseTensor<data=!InDataDistribution, sparsity_map=!InSparseMapDistribution>

!SparseDistributedOutput = !VPU.SparseTensor<data=!OutDataDistribution, sparsity_map=!OutSparseMapDistribution>

!SparseConvInput = !VPU.SparseTensor<
    data=tensor<1x256x28x28xf16, {order = #NHWC}>,
    sparsity_map=tensor<1x256x28x28xi1, {order = #NHWC}>
>

!SparseConvOutput = !VPU.SparseTensor<
    data=tensor<1x128x14x14xf16, {order = #NHWC}>,
    sparsity_map=tensor<1x128x14x14xi1, {order = #NHWC}>
>

!SparseConvInputCMX = !VPU.SparseTensor<
    data=tensor<1x256x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    sparsity_map=tensor<1x256x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>
>

!SparseConvOutputCMX = !VPU.SparseTensor<
    data=tensor<1x128x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    sparsity_map=tensor<1x128x14x14xi1, {mem_space = @CMX_NN, order = #NHWC}>
>

!ConcatOutDataDistribution = !VPU.DistributedTensor<
    1x256x14x14xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 256, 9, 14], [1, 256, 8, 14]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]],
    memory_shapes = [[1, 256, 9, 14], [1, 256, 8, 14]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]
}>

!ConcatOutSparseMapDistribution = !VPU.DistributedTensor<
    1x256x14x14xi1, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 256, 9, 14], [1, 256, 8, 14]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]],
    memory_shapes = [[1, 256, 9, 14], [1, 256, 8, 14]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]
}>

!SparseConcatOutputDist = !VPU.SparseTensor<data=!ConcatOutDataDistribution,sparsity_map=!ConcatOutSparseMapDistribution>

!SparseConcatOutput = !VPU.SparseTensor<
    data=tensor<1x256x14x14xf16, {order = #NHWC}>,
    sparsity_map=tensor<1x256x14x14xi1, {order = #NHWC}>
>

!SparseConcatOutputCMX = !VPU.SparseTensor<
    data=tensor<1x256x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    sparsity_map=tensor<1x256x14x14xi1, {mem_space = @CMX_NN, order = #NHWC}>
>

!Conv2OutDataDistribution = !VPU.DistributedTensor<
    1x128x6x6xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 3, 6], [1, 128, 3, 6]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]],
    memory_shapes = [[1, 128, 3, 6], [1, 128, 3, 6]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]
}>

!Conv2OutSparseMapDistribution = !VPU.DistributedTensor<
    1x128x6x6xi1, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 3, 6], [1, 128, 3, 6]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]],
    memory_shapes = [[1, 128, 3, 6], [1, 128, 3, 6]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]
}>

!SparseConv2OutDist = !VPU.SparseTensor<data=!Conv2OutDataDistribution, sparsity_map=!Conv2OutSparseMapDistribution>

!SparseConv2OutCMX = !VPU.SparseTensor<
    data=tensor<1x128x6x6xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    sparsity_map=tensor<1x128x6x6xi1, {mem_space = @CMX_NN, order = #NHWC}>
>

!SparseConv2OutDDR = !VPU.SparseTensor<
    data=tensor<1x128x6x6xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>,
    sparsity_map=tensor<1x128x6x6xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
>

// CHECK:      func.func @SparseConvolution([[INPUT:%.+]]: !VPU.SparseTensor<data=tensor<1x256x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x256x28x28xi1, {order = #NHWC}>>,
// CHECK-SAME:                         [[WEIGHTS_TABLE:%.+]]: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
// CHECK-SAME:                         [[WEIGHTS:%.+]]: tensor<128x256x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>)
// CHECK-SAME:         -> !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {order = #NHWC}>, sparsity_map=tensor<1x128x6x6xi1, {order = #NHWC}>> {
func.func @SparseConvolution(%input: !SparseConvInput,
                             %weightsTable: !WeightsTableType,
                             %weights: !WeightsType)
        -> !SparseConv2OutDDR {
    // Convolution 0 in copy
    %0 = VPU.Copy(%input) {out_mem_space = @CMX_NN} : !SparseConvInput -> !SparseDistributedInput

    // Convolution 0
    %1 = VPU.NCE.Convolution(%0, %weights, %weightsTable) {
            pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -128 : i64, clamp_high = 127 : i64,
                lrelu_mult = 1638 : i64, lrelu_shift = 14 : i64, fp_prelu_alpha = 0.0999755859375 : f64>,
            rawFilterShape = [128, 256, 3, 3], strides = [2, 2]}
                -> !SparseDistributedOutput

    // Convolution 0 output copy
    %2 = VPU.Copy(%1) : !SparseDistributedOutput -> !SparseConvOutput

    // Convolution 1
    %3 = VPU.NCE.Convolution(%0, %weights, %weightsTable) {
            pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -128 : i64, clamp_high = 127 : i64,
                lrelu_mult = 1638 : i64, lrelu_shift = 14 : i64, fp_prelu_alpha = 0.0999755859375 : f64>,
            rawFilterShape = [128, 256, 3, 3], strides = [2, 2]}
                -> !SparseDistributedOutput

    // Convolution 1 output copy
    %4 = VPU.Copy(%3) : !SparseDistributedOutput -> !SparseConvOutput

    // Concat
    %5 = VPU.Concat(%2, %4) {static_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]} : !SparseConvOutput, !SparseConvOutput -> !SparseConcatOutput

    // Concat output copy
    %6 = VPU.Copy(%5) {out_mem_space = @CMX_NN} : !SparseConcatOutput -> !SparseConcatOutputDist

    // Convolution 2
    %7 = VPU.NCE.Convolution(%6, %weights, %weightsTable){
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -128 : i64, clamp_high = 127 : i64,
                lrelu_mult = 1638 : i64, lrelu_shift = 14 : i64, fp_prelu_alpha = 0.0999755859375 : f64>,
            rawFilterShape = [128, 256, 3, 3], strides = [2, 2]}
             -> !SparseConv2OutDist

    %8 = VPU.Copy(%7) : !SparseConv2OutDist -> !SparseConv2OutDDR

    return %8 : !SparseConv2OutDDR

    // Copy DDR -> CMX
    // CHECK:       [[IN_CMX:%.+]] = VPU.Copy([[INPUT]])
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:              data=!VPU.DistributedTensor<1x256x28x28xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:            compute_shapes = [[1, 256, 16, 28], [1, 256, 15, 28]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0], [0, 0, 13, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[1, 256, 16, 28], [1, 256, 15, 28]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0], [0, 0, 13, 0]]}>
    // CHECK-SAME:              sparsity_map=!VPU.DistributedTensor<1x256x28x28xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:            compute_shapes = [[1, 256, 16, 28], [1, 256, 15, 28]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0], [0, 0, 13, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[1, 256, 16, 28], [1, 256, 15, 28]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0], [0, 0, 13, 0]]}>

    // Convolution 0
    // CHECK:       [[CONV0:%.+]] = VPU.NCE.Convolution([[IN_CMX]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:              data=!VPU.DistributedTensor<1x128x14x14xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:            compute_shapes = [[1, 128, 7, 14], [1, 128, 7, 14]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[1, 128, 9, 14], [1, 128, 8, 14]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]}>
    // CHECK-SAME:              sparsity_map=!VPU.DistributedTensor<1x128x14x14xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:            compute_shapes = [[1, 128, 7, 14], [1, 128, 7, 14]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[1, 128, 9, 14], [1, 128, 8, 14]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]}>

    // Convolution 1
    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[IN_CMX]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:              data=!VPU.DistributedTensor<1x128x14x14xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:            compute_shapes = [[1, 128, 7, 14], [1, 128, 7, 14]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[1, 128, 9, 14], [1, 128, 8, 14]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]}>
    // CHECK-SAME:              sparsity_map=!VPU.DistributedTensor<1x128x14x14xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:            compute_shapes = [[1, 128, 7, 14], [1, 128, 7, 14]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[1, 128, 9, 14], [1, 128, 8, 14]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]}>

    // Concat
    // CHECK:       [[CONCAT_CMX:%.+]] = VPU.Concat([[CONV0]], [[CONV1]])
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:              data=!VPU.DistributedTensor<1x256x14x14xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:            compute_shapes = [[1, 256, 9, 14], [1, 256, 8, 14]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[1, 256, 9, 14], [1, 256, 8, 14]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]}>
    // CHECK-SAME:              sparsity_map=!VPU.DistributedTensor<1x256x14x14xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:            compute_shapes = [[1, 256, 9, 14], [1, 256, 8, 14]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[1, 256, 9, 14], [1, 256, 8, 14]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]}>

    // Convolution 2
    // CHECK:       [[CONV2:%.+]] = VPU.NCE.Convolution([[CONCAT_CMX]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:              data=!VPU.DistributedTensor<1x128x6x6xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:            compute_shapes = [[1, 128, 3, 6], [1, 128, 3, 6]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[1, 128, 3, 6], [1, 128, 3, 6]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]
    // CHECK-SAME:              sparsity_map=!VPU.DistributedTensor<1x128x6x6xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:            compute_shapes = [[1, 128, 3, 6], [1, 128, 3, 6]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[1, 128, 3, 6], [1, 128, 3, 6]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ConvIn0 = !VPU.DistributedTensor<
    1x64x20x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[1, 64, 20, 32], [1, 64, 20, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 64, 20, 32], [1, 64, 20, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConvIn1 = !VPU.DistributedTensor<
    1x32x20x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConvOut0 = !VPU.DistributedTensor<
    1x64x20x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]],
    memory_shapes = [[1, 64, 20, 32], [1, 64, 20, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConvOut1 = !VPU.DistributedTensor<
    1x32x20x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[1, 16, 20, 32], [1, 16, 20, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]],
    memory_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConvIn23 = !VPU.DistributedTensor<
    1x96x12x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[1, 96, 12, 32], [1, 96, 12, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 96, 12, 32], [1, 96, 12, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConvOut23 = !VPU.DistributedTensor<
    1x64x12x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[1, 32, 12, 32], [1, 32, 12, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]],
    memory_shapes = [[1, 64, 12, 32], [1, 64, 12, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConvInOut0_DDR = tensor<1x64x20x32xf16, {mem_space = @DDR, order = #NHWC}>
!ConvInOut1_DDR = tensor<1x32x20x32xf16, {mem_space = @DDR, order = #NHWC}>
!ConvIn23_DDR = tensor<1x96x12x32xf16, {mem_space = @DDR, order = #NHWC}>

!ConvInOutStub0_CMX = tensor<1x64x20x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
!ConvInOutStub1_CMX = tensor<1x32x20x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
!ConvInStub23_CMX = tensor<1x96x12x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @SOKProducersSOKConsumersOfConcatWithExplicitDistributedAttrWithHSlice
// CHECK-SAME: ([[FUNC_IN0:%.+]]: tensor<1x64x20x32xf16, {mem_space = @DDR, order = #NHWC}>
// CHECK-SAME:  [[FUNC_IN1:%.+]]: tensor<1x32x20x32xf16, {mem_space = @DDR, order = #NHWC}>
// CHECK-SAME:  [[W0:%.+]]: tensor<64x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:  [[W1:%.+]]: tensor<32x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:  [[W2:%.+]]: tensor<64x96x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:  [[W3:%.+]]: tensor<64x96x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:  [[WT0:%.+]]: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
// CHECK-SAME:  [[WT1:%.+]]: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
func.func @SOKProducersSOKConsumersOfConcatWithExplicitDistributedAttrWithHSlice(
        %input0: !ConvInOut0_DDR,
        %input1: !ConvInOut1_DDR,
        %weights0: tensor<64x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weights1: tensor<32x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weights2: tensor<64x96x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weights3: tensor<64x96x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weightsTable0: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
        %weightsTable1: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
           -> (!ConvOut23, !ConvOut23) {

    %input0CMX = VPU.Copy(%input0) { out_mem_space = @CMX_NN } : !ConvInOut0_DDR -> !ConvIn0

    %conv0Output = VPU.NCE.Convolution(%input0CMX, %weights0, %weightsTable0) {
            pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
            rawFilterShape = [64, 64, 1, 1], strides = [1, 1]}
             -> !ConvOut0

    %conv0OutDDR = VPU.Copy(%conv0Output) : !ConvOut0 -> !ConvInOut0_DDR

    %input1CMX = VPU.Copy(%input1) { out_mem_space = @CMX_NN } : !ConvInOut1_DDR -> !ConvIn1

    %conv1Output = VPU.NCE.Convolution(%input1CMX, %weights1, %weightsTable1) {
            pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
            rawFilterShape = [32, 32, 1, 1], strides = [1, 1]}
             -> !ConvOut1

    %conv1OutDDR = VPU.Copy(%conv1Output) : !ConvOut1 -> !ConvInOut1_DDR

    %concatOut = VPU.Concat(%conv0OutDDR, %conv1OutDDR) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} :
        !ConvInOut0_DDR, !ConvInOut1_DDR -> tensor<1x96x20x32xf16, {mem_space = @DDR, order = #NHWC}>

    %slice0 = VPU.Slice %concatOut [0, 0, 0, 0] [1, 96, 12, 32] : tensor<1x96x20x32xf16, {mem_space = @DDR, order = #NHWC}> to !ConvIn23_DDR

    %input2CMX = VPU.Copy(%slice0) {out_mem_space = @CMX_NN} : !ConvIn23_DDR -> !ConvIn23

    %output0 = VPU.NCE.Convolution(%input2CMX, %weights2, %weightsTable0) {
            pad = #VPU.Padding<left = 1, right = 1, top = 1, bottom = 1>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
            rawFilterShape = [64, 96, 3, 3], strides = [1, 1]}
             -> !ConvOut23

    %slice1 = VPU.Slice %concatOut [0, 0, 8, 0] [1, 96, 12, 32] : tensor<1x96x20x32xf16, {mem_space = @DDR, order = #NHWC}> to !ConvIn23_DDR

    %input3CMX = VPU.Copy(%slice1) {out_mem_space = @CMX_NN} : !ConvIn23_DDR -> !ConvIn23

    %output1 = VPU.NCE.Convolution(%input3CMX, %weights3, %weightsTable0) {
            pad = #VPU.Padding<left = 2, right = 2, top = 2, bottom = 2>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
            rawFilterShape = [64, 96, 5, 5], strides = [1, 1]}
             -> !ConvOut23

    return %output0, %output1: !ConvOut23, !ConvOut23

    //CHECK:        [[IN0_CMX:%.*]] =  VPU.Copy([[FUNC_IN0]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x20x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "DUPLICATED", num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 64, 20, 32], [1, 64, 20, 32]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 64, 20, 32], [1, 64, 20, 32]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[CONV0_OUT:%.*]] = VPU.NCE.Convolution([[IN0_CMX]], [[W0]], [[WT0]])
    //CHECK-SAME:                -> !VPU.DistributedTensor<1x64x20x32xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                        {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    //CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]],
    //CHECK-SAME{LITERAL}:                memory_shapes = [[1, 64, 20, 32], [1, 64, 20, 32]],
    //CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[IN1_CMX:%.*]] = VPU.Copy([[FUNC_IN1]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x20x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "DUPLICATED", num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[CONV1_OUT:%.*]] = VPU.NCE.Convolution([[IN1_CMX]], [[W1]], [[WT1]])
    //CHECK-SAME:                -> !VPU.DistributedTensor<1x32x20x32xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                        {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:                compute_shapes = [[1, 16, 20, 32], [1, 16, 20, 32]],
    //CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]],
    //CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    //CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[D_CAST0:%.*]] = VPU.DistributedCast([[CONV0_OUT]] :
    //CHECK-SAME:          !VPU.DistributedTensor<1x64x20x32xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:               {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 64, 20, 32], [1, 64, 20, 32]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
    //CHECK-SAME:          -> !VPU.DistributedTensor<1x64x20x32xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                 {mode = "DUPLICATED", num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:         compute_shapes = [[1, 64, 20, 32], [1, 64, 20, 32]],
    //CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:         memory_shapes = [[1, 64, 20, 32], [1, 64, 20, 32]],
    //CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[D_CAST1:%.*]] = VPU.DistributedCast([[CONV1_OUT]] :
    //CHECK-SAME:          !VPU.DistributedTensor<1x32x20x32xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:               {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 16, 20, 32], [1, 16, 20, 32]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
    //CHECK-SAME:          -> !VPU.DistributedTensor<1x32x20x32xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                 {mode = "DUPLICATED", num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:         compute_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    //CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:         memory_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    //CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[CONCAT_OUT:%.*]] = VPU.Concat([[D_CAST0]], [[D_CAST1]])
    //CHECK-SAME{LITERAL}:             {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} :
    //CHECK-SAME:       !VPU.DistributedTensor<1x64x20x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "DUPLICATED", num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 64, 20, 32], [1, 64, 20, 32]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 64, 20, 32], [1, 64, 20, 32]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
    //CHECK-SAME:       !VPU.DistributedTensor<1x32x20x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "DUPLICATED", num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x20x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "DUPLICATED", num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 96, 20, 32], [1, 96, 20, 32]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 96, 20, 32], [1, 96, 20, 32]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK: [[SLICE0:%.*]] = VPU.Slice [[CONCAT_OUT]] [0, 0, 0, 0] [1, 96, 12, 32]
    //CHECK-SAME:           to !VPU.DistributedTensor<1x96x12x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "DUPLICATED", num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 96, 12, 32], [1, 96, 12, 32]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 96, 12, 32], [1, 96, 12, 32]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        VPU.NCE.Convolution([[SLICE0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x12x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 32, 12, 32], [1, 32, 12, 32]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 64, 12, 32], [1, 64, 12, 32]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK: [[SLICE1:%.*]] = VPU.Slice [[CONCAT_OUT]] [0, 0, 8, 0] [1, 96, 12, 32]
    //CHECK-SAME:           to !VPU.DistributedTensor<1x96x12x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "DUPLICATED", num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 96, 12, 32], [1, 96, 12, 32]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 96, 12, 32], [1, 96, 12, 32]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        VPU.NCE.Convolution([[SLICE1]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x12x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 32, 12, 32], [1, 32, 12, 32]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 64, 12, 32], [1, 64, 12, 32]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ConvIn0 = !VPU.DistributedTensor<
    1x64x20x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[1, 64, 20, 32], [1, 64, 20, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 64, 20, 32], [1, 64, 20, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConvIn1 = !VPU.DistributedTensor<
    1x32x20x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConvOut0 = !VPU.DistributedTensor<
    1x64x20x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]],
    memory_shapes = [[1, 64, 20, 32], [1, 64, 20, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConvOut1 = !VPU.DistributedTensor<
    1x32x20x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[1, 16, 20, 32], [1, 16, 20, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]],
    memory_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConvIn2 = !VPU.DistributedTensor<
    1x96x20x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[1, 96, 20, 32], [1, 96, 20, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 96, 20, 32], [1, 96, 20, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConvOut23 = !VPU.DistributedTensor<
    1x64x20x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]],
    memory_shapes = [[1, 64, 20, 32], [1, 64, 20, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!InterpIn = !VPU.DistributedTensor<
    1x16x10x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[1, 16, 10, 16], [1, 16, 10, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 16, 10, 16], [1, 16, 10, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!InterpOut = !VPU.DistributedTensor<
    1x16x20x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[1, 16, 20, 32], [1, 16, 20, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 16, 20, 32], [1, 16, 20, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConvIn3 = !VPU.DistributedTensor<
    1x80x20x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[1, 80, 20, 32], [1, 80, 20, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 80, 20, 32], [1, 80, 20, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConvInOut0_DDR = tensor<1x64x20x32xf16, {mem_space = @DDR, order = #NHWC}>
!ConvInOut1_DDR = tensor<1x32x20x32xf16, {mem_space = @DDR, order = #NHWC}>
!InterpIn_DDR = tensor<1x16x10x16xf16, {mem_space = @DDR, order = #NHWC}>
!InterpOut_DDR = tensor<1x16x20x32xf16, {mem_space = @DDR, order = #NHWC}>
!ConvIn2_DDR = tensor<1x96x20x32xf16, {mem_space = @DDR, order = #NHWC}>
!ConvIn3_DDR = tensor<1x80x20x32xf16, {mem_space = @DDR, order = #NHWC}>

!ConvInOutStub0_CMX = tensor<1x64x20x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
!ConvInOutStub1_CMX = tensor<1x32x20x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
!ConvInStub2_CMX = tensor<1x96x20x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
!ConvInStub3_CMX = tensor<1x80x20x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
!InterpInStub_CMX = tensor<1x16x10x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
!InterpOutStub_CMX = tensor<1x16x20x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @ComplexConcatWithExplicitDistribution
// CHECK-SAME: ([[FUNC_IN0:%.+]]: tensor<1x64x20x32xf16, {mem_space = @DDR, order = #NHWC}>
// CHECK-SAME:  [[FUNC_IN1:%.+]]: tensor<1x32x20x32xf16, {mem_space = @DDR, order = #NHWC}>
// CHECK-SAME:  [[FUNC_IN2:%.+]]: tensor<1x16x10x16xf16, {mem_space = @DDR, order = #NHWC}>
// CHECK-SAME:  [[W0:%.+]]: tensor<64x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:  [[W1:%.+]]: tensor<32x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:  [[W2:%.+]]: tensor<64x96x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:  [[W3:%.+]]: tensor<64x80x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:  [[WT0:%.+]]: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
// CHECK-SAME:  [[WT1:%.+]]: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
func.func @ComplexConcatWithExplicitDistribution(
        %input0: !ConvInOut0_DDR,
        %input1: !ConvInOut1_DDR,
        %input2: !InterpIn_DDR,
        %weights0: tensor<64x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weights1: tensor<32x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weights2: tensor<64x96x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weights3: tensor<64x80x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weightsTable0: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
        %weightsTable1: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
           -> (!ConvOut23, !ConvOut23) {

    %input0CMX = VPU.Copy(%input0) { out_mem_space = @CMX_NN } : !ConvInOut0_DDR -> !ConvIn0

    %conv0Output = VPU.NCE.Convolution(%input0CMX, %weights0, %weightsTable0) {
            pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
            rawFilterShape = [64, 64, 1, 1], strides = [1, 1]}
             -> !ConvOut0

    %conv0OutDDR = VPU.Copy(%conv0Output) : !ConvOut0 -> !ConvInOut0_DDR

    %input1CMX = VPU.Copy(%input1) { out_mem_space = @CMX_NN } : !ConvInOut1_DDR -> !ConvIn1

    %conv1Output = VPU.NCE.Convolution(%input1CMX, %weights1, %weightsTable1) {
            pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
            rawFilterShape = [32, 32, 1, 1], strides = [1, 1]}
             -> !ConvOut1

    %conv1OutDDR = VPU.Copy(%conv1Output) : !ConvOut1 -> !ConvInOut1_DDR

    %concatOut0 = VPU.Concat(%conv0OutDDR, %conv1OutDDR) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} :
        !ConvInOut0_DDR, !ConvInOut1_DDR -> !ConvIn2_DDR


    %input2CMX = VPU.Copy(%concatOut0) {out_mem_space = @CMX_NN} : !ConvIn2_DDR -> !ConvIn2

    %output0 = VPU.NCE.Convolution(%input2CMX, %weights2, %weightsTable0) {
            pad = #VPU.Padding<left = 1, right = 1, top = 1, bottom = 1>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
            rawFilterShape = [64, 96, 3, 3], strides = [1, 1]}
             -> !ConvOut23

    %interpInCMX = VPU.Copy(%input2) { out_mem_space = @CMX_NN } : !InterpIn_DDR -> !InterpIn

    %outInterp = VPU.Interpolate(%interpInCMX) {
            attr = #IE.Interpolate<mode = <NEAREST>,
            shape_calc_mode = <SIZES>,
            coord_mode = <TF_HALF_PIXEL_FOR_NN>,
            nearest_mode = <FLOOR>, antialias = false,
            pads_begin = [0, 0, 0, 0],
            pads_end = [0, 0, 0, 0],
            cube_coeff = -7.500000e-01 : f64>,
            axes_attr = [2, 3],
            initial_input_dims_attr = [1, 16, 10, 16],
            initial_output_dims_attr = [1, 16, 20, 32],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            scales_attr = [2.000000e+00, 2.000000e+00],
            sizes_attr = [20, 32],
            tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]}
             : !InterpIn
            -> !InterpOut

    %interpOutDDR = VPU.Copy(%outInterp) : !InterpOut -> !InterpOut_DDR

    %concatOut1 = VPU.Concat(%conv0OutDDR, %interpOutDDR) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} :
        !ConvInOut0_DDR, !InterpOut_DDR -> !ConvIn3_DDR

    %input3CMX = VPU.Copy(%concatOut1) {out_mem_space = @CMX_NN} : !ConvIn3_DDR -> !ConvIn3

    %output1 = VPU.NCE.Convolution(%input3CMX, %weights3, %weightsTable0) {
            pad = #VPU.Padding<left = 1, right = 1, top = 1, bottom = 1>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
            rawFilterShape = [64, 80, 3, 3], strides = [1, 1]}
             -> !ConvOut23


    return %output0, %output1: !ConvOut23, !ConvOut23

    //CHECK:        [[IN0_CMX:%.*]] =  VPU.Copy([[FUNC_IN0]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x20x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "DUPLICATED", num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 64, 20, 32], [1, 64, 20, 32]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 64, 20, 32], [1, 64, 20, 32]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[CONV0_OUT:%.*]] = VPU.NCE.Convolution([[IN0_CMX]], [[W0]], [[WT0]])
    //CHECK-SAME:                -> !VPU.DistributedTensor<1x64x20x32xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                        {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    //CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]],
    //CHECK-SAME{LITERAL}:                memory_shapes = [[1, 64, 20, 32], [1, 64, 20, 32]],
    //CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[CONV0_DDR:%.*]] =  VPU.Copy([[CONV0_OUT]])
    //CHECK-SAME:       -> tensor<1x64x20x32xf16, {mem_space = @DDR, order = #NHWC}>

    //CHECK:        [[IN1_CMX:%.*]] = VPU.Copy([[FUNC_IN1]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x20x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "DUPLICATED", num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[CONV1_OUT:%.*]] = VPU.NCE.Convolution([[IN1_CMX]], [[W1]], [[WT1]])
    //CHECK-SAME:                -> !VPU.DistributedTensor<1x32x20x32xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                        {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:                compute_shapes = [[1, 16, 20, 32], [1, 16, 20, 32]],
    //CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]],
    //CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    //CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[D_CAST0:%.*]] = VPU.DistributedCast([[CONV0_OUT]] :
    //CHECK-SAME:          !VPU.DistributedTensor<1x64x20x32xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:               {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 64, 20, 32], [1, 64, 20, 32]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
    //CHECK-SAME:          -> !VPU.DistributedTensor<1x64x20x32xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                 {mode = "DUPLICATED", num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:         compute_shapes = [[1, 64, 20, 32], [1, 64, 20, 32]],
    //CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:         memory_shapes = [[1, 64, 20, 32], [1, 64, 20, 32]],
    //CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[D_CAST1:%.*]] = VPU.DistributedCast([[CONV1_OUT]] :
    //CHECK-SAME:          !VPU.DistributedTensor<1x32x20x32xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:               {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 16, 20, 32], [1, 16, 20, 32]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
    //CHECK-SAME:          -> !VPU.DistributedTensor<1x32x20x32xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                 {mode = "DUPLICATED", num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:         compute_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    //CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:         memory_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    //CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[CONCAT_OUT:%.*]] = VPU.Concat([[D_CAST0]], [[D_CAST1]])
    //CHECK-SAME{LITERAL}:             {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} :
    //CHECK-SAME:       !VPU.DistributedTensor<1x64x20x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "DUPLICATED", num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 64, 20, 32], [1, 64, 20, 32]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 64, 20, 32], [1, 64, 20, 32]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
    //CHECK-SAME:       !VPU.DistributedTensor<1x32x20x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "DUPLICATED", num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x20x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "DUPLICATED", num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 96, 20, 32], [1, 96, 20, 32]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 96, 20, 32], [1, 96, 20, 32]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        VPU.NCE.Convolution([[CONCAT_OUT]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x20x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 32, 20, 32], [1, 32, 20, 32]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 64, 20, 32], [1, 64, 20, 32]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[IN2_CMX:%.*]] = VPU.Copy([[FUNC_IN2]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x10x16xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "DUPLICATED", num_clusters = 2 : i64

    //CHECK:        [[OUT_INTERP:%.*]] = VPU.Interpolate([[IN2_CMX]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x20x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:               {mode = "DUPLICATED", num_clusters = 2 : i64

    //CHECK:        [[INTERP_DDR:%.*]] =  VPU.Copy([[OUT_INTERP]])
    //CHECK-SAME:       -> tensor<1x16x20x32xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:       VPU.Concat([[CONV0_DDR]], [[INTERP_DDR]])
    // CHECK-SAME:      -> tensor<1x80x20x32xf16, {mem_space = @DDR, order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Distributed = !VPU.DistributedTensor<
    1x16x90x160xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 23, 160], [1, 16, 23, 160], [1, 16, 22, 160], [1, 16, 22, 160]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 46, 0], [0, 0, 68, 0]],
    memory_shapes = [[1, 16, 24, 160], [1, 16, 25, 160], [1, 16, 24, 160], [1, 16, 23, 160]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 45, 0], [0, 0, 67, 0]]
}>

!Distributed1 = !VPU.DistributedTensor<
    1x16x90x160xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 23, 160], [1, 16, 23, 160], [1, 16, 22, 160], [1, 16, 22, 160]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 46, 0], [0, 0, 68, 0]],
    memory_shapes = [[1, 16, 24, 160], [1, 16, 25, 160], [1, 16, 24, 160], [1, 16, 23, 160]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 45, 0], [0, 0, 67, 0]]
}>

!Distributed2 = !VPU.DistributedTensor<
    1x32x90x160xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 23, 160], [1, 32, 23, 160], [1, 32, 22, 160], [1, 32, 22, 160]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 46, 0], [0, 0, 68, 0]],
    memory_shapes = [[1, 32, 24, 160], [1, 32, 25, 160], [1, 32, 24, 160], [1, 32, 23, 160]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 45, 0], [0, 0, 67, 0]]
}>

// CHECK-LABEL: @SkipCMXConcatForNCEPermute
// CHECK-SAME:  ([[INPUT0:%.+]]: tensor<1x16x90x160xf16, {order = #NHWC}>,
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<1x16x90x160xf16, {order = #NCHW}>)
func.func @SkipCMXConcatForNCEPermute(%arg0: tensor<1x16x90x160xf16, {order = #NHWC}>,
           %arg1: tensor<1x16x90x160xf16, {order = #NCHW}>)
           -> tensor<1x32x90x160xf16, {order = #NHWC}> {
    %maxPoolWeightsTable = const.Declare tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> =
        dense<1> : tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    %maxPoolWeightsTable1 = const.Declare tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> =
        dense<1> : tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    // Input 1 of Concat
    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x90x160xf16, {order = #NHWC}> -> !Distributed1

    %1 = VPU.NCE.MaxPool(%0, %maxPoolWeightsTable) {
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
                strides = [1, 1],
                kernel_size = [1, 1]
            } -> !Distributed1

    %2 = VPU.Copy(%1) : !Distributed1 -> tensor<1x16x90x160xf16, {order = #NHWC}>

    // Input 2 of Concat
    %3 = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} :  tensor<1x16x90x160xf16, {order = #NCHW}> -> !Distributed

    %4 = VPU.NCE.Permute(%3) {
            dstElemType = f16,
            dstOrder = #NHWC,
            expandedChannels = 16 : i64,
            ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -128 : i64, clamp_high = 127 : i64,
                lrelu_mult = 1638 : i64, lrelu_shift = 14 : i64, fp_prelu_alpha = 0.0999755859375 : f64>
        } -> !Distributed1

    %5 = VPU.Copy(%4) : !Distributed1 -> tensor<1x16x90x160xf16, {order = #NHWC}>

    %6 = VPU.Concat(%2, %5) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x90x160xf16, {order = #NHWC}>, tensor<1x16x90x160xf16, {order = #NHWC}> -> tensor<1x32x90x160xf16, {order = #NHWC}>

    // Concat output
    %7 = VPU.Copy(%6) {out_mem_space = @CMX_NN} : tensor<1x32x90x160xf16, {order = #NHWC}> -> !Distributed2

    %8 = VPU.NCE.MaxPool(%7, %maxPoolWeightsTable1) {
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
                strides = [1, 1],
                kernel_size = [1, 1]
            } -> !Distributed2

    %9 = VPU.Copy(%8) : !Distributed2 -> tensor<1x32x90x160xf16, {order = #NHWC}>

    return %9 : tensor<1x32x90x160xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = dense<1> : tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:       [[CST_1:%.+]] = const.Declare tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = dense<1> : tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    // CHECK:       [[COPY_IN_0:%.+]] = VPU.Copy([[INPUT0]])
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          1x16x90x160xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 16, 23, 160], [1, 16, 23, 160], [1, 16, 22, 160], [1, 16, 22, 160]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 46, 0], [0, 0, 68, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 16, 24, 160], [1, 16, 25, 160], [1, 16, 24, 160], [1, 16, 23, 160]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 45, 0], [0, 0, 67, 0]]}>

    // CHECK:       [[MAXPOOL_0:%.+]] = VPU.NCE.MaxPool([[COPY_IN_0]], [[CST]] ) {
    // CHECK-SAME:                  kernel_size = [1, 1],
    // CHECK-SAME:                  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                  strides = [1, 1]}
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          1x16x90x160xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 16, 23, 160], [1, 16, 23, 160], [1, 16, 22, 160], [1, 16, 22, 160]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 46, 0], [0, 0, 68, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 16, 24, 160], [1, 16, 25, 160], [1, 16, 24, 160], [1, 16, 23, 160]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 45, 0], [0, 0, 67, 0]]}>

    // CHECK:       [[COPY_OUT_0:%.+]] = VPU.Copy([[MAXPOOL_0]])
    // CHECK-SAME:                       -> tensor<1x16x90x160xf16, {order = #NHWC}>

    // CHECK:       [[COPY_IN_1:%.+]] = VPU.Copy([[INPUT1]])
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          1x16x90x160xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 16, 23, 160], [1, 16, 23, 160], [1, 16, 22, 160], [1, 16, 22, 160]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 46, 0], [0, 0, 68, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 16, 24, 160], [1, 16, 25, 160], [1, 16, 24, 160], [1, 16, 23, 160]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 45, 0], [0, 0, 67, 0]]}>

    // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[COPY_IN_1]]) {
    // CHECK-SAME:                  dstElemType = f16,
    // CHECK-SAME:                  dstOrder = #NHWC,
    // CHECK-SAME:                  expandedChannels = 16 : i64,
    // CHECK-SAME:                  ppe = #VPU.PPEInt<mode = <LPRELU>,
    // CHECK-SAME:                    clamp_low = -128 : i64, clamp_high = 127 : i64,
    // CHECK-SAME:                    lrelu_mult = 1638 : i64, lrelu_shift = 14 : i64,
    // CHECK-SAME:                    fp_prelu_alpha = 0.0999755859375 : f64>}
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          1x16x90x160xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 16, 23, 160], [1, 16, 23, 160], [1, 16, 22, 160], [1, 16, 22, 160]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 46, 0], [0, 0, 68, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 16, 24, 160], [1, 16, 25, 160], [1, 16, 24, 160], [1, 16, 23, 160]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 45, 0], [0, 0, 67, 0]]}>

    // CHECK:       [[COPY_OUT_1:%.+]] = VPU.Copy([[NCE_PERMUTE]])
    // CHECK-SAME:                       -> tensor<1x16x90x160xf16, {order = #NHWC}>

    // CHECK:       [[CMX_CONCAT:%.+]] = VPU.Concat([[COPY_OUT_0]], [[COPY_OUT_1]]) {
    // CHECK-SAME:          static_offsets = [
    // CHECK-SAME:              [0, 0, 0, 0],
    // CHECK-SAME:              [0, 16, 0, 0]
    // CHECK-SAME:          ]
    // CHECK-SAME:  } : tensor<1x16x90x160xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x16x90x160xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x32x90x160xf16, {order = #NHWC}>

    // CHECK:       [[COPY_IN_2:%.+]] = VPU.Copy([[CMX_CONCAT]])
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          1x32x90x160xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 32, 23, 160], [1, 32, 23, 160], [1, 32, 22, 160], [1, 32, 22, 160]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 46, 0], [0, 0, 68, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 32, 24, 160], [1, 32, 25, 160], [1, 32, 24, 160], [1, 32, 23, 160]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 45, 0], [0, 0, 67, 0]]}>

    // CHECK:       [[MAXPOOL_1:%.+]] = VPU.NCE.MaxPool([[COPY_IN_2]], [[CST_1]] )
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          1x32x90x160xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 32, 23, 160], [1, 32, 23, 160], [1, 32, 22, 160], [1, 32, 22, 160]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 46, 0], [0, 0, 68, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 32, 24, 160], [1, 32, 25, 160], [1, 32, 24, 160], [1, 32, 23, 160]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 45, 0], [0, 0, 67, 0]]}>

    // CHECK:       [[COPY_OUT_2:%.+]] = VPU.Copy([[MAXPOOL_1]])
    // CHECK-SAME:                       -> tensor<1x32x90x160xf16, {order = #NHWC}>

    // CHECK:       return [[COPY_OUT_2]] : tensor<1x32x90x160xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Distributed = !VPU.DistributedTensor<
    1x64x36x36xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 6, 1],
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]],
    memory_shapes = [[1, 64, 7, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 7, 36]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]]
}>

!Distributed2 = !VPU.DistributedTensor<
    64x64x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!Distributed3 = !VPU.DistributedTensor<
    64x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!Distributed4 = !VPU.DistributedTensor<
    64x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 6 : i64,
    uniform_distributed_segments, compute_shapes = [[64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!Distributed5 = !VPU.DistributedTensor<
    1x128x36x36xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 6, 1],
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 128, 6, 36], [1, 128, 6, 36], [1, 128, 6, 36], [1, 128, 6, 36], [1, 128, 6, 36], [1, 128, 6, 36]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]],
    memory_shapes = [[1, 128, 7, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 7, 36]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]]
}>

// CHECK-LABEL: @InsertAvgPoolingWhenNCEOpHasExtraUser
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x64x36x36xf16, {order = #NHWC}>
func.func @InsertAvgPoolingWhenNCEOpHasExtraUser(%arg0: tensor<1x64x36x36xf16, {order = #NHWC}>)
           -> tensor<1x128x36x36xf16, {order = #NHWC}> {
    %convWeights = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %convWeightsTable = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %dwConvWeights = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %dwConvWeightsTable = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %maxPoolWeightsTable = const.Declare tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> =
        dense<1> : tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    // Input 1 of Concat
    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x64x36x36xf16, {order = #NHWC}> -> !Distributed
    %1 = VPU.Copy(%convWeights) {out_mem_space = @CMX_NN} : tensor<64x64x1x1xf16, {order = #NHWC}> -> !Distributed2
    %2 = VPU.Copy(%convWeightsTable) {out_mem_space = @CMX_NN} : tensor<64x1x1x4xsi32> -> !Distributed3

    %3 = VPU.NCE.Convolution(%0, %1, %2) {
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                rawFilterShape = [64, 64, 1, 1],
                strides = [1, 1]
            } -> !Distributed

    %4 = VPU.Copy(%3) : !Distributed -> tensor<1x64x36x36xf16, {order = #NHWC}>
    // Input 2 of Concat
    %5 = VPU.Copy(%dwConvWeights) {out_mem_space = @CMX_NN} : tensor<64x16x1x1xf16, {order = #NHWC}> -> !Distributed4
    %6 = VPU.Copy(%dwConvWeightsTable) {out_mem_space = @CMX_NN} : tensor<64x1x1x4xsi32> -> !Distributed3

    %7 = VPU.NCE.DepthConvolution(%3, %5, %6) {
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -128 : i64, clamp_high = 127 : i64,
                lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                rawFilterShape = [64, 1, 3, 3],
                strides = [1, 1]
            } -> !Distributed

    %8 = VPU.Copy(%7) : !Distributed -> tensor<1x64x36x36xf16, {order = #NHWC}>

    %9 = VPU.Concat(%4, %8) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x36x36xf16, {order = #NHWC}>, tensor<1x64x36x36xf16, {order = #NHWC}> -> tensor<1x128x36x36xf16, {order = #NHWC}>

    // Concat output
    %10 = VPU.Copy(%9) {out_mem_space = @CMX_NN} : tensor<1x128x36x36xf16, {order = #NHWC}> -> !Distributed5

    %11 = VPU.NCE.MaxPool(%10, %maxPoolWeightsTable) {
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
                strides = [1, 1],
                kernel_size = [1, 1]
            } -> !Distributed5

    %12 = VPU.Copy(%11) : !Distributed5 -> tensor<1x128x36x36xf16, {order = #NHWC}>

    return %12 : tensor<1x128x36x36xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[CST_0:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    // CHECK:       [[CST_1:%.+]] = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK:       [[CST_2:%.+]] = const.Declare tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = dense<1> : tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    // CHECK:       [[COPY_IN_0:%.+]] = VPU.Copy([[INPUT]])
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          1x64x36x36xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 64, 7, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 7, 36]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]]}>

    // CHECK:       [[COPY_IN_1:%.+]] = VPU.Copy([[CST]])
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          64x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[COPY_IN_2:%.+]] = VPU.Copy([[CST_0]])
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          64x1x1x4xsi32, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[COPY_IN_0]], [[COPY_IN_1]], [[COPY_IN_2]]) {
    // CHECK-SAME:                  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                  ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
    // CHECK-SAME:                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:                  rawFilterShape = [64, 64, 1, 1],
    // CHECK-SAME:                  strides = [1, 1]}
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          1x64x36x36xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 64, 7, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 7, 36]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]]}>

    // CHECK:       [[AVGPOOL:%.+]] = VPU.NCE.AveragePool([[CONV]]) {
    // CHECK-SAME:                  kernel_size = [1, 1],
    // CHECK-SAME:                  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                  ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
    // CHECK-SAME:                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:                  strides = [1, 1]}
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          1x64x36x36xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 64, 7, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 7, 36]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]]}>

    // CHECK:       [[COPY_IN_3:%.+]] = VPU.Copy([[CST_1]])
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          64x16x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[COPY_IN_4:%.+]] = VPU.Copy([[CST_0]])
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          64x1x1x4xsi32, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[DW_CONV:%.+]] = VPU.NCE.DepthConvolution([[CONV]], [[COPY_IN_3]], [[COPY_IN_4]]) {
    // CHECK-SAME:                          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:                          ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -128 : i64, clamp_high = 127 : i64,
    // CHECK-SAME:                              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:                          rawFilterShape = [64, 1, 3, 3],
    // CHECK-SAME:                          strides = [1, 1]}
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          1x64x36x36xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 64, 7, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 7, 36]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]]}>

    // CHECK:       [[CMX_CONCAT:%.+]] = VPU.Concat([[AVGPOOL]], [[DW_CONV]]) {
    // CHECK-SAME:          static_offsets = [
    // CHECK-SAME:              [0, 0, 0, 0],
    // CHECK-SAME:              [0, 64, 0, 0]
    // CHECK-SAME:          ]} :
    // CHECK-SAME:              !VPU.DistributedTensor<
    // CHECK-SAME:                          1x64x36x36xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 64, 7, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 7, 36]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]]}>,
    // CHECK-SAME:              !VPU.DistributedTensor<
    // CHECK-SAME:                          1x64x36x36xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 64, 7, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 7, 36]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]]}>
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          1x128x36x36xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                          compute_shapes = [[1, 128, 7, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 7, 36]],
    // CHECK-SAME{LITERAL}:                          compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]],
    // CHECK-SAME{LITERAL}:                          memory_shapes = [[1, 128, 7, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 7, 36]],
    // CHECK-SAME{LITERAL}:                          memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]]}>

    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[CMX_CONCAT]], [[CST_2]] ) {
    // CHECK-SAME:                          kernel_size = [1, 1],
    // CHECK-SAME:                          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                          strides = [1, 1]}
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          1x128x36x36xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 128, 6, 36], [1, 128, 6, 36], [1, 128, 6, 36], [1, 128, 6, 36], [1, 128, 6, 36], [1, 128, 6, 36]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 128, 7, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 7, 36]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]]}>

    // CHECK:       [[COPY_OUT:%.+]] = VPU.Copy([[MAXPOOL]])
    // CHECK-SAME:                       -> tensor<1x128x36x36xf16, {order = #NHWC}>

    // CHECK:       return [[COPY_OUT]] : tensor<1x128x36x36xf16, {order = #NHWC}>
}
