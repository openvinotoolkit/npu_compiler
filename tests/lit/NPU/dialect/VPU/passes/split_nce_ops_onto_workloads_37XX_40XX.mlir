//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW  allow-custom-values=true" --split-NCE-ops-onto-workloads %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvRewriter
func.func @ConvRewriter(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}> =
        dense<10> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x16x16xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<16x16x1x1xf16, {order = #NHWC}>
        -> tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = VPU.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32, {order = #NHWC}>
        -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    %3 = VPU.NCE.Convolution(%0, %1, %2) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.Copy(%3) : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %4 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = VPU.Copy([[CST0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL0]], [[VAL1]], [[VAL2]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 16, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
    // CHECK:           }

    // CHECK:       [[VAL4:%.+]] = VPU.Copy([[VAL3]])
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL4]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvRewriter
func.func @DepthConvRewriter(%arg0: tensor<1x16x40x80xf16, {order = #NHWC}>) -> tensor<1x16x37x73xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x1x4x8xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}> =
        dense<10> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x40x80xf16, {order = #NHWC}>
        -> tensor<1x16x40x80xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<16x1x4x8xf16, {order = #NHWC}>
        -> tensor<16x1x4x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = VPU.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32, {order = #NHWC}>
        -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    %3 = VPU.NCE.DepthConvolution(%0, %1, %2) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>,
            rawFilterShape = [16, 1, 4, 8],
            strides = [1, 1]
        } -> tensor<1x16x37x73xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.Copy(%3) : tensor<1x16x37x73xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x16x37x73xf16, {order = #NHWC}>

    return %4 : tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}>
    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x16x40x80xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x1x4x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = VPU.Copy([[CST0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.DepthConvolution([[VAL0]], [[VAL1]], [[VAL2]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x37x73xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 37, 73] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
    // CHECK:           }

    // CHECK:       [[VAL4:%.+]] = VPU.Copy([[VAL3]])
    // CHECK-SAME:      -> tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK:       return [[VAL4]] : tensor<1x16x37x73xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolRewriter
func.func @MaxPoolRewriter(%arg0: tensor<1x16x1x4xf16, {order = #NHWC}>) -> tensor<1x16x1x4xf16, {order = #NHWC}> {
    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x1x4xf16, {order = #NHWC}>
        -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %1 = VPU.NCE.MaxPool(%0) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>,
            strides = [1, 1],
            kernel_size = [1, 1]
        } -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %2 = VPU.Copy(%1) : tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x16x1x4xf16, {order = #NHWC}>

    return %2 : tensor<1x16x1x4xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL1:%.+]] = VPU.NCE.MaxPool([[VAL0]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 1, 4] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
    // CHECK:           }

    // CHECK:       [[VAL2:%.+]] = VPU.Copy([[VAL1]])
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {order = #NHWC}>

    // CHECK:       return [[VAL2]] : tensor<1x16x1x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddRewriter
func.func @EltwiseAddRewriter(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>)
        -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x64x28x28xf16, {order = #NHWC}>
        -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x64x28x28xf16, {order = #NHWC}>
        -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %2 = VPU.NCE.Eltwise(%0, %1) { op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} :
        tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %3 = VPU.Copy(%2) : tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %3 : tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL2:%.+]] = VPU.NCE.Eltwise([[VAL0]], [[VAL1]])
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 28, 28] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_8x16>
    // CHECK:           }

    // CHECK:       [[VAL3:%.+]] = VPU.Copy([[VAL2]])
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[VAL3]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPU.DistributedTensor<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 4
}>

!WeightsDistributed = !VPU.DistributedTensor<
    64x32x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!WeightsTableDistributed = !VPU.DistributedTensor<
    64x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4
}>

!Input_DDR = tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>
!Weights_DDR = tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}>
!WeightsTable_DDR = tensor<64x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}>
!Output_DDR = tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>

!InputStub_CMX = tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsStub_CMX = tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTableStub_CMX = tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
!OutputStub_CMX = tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>


// CHECK-LABEL: @ConvolutionWithDistributedTensor
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>
func.func @ConvolutionWithDistributedTensor(%arg0: !Input_DDR) -> !Output_DDR {
    %weights = const.Declare tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16, {mem_space = @DDR}>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = dense<10> : tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>

    %input_cmx = VPU.Copy(%arg0) { out_mem_space = @CMX_NN } : !Input_DDR -> !InputDistributed

    %weights_cmx = VPU.Copy(%weights) { out_mem_space = @CMX_NN } : !Weights_DDR -> !WeightsDistributed

    %wt_cmx = VPU.Copy(%wt) { out_mem_space = @CMX_NN } : tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> -> !WeightsTableDistributed

        // Generate different workloads due to different pads on each cluster
    %output_cmx = VPU.NCE.Convolution(%input_cmx, %weights_cmx, %wt_cmx) {
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>,
                rawFilterShape = [64, 32, 3, 3],
                strides = [1, 1]
            } : !InputDistributed, !WeightsDistributed, !WeightsTableDistributed -> !OutputDistributed

    %output = VPU.Copy(%output_cmx) { out_mem_space = @DDR } : !OutputDistributed -> !Output_DDR

    return %output: !Output_DDR

    // CHECK-DAG:        [[WEIGHTS:%.*]] = const.Declare tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK-SAME:       = dense<1.000000e+00> : tensor<64x32x3x3xf16, {mem_space = @DDR}>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:        [[WEIGHTS_TABLE:%.*]] = const.Declare tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    // CHECK:        [[INPUT_CMX:%.*]] = VPU.Copy([[INPUT]])
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x32x16x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], kernel = [3, 3],
    // CHECK-SAME:               pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, strides = [1, 1], num_clusters = 4 : i64}>

    // CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.Copy([[WEIGHTS]])
    // CHECK-SAME:           -> !VPU.DistributedTensor<64x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:        [[WEIGHTS_TABLE_CMX:%.*]] = VPU.Copy([[WEIGHTS_TABLE]])
    // CHECK-SAME:           -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.Convolution([[INPUT_CMX]], [[WEIGHTS_CMX]], [[WEIGHTS_TABLE_CMX]])
    // CHECK-SAME:                            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:                            strides = [1, 1]
    // CHECK-SAME:             -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:                    DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 4, 16] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK:                    DPU.Workload outOffsets [0, 0, 4, 0] outSizes [1, 64, 4, 16] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK:                    DPU.Workload outOffsets [0, 0, 8, 0] outSizes [1, 64, 4, 16] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 2 : i64}
    // CHECK:                    DPU.Workload outOffsets [0, 0, 12, 0] outSizes [1, 64, 4, 16] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 3 : i64}

    // CHECK:        [[OUT:%.*]] = VPU.Copy([[OUT_CMX]])
    // CHECK-SAME:           -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:        return [[OUT]] : tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvRewriterCUBOID_16x16
func.func @ConvRewriterCUBOID_16x16(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}> =
        dense<10> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x16x16xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<16x16x1x1xf16, {order = #NHWC}>
        -> tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = VPU.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32, {order = #NHWC}>
        -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    %3 = VPU.NCE.Convolution(%0, %1, %2) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.Copy(%3) : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %4 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = VPU.Copy([[CST0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL0]], [[VAL1]], [[VAL2]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 16, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
    // CHECK:           }

    // CHECK:       [[VAL4:%.+]] = VPU.Copy([[VAL3]])
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL4]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----



#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvRewriterCUBOID_8x16
func.func @ConvRewriterCUBOID_8x16(%arg0: tensor<1x128x56x28xf16, {order = #NHWC}>) -> tensor<1x128x56x28xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<128x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<128x1x1x4xsi32, {order = #NHWC}> =
        dense<10> : tensor<128x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x128x56x28xf16, {order = #NHWC}>
        -> tensor<1x128x56x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<128x128x1x1xf16, {order = #NHWC}>
        -> tensor<128x128x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = VPU.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<128x1x1x4xsi32, {order = #NHWC}>
        -> tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    %3 = VPU.NCE.Convolution(%0, %1, %2) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>,
            rawFilterShape = [128, 128, 1, 1],
            strides = [1, 1]
        } : tensor<1x128x56x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<128x128x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x128x56x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.Copy(%3) : tensor<1x128x56x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x128x56x28xf16, {order = #NHWC}>

    return %4 : tensor<1x128x56x28xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}>
    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<128x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x128x56x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<128x128x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = VPU.Copy([[CST0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL0]], [[VAL1]], [[VAL2]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x128x56x28xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 128, 56, 28] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_8x16>
    // CHECK:           }

    // CHECK:       [[VAL4:%.+]] = VPU.Copy([[VAL3]])
    // CHECK-SAME:      -> tensor<1x128x56x28xf16, {order = #NHWC}>

    // CHECK:       return [[VAL4]] : tensor<1x128x56x28xf16, {order = #NHWC}>
}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvRewriterCUBOID_4x16
func.func @ConvRewriterCUBOID_4x16(%arg0: tensor<1x256x8x8xf16, {order = #NHWC}>) -> tensor<1x256x8x8xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<256x256x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<256x256x1x1xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<256x1x1x4xsi32, {order = #NHWC}> =
        dense<10> : tensor<256x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x256x8x8xf16, {order = #NHWC}>
        -> tensor<1x256x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<256x256x1x1xf16, {order = #NHWC}>
        -> tensor<256x256x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = VPU.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<256x1x1x4xsi32, {order = #NHWC}>
        -> tensor<256x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    %3 = VPU.NCE.Convolution(%0, %1, %2) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>,
            rawFilterShape = [256, 256, 1, 1],
            strides = [1, 1]
        } : tensor<1x256x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<256x256x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<256x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x256x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.Copy(%3) : tensor<1x256x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x256x8x8xf16, {order = #NHWC}>

    return %4 : tensor<1x256x8x8xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<256x256x1x1xf16, {order = #NHWC}>
    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<256x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x256x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<256x256x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = VPU.Copy([[CST0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<256x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL0]], [[VAL1]], [[VAL2]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x256x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 256, 8, 8] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK:           }

    // CHECK:       [[VAL4:%.+]] = VPU.Copy([[VAL3]])
    // CHECK-SAME:      -> tensor<1x256x8x8xf16, {order = #NHWC}>

    // CHECK:       return [[VAL4]] : tensor<1x256x8x8xf16, {order = #NHWC}>
}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPU.SparseTensor<
    data=!VPU.DistributedTensor<1x16x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    sparsity_map=!VPU.DistributedTensor<1x16x224x224xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    >

!WeightsDistributed = !VPU.DistributedTensor<
    32x16x7x7xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!WeightsTableDistributed = !VPU.DistributedTensor<
    32x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!OutputDistributed = !VPU.SparseTensor<
    data=!VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    sparsity_map=!VPU.DistributedTensor<1x32x112x112xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    >

!Input_DDR = !VPU.SparseTensor<data=tensor<1x16x224x224xf16, {mem_space = @DDR, order = #NHWC}>, sparsity_map=tensor<1x16x224x224xi1, {mem_space = @DDR, order = #NHWC}>>
!Weights_DDR = tensor<32x16x7x7xf16, {mem_space = @DDR, order = #NHWC}>
!WeightsTable_DDR = tensor<32x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}>
!Output_DDR = !VPU.SparseTensor<
    data=tensor<1x32x112x112xf16, {mem_space = @DDR, order = #NHWC}>,
    sparsity_map=tensor<1x32x112x112xi1, {mem_space = @DDR, order = #NHWC}>
    >

!InputStub_CMX = !VPU.SparseTensor<data=tensor<1x16x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x16x224x224xi1, {mem_space = @CMX_NN, order = #NHWC}>>
!WeightsStub_CMX = tensor<32x16x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTableStub_CMX = tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
!OutputStub_CMX = !VPU.SparseTensor<
    data=tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    sparsity_map=tensor<1x32x112x112xi1, {mem_space = @CMX_NN, order = #NHWC}>
    >


// CHECK-LABEL: @ConvolutionWithSparseDistributedTensor
// CHECK-SAME:  ([[INPUT:%.+]]: !VPU.SparseTensor<data=tensor<1x16x224x224xf16, {mem_space = @DDR, order = #NHWC}>, sparsity_map=tensor<1x16x224x224xi1, {mem_space = @DDR, order = #NHWC}>>
func.func @ConvolutionWithSparseDistributedTensor(%arg0: !Input_DDR) -> !Output_DDR {
    %weights = const.Declare !Weights_DDR = dense<1.000000e+00> : tensor<32x16x7x7xf16, {mem_space = @DDR}>, [#const.Reorder<#NHWC>]
    %wt = const.Declare !WeightsTable_DDR = dense<10> : tensor<32x1x1x4xsi32, {mem_space = @CMX_NN}>

    // copy input DDR -> CMX
    %input_cmx = VPU.Copy(%arg0) { out_mem_space = @CMX_NN } : !Input_DDR -> !InputDistributed

    // copy dense weights DDR->CMX
    %weights_cmx = VPU.Copy(%weights) { out_mem_space = @CMX_NN } : !Weights_DDR -> !WeightsDistributed

    // copy weights table DDR->CMX
    %wt_cmx = VPU.Copy(%wt) { out_mem_space = @CMX_NN } : !WeightsTable_DDR -> !WeightsTableDistributed

        // Generate different workloads due to different pads on each cluster
    %output_cmx = VPU.NCE.Convolution(%input_cmx, %weights_cmx, %wt_cmx) {
            pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>,
            rawFilterShape = [32, 16, 7, 7],
            strides = [2, 2]
            } : !InputDistributed, !WeightsDistributed, !WeightsTableDistributed -> !OutputDistributed

    %output = VPU.Copy(%output_cmx) { out_mem_space = @DDR } : !OutputDistributed -> !Output_DDR

    return %output: !Output_DDR

    // CHECK-DAG:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x7x7xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK-SAME:       = dense<1.000000e+00> : tensor<32x16x7x7xf16, {mem_space = @DDR}>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:        [[WEIGHTS_TABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}>
    // CHECK-SAME:       = dense<10> : tensor<32x1x1x4xsi32, {mem_space = @CMX_NN}>
    // CHECK:        [[INPUT_CMX:%.*]] = VPU.Copy([[INPUT]])
    // CHECK-SAME:           -> !VPU.SparseTensor<
    // CHECK-SAME:              data=!VPU.DistributedTensor<1x16x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:              sparsity_map=!VPU.DistributedTensor<1x16x224x224xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>>

    // CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.Copy([[WEIGHTS]])
    // CHECK-SAME:           -> !VPU.DistributedTensor<32x16x7x7xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[WEIGHTS_TABLE_CMX:%.*]] = VPU.Copy([[WEIGHTS_TABLE]])
    // CHECK-SAME:           -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.Convolution([[INPUT_CMX]], [[WEIGHTS_CMX]], [[WEIGHTS_TABLE_CMX]])
    // CHECK-SAME:                            pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
    // CHECK-SAME:                            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:                            rawFilterShape = [32, 16, 7, 7], strides = [2, 2]}
    // CHECK-SAME:             -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                  sparsity_map=!VPU.DistributedTensor<1x32x112x112xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>>

    // CHECK:                    VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 56, 112] <left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK:                    VPU.DPU.Workload outOffsets [0, 0, 56, 0] outSizes [1, 32, 56, 112] <left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 2 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}

    // CHECK:        [[OUT:%.*]] = VPU.Copy([[OUT_CMX]])
    // CHECK-SAME:           -> !VPU.SparseTensor<data=tensor<1x32x112x112xf16, {mem_space = @DDR, order = #NHWC}>,
    // CHECK-SAME:                                sparsity_map=tensor<1x32x112x112xi1, {mem_space = @DDR, order = #NHWC}>>

    // CHECK:        return [[OUT]] : !VPU.SparseTensor<data=tensor<1x32x112x112xf16, {mem_space = @DDR, order = #NHWC}>,
    // CHECK-SAME:                                      sparsity_map=tensor<1x32x112x112xi1, {mem_space = @DDR, order = #NHWC}>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @InterpolateBilinear(
    %arg0: tensor<1x64x5x10xf16, {order = #NHWC}>,                       // data
    %arg1: tensor<64x64x2x2xf16, {order = #NHWC, mem_space = @CMX_NN}>,  // weights
    %arg2: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>                  // weight table
) -> tensor<1x64x10x20xf16, {order = #NHWC, mem_space = @CMX_NN}> {
    %sparsityMap = const.Declare tensor<1x64x11x21xi1> = dense<1> : tensor<1x64x11x21xi1>

    %storageElement = VPU.StorageElementTable {
        dataElemType = f16,
        seDepth = 1,
        seSize = [64],
        dataShape = [1, 64, 5, 10],
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 11, 21]>
    } -> tensor<1x1x11x21xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsityMap, %storageElement) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 11, 21]>
    } ->
        !VPU.SparseTensor<
            data=tensor<1x64x5x10xf16, {order = #NHWC}>,
            sparsity_map=tensor<1x64x11x21xi1>,
            storage_element_table=tensor<1x1x11x21xi32, {order = #NHWC}>,
            #VPU.SEInterpolate<
                mode = <BILINEAR>,
                coordinate_transformation_mode = <ASYMMETRIC>,
                scale = [1.0, 1.0, 2.0, 2.0],
                offsets = [0, 0, 0, 0],
                sizes = [1, 64, 11, 21]>>

    %input_cmx = VPU.Copy(%input) {out_mem_space = @CMX_NN} : !VPU.SparseTensor<
        data=tensor<1x64x5x10xf16, {order = #NHWC}>,
        sparsity_map=tensor<1x64x11x21xi1>,
        storage_element_table=tensor<1x1x11x21xi32, {order = #NHWC}>,
        #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 11, 21]>>
        -> !VPU.SparseTensor<
            data=tensor<1x64x5x10xf16, {order = #NHWC, mem_space = @CMX_NN}>,
            sparsity_map=tensor<1x64x11x21xi1, {mem_space = @CMX_NN}>,
            storage_element_table=tensor<1x1x11x21xi32, {order = #NHWC, mem_space = @CMX_NN}>,
            #VPU.SEInterpolate<
                mode = <BILINEAR>,
                coordinate_transformation_mode = <ASYMMETRIC>,
                scale = [1.0, 1.0, 2.0, 2.0],
                offsets = [0, 0, 0, 0],
                sizes = [1, 64, 11, 21]>>

    %task = VPU.NCE.Interpolate(%input_cmx, %arg1, %arg2) {
        rawFilterShape = [64, 64, 2, 2],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>
    } -> tensor<1x64x10x20xf16, {order = #NHWC, mem_space = @CMX_NN}>

    return %task : tensor<1x64x10x20xf16, {order = #NHWC, mem_space = @CMX_NN}>

    // CHECK:       [[SPARSITY_MAP:%.+]] = const.Declare tensor<1x64x11x21xi1> = dense<true> : tensor<1x64x11x21xi1>

    // CHECK:       [[STORAGE_ELEMENT:%.+]] = VPU.StorageElementTable {
    // CHECK-SAME:      dataElemType = f16,
    // CHECK-SAME:      dataShape = [1, 64, 5, 10],
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<
    // CHECK-SAME:          mode = <BILINEAR>,
    // CHECK-SAME:          coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 0, 0],
    // CHECK-SAME:          sizes = [1, 64, 11, 21]>,
    // CHECK-SAME:      seDepth = 1 : i64,
    // CHECK-SAME:      seSize = [64]
    // CHECK-SAME:  } -> tensor<1x1x11x21xi32, {order = #NHWC}>

    // CHECK:       [[SPARSE_TENSOR:%.+]] = VPU.GroupSparseTensor(%arg0, [[SPARSITY_MAP]], [[STORAGE_ELEMENT]]) {
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<
    // CHECK-SAME:          mode = <BILINEAR>,
    // CHECK-SAME:          coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 0, 0],
    // CHECK-SAME:          sizes = [1, 64, 11, 21]>
    // CHECK-SAME:  } -> !VPU.SparseTensor<
    // CHECK-SAME:      data=tensor<1x64x5x10xf16, {order = #NHWC}>,
    // CHECK-SAME:      sparsity_map=tensor<1x64x11x21xi1>,
    // CHECK-SAME:      storage_element_table=tensor<1x1x11x21xi32, {order = #NHWC}>,
    // CHECK-SAME:      #VPU.SEInterpolate<
    // CHECK-SAME:          mode = <BILINEAR>,
    // CHECK-SAME:          coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 0, 0],
    // CHECK-SAME:          sizes = [1, 64, 11, 21]>>

    // CHECK:       [[SPARSE_TENSOR_CMX:%.+]] = VPU.Copy([[SPARSE_TENSOR]]) {out_mem_space = @CMX_NN} : !VPU.SparseTensor<
    // CHECK-SAME:      data=tensor<1x64x5x10xf16, {order = #NHWC}>,
    // CHECK-SAME:      sparsity_map=tensor<1x64x11x21xi1>,
    // CHECK-SAME:      storage_element_table=tensor<1x1x11x21xi32, {order = #NHWC}>,
    // CHECK-SAME:      #VPU.SEInterpolate<
    // CHECK-SAME:          mode = <BILINEAR>,
    // CHECK-SAME:          coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 0, 0],
    // CHECK-SAME:          sizes = [1, 64, 11, 21]>> ->
    // CHECK-SAME:  !VPU.SparseTensor<
    // CHECK-SAME:      data=tensor<1x64x5x10xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:      sparsity_map=tensor<1x64x11x21xi1, {mem_space = @CMX_NN}>,
    // CHECK-SAME:      storage_element_table=tensor<1x1x11x21xi32, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:      #VPU.SEInterpolate<
    // CHECK-SAME:          mode = <BILINEAR>,
    // CHECK-SAME:          coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 0, 0],
    // CHECK-SAME:          sizes = [1, 64, 11, 21]>>

    // CHECK:       [[INTERPOLATE:%.+]] = VPU.NCE.Interpolate([[SPARSE_TENSOR_CMX]], %arg1, %arg2) {
    // CHECK-SAME:      mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:      rawFilterShape = [64, 64, 2, 2],
    // CHECK-SAME:      scales_attr = [2, 2]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } -> tensor<1x64x10x20xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 10, 20] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
    // CHECK:       }

    // CHECK:       return [[INTERPOLATE]] : tensor<1x64x10x20xf16, {mem_space = @CMX_NN, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @InterpolateNearest(
    %arg0: tensor<1x64x5x10xf16, {order = #NHWC}>,                       // data
    %arg1: tensor<64x64x1x1xf16, {order = #NHWC, mem_space = @CMX_NN}>,  // weights
    %arg2: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>                  // weight table
) -> tensor<1x64x10x20xf16, {order = #NHWC, mem_space = @CMX_NN}> {
    %sparsityMap = const.Declare tensor<1x64x10x20xi1> = dense<1> : tensor<1x64x10x20xi1>

    %storageElement = VPU.StorageElementTable {
        dataElemType = f16,
        seDepth = 1,
        seSize = [64],
        dataShape = [1, 64, 5, 10],
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 10, 20]>
    } -> tensor<1x1x10x20xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsityMap, %storageElement) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 10, 20]>
    } ->
        !VPU.SparseTensor<
            data=tensor<1x64x5x10xf16, {order = #NHWC}>,
            sparsity_map=tensor<1x64x10x20xi1>,
            storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
            #VPU.SEInterpolate<
                mode = <NEAREST>,
                coordinate_transformation_mode = <ASYMMETRIC>,
                scale = [1.0, 1.0, 2.0, 2.0],
                nearest_mode = <FLOOR>,
                offsets = [0, 0, 0, 0],
                sizes = [1, 64, 10, 20]>>

    %input_cmx = VPU.Copy(%input) {out_mem_space = @CMX_NN} : !VPU.SparseTensor<
        data=tensor<1x64x5x10xf16, {order = #NHWC}>,
        sparsity_map=tensor<1x64x10x20xi1>,
        storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
        #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 10, 20]>>
        -> !VPU.SparseTensor<
            data=tensor<1x64x5x10xf16, {order = #NHWC, mem_space = @CMX_NN}>,
            sparsity_map=tensor<1x64x10x20xi1, {mem_space = @CMX_NN}>,
            storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC, mem_space = @CMX_NN}>,
            #VPU.SEInterpolate<
                mode = <NEAREST>,
                coordinate_transformation_mode = <ASYMMETRIC>,
                scale = [1.0, 1.0, 2.0, 2.0],
                nearest_mode = <FLOOR>,
                offsets = [0, 0, 0, 0],
                sizes = [1, 64, 10, 20]>>

    %task = VPU.NCE.Interpolate(%input_cmx, %arg1, %arg2) {
        rawFilterShape = [64, 64, 1, 1],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>
    } -> tensor<1x64x10x20xf16, {order = #NHWC, mem_space = @CMX_NN}>

    return %task : tensor<1x64x10x20xf16, {order = #NHWC, mem_space = @CMX_NN}>

    // CHECK:       [[SPARSITY_MAP:%.+]] = const.Declare tensor<1x64x10x20xi1> = dense<true> : tensor<1x64x10x20xi1>

    // CHECK:       [[STORAGE_ELEMENT:%.+]] = VPU.StorageElementTable {
    // CHECK-SAME:      dataElemType = f16,
    // CHECK-SAME:      dataShape = [1, 64, 5, 10],
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<
    // CHECK-SAME:          mode = <NEAREST>,
    // CHECK-SAME:          coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          nearest_mode = <FLOOR>,
    // CHECK-SAME:          offsets = [0, 0, 0, 0],
    // CHECK-SAME:          sizes = [1, 64, 10, 20]>,
    // CHECK-SAME:      seDepth = 1 : i64,
    // CHECK-SAME:      seSize = [64]
    // CHECK-SAME:  } -> tensor<1x1x10x20xi32, {order = #NHWC}>

    // CHECK:       [[SPARSE_TENSOR:%.+]] = VPU.GroupSparseTensor(%arg0, [[SPARSITY_MAP]], [[STORAGE_ELEMENT]]) {
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<
    // CHECK-SAME:          mode = <NEAREST>,
    // CHECK-SAME:          coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          nearest_mode = <FLOOR>,
    // CHECK-SAME:          offsets = [0, 0, 0, 0],
    // CHECK-SAME:          sizes = [1, 64, 10, 20]>
    // CHECK-SAME:  } -> !VPU.SparseTensor<
    // CHECK-SAME:      data=tensor<1x64x5x10xf16, {order = #NHWC}>,
    // CHECK-SAME:      sparsity_map=tensor<1x64x10x20xi1>,
    // CHECK-SAME:      storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
    // CHECK-SAME:      #VPU.SEInterpolate<
    // CHECK-SAME:          mode = <NEAREST>,
    // CHECK-SAME:          coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          nearest_mode = <FLOOR>,
    // CHECK-SAME:          offsets = [0, 0, 0, 0],
    // CHECK-SAME:          sizes = [1, 64, 10, 20]>>

    // CHECK:       [[SPARSE_TENSOR_CMX:%.+]] = VPU.Copy([[SPARSE_TENSOR]]) {out_mem_space = @CMX_NN} : !VPU.SparseTensor<
    // CHECK-SAME:      data=tensor<1x64x5x10xf16, {order = #NHWC}>,
    // CHECK-SAME:      sparsity_map=tensor<1x64x10x20xi1>,
    // CHECK-SAME:      storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
    // CHECK-SAME:      #VPU.SEInterpolate<
    // CHECK-SAME:          mode = <NEAREST>,
    // CHECK-SAME:          coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          nearest_mode = <FLOOR>,
    // CHECK-SAME:          offsets = [0, 0, 0, 0],
    // CHECK-SAME:          sizes = [1, 64, 10, 20]>> ->
    // CHECK-SAME:  !VPU.SparseTensor<
    // CHECK-SAME:      data=tensor<1x64x5x10xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:      sparsity_map=tensor<1x64x10x20xi1, {mem_space = @CMX_NN}>,
    // CHECK-SAME:      storage_element_table=tensor<1x1x10x20xi32, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:      #VPU.SEInterpolate<
    // CHECK-SAME:          mode = <NEAREST>,
    // CHECK-SAME:          coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          nearest_mode = <FLOOR>,
    // CHECK-SAME:          offsets = [0, 0, 0, 0],
    // CHECK-SAME:          sizes = [1, 64, 10, 20]>>

    // CHECK:       [[INTERPOLATE:%.+]] = VPU.NCE.Interpolate([[SPARSE_TENSOR_CMX]], %arg1, %arg2) {
    // CHECK-SAME:      mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:      rawFilterShape = [64, 64, 1, 1],
    // CHECK-SAME:      scales_attr = [2, 2]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } -> tensor<1x64x10x20xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 10, 20] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
    // CHECK:       }

    // CHECK:       return [[INTERPOLATE]] : tensor<1x64x10x20xf16, {mem_space = @CMX_NN, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: func.func @SplitNCEPermute
func.func @SplitNCEPermute(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x4x224x224x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>
    } -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute(%arg0) {
    // CHECK-SAME:      dstElemType = !qElemType, dstOrder = #NHWC,
    // CHECK-SAME:      expandedChannels = 4 : i64,
    // CHECK-SAME:      minimumHardwareExecutionCost = {{[1-9][0-9]+}} : i64
    // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>}
    // CHECK-SAME:      -> tensor<1x4x224x224x!qElemType, {order = #NHWC}> {
    // CHECK:               DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 4, 224, 224] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
    // CHECK:           }

    // CHECK:       return [[NCE_PERMUTE]] : tensor<1x4x224x224x!qElemType, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Input_CMX = !VPU.DistributedTensor<
    1x3x224x224xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [1, 1],
    num_clusters = 2
}>

!Output_CMX = !VPU.DistributedTensor<
    1x4x224x224xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!Input_DDR = tensor<1x3x224x224xf16>
!InputStub_CMX = tensor<1x3x224x224xf16, {mem_space = @CMX_NN}>
!OutputStub_CMX = tensor<1x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @NCEPermuteOverlapSegmented
func.func @NCEPermuteOverlapSegmented(%arg0: !Input_DDR) -> !Output_CMX {

    %input_cmx = VPU.Copy(%arg0) { out_mem_space = @CMX_NN } : !Input_DDR -> !Input_CMX

    %output = VPU.NCE.Permute(%input_cmx) {
                dstElemType = !quant.uniform<u8:f16, 1.000000e+00>,
                dstOrder = #NHWC,
                expandedChannels = 4 : i64,
                ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>
        } -> !Output_CMX

    return %output : !Output_CMX

    // CHECK:       VPU.NCE.Permute
    // CHECK-SAME:      dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x4x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [0, 0, 0, 0] outSizes [1, 4, 112, 224]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 0

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [0, 0, 112, 0] outSizes [1, 4, 112, 224]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 1

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Input_CMX = !VPU.DistributedTensor<
    1x3x224x224xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [7, 7],
    pads = #VPU.Padding<left = 3 , right = 2, top = 3, bottom = 2>,
    strides = [2, 2],
    num_clusters = 2
}>

!Output_CMX = !VPU.DistributedTensor<
    1x4x224x224xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [7, 7],
    pads = #VPU.Padding<left = 3 , right = 2, top = 3, bottom = 2>,
    strides = [2, 2],
    num_clusters = 2,
    equal_memory_and_compute_view
}>

!Input_DDR = tensor<1x3x224x224xf16>
!InputStub_CMX = tensor<1x3x224x224xf16, {mem_space = @CMX_NN}>
!OutputStub_CMX = tensor<1x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @NCEPermuteOverlap
func.func @NCEPermuteOverlap(%arg0: !Input_DDR) -> !Output_CMX {

    %input_cmx = VPU.Copy(%arg0) { out_mem_space = @CMX_NN } : !Input_DDR -> !Input_CMX

    %output = VPU.NCE.Permute(%input_cmx) {
                dstElemType = !quant.uniform<u8:f16, 1.000000e+00>,
                dstOrder = #NHWC,
                expandedChannels = 4 : i64,
                ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>
        } -> !Output_CMX

    return %output : !Output_CMX

    // CHECK:       VPU.NCE.Permute
    // CHECK-SAME:      dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x4x224x224xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:          mode = "OVERLAPPED",
    // CHECK-SAME:          num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:          kernel = [7, 7],
    // CHECK-SAME:          pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
    // CHECK-SAME:          strides = [2, 2],
    // CHECK-SAME:          num_clusters = 2 : i64,
    // CHECK-SAME:          equal_memory_and_compute_view}>

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [0, 0, 0, 0] outSizes [1, 4, 114, 224]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 0

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [0, 0, 109, 0] outSizes [1, 4, 115, 224]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 1

}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
#GNCHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

!InputDistributed = !VPU.DistributedTensor<6x1x16x4x1xf16, #GNHWC, @CMX_NN,
{mode = "SEGMENTED", num_tiles = [6, 1, 1, 1, 1], num_clusters = 6 : i64}>

!WeightsDistributed = !VPU.DistributedTensor<6x32x16x1x1xf16, #GNHWC, @CMX_NN,
 {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1, 1], num_clusters = 6 : i64}>


 !OutputDistributed = !VPU.DistributedTensor<6x1x32x4x1xf16, #GNHWC, @CMX_NN,
 {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1, 1], num_clusters = 6 : i64}>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz
// CHECK-LABEL: @MatMulRewriter
func.func @MatMulRewriter(%arg0: !InputDistributed,%arg1: !WeightsDistributed) -> tensor<6x1x32x4x1xf16, {order =#GNHWC}> {
    %cst = const.Declare tensor<6x32x1x1x4xsi32, {mem_space = @CMX, order = #GNCHW}> = dense<1> : tensor<6x32x1x1x4xsi32,  {mem_space = @CMX}>

    %10 = VPU.NCE.MatMul(%arg0, %arg1, %cst)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>,
         rawFilterShape = [6, 32, 16, 1, 1], strides = [1, 1]}
          -> !OutputDistributed

    %4 = VPU.Copy(%10) : !OutputDistributed
        -> tensor<6x1x32x4x1xf16, {order = #GNHWC}>

    // CHECK:       VPU.NCE.MatMul

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [0, 0, 0, 0, 0] outSizes [1, 1, 32, 4, 1]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 0

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [1, 0, 0, 0, 0] outSizes [1, 1, 32, 4, 1]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 1

        // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [2, 0, 0, 0, 0] outSizes [1, 1, 32, 4, 1]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 2

        // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [3, 0, 0, 0, 0] outSizes [1, 1, 32, 4, 1]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 3

        // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [4, 0, 0, 0, 0] outSizes [1, 1, 32, 4, 1]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 4

        // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [5, 0, 0, 0, 0] outSizes [1, 1, 32, 4, 1]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 5

    return %4 : tensor<6x1x32x4x1xf16, {order = #GNHWC}>

}
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
#GNCHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

!InputDistributed = !VPU.DistributedTensor<6x1x16x4x1xf16, #GNHWC,
 @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
 compute_shapes = [[2, 1, 16, 4, 1], [2, 1, 16, 4, 1], [1, 1, 16, 4, 1], [1, 1, 16, 4, 1]],
 compute_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [5, 0, 0, 0, 0]],
 memory_shapes = [[2, 1, 16, 4, 1], [2, 1, 16, 4, 1], [1, 1, 16, 4, 1], [1, 1, 16, 4, 1]],
 memory_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [5, 0, 0, 0, 0]]}>

!WeightsDistributed = !VPU.DistributedTensor<6x32x16x1x1xf16, #GNHWC,
 @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
  compute_shapes = [[2, 32, 16, 1, 1], [2, 32, 16, 1, 1], [1, 32, 16, 1, 1], [1, 32, 16, 1, 1]],
  compute_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [5, 0, 0, 0, 0]],
  memory_shapes = [[2, 32, 16, 1, 1], [2, 32, 16, 1, 1], [1, 32, 16, 1, 1], [1, 32, 16, 1, 1]],
  memory_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [5, 0, 0, 0, 0]]}>


!WeightsTableDistributed = !VPU.DistributedTensor<6x32x1x1x4xsi32, #GNCHW,
  @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
   compute_shapes = [[2, 32, 1, 1, 4], [2, 32, 1, 1, 4], [1, 32, 1, 1, 4], [1, 32, 1, 1, 4]],
   compute_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [5, 0, 0, 0, 0]],
   memory_shapes = [[2, 32, 1, 1, 4], [2, 32, 1, 1, 4], [1, 32, 1, 1, 4], [1, 32, 1, 1, 4]],
   memory_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [5, 0, 0, 0, 0]]}>

 !OutputDistributed = !VPU.DistributedTensor<6x1x32x4x1xf16, #GNHWC,
  @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
   compute_shapes = [[2, 1, 32, 4, 1], [2, 1, 32, 4, 1], [1, 1, 32, 4, 1], [1, 1, 32, 4, 1]],
   compute_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [5, 0, 0, 0, 0]],
   memory_shapes = [[2, 1, 32, 4, 1], [2, 1, 32, 4, 1], [1, 1, 32, 4, 1], [1, 1, 32, 4, 1]],
   memory_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [5, 0, 0, 0, 0]]}>


module @executors {
config.Resources 4 of @NCE at 1.700000e+03 MHz
// CHECK-LABEL: @MatMulUnevenDistributionRewriter
func.func @MatMulUnevenDistributionRewriter(%arg0: !InputDistributed,%arg1: !WeightsDistributed) -> tensor<6x1x32x4x1xf16, {order =#GNHWC}> {
    %cst = const.Declare tensor<6x32x1x1x4xsi32, {mem_space = @CMX, order = #GNCHW}> = dense<1> : tensor<6x32x1x1x4xsi32,  {mem_space = @CMX}>

    %10 = VPU.NCE.MatMul(%arg0, %arg1, %cst)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>,
         rawFilterShape = [6, 32, 16, 1, 1], strides = [1, 1]}
          -> !OutputDistributed

    %4 = VPU.Copy(%10) :!OutputDistributed
        -> tensor<6x1x32x4x1xf16, {order = #GNHWC}>

    // CHECK:       VPU.NCE.MatMul

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [0, 0, 0, 0, 0] outSizes [2, 1, 32, 4, 1]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 0

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [2, 0, 0, 0, 0] outSizes [2, 1, 32, 4, 1]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 1

        // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [4, 0, 0, 0, 0] outSizes [1, 1, 32, 4, 1]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 2

        // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [5, 0, 0, 0, 0] outSizes [1, 1, 32, 4, 1]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 3

    return %4 : tensor<6x1x32x4x1xf16, {order = #GNHWC}>

}
}
