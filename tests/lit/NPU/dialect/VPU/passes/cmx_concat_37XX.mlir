//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --cmx-concat --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720

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

!SparseInput = !VPU.SparseTensor<
    data=tensor<1x256x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    sparsity_map=tensor<1x256x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>
>

!SparseOutput = !VPU.SparseTensor<
    data=!VPU.DistributedTensor<1x256x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    sparsity_map=!VPU.DistributedTensor<1x256x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
>

!SparseConvOutput = !VPU.SparseTensor<
    data=tensor<1x128x14x14xf16, {order = #NHWC}>,
    sparsity_map=tensor<1x128x14x14xi1, {order = #NHWC}>
>

!SparseConvOutputCMX = !VPU.SparseTensor<
    data=tensor<1x128x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    sparsity_map=tensor<1x128x14x14xi1, {mem_space = @CMX_NN, order = #NHWC}>
>

!SparseConvOutputDist = !VPU.SparseTensor<
    data=!VPU.DistributedTensor<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    sparsity_map=!VPU.DistributedTensor<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
>

!SparseConcatOutput = !VPU.SparseTensor<
    data=tensor<1x256x14x14xf16, {order = #NHWC}>,
    sparsity_map=tensor<1x256x14x14xi1, {order = #NHWC}>
>

!SparseConcatOutputCMX = !VPU.SparseTensor<
    data=tensor<1x256x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    sparsity_map=tensor<1x256x14x14xi1, {mem_space = @CMX_NN, order = #NHWC}>
>

// CHECK:      func.func @SparseConvolution([[INPUT:%.+]]: !VPU.SparseTensor<data=tensor<1x256x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x256x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>,
// CHECK-SAME:                         [[WEIGHTS_TABLE:%.+]]: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
// CHECK-SAME:                         [[WEIGHTS:%.+]]: tensor<128x256x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>)
// CHECK-SAME:         -> !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {order = #NHWC}>, sparsity_map=tensor<1x128x6x6xi1, {order = #NHWC}>> {
func.func @SparseConvolution(%input: !SparseInput,
                        %weightsTable: !WeightsTableType,
                        %weights: !WeightsType)
        -> !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {order = #NHWC}>, sparsity_map=tensor<1x128x6x6xi1, {order = #NHWC}>> {

    // Convolution 0
    %0 = VPU.NCE.Convolution(%input, %weights, %weightsTable) rawFilterShape [128, 256, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                 strides = [2, 2]}
                    : !SparseInput, !WeightsType, !WeightsTableType -> !SparseConvOutputDist

    // Convolution 0 output copy
    %1 = VPU.Copy(%0) : !SparseConvOutputDist -> !SparseConvOutput

    // Convolution 1
    %2 = VPU.NCE.Convolution(%input, %weights, %weightsTable) rawFilterShape [128, 256, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                 strides = [2, 2]}
                    : !SparseInput, !WeightsType, !WeightsTableType -> !SparseConvOutputDist

    // Convolution 1 output copy
    %3 = VPU.Copy(%2) : !SparseConvOutputDist -> !SparseConvOutput

    // Concat
    %4 = VPU.Concat(%1, %3) {static_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]} : !SparseConvOutput, !SparseConvOutput -> !SparseConcatOutput

    // Concat output copy
    %5 = VPU.Copy(%4) {out_mem_space = @CMX_NN} : !SparseConcatOutput -> !SparseOutput

    // Convolution 2
    %6 = VPU.NCE.Convolution(%5, %weights, %weightsTable) rawFilterShape [128, 256, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                 strides = [2, 2]}
            : !SparseOutput, !WeightsType, !WeightsTableType -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x128x6x6xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPU.DistributedTensor<1x128x6x6xi1, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>


    %7 = VPU.Copy(%6) : !VPU.SparseTensor<data=!VPU.DistributedTensor<1x128x6x6xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPU.DistributedTensor<1x128x6x6xi1, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
             -> !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, sparsity_map=tensor<1x128x6x6xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>>

    return %7 : !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, sparsity_map=tensor<1x128x6x6xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>>

    // Convolution 0
    // CHECK:       [[CONV0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x128x14x14xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                                  {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:                               sparsity_map=!VPU.DistributedTensor<1x128x14x14xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                                  {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // Convolution 1
    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x128x14x14xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                                  {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:                               sparsity_map=!VPU.DistributedTensor<1x128x14x14xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                                  {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // DistributedCast for Conv0
    // CHECK:       [[DISTR_CAST0:%.+]] = VPU.DistributedCast([[CONV0]] :
    // CHECK-SAME:         !VPU.SparseTensor<
    // CHECK-SAME:              data=!VPU.DistributedTensor<1x128x14x14xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                    {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:              sparsity_map=!VPU.DistributedTensor<1x128x14x14xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                    {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x128x14x14xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                                 {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:                           sparsity_map=!VPU.DistributedTensor<1x128x14x14xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                                 {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // DistributedCast for Conv1
    // CHECK:       [[DISTR_CAST1:%.+]] = VPU.DistributedCast([[CONV1]] :
    // CHECK-SAME:         !VPU.SparseTensor<
    // CHECK-SAME:              data=!VPU.DistributedTensor<1x128x14x14xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                    {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:              sparsity_map=!VPU.DistributedTensor<1x128x14x14xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                    {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x128x14x14xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                                 {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:                           sparsity_map=!VPU.DistributedTensor<1x128x14x14xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                                 {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // Concat
    // CHECK:       [[CONCAT_CMX:%.+]] = VPU.Concat([[DISTR_CAST0]], [[DISTR_CAST1]])
    // CHECK-SAME:      -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x256x14x14xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                                 {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:                           sparsity_map=!VPU.DistributedTensor<1x256x14x14xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                                 {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // Convolution 2
    // CHECK:       [[CONV2:%.+]] = VPU.NCE.Convolution([[CONCAT_CMX]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x128x6x6xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:                               sparsity_map=!VPU.DistributedTensor<1x128x6x6xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // CHECK:       [[COPY_OUT:%.+]] = VPU.Copy([[CONV2]])
    // CHECK-SAME:          -> !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {order = #NHWC}>, sparsity_map=tensor<1x128x6x6xi1, {order = #NHWC}>>

    // CHECK:       return [[COPY_OUT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ConcatInputsDistributed = !VPU.DistributedTensor<
    1x640x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!ConcatUserDistributed = !VPU.DistributedTensor<
    1x1280x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK-LABEL: func.func @ConcatUsersHaveDifferentOperandNums
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x640x16x16xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x640x16x16xf16, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<1x640x16x16xf16, {order = #NHWC}>, [[INPUT3:%.+]]: tensor<1x640x16x16xf16, {order = #NHWC}>, [[INPUT4:%.+]]: tensor<1x1280x16x16xf16, {order = #NHWC}>)
func.func @ConcatUsersHaveDifferentOperandNums(
            %input0: tensor<1x640x16x16xf16, {order = #NHWC}>,
            %input1: tensor<1x640x16x16xf16, {order = #NHWC}>,
            %input2: tensor<1x640x16x16xf16, {order = #NHWC}>,
            %input3: tensor<1x640x16x16xf16, {order = #NHWC}>,
            %input4: tensor<1x1280x16x16xf16, {order = #NHWC}>)
           -> (tensor<1x1280x16x16xf16, {order = #NHWC}>, tensor<1x1280x16x16xf16, {order = #NHWC}>) {
    // One input of Concat
    %0 = VPU.Copy(%input0) {out_mem_space = @CMX_NN} : tensor<1x640x16x16xf16, {order = #NHWC}> -> !ConcatInputsDistributed
    %1 = VPU.Copy(%input1) {out_mem_space = @CMX_NN} : tensor<1x640x16x16xf16, {order = #NHWC}> -> !ConcatInputsDistributed

    %2 = VPU.NCE.Eltwise(%0, %1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
            lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>
        } -> !ConcatInputsDistributed

    %3 = VPU.Copy(%2) : !ConcatInputsDistributed -> tensor<1x640x16x16xf16, {order = #NHWC}>

    // Another input of Concat
    %4 = VPU.Copy(%input2) {out_mem_space = @CMX_NN} : tensor<1x640x16x16xf16, {order = #NHWC}> -> !ConcatInputsDistributed
    %5 = VPU.Copy(%input3) {out_mem_space = @CMX_NN} : tensor<1x640x16x16xf16, {order = #NHWC}> -> !ConcatInputsDistributed

    %6 = VPU.NCE.Eltwise(%4, %5) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
            lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>
        } -> !ConcatInputsDistributed

    %7 = VPU.Copy(%6) : !ConcatInputsDistributed -> tensor<1x640x16x16xf16, {order = #NHWC}>

    // Concat
    %8 = VPU.Concat(%3, %7) {static_offsets = [[0, 0, 0, 0], [0, 640, 0, 0]]} :
        tensor<1x640x16x16xf16, {order = #NHWC}>,
        tensor<1x640x16x16xf16, {order = #NHWC}> -> tensor<1x1280x16x16xf16, {order = #NHWC}>

    // One user is MaxPool
    %9 = VPU.Copy(%8) {out_mem_space = @CMX_NN} : tensor<1x1280x16x16xf16, {order = #NHWC}> -> !ConcatUserDistributed

    %10 = VPU.NCE.MaxPool(%9) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            kernel_size = [1, 1],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
            lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]
        } -> !ConcatUserDistributed

    %11 = VPU.Copy(%10) : !ConcatUserDistributed -> tensor<1x1280x16x16xf16, {order = #NHWC}>

    // Another user is Eltwise
    %12 = VPU.Copy(%8) {out_mem_space = @CMX_NN} : tensor<1x1280x16x16xf16, {order = #NHWC}> -> !ConcatUserDistributed
    %13 = VPU.Copy(%input4) {out_mem_space = @CMX_NN} : tensor<1x1280x16x16xf16, {order = #NHWC}> -> !ConcatUserDistributed

    %14 = VPU.NCE.Eltwise(%13, %12) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
            lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>
        } -> !ConcatUserDistributed

    %15 = VPU.Copy(%14) : !ConcatUserDistributed -> tensor<1x1280x16x16xf16, {order = #NHWC}>

    return %11, %15 : tensor<1x1280x16x16xf16, {order = #NHWC}>, tensor<1x1280x16x16xf16, {order = #NHWC}>

    // CHECK:       [[COPY_IN_0:%.+]] = VPU.Copy([[INPUT0]])
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x640x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[COPY_IN_1:%.+]] = VPU.Copy([[INPUT1]])
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x640x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[ELTWISE_0:%.+]] = VPU.NCE.Eltwise([[COPY_IN_0]], [[COPY_IN_1]])
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x640x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[COPY_IN_2:%.+]] = VPU.Copy([[INPUT2]])
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x640x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[COPY_IN_3:%.+]] = VPU.Copy([[INPUT3]])
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x640x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[ELTWISE_1:%.+]] = VPU.NCE.Eltwise([[COPY_IN_2]], [[COPY_IN_3]])
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x640x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[ELTWISE_0]], [[ELTWISE_1]]) {
    // CHECK-SAME:                          static_offsets = [
    // CHECK-SAME:                              [0, 0, 0, 0],
    // CHECK-SAME:                              [0, 640, 0, 0]
    // CHECK-SAME:                          ]
    // CHECK-SAME:                      } : !VPU.DistributedTensor<1x640x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                          !VPU.DistributedTensor<1x640x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x1280x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[USER_MAX_POOL:%.+]] = VPU.NCE.MaxPool([[CONCAT]]) {
    // CHECK-SAME:                      kernel_size = [1, 1],
    // CHECK-SAME:                      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
    // CHECK-SAME:                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:                      strides = [1, 1]}
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x1280x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[COPY_OUT_0:%.+]] = VPU.Copy([[USER_MAX_POOL]])
    // CHECK-SAME:                      -> tensor<1x1280x16x16xf16, {order = #NHWC}>

    // CHECK:       [[COPY_IN_4:%.+]] = VPU.Copy([[INPUT4]])
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x1280x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[USER_ELTWISE:%.+]] = VPU.NCE.Eltwise([[COPY_IN_4]], [[CONCAT]]) {
    // CHECK-SAME:                      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:                      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
    // CHECK-SAME:                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>}
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x1280x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[COPY_OUT_1:%.+]] = VPU.Copy([[USER_ELTWISE]])
    // CHECK-SAME:                      -> tensor<1x1280x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[COPY_OUT_0]], [[COPY_OUT_1]] : tensor<1x1280x16x16xf16, {order = #NHWC}>, tensor<1x1280x16x16xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Distributed = !VPU.DistributedTensor<
    1x16x90x160xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>
!Distributed1 = !VPU.DistributedTensor<
    1x16x90x160xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!Distributed2 = !VPU.DistributedTensor<
    1x32x90x160xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
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
    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x90x160xf16, {order = #NHWC}> -> !Distributed

    %1 = VPU.NCE.MaxPool(%0, %maxPoolWeightsTable) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                strides = [1, 1],
                kernel_size = [1, 1]
            } -> !Distributed

    %2 = VPU.Copy(%1) : !Distributed -> tensor<1x16x90x160xf16, {order = #NHWC}>

    // Input 2 of Concat
    %3 = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x16x90x160xf16, {order = #NCHW}> -> !Distributed1

    %4 = VPU.NCE.Permute(%3) {
            dstElemType = f16,
            dstOrder = #NHWC,
            expandedChannels = 16 : i64,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
            lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>
        } -> !Distributed

    %5 = VPU.Copy(%4) : !Distributed -> tensor<1x16x90x160xf16, {order = #NHWC}>

    %6 = VPU.Concat(%2, %5) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x90x160xf16, {order = #NHWC}>, tensor<1x16x90x160xf16, {order = #NHWC}> -> tensor<1x32x90x160xf16, {order = #NHWC}>

    // Concat output
    %7 = VPU.Copy(%6) {out_mem_space = @CMX_NN} : tensor<1x32x90x160xf16, {order = #NHWC}> -> !Distributed2

    %8 = VPU.NCE.MaxPool(%7, %maxPoolWeightsTable1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                strides = [1, 1],
                kernel_size = [1, 1]
            } -> !Distributed2

    %9 = VPU.Copy(%8) : !Distributed2 -> tensor<1x32x90x160xf16, {order = #NHWC}>

    return %9 : tensor<1x32x90x160xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = dense<1> : tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:       [[CST_1:%.+]] = const.Declare tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = dense<1> : tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    // CHECK:       [[COPY_IN_0:%.+]] = VPU.Copy([[INPUT0]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x90x160xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[MAXPOOL_0:%.+]] = VPU.NCE.MaxPool([[COPY_IN_0]], [[CST]] ) {
    // CHECK-SAME:                  kernel_size = [1, 1],
    // CHECK-SAME:                  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                  strides = [1, 1]}
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x16x90x160xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[COPY_OUT_0:%.+]] = VPU.Copy([[MAXPOOL_0]])
    // CHECK-SAME:                       -> tensor<1x16x90x160xf16, {order = #NHWC}>

    // CHECK:       [[COPY_IN_1:%.+]] = VPU.Copy([[INPUT1]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x90x160xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[COPY_IN_1]]) {
    // CHECK-SAME:                  dstElemType = f16,
    // CHECK-SAME:                  dstOrder = #NHWC,
    // CHECK-SAME:                  expandedChannels = 16 : i64,
    // CHECK-SAME:                  ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
    // CHECK-SAME:                  lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x90x160xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

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
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x90x160xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[MAXPOOL_1:%.+]] = VPU.NCE.MaxPool([[COPY_IN_2]], [[CST_1]] ) {
    // CHECK-SAME:                  kernel_size = [1, 1],
    // CHECK-SAME:                  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                  strides = [1, 1]}
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x32x90x160xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[COPY_OUT_2:%.+]] = VPU.Copy([[MAXPOOL_1]])
    // CHECK-SAME:                        -> tensor<1x32x90x160xf16, {order = #NHWC}>

    // CHECK:       return [[COPY_OUT_2]] : tensor<1x32x90x160xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.13537439832500384:128>
!qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType2 = !quant.uniform<u8:f16, 0.12762966529995787:128>
!qElemType3 = !quant.uniform<u8:f16, 2.4627450980392158>

!Distributed = !VPU.DistributedTensor<
    1x32x104x104x!qElemType, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!Distributed1 = !VPU.DistributedTensor<
    32x32x3x3x!qElemType1, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>


!Distributed2 = !VPU.DistributedTensor<
    32x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!Distributed3 = !VPU.DistributedTensor<
    1x32x104x104x!qElemType2, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!Distributed4 = !VPU.DistributedTensor<
    32x32x3x3x!qElemType3, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!Distributed5 = !VPU.DistributedTensor<
    1x64x104x104x!qElemType2, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK-LABEL: @InsertAvgPoolingWhenNCEOpHasExtraUser
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x32x104x104x!qElemType, {order = #NHWC}>
func.func @InsertAvgPoolingWhenNCEOpHasExtraUser(%arg0: tensor<1x32x104x104x!qElemType, {order = #NHWC}>)
           -> tensor<1x64x104x104x!qElemType2, {order = #NHWC}> {
    %convWeights = const.Declare tensor<32x32x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %convWeightsTable = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %convWeights1 = const.Declare tensor<32x32x3x3x!qElemType3, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType3>, #const.Reorder<#NHWC>]
    %convWeightsTable1 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %maxPoolWeightsTable = const.Declare tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> =
        dense<1> : tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    // Input 1 of Concat
    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x32x104x104x!qElemType, {order = #NHWC}> -> !Distributed
    %1 = VPU.Copy(%convWeights) {out_mem_space = @CMX_NN} : tensor<32x32x3x3x!qElemType1, {order = #NHWC}> -> !Distributed1
    %2 = VPU.Copy(%convWeightsTable) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32> -> !Distributed2
    %3 = VPU.NCE.Convolution(%0, %1, %2) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = 0 : i64, clamp_high = 255 : i64,
                lrelu_mult = 1638 : i64, lrelu_shift = 14 : i64, fp_prelu_alpha = 0.0999755859375 : f64>,

                strides = [1, 1]
            } : !Distributed, !Distributed1, !Distributed2 -> !Distributed3
    %4 = VPU.Copy(%3) : !Distributed3 -> tensor<1x32x104x104x!qElemType2, {order = #NHWC}>

    // Input 2 of Concat
    %5 = VPU.Copy(%convWeights1) {out_mem_space = @CMX_NN} : tensor<32x32x3x3x!qElemType3, {order = #NHWC}> -> !Distributed4
    %6 = VPU.Copy(%convWeightsTable1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32> -> !Distributed2
    %7 = VPU.NCE.Convolution(%3, %5, %6) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -128 : i64, clamp_high = 127 : i64,
                lrelu_mult = 1638 : i64, lrelu_shift = 14 : i64, fp_prelu_alpha = 0.0999755859375 : f64>,

                strides = [1, 1]
            } : !Distributed3, !Distributed4, !Distributed2 -> !Distributed3
    %8 = VPU.Copy(%7) : !Distributed3 -> tensor<1x32x104x104x!qElemType2, {order = #NHWC}>

    %9 = VPU.Concat(%4, %8) {static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]]} : tensor<1x32x104x104x!qElemType2, {order = #NHWC}>, tensor<1x32x104x104x!qElemType2, {order = #NHWC}> -> tensor<1x64x104x104x!qElemType2, {order = #NHWC}>

    // Concat output
    %10 = VPU.Copy(%9) {out_mem_space = @CMX_NN} : tensor<1x64x104x104x!qElemType2, {order = #NHWC}> -> !Distributed5
    %11 = VPU.NCE.MaxPool(%10, %maxPoolWeightsTable) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -128 : i64, clamp_high = 127 : i64,
                lrelu_mult = 1638 : i64, lrelu_shift = 14 : i64, fp_prelu_alpha = 0.0999755859375 : f64>,
                strides = [1, 1],
                kernel_size = [1, 1]
            } -> !Distributed5
    %12 = VPU.Copy(%11) : !Distributed5 -> tensor<1x64x104x104x!qElemType2, {order = #NHWC}>

    return %12 : tensor<1x64x104x104x!qElemType2, {order = #NHWC}>
    // CHECK:       [[CST:%.+]] = const.Declare tensor<32x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]
    // CHECK:       [[CST_0:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    // CHECK:       [[CST_1:%.+]] = const.Declare tensor<32x32x3x3x!qElemType3, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType3>, #const.Reorder<#NHWC>]
    // CHECK:       [[CST_2:%.+]] = const.Declare tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = dense<1> : tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    // CHECK:       [[COPY_IN_0:%.+]] = VPU.Copy([[INPUT]])
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x32x104x104x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[COPY_IN_1:%.+]] = VPU.Copy([[CST]])
    // CHECK-SAME:                      -> !VPU.DistributedTensor<32x32x3x3x!qElemType2, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[COPY_IN_2:%.+]] = VPU.Copy([[CST_0]])
    // CHECK-SAME:                      -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[COPY_IN_0]], [[COPY_IN_1]], [[COPY_IN_2]]) rawFilterShape [32, 32, 3, 3] {
    // CHECK-SAME:                  pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:                  ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = 0 : i64, clamp_high = 255 : i64,
    // CHECK-SAME:                      lrelu_mult = 1638 : i64, lrelu_shift = 14 : i64, fp_prelu_alpha = 0.0999755859375 : f64>,
    // CHECK-SAME:                  strides = [1, 1]}
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x32x104x104x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[AVGPOOL:%.+]] = VPU.NCE.AveragePool([[CONV]]) {
    // CHECK-SAME:              kernel_size = [1, 1],
    // CHECK-SAME:              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:              ppe = #VPU.PPEInt<mode = <NOOP>,
    // CHECK-SAME:                  clamp_low = 0 : i64,
    // CHECK-SAME:                  lamp_high = 255 : i64,
    // CHECK-SAME:                  lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
    // CHECK-SAME:                  quant_mult = [16384], quant_shift = [14], quant_post_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:              strides = [1, 1]}
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x32x104x104x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[COPY_IN_3:%.+]] = VPU.Copy([[CST_1]])
    // CHECK-SAME:                      -> !VPU.DistributedTensor<32x32x3x3x!qElemType3, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[COPY_IN_4:%.+]] = VPU.Copy([[CST_0]])
    // CHECK-SAME:                      -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[CONV2:%.+]] = VPU.NCE.Convolution([[CONV]], [[COPY_IN_3]], [[COPY_IN_4]]) rawFilterShape [32, 32, 3, 3] {
    // CHECK-SAME:                          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:                          ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -128 : i64, clamp_high = 127 : i64,
    // CHECK-SAME:                              lrelu_mult = 1638 : i64, lrelu_shift = 14 : i64, fp_prelu_alpha = 0.0999755859375 : f64>,
    // CHECK-SAME:                          strides = [1, 1]}
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x32x104x104x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[CMX_CONCAT:%.+]] = VPU.Concat([[AVGPOOL]], [[CONV2]]) {
    // CHECK-SAME:          static_offsets = [
    // CHECK-SAME:              [0, 0, 0, 0],
    // CHECK-SAME:              [0, 32, 0, 0]
    // CHECK-SAME:          ]} :
    // CHECK-SAME:              !VPU.DistributedTensor<1x32x104x104x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:              !VPU.DistributedTensor<1x32x104x104x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x64x104x104x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[CMX_CONCAT]], [[CST_2]] ) {
    // CHECK-SAME:                          kernel_size = [1, 1],
    // CHECK-SAME:                          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                          strides = [1, 1]}
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x64x104x104x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[COPY_OUT:%.+]] = VPU.Copy([[MAXPOOL]])
    // CHECK-SAME:                      -> tensor<1x64x104x104x!qElemType1, {order = #NHWC}>

    // CHECK:       return [[COPY_OUT]] : tensor<1x64x104x104x!qElemType1, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0057493812897626093:128>
!qElemType1 = !quant.uniform<u8:f16, 0.0087224820080925441:128>
!qElemType2 = !quant.uniform<u8:f16, 0.0088351567586263026:128>
!qElemType3 = !quant.uniform<u8:f16, 0.0033942965900196748:128>
!qElemType4 = !quant.uniform<u8:f16, 0.0020254650536705465:128>

!Distributed = !VPU.DistributedTensor<
    1x32x128x128x!qElemType, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!Distributed1 = !VPU.DistributedTensor<
    1x32x128x128x!qElemType1, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!Distributed2 = !VPU.DistributedTensor<
    1x32x128x128x!qElemType2, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!Distributed3 = !VPU.DistributedTensor<
    1x16x128x128x!qElemType2, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!Distributed4 = !VPU.DistributedTensor<
    1x48x128x128x!qElemType2, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!Distributed5 = !VPU.DistributedTensor<
    1x64x128x128x!qElemType2, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK-LABEL: @InsertAvgPoolingInCaseCopyOpHasExtraUser
// CHECK-SAME:  [[INPUT0:%.+]]: !VPU.DistributedTensor<1x32x128x128x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK-SAME:  [[INPUT1:%.+]]: !VPU.DistributedTensor<1x32x128x128x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
func.func @InsertAvgPoolingInCaseCopyOpHasExtraUser(%arg0: !Distributed, %arg1: !Distributed1)
                                                    -> tensor<1x64x128x128x!qElemType2, {order = #NHWC}> {
    %convWeights = const.Declare tensor<16x32x3x3x!qElemType3, {order = #NHWC}> = dense<1.0> : tensor<16x32x3x3xf32>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType3>, #const.Reorder<#NHWC>]
    %convWeightsTable = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %convWeights1 = const.Declare tensor<16x48x3x3x!qElemType4, {order = #NHWC}> = dense<1.0> : tensor<16x48x3x3xf32>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType4>, #const.Reorder<#NHWC>]
    %convWeightsTable1 = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // Input 1 of Concat_0, Input 1 of Concat_1
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                op_type = #VPU.eltwise_type<ADD>,
                ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64,
                lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
                quant_mult = [28975], quant_shift = [30], quant_post_shift = 0 : i64,
                in1_quant_mult = [24114], in2_quant_mult = [36584], fp_prelu_alpha = 1.000000e+00 : f64>
            } -> !Distributed2
    %1 = VPU.Copy(%0) : !Distributed2 -> tensor<1x32x128x128x!qElemType2, {order = #NHWC}>

    // Input 2 of Concat_0, Input 2 of Concat_1
    %2 = VPU.Copy(%convWeights) {out_mem_space = @CMX_NN} : tensor<16x32x3x3x!qElemType3, {order = #NHWC}>
             -> !VPU.DistributedTensor<16x32x3x3x!qElemType3, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %3 = VPU.Copy(%convWeightsTable) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32>
             -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %4 = VPU.NCE.Convolution(%0, %2, %3) rawFilterShape [16, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = 0 : i64, clamp_high = 255 : i64,
                lrelu_mult = 1638 : i64, lrelu_shift = 13 : i64, fp_prelu_alpha = 0.20000000298023224 : f64>,
                 strides = [1, 1]
            } : !Distributed2, !VPU.DistributedTensor<16x32x3x3x!qElemType3, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> -> !Distributed3
    %5 = VPU.Copy(%4) : !Distributed3 -> tensor<1x16x128x128x!qElemType2, {order = #NHWC}>

    // DDR Concat_0
    %6 = VPU.Concat(%1, %5) {static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]]} : tensor<1x32x128x128x!qElemType2, {order = #NHWC}>, tensor<1x16x128x128x!qElemType2, {order = #NHWC}> -> tensor<1x48x128x128x!qElemType2, {order = #NHWC}>

    // Input 3 of Concat_1
    %7 = VPU.Copy(%6) {out_mem_space = @CMX_NN} : tensor<1x48x128x128x!qElemType2, {order = #NHWC}> -> !Distributed4
    %8 = VPU.Copy(%convWeights1) {out_mem_space = @CMX_NN} : tensor<16x48x3x3x!qElemType4, {order = #NHWC}>
             -> !VPU.DistributedTensor<16x48x3x3x!qElemType4, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %9 = VPU.Copy(%convWeightsTable1) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32>
             -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %10 = VPU.NCE.Convolution(%7, %8, %9) rawFilterShape [16, 48, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = 0 : i64, clamp_high = 255 : i64,
                lrelu_mult = 1638 : i64, lrelu_shift = 13 : i64, fp_prelu_alpha = 0.20000000298023224 : f64>,
                 strides = [1, 1]
            } : !Distributed4, !VPU.DistributedTensor<16x48x3x3x!qElemType4, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> -> !Distributed3
    %11 = VPU.Copy(%10) : !Distributed3 -> tensor<1x16x128x128x!qElemType2, {order = #NHWC}>

    // DDR Concat_1
    %12 = VPU.Concat(%1, %5, %11) {static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]]} : tensor<1x32x128x128x!qElemType2, {order = #NHWC}>, tensor<1x16x128x128x!qElemType2, {order = #NHWC}>, tensor<1x16x128x128x!qElemType2, {order = #NHWC}> -> tensor<1x64x128x128x!qElemType2, {order = #NHWC}>

    // Output
    %13 = VPU.Copy(%12) {out_mem_space = @CMX_NN} : tensor<1x64x128x128x!qElemType2, {order = #NHWC}> -> !Distributed5
    %14 = VPU.NCE.MaxPool(%13) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -128 : i64, clamp_high = 127 : i64,
                lrelu_mult = 1638 : i64, lrelu_shift = 14 : i64, fp_prelu_alpha = 0.0999755859375 : f64>,
                strides = [1, 1],
                kernel_size = [1, 1]
            } -> !Distributed5
    %15 = VPU.Copy(%14) : !Distributed5 -> tensor<1x64x128x128x!qElemType2, {order = #NHWC}>

    return %15 : tensor<1x64x128x128x!qElemType2, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x32x3x3x!qElemType3, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x3x3xf32>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType3>, #const.Reorder<#NHWC>]
    // CHECK:       [[CST_0:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK:       [[CST_1:%.+]] = const.Declare tensor<16x48x3x3x!qElemType4, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x48x3x3xf32>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType4>, #const.Reorder<#NHWC>]

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT0]], [[INPUT1]]) {
    // CHECK-SAME:                  op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64,
    // CHECK-SAME:                  lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [28975], quant_shift = [30], quant_post_shift = 0 : i64,
    // CHECK-SAME:                  in1_quant_mult = [24114], in2_quant_mult = [36584], fp_prelu_alpha = 1.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>}
    // CHECK-SAME:                        -> !VPU.DistributedTensor<1x32x128x128x!qElemType2, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[AVGPOOL_0:%.+]] = VPU.NCE.AveragePool([[ELTWISE]]) {
    // CHECK-SAME:                  kernel_size = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>, activation_zp = 128 : i64>,
    // CHECK-SAME:                  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                  ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
    // CHECK-SAME:                  quant_scale = [1.000000e+00], quant_mult = [16384], quant_shift = [14], quant_post_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:                  strides = [1, 1]}
    // CHECK-SAME:                        -> !VPU.DistributedTensor<1x32x128x128x!qElemType2, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[AVGPOOL_1:%.+]] = VPU.NCE.AveragePool([[ELTWISE]]) {
    // CHECK-SAME:                  kernel_size = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>, activation_zp = 128 : i64>,
    // CHECK-SAME:                  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                  ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
    // CHECK-SAME:                  quant_scale = [1.000000e+00], quant_mult = [16384], quant_shift = [14], quant_post_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:                  strides = [1, 1]}
    // CHECK-SAME:                        -> !VPU.DistributedTensor<1x32x128x128x!qElemType2, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[COPY_IN_0:%.+]] = VPU.Copy([[CST]])
    // CHECK-SAME:                        -> !VPU.DistributedTensor<16x32x3x3x!qElemType3, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[COPY_IN_1:%.+]] = VPU.Copy([[CST_0]])
    // CHECK-SAME:                        -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[CONV_0:%.+]] = VPU.NCE.Convolution([[ELTWISE]], [[COPY_IN_0]], [[COPY_IN_1]]) rawFilterShape [16, 32, 3, 3] {
    // CHECK-SAME:                  pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:                  ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = 0 : i64, clamp_high = 255 : i64,
    // CHECK-SAME:                  lrelu_mult = 1638 : i64, lrelu_shift = 13 : i64, fp_prelu_alpha = 0.20000000298023224 : f64>,
    // CHECK-SAME:                  strides = [1, 1]}
    // CHECK-SAME:                        -> !VPU.DistributedTensor<1x16x128x128x!qElemType2, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK-DAG:   [[AVGPOOL_2:%.+]] = VPU.NCE.AveragePool([[CONV_0]]) {
    // CHECK-SAME:                  kernel_size = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>, activation_zp = 128 : i64>,
    // CHECK-SAME:                  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                  ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
    // CHECK-SAME:                  quant_scale = [1.000000e+00], quant_mult = [16384], quant_shift = [14], quant_post_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:                  strides = [1, 1]}
    // CHECK-SAME:                        -> !VPU.DistributedTensor<1x16x128x128x!qElemType2, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK-DAG:   [[AVGPOOL_3:%.+]] = VPU.NCE.AveragePool([[CONV_0]]) {
    // CHECK-SAME:                  kernel_size = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>, activation_zp = 128 : i64>,
    // CHECK-SAME:                  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                  ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
    // CHECK-SAME:                  quant_scale = [1.000000e+00], quant_mult = [16384], quant_shift = [14], quant_post_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:                  strides = [1, 1]}
    // CHECK-SAME:                        -> !VPU.DistributedTensor<1x16x128x128x!qElemType2, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[CMX_CONCAT_0:%.+]] = VPU.Concat([[AVGPOOL_1]], [[AVGPOOL_3]]) {
    // CHECK-SAME:          static_offsets = [
    // CHECK-SAME:              [0, 0, 0, 0],
    // CHECK-SAME:              [0, 32, 0, 0]
    // CHECK-SAME:          ]} :
    // CHECK-SAME:              !VPU.DistributedTensor<1x32x128x128x!qElemType2, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:              !VPU.DistributedTensor<1x16x128x128x!qElemType2, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x48x128x128x!qElemType2, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[COPY_IN_2:%.+]] = VPU.Copy([[CST_1]])
    // CHECK-SAME:                        -> !VPU.DistributedTensor<16x48x3x3x!qElemType4, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[COPY_IN_3:%.+]] = VPU.Copy([[CST_0]])
    // CHECK-SAME:                        -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[CONV_1:%.+]] = VPU.NCE.Convolution([[CMX_CONCAT_0]], [[COPY_IN_2]], [[COPY_IN_3]]) rawFilterShape [16, 48, 3, 3] {
    // CHECK-SAME:                  pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:                  ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = 0 : i64, clamp_high = 255 : i64,
    // CHECK-SAME:                  lrelu_mult = 1638 : i64, lrelu_shift = 13 : i64, fp_prelu_alpha = 0.20000000298023224 : f64>,
    // CHECK-SAME:                  strides = [1, 1]}
    // CHECK-SAME:                        -> !VPU.DistributedTensor<1x16x128x128x!qElemType2, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[CMX_CONCAT_1:%.+]] = VPU.Concat([[AVGPOOL_0]], [[AVGPOOL_2]], [[CONV_1]]) {
    // CHECK-SAME:          static_offsets = [
    // CHECK-SAME:              [0, 0, 0, 0],
    // CHECK-SAME:              [0, 32, 0, 0],
    // CHECK-SAME:              [0, 48, 0, 0]
    // CHECK-SAME:          ]} :
    // CHECK-SAME:              !VPU.DistributedTensor<1x32x128x128x!qElemType2, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:              !VPU.DistributedTensor<1x16x128x128x!qElemType2, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:              !VPU.DistributedTensor<1x16x128x128x!qElemType2, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x64x128x128x!qElemType2, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[CMX_CONCAT_1]]) {
    // CHECK-SAME:                  kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                  ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -128 : i64, clamp_high = 127 : i64, lrelu_mult = 1638 : i64,
    // CHECK-SAME:                  lrelu_shift = 14 : i64, fp_prelu_alpha = 0.0999755859375 : f64>,
    // CHECK-SAME:                  strides = [1, 1]}
    // CHECK-SAME:                        -> !VPU.DistributedTensor<1x64x128x128x!qElemType2, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[COPY_OUT:%.+]] = VPU.Copy([[MAXPOOL]])
    // CHECK-SAME:                        -> tensor<1x64x128x128x!qElemType2, {order = #NHWC}>

    // CHECK:       return [[COPY_OUT]] : tensor<1x64x128x128x!qElemType2, {order = #NHWC}>
}
