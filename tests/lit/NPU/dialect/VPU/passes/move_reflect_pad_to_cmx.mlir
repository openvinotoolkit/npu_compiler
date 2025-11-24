
//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --move-reflect-pad-to-cmx %s | FileCheck %s
// REQUIRES: arch-NPU40XX

!qElemType = !quant.uniform<u8:f16, 0.0038406767097173954>
!qElemType1 = !quant.uniform<u8:f16, 0.0076813534194347909>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @moveReflectPadToCmxW
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x32x32x3x!qElemType, {order = #NHWC}>
func.func @moveReflectPadToCmxW(%arg0: tensor<1x32x32x3x!qElemType, {order = #NHWC}>) -> tensor<1x32x32x5x!qElemType1, {order = #NHWC}> {
    %copy_0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x32x32x3x!qElemType, {order = #NHWC}> -> tensor<1x32x32x3x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>
    %input_copy_to_ddr = VPU.Copy(%copy_0) : tensor<1x32x32x3x!qElemType, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x32x3x!qElemType, {order = #NHWC}>

    %quantize_cast = VPU.QuantizeCast(%input_copy_to_ddr) {dstElemType = !qElemType1} : tensor<1x32x32x3x!qElemType, {order = #NHWC}> -> tensor<1x32x32x3x!qElemType1, {order = #NHWC}>

    %input_pad_0 = VPU.Slice %quantize_cast [0, 0, 0, 1] [1, 32, 32, 1]: tensor<1x32x32x3x!qElemType1, {order = #NHWC}>
                                to tensor<1x32x32x1x!qElemType1, {order = #NHWC}>
    %input_pad_1 = VPU.Slice %quantize_cast [0, 0, 0, 2] [1, 32, 32, 1]: tensor<1x32x32x3x!qElemType1, {order = #NHWC}>
                                to tensor<1x32x32x1x!qElemType1, {order = #NHWC}>

    %concat_view = VPU.Concat (%input_pad_0, %quantize_cast, %input_pad_1)
                            {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 4]]} :
                                    tensor<1x32x32x1x!qElemType1, {order = #NHWC}>,
                                    tensor<1x32x32x3x!qElemType1, {order = #NHWC}>,
                                    tensor<1x32x32x1x!qElemType1, {order = #NHWC}>
                            -> tensor<1x32x32x5x!qElemType1, {order = #NHWC}>

    return %concat_view: tensor<1x32x32x5x!qElemType1, {order = #NHWC}>

    // CHECK:   [[ARG_COPY:%.+]] = VPU.Copy([[INPUT]]) {out_mem_space = @CMX_NN} : tensor<1x32x32x3x!qElemType, {order = #NHWC}> -> tensor<1x32x32x3x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:   [[CMX_TO_DDR_COPY:%.+]] = VPU.Copy([[ARG_COPY]]) : tensor<1x32x32x3x!qElemType, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x32x3x!qElemType, {order = #NHWC}>
    // CHECK:   [[QUANT_CAST:%.+]] = VPU.QuantizeCast([[CMX_TO_DDR_COPY]]) {dstElemType = !qElemType1} : tensor<1x32x32x3x!qElemType, {order = #NHWC}> -> tensor<1x32x32x3x!qElemType1, {order = #NHWC}>
    // CHECK:   [[DDR_TO_CMX_COPY:%.+]] = VPU.Copy([[QUANT_CAST]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x32x3x!qElemType1, {order = #NHWC}> -> tensor<1x32x32x3x!qElemType1, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:   [[INPUT_SLICE_0:%.+]] = VPU.Slice [[DDR_TO_CMX_COPY]] [0, 0, 0, 1] [1, 32, 32, 1] : tensor<1x32x32x3x!qElemType1, {mem_space = [@CMX_NN, 0], order = #NHWC}> to tensor<1x32x32x1x!qElemType1, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:   [[INPUT_SLICE_1:%.+]] = VPU.Slice [[DDR_TO_CMX_COPY]] [0, 0, 0, 2] [1, 32, 32, 1] : tensor<1x32x32x3x!qElemType1, {mem_space = [@CMX_NN, 0], order = #NHWC}> to tensor<1x32x32x1x!qElemType1, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:   [[CONCAT:%.+]] = VPU.Concat([[INPUT_SLICE_0]], [[DDR_TO_CMX_COPY]], [[INPUT_SLICE_1]])
    // CHECK-SAME(LITERAL):     {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 4]]}
    // CHECK:   tensor<1x32x32x1x!qElemType1, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x32x32x3x!qElemType1, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x32x32x1x!qElemType1, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x32x32x5x!qElemType1, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:   [[CONCAT_OUT_COPY:%.+]] = VPU.Copy([[CONCAT]]) : tensor<1x32x32x5x!qElemType1, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x32x32x5x!qElemType1, {order = #NHWC}>
    // CHECK:   return [[CONCAT_OUT_COPY]] : tensor<1x32x32x5x!qElemType1, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0038406767097173954>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @moveReflectPadToCmxH
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x32x3x5x!qElemType, {order = #NHWC}>)
func.func @moveReflectPadToCmxH(%arg0: tensor<1x32x3x5x!qElemType, {order = #NHWC}>) -> tensor<1x32x5x5x!qElemType, {order = #NHWC}> {
    %copy_0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x32x3x5x!qElemType, {order = #NHWC}> -> tensor<1x32x3x5x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>
    %input_copy_to_ddr = VPU.Copy(%copy_0) : tensor<1x32x3x5x!qElemType, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x3x5x!qElemType, {order = #NHWC}>

    %input_pad_0 = VPU.Slice %input_copy_to_ddr [0, 0, 1, 0] [1, 32, 1, 5]: tensor<1x32x3x5x!qElemType, {order = #NHWC}>
                                to tensor<1x32x1x5x!qElemType, {order = #NHWC}>
    %input_pad_1 = VPU.Slice %input_copy_to_ddr [0, 0, 2, 0] [1, 32, 1, 5]: tensor<1x32x3x5x!qElemType, {order = #NHWC}>
                                to tensor<1x32x1x5x!qElemType, {order = #NHWC}>
    %permute_cast_0 = VPU.PermuteCast(%input_pad_0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x32x1x5x!qElemType, {order = #NHWC}> -> tensor<1x32x1x5x!qElemType, {order = #NHWC}>
    %permute_cast_1 = VPU.PermuteCast(%input_pad_1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x32x1x5x!qElemType, {order = #NHWC}> -> tensor<1x32x1x5x!qElemType, {order = #NHWC}>

    %copy_1 = VPU.Copy(%input_copy_to_ddr) {out_mem_space = @CMX_NN} : tensor<1x32x3x5x!qElemType, {order = #NHWC}> -> tensor<1x32x3x5x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>
    %max_pool = VPU.NCE.MaxPool (%copy_1) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.0 : f64>, strides = [1, 1]} -> !VPU.DistributedTensor<1x32x3x5x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 1, 5], [1, 32, 1, 5], [1, 32, 1, 5]], compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0]], memory_shapes = [[1, 32, 1, 5], [1, 32, 1, 5], [1, 32, 1, 5]], memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0]]}>
    %copy_2 = VPU.Copy(%max_pool) : !VPU.DistributedTensor<1x32x3x5x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 1, 5], [1, 32, 1, 5], [1, 32, 1, 5]], compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0]], memory_shapes = [[1, 32, 1, 5], [1, 32, 1, 5], [1, 32, 1, 5]], memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0]]}> -> tensor<1x32x3x5x!qElemType, {order = #NHWC}>

    %concat_view = VPU.Concat (%permute_cast_0, %copy_2, %permute_cast_1)
                                {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 4, 0]]} :
                                    tensor<1x32x1x5x!qElemType, {order = #NHWC}>,
                                    tensor<1x32x3x5x!qElemType, {order = #NHWC}>,
                                    tensor<1x32x1x5x!qElemType, {order = #NHWC}>
                                ->  tensor<1x32x5x5x!qElemType, {order = #NHWC}>

    return %concat_view: tensor<1x32x5x5x!qElemType, {order = #NHWC}>

    // CHECK:   [[ARG_COPY:%.+]] = VPU.Copy([[INPUT]]) {out_mem_space = @CMX_NN} : tensor<1x32x3x5x!qElemType, {order = #NHWC}> -> tensor<1x32x3x5x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:   [[CMX_TO_DDR_COPY:%.+]] = VPU.Copy([[ARG_COPY]]) : tensor<1x32x3x5x!qElemType, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x3x5x!qElemType, {order = #NHWC}>
    // CHECK:   [[DDR_TO_CMX_COPY:%.+]] = VPU.Copy([[CMX_TO_DDR_COPY]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x3x5x!qElemType, {order = #NHWC}> -> tensor<1x32x3x5x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:   [[INPUT_SLICE_0:%.+]] = VPU.Slice [[DDR_TO_CMX_COPY]] [0, 0, 1, 0] [1, 32, 1, 5] : tensor<1x32x3x5x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}> to tensor<1x32x1x5x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:   [[INPUT_SLICE_1:%.+]] = VPU.Slice [[DDR_TO_CMX_COPY]] [0, 0, 2, 0] [1, 32, 1, 5] : tensor<1x32x3x5x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}> to tensor<1x32x1x5x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:   [[PERMUTE_CAST_0:%.+]] = VPU.PermuteCast([[INPUT_SLICE_0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x32x1x5x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x32x1x5x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:   [[PERMUTE_CAST_1:%.+]] = VPU.PermuteCast([[INPUT_SLICE_1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x32x1x5x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x32x1x5x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:   [[COPY_0:%.+]] = VPU.Copy([[CMX_TO_DDR_COPY]]) {out_mem_space = @CMX_NN} : tensor<1x32x3x5x!qElemType, {order = #NHWC}> -> tensor<1x32x3x5x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:   [[MAX_POOL:%.+]] = VPU.NCE.MaxPool([[COPY_0]]) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} -> !VPU.DistributedTensor<1x32x3x5x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 32, 1, 5], [1, 32, 1, 5], [1, 32, 1, 5]], compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0]], memory_shapes = [[1, 32, 1, 5], [1, 32, 1, 5], [1, 32, 1, 5]], memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0]]}>
    // CHECK:   [[COPY_1:%.+]] = VPU.Copy([[MAX_POOL]]) : !VPU.DistributedTensor<1x32x3x5x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 32, 1, 5], [1, 32, 1, 5], [1, 32, 1, 5]], compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0]], memory_shapes = [[1, 32, 1, 5], [1, 32, 1, 5], [1, 32, 1, 5]], memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0]]}> -> tensor<1x32x3x5x!qElemType, {order = #NHWC}>
    // CHECK:   [[INPUT_COPY_TO_CMX:%.+]] = VPU.Copy([[COPY_1]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x3x5x!qElemType, {order = #NHWC}> -> tensor<1x32x3x5x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:   [[CONCAT:%.+]] = VPU.Concat([[PERMUTE_CAST_0]], [[INPUT_COPY_TO_CMX]], [[PERMUTE_CAST_1]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 4, 0]]}
    // CHECK:   tensor<1x32x1x5x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x32x3x5x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x32x1x5x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x32x5x5x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:   [[CONCAT_OUT_COPY:%.+]] = VPU.Copy([[CONCAT]]) : tensor<1x32x5x5x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x32x5x5x!qElemType, {order = #NHWC}>
    // CHECK:   return [[CONCAT_OUT_COPY]] : tensor<1x32x5x5x!qElemType, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0038406767097173954>
!qElemType1 = !quant.uniform<u8:f16, 0.0076813534194347909>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @dontMoveReflectPadToCmxNoCmxToDdrCopy
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x32x32x3x!qElemType, {order = #NHWC}>
func.func @dontMoveReflectPadToCmxNoCmxToDdrCopy(%arg0: tensor<1x32x32x3x!qElemType, {order = #NHWC}>) -> tensor<1x32x32x5x!qElemType1, {order = #NHWC}> {
    %copy_0 = VPU.Copy(%arg0)  : tensor<1x32x32x3x!qElemType, {order = #NHWC}> -> tensor<1x32x32x3x!qElemType, {order = #NHWC}>

    %quantize_cast = VPU.QuantizeCast(%copy_0) {dstElemType = !qElemType1} : tensor<1x32x32x3x!qElemType, {order = #NHWC}> -> tensor<1x32x32x3x!qElemType1, {order = #NHWC}>

    %input_pad_0 = VPU.Slice %quantize_cast [0, 0, 0, 1] [1, 32, 32, 1]: tensor<1x32x32x3x!qElemType1, {order = #NHWC}>
                                to tensor<1x32x32x1x!qElemType1, {order = #NHWC}>
    %input_pad_1 = VPU.Slice %quantize_cast [0, 0, 0, 2] [1, 32, 32, 1]: tensor<1x32x32x3x!qElemType1, {order = #NHWC}>
                                to tensor<1x32x32x1x!qElemType1, {order = #NHWC}>

    %concat_view = VPU.Concat (%input_pad_0, %quantize_cast, %input_pad_1)
                            {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 4]]} :
                                    tensor<1x32x32x1x!qElemType1, {order = #NHWC}>,
                                    tensor<1x32x32x3x!qElemType1, {order = #NHWC}>,
                                    tensor<1x32x32x1x!qElemType1, {order = #NHWC}>
                            -> tensor<1x32x32x5x!qElemType1, {order = #NHWC}>

    return %concat_view: tensor<1x32x32x5x!qElemType1, {order = #NHWC}>

    // CHECK:   VPU.Concat
    // CHECK-NOT:   tensor<1x32x32x1x!qElemType1, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x32x32x32x!qElemType1, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x32x32x1x!qElemType1, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x32x32x34x!qElemType1, {mem_space = [@CMX_NN, 0], order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0038406767097173954>
!qElemType1 = !quant.uniform<u8:f16, 0.0076813534194347909>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @dontMoveReflectPadToCmxPaddingWithMoreThan1
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x32x32x3x!qElemType, {order = #NHWC}>
func.func @dontMoveReflectPadToCmxPaddingWithMoreThan1(%arg0: tensor<1x32x32x3x!qElemType, {order = #NHWC}>) -> tensor<1x32x32x6x!qElemType1, {order = #NHWC}> {
    %copy_0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x32x32x3x!qElemType, {order = #NHWC}> -> tensor<1x32x32x3x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>
    %input_copy_to_ddr = VPU.Copy(%copy_0) : tensor<1x32x32x3x!qElemType, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x32x3x!qElemType, {order = #NHWC}>

    %quantize_cast = VPU.QuantizeCast(%input_copy_to_ddr) {dstElemType = !qElemType1} : tensor<1x32x32x3x!qElemType, {order = #NHWC}> -> tensor<1x32x32x3x!qElemType1, {order = #NHWC}>

    %input_pad_0 = VPU.Slice %quantize_cast [0, 0, 0, 0] [1, 32, 32, 1]: tensor<1x32x32x3x!qElemType1, {order = #NHWC}>
                                to tensor<1x32x32x1x!qElemType1, {order = #NHWC}>
    %input_pad_1 = VPU.Slice %quantize_cast [0, 0, 0, 1] [1, 32, 32, 2]: tensor<1x32x32x3x!qElemType1, {order = #NHWC}>
                                to tensor<1x32x32x2x!qElemType1, {order = #NHWC}>

    %concat_view = VPU.Concat (%input_pad_0, %quantize_cast, %input_pad_1)
                            {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 4]]} :
                                    tensor<1x32x32x1x!qElemType1, {order = #NHWC}>,
                                    tensor<1x32x32x3x!qElemType1, {order = #NHWC}>,
                                    tensor<1x32x32x2x!qElemType1, {order = #NHWC}>
                            -> tensor<1x32x32x6x!qElemType1, {order = #NHWC}>

    return %concat_view: tensor<1x32x32x6x!qElemType1, {order = #NHWC}>

    // CHECK:   VPU.Concat
    // CHECK-NOT:   tensor<1x32x32x1x!qElemType1, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x32x32x32x!qElemType1, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x32x32x1x!qElemType1, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x32x32x34x!qElemType1, {mem_space = [@CMX_NN, 0], order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0038406767097173954>
!qElemType1 = !quant.uniform<u8:f16, 0.0076813534194347909>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @dontMoveReflectPadToCmxDoesntFitInCmx
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x1024x1024x3x!qElemType, {order = #NHWC}>
func.func @dontMoveReflectPadToCmxDoesntFitInCmx(%arg0: tensor<1x1024x1024x3x!qElemType, {order = #NHWC}>) -> tensor<1x1024x1024x5x!qElemType1, {order = #NHWC}> {
    %copy_0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x1024x1024x3x!qElemType, {order = #NHWC}> -> tensor<1x1024x1024x3x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>
    %input_copy_to_ddr = VPU.Copy(%copy_0) : tensor<1x1024x1024x3x!qElemType, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x1024x1024x3x!qElemType, {order = #NHWC}>

    %quantize_cast = VPU.QuantizeCast(%input_copy_to_ddr) {dstElemType = !qElemType1} : tensor<1x1024x1024x3x!qElemType, {order = #NHWC}> -> tensor<1x1024x1024x3x!qElemType1, {order = #NHWC}>

    %input_pad_0 = VPU.Slice %quantize_cast [0, 0, 0, 1] [1, 1024, 1024, 1]: tensor<1x1024x1024x3x!qElemType1, {order = #NHWC}>
                                to tensor<1x1024x1024x1x!qElemType1, {order = #NHWC}>
    %input_pad_1 = VPU.Slice %quantize_cast [0, 0, 0, 2] [1, 1024, 1024, 1]: tensor<1x1024x1024x3x!qElemType1, {order = #NHWC}>
                                to tensor<1x1024x1024x1x!qElemType1, {order = #NHWC}>

    %concat_view = VPU.Concat (%input_pad_0, %quantize_cast, %input_pad_1)
                            {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 4]]} :
                                    tensor<1x1024x1024x1x!qElemType1, {order = #NHWC}>,
                                    tensor<1x1024x1024x3x!qElemType1, {order = #NHWC}>,
                                    tensor<1x1024x1024x1x!qElemType1, {order = #NHWC}>
                            -> tensor<1x1024x1024x5x!qElemType1, {order = #NHWC}>

    return %concat_view: tensor<1x1024x1024x5x!qElemType1, {order = #NHWC}>

    // CHECK:   VPU.Concat
    // CHECK-NOT:   tensor<1x1024x1024x1x!qElemType1, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x1024x1024x3x!qElemType1, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x1024x1024x1x!qElemType1, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x1024x1024x5x!qElemType1, {mem_space = [@CMX_NN, 0], order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @dontMoveReflectPadToCmxNotTheSameCopySource
func.func @dontMoveReflectPadToCmxNotTheSameCopySource(%arg0: tensor<1x1x1x24xf16>, %arg1: tensor<1x16x1x24xf16, {order = #NHWC}>) -> tensor<1x1x2x24xf16> {
    %copy_0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x1x1x24xf16> -> tensor<1x1x1x24xf16, {mem_space = @CMX_NN, order = #NCHW}>
    %copy_1 = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x16x1x24xf16, {order = #NHWC}> -> tensor<1x16x1x24xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %copy_cmx_to_ddr_0 = VPU.Copy(%copy_0) : tensor<1x1x1x24xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x1x24xf16>
    %copy_cmx_to_ddr_1 = VPU.Copy(%copy_1) : tensor<1x16x1x24xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x1x24xf16, {order = #NHWC}>

    %slice = VPU.Slice %copy_cmx_to_ddr_1 [0, 0, 0, 0] [1, 1, 1, 24] : tensor<1x16x1x24xf16, {order = #NHWC}> to tensor<1x1x1x24xf16, {order = #NHWC}>
    %permute_cast = VPU.PermuteCast(%slice) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x1x24xf16, {order = #NHWC}> -> tensor<1x1x1x24xf16>

    %concat = VPU.Concat(%copy_cmx_to_ddr_0, %permute_cast) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0]]} : tensor<1x1x1x24xf16>, tensor<1x1x1x24xf16> -> tensor<1x1x2x24xf16>

    return %concat: tensor<1x1x2x24xf16>

    // CHECK:   VPU.Concat
    // CHECK-NOT:    tensor<1x1x1x24xf16, {mem_space = [@CMX_NN, 0]}>, tensor<1x1x1x24xf16, {mem_space = [@CMX_NN, 0]}> -> tensor<1x1x2x24xf16, {mem_space = [@CMX_NN, 0]}>
}
