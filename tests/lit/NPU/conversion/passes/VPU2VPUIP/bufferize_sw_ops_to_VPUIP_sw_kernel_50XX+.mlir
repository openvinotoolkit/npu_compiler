//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: platform-NPU5010

// CHECK: func.func nested @cache_prefetch() attributes {VPU.task_type = @CACHE_PREFETCH}

// CHECK-LABEL:  func.func @StridedSliceLevel3StrideToSWKernel
// CHECK-SAME:      ([[ARG:%.+]]: memref<3x40x40x15xf16>)
func.func @StridedSliceLevel3StrideToSWKernel(%input: tensor<3x40x40x15xf16>) -> tensor<3x30x30x4xf16> {
    %output = VPU.StridedSlice(%input) {
        begin_mask = [0, 0, 0, 0],
        begins_attr = [0, 10, 10, 5],
        ellipsis_mask = [0, 0, 0, 0],
        end_mask = [0, 0, 0, 0],
        ends_attr = [3, 40, 40, 15],
        new_axis_mask = [0, 0, 0, 0],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        shrink_axis_mask = [0, 0, 0, 0],
        strides_attr = [1, 1, 1, 3]
    } : tensor<3x40x40x15xf16> -> tensor<3x30x30x4xf16>
    return %output : tensor<3x30x30x4xf16>

    // CHECK: [[SW_OUTPUT:%.+]] = memref.alloc() : memref<3x30x30x4xf16>
    // CHECK: [[SW_RESULT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_StridedSlice inputs([[ARG]] as [[INNER_ARG0:[^:]+]]: memref<3x40x40x15xf16>) outputs([[SW_OUTPUT]] as [[INNER_ARG1:[^:]+]]: memref<3x30x30x4xf16>) on tile 0 -> memref<3x30x30x4xf16>
    // CHECK: VPUIP.SW.Kernel.run {attrs = [9223372036854775807, [0, 10, 10, 5], [3, 40, 40, 15], [1, 1, 1, 3], 1, 1, 1]}([[INNER_ARG0]], [[INNER_ARG1]]) : memref<3x40x40x15xf16>, memref<3x30x30x4xf16>
    // CHECK: return [[SW_RESULT]] : memref<3x30x30x4xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK: func.func nested @cache_prefetch() attributes {VPU.task_type = @CACHE_PREFETCH}

config.Resources 3 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @FlashSDPAInOutQueryBuffer
func.func @FlashSDPAInOutQueryBuffer(%arg0: tensor<1x2x1024x32xf16>, %arg1: tensor<1x2x128x32xf16>, %arg2: tensor<1x2x128x64xf16, {order = #NCWH}>) -> tensor<1x2x1024x64xf16> {
    // Constants with weights for DPU MatMuls from SHAVE will have actual data here instead of 0
    %cst = const.Declare tensor<1x1x64x4xsi32> = dense<0> : tensor<1x1x64x4xsi32>
    %cst_0 = const.Declare tensor<1x1x128x4xsi32> = dense<0> : tensor<1x1x128x4xsi32>

    %cst_1 = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
    %cst_2 = const.Declare tensor<1x2x1024x128xf16> = dense<0.000000e+00> : tensor<1x2x1024x128xf16>
    %cst_3 = const.Declare tensor<1x2x1024x1xf16> = dense<0.000000e+00> : tensor<1x2x1024x1xf16>
    %cst_4 = const.Declare tensor<1x2x1024x1xf16> = dense<0xFC00> : tensor<1x2x1024x1xf16>
    %cst_5 = const.Declare tensor<1x2x1024x64xf16> = dense<0.000000e+00> : tensor<1x2x1024x64xf16>

    %9 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x2x1024x32xf16> -> !VPU.DistributedTensor<1x2x1024x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 2, 342, 32], [1, 2, 341, 32], [1, 2, 341, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]], memory_shapes = [[1, 2, 342, 32], [1, 2, 341, 32], [1, 2, 341, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]]}>
    %11 = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x2x128x32xf16> -> !VPU.DistributedTensor<1x2x128x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 2, 128, 32], [1, 2, 128, 32], [1, 2, 128, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 2, 128, 32], [1, 2, 128, 32], [1, 2, 128, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    %15 = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x2x128x64xf16, {order = #NCWH}> -> !VPU.DistributedTensor<1x2x128x64xf16, #NCWH, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 2, 128, 64], [1, 2, 128, 64], [1, 2, 128, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 2, 128, 64], [1, 2, 128, 64], [1, 2, 128, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    %16 = VPU.Copy(%cst_2) {out_mem_space = @CMX_NN} : tensor<1x2x1024x128xf16> -> !VPU.DistributedTensor<1x2x1024x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 2, 342, 128], [1, 2, 341, 128], [1, 2, 341, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]], memory_shapes = [[1, 2, 342, 128], [1, 2, 341, 128], [1, 2, 341, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]]}>
    %17 = VPU.Copy(%cst_1) {out_mem_space = @CMX_NN} : tensor<1x1x2x256xsi32> -> !VPU.DistributedTensor<1x1x2x256xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 2, 256], [1, 1, 2, 256], [1, 1, 2, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 2, 256], [1, 1, 2, 256], [1, 1, 2, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    %18 = VPU.Copy(%cst_0) {out_mem_space = @CMX_NN} : tensor<1x1x128x4xsi32> -> !VPU.DistributedTensor<1x1x128x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 128, 4], [1, 1, 128, 4], [1, 1, 128, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 128, 4], [1, 1, 128, 4], [1, 1, 128, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    %19 = VPU.Copy(%cst) {out_mem_space = @CMX_NN} : tensor<1x1x64x4xsi32> -> !VPU.DistributedTensor<1x1x64x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 64, 4], [1, 1, 64, 4], [1, 1, 64, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 64, 4], [1, 1, 64, 4], [1, 1, 64, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    %20 = VPU.Copy(%cst_5) {out_mem_space = @CMX_NN} : tensor<1x2x1024x64xf16> -> !VPU.DistributedTensor<1x2x1024x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 2, 342, 64], [1, 2, 341, 64], [1, 2, 341, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]], memory_shapes = [[1, 2, 342, 64], [1, 2, 341, 64], [1, 2, 341, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]]}>
    %21 = VPU.Copy(%cst_4) {out_mem_space = @CMX_NN} : tensor<1x2x1024x1xf16> -> !VPU.DistributedTensor<1x2x1024x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 2, 342, 1], [1, 2, 341, 1], [1, 2, 341, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]], memory_shapes = [[1, 2, 342, 1], [1, 2, 341, 1], [1, 2, 341, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]]}>
    %22 = VPU.Copy(%cst_3) {out_mem_space = @CMX_NN} : tensor<1x2x1024x1xf16> -> !VPU.DistributedTensor<1x2x1024x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 2, 342, 1], [1, 2, 341, 1], [1, 2, 341, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]], memory_shapes = [[1, 2, 342, 1], [1, 2, 341, 1], [1, 2, 341, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]]}>

    %result_running_output, %result_running_max, %result_running_sum = VPU.FlashSDPA(%9, %11, %15, %16, %17, %18, %19, %20, %21, %22) {is_head = true, is_tail = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0>, source_seq_len_pad_size = 0 : i64} : !VPU.DistributedTensor<1x2x1024x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 2, 342, 32], [1, 2, 341, 32], [1, 2, 341, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]], memory_shapes = [[1, 2, 342, 32], [1, 2, 341, 32], [1, 2, 341, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]]}>, !VPU.DistributedTensor<1x2x128x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 2, 128, 32], [1, 2, 128, 32], [1, 2, 128, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 2, 128, 32], [1, 2, 128, 32], [1, 2, 128, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>, !VPU.DistributedTensor<1x2x128x64xf16, #NCWH, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 2, 128, 64], [1, 2, 128, 64], [1, 2, 128, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 2, 128, 64], [1, 2, 128, 64], [1, 2, 128, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>, !VPU.DistributedTensor<1x2x1024x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 2, 342, 128], [1, 2, 341, 128], [1, 2, 341, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]], memory_shapes = [[1, 2, 342, 128], [1, 2, 341, 128], [1, 2, 341, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]]}>, !VPU.DistributedTensor<1x1x2x256xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 2, 256], [1, 1, 2, 256], [1, 1, 2, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 2, 256], [1, 1, 2, 256], [1, 1, 2, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>, !VPU.DistributedTensor<1x1x128x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 128, 4], [1, 1, 128, 4], [1, 1, 128, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 128, 4], [1, 1, 128, 4], [1, 1, 128, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>, !VPU.DistributedTensor<1x1x64x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 64, 4], [1, 1, 64, 4], [1, 1, 64, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 64, 4], [1, 1, 64, 4], [1, 1, 64, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>, !VPU.DistributedTensor<1x2x1024x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 2, 342, 64], [1, 2, 341, 64], [1, 2, 341, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]], memory_shapes = [[1, 2, 342, 64], [1, 2, 341, 64], [1, 2, 341, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]]}>, !VPU.DistributedTensor<1x2x1024x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 2, 342, 1], [1, 2, 341, 1], [1, 2, 341, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]], memory_shapes = [[1, 2, 342, 1], [1, 2, 341, 1], [1, 2, 341, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]]}>, !VPU.DistributedTensor<1x2x1024x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 2, 342, 1], [1, 2, 341, 1], [1, 2, 341, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]], memory_shapes = [[1, 2, 342, 1], [1, 2, 341, 1], [1, 2, 341, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]]}> -> !VPU.DistributedTensor<1x2x1024x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 2, 342, 64], [1, 2, 341, 64], [1, 2, 341, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]], memory_shapes = [[1, 2, 342, 64], [1, 2, 341, 64], [1, 2, 341, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]]}>, !VPU.DistributedTensor<1x2x1024x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 2, 342, 1], [1, 2, 341, 1], [1, 2, 341, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]], memory_shapes = [[1, 2, 342, 1], [1, 2, 341, 1], [1, 2, 341, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]]}>, !VPU.DistributedTensor<1x2x1024x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 2, 342, 1], [1, 2, 341, 1], [1, 2, 341, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]], memory_shapes = [[1, 2, 342, 1], [1, 2, 341, 1], [1, 2, 341, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]]}>

    %24 = VPU.Copy(%result_running_output) : !VPU.DistributedTensor<1x2x1024x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 2, 342, 64], [1, 2, 341, 64], [1, 2, 341, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]], memory_shapes = [[1, 2, 342, 64], [1, 2, 341, 64], [1, 2, 341, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 342, 0], [0, 0, 683, 0]]}> -> tensor<1x2x1024x64xf16>
    return %24 : tensor<1x2x1024x64xf16>

    // Input allocs
    // CHECK:       [[QUERY_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x2x1024x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK:       [[KEY_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x2x128x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK:       [[VALUE_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x2x128x64xf16, #NCWH, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK:       [[AUX_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x2x1024x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK:       [[DPU_DESCRIPTORS_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x2x256xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK:       [[WEIGHTS_TABLE_0_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x128x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK:       [[WEIGHTS_TABLE_1_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x64x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK:       [[RUNNING_OUT_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x2x1024x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK:       [[RUNNING_MAX_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x2x1024x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK:       [[RUNNING_SUM_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x2x1024x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments

    // Output allocs
    // CHECK:       [[OUT_ALLOC_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x2x1024x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK:       [[MAX_ALLOC_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x2x1024x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK:       [[SUM_ALLOC_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x2x1024x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK-NOT:   VPURT.AllocDistributed

    // Kernel outputs and auxiliary buffers
    // CHECK:       [[RESULTS_0:%[^:]+]]:5 = VPUIP.SW.Kernel
    // CHECK-SAME:      @VPU.SW::@builtin_FlashSDPA
    // CHECK-SAME:      inputs([[QUERY_0:%[^ ]+]] as
    // CHECK-SAME:             [[KEY:%[^ ]+]] as
    // CHECK-SAME:             [[VALUE:%[^ ]+]] as
    // CHECK-SAME:             [[AUX_BUFFER:%[^ ]+]] as
    // CHECK-SAME:             [[DPU_DESCRIPTOR_AUX_BUFFER:%[^ ]+]] as
    // CHECK-SAME:             [[DPU_WEIGHTS_TABLE0:%[^ ]+]] as
    // CHECK-SAME:             [[DPU_WEIGHTS_TABLE1:%[^ ]+]] as
    // CHECK-SAME:             [[INPUT_RUNNING_OUTPUT:%[^ ]+]] as
    // CHECK-SAME:             [[INPUT_RUNNING_MAX:%[^ ]+]] as
    // CHECK-SAME:             [[INPUT_RUNNING_SUM:%[^ ]+]] as
    // CHECK-SAME:      outputs([[OUT_ALLOC_0]] as
    // CHECK-SAME:              [[MAX_ALLOC_0]] as
    // CHECK-SAME:              [[SUM_ALLOC_0]] as
    // CHECK-SAME:              [[AUX_BUFFER]] as
    // CHECK-SAME:              [[DPU_DESCRIPTOR_AUX_BUFFER]] as

    // CHECK:       [[RESULT_ALLOC:%.+]] = memref.alloc() : memref<1x2x1024x64xf16>
    // CHECK:       [[RESULT_COPY:%.+]] = VPUIP.Copy inputs([[RESULTS_0]]#0
    // CHECK-SAME:                                   outputs([[RESULT_ALLOC]]

    // CHECK:       return [[RESULT_COPY]] : memref<1x2x1024x64xf16>
}
