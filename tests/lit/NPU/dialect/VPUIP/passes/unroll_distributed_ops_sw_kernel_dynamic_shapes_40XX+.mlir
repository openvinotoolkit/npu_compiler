//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-distributed-ops --canonicalize  %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#C = affine_map<(d0) -> (d0)>

module @VPU.SW {
  func.func private @builtin_LSTMSequence(memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64) attributes {VPU.kernel_code = "lstm_sequence.cpp", VPU.kernel_entry = "lstm_sequence"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!InputDistributed1 = !VPUIP.DistributedBuffer<
     1x2x35x512xf16, #NCHW, @CMX_NN, {
     mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
     compute_shapes = [[1, 1, 35, 512], [1, 1, 35, 512]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
     memory_shapes = [[1, 1, 35, 512], [1, 1, 35, 512]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]
}>

!InputDistributed2 = !VPUIP.DistributedBuffer<
     1x2x1x128xf16, #NCHW, @CMX_NN, {
     mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
     compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
     memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]
}>

!InputDistributed3 = !VPUIP.DistributedBuffer<
     1x2x1x128xf16, #NCHW, @CMX_NN, {
     mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
     compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
     memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]
}>

!InputDistributed4 = !VPUIP.DistributedBuffer<
     2x4x128x128xf16, #NCHW, @CMX_NN, {
     mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
     compute_shapes = [[1, 4, 128, 128], [1, 4, 128, 128]], compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]],
     memory_shapes = [[1, 4, 128, 128], [1, 4, 128, 128]], memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]
}>

!InputDistributed5 = !VPUIP.DistributedBuffer<
     1x1x1x2xsi32, #NCHW, @CMX_NN, {
     mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
     compute_shapes = [[1, 1, 1, 2], [1, 1, 1, 2]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
     memory_shapes = [[1, 1, 1, 2], [1, 1, 1, 2]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedShape = !VPUIP.DistributedBuffer<4xsi32, #C, @CMX_NN, {
     mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
}>

!OutputDistributed1 = !VPUIP.DistributedBuffer<
     1x2x35x128xf16, #NCHW, @CMX_NN, {
     mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
     compute_shapes = [[1, 1, 35, 128], [1, 1, 35, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
     memory_shapes = [[1, 1, 35, 128], [1, 1, 35, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]
}>

!OutputDistributed2 = !VPUIP.DistributedBuffer<
     1x2x1x128xf16, #NCHW, @CMX_NN, {
     mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
     compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
     memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]
}>

!OutputDistributed3 = !VPUIP.DistributedBuffer<
     1x2x1x128xf16, #NCHW, @CMX_NN, {
     mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
     compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
     memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]
}>

func.func @UnrollSWKernelWithDynamicShapes() -> !DistributedShape {
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input1 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed1
    %input2 = VPURT.DeclareBuffer <CMX_NN> <71680> -> !InputDistributed2
    %input3 = VPURT.DeclareBuffer <CMX_NN> <72192> -> !InputDistributed3
    %input4 = VPURT.DeclareBuffer <CMX_NN> <72704> -> !InputDistributed4
    %input5 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <334848> -> !InputDistributed5

    %inputShape = VPURT.DeclareBuffer <CMX_NN> [0, 1] <334856> -> !DistributedShape
    %outputShape = VPURT.DeclareBuffer <CMX_NN> [0, 1] <334864> -> !DistributedShape
    // CHECK: [[INPUT_SHAPE_CMX_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <334856> -> memref<4xsi32, [@CMX_NN, 0]>
    // CHECK: [[INPUT_SHAPE_CMX_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <334856> -> memref<4xsi32, [@CMX_NN, 1]>
    // CHECK: [[OUTPUT_SHAPE_CMX_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <334864> -> memref<4xsi32, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT_SHAPE_CMX_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <334864> -> memref<4xsi32, [@CMX_NN, 1]>

    %output1 = VPURT.DeclareBuffer <CMX_NN> <334872> -> !OutputDistributed1
    %output2 = VPURT.DeclareBuffer <CMX_NN> <337000> -> !OutputDistributed2
    %output3 = VPURT.DeclareBuffer <CMX_NN> <339128> -> !OutputDistributed3


    VPURT.Task waits(%bar1, %bar2, %bar3, %bar4 : !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier)
               updates(%bar5: !VPURT.Barrier) {
    %results:3, %dynamicOutputShapes = VPUIP.SW.Kernel {
        dynamicInputShapesMap = array<i32: 0, -1, -1, -1, -1>,
        dynamicOutputShapesMap = array<i32: 0, -1, -1>,
        resultSegmentSizes = array<i32: 3, 1, 0>}
        @VPU.SW::@builtin_LSTMSequence
        inputs(
            %input1 as %arg8: !InputDistributed1,
            %input2 as %arg9: !InputDistributed2,
            %input3 as %arg10: !InputDistributed3,
            %input4 as %arg11: !InputDistributed4,
            %input5 as %arg12: !InputDistributed5)
        dynamicInputShapes(
            %inputShape : !DistributedShape)
        outputs(
            %output1 as %arg13: !OutputDistributed1,
            %output2 as %arg14: !OutputDistributed2,
            %output3 as %arg15: !OutputDistributed3)
        dynamicOutputShapes(
            %outputShape: !DistributedShape)
        on tile 0 -> (
            !OutputDistributed1, !OutputDistributed2, !OutputDistributed3, !DistributedShape) {
      VPUIP.SW.Kernel.run {attrs = [2]}(%arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) :
        !InputDistributed1, !InputDistributed2, !InputDistributed3, !InputDistributed4, !InputDistributed5, !OutputDistributed1, !OutputDistributed2, !OutputDistributed3
        }
    }

    // CHECK: VPURT.Task waits(
    // CHECK: VPUIP.SW.Kernel
    // CHECK-SAME: dynamicInputShapesMap
    // CHECK-SAME: dynamicOutputShapesMap
    // CHECK-SAME: resultSegmentSizes
    // CHECK-SAME: @VPU.SW::@builtin_LSTMSequence
    // CHECK-SAME: inputs(
    // CHECK-SAME: dynamicInputShapes([[INPUT_SHAPE_CMX_0]] : memref<4xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME: outputs(
    // CHECK-SAME: dynamicOutputShapes([[OUTPUT_SHAPE_CMX_0]] : memref<4xsi32, [@CMX_NN, 0]>)

    // CHECK: VPURT.Task waits(
    // CHECK: VPUIP.SW.Kernel
    // CHECK-SAME: dynamicInputShapesMap
    // CHECK-SAME: dynamicOutputShapesMap
    // CHECK-SAME: resultSegmentSizes
    // CHECK-SAME: @VPU.SW::@builtin_LSTMSequence
    // CHECK-SAME: inputs(
    // CHECK-SAME: dynamicInputShapes([[INPUT_SHAPE_CMX_1]] : memref<4xsi32, [@CMX_NN, 1]>)
    // CHECK-SAME: outputs(
    // CHECK-SAME: dynamicOutputShapes([[OUTPUT_SHAPE_CMX_1]] : memref<4xsi32, [@CMX_NN, 1]>)



    return %outputShape: !DistributedShape
}
