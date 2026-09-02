//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --unroll-distributed-ops --canonicalize  %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#C = affine_map<(d0) -> (d0)>

module @VPU.SW {
  func.func nested @builtin_LSTMSequence(memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none, i64) attributes {VPU.kernel_code = "lstm_sequence.cpp", VPU.kernel_entry = "lstm_sequence"}
  func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
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
    // CHECK: [[INPUT_SHAPE_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <334856> -> memref<4xsi32, [@CMX_NN, 0]>
    // CHECK: [[INPUT_SHAPE_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <334856> -> memref<4xsi32, [@CMX_NN, 1]>
    // CHECK: [[OUTPUT_SHAPE_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <334864> -> memref<4xsi32, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT_SHAPE_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <334864> -> memref<4xsi32, [@CMX_NN, 1]>

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
      VPUIP.SW.Kernel.run {attrs = [[-9223372036854775808, 1023], 2, [-1, -1]]}(%arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) :
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

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func nested @builtin_AtanDma(memref<*xf16>, memref<*xsi32>, memref<*xui8, @CMX_NN>, memref<*xf16>, memref<*xsi32>, memref<*xui8, @CMX_NN>, i64)
                      attributes {VPU.kernel_code = "activation_atan_dma.cpp",
                                  VPU.kernel_entry = "activation_atan_dma",
                                  VPU.kernel_name = "activation_atan_dma",
                                  VPU.task_type = @COMPUTE}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!AuxDistributed = !VPUIP.DistributedBuffer<1x1x1x524288xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes  = [[1, 1, 1, 524288], [1, 1, 1, 524288]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes   = [[1, 1, 1, 524288], [1, 1, 1, 524288]],
    memory_offsets  = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

// CHECK-LABEL: UnrollAtanDma
func.func @UnrollAtanDma() -> (memref<1x1x1x4194304xf16, @DDR>, memref<4xsi32, @DDR>) {
    %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x1x4194304xf16, @DDR>
    %1 = VPURT.DeclareBuffer <NetworkInput> [1] <0> -> memref<4xsi32, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x1x1x4194304xf16, @DDR>
    %3 = VPURT.DeclareBuffer <NetworkOutput> [1] <0> -> memref<4xsi32, @DDR>

    %8 = VPURT.DeclareBuffer <CMX_NN> <0> -> !AuxDistributed

    VPURT.Task {
      %results:2, %dynamicOutputShapes = VPUIP.SW.Kernel
      {dynamicInputShapesMap = array<i32: 0, -1>, dynamicOutputShapesMap = array<i32: 0, -1>, resultSegmentSizes = array<i32: 2, 1, 0>}
       @VPU.SW::@builtin_AtanDma
           inputs(%0 as %arg4: memref<1x1x1x4194304xf16, @DDR>, %8 as %arg5: !AuxDistributed)
           dynamicInputShapes(%1 : memref<4xsi32, @DDR>)
           outputs(%2 as %arg6: memref<1x1x1x4194304xf16, @DDR>, %8 as %arg7: !AuxDistributed)
           dynamicOutputShapes(%3 : memref<4xsi32, @DDR>)
           on tile 0 list 0 -> (memref<1x1x1x4194304xf16, @DDR>, !AuxDistributed, memref<4xsi32, @DDR>)
      {
        VPUIP.SW.Kernel.run {attrs = [8589934594]}(%arg4, %arg5, %arg6, %arg7) : memref<1x1x1x4194304xf16, @DDR>, !AuxDistributed, memref<1x1x1x4194304xf16, @DDR>, !AuxDistributed
      }
    }
    VPURT.Task {
      %results:2, %dynamicOutputShapes = VPUIP.SW.Kernel
      {dynamicInputShapesMap = array<i32: 0, -1>, dynamicOutputShapesMap = array<i32: 0, -1>, resultSegmentSizes = array<i32: 2, 1, 0>}
       @VPU.SW::@builtin_AtanDma
           inputs(%0 as %arg4: memref<1x1x1x4194304xf16, @DDR>, %8 as %arg5: !AuxDistributed)
           dynamicInputShapes(%1 : memref<4xsi32, @DDR>)
           outputs(%2 as %arg6: memref<1x1x1x4194304xf16, @DDR>, %8 as %arg7: !AuxDistributed)
           dynamicOutputShapes(%3 : memref<4xsi32, @DDR>)
           on tile 0 list 1 -> (memref<1x1x1x4194304xf16, @DDR>, !AuxDistributed, memref<4xsi32, @DDR>)
      {
        VPUIP.SW.Kernel.run {attrs = [8589934594]}(%arg4, %arg5, %arg6, %arg7) : memref<1x1x1x4194304xf16, @DDR>, !AuxDistributed, memref<1x1x1x4194304xf16, @DDR>, !AuxDistributed
      }
    }

    return %2, %3 : memref<1x1x1x4194304xf16, @DDR>, memref<4xsi32, @DDR>

    // CHECK:   [[IN_DATA:%.*]]   = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x1x4194304xf16, @DDR>
    // CHECK:   [[IN_SHAPE:%.*]]  = VPURT.DeclareBuffer <NetworkInput> [1] <0> -> memref<4xsi32, @DDR>
    // CHECK:   [[OUT_DATA:%.*]]  = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x1x1x4194304xf16, @DDR>
    // CHECK:   [[OUT_SHAPE:%.*]] = VPURT.DeclareBuffer <NetworkOutput> [1] <0> -> memref<4xsi32, @DDR>
    // CHECK:   [[AUX_0a:%.*]]    = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x524288xui8, [@CMX_NN, 0]>
    // CHECK:   [[AUX_1a:%.*]]    = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1x1x524288xui8, [@CMX_NN, 1]>
    // CHECK:   [[AUX_0b:%.*]]    = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x524288xui8, [@CMX_NN, 0]>
    // CHECK:   [[AUX_1b:%.*]]    = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1x1x524288xui8, [@CMX_NN, 1]>

    // CHECK:      @VPU.SW::@builtin_AtanDma inputs([[IN_DATA]] as [[IN_1:[^:]+]]: memref<1x1x1x4194304xf16, @DDR>, [[AUX_0b]]
    // CHECK-SAME:                           dynamicInputShapes([[IN_SHAPE]]
    // CHECK-SAME:                           outputs([[OUT_DATA]] as [[OUT_1:[^:]+]]: memref<1x1x1x4194304xf16, @DDR>, [[AUX_0b]]
    // CHECK-SAME:                           dynamicOutputShapes([[OUT_SHAPE]]
    // CHECK-SAME:                           on tile 0 list 0

    // CHECK:      @VPU.SW::@builtin_AtanDma inputs([[IN_DATA]] as [[IN_2:[^:]+]]: memref<1x1x1x4194304xf16, @DDR>, [[AUX_1b]]
    // CHECK-SAME:                           dynamicInputShapes([[IN_SHAPE]]
    // CHECK-SAME:                           outputs([[OUT_DATA]] as [[OUT_2:[^:]+]]: memref<1x1x1x4194304xf16, @DDR>, [[AUX_1b]]
    // CHECK-SAME:                           dynamicOutputShapes([[OUT_SHAPE]]
    // CHECK-SAME:                           on tile 1 list 0

    // CHECK:      @VPU.SW::@builtin_AtanDma inputs([[IN_DATA]] as [[IN_3:[^:]+]]: memref<1x1x1x4194304xf16, @DDR>, [[AUX_0a]]
    // CHECK-SAME:                           dynamicInputShapes([[IN_SHAPE]]
    // CHECK-SAME:                           outputs([[OUT_DATA]] as [[OUT_3:[^:]+]]: memref<1x1x1x4194304xf16, @DDR>, [[AUX_0a]]
    // CHECK-SAME:                           dynamicOutputShapes([[OUT_SHAPE]]
    // CHECK-SAME:                           on tile 0 list 1

    // CHECK:      @VPU.SW::@builtin_AtanDma inputs([[IN_DATA]] as [[IN_4:[^:]+]]: memref<1x1x1x4194304xf16, @DDR>, [[AUX_1a]]
    // CHECK-SAME:                           dynamicInputShapes([[IN_SHAPE]]
    // CHECK-SAME:                           outputs([[OUT_DATA]] as [[OUT_4:[^:]+]]: memref<1x1x1x4194304xf16, @DDR>, [[AUX_1a]]
    // CHECK-SAME:                           dynamicOutputShapes([[OUT_SHAPE]]
    // CHECK-SAME:                           on tile 1 list 1

    // CHECK:      return [[OUT_DATA]], [[OUT_SHAPE]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func nested @builtin_InterpolateDMA(memref<*xf16>, memref<*xf16>, memref<*xsi32>, memref<*xf16>, memref<*xui8, @CMX_NN>, memref<*xf16>, memref<*xui8, @CMX_NN>) attributes {VPU.kernel_code = "interpolate_dma.cpp", VPU.kernel_entry = "interpolate_dma", VPU.kernel_name = "interpolate_dma", VPU.task_type = @COMPUTE}
  func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!AuxDistributed = !VPUIP.DistributedBuffer<1x1x1x524288xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes  = [[1, 1, 1, 524288], [1, 1, 1, 524288]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes   = [[1, 1, 1, 524288], [1, 1, 1, 524288]],
    memory_offsets  = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

// CHECK-LABEL: @UnrollSWKernelNoDynInputOneDynOutput
func.func @UnrollSWKernelNoDynInputOneDynOutput() -> (memref<1x3x32x48xf16, @DDR>, memref<4xsi32, @DDR>) {
    %data_in = VPURT.DeclareBuffer <DDR> <64> -> memref<1x3x4x6xf16, @DDR>
    %scales_in = VPURT.DeclareBuffer <DDR> <0> -> memref<4xf16, @DDR>
    %coords = VPURT.DeclareBuffer <DDR> <200> -> memref<1x1x1x48xsi32, @DDR>
    %lambdas = VPURT.DeclareBuffer <DDR> <500> -> memref<1x1x1x96xf16, @DDR>
    %aux = VPURT.DeclareBuffer <CMX_NN> <0> -> !AuxDistributed
    %data_out = VPURT.DeclareBuffer <DDR> <1024> -> memref<1x3x32x48xf16, @DDR>
    %shape_out = VPURT.DeclareBuffer <DDR> <100000> -> memref<4xsi32, @DDR>

    VPURT.Task {
      %results:2, %dynamicOutputShapes = VPUIP.SW.Kernel {
          dynamicInputShapesMap = array<i32: -1, -1, -1, -1, -1>,
          dynamicOutputShapesMap = array<i32: 0, -1>,
          resultSegmentSizes = array<i32: 2, 1, 0>}
          @VPU.SW::@builtin_InterpolateDMA
          inputs(
              %data_in as %arg0: memref<1x3x4x6xf16, @DDR>,
              %scales_in as %arg1: memref<4xf16, @DDR>,
              %coords as %arg2: memref<1x1x1x48xsi32, @DDR>,
              %lambdas as %arg3: memref<1x1x1x96xf16, @DDR>,
              %aux as %arg4: !AuxDistributed)
          outputs(
              %data_out as %arg5: memref<1x3x32x48xf16, @DDR>,
              %aux as %arg6: !AuxDistributed)
          dynamicOutputShapes(
              %shape_out : memref<4xsi32, @DDR>)
          on tile 0 list 0 -> (memref<1x3x32x48xf16, @DDR>, !AuxDistributed, memref<4xsi32, @DDR>) {
        VPUIP.SW.Kernel.run {attrs = [[-9223372036854775808, 127], 2, 0, 2, 0, [2, 3], -7.500000e-01, 0.000000e+00, 8589934593]}
            (%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) :
            memref<1x3x4x6xf16, @DDR>, memref<4xf16, @DDR>, memref<1x1x1x48xsi32, @DDR>, memref<1x1x1x96xf16, @DDR>,
            !AuxDistributed, memref<1x3x32x48xf16, @DDR>, !AuxDistributed
      }
    }
    VPURT.Task {
      %results:2, %dynamicOutputShapes = VPUIP.SW.Kernel {
          dynamicInputShapesMap = array<i32: -1, -1, -1, -1, -1>,
          dynamicOutputShapesMap = array<i32: 0, -1>,
          resultSegmentSizes = array<i32: 2, 1, 0>}
          @VPU.SW::@builtin_InterpolateDMA
          inputs(
              %data_in as %arg0: memref<1x3x4x6xf16, @DDR>,
              %scales_in as %arg1: memref<4xf16, @DDR>,
              %coords as %arg2: memref<1x1x1x48xsi32, @DDR>,
              %lambdas as %arg3: memref<1x1x1x96xf16, @DDR>,
              %aux as %arg4: !AuxDistributed)
          outputs(
              %data_out as %arg5: memref<1x3x32x48xf16, @DDR>,
              %aux as %arg6: !AuxDistributed)
          dynamicOutputShapes(
              %shape_out : memref<4xsi32, @DDR>)
          on tile 0 list 1 -> (memref<1x3x32x48xf16, @DDR>, !AuxDistributed, memref<4xsi32, @DDR>) {
        VPUIP.SW.Kernel.run {attrs = [[-9223372036854775808, 127], 2, 0, 2, 0, [2, 3], -7.500000e-01, 0.000000e+00, 8589934593]}
            (%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) :
            memref<1x3x4x6xf16, @DDR>, memref<4xf16, @DDR>, memref<1x1x1x48xsi32, @DDR>, memref<1x1x1x96xf16, @DDR>,
            !AuxDistributed, memref<1x3x32x48xf16, @DDR>, !AuxDistributed
      }
    }

    // DDR buffers remain shared across all unrolled tasks
    // CHECK:   [[DATA_IN:%.+]]  = VPURT.DeclareBuffer <DDR> <64> -> memref<1x3x4x6xf16, @DDR>
    // CHECK:   [[SCALES:%.+]]   = VPURT.DeclareBuffer <DDR> <0> -> memref<4xf16, @DDR>
    // CHECK:   [[COORDS:%.+]]   = VPURT.DeclareBuffer <DDR> <200> -> memref<1x1x1x48xsi32, @DDR>
    // CHECK:   [[LAMBDAS:%.+]]  = VPURT.DeclareBuffer <DDR> <500> -> memref<1x1x1x96xf16, @DDR>
    // Aux buffer split to per-cluster (2 pairs for 2 original tasks)
    // CHECK:   [[AUX_0a:%.+]]   = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x524288xui8, [@CMX_NN, 0]>
    // CHECK:   [[AUX_1a:%.+]]   = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1x1x524288xui8, [@CMX_NN, 1]>
    // CHECK:   [[AUX_0b:%.+]]   = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x524288xui8, [@CMX_NN, 0]>
    // CHECK:   [[AUX_1b:%.+]]   = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1x1x524288xui8, [@CMX_NN, 1]>
    // CHECK:   [[DATA_OUT:%.+]] = VPURT.DeclareBuffer <DDR> <1024> -> memref<1x3x32x48xf16, @DDR>
    // CHECK:   [[SHAPE_OUT:%.+]] = VPURT.DeclareBuffer <DDR> <100000> -> memref<4xsi32, @DDR>

    // Task 1: tile 0 list 0 — no dynamicInputShapes, has dynamicOutputShapes
    // CHECK:      @VPU.SW::@builtin_InterpolateDMA
    // CHECK-SAME: inputs([[DATA_IN]] as {{[^:]+}}: memref<1x3x4x6xf16, @DDR>, [[SCALES]] as {{[^:]+}}: memref<4xf16, @DDR>
    // CHECK-SAME: [[COORDS]] as {{[^:]+}}: memref<1x1x1x48xsi32, @DDR>, [[LAMBDAS]] as {{[^:]+}}: memref<1x1x1x96xf16, @DDR>
    // CHECK-SAME: [[AUX_0b]] as {{[^:]+}}: memref<1x1x1x524288xui8, [@CMX_NN, 0]>)
    // CHECK-SAME: outputs([[DATA_OUT]] as {{[^:]+}}: memref<1x3x32x48xf16, @DDR>, [[AUX_0b]] as {{[^:]+}}: memref<1x1x1x524288xui8, [@CMX_NN, 0]>)
    // CHECK-SAME: dynamicOutputShapes([[SHAPE_OUT]] : memref<4xsi32, @DDR>)
    // CHECK-SAME: on tile 0 list 0

    // Task 2: tile 1 list 0
    // CHECK:      @VPU.SW::@builtin_InterpolateDMA
    // CHECK-SAME: [[AUX_1b]] as {{[^:]+}}: memref<1x1x1x524288xui8, [@CMX_NN, 1]>)
    // CHECK-SAME: outputs([[DATA_OUT]] as {{[^:]+}}: memref<1x3x32x48xf16, @DDR>, [[AUX_1b]] as {{[^:]+}}: memref<1x1x1x524288xui8, [@CMX_NN, 1]>)
    // CHECK-SAME: dynamicOutputShapes([[SHAPE_OUT]] : memref<4xsi32, @DDR>)
    // CHECK-SAME: on tile 1 list 0

    // Task 3: tile 0 list 1
    // CHECK:      @VPU.SW::@builtin_InterpolateDMA
    // CHECK-SAME: [[AUX_0a]] as {{[^:]+}}: memref<1x1x1x524288xui8, [@CMX_NN, 0]>)
    // CHECK-SAME: outputs([[DATA_OUT]] as {{[^:]+}}: memref<1x3x32x48xf16, @DDR>, [[AUX_0a]] as {{[^:]+}}: memref<1x1x1x524288xui8, [@CMX_NN, 0]>)
    // CHECK-SAME: dynamicOutputShapes([[SHAPE_OUT]] : memref<4xsi32, @DDR>)
    // CHECK-SAME: on tile 0 list 1

    // Task 4: tile 1 list 1
    // CHECK:      @VPU.SW::@builtin_InterpolateDMA
    // CHECK-SAME: [[AUX_1a]] as {{[^:]+}}: memref<1x1x1x524288xui8, [@CMX_NN, 1]>)
    // CHECK-SAME: outputs([[DATA_OUT]] as {{[^:]+}}: memref<1x3x32x48xf16, @DDR>, [[AUX_1a]] as {{[^:]+}}: memref<1x1x1x524288xui8, [@CMX_NN, 1]>)
    // CHECK-SAME: dynamicOutputShapes([[SHAPE_OUT]] : memref<4xsi32, @DDR>)
    // CHECK-SAME: on tile 1 list 1

    // CHECK:      return [[DATA_OUT]], [[SHAPE_OUT]]

    return %data_out, %shape_out : memref<1x3x32x48xf16, @DDR>, memref<4xsi32, @DDR>
}

// -----

// Test that multiple dynamicInputShapes are correctly unrolled per-cluster.
// Previously only a single dynamicInputShape was supported (VPUX_THROW_UNLESS size <= 1).

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#C = affine_map<(d0) -> (d0)>

module @VPU.SW {
  func.func nested @builtin_AttentionDMA(memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xui8, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "attention_dma.cpp", VPU.kernel_entry = "attention_dma"}
  func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!InputQ = !VPUIP.DistributedBuffer<
    1x6x64x64xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 6, 64, 64], [1, 6, 64, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 6, 64, 64], [1, 6, 64, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!InputK = !VPUIP.DistributedBuffer<
    1x6x128x64xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 6, 128, 64], [1, 6, 128, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 6, 128, 64], [1, 6, 128, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!InputV = !VPUIP.DistributedBuffer<
    1x6x128x64xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 6, 128, 64], [1, 6, 128, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 6, 128, 64], [1, 6, 128, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!InputScale = !VPUIP.DistributedBuffer<
    1x1x1x1xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 1], [1, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1, 1, 1], [1, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!AuxBuffer = !VPUIP.DistributedBuffer<
    1x1x1x524288xui8, #NCHW, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 524288], [1, 1, 1, 524288]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1, 1, 524288], [1, 1, 1, 524288]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!OutputDist = !VPUIP.DistributedBuffer<
    1x6x64x64xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 6, 64, 64], [1, 6, 64, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 6, 64, 64], [1, 6, 64, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedShape = !VPUIP.DistributedBuffer<4xsi32, #C, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
}>

// CHECK-LABEL: @UnrollSWKernelMultipleDynamicInputShapes
func.func @UnrollSWKernelMultipleDynamicInputShapes() -> !OutputDist {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %inputQ = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputQ
    %inputK = VPURT.DeclareBuffer <CMX_NN> <49152> -> !InputK
    %inputV = VPURT.DeclareBuffer <CMX_NN> <147456> -> !InputV
    %inputScale = VPURT.DeclareBuffer <CMX_NN> [0, 1] <245760> -> !InputScale
    %auxBuf = VPURT.DeclareBuffer <CMX_NN> <245762> -> !AuxBuffer
    %output = VPURT.DeclareBuffer <CMX_NN> <770050> -> !OutputDist

    %shapeQ = VPURT.DeclareBuffer <CMX_NN> [0, 1] <819202> -> !DistributedShape
    %shapeK = VPURT.DeclareBuffer <CMX_NN> [0, 1] <819218> -> !DistributedShape
    %shapeV = VPURT.DeclareBuffer <CMX_NN> [0, 1] <819234> -> !DistributedShape
    %shapeOut = VPURT.DeclareBuffer <CMX_NN> [0, 1] <819250> -> !DistributedShape

    // CHECK-DAG: [[SHAPE_Q_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <819202> -> memref<4xsi32, [@CMX_NN, 0]>
    // CHECK-DAG: [[SHAPE_Q_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <819202> -> memref<4xsi32, [@CMX_NN, 1]>
    // CHECK-DAG: [[SHAPE_K_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <819218> -> memref<4xsi32, [@CMX_NN, 0]>
    // CHECK-DAG: [[SHAPE_K_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <819218> -> memref<4xsi32, [@CMX_NN, 1]>
    // CHECK-DAG: [[SHAPE_V_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <819234> -> memref<4xsi32, [@CMX_NN, 0]>
    // CHECK-DAG: [[SHAPE_V_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <819234> -> memref<4xsi32, [@CMX_NN, 1]>
    // CHECK-DAG: [[SHAPE_OUT_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <819250> -> memref<4xsi32, [@CMX_NN, 0]>
    // CHECK-DAG: [[SHAPE_OUT_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <819250> -> memref<4xsi32, [@CMX_NN, 1]>

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
    %results, %dynamicOutputShapes = VPUIP.SW.Kernel {
        dynamicInputShapesMap = array<i32: 0, 1, 2, -1, -1>,
        dynamicOutputShapesMap = array<i32: 0>,
        resultSegmentSizes = array<i32: 1, 1, 0>}
        @VPU.SW::@builtin_AttentionDMA
        inputs(
            %inputQ as %arg0: !InputQ,
            %inputK as %arg1: !InputK,
            %inputV as %arg2: !InputV,
            %inputScale as %arg3: !InputScale,
            %auxBuf as %arg4: !AuxBuffer)
        dynamicInputShapes(
            %shapeQ : !DistributedShape,
            %shapeK : !DistributedShape,
            %shapeV : !DistributedShape)
        outputs(
            %output as %arg5: !OutputDist)
        dynamicOutputShapes(
            %shapeOut : !DistributedShape)
        on tile 0 -> (!OutputDist, !DistributedShape) {
      VPUIP.SW.Kernel.run {attrs = []}(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) :
        !InputQ, !InputK, !InputV, !InputScale, !AuxBuffer, !OutputDist
        }
    }

    // Verify cluster 0: all 3 dynamic input shape buffers unrolled to cluster 0
    // CHECK: VPURT.Task waits(
    // CHECK: VPUIP.SW.Kernel
    // CHECK-SAME: dynamicInputShapesMap = array<i32: 0, 1, 2, -1, -1>
    // CHECK-SAME: @VPU.SW::@builtin_AttentionDMA
    // CHECK-SAME: dynamicInputShapes([[SHAPE_Q_0]] : memref<4xsi32, [@CMX_NN, 0]>, [[SHAPE_K_0]] : memref<4xsi32, [@CMX_NN, 0]>, [[SHAPE_V_0]] : memref<4xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME: dynamicOutputShapes([[SHAPE_OUT_0]] : memref<4xsi32, [@CMX_NN, 0]>)

    // Verify cluster 1: all 3 dynamic input shape buffers unrolled to cluster 1
    // CHECK: VPURT.Task waits(
    // CHECK: VPUIP.SW.Kernel
    // CHECK-SAME: dynamicInputShapesMap = array<i32: 0, 1, 2, -1, -1>
    // CHECK-SAME: @VPU.SW::@builtin_AttentionDMA
    // CHECK-SAME: dynamicInputShapes([[SHAPE_Q_1]] : memref<4xsi32, [@CMX_NN, 1]>, [[SHAPE_K_1]] : memref<4xsi32, [@CMX_NN, 1]>, [[SHAPE_V_1]] : memref<4xsi32, [@CMX_NN, 1]>)
    // CHECK-SAME: dynamicOutputShapes([[SHAPE_OUT_1]] : memref<4xsi32, [@CMX_NN, 1]>)

    return %output : !OutputDist
}
