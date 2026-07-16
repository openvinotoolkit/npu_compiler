//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --unroll-distributed-ops --canonicalize  %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

!qElemType  = !quant.uniform<u8:f16:0, {1.000000e-02:128,2.000000e-02:128,3.000000e-02:128,4.000000e-02:128,5.000000e-02:128,6.000000e-02:128,7.000000e-02:128,8.000000e-02:128,0.089999999999999996:128,1.000000e-01:128,1.100000e-01:128,1.200000e-01:128,1.300000e-01:128,1.400000e-01:128,1.500000e-01:128,1.600000e-01:128}>
//CHECK-DAG: [[QTYPE_1:!.+]] = !quant.uniform<u8:f16:0, {1.000000e-02:128,2.000000e-02:128,3.000000e-02:128,4.000000e-02:128,5.000000e-02:128,6.000000e-02:128,7.000000e-02:128,8.000000e-02:128}>
//CHECK-DAG: [[QTYPE_2:!.+]] = !quant.uniform<u8:f16:0, {0.089999999999999996:128,1.000000e-01:128,1.100000e-01:128,1.200000e-01:128,1.300000e-01:128,1.400000e-01:128,1.500000e-01:128,1.600000e-01:128}>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
   16x48x3x3x!qElemType, #NHWC, @CMX_NN,
   {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes  = [[8, 48, 3, 3], [8, 48, 3, 3]],
    compute_offsets = [[0,  0, 0, 0], [8,  0, 0, 0]],
    memory_shapes   = [[8, 48, 3, 3], [8, 48, 3, 3]],
    memory_offsets  = [[0,  0, 0, 0], [8,  0, 0, 0]]}
>

!OutputDistributed = !VPUIP.DistributedBuffer<
   16x48x3x3xf16, #NHWC, @CMX_NN,
   {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes  = [[8, 48, 3, 3], [8, 48, 3, 3]],
    compute_offsets = [[0,  0, 0, 0], [8,  0, 0, 0]],
    memory_shapes   = [[8, 48, 3, 3], [8, 48, 3, 3]],
    memory_offsets  = [[0,  0, 0, 0], [8,  0, 0, 0]]}
>


module @VPU.SW {
    func.func nested @builtin_Dequantize(memref<*x!qElemType, @CMX_NN>, memref<*xf16, @CMX_NN>, none) attributes {VPU.kernel_code = "dequantize.cpp", VPU.kernel_entry = "dequantize", VPU.kernel_name = "dequantize", VPU.task_type = @COMPUTE}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @UnrollDequantOverQuantAxis(%arg0: memref<16x48x3x3xui8, {order = #NHWC}, @DDR>, %arg1: memref<16x48x3x3xf16, {order = #NHWC}, @DDR>) -> memref<16x48x3x3xf16, {order = #NHWC}, @DDR> {
  %0 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<16x48x3x3xf16, {order = #NHWC}, @DDR>
  %1 = VPURT.DeclareBuffer <CMX_NN> <13824> -> !InputDistributed
  %2 = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed
  %3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %4 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<16x48x3x3x!qElemType, {order = #NHWC}, @DDR>
  %5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  VPURT.Task updates(%3 : !VPURT.Barrier) {
    %6 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%4 : memref<16x48x3x3x!qElemType, {order = #NHWC}, @DDR>) outputs(%1 : !InputDistributed) -> !InputDistributed
  }
  VPURT.Task waits(%3 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) {
    %results = VPUIP.SW.Kernel {listIndex = 0 : i64, resultSegmentSizes = array<i32: 1, 0, 0>}
       @VPU.SW::@builtin_Dequantize
         inputs(%1 as %arg2: !InputDistributed)
         outputs(%2 as %arg3: !OutputDistributed) -> !OutputDistributed
    {
      VPUIP.SW.Kernel.run {attrs = [[3, 16, 2963130708733665567, 3251366363510221414, 3435735286504893891, 3539601489976307753, 6341165033837320192, 6341165033837320192, 6341165033837320192, 6341165033837320192]]}(%arg2, %arg3) : !InputDistributed, !OutputDistributed
    }
  }
  VPURT.Task waits(%5 : !VPURT.Barrier) {
    %6 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%2 : !OutputDistributed) outputs(%0 : memref<16x48x3x3xf16, {order = #NHWC}, @DDR>) -> memref<16x48x3x3xf16, {order = #NHWC}, @DDR>
  }
  return %arg1 : memref<16x48x3x3xf16, {order = #NHWC}, @DDR>

  //CHECK: @VPU.SW::@builtin_Dequantize
  //CHECK-SAME:  memref<8x48x3x3x[[QTYPE_1]], {order = #NHWC}, [@CMX_NN, 0]>
  //CHECK{LITERAL}: VPUIP.SW.Kernel.run {attrs = [[3, 8, 2963130708733665567, 3251366363510221414, 6341165033837320192, 6341165033837320192]]}

  //CHECK: @VPU.SW::@builtin_Dequantize
  //CHECK-SAME:  memref<8x48x3x3x[[QTYPE_2]], {order = #NHWC}, [@CMX_NN, 1]>
  //CHECK{LITERAL}: VPUIP.SW.Kernel.run {attrs = [[3, 8, 3435735286504893891, 3539601489976307753, 6341165033837320192, 6341165033837320192]]}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<1x1x5x13xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
!AuxDistributed = !VPUIP.DistributedBuffer<1x1x1x40xui8, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [80, 80, 80, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
!OutValuesDistributed = !VPUIP.DistributedBuffer<1x1x1x13xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
!TargetShapeDistributed = !VPUIP.DistributedBuffer<1x1x1x13xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

module @VPU.SW {
  func.func nested @builtin_TopK(memref<*xf16, @CMX_NN>, memref<*xui8, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, memref<*xui8, @CMX_NN>, i64, i64, i64, i64) attributes {VPU.kernel_code = "topk.cpp", VPU.kernel_entry = "topk", VPU.kernel_name = "topk", VPU.task_type = @COMPUTE}
  func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @UnrollKernelSharedInputOutputBuffers
func.func @UnrollKernelSharedInputOutputBuffers() -> !OutValuesDistributed {
    %input = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %aux = VPURT.DeclareBuffer <CMX_NN> <1> -> !AuxDistributed
    %out_values = VPURT.DeclareBuffer <CMX_NN> <2> -> !OutValuesDistributed
    %target_shape = VPURT.DeclareBuffer <CMX_NN> <3> -> !TargetShapeDistributed

    VPURT.Task {
      %results:3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 3, 0, 0>} @VPU.SW::@builtin_TopK
        inputs(%input as %inner_input: !InputDistributed,
               %aux as %inner_aux: !AuxDistributed)
        outputs(%out_values as %inner_out_values: !OutValuesDistributed,
                %target_shape as %inner_target_shape: !TargetShapeDistributed,
                %aux as %inner_aux_alias: !AuxDistributed)
        on tile 0 -> (!OutValuesDistributed, !TargetShapeDistributed, !AuxDistributed){
        VPUIP.SW.Kernel.run {attrs = [1, 1, 2, 1]}(%inner_input, %inner_aux, %inner_out_values, %inner_target_shape, %inner_aux_alias) : !InputDistributed, !AuxDistributed, !OutValuesDistributed, !TargetShapeDistributed, !AuxDistributed
      }
    }

    return %out_values : !OutValuesDistributed

    // CHECK:      [[INPUT_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x5x13xf16, [@CMX_NN, 0]>
    // CHECK:      [[INPUT_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1x5x13xf16, [@CMX_NN, 1]>
    // CHECK:      [[AUX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1> -> memref<1x1x1x40xui8, [@CMX_NN, 0]>
    // CHECK:      [[AUX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <1> -> memref<1x1x1x40xui8, [@CMX_NN, 1]>
    // CHECK:      [[OUT_VALUES_DUPLICATED:%.+]] = VPURT.DeclareBuffer <CMX_NN> <2> -> !VPUIP.DistributedBuffer<1x1x1x13xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[OUT_VALUES_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2> -> memref<1x1x1x13xf16, [@CMX_NN, 0]>
    // CHECK:      [[OUT_VALUES_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <2> -> memref<1x1x1x13xf16, [@CMX_NN, 1]>
    // CHECK:      [[TARGET_SHAPE_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <3> -> memref<1x1x1x13xsi32, [@CMX_NN, 0]>
    // CHECK:      [[TARGET_SHAPE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <3> -> memref<1x1x1x13xsi32, [@CMX_NN, 1]>

    // CHECK:      VPURT.Task
    // CHECK-NEXT:   [[RESULTS_0:%.+]]:3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 3, 0, 0>} @VPU.SW::@builtin_TopK
    // CHECK-SAME:     inputs([[INPUT_0]] as [[INNER_INPUT_0:[^:]+]]
    // CHECK-SAME:            [[AUX_0]] as  [[INNER_AUX_0:[^:]+]]
    // CHECK-SAME:     outputs([[OUT_VALUES_0]] as [[INNER_OUT_VALUES_0:[^:]+]]
    // CHECK-SAME:             [[TARGET_SHAPE_0]] as [[INNER_TARGET_SHAPE_0:[^:]+]]
    // CHECK-SAME:             [[AUX_0]] as [[INNER_AUX_0_ALIAS:[^:]+]]
    // CHECK-SAME:     on tile 0
    // CHECK-NEXT:     VPUIP.SW.Kernel.run {attrs = [1, 1, 2, 1]}([[INNER_INPUT_0]], [[INNER_AUX_0]], [[INNER_OUT_VALUES_0]], [[INNER_TARGET_SHAPE_0]], [[INNER_AUX_0_ALIAS]])

    // CHECK:      VPURT.Task
    // CHECK-NEXT:   [[RESULTS_1:%.+]]:3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 3, 0, 0>} @VPU.SW::@builtin_TopK
    // CHECK-SAME:     inputs([[INPUT_1]] as [[INNER_INPUT_1:[^:]+]]
    // CHECK-SAME:            [[AUX_1]] as  [[INNER_AUX_1:[^:]+]]
    // CHECK-SAME:     outputs([[OUT_VALUES_1]] as [[INNER_OUT_VALUES_1:[^:]+]]
    // CHECK-SAME:             [[TARGET_SHAPE_1]] as [[INNER_TARGET_SHAPE_1:[^:]+]]
    // CHECK-SAME:             [[AUX_1]] as [[INNER_AUX_1_ALIAS:[^:]+]]
    // CHECK-SAME:     on tile 1
    // CHECK-NEXT:     VPUIP.SW.Kernel.run {attrs = [1, 1, 2, 1]}([[INNER_INPUT_1]], [[INNER_AUX_1]], [[INNER_OUT_VALUES_1]], [[INNER_TARGET_SHAPE_1]], [[INNER_AUX_1_ALIAS]])
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<1x50x42x667xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 13, 42, 667], [1, 13, 42, 667], [1, 12, 42, 667], [1, 12, 42, 667]], compute_offsets = [[0, 0, 0, 0], [0, 13, 0, 0], [0, 26, 0, 0], [0, 38, 0, 0]], memory_shapes = [[1, 13, 42, 667], [1, 13, 42, 667], [1, 12, 42, 667], [1, 12, 42, 667]], memory_offsets = [[0, 0, 0, 0], [0, 13, 0, 0], [0, 26, 0, 0], [0, 38, 0, 0]]}>
!OutputDistributed = !VPUIP.DistributedBuffer<1x50x21x334xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 13, 21, 334], [1, 13, 21, 334], [1, 12, 21, 334], [1, 12, 21, 334]], compute_offsets = [[0, 0, 0, 0], [0, 13, 0, 0], [0, 26, 0, 0], [0, 38, 0, 0]], memory_shapes = [[1, 13, 21, 334], [1, 13, 21, 334], [1, 12, 21, 334], [1, 12, 21, 334]], memory_offsets = [[0, 0, 0, 0], [0, 13, 0, 0], [0, 26, 0, 0], [0, 38, 0, 0]]}>
!OutputIndexDistributed = !VPUIP.DistributedBuffer<1x50x21x334xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 13, 21, 334], [1, 13, 21, 334], [1, 12, 21, 334], [1, 12, 21, 334]], compute_offsets = [[0, 0, 0, 0], [0, 13, 0, 0], [0, 26, 0, 0], [0, 38, 0, 0]], memory_shapes = [[1, 13, 21, 334], [1, 13, 21, 334], [1, 12, 21, 334], [1, 12, 21, 334]], memory_offsets = [[0, 0, 0, 0], [0, 13, 0, 0], [0, 26, 0, 0], [0, 38, 0, 0]]}>

module @VPU.SW {
    func.func nested @builtin_MaxPool8(memref<*xf16>, memref<*xf16>, memref<*xsi32>, none, none, none, none, none, i64, none, none, none, none) attributes {VPU.kernel_code = "max_pool8.cpp", VPU.kernel_entry = "max_pool8", VPU.kernel_name = "max_pool8", VPU.task_type = @COMPUTE}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL:   @UnrollMaxPool8DistributedOpsWithTiling
func.func @UnrollMaxPool8DistributedOpsWithTiling() -> !OutputDistributed {
    %0 = VPURT.DeclareBuffer <CMX_NN> <547136> -> !InputDistributed
    %1 = VPURT.DeclareBuffer <CMX_NN> <364736> -> !OutputDistributed
    %2 = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputIndexDistributed
    %138 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
    %140 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
    VPURT.Task waits(%138 : !VPURT.Barrier) updates(%140 : !VPURT.Barrier)  {
      %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MaxPool8
      inputs(
        %0 as %arg3: !InputDistributed)
      outputs(
        %1 as %arg4: !OutputDistributed,
        %2 as %arg5: !OutputIndexDistributed) on tile 0 -> (!OutputDistributed, !OutputIndexDistributed){
        VPUIP.SW.Kernel.run {attrs = [[1, 3, 3], [1, 2, 2], [1, 1, 1], [0, 1, 1], [0, 0, 1], 1,
        [1, 1, 50, 667, 667], [1, 1, 50, 334, 334], [0, 0, 0, 41, 0], [0, 0, 0, 21, 0]]}(%arg3, %arg4, %arg5) :
        !InputDistributed, !OutputDistributed, !OutputIndexDistributed
      }
    }
    return %1 : !OutputDistributed

    // CHECK:   [[BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <547136> -> memref<1x13x42x667xf16, [@CMX_NN, 0]>
    // CHECK:   [[BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <547136> -> memref<1x13x42x667xf16, [@CMX_NN, 1]>
    // CHECK:   [[BUFFER_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <547136> -> memref<1x12x42x667xf16, [@CMX_NN, 2]>
    // CHECK:   [[BUFFER_3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [3] <547136> -> memref<1x12x42x667xf16, [@CMX_NN, 3]>

    // CHECK:   [[BUFFER_5:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <364736> -> memref<1x13x21x334xf16, [@CMX_NN, 0]>
    // CHECK:   [[BUFFER_6:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <364736> -> memref<1x13x21x334xf16, [@CMX_NN, 1]>
    // CHECK:   [[BUFFER_7:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <364736> -> memref<1x12x21x334xf16, [@CMX_NN, 2]>
    // CHECK:   [[BUFFER_8:%.+]] = VPURT.DeclareBuffer <CMX_NN> [3] <364736> -> memref<1x12x21x334xf16, [@CMX_NN, 3]>

    // CHECK:   [[BARRIER_13:%.+]] = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
    // CHECK:   [[BARRIER_14:%.+]] = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier

    // CHECK:         VPURT.Task waits([[BARRIER_13]] : !VPURT.Barrier) updates([[BARRIER_14]] : !VPURT.Barrier)  {
    // CHECK-NEXT:      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MaxPool8
    // CHECK-SAME:      inputs([[BUFFER_0]]
    // CHECK-SAME:      outputs([[BUFFER_5]]
    // CHECK-SAME:      on tile 0
    // CHECK-NEXT:    VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}:     {attrs = [[1, 3, 3], [1, 2, 2], [1, 1, 1], [0, 1, 1], [0, 0, 1], 1, [1, 1, 50, 667, 667], [1, 1, 50, 334, 334], [0, 0, 0, 41, 0], [0, 0, 0, 21, 0]]}
    // CHECK-NEXT:    }

    // CHECK:         VPURT.Task waits([[BARRIER_13]] : !VPURT.Barrier) updates([[BARRIER_14]] : !VPURT.Barrier)  {
    // CHECK-NEXT:      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MaxPool8
    // CHECK-SAME:      inputs([[BUFFER_1]]
    // CHECK-SAME:      outputs([[BUFFER_6]]
    // CHECK-SAME:      on tile 1
    // CHECK-NEXT:    VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}:     {attrs = [[1, 3, 3], [1, 2, 2], [1, 1, 1], [0, 1, 1], [0, 0, 1], 1, [1, 1, 50, 667, 667], [1, 1, 50, 334, 334], [0, 0, 13, 41, 0], [0, 0, 13, 21, 0]]}
    // CHECK-NEXT:    }

    // CHECK:         VPURT.Task waits([[BARRIER_13]] : !VPURT.Barrier) updates([[BARRIER_14]] : !VPURT.Barrier)  {
    // CHECK-NEXT:      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MaxPool8
    // CHECK-SAME:      inputs([[BUFFER_2]]
    // CHECK-SAME:      outputs([[BUFFER_7]]
    // CHECK-SAME:      on tile 2
    // CHECK-NEXT:    VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}:     {attrs = [[1, 3, 3], [1, 2, 2], [1, 1, 1], [0, 1, 1], [0, 0, 1], 1, [1, 1, 50, 667, 667], [1, 1, 50, 334, 334], [0, 0, 26, 41, 0], [0, 0, 26, 21, 0]]}
    // CHECK-NEXT:    }

    // CHECK:         VPURT.Task waits([[BARRIER_13]] : !VPURT.Barrier) updates([[BARRIER_14]] : !VPURT.Barrier)  {
    // CHECK-NEXT:      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MaxPool8
    // CHECK-SAME:      inputs([[BUFFER_3]]
    // CHECK-SAME:      outputs([[BUFFER_8]]
    // CHECK-SAME:      on tile 3
    // CHECK-NEXT:    VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}:     {attrs = [[1, 3, 3], [1, 2, 2], [1, 1, 1], [0, 1, 1], [0, 0, 1], 1, [1, 1, 50, 667, 667], [1, 1, 50, 334, 334], [0, 0, 38, 41, 0], [0, 0, 38, 21, 0]]}
    // CHECK-NEXT:    }

}
