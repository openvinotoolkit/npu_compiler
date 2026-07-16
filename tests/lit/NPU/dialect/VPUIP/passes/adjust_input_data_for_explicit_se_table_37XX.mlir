//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --adjust-input-data-for-explicit-se-table %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SparseConvSETable(%arg0: memref<1x48x10x10xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg1: memref<1x48x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x48x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]> {
  %cst_weights = const.Declare memref<48x48x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<48x48x1x1xf16, {order = #NHWC}>
  %cst_weights_table = const.Declare memref<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
  %cst_input_sm = const.Declare memref<1x48x12x21xi1, {order = #NHWC}> = dense<1> : tensor<1x48x12x21xi1, {order = #NHWC}>
  %cst_input_se = const.Declare memref<1x3x12x21xi32, {order = #NHWC}> = dense<1> : tensor<1x3x12x21xi32, {order = #NHWC}>

  %barrier = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

  %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x48x6x10xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %input_sm = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x48x12x21xi1, {order = #NHWC}, [@CMX_NN, 0]>
  %input_se = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x12x21xi32, {order = #NHWC}, [@CMX_NN, 0]>
  %output = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x48x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %parent_input = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x48x10x10xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
  %parent_input_sm = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x48x21x21xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
  %parent_input_se = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x3x21x21xi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>
  %parent_output = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x48x20x20xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
  %weights = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<48x48x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %weights_table = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<48x1x1x4xsi32, [@CMX_NN, 0]>

  VPURT.Task updates(%barrier : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %out = VPUIP.NNDMA inputs(%cst_weights : memref<48x48x1x1xf16, {order = #NHWC}>) outputs(%weights : memref<48x48x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<48x48x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
  }
  VPURT.Task updates(%barrier : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %out = VPUIP.NNDMA inputs(%cst_weights_table : memref<48x1x1x4xsi32>) outputs(%weights_table : memref<48x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<48x1x1x4xsi32, [@CMX_NN, 0]>
  }
  VPURT.Task updates(%barrier : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %out = VPUIP.NNDMA inputs(%cst_input_sm : memref<1x48x12x21xi1, {order = #NHWC}>) outputs(%input_sm : memref<1x48x12x21xi1, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x48x12x21xi1, {order = #NHWC}, [@CMX_NN, 0]>
  }
  VPURT.Task updates(%barrier : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %out = VPUIP.NNDMA inputs(%cst_input_se : memref<1x3x12x21xi32, {order = #NHWC}>) outputs(%input_se : memref<1x3x12x21xi32, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x3x12x21xi32, {order = #NHWC}, [@CMX_NN, 0]>
  }

  VPURT.Task waits(%barrier : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %out = VPUIP.NCEClusterTask {ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{input_se_size = 16 : i64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
            input(%input : memref<1x48x6x10xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            input_sparsity_map(%input_sm : memref<1x48x12x21xi1, {order = #NHWC}, [@CMX_NN, 0]>)
            input_storage_element_table(%input_se : memref<1x3x12x21xi32, {order = #NHWC}, [@CMX_NN, 0]>)
            weights(%weights : memref<48x48x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weight_table(%weights_table : memref<48x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%parent_input : !VPUIP.DistributedBuffer<1x48x10x10xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
            parent_input_sparsity_map(%parent_input_sm : !VPUIP.DistributedBuffer<1x48x21x21xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
            parent_input_storage_element_table(%parent_input_se : !VPUIP.DistributedBuffer<1x3x21x21xi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>)
            parent_output(%parent_output : !VPUIP.DistributedBuffer<1x48x20x20xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
            outputs(%output : memref<1x48x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    -> memref<1x48x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 2, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
  }

  return %arg1 : memref<1x48x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>

  // CHECK:       VPUIP.NCEClusterTask
  // CHECK-NOT:      input({{%.+}} : memref<1x48x6x10xf16, {order = #NHWC}, [@CMX_NN, 0]>)
  // CHECK:          input({{%.+}} : memref<1x48x12x21xf16, {order = #NHWC}, [@CMX_NN, 0]>)
  // CHECK-NOT:      parent_input({{%.+}} : !VPUIP.DistributedBuffer<1x48x10x10xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
  // CHECK:          parent_input({{%.+}} : !VPUIP.DistributedBuffer<1x48x21x21xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInput = !VPUIP.DistributedBuffer<
    1x16x80x144xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 40, 144], [1, 16, 40, 144]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 40, 0]],
    memory_shapes = [[1, 16, 44, 144], [1, 16, 36, 144]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 44, 0]]
}>

!ParentInputSM = !VPUIP.DistributedBuffer<
    1x16x161x289xi1, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    alignment = [1, 1, 8, 1],
    compute_shapes = [[1, 16, 88, 289], [1, 16, 73, 289]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 88, 0]],
    memory_shapes = [[1, 16, 88, 289], [1, 16, 73, 289]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 88, 0]]
}>

!ParentInputSET = !VPUIP.DistributedBuffer<
    1x1x161x289xi32, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    alignment = [1, 1, 8, 1],
    compute_shapes = [[1, 1, 88, 289], [1, 1, 73, 289]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 88, 0]],
    memory_shapes = [[1, 1, 88, 289], [1, 1, 73, 289]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 88, 0]]
}>

!ParentOutput = !VPUIP.DistributedBuffer<
    1x16x160x288xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 80, 288], [1, 16, 80, 288]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 80, 0]],
    memory_shapes = [[1, 16, 80, 288], [1, 16, 80, 288]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 80, 0]]
}>

func.func @SparseConvSETableWithExplictDistributed() -> memref<1x16x80x288xf16, {order = #NHWC}, [@CMX_NN, 0]> {
    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x36x144xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %input_sm = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x73x289xi1, {order = #NHWC}, [@CMX_NN, 0]>
    %input_se = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x73x289xi32, {order = #NHWC}, [@CMX_NN, 0]>
    %weights = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<16x16x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %weights_table = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x80x288xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %parent_input = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInput
    %parent_input_sm = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInputSM
    %parent_input_se = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInputSET
    %parent_output = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentOutput

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %out = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{input_se_size = 16 : i64, is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
              input(%input : memref<1x16x36x144xf16, {order = #NHWC}, [@CMX_NN, 0]>)
              input_sparsity_map(%input_sm : memref<1x16x73x289xi1, {order = #NHWC}, [@CMX_NN, 0]>)
              input_storage_element_table(%input_se : memref<1x1x73x289xi32, {order = #NHWC}, [@CMX_NN, 0]>)
              weights(%weights : memref<16x16x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]>)
              weight_table(%weights_table : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
              parent_input(%parent_input : !ParentInput)
              parent_input_sparsity_map(%parent_input_sm : !ParentInputSM)
              parent_input_storage_element_table(%parent_input_se : !ParentInputSET)
              parent_output(%parent_output : !ParentOutput)
              outputs(%output : memref<1x16x80x288xf16, {order = #NHWC}, [@CMX_NN, 0]>)
      -> memref<1x16x80x288xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [287, 159, 15], outStart = [0, 80, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }

    return %output : memref<1x16x80x288xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-NOT:      input({{%.+}} : memref<1x16x36x144xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:          input({{%.+}} : memref<1x16x73x289xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NOT:      parent_input({{%.+}} : !VPUIP.DistributedBuffer<1x16x80x144xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = {{\[\[}}1, 16, 40, 144], [1, 16, 40, 144]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 40, 0]], memory_shapes = {{\[\[}}1, 16, 44, 144], [1, 16, 36, 144]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 44, 0]]}>)
    // CHECK:          parent_input({{%.+}} : !VPUIP.DistributedBuffer<1x16x161x289xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = {{\[\[}}1, 16, 88, 289], [1, 16, 73, 289]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 88, 0]], memory_shapes = {{\[\[}}1, 16, 88, 289], [1, 16, 73, 289]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 88, 0]]}>)
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputType = memref<1x32x19x128xf16, {order = #NHWC}, [@CMX_NN, 1]>

!ParentInputSM = !VPUIP.DistributedBuffer<
    1x96x37x256xi1, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    compute_shapes = [[1, 64, 37, 256], [1, 32, 37, 256]],
    compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]],
    memory_shapes = [[1, 64, 37, 256], [1, 32, 37, 256]],
    memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]
}>

!ParentInputSET = !VPUIP.DistributedBuffer<
    1x2x37x256xi32, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 1, 1, 1],
    compute_shapes = [[1, 1, 37, 256], [1, 1, 37, 256]],
    compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
    memory_shapes = [[1, 1, 37, 256], [1, 1, 37, 256]],
    memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]
}>

!OutputType = memref<1x32x37x256xf16, {order = #NHWC}, [@CMX_NN, 1]>

func.func @SparseDWConvSETableWithExplictDistributedSOK() -> !OutputType {
    %input_sm = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x32x37x256xi1, {order = #NHWC}, [@CMX_NN, 1]>
    %input_se = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1x37x256xi32, {order = #NHWC}, [@CMX_NN, 1]>
    %weights = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<32x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>
    %weights_table = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<32x1x1x4xsi32, [@CMX_NN, 1]>

    %input = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> !InputType
    %parent_input_sm = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInputSM
    %parent_input_se = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInputSET
    %output = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> !OutputType

    VPURT.Task {
      %out = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{input_se_size = 32 : i64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<DWCONV>}>
              input(%input : !InputType)
              input_sparsity_map(%input_sm : memref<1x32x37x256xi1, {order = #NHWC}, [@CMX_NN, 1]>)
              input_storage_element_table(%input_se : memref<1x1x37x256xi32, {order = #NHWC}, [@CMX_NN, 1]>)
              weights(%weights : memref<32x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>)
              weight_table(%weights_table : memref<32x1x1x4xsi32, [@CMX_NN, 1]>)
              parent_input(%input : !InputType)
              parent_input_sparsity_map(%parent_input_sm : !ParentInputSM)
              parent_input_storage_element_table(%parent_input_se : !ParentInputSET)
              parent_output(%output : !OutputType)
              outputs(%output : !OutputType)
      -> !OutputType variants : {
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [255, 36, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }

    return %output : !OutputType

    // CHECK:       [[IN_VAL:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x32x37x256xf16, {order = #NHWC}, [@CMX_NN, 1]>
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:     input([[IN_VAL]] : memref<1x32x37x256xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    // CHECK-SAME:     parent_input([[IN_VAL]] : memref<1x32x37x256xf16, {order = #NHWC}, [@CMX_NN, 1]>)
}
