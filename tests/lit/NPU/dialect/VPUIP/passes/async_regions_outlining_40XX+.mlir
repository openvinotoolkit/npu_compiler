//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --async-regions-outlining="async-region-outlining-min-ops-in-block=2" %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!DistributedBufferType0 = !VPUIP.DistributedBuffer<1x32x256x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 64, 256], [1, 32, 64, 256], [1, 32, 64, 256], [1, 32, 64, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0], [0, 0, 128, 0], [0, 0, 192, 0]], memory_shapes = [[1, 32, 65, 256], [1, 32, 66, 256], [1, 32, 66, 256], [1, 32, 65, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0], [0, 0, 127, 0], [0, 0, 191, 0]]}>
!DistributedBufferType1 = !VPUIP.DistributedBuffer<1x32x256x128xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 64, 128], [1, 32, 64, 128], [1, 32, 64, 128], [1, 32, 64, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0], [0, 0, 128, 0], [0, 0, 192, 0]], memory_shapes = [[1, 32, 64, 128], [1, 32, 64, 128], [1, 32, 64, 128], [1, 32, 64, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 64, 0], [0, 0, 128, 0], [0, 0, 192, 0]]}>
module @AsyncRegionOutliningThroughDDR {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x32x256x256xf16>
    DataInfo "input1" : tensor<1x32x256x256xf16>
  } outputsInfo : {
    DataInfo "output1" : tensor<1x32x256x128xf16>
  }
  func.func @main(%arg0: memref<1x32x256x256xf16, #NHWC, @DDR>, %arg1: memref<1x32x256x256xf16, #NHWC, @DDR>, %arg2: memref<1x32x256x128xf16, #NHWC, @DDR>) -> memref<1x32x256x128xf16, #NHWC, @DDR> {
    %alloc = memref.alloc() : memref<1x32x256x256xf16, #NHWC, @DDR>
    %0 = VPURT.AllocDistributed -> !DistributedBufferType0
    %1 = VPURT.AllocDistributed -> !DistributedBufferType0
    %2 = VPURT.AllocDistributed -> !DistributedBufferType0
    %3 = VPURT.AllocDistributed -> !DistributedBufferType1
    %4 = VPURT.AllocDistributed -> !DistributedBufferType1
    // Input DDR -> CMX
    %token, %bodyResults = async.execute -> !async.value<!DistributedBufferType0> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
      %6 = VPUIP.NNDMA inputs(%arg0 : memref<1x32x256x256xf16, #NHWC, @DDR>) outputs(%0 : !DistributedBufferType0) -> !DistributedBufferType0
      async.yield %6 : !DistributedBufferType0
    }
    // Input DDR -> CMX
    %token_0, %bodyResults_1 = async.execute -> !async.value<!DistributedBufferType0> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 1 : i64} {
      %6 = VPUIP.NNDMA inputs(%arg1 : memref<1x32x256x256xf16, #NHWC, @DDR>) outputs(%1 : !DistributedBufferType0) -> !DistributedBufferType0
      async.yield %6 : !DistributedBufferType0
    }
    // NCE Op with CMX output
    %token_2, %bodyResults_3 = async.execute [%token, %token_0] (%bodyResults as %arg3: !async.value<!DistributedBufferType0>, %bodyResults_1 as %arg4: !async.value<!DistributedBufferType0>) -> !async.value<!DistributedBufferType0> attributes {VPUIP.executor = @DPU, "async-deps-index" = 2 : i64} {
      %6 = VPUIP.ViewOp %2 : !DistributedBufferType0 to !DistributedBufferType0
      %7 = VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, is_inplace = true, minimumHardwareExecutionCost = 13984 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%arg3 : !DistributedBufferType0) weights(%arg4 : !DistributedBufferType0) parent_input(%arg3 : !DistributedBufferType0) parent_output(%6 : !DistributedBufferType0) outputs(%6 : !DistributedBufferType0) -> !DistributedBufferType0 variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [255, 63, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [255, 63, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [255, 64, 31], inStart = [0, 1, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [255, 64, 31], outStart = [0, 1, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 2 : i64, inEnd = [255, 64, 31], inStart = [0, 1, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [255, 64, 31], outStart = [0, 1, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 3 : i64, inEnd = [255, 64, 31], inStart = [0, 1, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [255, 64, 31], outStart = [0, 1, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [16656], quant_shift = [30], quant_post_shift = 0 : i64, in1_quant_mult = [20834], in2_quant_mult = [54164], fp_prelu_alpha = 1.000000e+00 : f64>}
      }
      async.yield %7 : !DistributedBufferType0
    }
    // NCE output CMX -> DDR
    %token_4, %bodyResults_5 = async.execute [%token_2] (%bodyResults_3 as %arg3: !async.value<!DistributedBufferType0>) -> !async.value<memref<1x32x256x256xf16, #NHWC, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 3 : i64} {
      %6 = VPUIP.NNDMA inputs(%arg3 : !DistributedBufferType0) outputs(%alloc : memref<1x32x256x256xf16, #NHWC, @DDR>) -> memref<1x32x256x256xf16, #NHWC, @DDR>
      async.yield %6 : memref<1x32x256x256xf16, #NHWC, @DDR>
    }
    // Input DDR -> CMX
    %token_6, %bodyResults_7 = async.execute [%token_2, %token_4] (%bodyResults_5 as %arg3: !async.value<memref<1x32x256x256xf16, #NHWC, @DDR>>) -> !async.value<!DistributedBufferType1> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 4 : i64} {
      %6 = VPUIP.SubView %arg3 [0, 0, 0, 0] [1, 32, 256, 128] : memref<1x32x256x256xf16, #NHWC, @DDR> to memref<1x32x256x128xf16, {order = #NHWC, strides = [2097152, 1, 8192, 32]}, @DDR>
      %7 = VPUIP.NNDMA inputs(%6 : memref<1x32x256x128xf16, {order = #NHWC, strides = [2097152, 1, 8192, 32]}, @DDR>) outputs(%3 : !DistributedBufferType1) -> !DistributedBufferType1
      async.yield %7 : !DistributedBufferType1
    }
    // NCE Op with CMX output
    %token_8, %bodyResults_9 = async.execute [%token_6] (%bodyResults_7 as %arg3: !async.value<!DistributedBufferType1>) -> !async.value<!DistributedBufferType1> attributes {VPUIP.executor = @DPU, "async-deps-index" = 5 : i64} {
      %6 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 22748 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<AVEPOOL>} input(%arg3 : !DistributedBufferType1) parent_input(%arg3 : !DistributedBufferType1) parent_output(%4 : !DistributedBufferType1) outputs(%4 : !DistributedBufferType1) -> !DistributedBufferType1 variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [127, 63, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [127, 63, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [127, 63, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [127, 63, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 2 : i64, inEnd = [127, 63, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [127, 63, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 3 : i64, inEnd = [127, 63, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [127, 63, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [32231], quant_shift = [19], quant_post_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
      }
      async.yield %6 : !DistributedBufferType1
    }
    // NCE output CMX -> DDR
    %token_10, %bodyResults_11 = async.execute [%token_8] (%bodyResults_9 as %arg3: !async.value<!DistributedBufferType1>) -> !async.value<memref<1x32x256x128xf16, #NHWC, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 6 : i64} {
      %6 = VPUIP.NNDMA inputs(%arg3 : !DistributedBufferType1) outputs(%arg2 : memref<1x32x256x128xf16, #NHWC, @DDR>) -> memref<1x32x256x128xf16, #NHWC, @DDR>
      async.yield %6 : memref<1x32x256x128xf16, #NHWC, @DDR>
    }
    %5 = async.await %bodyResults_11 : !async.value<memref<1x32x256x128xf16, #NHWC, @DDR>>
    return %5 : memref<1x32x256x128xf16, #NHWC, @DDR>
  }

}

//CHECK-LABEL: @AsyncRegionOutliningThroughDDR

//CHECK: DataInfo "input0" : tensor<1x32x256x256xf16>
//CHECK: DataInfo "input1" : tensor<1x32x256x256xf16>
//CHECK: DataInfo "output1" : tensor<1x32x256x128xf16>

//CHECK:  func.func private @main_async_region1([[ARG0:%.+]]: memref<1x32x256x256xf16, #NHWC, @DDR>, [[ARG1:%.+]]: memref<1x32x256x256xf16, #NHWC, @DDR>, [[ARG2:%.+]]: memref<1x32x256x256xf16, #NHWC, @DDR>) -> memref<1x32x256x256xf16, #NHWC, @DDR> {
//CHECK:     [[func_output_buffer:%.*]] = memref.alloc() : memref<1x32x256x256xf16, #NHWC, @DDR>
//CHECK:     [[input_buffer_0:%.*]] = VPURT.AllocDistributed
//CHECK:     [[input_buffer_1:%.*]] = VPURT.AllocDistributed
//CHECK:     [[output_buffer:%.*]] = VPURT.AllocDistributed

//CHECK:     [[token:%.*]], [[bodyResults:%.*]] = async.execute
//CHECK:       [[NNDMA:%.*]] = VPUIP.NNDMA
//CHECK-SAME:                  inputs([[ARG0]]
//CHECK-SAME:                  outputs([[input_buffer_0]]
//CHECK:       async.yield [[NNDMA]]
//CHECK:     }

//CHECK:     [[token_0:%.*]], [[bodyResults_1:%.*]] = async.execute
//CHECK:       [[NNDMA:%.*]] = VPUIP.NNDMA
//CHECK-SAME:                  inputs([[ARG1]]
//CHECK-SAME:                  outputs([[input_buffer_1]]
//CHECK:       async.yield [[NNDMA]]
//CHECK:     }

//CHECK:     [[token_2:%.*]], [[bodyResults_3:%.*]] = async.execute [[[token]], [[token_0]]]
//CHECK-SAME:    ([[bodyResults]] as [[ARG3:%[^:]+]]
//CHECK-SAME:    [[bodyResults_1]] as [[ARG4:%[^:]+]]
//CHECK:       [[ViewOp:%.*]] = VPUIP.ViewOp [[output_buffer]]
//CHECK:       [[NCEOp:%.*]] = VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, is_inplace = true, minimumHardwareExecutionCost = 13984 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}
//CHECK-SAME:                 input([[ARG3]]
//CHECK-SAME:                 weights([[ARG4]]
//CHECK-SAME:                 parent_input([[ARG3]]
//CHECK-SAME:                 parent_output([[ViewOp]]
//CHECK-SAME:                 outputs([[ViewOp]]
//CHECK:       async.yield [[NCEOp]]

//CHECK:     [[token_4:%.*]], [[bodyResults_5:%.*]] = async.execute [[[token_2]]] ([[bodyResults_3]] as [[ARG3:%[^:]+]]
//CHECK:       [[NNDMA:%.*]] = VPUIP.NNDMA
//CHECK-SAME:                  inputs([[ARG3]]
//CHECK-SAME:                  outputs([[func_output_buffer]]
//CHECK:       async.yield [[NNDMA]]
//CHECK:     }

//CHECK:     [[token_6:%.*]], [[bodyResults_7:%.*]] = async.execute [[[token_4]]] ([[bodyResults_5]] as [[ARG3:%[^:]+]]
//CHECK:       [[NNDMA:%.*]] = VPUIP.NNDMA
//CHECK-SAME:                  inputs([[ARG3]]
//CHECK-SAME:                  outputs([[ARG2]]
//CHECK:       async.yield [[NNDMA]]
//CHECK:     }

//CHECK:     [[func_output:%.*]] = async.await [[bodyResults_7]] : !async.value<memref<1x32x256x256xf16, #NHWC, @DDR>>
//CHECK:     return [[func_output]]

//CHECK:   func.func private @main_async_region2([[ARG0:%.+]]: memref<1x32x256x256xf16, #NHWC, @DDR>, [[ARG1:%.+]]: memref<1x32x256x128xf16, #NHWC, @DDR>)
//CHECK-SAME:   -> memref<1x32x256x128xf16, #NHWC, @DDR> {
//CHECK:     [[NCE_INPUT:%.*]] = VPURT.AllocDistributed
//CHECK:     [[NCE_OUTPUT_CMX:%.*]] = VPURT.AllocDistributed
//CHECK:     [[NCE_OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x32x256x128xf16, #NHWC, @DDR>

//CHECK:     [[token:%.*]], [[bodyResults:%.*]] = async.execute
//CHECK:       [[SUBVIEW:%.*]] = VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [1, 32, 256, 128]
//CHECK:       [[NNDMA:%.*]] = VPUIP.NNDMA
//CHECK-SAME:       inputs([[SUBVIEW]]
//CHECK-SAME:       outputs([[NCE_INPUT]]
//CHECK:       async.yield [[NNDMA]]
//CHECK:     }

//CHECK:     [[token_0:%.*]], [[bodyResults_1:%.*]] = async.execute [[[token]]] ([[bodyResults]] as [[ARG2:%[^:]+]]
//CHECK:       [[NCEOp:%.*]] = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 22748 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<AVEPOOL>}
//CHECK-SAME:       input([[ARG2]]
//CHECK-SAME:       parent_input([[ARG2]]
//CHECK-SAME:       parent_output([[NCE_OUTPUT_CMX]]
//CHECK-SAME:       outputs([[NCE_OUTPUT_CMX]]
//CHECK:       async.yield [[NCEOp]]
//CHECK:     }

//CHECK:     [[token_2:%.*]], [[bodyResults_3:%.*]] = async.execute [[[token_0]]] ([[bodyResults_1]] as [[ARG2:%[^:]+]]
//CHECK:       [[NNDMA:%.*]] = VPUIP.NNDMA
//CHECK-SAME:       inputs([[ARG2]]
//CHECK-SAME:       outputs([[NCE_OUTPUT_DDR]]
//CHECK:       async.yield [[NNDMA]]
//CHECK:     }

//CHECK:     [[token_4:%.*]], [[bodyResults_5:%.*]] = async.execute [[[token_2]]] ([[bodyResults_3]] as [[ARG2:%[^:]+]]
//CHECK:       [[NNDMA:%.*]] = VPUIP.NNDMA
//CHECK-SAME:       inputs([[ARG2]]
//CHECK-SAME:       outputs([[ARG1]]
//CHECK:       async.yield [[NNDMA]]
//CHECK:     }
//CHECK:     [[func_output:%.*]] = async.await [[bodyResults_5]]
//CHECK:     return [[func_output]]

//CHECK:  func.func @main([[ARG0:%.+]]: memref<1x32x256x256xf16, #NHWC, @DDR>, [[ARG1:%.+]]: memref<1x32x256x256xf16, #NHWC, @DDR>, [[ARG2:%.+]]: memref<1x32x256x128xf16, #NHWC, @DDR>) -> memref<1x32x256x128xf16, #NHWC, @DDR> {
//CHECK:    [[FUNC1_OUTPUT_BUFFER:%.*]] = memref.alloc() : memref<1x32x256x256xf16, #NHWC, @DDR>
//CHECK:    [[FUNC2_OUTPUT_BUFFER:%.*]] = memref.alloc() : memref<1x32x256x128xf16, #NHWC, @DDR>
//CHECK:    [[SPILL_READ_DMA:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x256x128xf16, #NHWC, @CMX_NN
//CHECK:    [[token:%.*]], [[bodyResults:%.*]] = async.execute -> !async.value<memref<1x32x256x256xf16, #NHWC, @DDR>> attributes {VPUIP.executor = @NCE, "async-deps-index" = 0 : i64} {
//CHECK:      [[FUNC_RES:%.*]] = func.call @main_async_region1([[ARG0]], [[ARG1]], [[FUNC1_OUTPUT_BUFFER]]) : (memref<1x32x256x256xf16, #NHWC, @DDR>, memref<1x32x256x256xf16, #NHWC, @DDR>, memref<1x32x256x256xf16, #NHWC, @DDR>) -> memref<1x32x256x256xf16, #NHWC, @DDR>
//CHECK:      async.yield [[FUNC_RES]] : memref<1x32x256x256xf16, #NHWC, @DDR>
//CHECK:    }

//CHECK:    [[token_1:%.*]], [[bodyResults_2:%.*]] = async.execute [[[token]]] ([[bodyResults]] as [[ARG3:%[^:]+]]: !async.value<memref<1x32x256x256xf16, #NHWC, @DDR>>) -> !async.value<memref<1x32x256x128xf16, #NHWC, @DDR>>
//CHECK-SAME:     attributes {VPUIP.executor = @NCE, "async-deps-index" = 1 : i64} {
//CHECK:      [[FUNC_RES:%.*]] = func.call @main_async_region2([[ARG3]], [[FUNC2_OUTPUT_BUFFER]]) : (memref<1x32x256x256xf16, #NHWC, @DDR>, memref<1x32x256x128xf16, #NHWC, @DDR>) -> memref<1x32x256x128xf16, #NHWC, @DDR>
//CHECK:      async.yield [[FUNC_RES]] : memref<1x32x256x128xf16, #NHWC, @DDR>
//CHECK:    }
//CHECK:   [[token_3:%.*]], [[bodyResults_4:%.*]] = async.execute [[[token_1]]] ([[bodyResults_2]] as [[ARG3:%[^:]+]]: !async.value<memref<1x32x256x128xf16, #NHWC, @DDR>>) -> !async.value<!VPUIP.DistributedBuffer<1x32x256x128xf16, #NHWC, @CMX_NN
//CHECK:      [[NNDMA:%.*]] = VPUIP.NNDMA
//CHECK-SAME:                  inputs([[ARG3]]
//CHECK-SAME:                  outputs([[SPILL_READ_DMA]]
//CHECK:      async.yield [[NNDMA]] : !VPUIP.DistributedBuffer<1x32x256x128xf16, #NHWC, @CMX_NN
//CHECK:    }
//CHECK:    [[token_5:%.*]], [[bodyResults_6:%.*]] = async.execute [[[token_3]]] ([[bodyResults_4]] as [[ARG3:%[^:]+]]: !async.value<!VPUIP.DistributedBuffer<1x32x256x128xf16, #NHWC, @CMX_NN
//CHECK:      [[NNDMA:%.*]] = VPUIP.NNDMA
//CHECK-SAME:                  inputs([[ARG3]]
//CHECK-SAME:                  outputs([[ARG2]]
//CHECK:      async.yield [[NNDMA]] : memref<1x32x256x128xf16, #NHWC, @DDR>
//CHECK:    }

//CHECK:    [[FUNC_OUTPUT:%.*]] = async.await [[bodyResults_6]] : !async.value<memref<1x32x256x128xf16, #NHWC, @DDR>>
//CHECK:    return [[FUNC_OUTPUT]] : memref<1x32x256x128xf16, #NHWC, @DDR>
//CHECK:  }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!DistributedBufferType0 = !VPUIP.DistributedBuffer<1x32x256x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 64, 256], [1, 32, 64, 256], [1, 32, 64, 256], [1, 32, 64, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0], [0, 0, 128, 0], [0, 0, 192, 0]], memory_shapes = [[1, 32, 65, 256], [1, 32, 66, 256], [1, 32, 66, 256], [1, 32, 65, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0], [0, 0, 127, 0], [0, 0, 191, 0]]}>
module @DoNotSplitDueToLessOpsThanMinOpsInBlock {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x32x256x256xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x32x256x256xf16>
  }
  func.func @main(%arg0: memref<1x32x256x256xf16, #NHWC, @DDR>, %arg1: memref<1x32x256x256xf16, #NHWC, @DDR>) -> memref<1x32x256x256xf16, #NHWC, @DDR> {
    %token, %bodyResults = async.execute -> !async.value<memref<1x32x256x256xf16, #NHWC, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
      %6 = VPUIP.NNDMA inputs(%arg0 : memref<1x32x256x256xf16, #NHWC, @DDR>) outputs(%arg1 : memref<1x32x256x256xf16, #NHWC, @DDR>) -> memref<1x32x256x256xf16, #NHWC, @DDR>
      async.yield %6 : memref<1x32x256x256xf16, #NHWC, @DDR>
    }
    %0 = async.await %bodyResults : !async.value<memref<1x32x256x256xf16, #NHWC, @DDR>>
    return %0 : memref<1x32x256x256xf16, #NHWC, @DDR>
  }
}

//CHECK-LABEL: @DoNotSplitDueToLessOpsThanMinOpsInBlock

//CHECK: DataInfo "input" : tensor<1x32x256x256xf16>
//CHECK: DataInfo "output" : tensor<1x32x256x256xf16>

//CHECK-NOT: func.func private @main_async_region

//CHECK: func.func @main([[ARG0:%.+]]: memref<1x32x256x256xf16, #NHWC, @DDR>, [[ARG1:%.+]]: memref<1x32x256x256xf16, #NHWC, @DDR>) -> memref<1x32x256x256xf16, #NHWC, @DDR> {
//CHECK-NOT: func.call @main_async_region
//CHECK: return
