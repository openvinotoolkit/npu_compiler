//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true ppe-version=IntPPE" --calculate-async-region-cycle-cost  %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Distributed0 = !VPUIP.DistributedBuffer<1x16x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
!Distributed1 = !VPUIP.DistributedBuffer<1x1x1x4864xui8, {order = #NCHW, strides = [4864, 4864, 4864, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
!MemRef1 = memref<1x16x112x112xf16, #NHWC, @CMX_NN>
!MemRef0 = memref<1x16x112x112xf16, #NHWC>

// CHECK-LABEL: module @AddCycleCostForDistributedBuffers
module @AddCycleCostForDistributedBuffers attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
  config.Resources 2 of @NCE at 1.300000e+03 MHz {
      config.ExecutorResource 1 of @DPU
      config.ExecutorResource 2 of @SHAVE_ACT
      config.ExecutorResource 1 of @SHAVE_NN
      config.MemoryResource 1784217 bytes of @CMX_NN_FragmentationAware
      config.MemoryResource 1982464 bytes of @CMX_NN {config.bandwidth = 32 : i64, config.derateFactor = 1.000000e+00 : f64}
  }
  config.ExecutorResource 2 of @DMA_NN

func.func @main(%arg0: memref<1x112x112x16xf16, @DDR>, %arg1: memref<1x112x112x16xf16, @DDR>) -> memref<1x112x112x16xf16, @DDR> {
    %cst = const.Declare memref<1x1x1x4864xui8> = dense<1>  : tensor<1x1x1x4864xui8>
    %0 = VPURT.AllocDistributed -> !Distributed0
    %1 = VPURT.AllocDistributed -> !Distributed0
    %2 = VPURT.AllocDistributed -> !Distributed1
    %token, %bodyResults = async.execute -> !async.value<!Distributed0> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
        %3 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs(%arg0 : memref<1x112x112x16xf16, @DDR>) -> memref<1x16x112x112xf16, #NHWC, @DDR>
        %4 = VPUIP.Copy inputs(%3 : memref<1x16x112x112xf16, #NHWC, @DDR>) outputs(%0 : !Distributed0) -> !Distributed0
        async.yield %4 : !Distributed0
    }
    %token_0, %bodyResults_1 = async.execute -> !async.value<!Distributed1> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 1 : i64} {
        %4 = VPUIP.Copy inputs(%cst : memref<1x1x1x4864xui8>) outputs(%2 : !Distributed1) -> !Distributed1
        async.yield %4 : !Distributed1
    }
    %token_2, %bodyResults_3 = async.execute [%token, %token_0] (%bodyResults as %arg2: !async.value<!Distributed0>, %bodyResults_1 as %arg3: !async.value<!Distributed1>) -> !async.value<!Distributed0> attributes {VPUIP.executor = @DPU, "async-deps-index" = 2 : i64} {
        %3 = VPUIP.SubView %arg3 [0, 0, 0, 0] [1, 1, 1, 256] : !Distributed1 to !VPUIP.DistributedBuffer<1x1x1x256xui8, {order = #NCHW, strides = [4864, 4864, 4864, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        %4 = VPUIP.SubView %arg3 [0, 0, 0, 256] [1, 1, 1, 4608] : !Distributed1 to !VPUIP.DistributedBuffer<1x1x1x4608xui8, {order = #NCHW, strides = [4864, 4864, 4864, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        %5 = VPUIP.ViewOp %3 : !VPUIP.DistributedBuffer<1x1x1x256xui8, {order = #NCHW, strides = [4864, 4864, 4864, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        %6 = VPUIP.ViewOp %4 : !VPUIP.DistributedBuffer<1x1x1x4608xui8, {order = #NCHW, strides = [4864, 4864, 4864, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        %7 = VPUIP.NCEClusterTask {constantsFused = true, kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1],
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,minimumHardwareExecutionCost = 23244 : i64, task_type = #VPUIP.nce_task_type<CONV>} input(%arg2 : !Distributed0) weights(%6 : !VPUIP.DistributedBuffer<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) weight_table(%5 : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) parent_input(%arg2 : !Distributed0) parent_output(%1 : !Distributed0) outputs(%1 : !Distributed0) -> !Distributed0 variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 111, 15], outStart = [0, 56, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>}
        } PPE : {
            PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>}
        }
        async.yield %7 : !Distributed0
    }
    %token_4, %bodyResults_5 = async.execute [%token_2] (%bodyResults_3 as %arg2: !async.value<!Distributed0>) -> !async.value<memref<1x16x112x112xf16, #NHWC, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 3 : i64} {
        %3 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs(%arg1 : memref<1x112x112x16xf16, @DDR>) -> memref<1x16x112x112xf16, #NHWC, @DDR>
        %4 = VPUIP.Copy inputs(%arg2 : !Distributed0) outputs(%3 : memref<1x16x112x112xf16, #NHWC, @DDR>) -> memref<1x16x112x112xf16, #NHWC, @DDR>
        async.yield %4 : memref<1x16x112x112xf16, #NHWC, @DDR>
    }

    // CHECK: [[T1:%.+]], [[F1:%.+]] = async.execute -> !async.value
    // CHECK-SAME: attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64, cycleCost = 6075 : i64}
    // CHECK: [[T2:%.+]], [[F2:%.+]] = async.execute ->
    // CHECK-SAME: {VPUIP.executor = @DMA_NN, "async-deps-index" = 1 : i64, cycleCost = 719 : i64}
    // CHECK: async.execute [[[T1]], [[T2]]] ([[F1]] as %arg2: !async.value
    // CHECK-SAME: [[F2]] as %arg3:
    // CHECK-SAME: VPUIP.executor = @DPU, "async-deps-index" = 2 : i64, cycleCost = 15549 : i64

    return %arg1 : memref<1x112x112x16xf16, @DDR>
  }
}
