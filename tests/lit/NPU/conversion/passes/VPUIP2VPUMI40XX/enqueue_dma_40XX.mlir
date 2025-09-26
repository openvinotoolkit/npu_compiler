//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-VPUIP-to-VPUMI40XX %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Module0 {
  net.NetworkInfo entryPoint : @EnqueueDma inputsInfo : {
    DataInfo "input_0" : tensor<1x16x3x6xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x4x6x12xf16>
  }

  func.func @EnqueueDma(%arg0: memref<1x16x3x6xf16>, %arg1: memref<1x16x3x6xf16>) -> memref<1x16x3x6xf16> {
    %bar0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %bar1 = VPURT.ConfigureBarrier<1> <{isFinalBarrier}> -> !VPURT.Barrier

    %dummy_in = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %dummy_out = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>

    %dpu_in = VPURT.DeclareBuffer <CMX_NN> [0] <9216> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %dpu_out = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]>
    %dpu_par_in = VPURT.DeclareBuffer <CMX_NN> [0] <9216> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %dpu_par_out = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]>
    %dpu_wt = VPURT.DeclareBuffer <CMX_NN> [0] <42000> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      %enq_dma =  VPUIP.EnqueueDMA {port = 0 : i64} inputs(%dummy_in : memref<0x0x0x0xi32, @DDR>) outputs(%dummy_out : memref<0x0x0x0xi32, @DDR>) enqueue_dma_attr(<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 0 : i64, endTask = 0 : i64>) -> memref<0x0x0x0xi32, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
      %dpu = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%dpu_in : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%dpu_wt : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) parent_input(%dpu_par_in : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%dpu_par_out : memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]>) outputs(%dpu_out : memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {inStart = [0, 0, 0], inEnd = [15, 15, 15], outEnd = [7, 8, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 1 : i64>, outStart = [0, 0, 0]}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }

    return %arg1 : memref<1x16x3x6xf16>
  }
}

// CHECK: VPUMI40XX.NNDMA
// CHECK-SAME: enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 0 : i64, endTask = 0 : i64>
// CHECK-SAME: port = 0 : i64
// CHECK-SAME: !VPURegMapped.Index<0:0:0>
