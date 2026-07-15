//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @MultiOpRanges attributes {config.compilationMode = #config.compilation_mode<DefaultHW>, config.platform = #config.platform<NPU4000>} {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
        config.ExecutorResource 2 of @SHAVE_ACT
        config.ExecutorResource 1 of @DPU
    }
    config.ExecutorResource 2 of @DMA_NN
    config.MemoryResource 2306867200 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input_0" : tensor<1x2x3x4xf16>
    } outputsInfo : {
        DataInfo "output_0" : tensor<1x2x3x4xf16>
    }
    func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
        %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
        %1 = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
        %2 = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) previousDMA(%1 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
        %3 = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs(%0 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%0 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
        %4 = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs(%0 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%0 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) previousDMA(%3 : !VPURegMapped.Index<0:1:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:1>
        %5 = VPUMI40XX.NNDMA <{port = 1 : i64}> inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<1:0:0>
        %6 = VPUMI40XX.NNDMA <{port = 1 : i64}> inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) previousDMA(%5 : !VPURegMapped.Index<1:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<1:0:1>
        %7 = VPUMI40XX.NNDMA <{port = 1 : i64}> inputs(%0 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%0 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<1:1:0>
        %8 = VPUMI40XX.NNDMA <{port = 1 : i64}> inputs(%0 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%0 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) previousDMA(%7 : !VPURegMapped.Index<1:1:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<1:1:1>
        VPUMI40XX.OpRanges types([#VPURegMapped.task_type<DMA>, #VPURegMapped.task_type<DMA>, #VPURegMapped.task_type<DMA>, #VPURegMapped.task_type<DMA>]) begins(%1, %3, %5, %7 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>, !VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<1:1:0>) ends(%2, %4, %6, %8 : !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:1:1>, !VPURegMapped.Index<1:0:1>, !VPURegMapped.Index<1:1:1>)
    }
}
