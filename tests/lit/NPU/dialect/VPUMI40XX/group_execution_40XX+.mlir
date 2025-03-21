//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --group-execution-ops %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

module @TestConvolution {
  IE.TileResource 1 of @NCE at 1.700000e+03 MHz {
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        IE.MemoryResource 512 bytes of @CMX_NN offset 0
      }
    }
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 1 of @DMA_NN
  IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x14x14xf16>
  }
  func.func @main(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<1x16x14x14xf16, @DDR>) -> memref<1x16x14x14xf16, @DDR> {
    %cst = const.Declare memref<1x1x1x4864xui8> = dense<1> : tensor<1x1x1x4864xui8>
    %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x16x16xf16, @DDR>
    %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x16x14x14xf16, @DDR>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x16x16xf16, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x14x14xf16, [@CMX_NN, 0]>
    %5 = VPURT.DeclareBuffer <CMX_NN> [0] <16896> -> memref<1x1x1x4864xui8, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <16896> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <17152> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    %10 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <0, -1> -> !VPURegMapped.Index<0:0:0>
    %11 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 2 : ui8}(%10 : !VPURegMapped.Index<0:0:0>) <1, -1> -> !VPURegMapped.Index<0:0:1>
    %12 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}(%11 : !VPURegMapped.Index<0:0:1>) <2, -1> -> !VPURegMapped.Index<0:0:2>
    %13 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, isFinalBarrier, producer_count = 1 : ui8}(%12 : !VPURegMapped.Index<0:0:2>) <3, -1> -> !VPURegMapped.Index<0:0:3>
    %14 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x16x16x16xf16, @DDR>) outputs(%2 : memref<1x16x16x16xf16, [@CMX_NN, 0]>) updates(%10 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %15 = VPUMI40XX.NNDMA {port = 0 : i64, is_out_of_order} inputs(%cst : memref<1x1x1x4864xui8>) outputs(%5 : memref<1x1x1x4864xui8, [@CMX_NN, 0]>) previousDMA(%14 : !VPURegMapped.Index<0:0:0>) updates(%11 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %16 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x16x14x14xf16, @DDR>) waits(%12 : !VPURegMapped.Index<0:0:2>) updates(%13 : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
    %17 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64} input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%11 : !VPURegMapped.Index<0:0:1>) -> <0:0:0> PPE : {
    }
    %18 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} previousTask(%17 : !VPURegMapped.Index<0:0:0>) input(%7 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) outputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) waits(%11 : !VPURegMapped.Index<0:0:1>) updates(%12 : !VPURegMapped.Index<0:0:2>) -> <0:0:1> PPE : {
    }
    %19 = VPUMI40XX.DPUVariant calls(%17 : <0:0:0>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:0>
    %20 = VPUMI40XX.DPUVariant previousTask(%19 : !VPURegMapped.Index<0:0:0>) calls(%18 : <0:0:1>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:1>
    %21 = VPUMI40XX.MappedInference dmas((%14, %16) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) invariants(%17 : !VPURegMapped.Index<0:0:0>) variants(%19 : !VPURegMapped.Index<0:0:0>) barriers(%10 : !VPURegMapped.Index<0:0:0>) dmaCount([[2, 1]]) invariantCount([2]) variantCount([2]) actKernelRangesCount([0]) actKernelInvocationsCount([0]) mediaCount(0) barrierCount(4) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x16x14x14xf16, @DDR>
  }
}

//CHECK:  [[VAL10:%.*]] = VPUMI40XX.ConfigureBarrier
//CHECK:  [[VAL11:%.*]] = VPUMI40XX.ConfigureBarrier
//CHECK:  [[VAL12:%.*]] = VPUMI40XX.ConfigureBarrier
//CHECK:  [[VAL13:%.*]] = VPUMI40XX.ConfigureBarrier

//CHECK: %startIndexes:2, %endIndexes:2 = "VPURegMapped.ExecutionGroup"([[VAL10]], [[VAL12]]) <{operandSegmentSizes = array<i32: 0, 1, 1>, resultSegmentSizes = array<i32: 2, 2>, task_type = #VPURegMapped.task_type<DPUInvariant>}> ({
//CHECK:  [[VAL18:%.*]] = VPUMI40XX.DPUInvariant
//CHECK:  [[VAL19:%.*]] = VPUMI40XX.DPUInvariant
//CHECK:  [[VAL20:%.*]] = VPUMI40XX.DPUVariant
//CHECK:  [[VAL21:%.*]] = VPUMI40XX.DPUVariant
//CHECK:  "VPURegMapped.GroupYield"([[VAL18]], [[VAL20]], [[VAL19]], [[VAL21]]) <{operandSegmentSizes = array<i32: 2, 2>}> : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:1>) -> ()
//CHECK:}) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:2>) -> (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:1>)


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @Convolution attributes {VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  IE.TileResource 6 of @NCE at 1.700000e+03 MHz {
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        IE.MemoryResource 512 bytes of @CMX_NN offset 0
      }
    }
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 2 of @DMA_NN
  IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x32x16x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x32x14x14xf16>
  }
  func.func @main(%arg0: memref<1x32x16x16xf16, @DDR>, %arg1: memref<1x32x14x14xf16, @DDR>) -> memref<1x32x14x14xf16, @DDR> {
    %cst = const.Declare memref<1x1x1x9472xf16> = dense<1.0> : tensor<1x1x1x9472xf16>
    %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x32x3x16xf16, {order = #NCHW, strides = [8192, 256, 16, 1]}, @DDR>
    %1 = VPURT.DeclareBuffer <NetworkInput> [0] <96> -> memref<1x32x3x16xf16, {order = #NCHW, strides = [8192, 256, 16, 1]}, @DDR>
    %6 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x32x3x14xf16, {order = #NCHW, strides = [6272, 196, 14, 1]}, @DDR>
    %7 = VPURT.DeclareBuffer <NetworkOutput> [0] <84> -> memref<1x32x3x14xf16, {order = #NCHW, strides = [6272, 196, 14, 1]}, @DDR>
    %8 = VPURT.DeclareBuffer <NetworkOutput> [0] <168> -> memref<1x32x2x14xf16, {order = #NCHW, strides = [6272, 196, 14, 1]}, @DDR>
    %10 = VPURT.DeclareBuffer <NetworkOutput> [0] <280> -> memref<1x32x2x14xf16, {order = #NCHW, strides = [6272, 196, 14, 1]}, @DDR>
    %13 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %14 = VPURT.DeclareBuffer <CMX_NN> [0] <24576> -> memref<1x32x3x16xf16, [@CMX_NN, 0]>
    %16 = VPURT.DeclareBuffer <CMX_NN> [2] <24576> -> memref<1x32x3x16xf16, [@CMX_NN, 2]>
    %18 = VPURT.DeclareBuffer <CMX_NN> [4] <24576> -> memref<1x32x2x16xf16, [@CMX_NN, 4]>
    %20 = VPURT.DeclareBuffer <CMX_NN> [0] <24576> -> memref<1x32x3x14xf16, [@CMX_NN, 0]>
    %22 = VPURT.DeclareBuffer <CMX_NN> [2] <24576> -> memref<1x32x2x14xf16, [@CMX_NN, 2]>
    %24 = VPURT.DeclareBuffer <CMX_NN> [4] <24576> -> memref<1x32x2x14xf16, [@CMX_NN, 4]>
    %26 = VPURT.DeclareBuffer <CMX_NN> [0] <24576> -> memref<1x32x3x14xf16, [@CMX_NN, 0]>
    %27 = VPURT.DeclareBuffer <CMX_NN> [1] <24576> -> memref<1x32x3x14xf16, [@CMX_NN, 1]>
    %28 = VPURT.DeclareBuffer <CMX_NN> [2] <24576> -> memref<1x32x2x14xf16, [@CMX_NN, 2]>
    %29 = VPURT.DeclareBuffer <CMX_NN> [3] <24576> -> memref<1x32x2x14xf16, [@CMX_NN, 3]>
    %30 = VPURT.DeclareBuffer <CMX_NN> [4] <24576> -> memref<1x32x2x14xf16, [@CMX_NN, 4]>
    %31 = VPURT.DeclareBuffer <CMX_NN> [5] <24576> -> memref<1x32x2x14xf16, [@CMX_NN, 5]>
    %32 = VPURT.DeclareBuffer <CMX_NN> [0] <24576> -> memref<1x16x32x3xf16, #NHWC, [@CMX_NN, 0]>
    %33 = VPURT.DeclareBuffer <CMX_NN> [1] <24576> -> memref<1x16x32x3xf16, #NHWC, [@CMX_NN, 1]>
    %34 = VPURT.DeclareBuffer <CMX_NN> [2] <24576> -> memref<1x16x32x3xf16, #NHWC, [@CMX_NN, 2]>
    %35 = VPURT.DeclareBuffer <CMX_NN> [3] <24576> -> memref<1x16x32x3xf16, #NHWC, [@CMX_NN, 3]>
    %36 = VPURT.DeclareBuffer <CMX_NN> [4] <24576> -> memref<1x16x32x2xf16, #NHWC, [@CMX_NN, 4]>
    %37 = VPURT.DeclareBuffer <CMX_NN> [5] <24576> -> memref<1x16x32x2xf16, #NHWC, [@CMX_NN, 5]>
    %38 = VPURT.DeclareBuffer <CMX_NN> [0] <24576> -> memref<1x16x32x3xf16, #NHWC, [@CMX_NN, 0]>
    %39 = VPURT.DeclareBuffer <CMX_NN> [1] <24576> -> memref<1x16x32x3xf16, #NHWC, [@CMX_NN, 1]>
    %40 = VPURT.DeclareBuffer <CMX_NN> [2] <24576> -> memref<1x16x32x3xf16, #NHWC, [@CMX_NN, 2]>
    %41 = VPURT.DeclareBuffer <CMX_NN> [3] <24576> -> memref<1x16x32x3xf16, #NHWC, [@CMX_NN, 3]>
    %42 = VPURT.DeclareBuffer <CMX_NN> [4] <24576> -> memref<1x16x32x2xf16, #NHWC, [@CMX_NN, 4]>
    %43 = VPURT.DeclareBuffer <CMX_NN> [5] <24576> -> memref<1x16x32x2xf16, #NHWC, [@CMX_NN, 5]>
    %44 = VPURT.DeclareBuffer <CMX_NN> [0] <19456> -> memref<1x32x5x16xf16, #NHWC, [@CMX_NN, 0]>
    %45 = VPURT.DeclareBuffer <CMX_NN> [1] <19456> -> memref<1x32x5x16xf16, #NHWC, [@CMX_NN, 1]>
    %46 = VPURT.DeclareBuffer <CMX_NN> [2] <19456> -> memref<1x32x4x16xf16, #NHWC, [@CMX_NN, 2]>
    %47 = VPURT.DeclareBuffer <CMX_NN> [3] <19456> -> memref<1x32x4x16xf16, #NHWC, [@CMX_NN, 3]>
    %48 = VPURT.DeclareBuffer <CMX_NN> [4] <19456> -> memref<1x32x4x16xf16, #NHWC, [@CMX_NN, 4]>
    %49 = VPURT.DeclareBuffer <CMX_NN> [5] <19456> -> memref<1x32x4x16xf16, #NHWC, [@CMX_NN, 5]>
    %50 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>
    %51 = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> memref<32x1x1x4xsi32, [@CMX_NN, 1]>
    %52 = VPURT.DeclareBuffer <CMX_NN> [2] <512> -> memref<32x1x1x4xsi32, [@CMX_NN, 2]>
    %53 = VPURT.DeclareBuffer <CMX_NN> [3] <512> -> memref<32x1x1x4xsi32, [@CMX_NN, 3]>
    %54 = VPURT.DeclareBuffer <CMX_NN> [4] <512> -> memref<32x1x1x4xsi32, [@CMX_NN, 4]>
    %55 = VPURT.DeclareBuffer <CMX_NN> [5] <512> -> memref<32x1x1x4xsi32, [@CMX_NN, 5]>
    %56 = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<32x32x3x3xf16, #NHWC, [@CMX_NN, 0]>
    %57 = VPURT.DeclareBuffer <CMX_NN> [1] <1024> -> memref<32x32x3x3xf16, #NHWC, [@CMX_NN, 1]>
    %58 = VPURT.DeclareBuffer <CMX_NN> [2] <1024> -> memref<32x32x3x3xf16, #NHWC, [@CMX_NN, 2]>
    %59 = VPURT.DeclareBuffer <CMX_NN> [3] <1024> -> memref<32x32x3x3xf16, #NHWC, [@CMX_NN, 3]>
    %60 = VPURT.DeclareBuffer <CMX_NN> [4] <1024> -> memref<32x32x3x3xf16, #NHWC, [@CMX_NN, 4]>
    %61 = VPURT.DeclareBuffer <CMX_NN> [5] <1024> -> memref<32x32x3x3xf16, #NHWC, [@CMX_NN, 5]>
    %62 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x1x1x9472xf16, [@CMX_NN, 0]>
    %63 = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> memref<1x1x1x9472xf16, [@CMX_NN, 1]>
    %64 = VPURT.DeclareBuffer <CMX_NN> [2] <512> -> memref<1x1x1x9472xf16, [@CMX_NN, 2]>
    %65 = VPURT.DeclareBuffer <CMX_NN> [3] <512> -> memref<1x1x1x9472xf16, [@CMX_NN, 3]>
    %66 = VPURT.DeclareBuffer <CMX_NN> [4] <512> -> memref<1x1x1x9472xf16, [@CMX_NN, 4]>
    %67 = VPURT.DeclareBuffer <CMX_NN> [5] <512> -> memref<1x1x1x9472xf16, [@CMX_NN, 5]>
    %68 = VPURT.DeclareBuffer <CMX_NN> [0] <19456> -> memref<1x16x32x5xf16, {order = #NWCH}, [@CMX_NN, 0]>
    %69 = VPURT.DeclareBuffer <CMX_NN> [1] <19456> -> memref<1x16x32x5xf16, {order = #NWCH}, [@CMX_NN, 1]>
    %70 = VPURT.DeclareBuffer <CMX_NN> [2] <19456> -> memref<1x16x32x4xf16, {order = #NWCH}, [@CMX_NN, 2]>
    %71 = VPURT.DeclareBuffer <CMX_NN> [3] <19456> -> memref<1x16x32x4xf16, {order = #NWCH}, [@CMX_NN, 3]>
    %72 = VPURT.DeclareBuffer <CMX_NN> [4] <19456> -> memref<1x16x32x4xf16, {order = #NWCH}, [@CMX_NN, 4]>
    %73 = VPURT.DeclareBuffer <CMX_NN> [5] <19456> -> memref<1x16x32x4xf16, {order = #NWCH}, [@CMX_NN, 5]>
    %74 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <5, -1> -> !VPURegMapped.Index<0:0:0>
    %75 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}(%74 : !VPURegMapped.Index<0:0:0>) <0, -1> -> !VPURegMapped.Index<0:0:1>
    %76 = VPUMI40XX.ConfigureBarrier {consumer_count = 6 : ui8, producer_count = 6 : ui8}(%75 : !VPURegMapped.Index<0:0:1>) <1, -1> -> !VPURegMapped.Index<0:0:2>
    %77 = VPUMI40XX.ConfigureBarrier {consumer_count = 6 : ui8, producer_count = 7 : ui8}(%76 : !VPURegMapped.Index<0:0:2>) <2, -1> -> !VPURegMapped.Index<0:0:3>
    %78 = VPUMI40XX.ConfigureBarrier {consumer_count = 6 : ui8, producer_count = 6 : ui8}(%77 : !VPURegMapped.Index<0:0:3>) <3, -1> -> !VPURegMapped.Index<0:0:4>
    %79 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, isFinalBarrier, producer_count = 2 : ui8}(%78 : !VPURegMapped.Index<0:0:4>) <4, -1> -> !VPURegMapped.Index<0:0:5>
    %80 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xi32, @DDR>
    %81 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xi32, @DDR>
    %82 = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 0 : i32, srcWidth = 0 : i32, srcStride = 0 : i32, srcPlaneStride = 0 : i32, dstWidth = 0 : i32, dstStride = 0 : i32, dstPlaneStride = 0 : i32>, port = 0 : i64} inputs(%80 : memref<1x1x1x1xi32, @DDR>) outputs(%81 : memref<1x1x1x1xi32, @DDR>) updates(%74 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %97 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64} input(%38 : memref<1x16x32x3xf16, #NHWC, [@CMX_NN, 0]>) weights(%32 : memref<1x16x32x3xf16, #NHWC, [@CMX_NN, 0]>) outputs(%68 : memref<1x16x32x5xf16, {order = #NWCH}, [@CMX_NN, 0]>) waits(%76 : !VPURegMapped.Index<0:0:2>) updates(%77 : !VPURegMapped.Index<0:0:3>) -> <0:0:0> PPE : {}
    %98 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64} input(%39 : memref<1x16x32x3xf16, #NHWC, [@CMX_NN, 1]>) weights(%33 : memref<1x16x32x3xf16, #NHWC, [@CMX_NN, 1]>) outputs(%69 : memref<1x16x32x5xf16, {order = #NWCH}, [@CMX_NN, 1]>) waits(%76 : !VPURegMapped.Index<0:0:2>) updates(%77 : !VPURegMapped.Index<0:0:3>) -> <1:0:0> PPE : {}
    %99 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64} input(%40 : memref<1x16x32x3xf16, #NHWC, [@CMX_NN, 2]>) weights(%34 : memref<1x16x32x3xf16, #NHWC, [@CMX_NN, 2]>) outputs(%70 : memref<1x16x32x4xf16, {order = #NWCH}, [@CMX_NN, 2]>) waits(%76 : !VPURegMapped.Index<0:0:2>) updates(%77 : !VPURegMapped.Index<0:0:3>) -> <2:0:0> PPE : {}
    %100 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64} input(%41 : memref<1x16x32x3xf16, #NHWC, [@CMX_NN, 3]>) weights(%35 : memref<1x16x32x3xf16, #NHWC, [@CMX_NN, 3]>) outputs(%71 : memref<1x16x32x4xf16, {order = #NWCH}, [@CMX_NN, 3]>) waits(%76 : !VPURegMapped.Index<0:0:2>) updates(%77 : !VPURegMapped.Index<0:0:3>) -> <3:0:0> PPE : {}
    %101 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64} input(%42 : memref<1x16x32x2xf16, #NHWC, [@CMX_NN, 4]>) weights(%36 : memref<1x16x32x2xf16, #NHWC, [@CMX_NN, 4]>) outputs(%72 : memref<1x16x32x4xf16, {order = #NWCH}, [@CMX_NN, 4]>) waits(%76 : !VPURegMapped.Index<0:0:2>) updates(%77 : !VPURegMapped.Index<0:0:3>) -> <4:0:0> PPE : {
    }
    %102 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64} input(%43 : memref<1x16x32x2xf16, #NHWC, [@CMX_NN, 5]>) weights(%37 : memref<1x16x32x2xf16, #NHWC, [@CMX_NN, 5]>) outputs(%73 : memref<1x16x32x4xf16, {order = #NWCH}, [@CMX_NN, 5]>) waits(%76 : !VPURegMapped.Index<0:0:2>) updates(%77 : !VPURegMapped.Index<0:0:3>) -> <5:0:0> PPE : {
    }
    %103 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} previousTask(%97 : !VPURegMapped.Index<0:0:0>) input(%44 : memref<1x32x5x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%56 : memref<32x32x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%50 : memref<32x1x1x4xsi32, [@CMX_NN, 0]>) outputs(%26 : memref<1x32x3x14xf16, [@CMX_NN, 0]>) waits(%77 : !VPURegMapped.Index<0:0:3>) updates(%78 : !VPURegMapped.Index<0:0:4>) -> <0:0:1> PPE : {
    }
    %104 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} previousTask(%98 : !VPURegMapped.Index<1:0:0>) input(%45 : memref<1x32x5x16xf16, #NHWC, [@CMX_NN, 1]>) weights(%57 : memref<32x32x3x3xf16, #NHWC, [@CMX_NN, 1]>) weight_table(%51 : memref<32x1x1x4xsi32, [@CMX_NN, 1]>) outputs(%27 : memref<1x32x3x14xf16, [@CMX_NN, 1]>) waits(%77 : !VPURegMapped.Index<0:0:3>) updates(%78 : !VPURegMapped.Index<0:0:4>) -> <1:0:1> PPE : {
    }
    %105 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} previousTask(%99 : !VPURegMapped.Index<2:0:0>) input(%46 : memref<1x32x4x16xf16, #NHWC, [@CMX_NN, 2]>) weights(%58 : memref<32x32x3x3xf16, #NHWC, [@CMX_NN, 2]>) weight_table(%52 : memref<32x1x1x4xsi32, [@CMX_NN, 2]>) outputs(%28 : memref<1x32x2x14xf16, [@CMX_NN, 2]>) waits(%77 : !VPURegMapped.Index<0:0:3>) updates(%78 : !VPURegMapped.Index<0:0:4>) -> <2:0:1> PPE : {
    }
    %106 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} previousTask(%100 : !VPURegMapped.Index<3:0:0>) input(%47 : memref<1x32x4x16xf16, #NHWC, [@CMX_NN, 3]>) weights(%59 : memref<32x32x3x3xf16, #NHWC, [@CMX_NN, 3]>) weight_table(%53 : memref<32x1x1x4xsi32, [@CMX_NN, 3]>) outputs(%29 : memref<1x32x2x14xf16, [@CMX_NN, 3]>) waits(%77 : !VPURegMapped.Index<0:0:3>) updates(%78 : !VPURegMapped.Index<0:0:4>) -> <3:0:1> PPE : {
    }
    %107 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} previousTask(%101 : !VPURegMapped.Index<4:0:0>) input(%48 : memref<1x32x4x16xf16, #NHWC, [@CMX_NN, 4]>) weights(%60 : memref<32x32x3x3xf16, #NHWC, [@CMX_NN, 4]>) weight_table(%54 : memref<32x1x1x4xsi32, [@CMX_NN, 4]>) outputs(%30 : memref<1x32x2x14xf16, [@CMX_NN, 4]>) waits(%77 : !VPURegMapped.Index<0:0:3>) updates(%78 : !VPURegMapped.Index<0:0:4>) -> <4:0:1> PPE : {
    }
    %108 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} previousTask(%102 : !VPURegMapped.Index<5:0:0>) input(%49 : memref<1x32x4x16xf16, #NHWC, [@CMX_NN, 5]>) weights(%61 : memref<32x32x3x3xf16, #NHWC, [@CMX_NN, 5]>) weight_table(%55 : memref<32x1x1x4xsi32, [@CMX_NN, 5]>) outputs(%31 : memref<1x32x2x14xf16, [@CMX_NN, 5]>) waits(%77 : !VPURegMapped.Index<0:0:3>) updates(%78 : !VPURegMapped.Index<0:0:4>) -> <5:0:1> PPE : {
    }
    %109 = VPUMI40XX.DPUVariant calls(%97 : <0:0:0>) weights(%32 : memref<1x16x32x3xf16, #NHWC, [@CMX_NN, 0]>) {end = [2, 31, 15], haloRegions = [], inEnd = [2, 31, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:0>
    %110 = VPUMI40XX.DPUVariant calls(%98 : <1:0:0>) weights(%33 : memref<1x16x32x3xf16, #NHWC, [@CMX_NN, 1]>) {cluster_id = 1 : ui64, end = [2, 31, 15], haloRegions = [#VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 0 : i64, yEnd = 1 : i64, zStart = 0 : i64, zEnd = 31 : i64, targetOffset = 3072 : i64, targetClusters = [0], targetWidth = 16 : i64>], inEnd = [2, 31, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <1:0:0>
    %111 = VPUMI40XX.DPUVariant calls(%99 : <2:0:0>) weights(%34 : memref<1x16x32x3xf16, #NHWC, [@CMX_NN, 2]>) {cluster_id = 2 : ui64, end = [2, 31, 15], haloRegions = [#VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 0 : i64, yEnd = 1 : i64, zStart = 0 : i64, zEnd = 31 : i64, targetOffset = 3072 : i64, targetClusters = [1], targetWidth = 16 : i64>, #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 2 : i64, yEnd = 2 : i64, zStart = 0 : i64, zEnd = 31 : i64, targetOffset = -2048 : i64, targetClusters = [3], targetWidth = 16 : i64>], inEnd = [2, 31, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <2:0:0>
    %112 = VPUMI40XX.DPUVariant calls(%100 : <3:0:0>) weights(%35 : memref<1x16x32x3xf16, #NHWC, [@CMX_NN, 3]>) {cluster_id = 3 : ui64, end = [3, 31, 15], haloRegions = [#VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 1 : i64, yEnd = 1 : i64, zStart = 0 : i64, zEnd = 31 : i64, targetOffset = 2048 : i64, targetClusters = [2], targetWidth = 16 : i64>, #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 2 : i64, yEnd = 3 : i64, zStart = 0 : i64, zEnd = 31 : i64, targetOffset = -2048 : i64, targetClusters = [4], targetWidth = 16 : i64>], inEnd = [2, 31, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [1, 0, 0]} -> <3:0:0>
    %113 = VPUMI40XX.DPUVariant calls(%101 : <4:0:0>) weights(%36 : memref<1x16x32x2xf16, #NHWC, [@CMX_NN, 4]>) {cluster_id = 4 : ui64, end = [3, 31, 15], haloRegions = [#VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 2 : i64, yEnd = 3 : i64, zStart = 0 : i64, zEnd = 31 : i64, targetOffset = -2048 : i64, targetClusters = [5], targetWidth = 16 : i64>], inEnd = [1, 31, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [2, 0, 0]} -> <4:0:0>
    %114 = VPUMI40XX.DPUVariant calls(%102 : <5:0:0>) weights(%37 : memref<1x16x32x2xf16, #NHWC, [@CMX_NN, 5]>) {cluster_id = 5 : ui64, end = [3, 31, 15], haloRegions = [], inEnd = [1, 31, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [2, 0, 0]} -> <5:0:0>
    %115 = VPUMI40XX.DPUVariant previousTask(%109 : !VPURegMapped.Index<0:0:0>) calls(%103 : <0:0:1>) weights(%56 : memref<32x32x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%50 : memref<32x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 2, 31], inEnd = [15, 4, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:1>
    %116 = VPUMI40XX.DPUVariant previousTask(%110 : !VPURegMapped.Index<1:0:0>) calls(%104 : <1:0:1>) weights(%57 : memref<32x32x3x3xf16, #NHWC, [@CMX_NN, 1]>) weight_table(%51 : memref<32x1x1x4xsi32, [@CMX_NN, 1]>) {cluster_id = 1 : ui64, end = [13, 2, 31], inEnd = [15, 4, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <1:0:1>
    %117 = VPUMI40XX.DPUVariant previousTask(%111 : !VPURegMapped.Index<2:0:0>) calls(%105 : <2:0:1>) weights(%58 : memref<32x32x3x3xf16, #NHWC, [@CMX_NN, 2]>) weight_table(%52 : memref<32x1x1x4xsi32, [@CMX_NN, 2]>) {cluster_id = 2 : ui64, end = [13, 1, 31], inEnd = [15, 3, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <2:0:1>
    %118 = VPUMI40XX.DPUVariant previousTask(%112 : !VPURegMapped.Index<3:0:0>) calls(%106 : <3:0:1>) weights(%59 : memref<32x32x3x3xf16, #NHWC, [@CMX_NN, 3]>) weight_table(%53 : memref<32x1x1x4xsi32, [@CMX_NN, 3]>) {cluster_id = 3 : ui64, end = [13, 1, 31], inEnd = [15, 3, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <3:0:1>
    %119 = VPUMI40XX.DPUVariant previousTask(%113 : !VPURegMapped.Index<4:0:0>) calls(%107 : <4:0:1>) weights(%60 : memref<32x32x3x3xf16, #NHWC, [@CMX_NN, 4]>) weight_table(%54 : memref<32x1x1x4xsi32, [@CMX_NN, 4]>) {cluster_id = 4 : ui64, end = [13, 1, 31], inEnd = [15, 3, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <4:0:1>
    %120 = VPUMI40XX.DPUVariant previousTask(%114 : !VPURegMapped.Index<5:0:0>) calls(%108 : <5:0:1>) weights(%61 : memref<32x32x3x3xf16, #NHWC, [@CMX_NN, 5]>) weight_table(%55 : memref<32x1x1x4xsi32, [@CMX_NN, 5]>) {cluster_id = 5 : ui64, end = [13, 1, 31], inEnd = [15, 3, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <5:0:1>
    %121 = VPUMI40XX.MappedInference dmas((%82, %82) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>)) invariants(%97, %98, %99, %100, %101, %102 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<2:0:0>, !VPURegMapped.Index<3:0:0>, !VPURegMapped.Index<4:0:0>, !VPURegMapped.Index<5:0:0>) variants(%109, %110, %111, %112, %113, %114 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<2:0:0>, !VPURegMapped.Index<3:0:0>, !VPURegMapped.Index<4:0:0>, !VPURegMapped.Index<5:0:0>) barriers(%74 : !VPURegMapped.Index<0:0:0>) dmaCount([[6, 3], [3, 3]]) invariantCount([2, 2, 2, 2, 2, 2]) variantCount([2, 2, 2, 2, 2, 2]) actKernelRangesCount([0, 0, 0, 0, 0, 0]) actKernelInvocationsCount([0, 0, 0, 0, 0, 0]) mediaCount(0) barrierCount(6) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x32x14x14xf16, @DDR>
  }
}


//CHECK:  [[VAL61:%.*]] = VPUMI40XX.ConfigureBarrier
//CHECK:  [[VAL62:%.*]] = VPUMI40XX.ConfigureBarrier
//CHECK:  [[VAL63:%.*]] = VPUMI40XX.ConfigureBarrier
//CHECK:  [[VAL64:%.*]] = VPUMI40XX.ConfigureBarrier
//CHECK:  [[VAL65:%.*]] = VPUMI40XX.ConfigureBarrier
//CHECK:  [[VAL66:%.*]] = VPUMI40XX.ConfigureBarrier
//CHECK:  %startIndexes:2, %endIndexes:2 = "VPURegMapped.ExecutionGroup"([[VAL63]], [[VAL65]])
//CHECK:  VPUMI40XX.DPUInvariant
//CHECK:  VPUMI40XX.DPUInvariant
//CHECK:  VPUMI40XX.DPUVariant
//CHECK:  VPUMI40XX.DPUVariant
//CHECK:  "VPURegMapped.GroupYield"
//CHECK:  %startIndexes_0:2, %endIndexes_1:2 = "VPURegMapped.ExecutionGroup"([[VAL63]], [[VAL65]])
//CHECK:  VPUMI40XX.DPUInvariant
//CHECK:  VPUMI40XX.DPUInvariant
//CHECK:  VPUMI40XX.DPUVariant
//CHECK:  VPUMI40XX.DPUVariant
//CHECK:  "VPURegMapped.GroupYield"
//CHECK:  %startIndexes_2:2, %endIndexes_3:2 = "VPURegMapped.ExecutionGroup"([[VAL63]], [[VAL65]])
//CHECK:  VPUMI40XX.DPUInvariant
//CHECK:  VPUMI40XX.DPUInvariant
//CHECK:  VPUMI40XX.DPUVariant
//CHECK:  VPUMI40XX.DPUVariant
//CHECK:  "VPURegMapped.GroupYield"
//CHECK:  %startIndexes_4:2, %endIndexes_5:2 = "VPURegMapped.ExecutionGroup"([[VAL63]], [[VAL65]])
//CHECK:  VPUMI40XX.DPUInvariant
//CHECK:  VPUMI40XX.DPUInvariant
//CHECK:  VPUMI40XX.DPUVariant
//CHECK:  VPUMI40XX.DPUVariant
//CHECK:  "VPURegMapped.GroupYield"
//CHECK:  %startIndexes_6:2, %endIndexes_7:2 = "VPURegMapped.ExecutionGroup"([[VAL63]], [[VAL65]])
//CHECK:  VPUMI40XX.DPUInvariant
//CHECK:  VPUMI40XX.DPUInvariant
//CHECK:  VPUMI40XX.DPUVariant
//CHECK:  VPUMI40XX.DPUVariant
//CHECK:  "VPURegMapped.GroupYield"
//CHECK:  %startIndexes_8:2, %endIndexes_9:2 = "VPURegMapped.ExecutionGroup"([[VAL63]], [[VAL65]])
//CHECK:  VPUMI40XX.DPUInvariant
//CHECK:  VPUMI40XX.DPUInvariant
//CHECK:  VPUMI40XX.DPUVariant
//CHECK:  VPUMI40XX.DPUVariant
//CHECK:  "VPURegMapped.GroupYield"
