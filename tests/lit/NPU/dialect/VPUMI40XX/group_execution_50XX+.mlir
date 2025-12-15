//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true enable-sw-kernel-fifo-per-shave-engine=true" --group-execution-ops %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

module @TestConvolution {
  config.Resources 1 of @NCE at 1.700000e+03 MHz {
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        config.MemoryResource 512 bytes of @CMX_NN offset 0
      }
    }
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
  config.ExecutorResource 1 of @M2I
  config.ExecutorResource 1 of @DMA_NN
  config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  net.NetworkInfo entryPoint : @main inputsInfo : {
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
    %21 = VPUMI40XX.MappedInference dmas((%14, %16) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) invariants(%17 : !VPURegMapped.Index<0:0:0>) variants(%19 : !VPURegMapped.Index<0:0:0>) barriers(%10 : !VPURegMapped.Index<0:0:0>) dmaCount([[2, 1]]) invariantCount([2]) variantCount([2]) actKernelRangesCount([[0, 0]]) actKernelInvocationsCount([[0, 0]]) mediaCount(0) barrierCount(4) -> !VPURegMapped.Index<0:0:0>
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
module @Convolution attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
  config.Resources 6 of @NCE at 1.700000e+03 MHz {
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        config.MemoryResource 512 bytes of @CMX_NN offset 0
      }
    }
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
  config.ExecutorResource 1 of @M2I
  config.ExecutorResource 2 of @DMA_NN
  config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  net.NetworkInfo entryPoint : @main inputsInfo : {
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
    %121 = VPUMI40XX.MappedInference dmas((%82, %82) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>)) invariants(%97, %98, %99, %100, %101, %102 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<2:0:0>, !VPURegMapped.Index<3:0:0>, !VPURegMapped.Index<4:0:0>, !VPURegMapped.Index<5:0:0>) variants(%109, %110, %111, %112, %113, %114 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<2:0:0>, !VPURegMapped.Index<3:0:0>, !VPURegMapped.Index<4:0:0>, !VPURegMapped.Index<5:0:0>) barriers(%74 : !VPURegMapped.Index<0:0:0>) dmaCount([[6, 3], [3, 3]]) invariantCount([2, 2, 2, 2, 2, 2]) variantCount([2, 2, 2, 2, 2, 2]) actKernelRangesCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) actKernelInvocationsCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) mediaCount(0) barrierCount(6) -> !VPURegMapped.Index<0:0:0>
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

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

module @TestSoftmax {
  config.Resources 2 of @NCE at 1.700000e+03 MHz {
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        config.MemoryResource 512 bytes of @CMX_NN offset 0
      }
    }
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
  config.ExecutorResource 1 of @M2I
  config.ExecutorResource 1 of @DMA_NN
  config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1000x1x1xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1000x1x1xf16>
  }
  func.func @main(%arg0: memref<1x1000x1x1xf16, @DDR>, %arg1: memref<1x1000x1x1xf16, @DDR>) -> memref<1x1000x1x1xf16, @DDR> {
  %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
  %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x1000x1x1xf16, @DDR>
  %2 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
  %3 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
  %4 = VPURT.DeclareBuffer <CMX_NN> [0] <1473536> -> memref<16xui32, [@CMX_NN, 0]>
  %5 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %6 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %7 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %8 = VPURT.DeclareBuffer <CMX_NN> [1] <2048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %9 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %10 = VPURT.DeclareBuffer <CMX_NN> [1] <2048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %11 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %12 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %13 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %14 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %15 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %16 = VPURT.DeclareBuffer <CMX_NN> [1] <2048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %17 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %18 = VPURT.DeclareBuffer <CMX_NN> [1] <2048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %19 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %20 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %21 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1000x1x1xf16, [@CMX_NN, 0]>
  %22 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %23 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  // buffers for KernelParams of ActKernelInvocation ops in tile 0, list 1:
  %71 = VPURT.DeclareBuffer <CMX_NN> [0] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %72 = VPURT.DeclareBuffer <CMX_NN> [0] <6048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %73 = VPURT.DeclareBuffer <CMX_NN> [0] <6048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %74 = VPURT.DeclareBuffer <CMX_NN> [0] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %75 = VPURT.DeclareBuffer <CMX_NN> [0] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %76 = VPURT.DeclareBuffer <CMX_NN> [0] <6048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %77 = VPURT.DeclareBuffer <CMX_NN> [0] <6048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %78 = VPURT.DeclareBuffer <CMX_NN> [0] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  // buffers for KernelParams of ActKernelInvocation ops in tile 1, list 1:
  %79 = VPURT.DeclareBuffer <CMX_NN> [1] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %80 = VPURT.DeclareBuffer <CMX_NN> [1] <6048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %81 = VPURT.DeclareBuffer <CMX_NN> [1] <6048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %82 = VPURT.DeclareBuffer <CMX_NN> [1] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %83 = VPURT.DeclareBuffer <CMX_NN> [1] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %84 = VPURT.DeclareBuffer <CMX_NN> [1] <6048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %85 = VPURT.DeclareBuffer <CMX_NN> [1] <6048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %86 = VPURT.DeclareBuffer <CMX_NN> [1] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>

  %87 = VPURT.DeclareBuffer <CMX_NN> [0] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %88 = VPURT.DeclareBuffer <CMX_NN> [1] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>

  %24 = VPUMI40XX.DeclareKernelText kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
  %25 = VPUMI40XX.DeclareKernelEntry kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
  %26 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
  %89 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:1:0>
  %27 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:0:0>
  %90 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:1:0>
  %28 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:0:1>
  %91 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:1:1>
  %29 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:0:1>
  %92 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:1:1>
  %30 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:0:2>
  %93 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:1:2>
  %31 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:0:2>
  %94 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:1:2>
  %32 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:0:3>
  %95 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:1:3>
  %33 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:0:3>
  %96 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:1:3>
  %34 = VPUMI40XX.KernelParams inputs(%5 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%9 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:0:0>
  %97 = VPUMI40XX.KernelParams inputs(%71 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%72 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:1:0>
  %35 = VPUMI40XX.KernelParams inputs(%6 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%10 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:0:0>
  %98 = VPUMI40XX.KernelParams inputs(%79 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%80 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:1:0>
  %36 = VPUMI40XX.KernelParams inputs(%7 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%13 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:0:1>
  %99 = VPUMI40XX.KernelParams inputs(%73 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%74 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:1:1>
  %37 = VPUMI40XX.KernelParams inputs(%8 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%14 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:0:1>
  %100 = VPUMI40XX.KernelParams inputs(%81 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%82 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:1:1>
  %38 = VPUMI40XX.KernelParams inputs(%11 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%17 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:0:2>
  %101 = VPUMI40XX.KernelParams inputs(%75 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%76 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:1:2>
  %39 = VPUMI40XX.KernelParams inputs(%12 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%18 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:0:2>
  %102 = VPUMI40XX.KernelParams inputs(%83 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%84 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:1:2>
  %40 = VPUMI40XX.KernelParams inputs(%15 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%19 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:0:3>
  %103 = VPUMI40XX.KernelParams inputs(%77 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%78 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:1:3>
  %41 = VPUMI40XX.KernelParams inputs(%16 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%20 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:0:3>
  %104 = VPUMI40XX.KernelParams inputs(%85 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%86 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:1:3>
  %42 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, isStartBarrier, producer_count = 1 : ui8} <0, -1> -> !VPURegMapped.Index<0:0:0>
  %43 = VPUMI40XX.ConfigureBarrier {consumer_count = 4 : ui8, producer_count = 1 : ui8}(%42 : !VPURegMapped.Index<0:0:0>) <1, -1> -> !VPURegMapped.Index<0:0:1>
  %44 = VPUMI40XX.ConfigureBarrier {consumer_count = 4 : ui8, producer_count = 4 : ui8}(%43 : !VPURegMapped.Index<0:0:1>) <2, -1> -> !VPURegMapped.Index<0:0:2>
  %45 = VPUMI40XX.ConfigureBarrier {consumer_count = 4 : ui8, producer_count = 4 : ui8}(%44 : !VPURegMapped.Index<0:0:2>) <3, -1> -> !VPURegMapped.Index<0:0:3>
  %46 = VPUMI40XX.ConfigureBarrier {consumer_count = 4 : ui8, producer_count = 4 : ui8}(%45 : !VPURegMapped.Index<0:0:3>) <4, -1> -> !VPURegMapped.Index<0:0:4>
  %47 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 4 : ui8}(%46 : !VPURegMapped.Index<0:0:4>) <5, -1> -> !VPURegMapped.Index<0:0:5>
  %48 = VPUMI40XX.ConfigureBarrier {consumer_count = 0 : ui8, isFinalBarrier, producer_count = 1 : ui8}(%47 : !VPURegMapped.Index<0:0:5>) <6, -1> -> !VPURegMapped.Index<0:0:6>
  %49 = VPUMI40XX.ActKernelRange kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%26 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
  %105 = VPUMI40XX.ActKernelRange kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%89 : !VPURegMapped.Index<0:1:0>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:1:0>
  %50 = VPUMI40XX.ActKernelRange kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%27 : !VPURegMapped.Index<1:0:0>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:0:0>
  %106 = VPUMI40XX.ActKernelRange kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%90 : !VPURegMapped.Index<1:1:0>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:1:0>
  %51 = VPUMI40XX.ActKernelRange previousTask(%49 : !VPURegMapped.Index<0:0:0>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%28 : !VPURegMapped.Index<0:0:1>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:1>
  %107 = VPUMI40XX.ActKernelRange previousTask(%105 : !VPURegMapped.Index<0:1:0>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%91 : !VPURegMapped.Index<0:1:1>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:1:1>
  %52 = VPUMI40XX.ActKernelRange previousTask(%50 : !VPURegMapped.Index<1:0:0>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%29 : !VPURegMapped.Index<1:0:1>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:0:1>
  %108 = VPUMI40XX.ActKernelRange previousTask(%106 : !VPURegMapped.Index<1:1:0>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%92 : !VPURegMapped.Index<1:1:1>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:1:1>
  %53 = VPUMI40XX.ActKernelRange previousTask(%51 : !VPURegMapped.Index<0:0:1>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%30 : !VPURegMapped.Index<0:0:2>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:2>
  %109 = VPUMI40XX.ActKernelRange previousTask(%107 : !VPURegMapped.Index<0:1:1>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%93 : !VPURegMapped.Index<0:1:2>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:1:2>
  %54 = VPUMI40XX.ActKernelRange previousTask(%52 : !VPURegMapped.Index<1:0:1>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%31 : !VPURegMapped.Index<1:0:2>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:0:2>
  %110 = VPUMI40XX.ActKernelRange previousTask(%108 : !VPURegMapped.Index<1:1:1>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%94 : !VPURegMapped.Index<1:1:2>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:1:2>
  %55 = VPUMI40XX.ActKernelRange previousTask(%53 : !VPURegMapped.Index<0:0:2>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%32 : !VPURegMapped.Index<0:0:3>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:3>
  %111 = VPUMI40XX.ActKernelRange previousTask(%109 : !VPURegMapped.Index<0:1:2>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%95 : !VPURegMapped.Index<0:1:3>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:1:3>
  %56 = VPUMI40XX.ActKernelRange previousTask(%54 : !VPURegMapped.Index<1:0:2>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%33 : !VPURegMapped.Index<1:0:3>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:0:3>
  %112 = VPUMI40XX.ActKernelRange previousTask(%110 : !VPURegMapped.Index<1:1:2>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%96 : !VPURegMapped.Index<1:1:3>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:1:3>
  %57 = VPUMI40XX.ActKernelInvocation range_index(%49 : <0:0:0>) kernel_params(%34 : <0:0:0>) waits(%43 : !VPURegMapped.Index<0:0:1>) updates(%44 : !VPURegMapped.Index<0:0:2>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
  %113 = VPUMI40XX.ActKernelInvocation range_index(%105 : <0:1:0>) kernel_params(%97 : <0:1:0>) waits(%43 : !VPURegMapped.Index<0:0:1>) updates(%44 : !VPURegMapped.Index<0:0:2>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:1:0>
  %58 = VPUMI40XX.ActKernelInvocation range_index(%50 : <1:0:0>) kernel_params(%35 : <1:0:0>) waits(%43 : !VPURegMapped.Index<0:0:1>) updates(%44 : !VPURegMapped.Index<0:0:2>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:0>
  %114 = VPUMI40XX.ActKernelInvocation range_index(%106 : <1:1:0>) kernel_params(%98 : <1:1:0>) waits(%43 : !VPURegMapped.Index<0:0:1>) updates(%44 : !VPURegMapped.Index<0:0:2>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:1:0>
  %59 = VPUMI40XX.ActKernelInvocation previousTask(%57 : !VPURegMapped.Index<0:0:0>) range_index(%51 : <0:0:1>) kernel_params(%36 : <0:0:1>) waits(%44 : !VPURegMapped.Index<0:0:2>) updates(%45 : !VPURegMapped.Index<0:0:3>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
  %115 = VPUMI40XX.ActKernelInvocation previousTask(%113 : !VPURegMapped.Index<0:1:0>) range_index(%107 : <0:1:1>) kernel_params(%99 : <0:1:1>) waits(%44 : !VPURegMapped.Index<0:0:2>) updates(%45 : !VPURegMapped.Index<0:0:3>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:1:1>
  %60 = VPUMI40XX.ActKernelInvocation previousTask(%58 : !VPURegMapped.Index<1:0:0>) range_index(%52 : <1:0:1>) kernel_params(%37 : <1:0:1>) waits(%44 : !VPURegMapped.Index<0:0:2>) updates(%45 : !VPURegMapped.Index<0:0:3>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:1>
  %116 = VPUMI40XX.ActKernelInvocation previousTask(%114 : !VPURegMapped.Index<1:1:0>) range_index(%108 : <1:1:1>) kernel_params(%100 : <1:1:1>) waits(%44 : !VPURegMapped.Index<0:0:2>) updates(%45 : !VPURegMapped.Index<0:0:3>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:1:1>
  %61 = VPUMI40XX.ActKernelInvocation previousTask(%59 : !VPURegMapped.Index<0:0:1>) range_index(%53 : <0:0:2>) kernel_params(%38 : <0:0:2>) waits(%45 : !VPURegMapped.Index<0:0:3>) updates(%46 : !VPURegMapped.Index<0:0:4>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:2>
  %117 = VPUMI40XX.ActKernelInvocation previousTask(%115 : !VPURegMapped.Index<0:1:1>) range_index(%109 : <0:1:2>) kernel_params(%101 : <0:1:2>) waits(%45 : !VPURegMapped.Index<0:0:3>) updates(%46 : !VPURegMapped.Index<0:0:4>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:1:2>
  %62 = VPUMI40XX.ActKernelInvocation previousTask(%60 : !VPURegMapped.Index<1:0:1>) range_index(%54 : <1:0:2>) kernel_params(%39 : <1:0:2>) waits(%45 : !VPURegMapped.Index<0:0:3>) updates(%46 : !VPURegMapped.Index<0:0:4>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:2>
  %118 = VPUMI40XX.ActKernelInvocation previousTask(%116 : !VPURegMapped.Index<1:1:1>) range_index(%110 : <1:1:2>) kernel_params(%102 : <1:1:2>) waits(%45 : !VPURegMapped.Index<0:0:3>) updates(%46 : !VPURegMapped.Index<0:0:4>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:1:2>
  %63 = VPUMI40XX.ActKernelInvocation previousTask(%61 : !VPURegMapped.Index<0:0:2>) range_index(%55 : <0:0:3>) kernel_params(%40 : <0:0:3>) waits(%46 : !VPURegMapped.Index<0:0:4>) updates(%47 : !VPURegMapped.Index<0:0:5>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:3>
  %119 = VPUMI40XX.ActKernelInvocation previousTask(%117 : !VPURegMapped.Index<0:1:2>) range_index(%111 : <0:1:3>) kernel_params(%103 : <0:1:3>) waits(%46 : !VPURegMapped.Index<0:0:4>) updates(%47 : !VPURegMapped.Index<0:0:5>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:1:3>
  %64 = VPUMI40XX.ActKernelInvocation previousTask(%62 : !VPURegMapped.Index<1:0:2>) range_index(%56 : <1:0:3>) kernel_params(%41 : <1:0:3>) waits(%46 : !VPURegMapped.Index<0:0:4>) updates(%47 : !VPURegMapped.Index<0:0:5>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:3>
  %120 = VPUMI40XX.ActKernelInvocation previousTask(%118 : !VPURegMapped.Index<1:1:2>) range_index(%112 : <1:1:3>) kernel_params(%104 : <1:1:3>) waits(%46 : !VPURegMapped.Index<0:0:4>) updates(%47 : !VPURegMapped.Index<0:0:5>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:1:3>
  %65 = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64} inputs(%2 : memref<0x0x0x0xi32, @DDR>) outputs(%3 : memref<0x0x0x0xi32, @DDR>) updates(%42 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
  %66 = VPUMI40XX.NNDMA {is_out_of_order, port = 0 : i64} inputs(%0 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) outputs(%22, %23, %87, %88 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>, memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>, memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>, memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) previousDMA(%65 : !VPURegMapped.Index<0:0:0>) waits(%42 : !VPURegMapped.Index<0:0:0>) updates(%43 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>, outputType = !VPUIP.DistributedBuffer<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 1000, 1, 1], [1, 1000, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1000, 1, 1], [1, 1000, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>>) -> !VPURegMapped.Index<0:0:1>
  %67 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%21 : memref<1x1000x1x1xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x1000x1x1xf16, @DDR>) waits(%47 : !VPURegMapped.Index<0:0:5>) updates(%48 : !VPURegMapped.Index<0:0:6>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x1000x1x1xf16, [@CMX_NN, 0]>, outputType = memref<1x1000x1x1xf16, @DDR>>) -> !VPURegMapped.Index<0:1:0>
  %68 = VPUMI40XX.PlatformInfo -> <0:0:0>
  %69 = VPUMI40XX.ActShaveRt kernel("nnActEntry") -> !VPURegMapped.Index<0:0:0>
  %70 = VPUMI40XX.MappedInference dmas((%65, %67) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) actKernelRanges((%49, %105), (%50, %106) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>), (!VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<1:1:0>)) actKernelInvocations((%57, %113), (%58, %114) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>), (!VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<1:1:0>)) barriers(%42 : !VPURegMapped.Index<0:0:0>) actShaveRt(%69 : !VPURegMapped.Index<0:0:0>) dmaHwpBase(%4 : memref<16xui32, [@CMX_NN, 0]>) dmaCount([[2, 1], [0, 0]]) invariantCount([0, 0]) variantCount([0, 0]) actKernelRangesCount([[4, 4], [4, 4]]) actKernelInvocationsCount([[4, 4], [4, 4]]) mediaCount(0) barrierCount(7) -> !VPURegMapped.Index<0:0:0>
  return %arg1 : memref<1x1000x1x1xf16, @DDR>
  }
}

//CHECK:  [[VAL42:%.+]] = VPUMI40XX.ConfigureBarrier
//CHECK:  [[VAL43:%.+]] = VPUMI40XX.ConfigureBarrier
//CHECK:  [[VAL44:%.+]] = VPUMI40XX.ConfigureBarrier
//CHECK:  [[VAL45:%.+]] = VPUMI40XX.ConfigureBarrier
//CHECK:  [[VAL46:%.+]] = VPUMI40XX.ConfigureBarrier
//CHECK:  [[VAL47:%.+]] = VPUMI40XX.ConfigureBarrier
//CHECK:  [[VAL48:%.+]] = VPUMI40XX.ConfigureBarrier
//CHECK:  "VPURegMapped.ExecutionGroup"([[VAL43]], [[VAL47]]) <{operandSegmentSizes = array<i32: 0, 1, 1>, resultSegmentSizes = array<i32: 2, 2>, task_type = #VPURegMapped.task_type<ActKernelRange>}> ({
//CHECK:  [[VAL55:%.+]] = VPUMI40XX.ActKernelRange
//CHECK-SAME: !VPURegMapped.Index<0:0:0>
//CHECK:  [[VAL56:%.+]] = VPUMI40XX.ActKernelRange
//CHECK-SAME: !VPURegMapped.Index<0:0:1>
//CHECK:  [[VAL57:%.+]] = VPUMI40XX.ActKernelRange
//CHECK-SAME: !VPURegMapped.Index<0:0:2>
//CHECK:  [[VAL58:%.+]] = VPUMI40XX.ActKernelRange
//CHECK-SAME: !VPURegMapped.Index<0:0:3>
//CHECK:  [[VAL59:%.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: !VPURegMapped.Index<0:0:0>
//CHECK:  [[VAL60:%.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: !VPURegMapped.Index<0:0:1>
//CHECK:  [[VAL61:%.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: !VPURegMapped.Index<0:0:2>
//CHECK:  [[VAL62:%.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: !VPURegMapped.Index<0:0:3>
//CHECK:  "VPURegMapped.GroupYield"([[VAL55]], [[VAL59]], [[VAL58]], [[VAL62]]) <{operandSegmentSizes = array<i32: 2, 2>}> : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:3>, !VPURegMapped.Index<0:0:3>) -> ()
//CHECK:}) : (!VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:5>) -> (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:3>, !VPURegMapped.Index<0:0:3>)

//CHECK:  "VPURegMapped.ExecutionGroup"([[VAL43]], [[VAL47]]) <{operandSegmentSizes = array<i32: 0, 1, 1>, resultSegmentSizes = array<i32: 2, 2>, task_type = #VPURegMapped.task_type<ActKernelRange>}> ({
//CHECK:  [[VAL55:%.+]] = VPUMI40XX.ActKernelRange
//CHECK-SAME: !VPURegMapped.Index<0:1:0>
//CHECK:  [[VAL56:%.+]] = VPUMI40XX.ActKernelRange
//CHECK-SAME: !VPURegMapped.Index<0:1:1>
//CHECK:  [[VAL57:%.+]] = VPUMI40XX.ActKernelRange
//CHECK-SAME: !VPURegMapped.Index<0:1:2>
//CHECK:  [[VAL58:%.+]] = VPUMI40XX.ActKernelRange
//CHECK-SAME: !VPURegMapped.Index<0:1:3>
//CHECK:  [[VAL59:%.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: !VPURegMapped.Index<0:1:0>
//CHECK:  [[VAL60:%.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: !VPURegMapped.Index<0:1:1>
//CHECK:  [[VAL61:%.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: !VPURegMapped.Index<0:1:2>
//CHECK:  [[VAL62:%.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: !VPURegMapped.Index<0:1:3>
//CHECK:  "VPURegMapped.GroupYield"([[VAL55]], [[VAL59]], [[VAL58]], [[VAL62]]) <{operandSegmentSizes = array<i32: 2, 2>}> : (!VPURegMapped.Index<0:1:0>, !VPURegMapped.Index<0:1:0>, !VPURegMapped.Index<0:1:3>, !VPURegMapped.Index<0:1:3>) -> ()
//CHECK:}) : (!VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:5>) -> (!VPURegMapped.Index<0:1:0>, !VPURegMapped.Index<0:1:0>, !VPURegMapped.Index<0:1:3>, !VPURegMapped.Index<0:1:3>)

//CHECK:  "VPURegMapped.ExecutionGroup"([[VAL43]], [[VAL47]]) <{operandSegmentSizes = array<i32: 0, 1, 1>, resultSegmentSizes = array<i32: 2, 2>, task_type = #VPURegMapped.task_type<ActKernelRange>}> ({
//CHECK:  [[VAL55:%.+]] = VPUMI40XX.ActKernelRange
//CHECK-SAME: !VPURegMapped.Index<1:0:0>
//CHECK:  [[VAL56:%.+]] = VPUMI40XX.ActKernelRange
//CHECK-SAME: !VPURegMapped.Index<1:0:1>
//CHECK:  [[VAL57:%.+]] = VPUMI40XX.ActKernelRange
//CHECK-SAME: !VPURegMapped.Index<1:0:2>
//CHECK:  [[VAL58:%.+]] = VPUMI40XX.ActKernelRange
//CHECK-SAME: !VPURegMapped.Index<1:0:3>
//CHECK:  [[VAL59:%.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: !VPURegMapped.Index<1:0:0>
//CHECK:  [[VAL60:%.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: !VPURegMapped.Index<1:0:1>
//CHECK:  [[VAL61:%.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: !VPURegMapped.Index<1:0:2>
//CHECK:  [[VAL62:%.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: !VPURegMapped.Index<1:0:3>

//CHECK:  "VPURegMapped.ExecutionGroup"([[VAL43]], [[VAL47]]) <{operandSegmentSizes = array<i32: 0, 1, 1>, resultSegmentSizes = array<i32: 2, 2>, task_type = #VPURegMapped.task_type<ActKernelRange>}> ({
//CHECK:  [[VAL55:%.+]] = VPUMI40XX.ActKernelRange
//CHECK-SAME: !VPURegMapped.Index<1:1:0>
//CHECK:  [[VAL56:%.+]] = VPUMI40XX.ActKernelRange
//CHECK-SAME: !VPURegMapped.Index<1:1:1>
//CHECK:  [[VAL57:%.+]] = VPUMI40XX.ActKernelRange
//CHECK-SAME: !VPURegMapped.Index<1:1:2>
//CHECK:  [[VAL58:%.+]] = VPUMI40XX.ActKernelRange
//CHECK-SAME: !VPURegMapped.Index<1:1:3>
//CHECK:  [[VAL59:%.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: !VPURegMapped.Index<1:1:0>
//CHECK:  [[VAL60:%.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: !VPURegMapped.Index<1:1:1>
//CHECK:  [[VAL61:%.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: !VPURegMapped.Index<1:1:2>
//CHECK:  [[VAL62:%.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: !VPURegMapped.Index<1:1:3>
