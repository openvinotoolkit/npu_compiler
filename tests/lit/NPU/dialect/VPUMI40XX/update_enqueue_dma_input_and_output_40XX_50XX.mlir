//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --update-enqueue-dma-input-and-output %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

module @UpdateEnqueueDMAOpsDPU attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
  config.PipelineOptions @Options {
    config.Option @config.DpuFIFOAddrs : [788529152, 788529184, 788529216, 788529248, 788529280, 788529312, 788529344, 788529376]
    config.Option @config.ShvFIFOAddrs : [788578304, 788578336, 788578368, 788578400, 788578432, 788578464, 788578496, 788578528, 788578560, 788578592, 788578624, 788578656, 788578688, 788578720, 788578752, 788578784]
    config.Option @config.BarrierFIFOAddr : 788594688 : ui64
  }
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

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>

    %10 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8, wlmPage = 0 : i64} <4, -1> -> !VPURegMapped.Index<0:0:0>
    %11 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8, wlmPage = 0 : i64}(%10 : !VPURegMapped.Index<0:0:0>) <0, -1> -> !VPURegMapped.Index<0:0:1>
    %12 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 2 : ui8, wlmPage = 0 : i64}(%11 : !VPURegMapped.Index<0:0:1>) <1, -1> -> !VPURegMapped.Index<0:0:2>
    %13 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8, wlmPage = 0 : i64}(%12 : !VPURegMapped.Index<0:0:2>) <2, -1> -> !VPURegMapped.Index<0:0:3>
    %14 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, isFinalBarrier, producer_count = 1 : ui8, wlmPage = 0 : i64}(%13 : !VPURegMapped.Index<0:0:3>) <3, -1> -> !VPURegMapped.Index<0:0:4>

    %15 = VPUMI40XX.DeclareTaskBuffer {offset = 51200 : ui64} <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
    %16 = VPUMI40XX.DeclareTaskBuffer {offset = 51200 : ui64} <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
    %17 = VPUMI40XX.DeclareTaskBuffer {offset = 51200 : ui64} <DPUVariant> -> !VPURegMapped.Index<0:0:0>
    %18 = VPUMI40XX.DeclareTaskBuffer {offset = 51200 : ui64} <DPUVariant> -> !VPURegMapped.Index<0:0:1>
    %19 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64, wlmPage = 0 : i64} taskLocation(%15 : !VPURegMapped.Index<0:0:0>) input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%11 : !VPURegMapped.Index<0:0:1>) updates(%12 : !VPURegMapped.Index<0:0:2>) -> <0:0:0> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %20 = VPUMI40XX.DPUInvariant {clean_after = 3 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 4 : ui64, wlmPage = 0 : i64} taskLocation(%16 : !VPURegMapped.Index<0:0:1>) previousTask(%19 : !VPURegMapped.Index<0:0:0>) input(%7 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) outputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) waits(%12 : !VPURegMapped.Index<0:0:2>) updates(%13 : !VPURegMapped.Index<0:0:3>) -> <0:0:1> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %21 = VPUMI40XX.DPUVariant taskLocation(%17 : !VPURegMapped.Index<0:0:0>) calls(%19 : <0:0:0>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], wlmPage = 0, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:0>
    %22 = VPUMI40XX.DPUVariant taskLocation(%18 : !VPURegMapped.Index<0:0:1>) previousTask(%21 : !VPURegMapped.Index<0:0:0>) calls(%20 : <0:0:1>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], wlmPage = 0, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], taskLinkAttrName = #VPURegMapped.IndexType<<0:0:0>>} -> <0:0:1>
    %23 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xi32, @DDR>
    %24 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xi32, @DDR>
    %25 = VPURegMapped.ViewTaskRange(%19 -> %20 : <0:0:0> -> <0:0:1>) -> memref<2x352xui8>
    %26 = VPURegMapped.ViewTaskRange(%15 -> %16 : <0:0:0> -> <0:0:1>) -> memref<2x352xui8, [@CMX_NN, 0]>
    %27 = VPURegMapped.ViewTaskRange(%21 -> %22 : <0:0:0> -> <0:0:1>) -> memref<2x224xui8>
    %28 = VPURegMapped.ViewTaskRange(%17 -> %18 : <0:0:0> -> <0:0:1>) -> memref<2x224xui8, [@CMX_NN, 0]>
    %29 = VPUMI40XX.NNDMA {is_critical, is_out_of_order, port = 0 : i64, wlmPage = -1 : i64} inputs(%25 : memref<2x352xui8>) outputs(%26 : memref<2x352xui8, [@CMX_NN, 0]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %30 = VPUMI40XX.NNDMA {is_critical, is_out_of_order, port = 0 : i64, wlmPage = -1 : i64} inputs(%27 : memref<2x224xui8>) outputs(%28 : memref<2x224xui8, [@CMX_NN, 0]>) previousDMA(%29 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %31 = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 0 : i32, srcWidth = 0 : i32, srcStride = 0 : i32, srcPlaneStride = 0 : i32, dstWidth = 0 : i32, dstStride = 0 : i32, dstPlaneStride = 0 : i32>, port = 0 : i64, wlmPage = 0 : i64} inputs(%23 : memref<1x1x1x1xi32, @DDR>) outputs(%24 : memref<1x1x1x1xi32, @DDR>) previousDMA(%30 : !VPURegMapped.Index<0:0:1>) updates(%10 : !VPURegMapped.Index<0:0:0>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
    %32 = VPUMI40XX.NNDMA {port = 0 : i64, wlmPage = 0 : i64} inputs(%0 : memref<1x16x16x16xf16, @DDR>) outputs(%2 : memref<1x16x16x16xf16, [@CMX_NN, 0]>) previousDMA(%31 : !VPURegMapped.Index<0:0:2>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%11 : !VPURegMapped.Index<0:0:1>) start_after(2) clean_after(1) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>

    %enq_dma = VPUMI40XX.NNDMA {enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 0 : i64, endTask = 1 : i64>, port = 0 : i64} inputs(%buf0 : memref<0x0x0x0xi32, @DDR>) outputs(%buf1 : memref<0x0x0x0xi32, @DDR>) previousDMA(%32 : !VPURegMapped.Index<0:0:3>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%11 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:4>

    %33 = VPUMI40XX.NNDMA {is_out_of_order, port = 0 : i64, wlmPage = 0 : i64} inputs(%cst : memref<1x1x1x4864xui8>) outputs(%5 : memref<1x1x1x4864xui8, [@CMX_NN, 0]>) previousDMA(%enq_dma : !VPURegMapped.Index<0:0:4>) updates(%12 : !VPURegMapped.Index<0:0:2>) start_after(3) clean_after(2) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:5>
    %34 = VPUMI40XX.NNDMA {port = 0 : i64, wlmPage = 0 : i64} inputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x16x14x14xf16, @DDR>) waits(%13 : !VPURegMapped.Index<0:0:3>) updates(%14 : !VPURegMapped.Index<0:0:4>) start_after(5) clean_after(4) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>

    %35 = VPURegMapped.Enqueue (%29 -> %29 : <0:0:0> -> <0:0:0>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<DMA>}
    %36 = VPUMI40XX.MappedInference dmas((%29, %34) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) invariants(%19 : !VPURegMapped.Index<0:0:0>) variants(%21 : !VPURegMapped.Index<0:0:0>) barriers(%10 : !VPURegMapped.Index<0:0:0>) workItemTasks(%35 : !VPURegMapped.Index<0:0:0>) dmaCount([[6, 1]]) invariantCount([2]) variantCount([2]) actKernelRangesCount([[0, 0]]) actKernelInvocationsCount([[0, 0]]) mediaCount(0) barrierCount(5) workItemCount(1) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x16x14x14xf16, @DDR>
  }
}

// CHECK: [[EQDMA_CST:%.+]] = const.Declare memref<1xui32> = dense<2080> : tensor<1xui32>
// CHECK: [[EQDMA_REG_BUF:%.+]] = VPURT.DeclareBuffer <Register> <788529152> -> memref<1xui32, @Register>

// CHECK: VPUMI40XX.NNDMA
// CHECK: VPUMI40XX.NNDMA
// CHECK: VPUMI40XX.NNDMA
// CHECK: VPUMI40XX.NNDMA

// DPU Enqueue
// CHECK: [[EQDMA_0:%.+]] = VPUMI40XX.NNDMA {
// CHECK-SAME: dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 4 : i64, srcWidth = 4 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 4 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>
// CHECK-SAME: enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 0 : i64, endTask = 1 : i64>
// CHECK-SAME: inputs([[EQDMA_CST]] : memref<1xui32>) outputs([[EQDMA_REG_BUF]] : memref<1xui32, @Register>)

// DMA Enqueue
// CHECK: [[EQ_0:%.+]] = VPURegMapped.Enqueue
// CHECK-NOT: VPURegMapped.Enqueue
// CHECK: workItemTasks([[EQ_0]] : !VPURegMapped.Index<0:0:0>)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @UpdateEnqueueDMAOpsSHV attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
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
    DataInfo "input" : tensor<1x1000x1x1xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1000x1x1xf16>
  }
  func.func @main(%arg0: memref<1x1000x1x1xf16, @DDR>, %arg1: memref<1x1000x1x1xf16, @DDR>) -> memref<1x1000x1x1xf16, @DDR> {
    %cst = const.Declare memref<64xui32> = dense<[16842753, 0, 0, 0, 16908289, 0, 0, 0, 16908290, 0, 0, 0, 65793, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<64xui32>
    %0 = VPURT.DeclareBuffer <Register> <788594688> -> memref<64xui32, @Register>
    %1 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<3x3844xf16, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x62x62xf16, @DDR>
    %3 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %4 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %5 = VPURT.DeclareBuffer <CMX_NN> [0] <1473536> -> memref<16xui32, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<3844x3xf16, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <23104> -> memref<3844x3xf16, [@CMX_NN, 0]>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x62x62xf16, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<3x3844xf16, [@CMX_NN, 0]>
    %10 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x32x62xf16, {order = #NHWC, strides = [11532, 1, 186, 3]}, [@CMX_NN, 0]>
    %11 = VPURT.DeclareBuffer <CMX_NN> [0] <23104> -> memref<1x3x32x62xf16, {order = #NHWC, strides = [11532, 1, 186, 3]}, [@CMX_NN, 0]>
    %12 = VPURT.DeclareBuffer <CMX_NN> [0] <11904> -> memref<1x3x30x62xf16, {order = #NHWC, strides = [11532, 1, 186, 3]}, [@CMX_NN, 0]>
    %13 = VPURT.DeclareBuffer <CMX_NN> [0] <35008> -> memref<1x3x30x62xf16, {order = #NHWC, strides = [11532, 1, 186, 3]}, [@CMX_NN, 0]>

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>

    %14 = VPUMI40XX.DeclareKernelText kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
    %15 = VPUMI40XX.DeclareKernelEntry kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
    %16 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
    %17 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:0:1>
    %18 = VPUMI40XX.KernelParams inputs(%10 : memref<1x3x32x62xf16, {order = #NHWC, strides = [11532, 1, 186, 3]}, [@CMX_NN, 0]>) outputs(%11 : memref<1x3x32x62xf16, {order = #NHWC, strides = [11532, 1, 186, 3]}, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:0:0>
    %19 = VPUMI40XX.KernelParams inputs(%12 : memref<1x3x30x62xf16, {order = #NHWC, strides = [11532, 1, 186, 3]}, [@CMX_NN, 0]>) outputs(%13 : memref<1x3x30x62xf16, {order = #NHWC, strides = [11532, 1, 186, 3]}, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:0:1>
    %20 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, isStartBarrier, producer_count = 1 : ui8, wlmPage = 0 : i64} <0, -1> -> !VPURegMapped.Index<0:0:0>
    %21 = VPUMI40XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 1 : ui8, wlmPage = 0 : i64}(%20 : !VPURegMapped.Index<0:0:0>) <1, -1> -> !VPURegMapped.Index<0:0:1>
    %22 = VPUMI40XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8, wlmPage = 0 : i64}(%21 : !VPURegMapped.Index<0:0:1>) <2, -1> -> !VPURegMapped.Index<0:0:2>
    %23 = VPUMI40XX.ConfigureBarrier {consumer_count = 0 : ui8, isFinalBarrier, producer_count = 1 : ui8, wlmPage = 0 : i64}(%22 : !VPURegMapped.Index<0:0:2>) <3, -1> -> !VPURegMapped.Index<0:0:3>

    %24 = VPUMI40XX.DeclareTaskBuffer {offset = 51200 : ui64} <ActKernelInvocation> -> !VPURegMapped.Index<0:0:30>
    %25 = VPUMI40XX.DeclareTaskBuffer {offset = 51200 : ui64} <ActKernelInvocation> -> !VPURegMapped.Index<0:0:31>
    %26 = VPUMI40XX.DeclareTaskBuffer {offset = 51200 : ui64} <ActKernelRange> -> !VPURegMapped.Index<0:0:30>
    %27 = VPUMI40XX.DeclareTaskBuffer {offset = 51200 : ui64} <ActKernelRange> -> !VPURegMapped.Index<0:0:31>

    %28 = VPUMI40XX.ActKernelRange taskLocation(%26 : !VPURegMapped.Index<0:0:30>) kernel_text_index(%14 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%16 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%15 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>

    %29 = VPUMI40XX.ActKernelRange taskLocation(%27 : !VPURegMapped.Index<0:0:31>) previousTask(%28 : !VPURegMapped.Index<0:0:0>) kernel_text_index(%14 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%17 : !VPURegMapped.Index<0:0:1>) kernel_entry_index(%15 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:1>

    %30 = VPUMI40XX.ActKernelInvocation {wlmPage = 0 : i64} taskLocation(%24 : !VPURegMapped.Index<0:0:30>) range_index(%28 : <0:0:0>) kernel_params(%18 : <0:0:0>) waits(%21 : !VPURegMapped.Index<0:0:1>) updates(%22 : !VPURegMapped.Index<0:0:2>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>

    %31 = VPUMI40XX.ActKernelInvocation {taskLinkAttrName = #VPURegMapped.IndexType<<0:0:0>>, lastSecondaryTaskInExecutionGroup, wlmPage = 0 : i64} taskLocation(%25 : !VPURegMapped.Index<0:0:31>) previousTask(%30 : !VPURegMapped.Index<0:0:0>) range_index(%29 : <0:0:1>) kernel_params(%19 : <0:0:1>) waits(%21 : !VPURegMapped.Index<0:0:1>) updates(%22 : !VPURegMapped.Index<0:0:2>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>

    %32 = VPURegMapped.ViewTaskRange(%28 -> %29 : <0:0:0> -> <0:0:1>) -> memref<2x40xui8>
    %33 = VPURegMapped.ViewTaskRange(%26 -> %27 : <0:0:30> -> <0:0:31>) -> memref<2x40xui8, [@CMX_NN, 0]>
    %34 = VPURegMapped.ViewTaskRange(%30 -> %31 : <0:0:0> -> <0:0:1>) -> memref<2x96xui8>
    %35 = VPURegMapped.ViewTaskRange(%24 -> %25 : <0:0:30> -> <0:0:31>) -> memref<2x96xui8, [@CMX_NN, 0]>
    %36 = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 256 : i64, srcWidth = 256 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 16 : i64, dstStride = 32 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64, wlmPage = -1 : i64} inputs(%cst : memref<64xui32>) outputs(%0 : memref<64xui32, @Register>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %37 = VPUMI40XX.NNDMA {is_critical, is_out_of_order, port = 0 : i64, taskLinkAttrName = #VPURegMapped.IndexType<<0:0:0>>, wlmPage = -1 : i64} inputs(%32 : memref<2x40xui8>) outputs(%33 : memref<2x40xui8, [@CMX_NN, 0]>) previousDMA(%36 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>

    %38 = VPUMI40XX.NNDMA {is_critical, is_out_of_order, port = 0 : i64, taskLinkAttrName = #VPURegMapped.IndexType<<0:0:1>>, wlmPage = -1 : i64} inputs(%34 : memref<2x96xui8>) outputs(%35 : memref<2x96xui8, [@CMX_NN, 0]>) previousDMA(%37 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>

    %39 = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64, taskLinkAttrName = #VPURegMapped.IndexType<<0:0:2>>, wlmPage = 0 : i64} inputs(%3 : memref<0x0x0x0xi32, @DDR>) outputs(%4 : memref<0x0x0x0xi32, @DDR>) previousDMA(%38 : !VPURegMapped.Index<0:0:2>) updates(%20 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>

    %40 = VPUMI40XX.NNDMA {allow_different_in_out_shapes, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 3 : i64, len = 7688 : i64, srcWidth = 7688 : i64, srcStride = 2 : i64, srcPlaneStride = 7688 : i64, dstWidth = 2 : i64, dstStride = 6 : i64, dstPlaneStride = 2 : i64>, is_out_of_order, port = 0 : i64, taskLinkAttrName = #VPURegMapped.IndexType<<0:0:3>>, wlmPage = 0 : i64} inputs(%1 : memref<3x3844xf16, @DDR>) outputs(%6 : memref<3844x3xf16, [@CMX_NN, 0]>) previousDMA(%39 : !VPURegMapped.Index<0:0:3>) waits(%20 : !VPURegMapped.Index<0:0:0>) updates(%21 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:4>

    %enq_dma = VPUMI40XX.NNDMA {enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<SHAVE_ACT>, tile = 0 : i64, list = 0 : i64, startTask = 0 : i64, endTask = 1 : i64>, port = 0 : i64} inputs(%buf0 : memref<0x0x0x0xi32, @DDR>) outputs(%buf1 : memref<0x0x0x0xi32, @DDR>) previousDMA(%40 : !VPURegMapped.Index<0:0:4>) waits(%20 : !VPURegMapped.Index<0:0:0>) updates(%21 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:5>

    %41 = VPUMI40XX.NNDMA {allow_different_in_out_shapes, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 3844 : i64, len = 6 : i64, srcWidth = 6 : i64, srcStride = 2 : i64, srcPlaneStride = 6 : i64, dstWidth = 2 : i64, dstStride = 7688 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64, wlmPage = 0 : i64} inputs(%7 : memref<3844x3xf16, [@CMX_NN, 0]>) outputs(%9 : memref<3x3844xf16, [@CMX_NN, 0]>) waits(%22 : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>

    %42 = VPUMI40XX.NNDMA {port = 0 : i64, taskLinkAttrName = #VPURegMapped.IndexType<<0:1:0>>, wlmPage = 0 : i64} inputs(%8 : memref<1x3x62x62xf16, [@CMX_NN, 0]>) outputs(%2 : memref<1x3x62x62xf16, @DDR>) previousDMA(%41 : !VPURegMapped.Index<0:1:0>) waits(%22 : !VPURegMapped.Index<0:0:2>) updates(%23 : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x3x62x62xf16, [@CMX_NN, 0]>, outputType = memref<1x3x62x62xf16, @DDR>>) -> !VPURegMapped.Index<0:1:1>

    %43 = VPUMI40XX.PlatformInfo -> <0:0:0>
    %44 = VPUMI40XX.ActShaveRt kernel("nnActEntry") -> !VPURegMapped.Index<0:0:0>
    %45 = VPUMI40XX.Bootstrap inputs(%20 : <0:0:0>) -> !VPURegMapped.Index<0:0:0>
    %46 = VPUMI40XX.Bootstrap inputs(%21 : <0:0:1>) -> !VPURegMapped.Index<0:0:1>
    %47 = VPUMI40XX.Bootstrap inputs(%22 : <0:0:2>) -> !VPURegMapped.Index<0:0:2>
    %48 = VPUMI40XX.Bootstrap inputs(%23 : <0:0:3>) -> !VPURegMapped.Index<0:0:3>
    %49 = VPURegMapped.Enqueue (%36 -> %36 : <0:0:0> -> <0:0:0>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<DMA>}
    %50 = VPURegMapped.Enqueue previousTaskIdx(%49 : !VPURegMapped.Index<0:0:0>) (%41 -> %41 : <0:1:0> -> <0:1:0>) -> !VPURegMapped.Index<0:0:1> {taskType = #VPURegMapped.task_type<DMA>}

    %52 = VPUMI40XX.MappedInference {workloadManagementBarrierProgrammingMode = #VPURegMapped.workload_management_barrier_programming_mode<ALL_BARRIER_DMAS_SCHEDULED>} dmas((%36, %41) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) actKernelRanges((%28) : (!VPURegMapped.Index<0:0:0>)) actKernelInvocations((%30) : (!VPURegMapped.Index<0:0:0>)) barriers(%20 : !VPURegMapped.Index<0:0:0>) workItemTasks(%49 : !VPURegMapped.Index<0:0:0>) bootstrapBarriers(%45 : !VPURegMapped.Index<0:0:0>) actShaveRt(%44 : !VPURegMapped.Index<0:0:0>) dmaHwpBase(%5 : memref<16xui32, [@CMX_NN, 0]>) dmaCount([[6, 2]]) invariantCount([0]) variantCount([0]) actKernelRangesCount([[2, 0]]) actKernelInvocationsCount([[2, 0]]) mediaCount(0) barrierCount(4) workItemCount(2) bootstrapBarriersCount(4) bootsrapWorkItemsCount(2) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x1000x1x1xf16, @DDR>
  }
}

// CHECK: [[EQDMA_CST:%.+]] = const.Declare memref<1xui32> = dense<2080> : tensor<1xui32>
// CHECK: [[EQDMA_REG_BUF:%.+]] = VPURT.DeclareBuffer <Register> <788578304> -> memref<1xui32, @Register>

// CHECK: VPUMI40XX.NNDMA
// CHECK: VPUMI40XX.NNDMA
// CHECK: VPUMI40XX.NNDMA
// CHECK: VPUMI40XX.NNDMA
// CHECK: VPUMI40XX.NNDMA

// SHV Enqueue
// CHECK: [[EQDMA_0:%.+]] = VPUMI40XX.NNDMA {
// CHECK-SAME: dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 4 : i64, srcWidth = 4 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 4 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>
// CHECK-SAME: enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<SHAVE_ACT>, tile = 0 : i64, list = 0 : i64, startTask = 0 : i64, endTask = 1 : i64>
// CHECK-SAME: inputs([[EQDMA_CST]] : memref<1xui32>) outputs([[EQDMA_REG_BUF]] : memref<1xui32, @Register>)

// DMA Enqueue
// CHECK: [[EQ_0:%.+]] = VPURegMapped.Enqueue
// CHECK: [[EQ_1:%.+]] = VPURegMapped.Enqueue
// CHECK-NOT: VPURegMapped.Enqueue
// CHECK: workItemTasks([[EQ_0]] : !VPURegMapped.Index<0:0:0>)
