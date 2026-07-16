//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --platform=%platform% --update-fetch-dmas-for-skip-dmas %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

{-#
  dialect_resources: {
    builtin: {
            vpux_ow_0: "0x100000000AB0CE30"
        }
  }
#-}

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @Activation attributes {config.compilationMode = #config.compilation_mode<DefaultHW>, config.elf_version = #config.version<2 : 0 : 0>, config.platform = #config.platform<NPU5010>, config.revisionID = #config.revision_id<REVISION_NONE>} {
  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_AtanDma(memref<*xf16>, memref<*xsi32>, memref<*xui8, [@CMX_NN, 0]>, memref<*xf16>, memref<*xsi32>, memref<*xui8, [@CMX_NN, 0]>, i64, i64, i64) attributes {VPU.kernel_code = "activation_atan_dma.cpp", VPU.kernel_entry = "activation_atan_dma", VPU.kernel_name = "activation_atan_dma", VPU.task_type = @COMPUTE}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }
  config.PipelineOptions @Options {
    config.Option @config.FragmentationAvoidRatioPipeliningLargeWeights : 3.200000e-01 : f32
    config.Option @config.WorkloadManagementStatus : "ENABLED"
    config.Option @config.MetadataMaxVariantCount : 128 : ui64
    config.Option @config.MetadataMaxInvariantCount : 64 : ui64
    config.Option @config.MetadataMaxKernelInvocationCount : 32 : ui64
    config.Option @config.MetadataMaxKernelRangeCount : 32 : ui64
    config.Option @config.MetadataMaxDMACount : 80 : ui64
    config.Option @config.MetadataMaxMediaCount : 4 : ui64
  }
  config.Resources {activity_factor = 0.000000e+00 : f64} 1 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
  config.Resources 1 of @global {
    config.ExecutorResource 1 of @M2I
    config.ExecutorResource 1 of @DMA_NN
    config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  }
  net.NetworkInfo {inferenceTiming = 237047 : i64} entryPoint : @main inputsInfo : {
    DataInfo "Input" tensorNames = ["dyn_input"] : tensor<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 4194304]> : tensor<4xsi64>}>
    DataInfo "vpux_ie_shape_Input" : tensor<4xsi32>
  } outputsInfo : {
    DataInfo "Sigm_11" friendlyName = "Result_12" : tensor<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 4194304]> : tensor<4xsi64>}>
    DataInfo "vpux_ie_shape_Sigm_11" : tensor<4xsi32>
  }
  func.func @main(%arg0: memref<1x1x1x4194304xf16, @DDR>, %arg1: memref<4xsi32, @DDR>, %arg2: memref<1x1x1x4194304xf16, @DDR>, %arg3: memref<4xsi32, @DDR>) -> (memref<1x1x1x4194304xf16, @DDR>, memref<4xsi32, @DDR>) {
    %cst = const.Declare memref<64xui32> = dense_resource<vpux_ow_0> : tensor<64xui32>
    %0 = VPURT.DeclareBuffer <Register> <788594688> -> memref<64xui32, @Register>
    %1 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x1x2097152xf16, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkInput> [0] <4194304> -> memref<1x1x1x2097152xf16, @DDR>
    %3 = VPURT.DeclareBuffer <NetworkInput> [1] <0> -> memref<4xsi32, @DDR>
    %4 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x1x1x2097152xf16, @DDR>
    %5 = VPURT.DeclareBuffer <NetworkOutput> [0] <4194304> -> memref<1x1x1x2097152xf16, @DDR>
    %6 = VPURT.DeclareBuffer <NetworkOutput> [1] <0> -> memref<4xsi32, @DDR>

    %7 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %8 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %9 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %10 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %11 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %12 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %13 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %14 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %15 = VPURT.DeclareBuffer <DDR> <8388672> -> memref<1x1x1x2097152xf16, @DDR>
    %16 = VPURT.DeclareBuffer <DDR> <12582976> -> memref<1x1x1x2097152xf16, @DDR>
    %17 = VPURT.DeclareBuffer <DDR> <8388672> -> memref<1x1x1x4194304xf16, @DDR>
    %18 = VPURT.DeclareBuffer <DDR> <16777280> -> memref<4xsi32, @DDR>
    %19 = VPURT.DeclareBuffer <DDR> <0> -> memref<4xsi32, @DDR>
    %20 = VPURT.DeclareBuffer <DDR> <64> -> memref<1x1x1x2097152xf16, @DDR>
    %21 = VPURT.DeclareBuffer <DDR> <4194368> -> memref<1x1x1x2097152xf16, @DDR>
    %22 = VPURT.DeclareBuffer <DDR> <64> -> memref<1x1x1x4194304xf16, @DDR>
    %23 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<0x0x0x0xi32, [@CMX_NN, 0]>
    %24 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1049600xui8, [@CMX_NN, 0]>

    %25 = VPUMI40XX.DeclareKernelText kernel_path("activation_atan_dma") -> !VPURegMapped.Index<0:0:0>
    %26 = VPUMI40XX.DeclareKernelEntry kernel_path("activation_atan_dma") -> !VPURegMapped.Index<0:0:0>
    %27 = VPUMI40XX.DeclareKernelArgs kernel_path("activation_atan_dma") -> !VPURegMapped.Index<0:0:0>
    %28 = VPUMI40XX.DeclareKernelArgs kernel_path("activation_atan_dma") -> !VPURegMapped.Index<0:1:0>

    %29 = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32: 1, 0>, dynamicOutputShapesSize = array<i32: 1, 0>}> inputs(%22, %24 : memref<1x1x1x4194304xf16, @DDR>, memref<1x1x1x1049600xui8, [@CMX_NN, 0]>) outputs(%17, %24 : memref<1x1x1x4194304xf16, @DDR>, memref<1x1x1x1049600xui8, [@CMX_NN, 0]>) dynamicInputShapes((%19), () : (memref<4xsi32, @DDR>), ()) dynamicOutputShapes((%18), () : (memref<4xsi32, @DDR>), ()) kernel_type("activation_atan_dma") kernel_params([0]) -> !VPURegMapped.Index<0:0:0>
    %30 = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32: 1, 0>, dynamicOutputShapesSize = array<i32: 1, 0>}> inputs(%22, %24 : memref<1x1x1x4194304xf16, @DDR>, memref<1x1x1x1049600xui8, [@CMX_NN, 0]>) outputs(%17, %24 : memref<1x1x1x4194304xf16, @DDR>, memref<1x1x1x1049600xui8, [@CMX_NN, 0]>) dynamicInputShapes((%19), () : (memref<4xsi32, @DDR>), ()) dynamicOutputShapes((%18), () : (memref<4xsi32, @DDR>), ()) kernel_type("activation_atan_dma") kernel_params([0]) -> !VPURegMapped.Index<0:1:0>

    %31 = VPUMI40XX.ConfigureBarrier <{consumer_count = 2 : ui8, isStartBarrier, producer_count = 1 : ui8, wlmPage = 0 : i64}> <0, -1> -> !VPURegMapped.Index<0:0:0>
    %32 = VPUMI40XX.ConfigureBarrier <{consumer_count = 2 : ui8, producer_count = 1 : ui8, wlmPage = 0 : i64}> <1, -1> -> !VPURegMapped.Index<0:0:1>
    %33 = VPUMI40XX.ConfigureBarrier <{consumer_count = 2 : ui8, producer_count = 2 : ui8, wlmPage = 0 : i64}> <2, -1> -> !VPURegMapped.Index<0:0:2>
    %34 = VPUMI40XX.ConfigureBarrier <{consumer_count = 2 : ui8, producer_count = 2 : ui8, wlmPage = 0 : i64}> <3, -1> -> !VPURegMapped.Index<0:0:3>
    %35 = VPUMI40XX.ConfigureBarrier <{consumer_count = 0 : ui8, isFinalBarrier, producer_count = 3 : ui8, wlmPage = 0 : i64}> <4, -1> -> !VPURegMapped.Index<0:0:4>

    %51 = VPUMI40XX.DeclareTaskBuffer {offset = 600 : ui64} <ActKernelRange> -> !VPURegMapped.Index<0:0:15>
    %67 = VPUMI40XX.DeclareTaskBuffer {offset = 1240 : ui64} <ActKernelRange> -> !VPURegMapped.Index<0:0:31>
    %83 = VPUMI40XX.DeclareTaskBuffer {offset = 2720 : ui64} <ActKernelInvocation> -> !VPURegMapped.Index<0:0:15>
    %99 = VPUMI40XX.DeclareTaskBuffer {offset = 4256 : ui64} <ActKernelInvocation> -> !VPURegMapped.Index<0:0:31>
    %100 = VPUMI40XX.ActKernelRange {wlmPage = 0 : i64} taskLocation(%51 : !VPURegMapped.Index<0:0:15>) kernel_text_index(%25 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%27 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%26 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
    %101 = VPUMI40XX.ActKernelInvocation {lastSecondaryTaskInExecutionGroup, wlmPage = 0 : i64} taskLocation(%83 : !VPURegMapped.Index<0:0:15>) range_index(%100 : <0:0:0>) kernel_params(%29 : <0:0:0>) waits(%33 : !VPURegMapped.Index<0:0:2>) updates(%34 : !VPURegMapped.Index<0:0:3>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>

    %117 = VPUMI40XX.DeclareTaskBuffer {offset = 4952 : ui64} <ActKernelRange> -> !VPURegMapped.Index<0:1:15>
    %133 = VPUMI40XX.DeclareTaskBuffer {offset = 5592 : ui64} <ActKernelRange> -> !VPURegMapped.Index<0:1:31>
    %149 = VPUMI40XX.DeclareTaskBuffer {offset = 7072 : ui64} <ActKernelInvocation> -> !VPURegMapped.Index<0:1:15>
    %165 = VPUMI40XX.DeclareTaskBuffer {offset = 8608 : ui64} <ActKernelInvocation> -> !VPURegMapped.Index<0:1:31>
    %166 = VPUMI40XX.ActKernelRange {wlmPage = 0 : i64} taskLocation(%117 : !VPURegMapped.Index<0:1:15>) kernel_text_index(%25 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%28 : !VPURegMapped.Index<0:1:0>) kernel_entry_index(%26 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:1:0>
    %167 = VPUMI40XX.ActKernelInvocation {lastSecondaryTaskInExecutionGroup, wlmPage = 0 : i64} taskLocation(%149 : !VPURegMapped.Index<0:1:15>) range_index(%166 : <0:1:0>) kernel_params(%30 : <0:1:0>) waits(%33 : !VPURegMapped.Index<0:0:2>) updates(%34 : !VPURegMapped.Index<0:0:3>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:1:0>

    // Barrier Programming DMA
    %168 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 256 : i64, srcWidth = 256 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 16 : i64, dstStride = 32 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64, wlmPage = 0 : i64}> inputs(%cst : memref<64xui32>) outputs(%0 : memref<64xui32, @Register>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    %169 = VPURegMapped.ViewTaskRange(%166 -> %166 : <0:1:0> -> <0:1:0>) -> memref<1x40xui8>
    %170 = VPURegMapped.ViewTaskRange(%117 -> %117 : <0:1:15> -> <0:1:15>) -> memref<1x40xui8, [@CMX_NN, 0]>
    %171 = VPURegMapped.ViewTaskRange(%167 -> %167 : <0:1:0> -> <0:1:0>) -> memref<1x96xui8>
    %172 = VPURegMapped.ViewTaskRange(%149 -> %149 : <0:1:15> -> <0:1:15>) -> memref<1x96xui8, [@CMX_NN, 0]>
    // Fetch Kernel Range
    %173 = VPUMI40XX.NNDMA <{is_critical, is_out_of_order, port = 0 : i64, wlmPage = -1 : i64}> inputs(%169 : memref<1x40xui8>) outputs(%170 : memref<1x40xui8, [@CMX_NN, 0]>) previousDMA(%168 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    // Fetch Kernel Invocation
    %174 = VPUMI40XX.NNDMA <{is_critical, is_out_of_order, port = 0 : i64, wlmPage = -1 : i64}> inputs(%171 : memref<1x96xui8>) outputs(%172 : memref<1x96xui8, [@CMX_NN, 0]>) previousDMA(%173 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>


    %175 = VPURegMapped.ViewTaskRange(%100 -> %100 : <0:0:0> -> <0:0:0>) -> memref<1x40xui8>
    %176 = VPURegMapped.ViewTaskRange(%51 -> %51 : <0:0:15> -> <0:0:15>) -> memref<1x40xui8, [@CMX_NN, 0]>
    %177 = VPURegMapped.ViewTaskRange(%101 -> %101 : <0:0:0> -> <0:0:0>) -> memref<1x96xui8>
    %178 = VPURegMapped.ViewTaskRange(%83 -> %83 : <0:0:15> -> <0:0:15>) -> memref<1x96xui8, [@CMX_NN, 0]>
    // Fetch Kernel Range
    %179 = VPUMI40XX.NNDMA <{is_critical, is_out_of_order, port = 0 : i64, wlmPage = -1 : i64}> inputs(%175 : memref<1x40xui8>) outputs(%176 : memref<1x40xui8, [@CMX_NN, 0]>) previousDMA(%174 : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>
    // Fetch Kernel Invocation
    %180 = VPUMI40XX.NNDMA <{is_critical, is_out_of_order, port = 0 : i64, wlmPage = -1 : i64}> inputs(%177 : memref<1x96xui8>) outputs(%178 : memref<1x96xui8, [@CMX_NN, 0]>) previousDMA(%179 : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:4>

    %181 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, is_out_of_order, port = 0 : i64, wlmPage = 0 : i64}> inputs(%11 : memref<0x0x0x0xi32, @DDR>) outputs(%12 : memref<0x0x0x0xi32, @DDR>) previousDMA(%180 : !VPURegMapped.Index<0:0:4>) updates(%31 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:5>
    %182 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<SHAVE_ACT>, tile = 0 : i64, list = 0 : i64, startTask = 0 : i64, endTask = 0 : i64>, port = 0 : i64, wlmPage = 0 : i64}> inputs(%7 : memref<0x0x0x0xi32, @DDR>) outputs(%8 : memref<0x0x0x0xi32, @DDR>) previousDMA(%181 : !VPURegMapped.Index<0:0:5>) waits(%31 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:6>
    %183 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<SHAVE_ACT>, tile = 0 : i64, list = 1 : i64, startTask = 0 : i64, endTask = 0 : i64>, port = 0 : i64, wlmPage = 0 : i64}> inputs(%7 : memref<0x0x0x0xi32, @DDR>) outputs(%8 : memref<0x0x0x0xi32, @DDR>) previousDMA(%182 : !VPURegMapped.Index<0:0:6>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:7>
    %184 = VPUMI40XX.NNDMA <{is_out_of_order, port = 0 : i64, wlmPage = 0 : i64}> inputs(%3 : memref<4xsi32, @DDR>) outputs(%19 : memref<4xsi32, @DDR>) previousDMA(%183 : !VPURegMapped.Index<0:0:7>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<4xsi32, @DDR>, outputType = memref<4xsi32, @DDR>>) -> !VPURegMapped.Index<0:0:8>
    %185 = VPUMI40XX.NNDMA <{is_out_of_order, port = 0 : i64, wlmPage = 0 : i64}> inputs(%2 : memref<1x1x1x2097152xf16, @DDR>) outputs(%21 : memref<1x1x1x2097152xf16, @DDR>) previousDMA(%184 : !VPURegMapped.Index<0:0:8>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x1x1x2097152xf16, @DDR>, outputType = memref<1x1x1x2097152xf16, @DDR>>) -> !VPURegMapped.Index<0:0:9>

    // Fetch for Skip DMAs
    %186 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, fetch_dma = #VPUIP.FetchDMAAttr<<DMA_NN>, tile = 0 : i64, list = 0 : i64, fetchType = <SingleDescriptor>, logicalTaskIdx = 0 : i64, descId = 0 : i64>, is_out_of_order, port = 0 : i64, wlmPage = 0 : i64}> inputs(%14 : memref<0x0x0x0xi32, @DDR>) outputs(%13 : memref<0x0x0x0xi32, @DDR>) previousDMA(%185 : !VPURegMapped.Index<0:0:9>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:10>
    %187 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, fetch_dma = #VPUIP.FetchDMAAttr<<DMA_NN>, tile = 0 : i64, list = 0 : i64, fetchType = <SingleDescriptor>, logicalTaskIdx = 0 : i64, descId = 1 : i64>, is_out_of_order, port = 0 : i64, wlmPage = 0 : i64}> inputs(%14 : memref<0x0x0x0xi32, @DDR>) outputs(%13 : memref<0x0x0x0xi32, @DDR>) previousDMA(%186 : !VPURegMapped.Index<0:0:10>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:11>
    %188 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, fetch_dma = #VPUIP.FetchDMAAttr<<DMA_NN>, tile = 0 : i64, list = 1 : i64, fetchType = <SingleDescriptor>, logicalTaskIdx = 0 : i64, descId = 2 : i64>, is_out_of_order, port = 0 : i64, wlmPage = 0 : i64}> inputs(%14 : memref<0x0x0x0xi32, @DDR>) outputs(%13 : memref<0x0x0x0xi32, @DDR>) previousDMA(%187 : !VPURegMapped.Index<0:0:11>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:12>
    %189 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, fetch_dma = #VPUIP.FetchDMAAttr<<DMA_NN>, tile = 0 : i64, list = 1 : i64, fetchType = <SingleDescriptor>, logicalTaskIdx = 0 : i64, descId = 3 : i64>, is_out_of_order, port = 0 : i64, wlmPage = 0 : i64}> inputs(%14 : memref<0x0x0x0xi32, @DDR>) outputs(%13 : memref<0x0x0x0xi32, @DDR>) previousDMA(%188 : !VPURegMapped.Index<0:0:12>) updates(%32, %33 : !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:13>

    // Sync DMAs
    %190 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, is_out_of_order, port = 0 : i64, wlmPage = 0 : i64}> inputs(%14 : memref<0x0x0x0xi32, @DDR>) outputs(%13 : memref<0x0x0x0xi32, @DDR>) previousDMA(%189 : !VPURegMapped.Index<0:0:13>) waits(%32 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:14>
    %191 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, enable_msc, port = 0 : i64, wlmPage = 0 : i64}> inputs(%23 : memref<0x0x0x0xi32, [@CMX_NN, 0]>) outputs(%13 : memref<0x0x0x0xi32, @DDR>) waits(%32 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>

    // Skip DMAs
    %192 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, is_out_of_order, port = 0 : i64, skip_dma = #VPUIP.SkipDMAAttr<tile = 0 : i64, list = 0 : i64, logicalTask = 0 : i64, descId = 0 : i64>, wlmPage = 0 : i64}> inputs(%14 : memref<0x0x0x0xi32, @DDR>) outputs(%13 : memref<0x0x0x0xi32, @DDR>) previousDMA(%190 : !VPURegMapped.Index<0:0:14>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:15>
    %193 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, enable_msc, is_out_of_order, port = 0 : i64, skip_dma = #VPUIP.SkipDMAAttr<tile = 0 : i64, list = 0 : i64, logicalTask = 0 : i64, descId = 1 : i64>, wlmPage = 0 : i64}> inputs(%23 : memref<0x0x0x0xi32, [@CMX_NN, 0]>) outputs(%13 : memref<0x0x0x0xi32, @DDR>) previousDMA(%191 : !VPURegMapped.Index<0:1:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:1>
    %194 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, is_out_of_order, port = 0 : i64, skip_dma = #VPUIP.SkipDMAAttr<tile = 0 : i64, list = 1 : i64, logicalTask = 0 : i64, descId = 2 : i64>, wlmPage = 0 : i64}> inputs(%14 : memref<0x0x0x0xi32, @DDR>) outputs(%13 : memref<0x0x0x0xi32, @DDR>) previousDMA(%192 : !VPURegMapped.Index<0:0:15>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:16>
    %195 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, enable_msc, is_out_of_order, port = 0 : i64, skip_dma = #VPUIP.SkipDMAAttr<tile = 0 : i64, list = 1 : i64, logicalTask = 0 : i64, descId = 3 : i64>, wlmPage = 0 : i64}> inputs(%23 : memref<0x0x0x0xi32, [@CMX_NN, 0]>) outputs(%13 : memref<0x0x0x0xi32, @DDR>) previousDMA(%193 : !VPURegMapped.Index<0:1:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:2>

    %196 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, enable_msc, is_out_of_order, port = 0 : i64, wlmPage = 0 : i64}> inputs(%23 : memref<0x0x0x0xi32, [@CMX_NN, 0]>) outputs(%13 : memref<0x0x0x0xi32, @DDR>) previousDMA(%195 : !VPURegMapped.Index<0:1:2>) updates(%35 : !VPURegMapped.Index<0:0:4>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:3>
    %197 = VPUMI40XX.NNDMA <{is_out_of_order, port = 0 : i64, wlmPage = 0 : i64}> inputs(%18 : memref<4xsi32, @DDR>) outputs(%6 : memref<4xsi32, @DDR>) previousDMA(%194 : !VPURegMapped.Index<0:0:16>) waits(%34 : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<4xsi32, @DDR>, outputType = memref<4xsi32, @DDR>>) -> !VPURegMapped.Index<0:0:17>
    %198 = VPUMI40XX.NNDMA <{is_out_of_order, port = 0 : i64, wlmPage = 0 : i64}> inputs(%16 : memref<1x1x1x2097152xf16, @DDR>) outputs(%5 : memref<1x1x1x2097152xf16, @DDR>) previousDMA(%197 : !VPURegMapped.Index<0:0:17>) updates(%35 : !VPURegMapped.Index<0:0:4>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x1x1x2097152xf16, @DDR>, outputType = memref<1x1x1x2097152xf16, @DDR>>) -> !VPURegMapped.Index<0:0:18>

    %202 = VPUMI40XX.ActShaveRt kernel("nnActEntry") -> !VPURegMapped.Index<0:0:0>
    %203 = VPURegMapped.Enqueue (%168 -> %198 : <0:0:0> -> <0:0:18>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<DMA>}
    %205 = VPURegMapped.Enqueue previousTaskIdx(%203 : !VPURegMapped.Index<0:0:0>) (%191 -> %196 : <0:1:0> -> <0:1:3>) -> !VPURegMapped.Index<0:0:2> {taskType = #VPURegMapped.task_type<DMA>}
    %206 = VPUMI40XX.MappedInference dmas((%168, %191) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) actKernelRanges((%100, %166) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) actKernelInvocations((%101, %167) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) barriers(%31 : !VPURegMapped.Index<0:0:0>) workItemTasks(%203 : !VPURegMapped.Index<0:0:0>) actShaveRt(%202 : !VPURegMapped.Index<0:0:0>) dmaCount([[19, 4], [2, 0]]) invariantCount([0]) variantCount([0]) actKernelRangesCount([[1, 1]]) actKernelInvocationsCount([[1, 1]]) mediaCount(0) barrierCount(5) workItemCount(3) bootstrapWorkItemsCount(3) finalBarrierId(4) -> !VPURegMapped.Index<0:0:0>
    VPUMI40XX.OpRanges types([#VPURegMapped.task_type<DMA>, #VPURegMapped.task_type<DMA>, #VPURegMapped.task_type<ActKernelRange>, #VPURegMapped.task_type<ActKernelInvocation>, #VPURegMapped.task_type<ActKernelRange>, #VPURegMapped.task_type<ActKernelInvocation>]) begins(%168, %191, %100, %101, %166, %167 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>, !VPURegMapped.Index<0:1:0>) ends(%198, %196, %100, %101, %166, %167 : !VPURegMapped.Index<0:0:18>, !VPURegMapped.Index<0:1:3>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>, !VPURegMapped.Index<0:1:0>)
  }

// CHECK: [[DTB0:%.+]] = VPUMI40XX.DeclareTaskBuffer {offset = 60864 : ui64} <DMA> -> !VPURegMapped.Index<0:0:0>
// CHECK: [[DTB1:%.+]] = VPUMI40XX.DeclareTaskBuffer {offset = 61088 : ui64} <DMA> -> !VPURegMapped.Index<0:0:1>
// CHECK: [[DTB2:%.+]] = VPUMI40XX.DeclareTaskBuffer {offset = 61312 : ui64} <DMA> -> !VPURegMapped.Index<0:1:0>
// CHECK: [[DTB3:%.+]] = VPUMI40XX.DeclareTaskBuffer {offset = 61536 : ui64} <DMA> -> !VPURegMapped.Index<0:1:1>

// CHECK: [[VTR0:%.+]] = VPURegMapped.ViewTaskRange([[DTB0]] -> [[DTB0]] : <0:0:0> -> <0:0:0>) -> memref<1x224xui8, [@CMX_NN, 0]>
// CHECK: [[FD0:%.+]] = VPUMI40XX.NNDMA <{fetch_dma = #VPUIP.FetchDMAAttr<<DMA_NN>, tile = 0 : i64, list = 0 : i64, fetchType = <SingleDescriptor>, logicalTaskIdx = 0 : i64, descId = 0 : i64>, is_out_of_order, port = 0 : i64, wlmPage = 0 : i64}> inputs(%0 : memref<1x224xi8, @DDR>) outputs([[VTR0]] : memref<1x224xui8, [@CMX_NN, 0]>) 
// CHECK: [[VTR1:%.+]] = VPURegMapped.ViewTaskRange([[DTB1]] -> [[DTB1]] : <0:0:1> -> <0:0:1>) -> memref<1x224xui8, [@CMX_NN, 0]>
// CHECK: [[FD1:%.+]] = VPUMI40XX.NNDMA <{fetch_dma = #VPUIP.FetchDMAAttr<<DMA_NN>, tile = 0 : i64, list = 0 : i64, fetchType = <SingleDescriptor>, logicalTaskIdx = 0 : i64, descId = 1 : i64>, is_out_of_order, port = 0 : i64, wlmPage = 0 : i64}> inputs(%0 : memref<1x224xi8, @DDR>) outputs([[VTR1]] : memref<1x224xui8, [@CMX_NN, 0]>) 

// CHECK: [[VTR2:%.+]] = VPURegMapped.ViewTaskRange([[DTB2]] -> [[DTB2]] : <0:1:0> -> <0:1:0>) -> memref<1x224xui8, [@CMX_NN, 0]>
// CHECK: [[FD2:%.+]] = VPUMI40XX.NNDMA <{fetch_dma = #VPUIP.FetchDMAAttr<<DMA_NN>, tile = 0 : i64, list = 1 : i64, fetchType = <SingleDescriptor>, logicalTaskIdx = 0 : i64, descId = 2 : i64>, is_out_of_order, port = 0 : i64, wlmPage = 0 : i64}> inputs(%0 : memref<1x224xi8, @DDR>) outputs([[VTR2]] : memref<1x224xui8, [@CMX_NN, 0]>) 

// CHECK: [[VTR3:%.+]] = VPURegMapped.ViewTaskRange([[DTB3]] -> [[DTB3]] : <0:1:1> -> <0:1:1>) -> memref<1x224xui8, [@CMX_NN, 0]>
// CHECK: [[FD3:%.+]] = VPUMI40XX.NNDMA <{fetch_dma = #VPUIP.FetchDMAAttr<<DMA_NN>, tile = 0 : i64, list = 1 : i64, fetchType = <SingleDescriptor>, logicalTaskIdx = 0 : i64, descId = 3 : i64>, is_out_of_order, port = 0 : i64, wlmPage = 0 : i64}> inputs(%0 : memref<1x224xi8, @DDR>) outputs([[VTR3]] : memref<1x224xui8, [@CMX_NN, 0]>) 
}
