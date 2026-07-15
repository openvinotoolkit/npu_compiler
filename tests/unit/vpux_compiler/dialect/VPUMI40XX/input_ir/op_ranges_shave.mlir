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
    module @VPU.SW {
        func.func private @builtin_softmax(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax"}
    }
    func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
        %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
        %11 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
        %12 = VPURT.DeclareBuffer <CMX_NN> [0] <69632> -> memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
        %13 = VPURT.DeclareBuffer <CMX_NN> [0] <102400> -> memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
        %14 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <69632> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>
        %15 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <4096> -> !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
        %16 = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
        %40 = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs(%0 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%11 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) kernel_type("activation_softmax") kernel_params([0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]) -> !VPURegMapped.Index<0:0:0>
        %41 = VPUMI40XX.DeclareKernelText kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
        %42 = VPUMI40XX.DeclareKernelEntry kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
        %43 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
        %44 = VPUMI40XX.ActKernelRange kernel_text_index(%41 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%43 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%42 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
        %45 = VPUMI40XX.ActKernelInvocation range_index(%44 : <0:0:0>) kernel_params(%40 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
        %48 = VPUMI40XX.ActKernelRange previousTask(%44 : !VPURegMapped.Index<0:0:0>) kernel_text_index(%41 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%43 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%42 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:1>
        %49 = VPUMI40XX.ActKernelInvocation previousTask(%45 : !VPURegMapped.Index<0:0:0>) range_index(%48 : <0:0:1>) kernel_params(%40 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
        %52 = VPUMI40XX.ActKernelRange kernel_text_index(%41 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%43 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%42 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:0:0>
        %53 = VPUMI40XX.ActKernelInvocation range_index(%52 : <1:0:0>) kernel_params(%40 : <0:0:0>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:0>
        %56 = VPUMI40XX.ActKernelRange previousTask(%52 : !VPURegMapped.Index<1:0:0>) kernel_text_index(%41 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%43 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%42 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:0:1>
        %57 = VPUMI40XX.ActKernelInvocation previousTask(%53 : !VPURegMapped.Index<1:0:0>) range_index(%56 : <1:0:1>) kernel_params(%40 : <0:0:0>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:1>
        %60 = VPUMI40XX.ActKernelRange kernel_text_index(%41 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%43 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%42 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<2:0:0>
        %61 = VPUMI40XX.ActKernelInvocation range_index(%60 : <2:0:0>) kernel_params(%40 : <0:0:0>) tile(2) start_after(0) clean_after(0) -> !VPURegMapped.Index<2:0:0>
        %64 = VPUMI40XX.ActKernelRange previousTask(%60 : !VPURegMapped.Index<2:0:0>) kernel_text_index(%41 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%43 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%42 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<2:0:1>
        %65 = VPUMI40XX.ActKernelInvocation previousTask(%61 : !VPURegMapped.Index<2:0:0>) range_index(%64 : <2:0:1>) kernel_params(%40 : <0:0:0>) tile(2) start_after(0) clean_after(0) -> !VPURegMapped.Index<2:0:1>
        %68 = VPUMI40XX.ActKernelRange kernel_text_index(%41 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%43 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%42 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<3:0:0>
        %69 = VPUMI40XX.ActKernelInvocation range_index(%68 : <3:0:0>) kernel_params(%40 : <0:0:0>) tile(3) start_after(0) clean_after(0) -> !VPURegMapped.Index<3:0:0>
        %72 = VPUMI40XX.ActKernelRange previousTask(%68 : !VPURegMapped.Index<3:0:0>) kernel_text_index(%41 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%43 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%42 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<3:0:1>
        %73 = VPUMI40XX.ActKernelInvocation previousTask(%69 : !VPURegMapped.Index<3:0:0>) range_index(%72 : <3:0:1>) kernel_params(%40 : <0:0:0>) tile(3) start_after(0) clean_after(0) -> !VPURegMapped.Index<3:0:1>
        %76 = VPUMI40XX.ActKernelRange kernel_text_index(%41 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%43 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%42 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<4:0:0>
        %77 = VPUMI40XX.ActKernelInvocation range_index(%76 : <4:0:0>) kernel_params(%40 : <0:0:0>) tile(4) start_after(0) clean_after(0) -> !VPURegMapped.Index<4:0:0>
        %80 = VPUMI40XX.ActKernelRange previousTask(%76 : !VPURegMapped.Index<4:0:0>) kernel_text_index(%41 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%43 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%42 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<4:0:1>
        %81 = VPUMI40XX.ActKernelInvocation previousTask(%77 : !VPURegMapped.Index<4:0:0>) range_index(%80 : <4:0:1>) kernel_params(%40 : <0:0:0>) tile(4) start_after(0) clean_after(0) -> !VPURegMapped.Index<4:0:1>
        %84 = VPUMI40XX.ActKernelRange kernel_text_index(%41 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%43 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%42 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<5:0:0>
        %85 = VPUMI40XX.ActKernelInvocation range_index(%84 : <5:0:0>) kernel_params(%40 : <0:0:0>) tile(5) start_after(0) clean_after(0) -> !VPURegMapped.Index<5:0:0>
        %88 = VPUMI40XX.ActKernelRange previousTask(%84 : !VPURegMapped.Index<5:0:0>) kernel_text_index(%41 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%43 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%42 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<5:0:1>
        %89 = VPUMI40XX.ActKernelInvocation previousTask(%85 : !VPURegMapped.Index<5:0:0>) range_index(%88 : <5:0:1>) kernel_params(%40 : <0:0:0>) tile(5) start_after(0) clean_after(0) -> !VPURegMapped.Index<5:0:1>
        VPUMI40XX.OpRanges types([#VPURegMapped.task_type<ActKernelRange>, #VPURegMapped.task_type<ActKernelInvocation>, #VPURegMapped.task_type<ActKernelRange>, #VPURegMapped.task_type<ActKernelInvocation>, #VPURegMapped.task_type<ActKernelRange>, #VPURegMapped.task_type<ActKernelInvocation>, #VPURegMapped.task_type<ActKernelRange>, #VPURegMapped.task_type<ActKernelInvocation>, #VPURegMapped.task_type<ActKernelRange>, #VPURegMapped.task_type<ActKernelInvocation>, #VPURegMapped.task_type<ActKernelRange>, #VPURegMapped.task_type<ActKernelInvocation>]) begins(%44, %45, %52, %53, %60, %61, %68, %69, %76, %77, %84, %85 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<2:0:0>, !VPURegMapped.Index<2:0:0>, !VPURegMapped.Index<3:0:0>, !VPURegMapped.Index<3:0:0>, !VPURegMapped.Index<4:0:0>, !VPURegMapped.Index<4:0:0>, !VPURegMapped.Index<5:0:0>, !VPURegMapped.Index<5:0:0>) ends(%48, %49, %56, %57, %64, %65, %72, %73, %80, %81, %88, %89 : !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<1:0:1>, !VPURegMapped.Index<1:0:1>, !VPURegMapped.Index<2:0:1>, !VPURegMapped.Index<2:0:1>, !VPURegMapped.Index<3:0:1>, !VPURegMapped.Index<3:0:1>, !VPURegMapped.Index<4:0:1>, !VPURegMapped.Index<4:0:1>, !VPURegMapped.Index<5:0:1>, !VPURegMapped.Index<5:0:1>)
    }
}
