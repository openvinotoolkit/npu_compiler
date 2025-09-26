//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --intermediate-buffer-output="op-index=5 insertion-index=5 buffer-index=4" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType2 = !quant.uniform<u8:f16, 0.01269696927538105>
!qElemType3 = !quant.uniform<u8:f16, 0.0048000719033035573>
!qElemType4 = !quant.uniform<u8:f16, 0.0173492431640625:114>
!qElemType5 = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType6 = !quant.uniform<u8:f16, 0.0024000359516517787>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

module @testsubgraph attributes {config.arch = #config.arch_kind<NPU40XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
  config.Resources 4 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
  config.ExecutorResource 1 of @M2I
  config.ExecutorResource 2 of @DMA_NN
  config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "result.1" : tensor<1x3x224x224xf16>
  } outputsInfo : {
    DataInfo "Multiply_5095/fq_input_0" : tensor<1x64x56x56xf16>
    // CHECK: DataInfo "Multiply_5095/fq_input_0" : tensor<1x224x4x115xui8>
  }
  func.func @main(%arg0: memref<1x3x224x224xf16, @DDR>, %arg1: memref<1x64x56x56xf16, @DDR>) -> memref<1x64x56x56xf16, @DDR> {
  // CHECK: func.func @main([[ARG0:%.+]]: memref<1x3x224x224xf16, @DDR>, [[ARG1:%.+]]: memref<1x224x4x115x!qElemType, #NWCH, @DDR>) -> memref<1x224x4x115x!qElemType, #NWCH, @DDR> {

    %0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    %2 = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier
    %3 = VPURT.ConfigureBarrier<3> -> !VPURT.Barrier
    %4 = VPURT.ConfigureBarrier<4> -> !VPURT.Barrier
    %5 = VPURT.ConfigureBarrier<5> -> !VPURT.Barrier
    %6 = VPURT.ConfigureBarrier<6> <{isFinalBarrier}> -> !VPURT.Barrier
    %cst = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %cst_0 = const.Declare memref<64x1x1x160x!qElemType, #NHWC> = dense<1.0> : tensor<64x3x7x7xf32>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>, #const.SubView<[0, 0, 0, 0], [64, 3, 7, 7]>, #const.Reshape<[64, 1, 1, 147]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 13]>]
    %cst_1 = const.Declare memref<1x1x1x5120xui8> = dense<1> : tensor<1x1x1x5120xui8>
    %7 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %8 = VPURT.DeclareBuffer <NetworkInput> [0] <48832> -> memref<1x3x115x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %9 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>
    %10 = VPURT.DeclareBuffer <NetworkOutput> [0] <3136> -> memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>
    %11 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x114x224xf16, [@CMX_NN, 0]>
    %12 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x115x224xf16, [@CMX_NN, 1]>
    %13 = VPURT.DeclareBuffer <CMX_NN> <154560> -> !VPUIP.DistributedBuffer<1x224x4x224x!qElemType1, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>
    %14 = VPURT.DeclareBuffer <CMX_NN> [0] <154560> -> memref<1x224x4x114x!qElemType1, #NWCH, [@CMX_NN, 0]>
    %15 = VPURT.DeclareBuffer <CMX_NN> [1] <154560> -> memref<1x224x4x115x!qElemType1, #NWCH, [@CMX_NN, 1]>
    %16 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <257600> -> !VPUIP.DistributedBuffer<64x1x1x160x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %17 = VPURT.DeclareBuffer <CMX_NN> [0] <272960> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %18 = VPURT.DeclareBuffer <CMX_NN> [1] <272960> -> memref<64x1x1x4xsi32, [@CMX_NN, 1]>
    %19 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <272960> -> !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %20 = VPURT.DeclareBuffer <CMX_NN> <278528> {swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x64x112x112x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %21 = VPURT.DeclareBuffer <CMX_NN> [0] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %22 = VPURT.DeclareBuffer <CMX_NN> [1] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]>
    %23 = VPURT.DeclareBuffer <CMX_NN> [0] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %24 = VPURT.DeclareBuffer <CMX_NN> [1] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]>
    %25 = VPURT.DeclareBuffer <CMX_NN> <0> {swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x64x56x56x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %26 = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<1x64x28x56x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %27 = VPURT.DeclareBuffer <CMX_NN> [1] <0> {swizzlingKey = 5 : i64} -> memref<1x64x28x56x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]>
    %28 = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<1x64x28x56x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %29 = VPURT.DeclareBuffer <CMX_NN> [1] <0> {swizzlingKey = 5 : i64} -> memref<1x64x28x56x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]>
    %30 = VPURT.DeclareBuffer <CMX_NN> <100352> -> !VPUIP.DistributedBuffer<1x64x56x56x!qElemType3, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %31 = VPURT.DeclareBuffer <CMX_NN> [0] <100352> -> memref<1x64x28x56x!qElemType3, #NHWC, [@CMX_NN, 0]>
    %32 = VPURT.DeclareBuffer <CMX_NN> [1] <100352> -> memref<1x64x28x56x!qElemType3, #NHWC, [@CMX_NN, 1]>
    %33 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <267840> -> !VPUIP.DistributedBuffer<1x1x1x5120xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %34 = VPURT.DeclareBuffer <CMX_NN> <200704> -> !VPUIP.DistributedBuffer<1x64x56x56xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %35 = VPURT.DeclareBuffer <CMX_NN> [0] <200704> -> memref<1x64x28x56xf16, [@CMX_NN, 0]>
    %36 = VPURT.DeclareBuffer <CMX_NN> [1] <200704> -> memref<1x64x28x56xf16, [@CMX_NN, 1]>
    %37 = VPURT.DeclareBuffer <CMX_NN> [0] <200704> -> memref<1x64x28x56xf16, [@CMX_NN, 0]>
    %38 = VPURT.DeclareBuffer <CMX_NN> [1] <200704> -> memref<1x64x28x56xf16, [@CMX_NN, 1]>
    %39 = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x224x3x224xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
    %40 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>
    %41 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x224x3x115xf16, #NHWC, [@CMX_NN, 1]>
    %42 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>
    %43 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x224x3x115xf16, #NHWC, [@CMX_NN, 1]>
    %44 = VPURT.DeclareBuffer <CMX_NN> [0] <257600> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>
    %45 = VPURT.DeclareBuffer <CMX_NN> [1] <257600> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>
    %46 = VPURT.DeclareBuffer <CMX_NN> <154560> -> !VPUIP.DistributedBuffer<1x16x224x224x!qElemType4, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
    %47 = VPURT.DeclareBuffer <CMX_NN> [0] <154560> -> memref<1x16x114x224x!qElemType4, #NHWC, [@CMX_NN, 0]>
    %48 = VPURT.DeclareBuffer <CMX_NN> [1] <154560> -> memref<1x16x115x224x!qElemType4, #NHWC, [@CMX_NN, 1]>
    %49 = VPURT.DeclareBuffer <CMX_NN> [0] <267840> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %50 = VPURT.DeclareBuffer <CMX_NN> [1] <267840> -> memref<64x1x1x4xsi32, [@CMX_NN, 1]>
    %51 = VPURT.DeclareBuffer <CMX_NN> [0] <268864> -> memref<64x64x1x1x!qElemType5, #NHWC, [@CMX_NN, 0]>
    %52 = VPURT.DeclareBuffer <CMX_NN> [1] <268864> -> memref<64x64x1x1x!qElemType5, #NHWC, [@CMX_NN, 1]>
    %53 = VPURT.DeclareBuffer <CMX_NN> <100352> -> !VPUIP.DistributedBuffer<1x64x56x56x!qElemType6, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %54 = VPURT.DeclareBuffer <CMX_NN> [0] <100352> -> memref<1x64x28x56x!qElemType6, #NHWC, [@CMX_NN, 0]>
    %55 = VPURT.DeclareBuffer <CMX_NN> [1] <100352> -> memref<1x64x28x56x!qElemType6, #NHWC, [@CMX_NN, 1]>
    %56 = VPURT.DeclareBuffer <CMX_NN> [0] <100352> -> memref<1x64x28x56x!qElemType6, #NHWC, [@CMX_NN, 0]>
    %57 = VPURT.DeclareBuffer <CMX_NN> [1] <100352> -> memref<1x64x28x56x!qElemType6, #NHWC, [@CMX_NN, 1]>
    VPURT.Task updates(%0 : !VPURT.Barrier) {
      %58 = VPUIP.NNDMA {port = 0 : i64} inputs(%7 : memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%11 : memref<1x3x114x224xf16, [@CMX_NN, 0]>) -> memref<1x3x114x224xf16, [@CMX_NN, 0]>
    }
    VPURT.Task updates(%0 : !VPURT.Barrier) {
      %58 = VPUIP.NNDMA {port = 1 : i64} inputs(%8 : memref<1x3x115x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%12 : memref<1x3x115x224xf16, [@CMX_NN, 1]>) -> memref<1x3x115x224xf16, [@CMX_NN, 1]>
    }
    VPURT.Task {
      %58 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst : memref<64x1x1x4xsi32>) outputs(%19 : !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    }
    VPURT.Task updates(%1 : !VPURT.Barrier) {
      %58 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_0 : memref<64x1x1x160x!qElemType, #NHWC>) outputs(%16 : !VPUIP.DistributedBuffer<64x1x1x160x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<64x1x1x160x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %58 = VPUIP.NCEClusterTask {is_permute_quantize, is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%42 : memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>) weights(%40 : memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>) parent_input(%39 : !VPUIP.DistributedBuffer<1x224x3x224xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%13 : !VPUIP.DistributedBuffer<1x224x4x224x!qElemType1, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>) outputs(%14 : memref<1x224x4x114x!qElemType1, #NWCH, [@CMX_NN, 0]>) -> memref<1x224x4x114x!qElemType1, #NWCH, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1, %2 : !VPURT.Barrier, !VPURT.Barrier) {
      %58 = VPUIP.NCEClusterTask {is_permute_quantize, is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%43 : memref<1x224x3x115xf16, #NHWC, [@CMX_NN, 1]>) weights(%41 : memref<1x224x3x115xf16, #NHWC, [@CMX_NN, 1]>) parent_input(%39 : !VPUIP.DistributedBuffer<1x224x3x224xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%13 : !VPUIP.DistributedBuffer<1x224x4x224x!qElemType1, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>) outputs(%15 : memref<1x224x4x115x!qElemType1, #NWCH, [@CMX_NN, 1]>) -> memref<1x224x4x115x!qElemType1, #NWCH, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [114, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task updates(%2 : !VPURT.Barrier) {
      %58 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_1 : memref<1x1x1x5120xui8>) outputs(%33 : !VPUIP.DistributedBuffer<1x1x1x5120xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x1x5120xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    }
    VPURT.Task waits(%1 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
      %58 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%47 : memref<1x16x114x224x!qElemType4, #NHWC, [@CMX_NN, 0]>) weights(%44 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%17 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%46 : !VPUIP.DistributedBuffer<1x16x224x224x!qElemType4, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%20 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%23 : memref<1x64x56x112x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) -> memref<1x64x56x112x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%1 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
      %58 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 2 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%48 : memref<1x16x115x224x!qElemType4, #NHWC, [@CMX_NN, 1]>) weights(%45 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%18 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : !VPUIP.DistributedBuffer<1x16x224x224x!qElemType4, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%20 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%24 : memref<1x64x56x112x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]>) -> memref<1x64x56x112x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 111, 63], outStart = [0, 56, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 2 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%2 : !VPURT.Barrier) {
      %58 = VPUIP.NCEClusterTask {is_segmented, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%21 : memref<1x64x56x112x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) parent_input(%20 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%25 : !VPUIP.DistributedBuffer<1x64x56x56x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%28 : memref<1x64x28x56x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) -> memref<1x64x28x56x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%2 : !VPURT.Barrier) {
      %58 = VPUIP.NCEClusterTask {is_segmented, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%22 : memref<1x64x56x112x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]>) parent_input(%20 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%25 : !VPUIP.DistributedBuffer<1x64x56x56x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%29 : memref<1x64x28x56x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]>) -> memref<1x64x28x56x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 55, 63], outStart = [0, 28, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%2 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) {
      %58 = VPUIP.NCEClusterTask {is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>} input(%26 : memref<1x64x28x56x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) weights(%51 : memref<64x64x1x1x!qElemType5, #NHWC, [@CMX_NN, 0]>) weight_table(%49 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%25 : !VPUIP.DistributedBuffer<1x64x56x56x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%30 : !VPUIP.DistributedBuffer<1x64x56x56x!qElemType3, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%31 : memref<1x64x28x56x!qElemType3, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x56x!qElemType3, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%2 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) {
      %58 = VPUIP.NCEClusterTask {is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>} input(%27 : memref<1x64x28x56x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]>) weights(%52 : memref<64x64x1x1x!qElemType5, #NHWC, [@CMX_NN, 1]>) weight_table(%50 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%25 : !VPUIP.DistributedBuffer<1x64x56x56x!qElemType2, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%30 : !VPUIP.DistributedBuffer<1x64x56x56x!qElemType3, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%32 : memref<1x64x28x56x!qElemType3, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x56x!qElemType3, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 55, 63], outStart = [0, 28, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) {
      %58 = VPUIP.NCEClusterTask {is_segmented, is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%56 : memref<1x64x28x56x!qElemType6, #NHWC, [@CMX_NN, 0]>) weights(%54 : memref<1x64x28x56x!qElemType6, #NHWC, [@CMX_NN, 0]>) parent_input(%53 : !VPUIP.DistributedBuffer<1x64x56x56x!qElemType6, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%34 : !VPUIP.DistributedBuffer<1x64x56x56xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%37 : memref<1x64x28x56xf16, [@CMX_NN, 0]>) -> memref<1x64x28x56xf16, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) {
      %58 = VPUIP.NCEClusterTask {is_segmented, is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%57 : memref<1x64x28x56x!qElemType6, #NHWC, [@CMX_NN, 1]>) weights(%55 : memref<1x64x28x56x!qElemType6, #NHWC, [@CMX_NN, 1]>) parent_input(%53 : !VPUIP.DistributedBuffer<1x64x56x56x!qElemType6, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%34 : !VPUIP.DistributedBuffer<1x64x56x56xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%38 : memref<1x64x28x56xf16, [@CMX_NN, 1]>) -> memref<1x64x28x56xf16, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 55, 63], outStart = [0, 28, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) {
      %58 = VPUIP.NNDMA {port = 0 : i64} inputs(%35 : memref<1x64x28x56xf16, [@CMX_NN, 0]>) outputs(%9 : memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>) -> memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>
    }
    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) {
      %58 = VPUIP.NNDMA {port = 1 : i64} inputs(%36 : memref<1x64x28x56xf16, [@CMX_NN, 1]>) outputs(%10 : memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>) -> memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>
    }
    return %arg1 : memref<1x64x56x56xf16, @DDR>
  }

  // CHECK:       [[BAR0:%.*]] = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
  // CHECK:       [[BAR1:%.*]] = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
  // CHECK:       [[BAR2:%.*]] = VPURT.ConfigureBarrier<0> <{isFinalBarrier}> -> !VPURT.Barrier
  // CHECK-NOT:                  VPURT.ConfigureBarrier

  // CHECK-DAG:   [[OUTPUT:%.*]] = VPURT.DeclareBuffer <NetworkOutput> <0> -> memref<1x224x4x115x!qElemType, #NWCH, @DDR>
  // CHECK-DAG:   [[TARGET_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <154560> -> memref<1x224x4x115x!qElemType, #NWCH, [@CMX_NN, 1]>

  // Target TaskOp
  // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {opIndex = 5 : i64}
  // CHECK-NEXT:     outputs([[TARGET_BUFFER]] : memref<1x224x4x115x!qElemType, #NWCH, [@CMX_NN, 1]>)

  // Target insertion point and DMA out
  // CHECK:       VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
  // CHECK:          VPUIP.NNDMA
  // CHECK-SAME:        inputs([[TARGET_BUFFER]] : memref<1x224x4x115x!qElemType, #NWCH, [@CMX_NN, 1]>
  // CHECK-SAME:        outputs([[OUTPUT]] :  memref<1x224x4x115x!qElemType, #NWCH, @DDR>

  // CHECK-NOT:   VPURT.Task

  // Updated return
  // CHECK:   return [[ARG1]] : memref<1x224x4x115x!qElemType, #NWCH, @DDR>
}
