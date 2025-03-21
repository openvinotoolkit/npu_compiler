//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch% allow-custom-values=true" --split-input-file --legalize-schedule-for-wlm="workload-management-mode=PWLM_V0_LCA" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 0.01269696927538105>
!qElemType2 = !quant.uniform<u8:f16, 0.0173492431640625:114>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @NoLegalizationDueToFIFOOrdering attributes {VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  IE.PipelineOptions @Options {
    IE.Option @VPU.MetadataMaxVariantCount : 8
    IE.Option @VPU.MetadataMaxInvariantCount : 4
    IE.Option @VPU.MetadataMaxKernelInvocationCount : 4
    IE.Option @VPU.MetadataMaxKernelRangeCount : 4
  }
  IE.TileResource 2 of @NCE at 1.700000e+03 MHz {
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 2 of @DMA_NN
  IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "result.1" : tensor<1x3x224x224xf16>
  } outputsInfo : {
    DataInfo "Multiply_5095/fq_input_0" : tensor<1x64x56x56xf16>
  }
  func.func @main(%arg0: memref<1x3x224x224xf16, @DDR>, %arg1: memref<1x64x56x56xf16, @DDR>) -> memref<1x64x56x56xf16, @DDR> {
    %cst = const.Declare memref<1x1x1x5120xui8> = dense<1> : tensor<1x1x1x5120xui8>
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %x = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %6 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %7 = VPURT.DeclareBuffer <NetworkInput> [0] <48832> -> memref<1x3x115x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %8 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x114x224xf16, [@CMX_NN, 0]>
    %10 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x115x224xf16, [@CMX_NN, 1]>
    %11 = VPURT.DeclareBuffer <CMX_NN> <154560> -> !VPUIP.DistributedBuffer<1x224x4x224x!qElemType, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>
    %12 = VPURT.DeclareBuffer <CMX_NN> [0] <154560> -> memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 0]>
    %13 = VPURT.DeclareBuffer <CMX_NN> [0] <272960> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %14 = VPURT.DeclareBuffer <CMX_NN> <278528> {swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %15 = VPURT.DeclareBuffer <CMX_NN> [0] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %16 = VPURT.DeclareBuffer <CMX_NN> [0] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %17 = VPURT.DeclareBuffer <CMX_NN> <0> {swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x64x56x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %18 = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %19 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <267840> -> !VPUIP.DistributedBuffer<1x1x1x5120xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %20 = VPURT.DeclareBuffer <CMX_NN> [0] <200704> -> memref<1x64x28x56xf16, [@CMX_NN, 0]>
    %21 = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x224x3x224xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
    %22 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>
    %23 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>
    %24 = VPURT.DeclareBuffer <CMX_NN> [0] <257600> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>
    %25 = VPURT.DeclareBuffer <CMX_NN> <154560> -> !VPUIP.DistributedBuffer<1x16x224x224x!qElemType2, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
    %26 = VPURT.DeclareBuffer <CMX_NN> [0] <154560> -> memref<1x16x114x224x!qElemType2, #NHWC, [@CMX_NN, 0]>
    VPURT.Task updates(%0 : !VPURT.Barrier) {
      %27 = VPUIP.NNDMA {port = 0 : i64} inputs(%6 : memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%9 : memref<1x3x114x224xf16, [@CMX_NN, 0]>) -> memref<1x3x114x224xf16, [@CMX_NN, 0]>
    }
    VPURT.Task updates(%0 : !VPURT.Barrier) {
      %27 = VPUIP.NNDMA {port = 0 : i64} inputs(%7 : memref<1x3x115x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%10 : memref<1x3x115x224xf16, [@CMX_NN, 1]>) -> memref<1x3x115x224xf16, [@CMX_NN, 1]>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %27 = VPUIP.NCEClusterTask {is_permute_quantize, is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%23 : memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>) weights(%22 : memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>) parent_input(%21 : !VPUIP.DistributedBuffer<1x224x3x224xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%11 : !VPUIP.DistributedBuffer<1x224x4x224x!qElemType, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>) outputs(%12 : memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 0]>) -> memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%1 : !VPURT.Barrier) updates(%2 : !VPURT.Barrier) {
      %27 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst : memref<1x1x1x5120xui8>) outputs(%19 : !VPUIP.DistributedBuffer<1x1x1x5120xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x1x5120xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    }
    VPURT.Task waits(%2 : !VPURT.Barrier) updates(%x : !VPURT.Barrier) {
      %27 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst : memref<1x1x1x5120xui8>) outputs(%19 : !VPUIP.DistributedBuffer<1x1x1x5120xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x1x5120xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    }
    VPURT.Task waits(%x : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
      %27 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%26 : memref<1x16x114x224x!qElemType2, #NHWC, [@CMX_NN, 0]>) weights(%24 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%25 : !VPUIP.DistributedBuffer<1x16x224x224x!qElemType2, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%14 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%16 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) {
      %27 = VPUIP.NCEClusterTask {is_segmented, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%15 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) parent_input(%14 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%17 : !VPUIP.DistributedBuffer<1x64x56x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%18 : memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) {
      %27 = VPUIP.NNDMA {port = 0 : i64} inputs(%20 : memref<1x64x28x56xf16, [@CMX_NN, 0]>) outputs(%8 : memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>) -> memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>
    }
    return %arg1 : memref<1x64x56x56xf16, @DDR>
  }

  // CHECK: VPUIP.NNDMA
  // CHECK: VPUIP.NNDMA
  // CHECK: VPUIP.NNDMA
  // CHECK: VPUIP.NNDMA
  // CHECK: VPUIP.NNDMA
  // CHECK-NOT: VPUIP.NNDMA
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 0.01269696927538105>
!qElemType2 = !quant.uniform<u8:f16, 0.0173492431640625:114>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @LegalizeWithJustDMAs attributes {VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  IE.PipelineOptions @Options {
    IE.Option @VPU.MetadataMaxVariantCount : 8
    IE.Option @VPU.MetadataMaxInvariantCount : 4
    IE.Option @VPU.MetadataMaxKernelInvocationCount : 4
    IE.Option @VPU.MetadataMaxKernelRangeCount : 4
  }
  IE.TileResource 2 of @NCE at 1.700000e+03 MHz {
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 2 of @DMA_NN
  IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "result.1" : tensor<1x3x224x224xf16>
  } outputsInfo : {
    DataInfo "Multiply_5095/fq_input_0" : tensor<1x64x56x56xf16>
  }
  func.func @main(%arg0: memref<1x3x224x224xf16, @DDR>, %arg1: memref<1x64x56x56xf16, @DDR>) -> memref<1x64x56x56xf16, @DDR> {
    %cst = const.Declare memref<1x1x1x5120xui8> = dense<1> : tensor<1x1x1x5120xui8>
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %6 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %7 = VPURT.DeclareBuffer <NetworkInput> [0] <48832> -> memref<1x3x115x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x114x224xf16, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x115x224xf16, [@CMX_NN, 1]>
    %10 = VPURT.DeclareBuffer <CMX_NN> <154560> -> !VPUIP.DistributedBuffer<1x224x4x224x!qElemType, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>
    %11 = VPURT.DeclareBuffer <CMX_NN> [0] <154560> -> memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 0]>
    %12 = VPURT.DeclareBuffer <CMX_NN> [0] <272960> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %13 = VPURT.DeclareBuffer <CMX_NN> <278528> {swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %14 = VPURT.DeclareBuffer <CMX_NN> [0] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %15 = VPURT.DeclareBuffer <CMX_NN> [0] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %16 = VPURT.DeclareBuffer <CMX_NN> <0> {swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x64x56x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %17 = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %18 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <267840> -> !VPUIP.DistributedBuffer<1x1x1x5120xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %19 = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x224x3x224xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
    %20 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>
    %21 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>
    %22 = VPURT.DeclareBuffer <CMX_NN> [0] <257600> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>
    %23 = VPURT.DeclareBuffer <CMX_NN> <154560> -> !VPUIP.DistributedBuffer<1x16x224x224x!qElemType2, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
    %24 = VPURT.DeclareBuffer <CMX_NN> [0] <154560> -> memref<1x16x114x224x!qElemType2, #NHWC, [@CMX_NN, 0]>
    VPURT.Task updates(%0 : !VPURT.Barrier) {
      %25 = VPUIP.NNDMA {port = 0 : i64} inputs(%6 : memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%8 : memref<1x3x114x224xf16, [@CMX_NN, 0]>) -> memref<1x3x114x224xf16, [@CMX_NN, 0]>
    }
    VPURT.Task updates(%0 : !VPURT.Barrier) {
      %25 = VPUIP.NNDMA {port = 0 : i64} inputs(%7 : memref<1x3x115x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%9 : memref<1x3x115x224xf16, [@CMX_NN, 1]>) -> memref<1x3x115x224xf16, [@CMX_NN, 1]>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %25 = VPUIP.NCEClusterTask {is_permute_quantize, is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%21 : memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>) weights(%20 : memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>) parent_input(%19 : !VPUIP.DistributedBuffer<1x224x3x224xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%10 : !VPUIP.DistributedBuffer<1x224x4x224x!qElemType, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>) outputs(%11 : memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 0]>) -> memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%1 : !VPURT.Barrier) updates(%2 : !VPURT.Barrier) {
      %25 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst : memref<1x1x1x5120xui8>) outputs(%18 : !VPUIP.DistributedBuffer<1x1x1x5120xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x1x5120xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    }
    VPURT.Task waits(%2 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
      %25 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%24 : memref<1x16x114x224x!qElemType2, #NHWC, [@CMX_NN, 0]>) weights(%22 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%12 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%23 : !VPUIP.DistributedBuffer<1x16x224x224x!qElemType2, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%13 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%15 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) {
      %25 = VPUIP.NCEClusterTask {is_segmented, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%14 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) parent_input(%13 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%16 : !VPUIP.DistributedBuffer<1x64x56x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%17 : memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) {
      %25 = VPUIP.NNDMA {port = 0 : i64} inputs(%6 : memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%8 : memref<1x3x114x224xf16, [@CMX_NN, 0]>) -> memref<1x3x114x224xf16, [@CMX_NN, 0]>
    }
    return %arg1 : memref<1x64x56x56xf16, @DDR>
  }

  // CHECK: [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR4:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR5:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

  // CHECK: [[BUF0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
  // CHECK: [[BUF1:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>

  // Dummy DMAs
  // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
  // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)

  // Parent Group
  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 0.01269696927538105>
!qElemType2 = !quant.uniform<u8:f16, 0.0173492431640625:114>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @LegalizeGroupThreeWithSharedBarrier attributes {VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  IE.PipelineOptions @Options {
    IE.Option @VPU.MetadataMaxVariantCount : 8
    IE.Option @VPU.MetadataMaxInvariantCount : 4
    IE.Option @VPU.MetadataMaxKernelInvocationCount : 4
    IE.Option @VPU.MetadataMaxKernelRangeCount : 4
  }
  IE.TileResource 2 of @NCE at 1.700000e+03 MHz {
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 2 of @DMA_NN
  IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "result.1" : tensor<1x3x224x224xf16>
  } outputsInfo : {
    DataInfo "Multiply_5095/fq_input_0" : tensor<1x64x56x56xf16>
  }
  func.func @main(%arg0: memref<1x3x224x224xf16, @DDR>, %arg1: memref<1x64x56x56xf16, @DDR>) -> memref<1x64x56x56xf16, @DDR> {
    %cst = const.Declare memref<1x1x1x5120xui8> = dense<1> : tensor<1x1x1x5120xui8>
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %6 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %7 = VPURT.DeclareBuffer <NetworkInput> [0] <48832> -> memref<1x3x115x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %8 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x114x224xf16, [@CMX_NN, 0]>
    %10 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x115x224xf16, [@CMX_NN, 1]>
    %11 = VPURT.DeclareBuffer <CMX_NN> <154560> -> !VPUIP.DistributedBuffer<1x224x4x224x!qElemType, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>
    %12 = VPURT.DeclareBuffer <CMX_NN> [0] <154560> -> memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 0]>
    %13 = VPURT.DeclareBuffer <CMX_NN> [0] <272960> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %14 = VPURT.DeclareBuffer <CMX_NN> <278528> {swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %15 = VPURT.DeclareBuffer <CMX_NN> [0] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %16 = VPURT.DeclareBuffer <CMX_NN> [0] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %17 = VPURT.DeclareBuffer <CMX_NN> <0> {swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x64x56x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %18 = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %19 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <267840> -> !VPUIP.DistributedBuffer<1x1x1x5120xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %20 = VPURT.DeclareBuffer <CMX_NN> [0] <200704> -> memref<1x64x28x56xf16, [@CMX_NN, 0]>
    %21 = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x224x3x224xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
    %22 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>
    %23 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>
    %24 = VPURT.DeclareBuffer <CMX_NN> [0] <257600> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>
    %25 = VPURT.DeclareBuffer <CMX_NN> <154560> -> !VPUIP.DistributedBuffer<1x16x224x224x!qElemType2, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
    %26 = VPURT.DeclareBuffer <CMX_NN> [0] <154560> -> memref<1x16x114x224x!qElemType2, #NHWC, [@CMX_NN, 0]>
    VPURT.Task updates(%0 : !VPURT.Barrier) {
      %27 = VPUIP.NNDMA {port = 0 : i64} inputs(%6 : memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%9 : memref<1x3x114x224xf16, [@CMX_NN, 0]>) -> memref<1x3x114x224xf16, [@CMX_NN, 0]>
    }
    VPURT.Task updates(%0 : !VPURT.Barrier) {
      %27 = VPUIP.NNDMA {port = 0 : i64} inputs(%7 : memref<1x3x115x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%10 : memref<1x3x115x224xf16, [@CMX_NN, 1]>) -> memref<1x3x115x224xf16, [@CMX_NN, 1]>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %27 = VPUIP.NCEClusterTask {is_permute_quantize, is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%23 : memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>) weights(%22 : memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>) parent_input(%21 : !VPUIP.DistributedBuffer<1x224x3x224xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%11 : !VPUIP.DistributedBuffer<1x224x4x224x!qElemType, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>) outputs(%12 : memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 0]>) -> memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%1: !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
      %27 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%26 : memref<1x16x114x224x!qElemType2, #NHWC, [@CMX_NN, 0]>) weights(%24 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%25 : !VPUIP.DistributedBuffer<1x16x224x224x!qElemType2, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%14 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%16 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) {
      %27 = VPUIP.NCEClusterTask {is_segmented, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%15 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) parent_input(%14 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%17 : !VPUIP.DistributedBuffer<1x64x56x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%18 : memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) {
      %27 = VPUIP.NNDMA {port = 0 : i64} inputs(%20 : memref<1x64x28x56xf16, [@CMX_NN, 0]>) outputs(%8 : memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>) -> memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>
    }
    return %arg1 : memref<1x64x56x56xf16, @DDR>
  }

  // CHECK: [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR4:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR5:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

  // CHECK: [[BUF0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
  // CHECK: [[BUF1:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>

  // Dummy DMAs
  // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
  // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)

  // Parent Group
  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 0.01269696927538105>
!qElemType2 = !quant.uniform<u8:f16, 0.0173492431640625:114>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @LegalizeGroupsWithBothSharedBarriers attributes {VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  IE.PipelineOptions @Options {
    IE.Option @VPU.MetadataMaxVariantCount : 8
    IE.Option @VPU.MetadataMaxInvariantCount : 4
    IE.Option @VPU.MetadataMaxKernelInvocationCount : 4
    IE.Option @VPU.MetadataMaxKernelRangeCount : 4
  }
  IE.TileResource 2 of @NCE at 1.700000e+03 MHz {
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 2 of @DMA_NN
  IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "result.1" : tensor<1x3x224x224xf16>
  } outputsInfo : {
    DataInfo "Multiply_5095/fq_input_0" : tensor<1x64x56x56xf16>
  }
  func.func @main(%arg0: memref<1x3x224x224xf16, @DDR>, %arg1: memref<1x64x56x56xf16, @DDR>) -> memref<1x64x56x56xf16, @DDR> {
    %0 = VPURT.DeclareVirtualBarrier {"barrier-index" = 0 : i64} -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier {"barrier-index" = 1 : i64} -> !VPURT.Barrier
    %2 = VPURT.DeclareVirtualBarrier {"barrier-index" = 2 : i64} -> !VPURT.Barrier
    %3 = VPURT.DeclareVirtualBarrier {"barrier-index" = 3 : i64} -> !VPURT.Barrier
    %4 = VPURT.DeclareVirtualBarrier {"barrier-index" = 4 : i64} -> !VPURT.Barrier
    %11 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %12 = VPURT.DeclareBuffer <NetworkInput> [0] <48832> -> memref<1x3x115x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %13 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>
    %14 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x114x224xf16, [@CMX_NN, 0]>
    %15 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x115x224xf16, [@CMX_NN, 1]>
    %16 = VPURT.DeclareBuffer <CMX_NN> <154560> -> !VPUIP.DistributedBuffer<1x224x4x224x!qElemType, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>
    %17 = VPURT.DeclareBuffer <CMX_NN> [0] <154560> -> memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 0]>
    %18 = VPURT.DeclareBuffer <CMX_NN> [0] <272960> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %19 = VPURT.DeclareBuffer <CMX_NN> <278528> {swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %20 = VPURT.DeclareBuffer <CMX_NN> [0] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %21 = VPURT.DeclareBuffer <CMX_NN> [0] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %22 = VPURT.DeclareBuffer <CMX_NN> <0> {swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x64x56x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %23 = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %24 = VPURT.DeclareBuffer <CMX_NN> [0] <200704> -> memref<1x64x28x56xf16, [@CMX_NN, 0]>
    %25 = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x224x3x224xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
    %26 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>
    %27 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>
    %28 = VPURT.DeclareBuffer <CMX_NN> [0] <257600> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>
    %29 = VPURT.DeclareBuffer <CMX_NN> <154560> -> !VPUIP.DistributedBuffer<1x16x224x224x!qElemType2, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
    %30 = VPURT.DeclareBuffer <CMX_NN> [0] <154560> -> memref<1x16x114x224x!qElemType2, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%0 : !VPURT.Barrier) {
      %31 = VPUIP.NNDMA {port = 0 : i64} inputs(%11 : memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%14 : memref<1x3x114x224xf16, [@CMX_NN, 0]>) -> memref<1x3x114x224xf16, [@CMX_NN, 0]>
    }
    VPURT.Task updates(%0 : !VPURT.Barrier) {
      %31 = VPUIP.NNDMA {port = 0 : i64} inputs(%12 : memref<1x3x115x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%15 : memref<1x3x115x224xf16, [@CMX_NN, 1]>) -> memref<1x3x115x224xf16, [@CMX_NN, 1]>
    }

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier){
      %31 = VPUIP.NCEClusterTask {is_permute_quantize, is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%27 : memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>) weights(%26 : memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>) parent_input(%25 : !VPUIP.DistributedBuffer<1x224x3x224xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%16 : !VPUIP.DistributedBuffer<1x224x4x224x!qElemType, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>) outputs(%17 : memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 0]>) -> memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {"task-index" = 3 : i64} {
      %31 = VPUIP.NCEClusterTask {is_permute_quantize, is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%27 : memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>) weights(%26 : memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>) parent_input(%25 : !VPUIP.DistributedBuffer<1x224x3x224xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%16 : !VPUIP.DistributedBuffer<1x224x4x224x!qElemType, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>) outputs(%17 : memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 0]>) -> memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %31 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%30 : memref<1x16x114x224x!qElemType2, #NHWC, [@CMX_NN, 0]>) weights(%28 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%18 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%29 : !VPUIP.DistributedBuffer<1x16x224x224x!qElemType2, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%19 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%21 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %31 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%30 : memref<1x16x114x224x!qElemType2, #NHWC, [@CMX_NN, 0]>) weights(%28 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%18 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%29 : !VPUIP.DistributedBuffer<1x16x224x224x!qElemType2, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%19 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%21 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%2 : !VPURT.Barrier) {
      %31 = VPUIP.NCEClusterTask {is_segmented, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%20 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) parent_input(%19 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%22 : !VPUIP.DistributedBuffer<1x64x56x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%23 : memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%2 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
      %31 = VPUIP.NCEClusterTask {is_segmented, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%20 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) parent_input(%19 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%22 : !VPUIP.DistributedBuffer<1x64x56x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%23 : memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }

    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) {
      %31 = VPUIP.NNDMA {port = 0 : i64} inputs(%24 : memref<1x64x28x56xf16, [@CMX_NN, 0]>) outputs(%13 : memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>) -> memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>
    }
    return %arg1 : memref<1x64x56x56xf16, @DDR>
  }

  // CHECK: [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR4:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR5:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR6:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

  // CHECK: [[BUF0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
  // CHECK: [[BUF1:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>

  // Group 1
  // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
  // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)

  // Dummy DMAs
  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)
  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)

  // Group 2
  // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
  // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)

  // Group 3
  // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)
  // CHECK: VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier)
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 0.01269696927538105>
!qElemType2 = !quant.uniform<u8:f16, 0.0173492431640625:114>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @LegalizeWithJustDMAWithWaitingDMABeforeGrandParent attributes {VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  IE.PipelineOptions @Options {
    IE.Option @VPU.MetadataMaxVariantCount : 8
    IE.Option @VPU.MetadataMaxInvariantCount : 4
    IE.Option @VPU.MetadataMaxKernelInvocationCount : 4
    IE.Option @VPU.MetadataMaxKernelRangeCount : 4
  }
  IE.TileResource 2 of @NCE at 1.700000e+03 MHz {
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 2 of @DMA_NN
  IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "result.1" : tensor<1x3x224x224xf16>
  } outputsInfo : {
    DataInfo "Multiply_5095/fq_input_0" : tensor<1x64x56x56xf16>
  }
  func.func @main(%arg0: memref<1x3x224x224xf16, @DDR>, %arg1: memref<1x64x56x56xf16, @DDR>) -> memref<1x64x56x56xf16, @DDR> {
    %cst = const.Declare memref<1x1x1x5120xui8> = dense<1> : tensor<1x1x1x5120xui8>
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %26 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %6 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %7 = VPURT.DeclareBuffer <NetworkInput> [0] <48832> -> memref<1x3x115x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x114x224xf16, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x115x224xf16, [@CMX_NN, 1]>
    %10 = VPURT.DeclareBuffer <CMX_NN> <154560> -> !VPUIP.DistributedBuffer<1x224x4x224x!qElemType, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>
    %11 = VPURT.DeclareBuffer <CMX_NN> [0] <154560> -> memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 0]>
    %12 = VPURT.DeclareBuffer <CMX_NN> [0] <272960> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %13 = VPURT.DeclareBuffer <CMX_NN> <278528> {swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %14 = VPURT.DeclareBuffer <CMX_NN> [0] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %15 = VPURT.DeclareBuffer <CMX_NN> [0] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %16 = VPURT.DeclareBuffer <CMX_NN> <0> {swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x64x56x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %17 = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %18 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <267840> -> !VPUIP.DistributedBuffer<1x1x1x5120xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %19 = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x224x3x224xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
    %20 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>
    %21 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>
    %22 = VPURT.DeclareBuffer <CMX_NN> [0] <257600> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>
    %23 = VPURT.DeclareBuffer <CMX_NN> <154560> -> !VPUIP.DistributedBuffer<1x16x224x224x!qElemType2, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
    %24 = VPURT.DeclareBuffer <CMX_NN> [0] <154560> -> memref<1x16x114x224x!qElemType2, #NHWC, [@CMX_NN, 0]>
    VPURT.Task updates(%0 : !VPURT.Barrier) {
      %25 = VPUIP.NNDMA {port = 0 : i64} inputs(%6 : memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%8 : memref<1x3x114x224xf16, [@CMX_NN, 0]>) -> memref<1x3x114x224xf16, [@CMX_NN, 0]>
    }
    VPURT.Task updates(%0, %26 : !VPURT.Barrier, !VPURT.Barrier) {
      %25 = VPUIP.NNDMA {port = 0 : i64} inputs(%7 : memref<1x3x115x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%9 : memref<1x3x115x224xf16, [@CMX_NN, 1]>) -> memref<1x3x115x224xf16, [@CMX_NN, 1]>
    }
    VPURT.Task waits(%26 : !VPURT.Barrier) updates(%0 : !VPURT.Barrier) {
      %25 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst : memref<1x1x1x5120xui8>) outputs(%18 : !VPUIP.DistributedBuffer<1x1x1x5120xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x1x5120xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %25 = VPUIP.NCEClusterTask {is_permute_quantize, is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%21 : memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>) weights(%20 : memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>) parent_input(%19 : !VPUIP.DistributedBuffer<1x224x3x224xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%10 : !VPUIP.DistributedBuffer<1x224x4x224x!qElemType, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>) outputs(%11 : memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 0]>) -> memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%1 : !VPURT.Barrier) updates(%2 : !VPURT.Barrier) {
      %25 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst : memref<1x1x1x5120xui8>) outputs(%18 : !VPUIP.DistributedBuffer<1x1x1x5120xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x1x5120xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    }
    VPURT.Task waits(%2, %26 : !VPURT.Barrier, !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
      %25 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%24 : memref<1x16x114x224x!qElemType2, #NHWC, [@CMX_NN, 0]>) weights(%22 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%12 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%23 : !VPUIP.DistributedBuffer<1x16x224x224x!qElemType2, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%13 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%15 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) {
      %25 = VPUIP.NCEClusterTask {is_segmented, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%14 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) parent_input(%13 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%16 : !VPUIP.DistributedBuffer<1x64x56x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%17 : memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) {
      %25 = VPUIP.NNDMA {port = 0 : i64} inputs(%6 : memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%8 : memref<1x3x114x224xf16, [@CMX_NN, 0]>) -> memref<1x3x114x224xf16, [@CMX_NN, 0]>
    }
    return %arg1 : memref<1x64x56x56xf16, @DDR>
  }
  // CHECK: [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR4:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR5:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR6:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

  // CHECK: [[BUF0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
  // CHECK: [[BUF1:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
  
  // DMA Waiting on Bar1
  // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR0]] : !VPURT.Barrier)
  
  // Grand Parent
  // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
  
  // Dummy DMAs not updating Bar1
  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)
  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)

  // Parent Group waits Bar1
  // CHECK: VPURT.Task waits([[BAR3]], [[BAR1]] : !VPURT.Barrier, !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 0.01269696927538105>
!qElemType2 = !quant.uniform<u8:f16, 0.0173492431640625:114>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @LegalizeWithBarrierAndDMAWithWaitingDMABeforeGrandParent attributes {VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  IE.PipelineOptions @Options {
    IE.Option @VPU.MetadataMaxVariantCount : 8
    IE.Option @VPU.MetadataMaxInvariantCount : 4
    IE.Option @VPU.MetadataMaxKernelInvocationCount : 4
    IE.Option @VPU.MetadataMaxKernelRangeCount : 4
  }
  IE.TileResource 2 of @NCE at 1.700000e+03 MHz {
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 2 of @DMA_NN
  IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "result.1" : tensor<1x3x224x224xf16>
  } outputsInfo : {
    DataInfo "Multiply_5095/fq_input_0" : tensor<1x64x56x56xf16>
  }
  func.func @main(%arg0: memref<1x3x224x224xf16, @DDR>, %arg1: memref<1x64x56x56xf16, @DDR>) -> memref<1x64x56x56xf16, @DDR> {
    %cst = const.Declare memref<1x1x1x5120xui8> = dense<1> : tensor<1x1x1x5120xui8>
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %6 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %7 = VPURT.DeclareBuffer <NetworkInput> [0] <48832> -> memref<1x3x115x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x114x224xf16, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x115x224xf16, [@CMX_NN, 1]>
    %10 = VPURT.DeclareBuffer <CMX_NN> <154560> -> !VPUIP.DistributedBuffer<1x224x4x224x!qElemType, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>
    %11 = VPURT.DeclareBuffer <CMX_NN> [0] <154560> -> memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 0]>
    %12 = VPURT.DeclareBuffer <CMX_NN> [0] <272960> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %13 = VPURT.DeclareBuffer <CMX_NN> <278528> {swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %14 = VPURT.DeclareBuffer <CMX_NN> [0] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %15 = VPURT.DeclareBuffer <CMX_NN> [0] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %16 = VPURT.DeclareBuffer <CMX_NN> <0> {swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x64x56x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %17 = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %18 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <267840> -> !VPUIP.DistributedBuffer<1x1x1x5120xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %19 = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x224x3x224xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
    %20 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>
    %21 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>
    %22 = VPURT.DeclareBuffer <CMX_NN> [0] <257600> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>
    %23 = VPURT.DeclareBuffer <CMX_NN> <154560> -> !VPUIP.DistributedBuffer<1x16x224x224x!qElemType2, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
    %24 = VPURT.DeclareBuffer <CMX_NN> [0] <154560> -> memref<1x16x114x224x!qElemType2, #NHWC, [@CMX_NN, 0]>
    
    VPURT.Task updates(%0, %1 : !VPURT.Barrier, !VPURT.Barrier) {
      %25 = VPUIP.NNDMA {port = 0 : i64} inputs(%6 : memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%8 : memref<1x3x114x224xf16, [@CMX_NN, 0]>) -> memref<1x3x114x224xf16, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%1 : !VPURT.Barrier) {
      %25 = VPUIP.NNDMA {port = 0 : i64} inputs(%6 : memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%8 : memref<1x3x114x224xf16, [@CMX_NN, 0]>) -> memref<1x3x114x224xf16, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%2 : !VPURT.Barrier) {
      %25 = VPUIP.NCEClusterTask {is_permute_quantize, is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%21 : memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>) weights(%20 : memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>) parent_input(%19 : !VPUIP.DistributedBuffer<1x224x3x224xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%10 : !VPUIP.DistributedBuffer<1x224x4x224x!qElemType, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>) outputs(%11 : memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 0]>) -> memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%1 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
      %25 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%24 : memref<1x16x114x224x!qElemType2, #NHWC, [@CMX_NN, 0]>) weights(%22 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%12 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%23 : !VPUIP.DistributedBuffer<1x16x224x224x!qElemType2, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%13 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%15 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) {
      %25 = VPUIP.NCEClusterTask {is_segmented, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%14 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) parent_input(%13 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%16 : !VPUIP.DistributedBuffer<1x64x56x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%17 : memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) {
      %25 = VPUIP.NNDMA {port = 0 : i64} inputs(%6 : memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%8 : memref<1x3x114x224xf16, [@CMX_NN, 0]>) -> memref<1x3x114x224xf16, [@CMX_NN, 0]>
    }
    return %arg1 : memref<1x64x56x56xf16, @DDR>
  }
  
  // Have one extra barrier as no existing barrier was usable
  // CHECK: [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR4:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR5:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR6:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

  // CHECK: [[BUF0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
  // CHECK: [[BUF1:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
  
  // DMA Waiting on Bar1
  // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
  
  // Grand Parent
  // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
  
  // Dummy DMAs updating new barrier Bar3
  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]]
  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]]

  // Parent Group waits Bar3
  // CHECK: VPURT.Task waits([[BAR1]], [[BAR3]] : !VPURT.Barrier, !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 0.01269696927538105>
!qElemType2 = !quant.uniform<u8:f16, 0.0173492431640625:114>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @LegalizeGroupThreeWithSharedBarrierMultiTile attributes {VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  IE.PipelineOptions @Options {
    IE.Option @VPU.MetadataMaxVariantCount : 8
    IE.Option @VPU.MetadataMaxInvariantCount : 4
    IE.Option @VPU.MetadataMaxKernelInvocationCount : 4
    IE.Option @VPU.MetadataMaxKernelRangeCount : 4
  }
  IE.TileResource 2 of @NCE at 1.700000e+03 MHz {
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 4 of @SHAVE_ACT
    IE.ExecutorResource 2 of @DPU
  }
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 2 of @DMA_NN
  IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "result.1" : tensor<1x3x224x224xf16>
  } outputsInfo : {
    DataInfo "Multiply_5095/fq_input_0" : tensor<1x64x56x56xf16>
  }
  func.func @main(%arg0: memref<1x3x224x224xf16, @DDR>, %arg1: memref<1x64x56x56xf16, @DDR>) -> memref<1x64x56x56xf16, @DDR> {
    %cst = const.Declare memref<1x1x1x5120xui8> = dense<1> : tensor<1x1x1x5120xui8>
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %6 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %7 = VPURT.DeclareBuffer <NetworkInput> [0] <48832> -> memref<1x3x115x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %8 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x114x224xf16, [@CMX_NN, 0]>
    %10 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x115x224xf16, [@CMX_NN, 1]>
    %11 = VPURT.DeclareBuffer <CMX_NN> <154560> -> !VPUIP.DistributedBuffer<1x224x4x224x!qElemType, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>
    %12 = VPURT.DeclareBuffer <CMX_NN> [0] <154560> -> memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 0]>
    %122 = VPURT.DeclareBuffer <CMX_NN> [1] <154560> -> memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 1]>
    %13 = VPURT.DeclareBuffer <CMX_NN> [0] <272960> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %14 = VPURT.DeclareBuffer <CMX_NN> <278528> {swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %15 = VPURT.DeclareBuffer <CMX_NN> [0] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %16 = VPURT.DeclareBuffer <CMX_NN> [0] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %166 = VPURT.DeclareBuffer <CMX_NN> [1] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]>
    %17 = VPURT.DeclareBuffer <CMX_NN> <0> {swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x64x56x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %18 = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %188 = VPURT.DeclareBuffer <CMX_NN> [1] <0> {swizzlingKey = 5 : i64} -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]>
    %19 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <267840> -> !VPUIP.DistributedBuffer<1x1x1x5120xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %20 = VPURT.DeclareBuffer <CMX_NN> [0] <200704> -> memref<1x64x28x56xf16, [@CMX_NN, 0]>
    %21 = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x224x3x224xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
    %22 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>
    %23 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>
    %24 = VPURT.DeclareBuffer <CMX_NN> [0] <257600> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>
    %25 = VPURT.DeclareBuffer <CMX_NN> <154560> -> !VPUIP.DistributedBuffer<1x16x224x224x!qElemType2, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
    %26 = VPURT.DeclareBuffer <CMX_NN> [0] <154560> -> memref<1x16x114x224x!qElemType2, #NHWC, [@CMX_NN, 0]>
    VPURT.Task updates(%0 : !VPURT.Barrier) {
      %27 = VPUIP.NNDMA {port = 0 : i64} inputs(%6 : memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%9 : memref<1x3x114x224xf16, [@CMX_NN, 0]>) -> memref<1x3x114x224xf16, [@CMX_NN, 0]>
    }
    VPURT.Task updates(%0 : !VPURT.Barrier) {
      %27 = VPUIP.NNDMA {port = 0 : i64} inputs(%7 : memref<1x3x115x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%10 : memref<1x3x115x224xf16, [@CMX_NN, 1]>) -> memref<1x3x115x224xf16, [@CMX_NN, 1]>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %27 = VPUIP.NCEClusterTask {is_permute_quantize, is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%23 : memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>) weights(%22 : memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>) parent_input(%21 : !VPUIP.DistributedBuffer<1x224x3x224xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%11 : !VPUIP.DistributedBuffer<1x224x4x224x!qElemType, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>) outputs(%12 : memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 0]>) -> memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %27 = VPUIP.NCEClusterTask {is_permute_quantize, is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%23 : memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>) weights(%22 : memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>) parent_input(%21 : !VPUIP.DistributedBuffer<1x224x3x224xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%11 : !VPUIP.DistributedBuffer<1x224x4x224x!qElemType, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>) outputs(%122 : memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 1]>) -> memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    
    VPURT.Task waits(%1: !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
      %27 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%26 : memref<1x16x114x224x!qElemType2, #NHWC, [@CMX_NN, 0]>) weights(%24 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%25 : !VPUIP.DistributedBuffer<1x16x224x224x!qElemType2, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%14 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%16 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%1: !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
      %27 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%26 : memref<1x16x114x224x!qElemType2, #NHWC, [@CMX_NN, 0]>) weights(%24 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%25 : !VPUIP.DistributedBuffer<1x16x224x224x!qElemType2, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%14 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%166 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]>) -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) {
      %27 = VPUIP.NCEClusterTask {is_segmented, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%15 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) parent_input(%14 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%17 : !VPUIP.DistributedBuffer<1x64x56x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%18 : memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
        VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) {
      %27 = VPUIP.NCEClusterTask {is_segmented, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%15 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) parent_input(%14 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%17 : !VPUIP.DistributedBuffer<1x64x56x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%188 : memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]>) -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) {
      %27 = VPUIP.NNDMA {port = 0 : i64} inputs(%20 : memref<1x64x28x56xf16, [@CMX_NN, 0]>) outputs(%8 : memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>) -> memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>
    }
    return %arg1 : memref<1x64x56x56xf16, @DDR>
  }

  // CHECK: [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR4:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR5:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

  // CHECK: [[BUF0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
  // CHECK: [[BUF1:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
	
  // Group 0 Tile 0
  // CHECK: VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier)
  // Group 0 Tile 1
  // CHECK: VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier)
      
  // Dummy DMAs Tile 0
  // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
  // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
	
  // Group 1 Tile 0
  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)
  // Group 1 Tile 1
  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)
  
  // Group 2 Tile 0
  // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)
  // Group 2 Tile 1
  // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)

  // End DMA
  // CHECK: VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier)
}