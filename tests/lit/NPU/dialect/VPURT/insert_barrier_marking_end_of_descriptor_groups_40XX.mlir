//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt  --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --insert-barrier-to-mark-end-of-descriptor-group="workload-management-mode=PWLM_V0_LCA" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 0.01269696927538105>
!qElemType2 = !quant.uniform<u8:f16, 0.0173492431640625:114>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// Final State
//    DMA0            DMA1
//      \              /
//            b0
//            |
//          DPU1_0 (Group 1)
//            |
//            b1
//            |
//          DMA2
//            |
//            b2
//            |
//          DMA3
//            |
//            b3
//            |
//          DPU2_0 (Group 2)
//            |              
//            b4
//            |
//          DPU3_0 (Group 3)
//            |
//            b5
//            |
//           DMA
//            |

module @NoInsertionNeeded attributes {VPU.compilationMode = #VPU.compilation_mode<DefaultHW>, VPUIP.wlm_status = #VPUIP.wlm_status<ENABLED>} {
  IE.PipelineOptions @Options {
    IE.Option @VPU.MetadataMaxVariantCount : 8
    IE.Option @VPU.MetadataMaxInvariantCount : 4
    IE.Option @VPU.MetadataMaxKernelInvocationCount : 4
    IE.Option @VPU.MetadataMaxKernelRangeCount : 4
  }
  IE.TileResource 1 of @NCE at 1.700000e+03 MHz {
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

  // CHECK: VPURT.DeclareVirtualBarrier
  // CHECK: VPURT.DeclareVirtualBarrier
  // CHECK: VPURT.DeclareVirtualBarrier
  // CHECK: VPURT.DeclareVirtualBarrier
  // CHECK: VPURT.DeclareVirtualBarrier
  // CHECK: VPURT.DeclareVirtualBarrier
  // CHECK: VPURT.DeclareVirtualBarrier
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 0.01269696927538105>
!qElemType2 = !quant.uniform<u8:f16, 0.0173492431640625:114>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// Initial State
//      DMA0      DMA1
//          \    /
//            b0
//            /\
//      DPU1_0  DPU1_1 (Group1)
//            \/
//            b1
//            |
//          DMA2
//            |
//            b2
//            |
//          DMA3
//            |
//            b3
//            /\
//      DPU2_0  DPU2_1 (Group2)
//            \/              
//            b4
//            /\
//     DPU3_0   DPU3_1 (Group3)
//            \/
//            b5
//            |
//           DMA
//            |

module @NoInsertionNeededMultiTile attributes {VPU.compilationMode = #VPU.compilation_mode<DefaultHW>, VPUIP.wlm_status = #VPUIP.wlm_status<ENABLED>} {
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
    VPURT.Task waits(%x : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
      %27 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%26 : memref<1x16x114x224x!qElemType2, #NHWC, [@CMX_NN, 0]>) weights(%24 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%25 : !VPUIP.DistributedBuffer<1x16x224x224x!qElemType2, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%14 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%166 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]>) -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]> variants : {
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
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) {
      %27 = VPUIP.NCEClusterTask {is_segmented, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%15 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) parent_input(%14 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%17 : !VPUIP.DistributedBuffer<1x64x56x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%188 : memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]>) -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]> variants : {
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

  // CHECK: VPURT.DeclareVirtualBarrier
  // CHECK: VPURT.DeclareVirtualBarrier
  // CHECK: VPURT.DeclareVirtualBarrier
  // CHECK: VPURT.DeclareVirtualBarrier
  // CHECK: VPURT.DeclareVirtualBarrier
  // CHECK: VPURT.DeclareVirtualBarrier
  // CHECK: VPURT.DeclareVirtualBarrier
}

// -----

//
//  Below test is configured to have 3 groups with 1 invariant and 4 variants each
// Initial State
//          DMA0
//            |
//            b0
//            |
//          DPU1_0
//           
//          DPU2_0
//            
//          DPU3_0
//            |
//            b1
//

// Final State
//          DMA0
//            |
//            b0
//            |
//          DPU1_0
//            |
//            b1
//            |
//          DPU2_0
//            |
//            b2
//            |
//          DPU3_0
//            |
//            b1

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 0.01269696927538105>
!qElemType2 = !quant.uniform<u8:f16, 0.0173492431640625:114>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @InsertBarriersWhereNeeded attributes {VPU.compilationMode = #VPU.compilation_mode<DefaultHW>, VPUIP.wlm_status = #VPUIP.wlm_status<ENABLED>} {
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
    VPURT.Task waits(%0 : !VPURT.Barrier) {
      %27 = VPUIP.NCEClusterTask {is_permute_quantize, is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%23 : memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>) weights(%22 : memref<1x224x3x114xf16, #NHWC, [@CMX_NN, 0]>) parent_input(%21 : !VPUIP.DistributedBuffer<1x224x3x224xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%11 : !VPUIP.DistributedBuffer<1x224x4x224x!qElemType, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>) outputs(%12 : memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 0]>) -> memref<1x224x4x114x!qElemType, #NWCH, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task {
      %27 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%26 : memref<1x16x114x224x!qElemType2, #NHWC, [@CMX_NN, 0]>) weights(%24 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%25 : !VPUIP.DistributedBuffer<1x16x224x224x!qElemType2, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%14 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%16 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task updates(%1 : !VPURT.Barrier) {
      %27 = VPUIP.NCEClusterTask {is_segmented, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%15 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) parent_input(%14 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%17 : !VPUIP.DistributedBuffer<1x64x56x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%18 : memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    return %arg1 : memref<1x64x56x56xf16, @DDR>
  }

  // CHECK: [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

  // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
  // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)
}

// -----

//
//  Below test is configured to have 3 groups with 1 invariant each
// Initial State
//                DMA
//                 0
//        DPU1_0        DPU1_1
//        DPU2_0        /  x
//              \      /
//                 1
//              /     \
//        DPU3_0        DPU3_1
//        DPU4_0        DPU4_1
//        DPU5_0        DPU5_1
//        DPU6_0        DPU6_1
//        DPU7_0        DPU7_1
//             \        /
//                 2
//                 |
//                DMA
//
// [Group1](DPU1, DPU2, DPU3) [Group2](DPU4, DPU5, DPU6) [Group3](DPU7)
//
// Group Barrier Dependencies
//    Group 1 doesn't have update barrier
//    Group 2 doesn't have update barrier
//
//            0
//            |
//          Group1
//
//          Group2
//
//          Group3
//            |
//            2


// Final State
//                DMA
//                 |
//                b0
//              /     \
//        DPU1_0        DPU1_1
//        DPU2_0       /  x
//              \     /
//                b1
//              /     \
//        DPU3_0        DPU3_1
//              \     /
//                b2
//        DPU4_0  |      DPU4_1
//        DPU5_0  |      DPU5_1
//                |
//               /  \ 
//              /    \ 
//             /      \ 
//        DPU6_0        DPU6_1
//              \    /
//                b4
//                / \
//               /   \
//              /     \
//        DPU7_0        DPU7_1
//             \      /
//              \    /
//                b5
//                |
//               DMA
//

!qElemType = !quant.uniform<u8<0:254>:f16:0, {2.3954496608944391E-5:127,1.8652968519315946E-5:127,5.714536651851624E-6:127,9.2288640540415852E-6:127,5.2774985005536418E-5:127,1.0355251041922983E-5:127,4.7608623354453737E-5:127,4.2622483621432085E-5:127,6.3378041184793303E-5:127,9.0411328893946842E-6:127,3.7636343888410434E-5:127,9.1462623415969495E-6:127,3.0472522645484744E-5:127,1.3648806181992955E-6:127,8.4283783679872047E-5:127,5.5778683639886814E-5:127,1.6640490434301182E-5:127,5.4847537063238186E-5:127,9.0531476839320868E-5:127,1.0873389056348425E-5:127,8.4944597379429132E-5:127,5.5928868571604333E-5:127,7.9477865864911412E-5:127,1.5408973994217519E-5:127,2.7033287709153543E-5:127,7.3740801473302161E-6:127,4.5475997324064961E-5:127,4.5415923351377953E-5:127,1.4605484609528789E-5:127,8.4554116556963585E-6:127,1.9478985643762302E-5:127,4.6332051434854825E-6:127,9.6568911094365155E-6:127,1.3648806594488189E-4:127,4.8584825410617617E-6:127,1.0588037686085138E-5:127,9.6493818628506396E-6:127,1.0663130151943897E-5:127,3.8086898683562992E-5:127,4.142100416769193E-5:127,7.5805844284418062E-6:127,5.2684874046505907E-5:127,9.1462623415969495E-6:127,1.327634796382874E-5:127,6.1538275771253692E-6:127,2.9691561000553643E-5:127,5.6079053503321852E-5:127,1.4605484609528789E-5:127,2.8324878121924213E-5:127,6.1695969949557085E-5:127,5.0492174043430117E-5:127,2.3218590443528543E-5:127,1.5026002418337845E-5:127,4.4184406911294294E-5:127,4.0009265809547241E-5:127,2.0289984275036911E-5:127,2.0680465097502461E-5:127,5.5906340831846705E-6:127,1.6054769200602853E-5:127,1.0940972275621308E-5:127,2.3008331539124016E-5:127,3.4182090458907481E-5:127,9.4766691913754922E-6:127,8.9450145330954728E-5:127}>
!qElemType3 = !quant.uniform<u8:f16, 1.000000e+00:114>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

module @InsertBarriersWhereNeededMultiTile attributes {VPU.compilationMode = #VPU.compilation_mode<DefaultHW>, VPUIP.wlm_status = #VPUIP.wlm_status<ENABLED>} {
  IE.PipelineOptions @Options {
    IE.Option @VPU.MetadataMaxVariantCount : 12
    IE.Option @VPU.MetadataMaxInvariantCount : 6
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

func.func @main(%arg0: memref<1x3x224x224xf16, @DDR>, %arg1: memref<1x64x112x112xf16, @DDR>) -> memref<1x64x112x112xf16, @DDR> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // buffers
    %3 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %7 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>
    %11 = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<1x3x56x224xf16, [@CMX_NN, 0]>
    %20 = VPURT.DeclareBuffer <CMX_NN> [0] <140736> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %21 = VPURT.DeclareBuffer <CMX_NN> [1] <140736> -> memref<64x1x1x4xsi32, [@CMX_NN, 1]>
    %22 = VPURT.DeclareBuffer <CMX_NN> [2] <140736> -> memref<64x1x1x4xsi32, [@CMX_NN, 2]>
    %23 = VPURT.DeclareBuffer <CMX_NN> [3] <140736> -> memref<64x1x1x4xsi32, [@CMX_NN, 3]>
    %25 = VPURT.DeclareBuffer <CMX_NN> [0] <141760> -> memref<1x64x28x112xf16, [@CMX_NN, 0]>
    %29 = VPURT.DeclareBuffer <CMX_NN> [0] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>
    %30 = VPURT.DeclareBuffer <CMX_NN> [1] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>
    %31 = VPURT.DeclareBuffer <CMX_NN> [2] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 2]>
    %32 = VPURT.DeclareBuffer <CMX_NN> [3] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 3]>
    %41 = VPURT.DeclareBuffer <CMX_NN> [0] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>
    %42 = VPURT.DeclareBuffer <CMX_NN> [1] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>
    %43 = VPURT.DeclareBuffer <CMX_NN> [2] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 2]>
    %44 = VPURT.DeclareBuffer <CMX_NN> [3] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 3]>
    %45 = VPURT.DeclareBuffer <CMX_NN> [0] <75840> -> memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>
    %46 = VPURT.DeclareBuffer <CMX_NN> [1] <75840> -> memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>
    %47 = VPURT.DeclareBuffer <CMX_NN> [2] <75840> -> memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 2]>
    %48 = VPURT.DeclareBuffer <CMX_NN> [3] <75840> -> memref<1x16x59x224x!qElemType3, #NHWC, [@CMX_NN, 3]>

    // DMA
    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      %49 = VPUIP.NNDMA {port = 0 : i64} inputs(%3 : memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%11 : memref<1x3x56x224xf16, [@CMX_NN, 0]>) -> memref<1x3x56x224xf16, [@CMX_NN, 0]>
    }

    // DPU1
    VPURT.Task waits(%bar0: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }

    // DPU2 - Theoretical case running on 1 tile only
    VPURT.Task  updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }

    // DPU3
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }

    // DPU4
    VPURT.Task {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }

    // DPU5
    VPURT.Task {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }

    // DPU6
    VPURT.Task {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }

    // DPU7
    VPURT.Task updates(%bar2: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task updates(%bar2: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }

    // End DMA
    VPURT.Task waits(%bar2: !VPURT.Barrier) {
      %49 = VPUIP.NNDMA {port = 0 : i64} inputs(%25 : memref<1x64x28x112xf16, [@CMX_NN, 0]>) outputs(%7 : memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>) -> memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>
    }
    return %arg1 : memref<1x64x112x112xf16, @DDR>
  }
  // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK: [[BAR4:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  
  // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) {
  // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
  // CHECK: VPURT.Task updates([[BAR1]] : !VPURT.Barrier) {
  // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
  // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
  // CHECK: VPURT.Task {
  // CHECK: VPURT.Task {
  // CHECK: VPURT.Task {
  // CHECK: VPURT.Task {
  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)
  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)
  // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)
  // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)
}
