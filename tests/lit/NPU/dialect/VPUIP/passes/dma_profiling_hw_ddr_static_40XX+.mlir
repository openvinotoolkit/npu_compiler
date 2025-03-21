//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch% allow-custom-values=true" --dma-task-profiling-hw-ddr="dma-profiling=static" %s | FileCheck %s
// REQUIRES: arch-NPU40XX
!dataType = memref<1x16x4x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>

module @DMAGraph {
  IE.TileResource 1 of @NCE at 1.300000e+03 MHz {
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        IE.MemoryResource 512 bytes of @CMX_NN offset 0
      }
    }
  }

  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  } profilingOutputsInfo :  {
  }
  func.func @main(%arg0: !dataType, %arg1: !dataType) -> !dataType {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %buf0 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> !dataType
    %buf1 = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> !dataType

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %dma0 = VPUIP.NNDMA inputs(%arg0 : !dataType) outputs(%buf0 : !dataType) -> !dataType
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %dma0 = VPUIP.NNDMA inputs(%buf0 : !dataType) outputs(%buf1 : !dataType) -> !dataType
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %dma0 = VPUIP.NNDMA inputs(%buf1 : !dataType) outputs(%arg1 : !dataType) -> !dataType
    }

    return %arg1 : !dataType
  }
}

// CHECK:        profilingOutputsInfo
// CHECK-NEXT:   DataInfo "dmahw" : tensor<256xui8>
// CHECK:        func.func @main(%arg0: memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>,
// CHECK-SAME:       %arg1: memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>,
// CHECK-SAME:       %arg2: memref<256xui8, [@DDR, 0]>) ->
// CHECK-SAME:       (memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>,
// CHECK-SAME:       memref<256xui8, [@DDR, 0]>) {
// CHECK:    [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier
// CHECK:    [[BUF_DATA_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
// CHECK:    [[BUF_DATA_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>

// Profiled DMA task 1
// CHECK:  VPURT.Task
// CHECK-NEXT:    VPUIP.NNDMA {dma_hwp_id = 1 : si32,
// CHECK-SAME:        profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 1 : i64>}
// CHECK-SAME:        inputs(%arg0 :
// CHECK-SAME:        outputs([[BUF_DATA_0]] :

// Profiled DMA task 2
// CHECK:  VPURT.Task
// CHECK-NEXT:    VPUIP.NNDMA {dma_hwp_id = 2 : si32
// CHECK-SAME:        profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 2 : i64>}
// CHECK-SAME:        inputs([[BUF_DATA_0]] :
// CHECK-SAME:        outputs([[BUF_DATA_1]] :

// Profiled DMA task 3
// CHECK:  VPURT.Task
// CHECK-NEXT:    VPUIP.NNDMA {dma_hwp_id = 3 : si32
// CHECK-SAME:        profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 3 : i64>}
// CHECK-SAME:        inputs([[BUF_DATA_1]] :
// CHECK-SAME:        outputs(%arg1 :

// Check network output
// CHECK:   return %arg1, %arg2
// CHECK-SAME:    memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>,
// CHECK-SAME:    memref<256xui8, [@DDR, 0]>
