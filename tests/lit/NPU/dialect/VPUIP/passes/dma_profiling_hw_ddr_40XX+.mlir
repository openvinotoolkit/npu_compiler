//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --dma-task-profiling-hw-ddr="dma-profiling=true" %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

!dataType = memref<1x16x4x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>

// CHECK-LABEL: @DMAGraph
module @DMAGraph {
  config.Resources 1 of @global {
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        config.MemoryResource 4096 bytes of @DDR offset 0
      }
    }
  }

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  } profilingOutputsInfo :  {
  }
  func.func @main(%arg0: !dataType, %arg1: !dataType) -> !dataType {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %buf0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !dataType
    %buf1 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> !dataType

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %dma0 = VPUIP.NNDMA { port = 0 : i64 } inputs(%arg0 : !dataType) outputs(%buf0 : !dataType) -> !dataType
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %dma0 = VPUIP.NNDMA { port = 0 : i64 } inputs(%buf0 : !dataType) outputs(%buf1 : !dataType) -> !dataType
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %dma0 = VPUIP.NNDMA { port = 0 : i64 } inputs(%buf1 : !dataType) outputs(%arg1 : !dataType) -> !dataType
    }

    return %arg1 : !dataType
  }

// CHECK:        profilingOutputsInfo
// CHECK-NEXT:   DataInfo "dmahw" : tensor<256xui8>
// CHECK:        func.func @main(%arg0: memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>,
// CHECK-SAME:       %arg1: memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>,
// CHECK-SAME:       %arg2: memref<256xui8>) ->
// CHECK-SAME:       (memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>,
// CHECK-SAME:       memref<256xui8>) {
// CHECK:    [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier
// CHECK:    [[PROF_BAR:%.+]] = VPURT.DeclareVirtualBarrier
// CHECK:    [[PROF_DATA_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <64> -> memref<192xui8, @DDR>
// CHECK:    [[PROF_DATA_OUT:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <64> -> memref<192xui8>
// CHECK:    [[BUF_DATA_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
// CHECK:    [[BUF_DATA_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>

// Profiled DMA task 1
// CHECK:  VPURT.Task
// CHECK-SAME:        updates([[BAR0]] : !VPURT.Barrier)
// CHECK-NEXT:    VPUIP.NNDMA {dma_hwp_id = 1 : si32,
// CHECK-SAME:        profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 1 : i64>}
// CHECK-SAME:        inputs(%arg0 :
// CHECK-SAME:        outputs([[BUF_DATA_0]] :

// Profiled DMA task 2
// CHECK:  VPURT.Task
// CHECK-SAME:        waits([[BAR0]] : !VPURT.Barrier)
// CHECK-NEXT:    VPUIP.NNDMA {dma_hwp_id = 2 : si32
// CHECK-SAME:        profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 2 : i64>}
// CHECK-SAME:        inputs([[BUF_DATA_0]] :
// CHECK-SAME:        outputs([[BUF_DATA_1]] :

// Profiled DMA task 3
// CHECK:  VPURT.Task
// CHECK-SAME:        updates([[PROF_BAR]] : !VPURT.Barrier)
// CHECK-NEXT:    VPUIP.NNDMA {dma_hwp_id = 3 : si32
// CHECK-SAME:        profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 3 : i64>}
// CHECK-SAME:        inputs([[BUF_DATA_1]] :
// CHECK-SAME:        outputs(%arg1 :

// DMA HWP DDR2DDR data copy
// CHECK:  VPURT.Task
// CHECK-SAME:        waits([[PROF_BAR]] : !VPURT.Barrier)
// CHECK-NEXT:    VPUIP.NNDMA
// CHECK-SAME:        inputs([[PROF_DATA_DDR]] :
// CHECK-SAME:        outputs([[PROF_DATA_OUT]] :

// Check network output
// CHECK:   return %arg1, %arg2
// CHECK-SAME:    memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>,
// CHECK-SAME:    memref<256xui8>

}

// -----

!dataTypeCMX = memref<1x16x4x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
!dataTypeDDR = memref<1x16x4x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>

// CHECK-LABEL: @DMAComplexGraph
module @DMAComplexGraph {
  config.Resources 1 of @global {
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        config.MemoryResource 4096 bytes of @DDR offset 0
      }
    }
  }

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  } profilingOutputsInfo :  {
  }

  func.func @main(%arg0: !dataTypeCMX, %arg1: !dataTypeCMX) -> !dataTypeCMX {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %cmxbuf0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !dataTypeCMX
    %cmxbuf1 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> !dataTypeCMX
    %ddrbuf0 = VPURT.DeclareBuffer <DDR> <0> -> !dataTypeDDR
    %ddrbuf1 = VPURT.DeclareBuffer <DDR> <512> -> !dataTypeDDR

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %dma0 = VPUIP.NNDMA { port = 0 : i64 } inputs(%arg0 : !dataTypeCMX) outputs(%cmxbuf0 : !dataTypeCMX) -> !dataTypeCMX
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %dma0 = VPUIP.NNDMA { port = 1 : i64 } inputs(%cmxbuf0 : !dataTypeCMX) outputs(%ddrbuf0 : !dataTypeDDR) -> !dataTypeDDR
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %dma0 = VPUIP.NNDMA { port = 0 : i64 } inputs(%ddrbuf1 : !dataTypeDDR) outputs(%ddrbuf0 : !dataTypeDDR) -> !dataTypeDDR
    }

    VPURT.Task waits(%bar2 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %dma0 = VPUIP.NNDMA { port = 1 : i64 } inputs(%ddrbuf1 : !dataTypeDDR) outputs(%arg1 : !dataTypeCMX) -> !dataTypeCMX
    }

    return %arg1 : !dataTypeCMX
  }

// CHECK:        profilingOutputsInfo
// CHECK-NEXT:   DataInfo "dmahw" : tensor<320xui8>
// CHECK:        func.func @main(%arg0: memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>,
// CHECK-SAME:       %arg1: memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>,
// CHECK-SAME:       %arg2: memref<320xui8>) ->
// CHECK-SAME:       (memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>,
// CHECK-SAME:       memref<320xui8>) {
// CHECK:    [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier
// CHECK:    [[PROF_BAR:%.+]] = VPURT.DeclareVirtualBarrier
// CHECK:    [[PROF_DATA_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <64> -> memref<256xui8, @DDR>
// CHECK:    [[PROF_DATA_OUT:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <64> -> memref<256xui8>
// CHECK:    [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier
// CHECK:    [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier
// CHECK:    [[CMX_BUF_DATA_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
// CHECK:    [[CMX_BUF_DATA_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
// CHECK:    [[DDR_BUF_DATA_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x4x4xf16, #NHWC, @DDR>
// CHECK:    [[DDR_BUF_DATA_1:%.+]] = VPURT.DeclareBuffer <DDR> <512> -> memref<1x16x4x4xf16, #NHWC, @DDR>

// Profiled DMA task 1
// CHECK:  VPURT.Task
// CHECK-SAME:        updates([[BAR0]], [[PROF_BAR]]
// CHECK-NEXT:    VPUIP.NNDMA {dma_hwp_id = 1 : si32, port = 0 : i64
// CHECK-SAME:        profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 1 : i64>}
// CHECK-SAME:        inputs(%arg0 :
// CHECK-SAME:        outputs([[CMX_BUF_DATA_0]] :

// Profiled DMA task 2
// CHECK:  VPURT.Task
// CHECK-SAME:        waits([[BAR0]]
// CHECK-SAME:        updates([[BAR1]], [[PROF_BAR]]
// CHECK-NEXT:    VPUIP.NNDMA {dma_hwp_id = 2 : si32, port = 1 : i64
// CHECK-SAME:        profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 2 : i64>}
// CHECK-SAME:        inputs([[CMX_BUF_DATA_0]] :
// CHECK-SAME:        outputs([[DDR_BUF_DATA_0]] :

// Profiled DMA task 3
// CHECK:  VPURT.Task
// CHECK-SAME:        waits([[BAR1]]
// CHECK-SAME:        updates([[BAR2]], [[PROF_BAR]]
// CHECK-NEXT:    VPUIP.NNDMA {dma_hwp_id = 3 : si32, port = 0 : i64
// CHECK-SAME:        profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 3 : i64>}
// CHECK-SAME:        inputs([[DDR_BUF_DATA_1]] :
// CHECK-SAME:        outputs([[DDR_BUF_DATA_0]] :

// Profiled DMA task 4
// CHECK:  VPURT.Task
// CHECK-SAME:        waits([[BAR2]]
// CHECK-SAME:        updates([[PROF_BAR]]
// CHECK-NEXT:    VPUIP.NNDMA {dma_hwp_id = 4 : si32, port = 1 : i64
// CHECK-SAME:        profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 4 : i64>}
// CHECK-SAME:        inputs([[DDR_BUF_DATA_1]] :
// CHECK-SAME:        outputs(%arg1 :

// DMA HWP DDR2DDR data copy
// CHECK:  VPURT.Task
// CHECK-SAME:        waits([[PROF_BAR]]
// CHECK-NEXT:    VPUIP.NNDMA
// CHECK-SAME:        inputs([[PROF_DATA_DDR]] :
// CHECK-SAME:        outputs([[PROF_DATA_OUT]] :

// Check network output
// CHECK:   return %arg1, %arg2
// CHECK-SAME:    memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>,
// CHECK-SAME:    memref<320xui8>

}
