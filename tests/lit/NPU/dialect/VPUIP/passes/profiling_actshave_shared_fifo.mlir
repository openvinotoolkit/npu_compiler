//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --act-shave-profiling %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Output_DDR = memref<1x48x60x60xf16, #NHWC, @DDR>

!Input_CMX = memref<1x16x62x62xf16, #NHWC, [@CMX_NN, 0]>
!Output_CMX = memref<1x48x60x60xf16, #NHWC, [@CMX_NN, 0]>
!Weights_CMX = memref<48x16x3x3xf16, #NHWC, [@CMX_NN, 0]>

// CHECK-LABEL: module @ActShaveProfilingSimple
module @ActShaveProfilingSimple  {
  config.PipelineOptions @Options {
    config.Option @config.UseDedicatedFifoPerShaveEngine : false
  }

  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x16x62x62xf16>
    DataInfo "weights" : tensor<48x16x3x3xf16>
    DataInfo "output_cmx" : tensor<1x48x60x60xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x48x60x60xf16>
  } profilingOutputsInfo :  {
    // CHECK:       profilingOutputsInfo : {
    // CHECK-NEXT:    DataInfo "actshave" : tensor<8xui32>
  }
  config.Resources 2 of @NCE at 1.850000e+03 MHz {
    builtin.module @ReservedMemory {
      module @ActShaveProfilingReservedMemory {
        config.MemoryResource 512 bytes of @CMX_NN offset 1472000
      }
    }
  }

  func.func @main(%input: !Input_CMX, %weights: !Weights_CMX, %output_cmx: !Output_CMX, %output_ddr: !Output_DDR) -> !Output_DDR {

    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> !Output_CMX
    %1 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%input as %arg3: !Input_CMX) outputs(%output_cmx as %arg4: !Output_CMX) on tile 0 -> !Output_CMX {
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !Input_CMX, !Output_CMX
      }
    }
    VPURT.Task waits(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA inputs(%output_cmx : !Output_CMX) outputs(%output_ddr : !Output_DDR) -> !Output_DDR
    }
    return %output_ddr : !Output_DDR
  }
    // CHECK:       [[BARRIER_NCE:%.+]] = VPURT.DeclareVirtualBarrier
    // CHECK:       [[PROF_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <[[CMX_OFFSET:[0-9]+]]> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       [[BARRIER_FLUSH:%.+]] = VPURT.DeclareVirtualBarrier
    // CHECK:       VPURT.Task updates([[BARRIER_NCE]], [[BARRIER_FLUSH]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.SW.Kernel
    // CHECK-SAME:      #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 0 : i64, dataIndex = 0 : i64, tileId = 0 : i64, clusterId = 0 : i64>
    // CHECK-SAME:      profiling_data([[PROF_CMX]]
    // CHECK:       [[FLUSH_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <[[CMX_OFFSET]]> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       [[FLUSH_DDR:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<8xui32, @DDR>
    // CHECK:       VPURT.Task waits([[BARRIER_FLUSH]] : !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.NNDMA <{port = 0 : i64, profiling_buffer_mgmt}>
    // CHECK-SAME:      inputs([[FLUSH_CMX]]
    // CHECK-SAME:      outputs([[FLUSH_DDR]]
    // CHECK:       return %arg3, %arg4 : memref<1x48x60x60xf16, #NHWC, @DDR>, memref<8xui32>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Output_DDR = memref<1x48x60x60xf16, #NHWC, @DDR>

!Input_CMX = memref<1x16x62x62xf16, #NHWC, [@CMX_NN, 0]>
!Output_CMX = memref<1x48x60x60xf16, #NHWC, [@CMX_NN, 0]>
!Weights_CMX = memref<48x16x3x3xf16, #NHWC, [@CMX_NN, 0]>

// Twelve SW tasks and 128-byte CMX profiling subbuffers holding four records each, so the pass
// flushes in this order: subbuffer #0 -> subbuffer #1 -> subbuffer #0. The third group reuses an
// already flushed subbuffer, so every task in that group is held back by the end barrier of the
// flush that drained it. Since dedicated per-engine FIFOs are disabled here, every task in every
// group (not just boundary tasks) updates the barrier that gates its group's flush.

// CHECK-LABEL: module @ActShaveProfilingWithProfilingSubbufferReuse
module @ActShaveProfilingWithProfilingSubbufferReuse  {
  config.PipelineOptions @Options {
    config.Option @config.UseDedicatedFifoPerShaveEngine : false
  }

  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x16x62x62xf16>
    DataInfo "weights" : tensor<48x16x3x3xf16>
    DataInfo "output_cmx" : tensor<1x48x60x60xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x48x60x60xf16>
  } profilingOutputsInfo :  {
    // CHECK:       profilingOutputsInfo : {
    // CHECK-NEXT:    DataInfo "actshave" : tensor<96xui32>
  }
  config.Resources 2 of @NCE at 1.850000e+03 MHz {
    builtin.module @ReservedMemory {
      module @ActShaveProfilingReservedMemory {
        config.MemoryResource 256 bytes of @CMX_NN offset 1472000
      }
    }
  }

  func.func @main(%input: !Input_CMX, %weights: !Weights_CMX, %output_cmx: !Output_CMX, %output_ddr: !Output_DDR) -> !Output_DDR {

    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> !Output_CMX
    %1 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%input as %arg3: !Input_CMX) outputs(%output_cmx as %arg4: !Output_CMX) on tile 0 -> !Output_CMX {
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !Input_CMX, !Output_CMX
      }
    }
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%input as %arg3: !Input_CMX) outputs(%output_cmx as %arg4: !Output_CMX) on tile 0 -> !Output_CMX {
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !Input_CMX, !Output_CMX
      }
    }
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%input as %arg3: !Input_CMX) outputs(%output_cmx as %arg4: !Output_CMX) on tile 0 -> !Output_CMX {
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !Input_CMX, !Output_CMX
      }
    }
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%input as %arg3: !Input_CMX) outputs(%output_cmx as %arg4: !Output_CMX) on tile 0 -> !Output_CMX {
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !Input_CMX, !Output_CMX
      }
    }
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%input as %arg3: !Input_CMX) outputs(%output_cmx as %arg4: !Output_CMX) on tile 0 -> !Output_CMX {
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !Input_CMX, !Output_CMX
      }
    }
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%input as %arg3: !Input_CMX) outputs(%output_cmx as %arg4: !Output_CMX) on tile 0 -> !Output_CMX {
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !Input_CMX, !Output_CMX
      }
    }
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%input as %arg3: !Input_CMX) outputs(%output_cmx as %arg4: !Output_CMX) on tile 0 -> !Output_CMX {
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !Input_CMX, !Output_CMX
      }
    }
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%input as %arg3: !Input_CMX) outputs(%output_cmx as %arg4: !Output_CMX) on tile 0 -> !Output_CMX {
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !Input_CMX, !Output_CMX
      }
    }
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%input as %arg3: !Input_CMX) outputs(%output_cmx as %arg4: !Output_CMX) on tile 0 -> !Output_CMX {
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !Input_CMX, !Output_CMX
      }
    }
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%input as %arg3: !Input_CMX) outputs(%output_cmx as %arg4: !Output_CMX) on tile 0 -> !Output_CMX {
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !Input_CMX, !Output_CMX
      }
    }
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%input as %arg3: !Input_CMX) outputs(%output_cmx as %arg4: !Output_CMX) on tile 0 -> !Output_CMX {
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !Input_CMX, !Output_CMX
      }
    }
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%input as %arg3: !Input_CMX) outputs(%output_cmx as %arg4: !Output_CMX) on tile 0 -> !Output_CMX {
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !Input_CMX, !Output_CMX
      }
    }
    VPURT.Task waits(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA inputs(%output_cmx : !Output_CMX) outputs(%output_ddr : !Output_DDR) -> !Output_DDR
    }
    return %output_ddr : !Output_DDR
  }
    // CHECK:       [[BARRIER_NCE:%.+]] = VPURT.DeclareVirtualBarrier

    // CHECK:       [[PROF_CMX_A:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472000> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       [[BARRIER_FLUSH_0:%.+]] = VPURT.DeclareVirtualBarrier
    // CHECK:       VPURT.Task updates([[BARRIER_NCE]], [[BARRIER_FLUSH_0]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.SW.Kernel
    // CHECK-SAME:      #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 0 : i64, dataIndex = 0 : i64,  tileId = 0 : i64, clusterId = 0 : i64>
    // CHECK-SAME:      profiling_data([[PROF_CMX_A]]

    // CHECK:       [[PROF_CMX_B:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472032> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       VPURT.Task updates([[BARRIER_NCE]], [[BARRIER_FLUSH_0]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.SW.Kernel
    // CHECK-SAME:      #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 0 : i64, dataIndex = 1 : i64,  tileId = 0 : i64, clusterId = 0 : i64>
    // CHECK-SAME:      profiling_data([[PROF_CMX_B]]

    // CHECK:       [[PROF_CMX_C:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472064> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       VPURT.Task updates([[BARRIER_NCE]], [[BARRIER_FLUSH_0]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.SW.Kernel
    // CHECK-SAME:      #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 0 : i64, dataIndex = 2 : i64,  tileId = 0 : i64, clusterId = 0 : i64>
    // CHECK-SAME:      profiling_data([[PROF_CMX_C]]

    // CHECK:       [[PROF_CMX_D:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472096> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       VPURT.Task updates([[BARRIER_NCE]], [[BARRIER_FLUSH_0]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.SW.Kernel
    // CHECK-SAME:      #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 0 : i64, dataIndex = 3 : i64,  tileId = 0 : i64, clusterId = 0 : i64>
    // CHECK-SAME:      profiling_data([[PROF_CMX_D]]

    // CHECK:       [[FLUSH_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472000> -> memref<32xui32, [@CMX_NN, 0]>
    // CHECK:       [[FLUSH_DDR_0:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<32xui32, @DDR>
    // CHECK:       [[BARRIER_FLUSH_END_0:%.+]] = VPURT.DeclareVirtualBarrier
    // CHECK:       VPURT.Task waits([[BARRIER_FLUSH_0]] : !VPURT.Barrier) updates([[BARRIER_FLUSH_END_0]] : !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.NNDMA <{port = 0 : i64, profiling_buffer_mgmt}>
    // CHECK-SAME:      inputs([[FLUSH_CMX_0]]
    // CHECK-SAME:      outputs([[FLUSH_DDR_0]]

    // CHECK:       [[PROF_CMX_E:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472128> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       [[BARRIER_FLUSH_1:%.+]] = VPURT.DeclareVirtualBarrier
    // CHECK:       VPURT.Task updates([[BARRIER_NCE]], [[BARRIER_FLUSH_1]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.SW.Kernel
    // CHECK-SAME:      #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 4 : i64, clusterSize = 0 : i64, dataIndex = 0 : i64,  tileId = 0 : i64, clusterId = 0 : i64>
    // CHECK-SAME:      profiling_data([[PROF_CMX_E]]

    // CHECK:       [[PROF_CMX_F:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472160> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       VPURT.Task updates([[BARRIER_NCE]], [[BARRIER_FLUSH_1]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.SW.Kernel
    // CHECK-SAME:      #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 4 : i64, clusterSize = 0 : i64, dataIndex = 1 : i64,  tileId = 0 : i64, clusterId = 0 : i64>
    // CHECK-SAME:      profiling_data([[PROF_CMX_F]]

    // CHECK:       [[PROF_CMX_G:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472192> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       VPURT.Task updates([[BARRIER_NCE]], [[BARRIER_FLUSH_1]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.SW.Kernel
    // CHECK-SAME:      #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 4 : i64, clusterSize = 0 : i64, dataIndex = 2 : i64,  tileId = 0 : i64, clusterId = 0 : i64>
    // CHECK-SAME:      profiling_data([[PROF_CMX_G]]

    // CHECK:       [[PROF_CMX_H:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472224> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       VPURT.Task updates([[BARRIER_NCE]], [[BARRIER_FLUSH_1]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.SW.Kernel
    // CHECK-SAME:      #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 4 : i64, clusterSize = 0 : i64, dataIndex = 3 : i64,  tileId = 0 : i64, clusterId = 0 : i64>
    // CHECK-SAME:      profiling_data([[PROF_CMX_H]]

    // CHECK:       [[FLUSH_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472128> -> memref<32xui32, [@CMX_NN, 0]>
    // CHECK:       [[FLUSH_DDR_1:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <128> -> memref<32xui32, @DDR>
    // CHECK:       VPURT.Task waits([[BARRIER_FLUSH_1]] : !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.NNDMA <{port = 0 : i64, profiling_buffer_mgmt}>
    // CHECK-SAME:      inputs([[FLUSH_CMX_1]]
    // CHECK-SAME:      outputs([[FLUSH_DDR_1]]

    // CHECK:       [[PROF_CMX_I:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472000> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       [[BARRIER_FLUSH_2:%.+]] = VPURT.DeclareVirtualBarrier
    // CHECK:       VPURT.Task waits([[BARRIER_FLUSH_END_0]] : !VPURT.Barrier) updates([[BARRIER_NCE]], [[BARRIER_FLUSH_2]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.SW.Kernel
    // CHECK-SAME:      #VPUIP.SwProfilingMetadataAttr<bufferId = 2 : i64, bufferOffset = 8 : i64, clusterSize = 0 : i64, dataIndex = 0 : i64,  tileId = 0 : i64, clusterId = 0 : i64>
    // CHECK-SAME:      profiling_data([[PROF_CMX_I]]

    // CHECK:       [[PROF_CMX_J:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472032> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       VPURT.Task waits([[BARRIER_FLUSH_END_0]] : !VPURT.Barrier) updates([[BARRIER_NCE]], [[BARRIER_FLUSH_2]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.SW.Kernel
    // CHECK-SAME:      #VPUIP.SwProfilingMetadataAttr<bufferId = 2 : i64, bufferOffset = 8 : i64, clusterSize = 0 : i64, dataIndex = 1 : i64,  tileId = 0 : i64, clusterId = 0 : i64>
    // CHECK-SAME:      profiling_data([[PROF_CMX_J]]

    // CHECK:       [[PROF_CMX_K:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472064> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       VPURT.Task waits([[BARRIER_FLUSH_END_0]] : !VPURT.Barrier) updates([[BARRIER_NCE]], [[BARRIER_FLUSH_2]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.SW.Kernel
    // CHECK-SAME:      #VPUIP.SwProfilingMetadataAttr<bufferId = 2 : i64, bufferOffset = 8 : i64, clusterSize = 0 : i64, dataIndex = 2 : i64,  tileId = 0 : i64, clusterId = 0 : i64>
    // CHECK-SAME:      profiling_data([[PROF_CMX_K]]

    // CHECK:       [[PROF_CMX_L:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472096> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       VPURT.Task waits([[BARRIER_FLUSH_END_0]] : !VPURT.Barrier) updates([[BARRIER_NCE]], [[BARRIER_FLUSH_2]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.SW.Kernel
    // CHECK-SAME:      #VPUIP.SwProfilingMetadataAttr<bufferId = 2 : i64, bufferOffset = 8 : i64, clusterSize = 0 : i64, dataIndex = 3 : i64,  tileId = 0 : i64, clusterId = 0 : i64>
    // CHECK-SAME:      profiling_data([[PROF_CMX_L]]

    // CHECK:       [[FLUSH_CMX_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472000> -> memref<32xui32, [@CMX_NN, 0]>
    // CHECK:       [[FLUSH_DDR_2:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <256> -> memref<32xui32, @DDR>
    // CHECK:       VPURT.Task waits([[BARRIER_FLUSH_2]] : !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.NNDMA <{port = 0 : i64, profiling_buffer_mgmt}>
    // CHECK-SAME:      inputs([[FLUSH_CMX_2]]
    // CHECK-SAME:      outputs([[FLUSH_DDR_2]]

    // CHECK:       return %arg3, %arg4 : memref<1x48x60x60xf16, #NHWC, @DDR>, memref<96xui32>

}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Output_DDR = memref<1x48x60x60xf16, #NHWC, @DDR>

!Input_CMX = memref<1x16x62x62xf16, #NHWC, [@CMX_NN, 0]>
!Output_CMX = memref<1x48x60x60xf16, #NHWC, [@CMX_NN, 0]>
!Weights_CMX = memref<48x16x3x3xf16, #NHWC, [@CMX_NN, 0]>

// Twelve SW tasks and 128-byte CMX profiling subbuffers holding four records each, so the pass
// flushes in this order: subbuffer #0 -> subbuffer #1 -> subbuffer #0.
// Only the ninth through twelfth tasks reuse an already flushed subbuffer (#0), so they are the
// only ones held back by the end barrier of the first flush -- this shows that with a full
// four-task group in the reused buffer, every task in that group (not just a single task) must
// wait for the prior flush to complete before writing its profiling record.

// CHECK-LABEL: module @ActShaveProfilingWithTripleFlushAndSubbufferReuse
module @ActShaveProfilingWithTripleFlushAndSubbufferReuse  {
  config.PipelineOptions @Options {
    config.Option @config.UseDedicatedFifoPerShaveEngine : false
  }

  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x16x62x62xf16>
    DataInfo "weights" : tensor<48x16x3x3xf16>
    DataInfo "output_cmx" : tensor<1x48x60x60xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x48x60x60xf16>
  } profilingOutputsInfo :  {
    // CHECK:       profilingOutputsInfo : {
    // CHECK-NEXT:    DataInfo "actshave" : tensor<96xui32>
  }
  config.Resources 2 of @NCE at 1.850000e+03 MHz {
    builtin.module @ReservedMemory {
      module @ActShaveProfilingReservedMemory {
        config.MemoryResource 256 bytes of @CMX_NN offset 1472000
      }
    }
  }

  func.func @main(%input: !Input_CMX, %weights: !Weights_CMX, %output_cmx: !Output_CMX, %output_ddr: !Output_DDR) -> !Output_DDR {

    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> !Output_CMX
    %1 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%input as %arg3: !Input_CMX) outputs(%output_cmx as %arg4: !Output_CMX) on tile 0 -> !Output_CMX {
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !Input_CMX, !Output_CMX
      }
    }
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%input as %arg3: !Input_CMX) outputs(%output_cmx as %arg4: !Output_CMX) on tile 0 -> !Output_CMX {
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !Input_CMX, !Output_CMX
      }
    }
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%input as %arg3: !Input_CMX) outputs(%output_cmx as %arg4: !Output_CMX) on tile 0 -> !Output_CMX {
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !Input_CMX, !Output_CMX
      }
    }
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%input as %arg3: !Input_CMX) outputs(%output_cmx as %arg4: !Output_CMX) on tile 0 -> !Output_CMX {
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !Input_CMX, !Output_CMX
      }
    }
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%input as %arg3: !Input_CMX) outputs(%output_cmx as %arg4: !Output_CMX) on tile 0 -> !Output_CMX {
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !Input_CMX, !Output_CMX
      }
    }
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%input as %arg3: !Input_CMX) outputs(%output_cmx as %arg4: !Output_CMX) on tile 0 -> !Output_CMX {
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !Input_CMX, !Output_CMX
      }
    }
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%input as %arg3: !Input_CMX) outputs(%output_cmx as %arg4: !Output_CMX) on tile 0 -> !Output_CMX {
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !Input_CMX, !Output_CMX
      }
    }
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%input as %arg3: !Input_CMX) outputs(%output_cmx as %arg4: !Output_CMX) on tile 0 -> !Output_CMX {
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !Input_CMX, !Output_CMX
      }
    }
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%input as %arg3: !Input_CMX) outputs(%output_cmx as %arg4: !Output_CMX) on tile 0 -> !Output_CMX {
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !Input_CMX, !Output_CMX
      }
    }
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%input as %arg3: !Input_CMX) outputs(%output_cmx as %arg4: !Output_CMX) on tile 0 -> !Output_CMX {
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !Input_CMX, !Output_CMX
      }
    }
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%input as %arg3: !Input_CMX) outputs(%output_cmx as %arg4: !Output_CMX) on tile 0 -> !Output_CMX {
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !Input_CMX, !Output_CMX
      }
    }
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%input as %arg3: !Input_CMX) outputs(%output_cmx as %arg4: !Output_CMX) on tile 0 -> !Output_CMX {
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !Input_CMX, !Output_CMX
      }
    }
    VPURT.Task waits(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA inputs(%output_cmx : !Output_CMX) outputs(%output_ddr : !Output_DDR) -> !Output_DDR
    }
    return %output_ddr : !Output_DDR
  }
    // CHECK:       [[BARRIER_NCE:%.+]] = VPURT.DeclareVirtualBarrier

    // CHECK:       [[PROF_CMX_A:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <[[SUBBUF0_CMX_OFF:1472000]]> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       [[BARRIER_FLUSH_0:%.+]] = VPURT.DeclareVirtualBarrier
    // CHECK:       VPURT.Task updates([[BARRIER_NCE]], [[BARRIER_FLUSH_0]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.SW.Kernel
    // CHECK-SAME:      #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 0 : i64, dataIndex = 0 : i64, tileId = 0 : i64, clusterId = 0 : i64>
    // CHECK-SAME:      profiling_data([[PROF_CMX_A]]

    // CHECK:       [[PROF_CMX_B:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472032> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       VPURT.Task updates([[BARRIER_NCE]], [[BARRIER_FLUSH_0]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.SW.Kernel
    // CHECK-SAME:      #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 0 : i64, dataIndex = 1 : i64, tileId = 0 : i64, clusterId = 0 : i64>
    // CHECK-SAME:      profiling_data([[PROF_CMX_B]]

    // CHECK:       [[PROF_CMX_C:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472064> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       VPURT.Task updates([[BARRIER_NCE]], [[BARRIER_FLUSH_0]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.SW.Kernel
    // CHECK-SAME:      #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 0 : i64, dataIndex = 2 : i64, tileId = 0 : i64, clusterId = 0 : i64>
    // CHECK-SAME:      profiling_data([[PROF_CMX_C]]

    // CHECK:       [[PROF_CMX_D:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472096> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       VPURT.Task updates([[BARRIER_NCE]], [[BARRIER_FLUSH_0]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.SW.Kernel
    // CHECK-SAME:      #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 0 : i64, dataIndex = 3 : i64, tileId = 0 : i64, clusterId = 0 : i64>
    // CHECK-SAME:      profiling_data([[PROF_CMX_D]]

    // CHECK:       [[FLUSH_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <[[SUBBUF0_CMX_OFF]]> -> memref<32xui32, [@CMX_NN, 0]>
    // CHECK:       [[FLUSH_DDR_0:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<32xui32, @DDR>
    // CHECK:       [[BARRIER_CHAIN_0:%.+]] = VPURT.DeclareVirtualBarrier
    // CHECK:       VPURT.Task waits([[BARRIER_FLUSH_0]] : !VPURT.Barrier) updates([[BARRIER_CHAIN_0]] : !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.NNDMA <{port = 0 : i64, profiling_buffer_mgmt}>
    // CHECK-SAME:      inputs([[FLUSH_CMX_0]]
    // CHECK-SAME:      outputs([[FLUSH_DDR_0]]

    // CHECK:       [[PROF_CMX_E:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <[[SUBBUF1_CMX_OFF:1472128]]> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       [[BARRIER_FLUSH_1:%.+]] = VPURT.DeclareVirtualBarrier
    // CHECK:       VPURT.Task updates([[BARRIER_NCE]], [[BARRIER_FLUSH_1]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.SW.Kernel
    // CHECK-SAME:      #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 4 : i64, clusterSize = 0 : i64, dataIndex = 0 : i64, tileId = 0 : i64, clusterId = 0 : i64>
    // CHECK-SAME:      profiling_data([[PROF_CMX_E]]

    // CHECK:       [[PROF_CMX_F:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472160> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       VPURT.Task updates([[BARRIER_NCE]], [[BARRIER_FLUSH_1]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.SW.Kernel
    // CHECK-SAME:      #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 4 : i64, clusterSize = 0 : i64, dataIndex = 1 : i64, tileId = 0 : i64, clusterId = 0 : i64>
    // CHECK-SAME:      profiling_data([[PROF_CMX_F]]

    // CHECK:       [[PROF_CMX_G:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472192> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       VPURT.Task updates([[BARRIER_NCE]], [[BARRIER_FLUSH_1]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.SW.Kernel
    // CHECK-SAME:      #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 4 : i64, clusterSize = 0 : i64, dataIndex = 2 : i64, tileId = 0 : i64, clusterId = 0 : i64>
    // CHECK-SAME:      profiling_data([[PROF_CMX_G]]

    // CHECK:       [[PROF_CMX_H:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472224> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       VPURT.Task updates([[BARRIER_NCE]], [[BARRIER_FLUSH_1]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.SW.Kernel
    // CHECK-SAME:      #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 4 : i64, clusterSize = 0 : i64, dataIndex = 3 : i64, tileId = 0 : i64, clusterId = 0 : i64>
    // CHECK-SAME:      profiling_data([[PROF_CMX_H]]

    // CHECK:       [[FLUSH_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <[[SUBBUF1_CMX_OFF]]> -> memref<32xui32, [@CMX_NN, 0]>
    // CHECK:       [[FLUSH_DDR_1:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <128> -> memref<32xui32, @DDR>
    // CHECK:       VPURT.Task waits([[BARRIER_FLUSH_1]] : !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.NNDMA <{port = 0 : i64, profiling_buffer_mgmt}>
    // CHECK-SAME:      inputs([[FLUSH_CMX_1]]
    // CHECK-SAME:      outputs([[FLUSH_DDR_1]]

    // CHECK:       [[PROF_CMX_I:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <[[SUBBUF0_CMX_OFF]]> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       [[BARRIER_FLUSH_2:%.+]] = VPURT.DeclareVirtualBarrier
    // CHECK:       VPURT.Task waits([[BARRIER_CHAIN_0]] : !VPURT.Barrier) updates([[BARRIER_NCE]], [[BARRIER_FLUSH_2]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.SW.Kernel
    // CHECK-SAME:      #VPUIP.SwProfilingMetadataAttr<bufferId = 2 : i64, bufferOffset = 8 : i64, clusterSize = 0 : i64, dataIndex = 0 : i64, tileId = 0 : i64, clusterId = 0 : i64>
    // CHECK-SAME:      profiling_data([[PROF_CMX_I]]

    // CHECK:       [[PROF_CMX_J:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472032> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       VPURT.Task waits([[BARRIER_CHAIN_0]] : !VPURT.Barrier) updates([[BARRIER_NCE]], [[BARRIER_FLUSH_2]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.SW.Kernel
    // CHECK-SAME:      #VPUIP.SwProfilingMetadataAttr<bufferId = 2 : i64, bufferOffset = 8 : i64, clusterSize = 0 : i64, dataIndex = 1 : i64, tileId = 0 : i64, clusterId = 0 : i64>
    // CHECK-SAME:      profiling_data([[PROF_CMX_J]]

    // CHECK:       [[PROF_CMX_K:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472064> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       VPURT.Task waits([[BARRIER_CHAIN_0]] : !VPURT.Barrier) updates([[BARRIER_NCE]], [[BARRIER_FLUSH_2]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.SW.Kernel
    // CHECK-SAME:      #VPUIP.SwProfilingMetadataAttr<bufferId = 2 : i64, bufferOffset = 8 : i64, clusterSize = 0 : i64, dataIndex = 2 : i64, tileId = 0 : i64, clusterId = 0 : i64>
    // CHECK-SAME:      profiling_data([[PROF_CMX_K]]

    // CHECK:       [[PROF_CMX_L:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472096> -> memref<8xui32, [@CMX_NN, 0]>
    // CHECK:       VPURT.Task waits([[BARRIER_CHAIN_0]] : !VPURT.Barrier) updates([[BARRIER_NCE]], [[BARRIER_FLUSH_2]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.SW.Kernel
    // CHECK-SAME:      #VPUIP.SwProfilingMetadataAttr<bufferId = 2 : i64, bufferOffset = 8 : i64, clusterSize = 0 : i64, dataIndex = 3 : i64, tileId = 0 : i64, clusterId = 0 : i64>
    // CHECK-SAME:      profiling_data([[PROF_CMX_L]]

    // CHECK:       [[FLUSH_CMX_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <[[SUBBUF0_CMX_OFF]]> -> memref<32xui32, [@CMX_NN, 0]>
    // CHECK:       [[FLUSH_DDR_2:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <256> -> memref<32xui32, @DDR>
    // CHECK:       VPURT.Task waits([[BARRIER_FLUSH_2]] : !VPURT.Barrier)
    // CHECK-NEXT:    VPUIP.NNDMA <{port = 0 : i64, profiling_buffer_mgmt}>
    // CHECK-SAME:      inputs([[FLUSH_CMX_2]]
    // CHECK-SAME:      outputs([[FLUSH_DDR_2]]

    // CHECK:       return %arg3, %arg4 : memref<1x48x60x60xf16, #NHWC, @DDR>, memref<96xui32>

}
