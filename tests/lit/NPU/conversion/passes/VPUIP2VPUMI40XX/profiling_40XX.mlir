//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values" --convert-VPUIP-to-VPUMI40XX --setup-profiling-VPUMI40XX="dma-profiling=true" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {
  config.Resources 1 of @global {
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        config.MemoryResource 4096 bytes of @DDR offset 0
      }
    }
  }

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x16x16xf16>
    DataInfo "output_1" : tensor<1x16x16x16xf16>
  } profilingOutputsInfo : {
    DataInfo "profilingOutput" {
      VPUIP.ProfilingSection type 6 : 320 bytes from 0
      VPUIP.ProfilingSection type 5 : 64 bytes from 320
    } : tensor<96xui32>
  }
  func.func private @main(%arg0: memref<1x16x16x16xf16, #NHWC, @DDR>, %arg1: memref<1x16x16x16xf16, #NHWC, @DDR>, %arg2: memref<1x16x16x16xf16, #NHWC, @DDR>, %arg3: memref<96xui32>) -> (memref<1x16x16x16xf16, #NHWC, @DDR>, memref<1x16x16x16xf16, #NHWC, @DDR>, memref<96xui32>) {
    // CHECK: %[[DMA_HWP_BASE:.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1024xui32, @DDR>
    // CHECK: %[[WORKPOINT_CFG:.*]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <320> -> memref<16xui32>

    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: %[[PROF_BUF:.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>

    %2 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    // CHECK: %[[BAR0:.*]] = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <0, -1> -> !VPURegMapped.Index<0:0:0>

    VPURT.Task updates(%2 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    }
    // CHECK: %[[NNDMA0_DDR_0:.*]] = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%[[PROF_BUF]] : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) updates(%[[BAR0]] : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>){{.*}}-> !VPURegMapped.Index<0:0:0>

    %3 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    // CHECK: %[[BAR1:.*]] = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <1, -1> -> !VPURegMapped.Index<0:0:1>

    VPURT.Task waits(%2 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    }
    // CHECK: %[[NNDMA0_DDR_1:.*]] = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%[[PROF_BUF]] : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[NNDMA0_DDR_0]] : !VPURegMapped.Index<0:0:0>) waits(%[[BAR0]] : !VPURegMapped.Index<0:0:0>) updates(%[[BAR1]] : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>){{.*}}-> !VPURegMapped.Index<0:0:1>

    VPURT.Task waits(%3 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x16x16x16xf16, #NHWC, @DDR>) -> memref<1x16x16x16xf16, #NHWC, @DDR>
    }
    // CHECK: %[[NNDMA0_CMX_0:.*]] = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%[[PROF_BUF]] : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x16x16x16xf16, #NHWC, @DDR>) waits(%[[BAR1]] : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>){{.*}}-> !VPURegMapped.Index<0:1:0>

    // CHECK: VPUMI40XX.ProfilingMetadata {metadata = dense<"{{.+}}"> : vector<180xui8>} -> !VPURegMapped.Index<0:0:0>

    // CHECK: %[[MI:.*]] = VPUMI40XX.MappedInference dmas((%[[NNDMA0_DDR_0]], %[[NNDMA0_CMX_0]]) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) barriers(%[[BAR0]] : !VPURegMapped.Index<0:0:0>)
    // CHECK: dmaHwpBase(%[[DMA_HWP_BASE]] : memref<1024xui32, @DDR>) hwpWorkpointCfg(%[[WORKPOINT_CFG]] : memref<16xui32>)
    // CHECK-SAME{LITERAL}: dmaCount([[2, 1], [0, 0]])
    // CHECK-SAME{LITERAL}: invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) actKernelInvocationsCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) mediaCount(0) barrierCount(2)  -> !VPURegMapped.Index<0:0:0>

    return %arg1, %arg2, %arg3 : memref<1x16x16x16xf16, #NHWC, @DDR>, memref<1x16x16x16xf16, #NHWC, @DDR>, memref<96xui32>
  }
}
