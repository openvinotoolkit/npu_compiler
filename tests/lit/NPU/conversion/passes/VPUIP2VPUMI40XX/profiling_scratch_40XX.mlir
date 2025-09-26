//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch% allow-custom-values=true" --convert-VPUIP-to-VPUMI40XX --setup-profiling-VPUMI40XX="dma-profiling=false" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @inout {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x1x1x1xf16>
  } outputsInfo : {
    DataInfo "data" : tensor<1x1x1x1xf16>
  }
  config.Resources 6 of @NCE at 1.700000e+03 MHz {
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        config.MemoryResource 512 bytes of @CMX_NN offset 0
      }
    }
  }
  func.func @main(%arg0: memref<1x1x1x1xf16, @DDR>, %arg1: memref<1x1x1x1xf16, @DDR>) -> memref<1x1x1x1xf16, @DDR> {
    %0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %1 = VPURT.ConfigureBarrier<1> <{isFinalBarrier}> -> !VPURT.Barrier
    %2 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x1x1xf16, @DDR>
    %3 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x1x1x1xf16, @DDR>
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %6 = VPUIP.NNDMA {port = 0 : i64} inputs(%2 : memref<1x1x1x1xf16, @DDR>) outputs(%3 : memref<1x1x1x1xf16, @DDR>) -> memref<1x1x1x1xf16, @DDR>
    }

    // CHECK: %[[SCRATCH:.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<16xui32, [@CMX_NN, 0]>
    // CHECK: VPUMI40XX.MappedInference
    // CHECK-SAME: dmaHwpBase(%[[SCRATCH]] : memref<16xui32, [@CMX_NN, 0]>)
    return %arg1 : memref<1x1x1x1xf16, @DDR>
  }
}
