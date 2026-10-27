//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform% allow-custom-values=true" --reserve-profiling-mem="dma-profiling=false" %s | FileCheck %s
// REQUIRES: platform-NPU5010

// For specified platforms, dma-profiling=false maps to DMAProfilingMode::DISABLED.
// No DMA CMX or DDR reservation is added; only ActShave and DPU CMX blocks are present.

module @SimpleGraph {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  }
  func.func @main(%arg0: memref<1x16x4x4xf16>, %arg1: memref<1x16x4x4xf16>) -> memref<1x16x4x4xf16> {
    return %arg1 : memref<1x16x4x4xf16>
  }
    // CHECK:     config.Resources
    // CHECK:         ActShaveProfilingReservedMemory
    // CHECK-NEXT:         config.MemoryResource 512 bytes of @CMX_NN offset 1570304

    // CHECK:         DpuProfilingReservedMemory
    // CHECK-NEXT:         config.MemoryResource 1024 bytes of @CMX_NN offset 1570816

    // CHECK-NOT: DmaProfilingReservedMemory
}
