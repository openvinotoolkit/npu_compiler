//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --reserve-profiling-mem="dma-profiling=false" %s | FileCheck %s
// REQUIRES: platform-NPU3720

// For NPU37XX, dma-profiling=false maps to DMAProfilingMode::DISABLED.
// The DMA CMX block is still reserved unconditionally (arch == NPU37XX guard in the pass),
// so the CMX layout is identical to the dma-profiling=true case.
// No DDR block must be added (DDR reservation is only for DYNAMIC_HWP mode).

// CHECK-LABEL: module @SimpleGraph
module @SimpleGraph {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  }
  func.func @main(%arg0: memref<1x16x4x4xf16>, %arg1: memref<1x16x4x4xf16>) -> memref<1x16x4x4xf16> {
    return %arg1 : memref<1x16x4x4xf16>
  }
    // CHECK:       ActShaveProfilingReservedMemory
    // CHECK-NEXT:    config.MemoryResource 512 bytes of @CMX_NN offset 1980416

    // CHECK:       DpuProfilingReservedMemory
    // CHECK-NEXT:    config.MemoryResource 1024 bytes of @CMX_NN offset 1980928

    // CHECK:       DmaProfilingReservedMemory
    // CHECK-NEXT:    config.MemoryResource 512 bytes of @CMX_NN offset 1981952

}
