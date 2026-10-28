//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --reserve-profiling-mem="dma-profiling=true" %s | FileCheck %s
// REQUIRES: platform-NPU5010

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

    // CHECK:         ActShaveProfilingReservedMemory
    // CHECK-NEXT:         config.MemoryResource 512 bytes of @CMX_NN offset 1570304

    // CHECK:         DpuProfilingReservedMemory
    // CHECK-NEXT:         config.MemoryResource 1024 bytes of @CMX_NN offset 1570816

    // CHECK:         DmaProfilingReservedMemory
    // CHECK-NEXT:         config.MemoryResource 4096 bytes of @DDR offset 0
}

// -----

// CHECK-LABEL: module @SimpleGraphWithReservedMemory
module @SimpleGraphWithReservedMemory {
  config.Resources 1 of @global {
    builtin.module @ReservedMemory {
      module @DDRCustomReservedMemory {
        config.MemoryResource 40 bytes of @DDR offset 0
      }
    }
  }

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  }
  func.func @main(%arg0: memref<1x16x4x4xf16>, %arg1: memref<1x16x4x4xf16>) -> memref<1x16x4x4xf16> {
    return %arg1 : memref<1x16x4x4xf16>
  }

  // CHECK:       ActShaveProfilingReservedMemory
  // CHECK-NEXT:    config.MemoryResource 512 bytes of @CMX_NN offset 1570304

  // CHECK:       DpuProfilingReservedMemory
  // CHECK-NEXT:    config.MemoryResource 1024 bytes of @CMX_NN offset 1570816

  // CHECK:       DmaProfilingReservedMemory
  // CHECK-NEXT:    config.MemoryResource 4096 bytes of @DDR

  // CHECK:       DDRCustomReservedMemory
  // CHECK-NEXT:    config.MemoryResource 40 bytes of @DDR offset 0
}
