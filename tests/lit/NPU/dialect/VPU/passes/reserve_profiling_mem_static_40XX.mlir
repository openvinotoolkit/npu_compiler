//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --reserve-profiling-mem="dma-profiling=static" %s | FileCheck %s
// REQUIRES: platform-NPU4000

module @SimpleGraph {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  }
  func.func @main(%arg0: memref<1x16x4x4xf16>, %arg1: memref<1x16x4x4xf16>) -> memref<1x16x4x4xf16> {
    return %arg1 : memref<1x16x4x4xf16>
  }
    // CHECK:   config.Resources
    // CHECK:     @ReservedMemory
    // CHECK:       @ActShaveProfilingReservedMemory
    // CHECK-NEXT:    config.MemoryResource 512 bytes of @CMX_NN offset 1569792
    // CHECK:       @DpuProfilingReservedMemory
    // CHECK-NEXT:    config.MemoryResource 1024 bytes of @CMX_NN offset 1570304
    // CHECK:       @DmaProfilingReservedMemory
    // CHECK-NEXT:    config.MemoryResource 512 bytes of @CMX_NN offset 1571328
}

// -----

module @SimpleGraphWithReservedMemory {
  config.Resources 2 of @NCE at 1.300000e+03 MHz {
    builtin.module @ReservedMemory {
      module @CMXCustomReservedMemory {
        config.MemoryResource 40 bytes of @CMX_NN offset 1473024
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
  // CHECK:     @ActShaveProfilingReservedMemory
  // CHECK-NEXT:       config.MemoryResource 512 bytes of @CMX_NN offset 1470976

  // CHECK:     @DpuProfilingReservedMemory
  // CHECK-NEXT:       config.MemoryResource 1024 bytes of @CMX_NN offset 1471488

  // CHECK:     @DmaProfilingReservedMemory
  // CHECK-NEXT:       config.MemoryResource 512 bytes of @CMX_NN offset 1472512

  // CHECK:     @CMXCustomReservedMemory
  // CHECK-NEXT:       config.MemoryResource 40 bytes of @CMX_NN offset 1473024
}
