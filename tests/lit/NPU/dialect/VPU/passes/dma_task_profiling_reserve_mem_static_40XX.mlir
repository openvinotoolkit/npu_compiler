//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --dma-task-profiling-reserve-mem="dma-profiling=static" %s | FileCheck %s
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
    // CHECK:     config.Resources
    // CHECK:         ReservedMemory
    // CHECK-NEXT:         DmaProfilingReservedMemory
    // CHECK-NEXT:         config.MemoryResource 512 bytes of @CMX_NN offset 1571328
}

// -----

module @SimpleGraphWithReservedMemory {
  config.Resources 2 of @NCE at 1.300000e+03 MHz {
    builtin.module @ReservedMemory {
      module @CMXCustomReservedMemory {
        config.MemoryResource 40 bytes of @CMX_NN offset 1571800
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
    // CHECK: module @ReservedMemory {
    // CHECK-NEXT:     module @DmaProfilingReservedMemory {
    // CHECK-NEXT:       config.MemoryResource 512 bytes of @CMX_NN offset 1571264
    // CHECK-NEXT:     }
    // CHECK:           module @CMXCustomReservedMemory {
    // CHECK-NEXT:       config.MemoryResource 40 bytes of @CMX_NN offset 1571800
    // CHECK-NEXT:     }
}
