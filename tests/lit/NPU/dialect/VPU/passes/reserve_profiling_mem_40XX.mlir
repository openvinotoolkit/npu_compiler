//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --reserve-profiling-mem="dma-profiling=true" %s | FileCheck %s
// REQUIRES: platform-NPU4000

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

// CHECK-LABEL: module @SimpleGraphWithReservedMemory
module @SimpleGraphWithReservedMemory {
  config.Resources 1 of @global {
    builtin.module @ReservedMemory {
      module @DDRCustomReservedMemory {
        config.MemoryResource 40 bytes of @DDR offset 0
      }
    }
  }

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

    // CHECK:     @ReservedMemory
    // CHECK-DAG:   @DmaProfilingReservedMemory
    // CHECK-DAG:     config.MemoryResource 4096 bytes of @DDR offset 64
    // CHECK-DAG:   @DDRCustomReservedMemory
    // CHECK-DAG:       config.MemoryResource 40 bytes of @DDR offset 0

    // CHECK:   config.Resources
    // CHECK:     @ReservedMemory
    // CHECK:       @ActShaveProfilingReservedMemory
    // CHECK-NEXT:    config.MemoryResource 512 bytes of @CMX_NN offset 1569728
    // CHECK:       @DpuProfilingReservedMemory
    // CHECK-NEXT:    config.MemoryResource 1024 bytes of @CMX_NN offset 1570240
    // CHECK:       @DmaProfilingReservedMemory
    // CHECK-NEXT:    config.MemoryResource 512 bytes of @CMX_NN offset 1571264
    // CHECK:       @CMXCustomReservedMemory
    // CHECK-NEXT:    config.MemoryResource 40 bytes of @CMX_NN offset 1571800

}
