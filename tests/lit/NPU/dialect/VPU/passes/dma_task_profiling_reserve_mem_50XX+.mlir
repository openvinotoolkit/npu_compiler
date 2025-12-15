//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --dma-task-profiling-reserve-mem="dma-profiling=true" %s | FileCheck %s
// REQUIRES: arch-NPU50XX

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

    // CHECK:         ReservedMemory
    // CHECK-NEXT:         DmaProfilingReservedMemory
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

  // CHECK-DAG:   {{  }}config.Resources
  // CHECK-DAG:   {{    }}@ReservedMemory
  // CHECK-DAG:   {{      }}@DDRCustomReservedMemory
  // CHECK-DAG:   {{        }}config.MemoryResource 40 bytes of @DDR offset 0
  // CHECK-DAG:   {{      }}@DmaProfilingReservedMemory
  // CHECK-DAG:   {{        }}config.MemoryResource 4096 bytes of @DDR offset 64
}
