//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform% allow-custom-values=true" --reserve-profiling-mem="dma-profiling=true" %s | FileCheck %s
// REQUIRES: platform-NPU3720

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
