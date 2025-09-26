//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --sw-kernel-data-prefetch-reserve-mem %s | FileCheck %s
// REQUIRES: arch-NPU37XX

module @SimpleGraph {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  }

  func.func @main(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16> {
    %results = VPU.Gelu(%arg0) : tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>
    return %results: tensor<1x16x4x4xf16>
  }

    // reserve dummy memory at the end of CMX

    // CHECK:     config.Resources
    // CHECK:       ReservedMemory
    // CHECK:         SWKernelPrefetchingReservedMemory
    // CHECK:           config.MemoryResource 256 bytes of @CMX_NN offset 1982208
}
