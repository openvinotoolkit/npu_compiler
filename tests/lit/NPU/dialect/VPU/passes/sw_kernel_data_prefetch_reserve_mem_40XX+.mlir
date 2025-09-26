//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --sw-kernel-data-prefetch-reserve-mem %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @SimpleGraph {
  config.Resources 1 of @NCE at 1.300000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
  }

  module @VPU.SW {
    func.func private @builtin_Gelu(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_gelu.cpp", VPU.kernel_entry = "activation_gelu"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  }

  func.func @main(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16> {
    %results = VPU.Gelu(%arg0) : tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>
    return %results: tensor<1x16x4x4xf16>
  }

    // no reserved CMX by default

    // CHECK:     config.Resources
    // CHECK-NOT:       ReservedMemory
}
