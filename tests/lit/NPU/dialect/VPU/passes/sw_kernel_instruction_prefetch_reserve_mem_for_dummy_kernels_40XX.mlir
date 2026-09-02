//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --sw-kernel-instruction-prefetch-reserve-mem-for-dummy-kernels %s | FileCheck %s
// REQUIRES: platform-NPU4000

module @SimpleGraphAddFirstResMem {
  config.Resources 1 of @NCE at 1.300000e+03 MHz

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  }
  func.func @main(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16> {
    %results = VPU.Gelu(%arg0) : tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>
    return %results: tensor<1x16x4x4xf16>
  }

    // CHECK:     config.Resources
    // CHECK:         ReservedMemory
    // CHECK-NEXT:         DummySWKernelsForInstructionPrefetchReservedMemory
    // CHECK-NEXT:         config.MemoryResource 16 bytes of @CMX_NN offset 1571824
}

// -----

module @SimpleGraphAddSecondResMem {
  config.Resources 1 of @NCE at 1.300000e+03 MHz {
    builtin.module @ReservedMemory {
        module @CustomReservedMemory {
            config.MemoryResource 512 bytes of @CMX_NN offset 1571328
        }
    }
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

    // CHECK:     config.Resources
    // CHECK:         ReservedMemory
    // CHECK-NEXT:         DummySWKernelsForInstructionPrefetchReservedMemory
    // CHECK-NEXT:         config.MemoryResource 16 bytes of @CMX_NN offset 1571312
    // CHECK:              CustomReservedMemory
    // CHECK-NEXT:         config.MemoryResource 512 bytes of @CMX_NN offset 1571328
}

// -----

module @SimpleGraphNotAddResMem {
  config.Resources 1 of @NCE at 1.300000e+03 MHz

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  }
  func.func @main(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16> {
    return %arg0: tensor<1x16x4x4xf16>
  }

    // CHECK:     config.Resources
    // CHECK-NOT:         ReservedMemory
    // CHECK-NOT:         DummySWKernelsForInstructionPrefetchReservedMemory
}
