//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --cmx-metadata-reserve-mem %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010 || platform-NPU5020

module @AddReservedMemory {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  }
  func.func @main(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16> {
    return %arg0: tensor<1x16x4x4xf16>
  }

    // CHECK:   config.Resources
    // CHECK:     @ReservedMemory
    // CHECK-NEXT:  @CMXMetadataReservedMemory
    // CHECK-NEXT:    config.MemoryResource 82944 bytes of @CMX_NN offset 0
}

// -----

module @AddSecondReservedMemory {
  config.Resources 3 of @NCE at 2.100000e+03 MHz {
    builtin.module @ReservedMemory {
      // Note: Using CMXStackFramesReservedMemory module name just to test CMXMetadataReservedMemory alignment requirement.
      module @CMXStackFramesReservedMemory {
        config.MemoryResource 10 bytes of @CMX_NN offset 0
      }
    }
  }

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  }
  func.func @main(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16> {
    return %arg0: tensor<1x16x4x4xf16>
  }

    // CHECK:     config.Resources
    // CHECK:       @ReservedMemory
    // CHECK-NEXT:    @CMXMetadataReservedMemory
    // CHECK-NEXT:      config.MemoryResource 82944 bytes of @CMX_NN offset 32
    // CHECK:         @CMXStackFramesReservedMemory
    // CHECK-NEXT:      config.MemoryResource 10 bytes of @CMX_NN offset 0
}
