//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --sw-kernel-prefetching-reserve-mem %s | FileCheck %s
// REQUIRES: arch-NPU40XX
module @SimpleGraph {
  IE.TileResource 1 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
  }

  module @VPU.SW {
    func.func private @builtin_Gelu(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_gelu.cpp", VPU.kernel_entry = "activation_gelu"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  }

  func.func @main(%arg0: memref<1x16x4x4xf16>, %arg1: memref<1x16x4x4xf16>) -> memref<1x16x4x4xf16> {
    %0 = memref.alloc() : memref<1x16x4x4xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4x4xf16>) outputs(%0 : memref<1x16x4x4xf16, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x16x4x4xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gelu inputs(%1 as %arg2: memref<1x16x4x4xf16, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x16x4x4xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x4x4xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x16x4x4xf16, [@CMX_NN, 0]>, memref<1x16x4x4xf16, [@CMX_NN, 0]>
    }
    %3 = VPUIP.Copy inputs(%results : memref<1x16x4x4xf16, [@CMX_NN, 0]>) outputs(%arg1: memref<1x16x4x4xf16>) -> memref<1x16x4x4xf16>

    return %arg1: memref<1x16x4x4xf16>
  }

    // reserve dummy memory at the end of CMX

    // CHECK:     IE.TileResource
    // CHECK:       ReservedMemory
    // CHECK:         SWKernelPrefetchingReservedMemory
    // CHECK:           IE.MemoryResource 1024 bytes of @CMX_NN offset 1473536
}

// -----

module @SimpleGraphWithReservedMem {
  module @VPU.SW {
    func.func private @builtin_Gelu(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_gelu.cpp", VPU.kernel_entry = "activation_gelu"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

  IE.TileResource 1 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    builtin.module @ReservedMemory {
        module @CustomReservedMemory {
            IE.MemoryResource 512 bytes of @CMX_NN
        }
    }
  }

  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  }

  func.func @main(%arg0: memref<1x16x4x4xf16>, %arg1: memref<1x16x4x4xf16>) -> memref<1x16x4x4xf16> {
    %0 = memref.alloc() : memref<1x16x4x4xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4x4xf16>) outputs(%0 : memref<1x16x4x4xf16, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x16x4x4xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gelu inputs(%1 as %arg2: memref<1x16x4x4xf16, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x16x4x4xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x4x4xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x16x4x4xf16, [@CMX_NN, 0]>, memref<1x16x4x4xf16, [@CMX_NN, 0]>
    }
    %3 = VPUIP.Copy inputs(%results : memref<1x16x4x4xf16, [@CMX_NN, 0]>) outputs(%arg1: memref<1x16x4x4xf16>) -> memref<1x16x4x4xf16>

    return %arg1: memref<1x16x4x4xf16>
  }

    // enlarge the original reserved memory and put it at the end of CMX

    // CHECK:     IE.TileResource
    // CHECK:       ReservedMemory
    // CHECK:         CustomReservedMemory
    // CHECK:           IE.MemoryResource 1024 bytes of @CMX_NN offset 1473536
}

// -----

module @SimpleGraphWithReservedMemHasEnoughSize {
  module @VPU.SW {
    func.func private @builtin_Gelu(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_gelu.cpp", VPU.kernel_entry = "activation_gelu"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

  IE.TileResource 1 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    builtin.module @ReservedMemory {
        module @CustomReservedMemory {
            IE.MemoryResource 1024 bytes of @CMX_NN
        }
    }
  }

  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  }

  func.func @main(%arg0: memref<1x16x4x4xf16>, %arg1: memref<1x16x4x4xf16>) -> memref<1x16x4x4xf16> {
    %0 = memref.alloc() : memref<1x16x4x4xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4x4xf16>) outputs(%0 : memref<1x16x4x4xf16, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x16x4x4xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gelu inputs(%1 as %arg2: memref<1x16x4x4xf16, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x16x4x4xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x4x4xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x16x4x4xf16, [@CMX_NN, 0]>, memref<1x16x4x4xf16, [@CMX_NN, 0]>
    }
    %3 = VPUIP.Copy inputs(%results : memref<1x16x4x4xf16, [@CMX_NN, 0]>) outputs(%arg1: memref<1x16x4x4xf16>) -> memref<1x16x4x4xf16>

    return %arg1: memref<1x16x4x4xf16>
  }

    // no need to change the reserved memory size, just put it at the end of CMX

    // CHECK:     IE.TileResource
    // CHECK:       ReservedMemory
    // CHECK:         CustomReservedMemory
    // CHECK:           IE.MemoryResource 1024 bytes of @CMX_NN offset 1473536
}

// -----

module @SimpleGraphWith2ReservedMem {
  module @VPU.SW {
    func.func private @builtin_Gelu(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_gelu.cpp", VPU.kernel_entry = "activation_gelu"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

  IE.TileResource 1 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    builtin.module @ReservedMemory {
        module @CustomReservedMemory1 {
            IE.MemoryResource 512 bytes of @CMX_NN
        }

        module @CustomReservedMemory2 {
            IE.MemoryResource 64 bytes of @CMX_NN
        }
    }
  }

  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  }

  func.func @main(%arg0: memref<1x16x4x4xf16>, %arg1: memref<1x16x4x4xf16>) -> memref<1x16x4x4xf16> {
    %0 = memref.alloc() : memref<1x16x4x4xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4x4xf16>) outputs(%0 : memref<1x16x4x4xf16, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x16x4x4xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gelu inputs(%1 as %arg2: memref<1x16x4x4xf16, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x16x4x4xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x4x4xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x16x4x4xf16, [@CMX_NN, 0]>, memref<1x16x4x4xf16, [@CMX_NN, 0]>
    }
    %3 = VPUIP.Copy inputs(%results : memref<1x16x4x4xf16, [@CMX_NN, 0]>) outputs(%arg1: memref<1x16x4x4xf16>) -> memref<1x16x4x4xf16>

    return %arg1: memref<1x16x4x4xf16>
  }

    // enlarge reserved memory size, and put both of them at the end of CMX

    // CHECK:     IE.TileResource
    // CHECK:       ReservedMemory
    // CHECK:         CustomReservedMemory1
    // CHECK:           IE.MemoryResource 512 bytes of @CMX_NN offset 1474048
    // CHECK:         CustomReservedMemory2
    // CHECK:           IE.MemoryResource 512 bytes of @CMX_NN offset 1473536
}

// -----

module @SimpleGraphWith2ReservedMemHaveEnoughTotalSize {
  module @VPU.SW {
    func.func private @builtin_Gelu(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_gelu.cpp", VPU.kernel_entry = "activation_gelu"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

  IE.TileResource 1 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    builtin.module @ReservedMemory {
        module @CustomReservedMemory1 {
            IE.MemoryResource 128 bytes of @CMX_NN
        }

        module @CustomReservedMemory2 {
            IE.MemoryResource 896 bytes of @CMX_NN
        }
    }
  }

  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  }

  func.func @main(%arg0: memref<1x16x4x4xf16>, %arg1: memref<1x16x4x4xf16>) -> memref<1x16x4x4xf16> {
    %0 = memref.alloc() : memref<1x16x4x4xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4x4xf16>) outputs(%0 : memref<1x16x4x4xf16, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x16x4x4xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gelu inputs(%1 as %arg2: memref<1x16x4x4xf16, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x16x4x4xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x4x4xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x16x4x4xf16, [@CMX_NN, 0]>, memref<1x16x4x4xf16, [@CMX_NN, 0]>
    }
    %3 = VPUIP.Copy inputs(%results : memref<1x16x4x4xf16, [@CMX_NN, 0]>) outputs(%arg1: memref<1x16x4x4xf16>) -> memref<1x16x4x4xf16>

    return %arg1: memref<1x16x4x4xf16>
  }

    // not need to enlarge the reserved memory size, just put both of them at the end of CMX

    // CHECK:     IE.TileResource
    // CHECK:       ReservedMemory
    // CHECK:         CustomReservedMemory1
    // CHECK:           IE.MemoryResource 128 bytes of @CMX_NN offset 1474432
    // CHECK:         CustomReservedMemory2
    // CHECK:           IE.MemoryResource 896 bytes of @CMX_NN offset 1473536
}

// -----

module @SimpleGraphNoSWKernel {

  IE.TileResource 1 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
  }

  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  }
  func.func @main(%arg0: memref<1x16x4x4xf16>, %arg1: memref<1x16x4x4xf16>) -> memref<1x16x4x4xf16> {
    return %arg1 : memref<1x16x4x4xf16>
  }
    // not change if no SW Kernel

    // CHECK-NOT:     ReservedMemory
}
