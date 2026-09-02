//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --serialize-kernels-to-bytecode %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

module @BasicKernelSerialization {
  HostExec.Binary @SubModule {
    HostExec.BinaryData @serialized_compute <object = "\00\01\02\03">
    func.func nested @compute(memref<1x3x8x8xf16>, memref<1x3x8x8xf16>) -> memref<1x3x8x8xf16>
  }
  func.func @main(%arg0: memref<1x3x8x8xf16>, %arg1: memref<1x3x8x8xf16>) -> memref<1x3x8x8xf16> {
    %0 = Core.NestedCall @SubModule::@compute(%arg0, %arg1) : (memref<1x3x8x8xf16>, memref<1x3x8x8xf16>) -> memref<1x3x8x8xf16>
    return %arg1 : memref<1x3x8x8xf16>
  }
}

// CHECK-LABEL: @BasicKernelSerialization
// CHECK:   module @SubModule {
// CHECK:     func.func nested @compute(memref<1x3x8x8xf16>, memref<1x3x8x8xf16>) -> memref<1x3x8x8xf16>
// CHECK:   }
// CHECK:   bytecode.kernel_section @kernel_section {
// CHECK:     bytecode.kernel @compute "\00\01\02\03"
// CHECK:   }

// -----

module @MultipleKernels {
  HostExec.Binary @Module1 {
    HostExec.BinaryData @serialized_kernel_a <object = "\00\01">
    func.func nested @kernel_a(memref<1x16xf16>) -> memref<1x16xf16>
  }
  HostExec.Binary @Module2 {
    HostExec.BinaryData @serialized_kernel_b <object = "\02\03">
    func.func nested @kernel_b(memref<1x32xf16>) -> memref<1x32xf16>
  }
  func.func @main(%arg0: memref<1x16xf16>, %arg1: memref<1x32xf16>) -> memref<1x32xf16> {
    %0 = Core.NestedCall @Module1::@kernel_a(%arg0) : (memref<1x16xf16>) -> memref<1x16xf16>
    %1 = Core.NestedCall @Module2::@kernel_b(%arg1) : (memref<1x32xf16>) -> memref<1x32xf16>
    return %arg1 : memref<1x32xf16>
  }
}

// CHECK-LABEL: @MultipleKernels
// CHECK:      module @Module1 {
// CHECK:        func.func nested @kernel_a(memref<1x16xf16>) -> memref<1x16xf16>
// CHECK:      }
// CHECK:      module @Module2 {
// CHECK:        func.func nested @kernel_b(memref<1x32xf16>) -> memref<1x32xf16>
// CHECK:      }
// CHECK:      bytecode.kernel_section @kernel_section {
// CHECK:        bytecode.kernel @kernel_a "\00\01"
// CHECK:        bytecode.kernel @kernel_b "\02\03"
// CHECK:      }
