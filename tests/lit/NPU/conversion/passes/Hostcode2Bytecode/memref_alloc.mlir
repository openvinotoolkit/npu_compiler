//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-hostcode-to-bytecode %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

module {

func.func @alloc_static_float() -> () attributes {config.pureHostCompileFunc} {
  %0 = memref.alloc() : memref<2x3xf32>
  return
}

// CHECK:  bytecode.func_section @func_section {
// CHECK:    bytecode.ext.func @alloc_static_float () -> () {
// CHECK-NOT:   memref.alloc
// CHECK:       [[BUF:%.+]] = bytecode.virtual_general_register
// CHECK:       bytecode.ext.buffer.create [[BUF]], memref<2x3xf32>
// CHECK:       bytecode.ret
// CHECK:    }
// CHECK:  }

}

// -----

// Dynamic host scratch buffer: the runtime size is forwarded as a sizes() register operand.

module {

func.func @alloc_dynamic(%n: index) -> () attributes {config.pureHostCompileFunc} {
  %0 = memref.alloc(%n) : memref<?xi8>
  return
}

// CHECK:    bytecode.ext.func @alloc_dynamic (index) -> () {
// CHECK-NOT:   memref.alloc
// CHECK:       [[N:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:       [[BUF:%.+]] = bytecode.virtual_general_register
// CHECK:       bytecode.ext.buffer.create [[BUF]], memref<?xi8> sizes([[N]])
// CHECK:       bytecode.ret
// CHECK:    }

}
