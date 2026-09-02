//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-hostcode-to-bytecode %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// Static source dim with constant index: memref.dim folds to a constant and lowers to an immediate register.

module {

func.func @dim_static_alloc() -> index attributes {config.pureHostCompileFunc} {
  %c0 = arith.constant 0 : index
  %0 = memref.alloc() : memref<4x6xf32>
  %1 = memref.dim %0, %c0 : memref<4x6xf32>
  return %1 : index
}

}

// CHECK:      bytecode.ext.func @dim_static_alloc () -> index {
// CHECK-NOT:    memref.dim
// CHECK-NOT:    bytecode.buffer.get_dim
// CHECK:        [[DIM:%.+]] = bytecode.imm_register 4
// CHECK-NEXT:   bytecode.retv [[DIM]]
// CHECK:      }

// -----

// Dynamic source dim with constant index: memref.dim cannot fold, so it lowers via bytecode.buffer.get_dim.

module {

func.func @dim_dynamic_arg(%arg0: memref<?x6xf32>) -> index attributes {config.pureHostCompileFunc} {
  %c0 = arith.constant 0 : index
  %0 = memref.dim %arg0, %c0 : memref<?x6xf32>
  return %0 : index
}

}

// CHECK:      bytecode.ext.func @dim_dynamic_arg (memref<?x6xf32>) -> index {
// CHECK:        [[ARG:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:        [[IDX:%.+]] = bytecode.imm_register 0
// CHECK:        [[DIM:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT:   bytecode.buffer.get_dim [[DIM]], [[ARG]], [[IDX]]
// CHECK:        bytecode.retv [[DIM]]
// CHECK:      }

// -----

// Dynamic index on a static source: the index register comes from the function argument and the dim is read
// from the runtime buffer descriptor.

module {

func.func @dim_static_alloc_dynamic_index(%idx: index) -> index attributes {config.pureHostCompileFunc} {
  %0 = memref.alloc() : memref<4x6xf32>
  %1 = memref.dim %0, %idx : memref<4x6xf32>
  return %1 : index
}

}

// CHECK:      bytecode.ext.func @dim_static_alloc_dynamic_index (index) -> index {
// CHECK:        [[IDX:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:        [[BUF:%.+]] = bytecode.virtual_general_register
// CHECK:        bytecode.ext.buffer.create [[BUF]], memref<4x6xf32>
// CHECK:        [[DIM:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT:   bytecode.buffer.get_dim [[DIM]], [[BUF]], [[IDX]]
// CHECK:        bytecode.retv [[DIM]]
// CHECK:      }
