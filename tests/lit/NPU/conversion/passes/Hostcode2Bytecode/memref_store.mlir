//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-hostcode-to-bytecode %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

module {
func.func @store_static_i64(%arg0: i64, %arg1: index, %arg2: index) -> () attributes {config.pureHostCompileFunc} {
  %0 = memref.alloc() : memref<2x3xi64>
  memref.store %arg0, %0[%arg1, %arg2] : memref<2x3xi64>
  return
}

// CHECK-LABEL: bytecode.ext.func @store_static_i64 (i64, index, index) -> () {
// CHECK-NEXT:    %0 = bytecode.virtual_parameter_register 2
// CHECK-NEXT:    %1 = bytecode.virtual_parameter_register 1
// CHECK-NEXT:    %2 = bytecode.virtual_parameter_register 0
// CHECK-NEXT:    %3 = bytecode.virtual_general_register
// CHECK-NEXT:    bytecode.ext.buffer.create %3, memref<2x3xi64>
// CHECK-NEXT:    bytecode.buffer.store %3, %2 indices(%1, %0)
// CHECK-NEXT:    bytecode.ret
// CHECK-NEXT:  }

}

// -----

module {
func.func @store_static_f64(%arg0: f64, %arg1: index) -> () attributes {config.pureHostCompileFunc} {
  %0 = memref.alloc() : memref<4xf64>
  memref.store %arg0, %0[%arg1] : memref<4xf64>
  return
}

// CHECK-LABEL: bytecode.ext.func @store_static_f64 (f64, index) -> () {
// CHECK-NEXT:    %0 = bytecode.virtual_parameter_register 1
// CHECK-NEXT:    %1 = bytecode.virtual_parameter_register 0
// CHECK-NEXT:    %2 = bytecode.virtual_general_register
// CHECK-NEXT:    bytecode.ext.buffer.create %2, memref<4xf64>
// CHECK-NEXT:    bytecode.buffer.store %2, %1 indices(%0)
// CHECK-NEXT:    bytecode.ret
// CHECK-NEXT:  }

}

// -----

module {
func.func @store_static_f32_const(%arg0: index) -> () attributes {config.pureHostCompileFunc} {
  %0 = memref.alloc() : memref<4xf32>
  %one = arith.constant 1.0 : f32
  memref.store %one, %0[%arg0] : memref<4xf32>
  return
}

// CHECK-LABEL: bytecode.ext.func @store_static_f32_const (index) -> () {
// CHECK-NEXT:    %0 = bytecode.virtual_parameter_register 0
// CHECK-NEXT:    %1 = bytecode.virtual_general_register
// CHECK-NEXT:    bytecode.ext.buffer.create %1, memref<4xf32>
// CHECK-NEXT:    %2 = bytecode.virtual_general_register
// CHECK-NEXT:    bytecode.set_imm %2, 1065353216
// CHECK-NEXT:    bytecode.buffer.store %1, %2 indices(%0)
// CHECK-NEXT:    bytecode.ret
// CHECK-NEXT:  }

}

// -----

module {
func.func @store_static_f16_const(%arg0: index) -> () attributes {config.pureHostCompileFunc} {
  %0 = memref.alloc() : memref<4xf16>
  %one = arith.constant 1.0 : f16
  memref.store %one, %0[%arg0] : memref<4xf16>
  return
}

// CHECK-LABEL: bytecode.ext.func @store_static_f16_const (index) -> () {
// CHECK-NEXT:    %0 = bytecode.virtual_parameter_register 0
// CHECK-NEXT:    %1 = bytecode.virtual_general_register
// CHECK-NEXT:    bytecode.ext.buffer.create %1, memref<4xf16>
// CHECK-NEXT:    %2 = bytecode.virtual_general_register
// CHECK-NEXT:    bytecode.set_imm %2, 15360
// CHECK-NEXT:    bytecode.buffer.store %1, %2 indices(%0)
// CHECK-NEXT:    bytecode.ret
// CHECK-NEXT:  }

}

// -----

// Output-shape style function: build [1, 16, dim(%main, 2), 1280] in a new memref<4xi64>.
// arith.index_cast is elided because bytecode.buffer.get_dim already produces an i64 payload.

module {

func.func @output_shape(%main: memref<1x16x?x1280xf32>) -> memref<4xi64> attributes {config.pureHostCompileFunc} {
  %c1280_i64 = arith.constant 1280 : i64
  %c2 = arith.constant 2 : index
  %c16_i64 = arith.constant 16 : i64
  %c1_i64 = arith.constant 1 : i64
  %dim = memref.dim %main, %c2 : memref<1x16x?x1280xf32>
  %dim_i64 = arith.index_cast %dim : index to i64
  %out = memref.alloc() {alignment = 64 : i64} : memref<4xi64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2_idx = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  memref.store %c1_i64, %out[%c0] : memref<4xi64>
  memref.store %c16_i64, %out[%c1] : memref<4xi64>
  memref.store %dim_i64, %out[%c2_idx] : memref<4xi64>
  memref.store %c1280_i64, %out[%c3] : memref<4xi64>
  return %out : memref<4xi64>
}

// CHECK-LABEL: bytecode.ext.func @output_shape (memref<1x16x?x1280xf32>) -> memref<4xi64> {
// CHECK-NEXT:    [[MAIN:%.+]] = bytecode.virtual_parameter_register 0
// CHECK-NEXT:    [[V1280:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT:    bytecode.set_imm [[V1280]], 1280
// CHECK-NEXT:    [[DIM_IDX:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT:    bytecode.set_imm [[DIM_IDX]], 2
// CHECK-NEXT:    [[V16:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT:    bytecode.set_imm [[V16]], 16
// CHECK-NEXT:    [[V1:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT:    bytecode.set_imm [[V1]], 1
// CHECK-NEXT:    [[DIM:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT:    bytecode.buffer.get_dim [[DIM]], [[MAIN]], [[DIM_IDX]]
// CHECK-NEXT:    [[OUT:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT:    bytecode.ext.buffer.create [[OUT]], memref<4xi64>
// CHECK-NEXT:    [[I0:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT:    bytecode.set_imm [[I0]], 0
// CHECK-NEXT:    [[I1:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT:    bytecode.set_imm [[I1]], 1
// CHECK-NEXT:    [[I2:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT:    bytecode.set_imm [[I2]], 2
// CHECK-NEXT:    [[I3:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT:    bytecode.set_imm [[I3]], 3
// CHECK-NEXT:    bytecode.buffer.store [[OUT]], [[V1]] indices([[I0]])
// CHECK-NEXT:    bytecode.buffer.store [[OUT]], [[V16]] indices([[I1]])
// CHECK-NEXT:    bytecode.buffer.store [[OUT]], [[DIM]] indices([[I2]])
// CHECK-NEXT:    bytecode.buffer.store [[OUT]], [[V1280]] indices([[I3]])
// CHECK-NEXT:    bytecode.retv [[OUT]]
// CHECK-NEXT:  }

}
