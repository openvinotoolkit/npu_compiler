//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-hostcode-to-bytecode %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

module {

func.func @view_static_reshape() -> () attributes {config.pureHostCompileFunc} {
  %0 = memref.alloc() : memref<8xf32>
  %1 = memref.reinterpret_cast %0 to offset: [0], sizes: [2, 4], strides: [4, 1]
      : memref<8xf32> to memref<2x4xf32>
  return
}

}

// CHECK:  bytecode.func_section @func_section {
// CHECK:    bytecode.ext.func @view_static_reshape () -> () {
// CHECK-NOT:   memref.reinterpret_cast
// CHECK:       [[BUF:%.+]] = bytecode.virtual_general_register
// CHECK:       bytecode.ext.buffer.create [[BUF]], memref<8xf32>
// CHECK:       [[SZ0:%.+]] = bytecode.imm_register 2
// CHECK:       [[SZ1:%.+]] = bytecode.imm_register 4
// CHECK:       [[ST0:%.+]] = bytecode.imm_register 4
// CHECK:       [[ST1:%.+]] = bytecode.imm_register 1
// CHECK:       [[VIEW:%.+]] = bytecode.virtual_general_register
// CHECK:       [[ZERO:%.+]] = bytecode.imm_register 0
// CHECK:       bytecode.ext.buffer.view [[VIEW]], [[BUF]], f32 offset([[ZERO]]) shape([[SZ0]], [[SZ1]]) strides([[ST0]], [[ST1]])
// CHECK:       bytecode.ret
// CHECK:    }

// -----

module {

func.func @view_dynamic_size(%src: memref<1x?xf32>, %rows: index) -> ()
    attributes {config.pureHostCompileFunc} {
  %view = memref.reinterpret_cast %src to offset: [0], sizes: [%rows, 16], strides: [16, 1]
      : memref<1x?xf32> to memref<?x16xf32>
  return
}

}

// CHECK:      bytecode.ext.func @view_dynamic_size (memref<1x?xf32>, index) -> () {
// CHECK-NOT:    memref.reinterpret_cast
// CHECK-DAG:    [[ARG0:%.+]] = bytecode.virtual_parameter_register 0
// CHECK-DAG:    [[ARG1:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:        [[SZ1:%.+]] = bytecode.imm_register 16
// CHECK:        [[ST0:%.+]] = bytecode.imm_register 16
// CHECK:        [[ST1:%.+]] = bytecode.imm_register 1
// CHECK:        [[VIEW:%.+]] = bytecode.virtual_general_register
// CHECK:        [[ZERO:%.+]] = bytecode.imm_register 0
// CHECK:        bytecode.ext.buffer.view [[VIEW]], [[ARG0]], f32 offset([[ZERO]]) shape([[ARG1]], [[SZ1]]) strides([[ST0]], [[ST1]])
// CHECK:        bytecode.ret

// -----

// Static byte-buffer reinterpretation: reinterpret a 16-element i8 buffer as a 4-element i8 view.
// Element type is unchanged; all size/stride slots are static on the ext.buffer.view.

module {

func.func @view_element_type_cast() -> () attributes {config.pureHostCompileFunc} {
  %0 = memref.alloc() : memref<16xi8>
  %1 = memref.reinterpret_cast %0 to offset: [0], sizes: [4], strides: [1]
      : memref<16xi8> to memref<4xi8>
  return
}

}

// CHECK:      bytecode.ext.func @view_element_type_cast () -> () {
// CHECK-NOT:    memref.reinterpret_cast
// CHECK:        [[BUF:%.+]] = bytecode.virtual_general_register
// CHECK:        bytecode.ext.buffer.create [[BUF]], memref<16xi8>
// CHECK:        [[SZ0:%.+]] = bytecode.imm_register 4
// CHECK:        [[ST0:%.+]] = bytecode.imm_register 1
// CHECK:        [[VIEW:%.+]] = bytecode.virtual_general_register
// CHECK:        [[ZERO:%.+]] = bytecode.imm_register 0
// CHECK:        bytecode.ext.buffer.view [[VIEW]], [[BUF]], i8 offset([[ZERO]]) shape([[SZ0]]) strides([[ST0]])
// CHECK:        bytecode.ret

// -----

module {

func.func @view_2d(%base: memref<64xi8>, %rows: index, %cols: index) -> ()
    attributes {config.pureHostCompileFunc} {
  %c0 = arith.constant 0 : index
  %view = memref.view %base[%c0][%rows, %cols] : memref<64xi8> to memref<?x?xf32>
  return
}

}

// CHECK:      bytecode.ext.func @view_2d (memref<64xi8>, index, index) -> () {
// CHECK-NOT:    memref.view
// CHECK-DAG:    [[ARG0:%.+]] = bytecode.virtual_parameter_register 0
// CHECK-DAG:    [[ARG1:%.+]] = bytecode.virtual_parameter_register 1
// CHECK-DAG:    [[ARG2:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:        [[C0:%.+]] = bytecode.imm_register 0
// CHECK:        [[ST1:%.+]] = bytecode.imm_register 1
// CHECK:        [[ST0:%.+]] = bytecode.virtual_general_register
// CHECK:        bytecode.mul.i64 [[ST0]], [[ARG2]], [[ST1]]
// CHECK:        [[VIEW:%.+]] = bytecode.virtual_general_register
// CHECK:        bytecode.ext.buffer.view [[VIEW]], [[ARG0]], f32 offset([[C0]]) shape([[ARG1]], [[ARG2]]) strides([[ST0]], [[ST1]])
// CHECK:        bytecode.ret

// -----

// Mixed static/dynamic view (mirrors the dynamic 2K model): a rank-4 result memref<1x?x?x16xf16> with two
// dynamic dims gets two size operands; the static dims (1 and 16) are materialized as immediates and spliced
// into their slots, preserving dimension order in shape(...).

module {

func.func @view_4d_mixed(%base: memref<1024xi8>, %h: index, %w: index) -> ()
    attributes {config.pureHostCompileFunc} {
  %c0 = arith.constant 0 : index
  %v = memref.view %base[%c0][%h, %w] : memref<1024xi8> to memref<1x?x?x16xf16>
  return
}

}

// CHECK:      bytecode.ext.func @view_4d_mixed (memref<1024xi8>, index, index) -> () {
// CHECK-NOT:    memref.view
// CHECK-DAG:    [[W:%.+]] = bytecode.virtual_parameter_register 2
// CHECK-DAG:    [[H:%.+]] = bytecode.virtual_parameter_register 1
// CHECK-DAG:    [[BASE:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:        [[C0:%.+]] = bytecode.imm_register 0
// CHECK:        [[D0:%.+]] = bytecode.imm_register 1
// CHECK:        [[D3:%.+]] = bytecode.imm_register 16
// CHECK:        [[ST3:%.+]] = bytecode.imm_register 1
// CHECK:        [[ST2:%.+]] = bytecode.virtual_general_register
// CHECK:        bytecode.mul.i64 [[ST2]], [[D3]], [[ST3]]
// CHECK:        [[ST1:%.+]] = bytecode.virtual_general_register
// CHECK:        bytecode.mul.i64 [[ST1]], [[W]], [[ST2]]
// CHECK:        [[ST0:%.+]] = bytecode.virtual_general_register
// CHECK:        bytecode.mul.i64 [[ST0]], [[H]], [[ST1]]
// CHECK:        [[VIEW:%.+]] = bytecode.virtual_general_register
// CHECK:        bytecode.ext.buffer.view [[VIEW]], [[BASE]], f16 offset([[C0]]) shape([[D0]], [[H]], [[W]], [[D3]]) strides([[ST0]], [[ST1]], [[ST2]], [[ST3]])
// CHECK:        bytecode.ret
