//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-hostcode-to-bytecode %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

module {

func.func @subview_static_float() -> () attributes {config.pureHostCompileFunc} {
  %0 = memref.alloc() : memref<4x6xf32>
  %1 = memref.subview %0[1, 1] [2, 3] [1, 1] : memref<4x6xf32> to memref<2x3xf32, strided<[6, 1], offset: 7>>
  return
}

}

// CHECK:  bytecode.func_section @func_section {
// CHECK:    bytecode.ext.func @subview_static_float () -> () {
// CHECK-NOT:   memref.subview
// CHECK:       [[BUF:%.+]] = bytecode.virtual_general_register
// CHECK:       bytecode.ext.buffer.create [[BUF]], memref<4x6xf32>
// CHECK:       [[O0:%.+]] = bytecode.imm_register 1
// CHECK:       [[O1:%.+]] = bytecode.imm_register 1
// CHECK:       [[SZ0:%.+]] = bytecode.imm_register 2
// CHECK:       [[SZ1:%.+]] = bytecode.imm_register 3
// CHECK:       [[ST0:%.+]] = bytecode.imm_register 1
// CHECK:       [[ST1:%.+]] = bytecode.imm_register 1
// CHECK:       [[VIEW:%.+]] = bytecode.virtual_general_register
// CHECK:       bytecode.buffer.subview [[VIEW]], [[BUF]] offsets([[O0]], [[O1]]) sizes([[SZ0]], [[SZ1]]) strides([[ST0]], [[ST1]])
// CHECK:       bytecode.ret
// CHECK:    }

// -----

// Real-world example: slice a runtime sequence dimension out of a dynamic-shape input. The dynamic offset slot
// reuses the function-parameter register; remaining slots go through set_imm. SSA names mirror the production
// IR exactly so this test pins down the lowering for that specific pattern.

module {

func.func @subview_dynamic_offset(%Parameter_10: memref<1x16x?x1280xf32>, %Convert_11_5: index) -> ()
    attributes {config.pureHostCompileFunc} {
  %Convert_11_6 = memref.subview %Parameter_10[0, 0, %Convert_11_5, 0] [1, 16, 31, 1280] [1, 1, 1, 1]
      : memref<1x16x?x1280xf32> to memref<1x16x31x1280xf32, strided<[?, ?, 1280, 1], offset: ?>>
  return
}

}

// CHECK:      bytecode.ext.func @subview_dynamic_offset (memref<1x16x?x1280xf32>, index) -> () {
// CHECK-NOT:    memref.subview
// CHECK-DAG:    [[ARG0:%.+]] = bytecode.virtual_parameter_register 0
// CHECK-DAG:    [[ARG1:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:        [[O0:%.+]] = bytecode.imm_register 0
// CHECK:        [[O1:%.+]] = bytecode.imm_register 0
// CHECK:        [[O3:%.+]] = bytecode.imm_register 0
// CHECK:        [[SZ0:%.+]] = bytecode.imm_register 1
// CHECK:        [[SZ1:%.+]] = bytecode.imm_register 16
// CHECK:        [[SZ2:%.+]] = bytecode.imm_register 31
// CHECK:        [[SZ3:%.+]] = bytecode.imm_register 1280
// CHECK:        [[ST0:%.+]] = bytecode.imm_register 1
// CHECK:        [[ST1:%.+]] = bytecode.imm_register 1
// CHECK:        [[ST2:%.+]] = bytecode.imm_register 1
// CHECK:        [[ST3:%.+]] = bytecode.imm_register 1
// CHECK:        [[VIEW:%.+]] = bytecode.virtual_general_register
// CHECK:        bytecode.buffer.subview [[VIEW]], [[ARG0]]
// CHECK-SAME:     offsets([[O0]], [[O1]], [[ARG1]], [[O3]])
// CHECK-SAME:     sizes([[SZ0]], [[SZ1]], [[SZ2]], [[SZ3]])
// CHECK-SAME:     strides([[ST0]], [[ST1]], [[ST2]], [[ST3]])
// CHECK:        bytecode.ret

// -----

// Two dynamic offset slots: H and W spatial offsets on a runtime-shaped NCHW input. Pins down that each
// dynamic slot is paired with the correct parameter register in rank order, exercising the dynIdx advancement
// that the single-dynamic-offset case alone does not cover.

module {

func.func @subview_dynamic_hw_offsets(%src: memref<1x16x?x?xf32>, %offH: index, %offW: index) -> ()
    attributes {config.pureHostCompileFunc} {
  %0 = memref.subview %src[0, 0, %offH, %offW] [1, 16, 56, 56] [1, 1, 1, 1]
      : memref<1x16x?x?xf32> to memref<1x16x56x56xf32, strided<[?, ?, ?, 1], offset: ?>>
  return
}

}

// CHECK:      bytecode.ext.func @subview_dynamic_hw_offsets (memref<1x16x?x?xf32>, index, index) -> () {
// CHECK-NOT:    memref.subview
// CHECK-DAG:    [[ARG0:%.+]] = bytecode.virtual_parameter_register 0
// CHECK-DAG:    [[ARG1:%.+]] = bytecode.virtual_parameter_register 1
// CHECK-DAG:    [[ARG2:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:        [[O0:%.+]] = bytecode.imm_register 0
// CHECK:        [[O1:%.+]] = bytecode.imm_register 0
// CHECK:        [[SZ0:%.+]] = bytecode.imm_register 1
// CHECK:        [[SZ1:%.+]] = bytecode.imm_register 16
// CHECK:        [[SZ2:%.+]] = bytecode.imm_register 56
// CHECK:        [[SZ3:%.+]] = bytecode.imm_register 56
// CHECK:        [[ST0:%.+]] = bytecode.imm_register 1
// CHECK:        [[ST1:%.+]] = bytecode.imm_register 1
// CHECK:        [[ST2:%.+]] = bytecode.imm_register 1
// CHECK:        [[ST3:%.+]] = bytecode.imm_register 1
// CHECK:        [[VIEW:%.+]] = bytecode.virtual_general_register
// CHECK:        bytecode.buffer.subview [[VIEW]], [[ARG0]]
// CHECK-SAME:     offsets([[O0]], [[O1]], [[ARG1]], [[ARG2]])
// CHECK-SAME:     sizes([[SZ0]], [[SZ1]], [[SZ2]], [[SZ3]])
// CHECK-SAME:     strides([[ST0]], [[ST1]], [[ST2]], [[ST3]])
// CHECK:        bytecode.ret

// -----

// Dynamic size slot: the dynamic size register comes straight from the function argument; all other slots go
// through set_imm.

module {

func.func @subview_dynamic_size(%arg0: memref<4x6xf32>, %arg1: index) -> () attributes {config.pureHostCompileFunc} {
  %0 = memref.subview %arg0[0, 0] [%arg1, 3] [1, 1]
      : memref<4x6xf32> to memref<?x3xf32, strided<[6, 1], offset: 0>>
  return
}

}

// CHECK:      bytecode.ext.func @subview_dynamic_size (memref<4x6xf32>, index) -> () {
// CHECK-NOT:    memref.subview
// CHECK-DAG:    [[ARG0:%.+]] = bytecode.virtual_parameter_register 0
// CHECK-DAG:    [[ARG1:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:        [[O0:%.+]] = bytecode.imm_register 0
// CHECK:        [[O1:%.+]] = bytecode.imm_register 0
// CHECK:        [[SZ1:%.+]] = bytecode.imm_register 3
// CHECK:        [[ST0:%.+]] = bytecode.imm_register 1
// CHECK:        [[ST1:%.+]] = bytecode.imm_register 1
// CHECK:        [[VIEW:%.+]] = bytecode.virtual_general_register
// CHECK:        bytecode.buffer.subview [[VIEW]], [[ARG0]]
// CHECK-SAME:     offsets([[O0]], [[O1]])
// CHECK-SAME:     sizes([[ARG1]], [[SZ1]])
// CHECK-SAME:     strides([[ST0]], [[ST1]])
// CHECK:        bytecode.ret

// -----

// Dynamic stride slot: the runtime stride flows from a function argument; the remaining slots are immediate.

module {

func.func @subview_dynamic_stride(%arg0: memref<8x6xf32>, %arg1: index) -> () attributes {config.pureHostCompileFunc} {
  %0 = memref.subview %arg0[0, 0] [2, 3] [%arg1, 1]
      : memref<8x6xf32> to memref<2x3xf32, strided<[?, 1], offset: 0>>
  return
}

}

// CHECK:      bytecode.ext.func @subview_dynamic_stride (memref<8x6xf32>, index) -> () {
// CHECK-NOT:    memref.subview
// CHECK-DAG:    [[ARG0:%.+]] = bytecode.virtual_parameter_register 0
// CHECK-DAG:    [[ARG1:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:        [[O0:%.+]] = bytecode.imm_register 0
// CHECK:        [[O1:%.+]] = bytecode.imm_register 0
// CHECK:        [[SZ0:%.+]] = bytecode.imm_register 2
// CHECK:        [[SZ1:%.+]] = bytecode.imm_register 3
// CHECK:        [[ST1:%.+]] = bytecode.imm_register 1
// CHECK:        [[VIEW:%.+]] = bytecode.virtual_general_register
// CHECK:        bytecode.buffer.subview [[VIEW]], [[ARG0]]
// CHECK-SAME:     offsets([[O0]], [[O1]])
// CHECK-SAME:     sizes([[SZ0]], [[SZ1]])
// CHECK-SAME:     strides([[ARG1]], [[ST1]])
// CHECK:        bytecode.ret

// -----

// All subview offset, size, and stride slots are supplied through function arguments, and a single size argument
// feeds both rank-0 and rank-1 size slots. This pins down that every dynamic operand group is consumed in rank
// order without materializing immediate registers for those slots, and that a reused argument register is wired
// to every slot that references it.

module {

func.func @subview_all_dynamic_arguments(%src: memref<64x64xf32>, %off0: index, %off1: index, %size: index,
                                         %stride0: index, %stride1: index) -> ()
    attributes {config.pureHostCompileFunc} {
  %0 = memref.subview %src[%off0, %off1] [%size, %size] [%stride0, %stride1]
      : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  return
}

}

// CHECK:      bytecode.ext.func @subview_all_dynamic_arguments
// CHECK-SAME:    (memref<64x64xf32>, index, index, index, index, index) -> () {
// CHECK-NOT:    memref.subview
// CHECK-DAG:    [[ARG0:%.+]] = bytecode.virtual_parameter_register 0
// CHECK-DAG:    [[ARG1:%.+]] = bytecode.virtual_parameter_register 1
// CHECK-DAG:    [[ARG2:%.+]] = bytecode.virtual_parameter_register 2
// CHECK-DAG:    [[ARG3:%.+]] = bytecode.virtual_parameter_register 3
// CHECK-DAG:    [[ARG4:%.+]] = bytecode.virtual_parameter_register 4
// CHECK-DAG:    [[ARG5:%.+]] = bytecode.virtual_parameter_register 5
// CHECK:        [[VIEW:%.+]] = bytecode.virtual_general_register
// CHECK:        bytecode.buffer.subview [[VIEW]], [[ARG0]]
// CHECK-SAME:     offsets([[ARG1]], [[ARG2]])
// CHECK-SAME:     sizes([[ARG3]], [[ARG3]])
// CHECK-SAME:     strides([[ARG4]], [[ARG5]])
// CHECK:        bytecode.ret
