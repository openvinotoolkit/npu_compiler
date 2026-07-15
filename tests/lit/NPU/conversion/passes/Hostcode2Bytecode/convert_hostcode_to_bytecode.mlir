//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile_Interpreter" --convert-hostcode-to-bytecode %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

module {

func.func @fn1(%arg0 : i64) -> () attributes {config.pureHostCompileFunc} {
  %c32f = arith.constant 32.0 : f64
  %c32i = arith.constant 32 : i64
  %add = arith.addi %arg0, %c32i : i64
  return
}
func.func @fn2() -> () attributes {config.pureHostCompileFunc} {
  %false = arith.constant false
  %true = arith.constant true
  cf.assert %false, "Assertion failed"
  cf.assert %true, "Assertion failed"
  return
}
func.func @fn3(%arg0 : i64, %arg1 : i64) -> () attributes {config.pureHostCompileFunc} {
  %minus_one = arith.constant -1 : i64
  %mul = arith.muli %arg0, %arg1 : i64
  %min = arith.minsi %arg0, %arg1 : i64
  %max = arith.maxsi %arg0, %arg1 : i64
  %and = arith.andi %arg0, %arg1 : i64
  %or = arith.ori %arg0, %arg1 : i64
  %xor = arith.xori %arg0, %arg1 : i64
  %not = arith.xori %arg0, %minus_one : i64
  %not_swapped = arith.xori %minus_one, %arg1 : i64
  %sll = arith.shli %arg0, %arg1 : i64
  %srl = arith.shrui %arg0, %arg1 : i64
  %sra = arith.shrsi %arg0, %arg1 : i64
  return
}
func.func @fn4(%arg0 : i64, %arg1 : i64) -> () attributes {config.pureHostCompileFunc} {
  %eq  = arith.cmpi eq,  %arg0, %arg1 : i64
  %slt = arith.cmpi slt, %arg0, %arg1 : i64
  %ult = arith.cmpi ult, %arg0, %arg1 : i64
  return
}
func.func @fn5(%arg0 : i64, %arg1 : i64) -> () attributes {config.pureHostCompileFunc} {
  %sub = arith.subi %arg0, %arg1 : i64
  %div = arith.divsi %arg0, %arg1 : i64
  %divu = arith.divui %arg0, %arg1 : i64
  %minu = arith.minui %arg0, %arg1 : i64
  %maxu = arith.maxui %arg0, %arg1 : i64
  %remu = arith.remui %arg0, %arg1 : i64
  %addu, %addu_overflow = arith.addui_extended %arg0, %arg1 : i64, i1
  %mulu, %mulu_high = arith.mului_extended %arg0, %arg1 : i64
  %rem = arith.remsi %arg0, %arg1 : i64
  return
}
func.func @fn6(%arg0 : i64) -> () attributes {config.pureHostCompileFunc} {
  %abs = math.absi %arg0 : i64
  return
}
func.func @fn7(%arg0 : i1, %arg1 : i64, %arg2 : i64) -> () attributes {config.pureHostCompileFunc} {
  %sel = arith.select %arg0, %arg1, %arg2 : i64
  return
}
func.func @fn8(%arg0 : f64, %arg1 : f64) -> () attributes {config.pureHostCompileFunc} {
  %add_f64 = arith.addf %arg0, %arg1 : f64
  %sub_f64 = arith.subf %arg0, %arg1 : f64
  %mul_f64 = arith.mulf %arg0, %arg1 : f64
  %div_f64 = arith.divf %arg0, %arg1 : f64
  %rem_f64 = arith.remf %arg0, %arg1 : f64
  %max_f64 = arith.maximumf %arg0, %arg1 : f64
  %min_f64 = arith.minimumf %arg0, %arg1 : f64
  return
}
func.func @fn9(%arg0 : f64) -> () attributes {config.pureHostCompileFunc} {
  %abs = math.absf %arg0 : f64
  %neg = arith.negf %arg0 : f64
  %ceil = math.ceil %arg0 : f64
  %floor = math.floor %arg0 : f64
  return
}
func.func @fn10(%arg0 : f64) -> () attributes {config.pureHostCompileFunc} {
  %rne = math.roundeven %arg0 : f64
  %rna = math.round %arg0 : f64
  %rtz = math.trunc %arg0 : f64
  return
}

// CHECK:  bytecode.func_section @func_section {
// CHECK:    bytecode.ext.func @fn1 (i64) -> () {
// CHECK:      [[PARAM_REG:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:      [[REG_C32F:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[REG_C32F]], 4629700416936869888
// CHECK:      [[REG_C32I:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[REG_C32I]], 32
// CHECK:      [[REG_ADD:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.add.i64 [[REG_ADD]], [[PARAM_REG]], [[REG_C32I]]
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:    bytecode.ext.func @fn2 () -> () {
// CHECK:      [[REG_FALSE:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[REG_FALSE]], 0
// CHECK:      [[REG_TRUE:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[REG_TRUE]], 1
// CHECK:      bytecode.ext.assert [[REG_FALSE]], "Assertion failed"
// CHECK:      bytecode.ext.assert [[REG_TRUE]], "Assertion failed"
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:    bytecode.ext.func @fn3 (i64, i64) -> () {
// CHECK:      [[PARAM1:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:      [[PARAM0:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:      [[REG_MINUS_ONE:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[REG_MINUS_ONE]], -1
// CHECK:      [[REG_MUL:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.mul.i64 [[REG_MUL]], [[PARAM0]], [[PARAM1]]
// CHECK:      [[REG_MIN:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.min.i64 [[REG_MIN]], [[PARAM0]], [[PARAM1]]
// CHECK:      [[REG_MAX:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.max.i64 [[REG_MAX]], [[PARAM0]], [[PARAM1]]
// CHECK:      [[REG_AND:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.and.64 [[REG_AND]], [[PARAM0]], [[PARAM1]]
// CHECK:      [[REG_OR:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.or.64 [[REG_OR]], [[PARAM0]], [[PARAM1]]
// CHECK:      [[REG_XOR:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.xor.64 [[REG_XOR]], [[PARAM0]], [[PARAM1]]
// CHECK:      [[REG_NOT:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.not.64 [[REG_NOT]], [[PARAM0]]
// CHECK:      [[REG_NOT_SWAPPED:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.not.64 [[REG_NOT_SWAPPED]], [[PARAM1]]
// CHECK:      [[REG_SLL:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.sll.64 [[REG_SLL]], [[PARAM0]], [[PARAM1]]
// CHECK:      [[REG_SRL:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.srl.64 [[REG_SRL]], [[PARAM0]], [[PARAM1]]
// CHECK:      [[REG_SRA:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.sra.64 [[REG_SRA]], [[PARAM0]], [[PARAM1]]
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:    bytecode.ext.func @fn4 (i64, i64) -> () {
// CHECK:      [[PARAM1:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:      [[PARAM0:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:      [[REG_EQ:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.cmp.i64 [[REG_EQ]], [[PARAM0]], [[PARAM1]], 0
// CHECK:      [[REG_SLT:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.cmp.i64 [[REG_SLT]], [[PARAM0]], [[PARAM1]], 260
// CHECK:      [[REG_ULT:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.cmp.i64 [[REG_ULT]], [[PARAM0]], [[PARAM1]], 4
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:    bytecode.ext.func @fn5 (i64, i64) -> () {
// CHECK:      [[PARAM1:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:      [[PARAM0:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:      [[REG_SUB:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.sub.i64 [[REG_SUB]], [[PARAM0]], [[PARAM1]]
// CHECK:      [[REG_DIV:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.div.i64 [[REG_DIV]], [[PARAM0]], [[PARAM1]]
// CHECK:      [[REG_DIVU:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.div.u64 [[REG_DIVU]], [[PARAM0]], [[PARAM1]]
// CHECK:      [[REG_MINU:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.min.u64 [[REG_MINU]], [[PARAM0]], [[PARAM1]]
// CHECK:      [[REG_MAXU:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.max.u64 [[REG_MAXU]], [[PARAM0]], [[PARAM1]]
// CHECK:      [[REG_REMU:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.rem.u64 [[REG_REMU]], [[PARAM0]], [[PARAM1]]
// CHECK:      [[REG_ADDU:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.add.u64 [[REG_ADDU]], [[PARAM0]], [[PARAM1]]
// CHECK:      {{%.+}} = bytecode.virtual_general_register
// CHECK:      [[REG_MULU:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.mul.u64 [[REG_MULU]], [[PARAM0]], [[PARAM1]]
// CHECK:      {{%.+}} = bytecode.virtual_general_register
// CHECK:      [[REG_REM:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.rem.i64 [[REG_REM]], [[PARAM0]], [[PARAM1]]
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:    bytecode.ext.func @fn6 (i64) -> () {
// CHECK:      [[PARAM0:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:      [[REG_ABS:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.abs.i64 [[REG_ABS]], [[PARAM0]]
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:    bytecode.ext.func @fn7 (i1, i64, i64) -> () {
// CHECK:      [[PARAM2:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:      [[PARAM1:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:      [[PARAM0:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:      [[REG_SEL:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.select [[REG_SEL]], [[PARAM0]], [[PARAM1]], [[PARAM2]]
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:    bytecode.ext.func @fn8 (f64, f64) -> () {
// CHECK:      [[PARAM1:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:      [[PARAM0:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:      [[REG_ADD_F64:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.add.f64 [[REG_ADD_F64]], [[PARAM0]], [[PARAM1]]
// CHECK:      [[REG_SUB_F64:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.sub.f64 [[REG_SUB_F64]], [[PARAM0]], [[PARAM1]]
// CHECK:      [[REG_MUL_F64:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.mul.f64 [[REG_MUL_F64]], [[PARAM0]], [[PARAM1]]
// CHECK:      [[REG_DIV_F64:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.div.f64 [[REG_DIV_F64]], [[PARAM0]], [[PARAM1]]
// CHECK:      [[REG_REM_F64:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.rem.f64 [[REG_REM_F64]], [[PARAM0]], [[PARAM1]]
// CHECK:      [[REG_MAX_F64:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.max.f64 [[REG_MAX_F64]], [[PARAM0]], [[PARAM1]]
// CHECK:      [[REG_MIN_F64:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.min.f64 [[REG_MIN_F64]], [[PARAM0]], [[PARAM1]]
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:    bytecode.ext.func @fn9 (f64) -> () {
// CHECK:      [[PARAM_F:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:      [[REG_ABS:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.abs.f64 [[REG_ABS]], [[PARAM_F]]
// CHECK:      [[REG_NEG:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.neg.f64 [[REG_NEG]], [[PARAM_F]]
// CHECK:      [[REG_CEIL:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.ceil.f64 [[REG_CEIL]], [[PARAM_F]]
// CHECK:      [[REG_FLOOR:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.floor.f64 [[REG_FLOOR]], [[PARAM_F]]
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:    bytecode.ext.func @fn10 (f64) -> () {
// CHECK:      [[PARAM_F10:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:      [[REG_RNE:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.round.f64 [[REG_RNE]], [[PARAM_F10]], 0
// CHECK:      [[REG_RNA:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.round.f64 [[REG_RNA]], [[PARAM_F10]], 1
// CHECK:      [[REG_RTZ:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.round.f64 [[REG_RTZ]], [[PARAM_F10]], 4
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:  }

}

// -----

module @test_async_create_group {
  func.func @fn_create_group() -> () attributes {config.pureHostCompileFunc} {
    %size = arith.constant 4 : index
    %group = async.create_group %size : !async.group
    return
  }
}

// CHECK-LABEL: module @test_async_create_group
// CHECK:    bytecode.ext.func @fn_create_group () -> () {
// CHECK:      [[SIZE_REG:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[SIZE_REG]], 4
// CHECK:      [[CMD_LIST:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.cmd_list.create [[CMD_LIST]]
// CHECK:      bytecode.ret
// CHECK:    }

// -----

module @test_async_create_group {
    func.func @fn_create_group(%arg0: index) -> () attributes {config.pureHostCompileFunc} {
    %size = arith.constant 4 : index
    %group = async.create_group %size : !async.group

    %add1 = arith.addi %size, %arg0: index
    return
  }
}

// CHECK-LABEL: module @test_async_create_group
// CHECK:    bytecode.ext.func @fn_create_group (index) -> () {
// CHECK:      [[PARAM_REG:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:      [[CONST:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[CONST]], 4

// CHECK:      [[CMD_LIST:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.cmd_list.create [[CMD_LIST]]

// CHECK:      [[ADD_REG:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.add.i64 [[ADD_REG]], [[PARAM_REG]], [[CONST]]
// CHECK:      bytecode.ret
// CHECK:    }

// -----

module @test_async_create_group {
  func.func @fn_create_group() -> () attributes {config.pureHostCompileFunc} {
    %size = arith.constant 4 : index
    %group0 = async.create_group %size : !async.group
    %group1 = async.create_group %size : !async.group
    return
  }
}

// CHECK-LABEL: module @test_async_create_group
// CHECK:    bytecode.ext.func @fn_create_group () -> () {
// CHECK:      [[SIZE_REG:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[SIZE_REG]], 4
// CHECK:      [[CMD_LIST0:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.cmd_list.create [[CMD_LIST0]]

// CHECK:      [[CMD_LIST1:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.cmd_list.create [[CMD_LIST1]]
// CHECK:      bytecode.ret
// CHECK:    }

// -----

module @test_async_await_all {
  func.func @fn_await_all() -> () attributes {config.pureHostCompileFunc} {
    %size = arith.constant 4 : index
    %group = async.create_group %size : !async.group
    async.await_all %group
    return
  }
}

// CHECK-LABEL: module @test_async_await_all
// CHECK:    bytecode.ext.func @fn_await_all () -> () {
// CHECK:      [[SIZE_REG:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[SIZE_REG]], 4
// CHECK:      [[CMD_LIST:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.cmd_list.create [[CMD_LIST]]
// CHECK:      bytecode.cmd_list.close [[CMD_LIST]]
// CHECK:      bytecode.cmd_list.exec [[CMD_LIST]]
// CHECK:      bytecode.ret
// CHECK:    }

// -----

module @test_async_execute_kernel_create {
  bytecode.kernel_section @kernel_section {
    bytecode.kernel @compute "\00\01\02\03"
  }
  module @SubModule {
    func.func nested @compute(memref<16xf32>, memref<16xf32>) -> ()
  }
  func.func @fn(%arg0: memref<16xf32>, %arg1: memref<16xf32>) -> () attributes {config.pureHostCompileFunc} {
    %token = async.execute {
      Core.NestedCall @SubModule::@compute(%arg0, %arg1) : (memref<16xf32>, memref<16xf32>) -> ()
      async.yield
    }
    return
  }
}

// CHECK-LABEL: module @test_async_execute_kernel_create
// CHECK:    bytecode.ext.func @fn (memref<16xf32>, memref<16xf32>) -> () {
// CHECK:      [[PARAM1:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:      [[PARAM0:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:      [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.kernel.create [[DST]], @compute, inputs([[PARAM0]], [[PARAM1]]), outputs()
// CHECK:    }

// -----

module @test_async_add_to_group {
  func.func @fn_add_to_group(%token: !async.token) -> () attributes {config.pureHostCompileFunc} {
    %size = arith.constant 1 : index
    %group = async.create_group %size : !async.group
    %rank = async.add_to_group %token, %group : !async.token
    return
  }
}

// CHECK-LABEL: module @test_async_add_to_group
// CHECK:    bytecode.ext.func @fn_add_to_group (!async.token) -> () {
// CHECK:      [[PARAM_TOKEN:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:      [[SIZE_REG:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[SIZE_REG]], 1
// CHECK:      [[CMD_LIST:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.cmd_list.create [[CMD_LIST]]
// CHECK:      bytecode.cmd_list.add_kernel [[CMD_LIST]], [[PARAM_TOKEN]], (), ()
// CHECK:      bytecode.ret
// CHECK:    }

// -----

// CHECK-LABEL: bytecode.ext.func @test_no_block_args
module {
  func.func @test_no_block_args() attributes {config.pureHostCompileFunc} {
    cf.br ^bb1
  ^bb1:
    return
  }
}

// CHECK:      bytecode.jmp ^bb1
// CHECK-NOT:  cf.br
// CHECK-NOT:  bytecode.set %
// CHECK:      ^bb1:
// CHECK-NEXT: bytecode.ret

// -----

// CHECK-LABEL: bytecode.ext.func @test_one_block_arg
module {
  func.func @test_one_block_arg(%arg0: index) attributes {config.pureHostCompileFunc} {
    %c0 = arith.constant 0 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%x: index):
    %unused = arith.addi %x, %x : index
    return
  }
}

// CHECK:      [[CANON_X:%.+]] = bytecode.virtual_general_register
// CHECK:      [[C0:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT: bytecode.set_imm [[C0]], 0
// CHECK-NEXT: bytecode.set [[CANON_X]], [[C0]]
// CHECK:      bytecode.jmp ^bb1
// CHECK-NOT:  cf.br
// CHECK:      ^bb1:
// CHECK:      bytecode.add.i64 {{%.+}}, [[CANON_X]], [[CANON_X]]
// CHECK:      bytecode.ret

// -----

// CHECK-LABEL: bytecode.ext.func @test_two_block_args
module {
  func.func @test_two_block_args(%a: index, %b: index) attributes {config.pureHostCompileFunc} {
    cf.br ^bb1(%a, %b : index, index)
  ^bb1(%x: index, %y: index):
    %sum = arith.addi %x, %y : index
    return
  }
}

// CHECK-DAG: [[PARAM0:%.+]] = bytecode.virtual_parameter_register 0
// CHECK-DAG: [[PARAM1:%.+]] = bytecode.virtual_parameter_register 1
// CHECK-DAG: [[CANON_X:%.+]] = bytecode.virtual_general_register
// CHECK-DAG: [[CANON_Y:%.+]] = bytecode.virtual_general_register
// CHECK-DAG: bytecode.set [[CANON_X]], [[PARAM0]]
// CHECK-DAG: bytecode.set [[CANON_Y]], [[PARAM1]]
// CHECK:     bytecode.jmp ^bb1
// CHECK-NOT: cf.br
// CHECK:     ^bb1:
// CHECK:     bytecode.add.i64 {{%.+}}, [[CANON_X]], [[CANON_Y]]

// -----

// CHECK-LABEL: bytecode.ext.func @test_loop
module {
  func.func @test_loop() attributes {config.pureHostCompileFunc} {
    %c0 = arith.constant 0 : index
    cf.br ^loop(%c0 : index)
  ^loop(%i: index):
    %c1 = arith.constant 1 : index
    %next = arith.addi %i, %c1 : index
    cf.br ^loop(%next : index)
  }
}

// CHECK:      [[CANON_I:%.+]] = bytecode.virtual_general_register
// CHECK:      [[C0:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT: bytecode.set_imm [[C0]], 0
// CHECK-NEXT: bytecode.set [[CANON_I]], [[C0]]
// CHECK:      bytecode.jmp ^bb1
// CHECK-NOT:  cf.br
// CHECK:      ^bb1:
// CHECK:      bytecode.add.i64 {{%.+}}, [[CANON_I]], {{%.+}}
// CHECK:      bytecode.set [[CANON_I]], {{%.+}}
// CHECK:      bytecode.jmp ^bb1
// CHECK-NOT:  cf.br

// -----

module @test_kernel_submission_ops_together {
  bytecode.kernel_section @kernel_section {
    bytecode.kernel @compute "\00\01\02\03"
  }
  module @SubModule {
    func.func nested @compute(memref<16xf32>, memref<16xf32>) -> ()
  }
  func.func @fn(%arg0: memref<16xf32>, %arg1: memref<16xf32>) -> () attributes {config.pureHostCompileFunc} {
    %size = arith.constant 1 : index
    %group = async.create_group %size : !async.group
    %token = async.execute {
      Core.NestedCall @SubModule::@compute(%arg0, %arg1) : (memref<16xf32>, memref<16xf32>) -> ()
      async.yield
    }
    %rank = async.add_to_group %token, %group : !async.token
    async.await_all %group
    return
  }
}

// CHECK-LABEL: module @test_kernel_submission_ops_together
// CHECK:    bytecode.ext.func @fn (memref<16xf32>, memref<16xf32>) -> () {
// CHECK:      [[PARAM1:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:      [[PARAM0:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:      [[SIZE_REG:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[SIZE_REG]], 1
// CHECK:      [[CMD_LIST:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.cmd_list.create [[CMD_LIST]]
// CHECK:      [[KERNEL:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.kernel.create [[KERNEL]], @compute, inputs([[PARAM0]], [[PARAM1]]), outputs()
// CHECK:      bytecode.cmd_list.add_kernel [[CMD_LIST]], [[KERNEL]], (), ()
// CHECK:      bytecode.cmd_list.close [[CMD_LIST]]
// CHECK:      bytecode.cmd_list.exec [[CMD_LIST]]
// CHECK:      bytecode.ret
// CHECK:  }

// -----

module {

func.func @callee(%arg0 : i64) -> () attributes {config.pureHostCompileFunc} {
  return
}

func.func @caller(%arg0 : i64) -> () attributes {config.pureHostCompileFunc} {
  func.call @callee(%arg0) : (i64) -> ()
  return
}

// CHECK:  bytecode.func_section @func_section {
// CHECK:    bytecode.ext.func @callee (i64) -> () {
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:    bytecode.ext.func @caller (i64) -> () {
// CHECK:      [[CALLER_ARG:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:      [[FUNC_IDX:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[FUNC_IDX]], 0
// CHECK:      bytecode.call [[FUNC_IDX]], results(), args([[CALLER_ARG]] : !bytecode.Register)
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:  }

}

// -----

module {

func.func @callee_multi(%a : i64, %b : i64, %c : i64) -> () attributes {config.pureHostCompileFunc} {
  return
}

func.func @caller_multi(%arg0 : i64, %arg1 : i64) -> () attributes {config.pureHostCompileFunc} {
  %sum = arith.addi %arg0, %arg1 : i64
  func.call @callee_multi(%arg1, %sum, %arg0) : (i64, i64, i64) -> ()
  return
}

// CHECK:  bytecode.func_section @func_section {
// CHECK:    bytecode.ext.func @callee_multi (i64, i64, i64) -> () {
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:    bytecode.ext.func @caller_multi (i64, i64) -> () {
// CHECK-DAG:  [[CALLER_ARG1:%.+]] = bytecode.virtual_parameter_register 1
// CHECK-DAG:  [[CALLER_ARG0:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:      [[REG_SUM:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.add.i64 [[REG_SUM]], [[CALLER_ARG0]], [[CALLER_ARG1]]
// CHECK:      [[FUNC_IDX:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[FUNC_IDX]], 0
// CHECK:      bytecode.call [[FUNC_IDX]], results(), args([[CALLER_ARG1]], [[REG_SUM]], [[CALLER_ARG0]] : !bytecode.Register, !bytecode.Register, !bytecode.Register)
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:  }

}

// -----

module {

func.func @callee_a(%x : i64) -> () attributes {config.pureHostCompileFunc} {
  return
}

func.func @callee_b(%y : i64) -> () attributes {config.pureHostCompileFunc} {
  return
}

func.func @caller_chain(%arg0 : i64, %arg1 : i64) -> () attributes {config.pureHostCompileFunc} {
  func.call @callee_a(%arg0) : (i64) -> ()
  func.call @callee_b(%arg1) : (i64) -> ()
  func.call @callee_a(%arg1) : (i64) -> ()
  return
}

// CHECK:  bytecode.func_section @func_section {
// CHECK:    bytecode.ext.func @callee_a (i64) -> () {
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:    bytecode.ext.func @callee_b (i64) -> () {
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:    bytecode.ext.func @caller_chain (i64, i64) -> () {
// CHECK-DAG:  [[CHAIN_ARG1:%.+]] = bytecode.virtual_parameter_register 1
// CHECK-DAG:  [[CHAIN_ARG0:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:      [[FUNC_IDX_A0:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[FUNC_IDX_A0]], 0
// CHECK:      bytecode.call [[FUNC_IDX_A0]], results(), args([[CHAIN_ARG0]] : !bytecode.Register)
// CHECK:      [[FUNC_IDX_B:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[FUNC_IDX_B]], 1
// CHECK:      bytecode.call [[FUNC_IDX_B]], results(), args([[CHAIN_ARG1]] : !bytecode.Register)
// CHECK:      [[FUNC_IDX_A1:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[FUNC_IDX_A1]], 0
// CHECK:      bytecode.call [[FUNC_IDX_A1]], results(), args([[CHAIN_ARG1]] : !bytecode.Register)
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:  }

}

// -----

module {

func.func @callee_mem_a(%buf : memref<2x3xf32>) -> () attributes {config.pureHostCompileFunc} {
  return
}

func.func @callee_mem_b(%lhs : memref<2x3xf32>, %rhs : memref<2x3xf32>) -> () attributes {config.pureHostCompileFunc} {
  return
}

func.func @caller_mem_chain(%arg0 : memref<2x3xf32>, %arg1 : memref<2x3xf32>) -> () attributes {config.pureHostCompileFunc} {
  func.call @callee_mem_a(%arg0) : (memref<2x3xf32>) -> ()
  func.call @callee_mem_b(%arg0, %arg1) : (memref<2x3xf32>, memref<2x3xf32>) -> ()
  func.call @callee_mem_a(%arg1) : (memref<2x3xf32>) -> ()
  return
}

// CHECK:  bytecode.func_section @func_section {
// CHECK:    bytecode.ext.func @callee_mem_a (memref<2x3xf32>) -> () {
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:    bytecode.ext.func @callee_mem_b (memref<2x3xf32>, memref<2x3xf32>) -> () {
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:    bytecode.ext.func @caller_mem_chain (memref<2x3xf32>, memref<2x3xf32>) -> () {
// CHECK-DAG:  [[MEM_ARG1:%.+]] = bytecode.virtual_parameter_register 1
// CHECK-DAG:  [[MEM_ARG0:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:      [[FUNC_IDX_A0:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[FUNC_IDX_A0]], 0
// CHECK:      bytecode.call [[FUNC_IDX_A0]], results(), args([[MEM_ARG0]] : !bytecode.Register)
// CHECK:      [[FUNC_IDX_B:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[FUNC_IDX_B]], 1
// CHECK:      bytecode.call [[FUNC_IDX_B]], results(), args([[MEM_ARG0]], [[MEM_ARG1]] : !bytecode.Register, !bytecode.Register)
// CHECK:      [[FUNC_IDX_A1:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[FUNC_IDX_A1]], 0
// CHECK:      bytecode.call [[FUNC_IDX_A1]], results(), args([[MEM_ARG1]] : !bytecode.Register)
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:  }

}

// -----

module {

func.func @callee_dep_a(%arg0 : i64) -> i64 attributes {config.pureHostCompileFunc} {
  %c1 = arith.constant 1 : i64
  %sum = arith.addi %arg0, %c1 : i64
  return %sum : i64
}

func.func @callee_dep_b(%arg0 : i64) -> i64 attributes {config.pureHostCompileFunc} {
  %c2 = arith.constant 2 : i64
  %prod = arith.muli %arg0, %c2 : i64
  return %prod : i64
}

func.func @caller_chain_dep(%arg0 : i64, %arg1 : i64) -> i64 attributes {config.pureHostCompileFunc} {
  %a0 = func.call @callee_dep_a(%arg0) : (i64) -> i64
  %mix = arith.addi %a0, %arg1 : i64
  %b = func.call @callee_dep_b(%mix) : (i64) -> i64
  %a1 = func.call @callee_dep_a(%b) : (i64) -> i64
  return %a1 : i64
}

// CHECK:  bytecode.func_section @func_section {
// CHECK:    bytecode.ext.func @callee_dep_a (i64) -> i64 {
// CHECK-DAG:  [[A_ARG:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:      [[A_C1:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[A_C1]], 1
// CHECK:      [[A_SUM:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.add.i64 [[A_SUM]], [[A_ARG]], [[A_C1]]
// CHECK:      bytecode.retv [[A_SUM]]
// CHECK:    }
// CHECK:    bytecode.ext.func @callee_dep_b (i64) -> i64 {
// CHECK-DAG:  [[B_ARG:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:      [[B_C2:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[B_C2]], 2
// CHECK:      [[B_PROD:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.mul.i64 [[B_PROD]], [[B_ARG]], [[B_C2]]
// CHECK:      bytecode.retv [[B_PROD]]
// CHECK:    }
// CHECK:    bytecode.ext.func @caller_chain_dep (i64, i64) -> i64 {
// CHECK-DAG:  [[CHAIN_ARG1:%.+]] = bytecode.virtual_parameter_register 1
// CHECK-DAG:  [[CHAIN_ARG0:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:      [[FUNC_IDX_A0:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[FUNC_IDX_A0]], 0
// CHECK:      [[A0_RES:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.call [[FUNC_IDX_A0]], results([[A0_RES]] : !bytecode.Register), args([[CHAIN_ARG0]] : !bytecode.Register)
// CHECK:      [[MIX:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.add.i64 [[MIX]], [[A0_RES]], [[CHAIN_ARG1]]
// CHECK:      [[FUNC_IDX_B:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[FUNC_IDX_B]], 1
// CHECK:      [[B_RES:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.call [[FUNC_IDX_B]], results([[B_RES]] : !bytecode.Register), args([[MIX]] : !bytecode.Register)
// CHECK:      [[FUNC_IDX_A1:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[FUNC_IDX_A1]], 0
// CHECK:      [[A1_RES:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.call [[FUNC_IDX_A1]], results([[A1_RES]] : !bytecode.Register), args([[B_RES]] : !bytecode.Register)
// CHECK:      bytecode.retv [[A1_RES]]
// CHECK:    }
// CHECK:  }

}

// -----

// CHECK-LABEL: bytecode.ext.func @test_arith_index_cast
module {
    func.func @test_arith_index_cast(%main: memref<1x16x?x1280xf32>) -> () attributes {config.pureHostCompileFunc} {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 16 : i64
    %Convert_11 = memref.dim %main, %c2 : memref<1x16x?x1280xf32>
    %Convert_11_3 = arith.index_cast %Convert_11 : index to i64
    %add = arith.addi %c1, %Convert_11_3 : i64
    return
  }
  // CHECK:      [[ARG0:%.+]] = bytecode.virtual_parameter_register 0
  // CHECK:      [[REG0:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.set_imm [[REG0]], 2
  // CHECK:      [[REG1:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.set_imm [[REG1]], 16
  // CHECK:      [[REG2:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.buffer.get_dim [[REG2]], [[ARG0]], [[REG0]]
  // CHECK-NOT:  arith.index_cast
  // CHECK:      [[REG3:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.add.i64 [[REG3]], [[REG2]], [[REG1]]
  // CHECK:      bytecode.ret
}

// -----

// CHECK-LABEL: bytecode.ext.func @test_cond_br_no_args
module {
  func.func @test_cond_br_no_args(%cond: i1) -> (index) attributes {config.pureHostCompileFunc} {
    cf.br ^check
  ^true_dest:
    %c1 = arith.constant 1 : index
    return %c1 : index
  ^false_dest:
    %c2 = arith.constant 2 : index
    return %c2 : index
  ^check:
    cf.cond_br %cond, ^true_dest, ^false_dest
  }
}

// CHECK:      [[ARG:%.+]] = bytecode.virtual_parameter_register
// CHECK:      [[ONE:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT: bytecode.set_imm [[ONE]], 1
// CHECK:      bytecode.jmp ^bb3
// CHECK-NOT:  cf.br
// CHECK:      ^bb1:
// CHECK:      [[TRUE_RET:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[TRUE_RET]], 1
// CHECK:      bytecode.retv [[TRUE_RET]]
// CHECK:      ^bb2:
// CHECK:      [[FALSE_RET:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[FALSE_RET]], 2
// CHECK:      bytecode.retv [[FALSE_RET]]
// CHECK:      ^bb3:
// CHECK-NEXT: bytecode.je [[ARG]], [[ONE]], ^bb5, ^bb4
// CHECK-NOT:  cf.cond_br
// CHECK:      ^bb4:
// CHECK-NEXT: bytecode.jmp ^bb2
// CHECK:      ^bb5:
// CHECK-NEXT: bytecode.jmp ^bb1

// -----

// CHECK-LABEL: bytecode.ext.func @test_cond_br_both_args
module {
  func.func @test_cond_br_both_args(%cond: i1, %a: index, %b: index) -> index attributes {config.pureHostCompileFunc} {
    cf.br ^check
  ^true_dest(%x: index):
    return %x : index
  ^false_dest(%y: index):
    return %y : index
  ^check:
    cf.cond_br %cond, ^true_dest(%a : index), ^false_dest(%b : index)
  }
}

// CHECK-DAG:  [[CANON_X:%.+]] = bytecode.virtual_general_register
// CHECK-DAG:  [[CANON_Y:%.+]] = bytecode.virtual_general_register
// CHECK:      [[ONE:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT: bytecode.set_imm [[ONE]], 1
// CHECK:      bytecode.jmp ^bb3
// CHECK-NOT:  cf.br
// CHECK:      ^bb1:
// CHECK:      bytecode.retv [[CANON_X]]
// CHECK:      ^bb2:
// CHECK:      bytecode.retv [[CANON_Y]]
// CHECK:      ^bb3:
// CHECK-NEXT: bytecode.je {{%.+}}, [[ONE]], ^bb5, ^bb4
// CHECK-NOT:  cf.cond_br
// CHECK:      ^bb4:
// CHECK:      bytecode.set [[CANON_Y]], {{%.+}}
// CHECK-NEXT: bytecode.jmp ^bb2
// CHECK:      ^bb5:
// CHECK:      bytecode.set [[CANON_X]], {{%.+}}
// CHECK-NEXT: bytecode.jmp ^bb1

// -----

// CHECK-LABEL: bytecode.ext.func @test_cond_br_entry_block
module {
  func.func @test_cond_br_entry_block(%cond: i1) -> index attributes {config.pureHostCompileFunc} {
    cf.cond_br %cond, ^true_dest, ^false_dest
  ^true_dest:
    %c1 = arith.constant 1 : index
    return %c1 : index
  ^false_dest:
    %c2 = arith.constant 2 : index
    return %c2 : index
  }
}

// CHECK:      [[ARG:%.+]] = bytecode.virtual_parameter_register
// CHECK:      [[ONE:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT: bytecode.set_imm [[ONE]], 1
// CHECK-NEXT: bytecode.je [[ARG]], [[ONE]], ^bb2, ^bb1
// CHECK-NOT:  cf.cond_br
// CHECK:      ^bb1:
// CHECK-NEXT: bytecode.jmp ^bb4
// CHECK:      ^bb2:
// CHECK-NEXT: bytecode.jmp ^bb3
// CHECK:      ^bb3:
// CHECK:      [[TRUE_RET:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[TRUE_RET]], 1
// CHECK:      bytecode.retv [[TRUE_RET]]
// CHECK:      ^bb4:
// CHECK:      [[FALSE_RET:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[FALSE_RET]], 2
// CHECK:      bytecode.retv [[FALSE_RET]]

// -----

// CHECK-LABEL: bytecode.ext.func @test_cond_br_loop_aliasing
module {
  func.func @test_cond_br_loop_aliasing(%cond: i1, %init_i: index, %init_sum: index) -> index
      attributes {config.pureHostCompileFunc} {
    cf.br ^loop(%init_i, %init_sum : index, index)
  ^loop(%i: index, %sum: index):
    %c1 = arith.constant 1 : index
    %new_i = arith.addi %i, %c1 : index
    %new_sum = arith.addi %sum, %i : index
    cf.cond_br %cond, ^loop(%new_i, %new_sum : index, index), ^done(%sum : index)
  ^done(%result: index):
    return %result : index
  }
}

// CHECK-DAG:   [[CANON_I:%.+]] = bytecode.virtual_general_register
// CHECK-DAG:   [[CANON_SUM:%.+]] = bytecode.virtual_general_register
// CHECK-DAG:   [[CANON_RESULT:%.+]] = bytecode.virtual_general_register
// CHECK:       [[ONE:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT:  bytecode.set_imm [[ONE]], 1
// CHECK:       bytecode.set [[CANON_I]], {{%.+}}
// CHECK-NEXT:  bytecode.set [[CANON_SUM]], {{%.+}}
// CHECK-NEXT:  bytecode.jmp ^bb1
// CHECK-NOT:   cf.br
// CHECK:       ^bb1:
// CHECK:       [[VGR_C1:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT:  bytecode.set_imm [[VGR_C1]], 1
// CHECK:       [[VGR_NEW_I:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT:  bytecode.add.i64 [[VGR_NEW_I]], [[CANON_I]], [[VGR_C1]]
// CHECK:       [[VGR_NEW_SUM:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT:  bytecode.add.i64 [[VGR_NEW_SUM]], [[CANON_SUM]], [[CANON_I]]
// CHECK-NEXT:  bytecode.je {{%.+}}, [[ONE]], ^bb3, ^bb2
// CHECK-NOT:   cf.cond_br
// CHECK:       ^bb2:
// CHECK-NEXT:  bytecode.set [[CANON_RESULT]], [[CANON_SUM]]
// CHECK-NEXT:  bytecode.jmp ^bb4
// CHECK:       ^bb3:
// CHECK-NEXT:  bytecode.set [[CANON_I]], [[VGR_NEW_I]]
// CHECK-NEXT:  bytecode.set [[CANON_SUM]], [[VGR_NEW_SUM]]
// CHECK-NEXT:  bytecode.jmp ^bb1
// CHECK:       ^bb4:
// CHECK-NEXT:  bytecode.retv [[CANON_RESULT]]

// -----

// CHECK-LABEL: bytecode.ext.func @test_switch_no_cases
module {
  func.func @test_switch_no_cases(%flag: i32) -> index attributes {config.pureHostCompileFunc} {
    cf.switch %flag : i32, [
      default: ^bb_default
    ]
  ^bb_default:
    %c0 = arith.constant 0 : index
    return %c0 : index
  }
}

// CHECK:      bytecode.jmp ^bb1
// CHECK-NOT:  cf.switch
// CHECK:      ^bb1:
// CHECK:      [[C0:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT: bytecode.set_imm [[C0]], 0
// CHECK-NEXT: bytecode.retv [[C0]]

// -----

// CHECK-LABEL: bytecode.ext.func @test_switch_one_case
module {
  func.func @test_switch_one_case(%flag: i32) -> index attributes {config.pureHostCompileFunc} {
    cf.switch %flag : i32, [
      default: ^bb_default,
      42: ^bb_case
    ]
  ^bb_case:
    %c1 = arith.constant 1 : index
    return %c1 : index
  ^bb_default:
    %c0 = arith.constant 0 : index
    return %c0 : index
  }
}

// CHECK:      [[FLAG:%.+]] = bytecode.virtual_parameter_register
// CHECK:      [[COMP_REG:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[COMP_REG]], 42
// CHECK-NEXT: bytecode.je [[FLAG]], [[COMP_REG]], ^bb2, ^bb1
// CHECK-NOT:  cf.switch
// CHECK:      ^bb1:
// CHECK-NEXT: bytecode.jmp ^bb4
// CHECK:      ^bb2:
// CHECK-NEXT: bytecode.jmp ^bb3
// CHECK:      ^bb3:
// CHECK:      [[C1:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT: bytecode.set_imm [[C1]], 1
// CHECK-NEXT: bytecode.retv [[C1]]
// CHECK:      ^bb4:
// CHECK:      [[C0:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT: bytecode.set_imm [[C0]], 0
// CHECK-NEXT: bytecode.retv [[C0]]

// -----

// CHECK-LABEL: bytecode.ext.func @test_switch_three_cases
module {
  func.func @test_switch_three_cases(%flag: i32) -> index attributes {config.pureHostCompileFunc} {
    cf.switch %flag : i32, [
      default: ^bb_default,
      10: ^bb0,
      20: ^bb1,
      30: ^bb2
    ]
  ^bb0:
    %c10 = arith.constant 10 : index
    return %c10 : index
  ^bb1:
    %c20 = arith.constant 20 : index
    return %c20 : index
  ^bb2:
    %c30 = arith.constant 30 : index
    return %c30 : index
  ^bb_default:
    %c0 = arith.constant 0 : index
    return %c0 : index
  }
}

// CHECK:      [[FLAG:%.+]] = bytecode.virtual_parameter_register
// CHECK:      [[COMP_REG:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[COMP_REG]], 10
// CHECK-NEXT: bytecode.je [[FLAG]], [[COMP_REG]], ^bb4, ^bb1
// CHECK-NOT:  cf.switch
// CHECK:      ^bb1:
// CHECK-NEXT: bytecode.set_imm [[COMP_REG]], 20
// CHECK-NEXT: bytecode.je [[FLAG]], [[COMP_REG]], ^bb5, ^bb2
// CHECK:      ^bb2:
// CHECK-NEXT: bytecode.set_imm [[COMP_REG]], 30
// CHECK-NEXT: bytecode.je [[FLAG]], [[COMP_REG]], ^bb6, ^bb3
// CHECK:      ^bb3:
// CHECK-NEXT: bytecode.jmp ^bb10
// CHECK:      ^bb4:
// CHECK-NEXT: bytecode.jmp ^bb7
// CHECK:      ^bb5:
// CHECK-NEXT: bytecode.jmp ^bb8
// CHECK:      ^bb6:
// CHECK-NEXT: bytecode.jmp ^bb9
// CHECK:      ^bb7:
// CHECK:      [[C10:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT: bytecode.set_imm [[C10]], 10
// CHECK-NEXT: bytecode.retv [[C10]]
// CHECK:      ^bb8:
// CHECK:      [[C20:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT: bytecode.set_imm [[C20]], 20
// CHECK-NEXT: bytecode.retv [[C20]]
// CHECK:      ^bb9:
// CHECK:      [[C30:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT: bytecode.set_imm [[C30]], 30
// CHECK-NEXT: bytecode.retv [[C30]]
// CHECK:      ^bb10:
// CHECK:      [[C0:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT: bytecode.set_imm [[C0]], 0
// CHECK-NEXT: bytecode.retv [[C0]]

// -----

// CHECK-LABEL: bytecode.ext.func @test_switch_with_args
module {
  func.func @test_switch_with_args(%flag: i32, %val: index) -> index attributes {config.pureHostCompileFunc} {
    cf.switch %flag : i32, [
      default: ^bb_default(%val : index),
      42: ^bb_case(%val : index)
    ]
  ^bb_case(%x: index):
    return %x : index
  ^bb_default(%y: index):
    return %y : index
  }
}

// CHECK:      [[VAL:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:      [[FLAG:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:      [[CANON_X:%.+]] = bytecode.virtual_general_register
// CHECK:      [[CANON_Y:%.+]] = bytecode.virtual_general_register
// CHECK:      [[COMP_REG:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[COMP_REG]], 42
// CHECK-NEXT: bytecode.je [[FLAG]], [[COMP_REG]], ^bb2, ^bb1
// CHECK-NOT:  cf.switch
// CHECK:      ^bb1:
// CHECK-NEXT: bytecode.set [[CANON_Y]], [[VAL]]
// CHECK-NEXT: bytecode.jmp ^bb4
// CHECK:      ^bb2:
// CHECK-NEXT: bytecode.set [[CANON_X]], [[VAL]]
// CHECK-NEXT: bytecode.jmp ^bb3
// CHECK:      ^bb3:
// CHECK-NEXT: bytecode.retv [[CANON_X]]
// CHECK:      ^bb4:
// CHECK-NEXT: bytecode.retv [[CANON_Y]]

// -----

// CHECK-LABEL: bytecode.ext.func @test_assert
module {
  func.func @test_assert(%cond: i1) -> () attributes {config.pureHostCompileFunc} {
    cf.assert %cond, "condition must hold"
    return
  }
}

// CHECK:      [[FLAG:%.+]] = bytecode.virtual_parameter_register
// CHECK:      bytecode.ext.assert [[FLAG]], "condition must hold"
// CHECK-NOT:  cf.assert
// CHECK:      bytecode.ret

// -----

// CHECK-LABEL: bytecode.ext.func @memref_cast_op

module {
func.func @memref_cast_op(%Parameter_10: memref<1x16x?x1280xf32>, %main: memref<1x16x?x1280xf16>) -> memref<1x16x?x1280xf16> attributes {HostExec.HostCompileInferenceExec, config.pureHostCompileFunc} {
    %c2 = arith.constant 2 : index
    %Convert_11 = memref.dim %Parameter_10, %c2 : memref<1x16x?x1280xf32>
    %Convert_11_4 = arith.constant 0 : index
    %Convert_11_6 = arith.constant -31 : index
    %Convert_11_7 = arith.addi %Convert_11, %Convert_11_6 : index
    %Convert_11_8 = arith.minsi %Convert_11_7, %Convert_11_4 : index
    %Convert_11_9 = memref.subview %Parameter_10[0, 0, %Convert_11_8, 0] [1, 16, 31, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf32> to memref<1x16x31x1280xf32, strided<[?, ?, 1280, 1], offset: ?>>
    %Convert_11_10 = memref.cast %Convert_11_9 : memref<1x16x31x1280xf32, strided<[?, ?, 1280, 1], offset: ?>> to memref<1x16x31x1280xf32, strided<[?, ?, ?, ?], offset: ?>>
    %Convert_11_11 = memref.subview %main[0, 0, %Convert_11_8, 0] [1, 16, 31, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf16> to memref<1x16x31x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
    %Convert_11_12 = memref.cast %Convert_11_11 : memref<1x16x31x1280xf16, strided<[?, ?, 1280, 1], offset: ?>> to memref<1x16x31x1280xf16, strided<[?, ?, ?, ?], offset: ?>>
    return %main : memref<1x16x?x1280xf16>
  }
}

// CHECK:      [[REG0:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.buffer.subview
// CHECK-NOT:  memref.cast
// CHECK:      bytecode.buffer.subview
// CHECK-NOT:  memref.cast

// -----

// test no compilation error when entry point is not seen during bytecode conversion

// CHECK-LABEL: bytecode.ext.func @test_network_info_main

module {
  bytecode.kernel_section @kernel_section {
    bytecode.kernel @main_func0_static "\7FELF\02\01"
  }
  net.NetworkInfo entryPoint : @test_network_info_main inputsInfo : {
    DataInfo "Parameter_10" : tensor<1x16x?x1280xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>}>
  } outputsInfo : {
    DataInfo "Convert_11" friendlyName = "Result_12" : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>}>
  }
  func.func @test_network_info_main(%Parameter_10: memref<1x16x?x1280xf32>, %main: memref<1x16x?x1280xf16>) -> memref<1x16x?x1280xf16> attributes {config.pureHostCompileFunc} {
    return %main : memref<1x16x?x1280xf16>
  }
  // CHECK-NOT: error:
}

// -----

// CHECK-LABEL: bytecode.ext.func @test_arith_extsi
module {
  func.func @test_arith_extsi(%arg0: i8) -> () attributes {config.pureHostCompileFunc} {
    %ext = arith.extsi %arg0 : i8 to i64
    return
  }
  // arith.extsi is a no-op: registers are always sign-extended to 64 bits.
  // The unused parameter register is not emitted either.
  // CHECK-NOT:  bytecode.convert
  // CHECK:      bytecode.ret
}

// -----

// CHECK-LABEL: bytecode.ext.func @test_arith_extsi_i1
module {
  func.func @test_arith_extsi_i1(%arg0: i1) -> (i64) attributes {config.pureHostCompileFunc} {
    %ext = arith.extsi %arg0 : i1 to i64
    return %ext : i64
  }
  // i1 values are stored as 0/1 (zero-extended). extsi i1 must produce 0/-1: emit 0 - val.
  // CHECK:      [[ARG:%.+]] = bytecode.virtual_parameter_register 0
  // CHECK:      [[ZERO:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.set_imm [[ZERO]], 0
  // CHECK:      [[RES:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.sub.i64 [[RES]], [[ZERO]], [[ARG]]
  // CHECK:      bytecode.retv [[RES]]
}

// -----

// CHECK-LABEL: bytecode.ext.func @test_arith_trunci
module {
  func.func @test_arith_trunci(%arg0: i64, %arg1: i32) -> () attributes {config.pureHostCompileFunc} {
    %t8  = arith.trunci %arg0 : i64 to i8
    %t16 = arith.trunci %arg0 : i64 to i16
    %t32 = arith.trunci %arg0 : i64 to i32
    %t8b = arith.trunci %arg1 : i32 to i8
    return
  }
  // CHECK:      [[ARG1:%.+]] = bytecode.virtual_parameter_register 1
  // CHECK:      [[ARG0:%.+]] = bytecode.virtual_parameter_register 0
  // CHECK:      [[R0:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.convert.i64toi8 [[R0]], [[ARG0]]
  // CHECK:      [[R1:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.convert.i64toi16 [[R1]], [[ARG0]]
  // CHECK:      [[R2:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.convert.i64toi32 [[R2]], [[ARG0]]
  // CHECK:      [[R3:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.convert.i32toi8 [[R3]], [[ARG1]]
  // CHECK:      bytecode.ret
}

// -----

// CHECK-LABEL: bytecode.ext.func @test_arith_sitofp
module {
  func.func @test_arith_sitofp(%arg0: i32, %arg1: i64) -> () attributes {config.pureHostCompileFunc} {
    %f32a = arith.sitofp %arg0 : i32 to f32
    %f64a = arith.sitofp %arg0 : i32 to f64
    %f32b = arith.sitofp %arg1 : i64 to f32
    %f64b = arith.sitofp %arg1 : i64 to f64
    return
  }
  // CHECK:      [[ARG1:%.+]] = bytecode.virtual_parameter_register 1
  // CHECK:      [[ARG0:%.+]] = bytecode.virtual_parameter_register 0
  // CHECK:      [[R0:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.convert.i32tof32 [[R0]], [[ARG0]]
  // CHECK:      [[R1:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.convert.i32tof64 [[R1]], [[ARG0]]
  // CHECK:      [[R2:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.convert.i64tof32 [[R2]], [[ARG1]]
  // CHECK:      [[R3:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.convert.i64tof64 [[R3]], [[ARG1]]
  // CHECK:      bytecode.ret
}

// -----

// CHECK-LABEL: bytecode.ext.func @test_arith_fptosi
module {
  func.func @test_arith_fptosi(%arg0: f32, %arg1: f64) -> () attributes {config.pureHostCompileFunc} {
    %i32a = arith.fptosi %arg0 : f32 to i32
    %i64a = arith.fptosi %arg0 : f32 to i64
    %i32b = arith.fptosi %arg1 : f64 to i32
    %i64b = arith.fptosi %arg1 : f64 to i64
    return
  }
  // CHECK:      [[ARG1:%.+]] = bytecode.virtual_parameter_register 1
  // CHECK:      [[ARG0:%.+]] = bytecode.virtual_parameter_register 0
  // CHECK:      [[R0:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.convert.f32toi32 [[R0]], [[ARG0]]
  // CHECK:      [[R1:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.convert.f32toi64 [[R1]], [[ARG0]]
  // CHECK:      [[R2:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.convert.f64toi32 [[R2]], [[ARG1]]
  // CHECK:      [[R3:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.convert.f64toi64 [[R3]], [[ARG1]]
  // CHECK:      bytecode.ret
}

// -----

// CHECK-LABEL: bytecode.ext.func @test_arith_extf_truncf
module {
  func.func @test_arith_extf_truncf(%arg0: f32, %arg1: f64) -> () attributes {config.pureHostCompileFunc} {
    %ext = arith.extf %arg0 : f32 to f64
    %trunc = arith.truncf %arg1 : f64 to f32
    return
  }
  // CHECK:      [[ARG1:%.+]] = bytecode.virtual_parameter_register 1
  // CHECK:      [[ARG0:%.+]] = bytecode.virtual_parameter_register 0
  // CHECK:      [[R0:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.convert.f32tof64 [[R0]], [[ARG0]]
  // CHECK:      [[R1:%.+]] = bytecode.virtual_general_register
  // CHECK:      bytecode.convert.f64tof32 [[R1]], [[ARG1]]
  // CHECK:      bytecode.ret
}

// -----

module @test_preprocessing_single_group {
  bytecode.kernel_section @kernel_section {
    bytecode.kernel @compute "\00\01\02\03"
  }
  module @SubModule {
    func.func nested @compute(memref<16xf32>, memref<16xf32>) -> ()
  }
  func.func @main(%arg0: memref<16xf32>, %arg1: memref<16xf32>) -> ()
      attributes {config.pureHostCompileFunc, HostExec.HostCompileInferenceExec,
                  disable_pipelined_cmdlist_recording = true} {
    %size = arith.constant 1 : index
    %group = async.create_group %size : !async.group
    %token = async.execute {
      Core.NestedCall @SubModule::@compute(%arg0, %arg1) : (memref<16xf32>, memref<16xf32>) -> ()
      async.yield
    }
    %rank = async.add_to_group %token, %group : !async.token
    async.await_all %group
    return
  }
}

// CHECK-LABEL: module @test_preprocessing_single_group
// CHECK:    bytecode.ext.func @main (memref<16xf32>, memref<16xf32>) -> () {
// CHECK:      [[PARAM1:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:      [[PARAM0:%.+]] = bytecode.virtual_parameter_register 0
// Injected create_group at entry: arith.constant 1 (set_imm), then cmd_list.create.
// CHECK:      [[SIZE0:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[SIZE0]], 1
// CHECK:      [[CMD_LIST:%.+]] = bytecode.virtual_general_register
// Existing create_group has no_reset_cmdlist → assert exactly one cmd_list.create in the chunk.
// CHECK-COUNT-1: bytecode.cmd_list.create [[CMD_LIST]]
// CHECK-NOT:     bytecode.cmd_list.create
// Existing arith.constant 1 for original group size.
// CHECK:      bytecode.set_imm {{%.+}}, 1
// Kernel is submitted through the existing create_group (reuses CMD_LIST).
// CHECK:      [[KERNEL:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.kernel.create [[KERNEL]], @compute, inputs([[PARAM0]], [[PARAM1]]), outputs()
// CHECK:      bytecode.cmd_list.add_kernel [[CMD_LIST]], [[KERNEL]], (), ()
// Existing await_all is noop → no close/exec emitted for it.
// Injected await_all (no attributes) → emits close + exec.
// CHECK:      bytecode.cmd_list.close [[CMD_LIST]]
// CHECK:      bytecode.cmd_list.exec [[CMD_LIST]]
// CHECK:      bytecode.ret
// CHECK:    }

// -----

module @test_preprocessing_multiple_groups {
  bytecode.kernel_section @kernel_section {
    bytecode.kernel @computeA "\00\01\02\03"
    bytecode.kernel @computeB "\04\05\06\07"
  }
  module @SubModule {
    func.func nested @computeA(memref<16xf32>, memref<16xf32>) -> ()
    func.func nested @computeB(memref<16xf32>, memref<16xf32>) -> ()
  }
  func.func @main(%arg0: memref<16xf32>, %arg1: memref<16xf32>) -> ()
      attributes {config.pureHostCompileFunc, HostExec.HostCompileInferenceExec,
                  disable_pipelined_cmdlist_recording = true} {
    %size = arith.constant 1 : index
    %groupA = async.create_group %size : !async.group
    %tokenA = async.execute {
      Core.NestedCall @SubModule::@computeA(%arg0, %arg1) : (memref<16xf32>, memref<16xf32>) -> ()
      async.yield
    }
    %rankA = async.add_to_group %tokenA, %groupA : !async.token
    async.await_all %groupA

    %groupB = async.create_group %size : !async.group
    %tokenB = async.execute {
      Core.NestedCall @SubModule::@computeB(%arg0, %arg1) : (memref<16xf32>, memref<16xf32>) -> ()
      async.yield
    }
    %rankB = async.add_to_group %tokenB, %groupB : !async.token
    async.await_all %groupB
    return
  }
}

// CHECK-LABEL: module @test_preprocessing_multiple_groups
// CHECK:    bytecode.ext.func @main (memref<16xf32>, memref<16xf32>) -> () {
// CHECK:      [[PARAM1:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:      [[PARAM0:%.+]] = bytecode.virtual_parameter_register 0
// Injected create_group at entry: arith.constant 1 (set_imm), then cmd_list.create.
// CHECK:      [[SIZE0:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[SIZE0]], 1
// CHECK:      [[CMD_LIST:%.+]] = bytecode.virtual_general_register
// Both existing create_groups have no_reset_cmdlist → assert exactly one cmd_list.create in the chunk.
// CHECK-COUNT-1: bytecode.cmd_list.create [[CMD_LIST]]
// CHECK-NOT:     bytecode.cmd_list.create
// Existing arith.constant 1 for groupA size, then kernel A submitted (reuses CMD_LIST).
// CHECK:      bytecode.set_imm {{%.+}}, 1
// CHECK:      [[KERNEL_A:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.kernel.create [[KERNEL_A]], @computeA, inputs([[PARAM0]], [[PARAM1]]), outputs()
// CHECK:      bytecode.cmd_list.add_kernel [[CMD_LIST]], [[KERNEL_A]], (), ()
// First await_all is "barrier" → erased (no close/exec per E-221988).
// groupB's arith.constant 1 is CSE'd with groupA's (same value), so no second set_imm.
// Kernel B submitted (reuses CMD_LIST).
// CHECK:      [[KERNEL_B:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.kernel.create [[KERNEL_B]], @computeB, inputs([[PARAM0]], [[PARAM1]]), outputs()
// CHECK:      bytecode.cmd_list.add_kernel [[CMD_LIST]], [[KERNEL_B]], (), ()
// Second await_all is "noop" → erased. Injected final await_all emits close + exec.
// CHECK:      bytecode.cmd_list.close [[CMD_LIST]]
// CHECK:      bytecode.cmd_list.exec [[CMD_LIST]]
// CHECK:      bytecode.ret
// CHECK:    }
