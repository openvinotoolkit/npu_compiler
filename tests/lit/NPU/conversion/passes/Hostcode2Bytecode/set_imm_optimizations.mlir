//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform% compilation-mode=HostCompile_Interpreter allow-custom-values=true" \
// RUN:   --convert-hostcode-to-bytecode \
// RUN:   --inject-bytecode-metadata \
// RUN:   --convert-intermediate-bytecode-ops \
// RUN:   --deduplicate-and-lower-imm-register-ops \
// RUN:   %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

module @demo attributes {config.compilationMode = #config.compilation_mode<HostCompile_Interpreter>} {

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1xf32>
  }

  // CHECK-LABEL: bytecode.func @main
  func.func @main(%in: memref<1x1xf32>, %out: memref<1x1xf32>)
      attributes {HostExec.HostCompileInferenceExec, config.pureHostCompileFunc} {

    %c0   = arith.constant 0   : index
    %c1   = arith.constant 1   : index
    %c2   = arith.constant 2   : index
    %c342 = arith.constant 342 : index
    %group = async.create_group %c1 : !async.group
    cf.br ^loop(%c0 : index)
  ^loop(%iv: index):
    %cond = arith.cmpi slt, %iv, %c342 : index
    cf.cond_br %cond, ^body, ^exit
  ^body:
    %c0_b   = arith.constant 0   : index
    %c1_b   = arith.constant 1   : index
    %c2_b   = arith.constant 2   : index
    %c342_b = arith.constant 342 : index
    %step   = arith.muli  %iv,    %c2_b   : index
    %next   = arith.addi  %step,  %c1_b   : index
    %clamp  = arith.minsi %next,  %c342_b : index
    %offset = arith.addi  %clamp, %c0_b   : index
    cf.br ^loop(%offset : index)
  ^exit:
    async.await_all %group
    return
  }

// Entry-block constants (C1, C0, C342) are materialised once and reused in all blocks.
// C2 first appears in the loop body (the entry-block arith.constant 2 is dead and eliminated).
// CHECK:         {{%.+}} = bytecode.virtual_general_register
// CHECK:         [[C1:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT:    bytecode.set_imm [[C1]], 1
// CHECK:         [[C0:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT:    bytecode.set_imm [[C0]], 0
// CHECK:         [[C342:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT:    bytecode.set_imm [[C342]], 342

// CHECK:       ^bb1:
// CHECK:         bytecode.cmp.i64 {{%.+}}, {{%.+}}, [[C342]], 260
// CHECK:         bytecode.je {{%.+}}, [[C1]],

// C2 is first used in the loop body; C1 and C342 are reused from entry block (no new set_imm).
// CHECK:       ^bb4:
// CHECK-NEXT:    [[C2:%.+]] = bytecode.virtual_general_register
// CHECK-NEXT:    bytecode.set_imm [[C2]], 2
// CHECK:         bytecode.mul.i64 {{%.+}}, {{%.+}}, [[C2]]
// CHECK:         bytecode.add.i64 {{%.+}}, {{%.+}}, [[C1]]
// CHECK:         bytecode.min.i64 {{%.+}}, {{%.+}}, [[C342]]

// CHECK:       ^bb5:
// CHECK-NOT:   bytecode.set_imm
// CHECK:         bytecode.cmd_list.close
}
