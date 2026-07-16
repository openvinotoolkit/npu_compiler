//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

bytecode.func_section @function_section {
    bytecode.ext.func @jmp_test () -> () {
        bytecode.jmp ^target
      ^target:
        bytecode.ret
    }
}

// CHECK-LABEL: bytecode.ext.func @jmp_test
// CHECK:         bytecode.jmp ^[[TARGET:[a-zA-Z0-9_]+]]
// CHECK:       ^[[TARGET]]:
// CHECK:         bytecode.ret

// -----

bytecode.func_section @function_section_je {
    bytecode.ext.func @je_test () -> () {
        %lhs = bytecode.virtual_general_register
        %rhs = bytecode.virtual_general_register
        bytecode.je %lhs, %rhs, ^trueDest, ^falseDest
      ^falseDest:
        bytecode.ret
      ^trueDest:
        bytecode.ret
    }
}

// CHECK-LABEL: bytecode.ext.func @je_test
// CHECK:         [[LHS:%.+]] = bytecode.virtual_general_register
// CHECK:         [[RHS:%.+]] = bytecode.virtual_general_register
// CHECK:         bytecode.je [[LHS]], [[RHS]], ^[[TRUE:[a-zA-Z0-9_]+]], ^[[FALSE:[a-zA-Z0-9_]+]]
// CHECK:       ^[[FALSE]]:
// CHECK:         bytecode.ret
// CHECK:       ^[[TRUE]]:
// CHECK:         bytecode.ret

// -----

bytecode.func_section @function_section_jne {
    bytecode.ext.func @jne_test () -> () {
        %lhs = bytecode.virtual_general_register
        %rhs = bytecode.virtual_general_register
        bytecode.jne %lhs, %rhs, ^trueDest, ^falseDest
      ^falseDest:
        bytecode.ret
      ^trueDest:
        bytecode.ret
    }
}

// CHECK-LABEL: bytecode.ext.func @jne_test
// CHECK:         [[LHS:%.+]] = bytecode.virtual_general_register
// CHECK:         [[RHS:%.+]] = bytecode.virtual_general_register
// CHECK:         bytecode.jne [[LHS]], [[RHS]], ^[[TRUE:[a-zA-Z0-9_]+]], ^[[FALSE:[a-zA-Z0-9_]+]]
// CHECK:       ^[[FALSE]]:
// CHECK:         bytecode.ret
// CHECK:       ^[[TRUE]]:
// CHECK:         bytecode.ret
