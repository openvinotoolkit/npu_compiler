//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

bytecode.func_section @function_section {
    bytecode.ext.func @and (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 1
        %rhs = bytecode.virtual_parameter_register 2
        bytecode.and.64 %dst, %lhs, %rhs
        bytecode.ret
    }
}

// CHECK-LABEL: bytecode.ext.func @and
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.and.64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// -----

bytecode.func_section @function_section {
    bytecode.ext.func @not (i64) -> () {
        %dst = bytecode.virtual_general_register
        %src = bytecode.virtual_parameter_register 1
        bytecode.not.64 %dst, %src
        bytecode.ret
    }
}

// CHECK-LABEL: bytecode.ext.func @not
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[SRC:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         bytecode.not.64 [[DST]], [[SRC]]
// CHECK:         bytecode.ret

// -----

bytecode.func_section @function_section {
    bytecode.ext.func @or (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 1
        %rhs = bytecode.virtual_parameter_register 2
        bytecode.or.64 %dst, %lhs, %rhs
        bytecode.ret
    }
}

// CHECK-LABEL: bytecode.ext.func @or
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.or.64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// -----

bytecode.func_section @function_section {
    bytecode.ext.func @xor (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 1
        %rhs = bytecode.virtual_parameter_register 2
        bytecode.xor.64 %dst, %lhs, %rhs
        bytecode.ret
    }
}

// CHECK-LABEL: bytecode.ext.func @xor
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.xor.64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// -----

bytecode.func_section @function_section {
    bytecode.ext.func @sll (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 1
        %rhs = bytecode.virtual_parameter_register 2
        bytecode.sll.64 %dst, %lhs, %rhs
        bytecode.ret
    }
}

// CHECK-LABEL: bytecode.ext.func @sll
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.sll.64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// -----

bytecode.func_section @function_section {
    bytecode.ext.func @srl (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 1
        %rhs = bytecode.virtual_parameter_register 2
        bytecode.srl.64 %dst, %lhs, %rhs
        bytecode.ret
    }
}

// CHECK-LABEL: bytecode.ext.func @srl
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.srl.64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// -----

bytecode.func_section @function_section {
    bytecode.ext.func @sra (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 1
        %rhs = bytecode.virtual_parameter_register 2
        bytecode.sra.64 %dst, %lhs, %rhs
        bytecode.ret
    }
}

// CHECK-LABEL: bytecode.ext.func @sra
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.sra.64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret
