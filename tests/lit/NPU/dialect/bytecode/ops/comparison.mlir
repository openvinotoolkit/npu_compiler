//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

bytecode.func_section @function_section {
    bytecode.ext.func @cmp (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 1
        %rhs = bytecode.virtual_parameter_register 2
        bytecode.cmp.i64 %dst, %lhs, %rhs, 260
        bytecode.cmp.f64 %dst, %lhs, %rhs, 2
        bytecode.ret
    }
}

// CHECK-LABEL: bytecode.ext.func @cmp
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.cmp.i64 [[DST]], [[LHS]], [[RHS]], 260
// CHECK:         bytecode.cmp.f64 [[DST]], [[LHS]], [[RHS]], 2
// CHECK:         bytecode.ret
