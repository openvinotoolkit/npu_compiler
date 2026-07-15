//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

bytecode.func_section @function_section {
    bytecode.ext.func @add (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 1
        %rhs = bytecode.virtual_parameter_register 2
        bytecode.add.i64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.ext.func @mul (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 1
        %rhs = bytecode.virtual_parameter_register 2
        bytecode.mul.i64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.ext.func @min (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 1
        %rhs = bytecode.virtual_parameter_register 2
        bytecode.min.i64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.ext.func @max (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 1
        %rhs = bytecode.virtual_parameter_register 2
        bytecode.max.i64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.ext.func @sub (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 1
        %rhs = bytecode.virtual_parameter_register 2
        bytecode.sub.i64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.ext.func @div (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 1
        %rhs = bytecode.virtual_parameter_register 2
        bytecode.div.i64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.ext.func @div_u (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 1
        %rhs = bytecode.virtual_parameter_register 2
        bytecode.div.u64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.ext.func @min_u (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 1
        %rhs = bytecode.virtual_parameter_register 2
        bytecode.min.u64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.ext.func @add_u (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 1
        %rhs = bytecode.virtual_parameter_register 2
        bytecode.add.u64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.ext.func @max_u (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 1
        %rhs = bytecode.virtual_parameter_register 2
        bytecode.max.u64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.ext.func @mul_u (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 1
        %rhs = bytecode.virtual_parameter_register 2
        bytecode.mul.u64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.ext.func @rem_u (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 1
        %rhs = bytecode.virtual_parameter_register 2
        bytecode.rem.u64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.ext.func @sub_u (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 1
        %rhs = bytecode.virtual_parameter_register 2
        bytecode.sub.u64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.ext.func @rem (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 1
        %rhs = bytecode.virtual_parameter_register 2
        bytecode.rem.i64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.ext.func @abs (i64) -> () {
        %dst = bytecode.virtual_general_register
        %src = bytecode.virtual_parameter_register 0
        bytecode.abs.i64 %dst, %src
        bytecode.ret
    }
    bytecode.ext.func @add_f (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 0
        %rhs = bytecode.virtual_parameter_register 1
        bytecode.add.f64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.ext.func @sub_f (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 0
        %rhs = bytecode.virtual_parameter_register 1
        bytecode.sub.f64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.ext.func @mul_f (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 0
        %rhs = bytecode.virtual_parameter_register 1
        bytecode.mul.f64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.ext.func @div_f (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 0
        %rhs = bytecode.virtual_parameter_register 1
        bytecode.div.f64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.ext.func @rem_f (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 0
        %rhs = bytecode.virtual_parameter_register 1
        bytecode.rem.f64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.ext.func @max_f (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 0
        %rhs = bytecode.virtual_parameter_register 1
        bytecode.max.f64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.ext.func @min_f (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 0
        %rhs = bytecode.virtual_parameter_register 1
        bytecode.min.f64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.ext.func @abs_f (i64) -> () {
        %dst = bytecode.virtual_general_register
        %src = bytecode.virtual_parameter_register 0
        bytecode.abs.f64 %dst, %src
        bytecode.ret
    }
    bytecode.ext.func @neg_f (i64) -> () {
        %dst = bytecode.virtual_general_register
        %src = bytecode.virtual_parameter_register 0
        bytecode.neg.f64 %dst, %src
        bytecode.ret
    }
    bytecode.ext.func @ceil_f (i64) -> () {
        %dst = bytecode.virtual_general_register
        %src = bytecode.virtual_parameter_register 0
        bytecode.ceil.f64 %dst, %src
        bytecode.ret
    }
    bytecode.ext.func @floor_f (i64) -> () {
        %dst = bytecode.virtual_general_register
        %src = bytecode.virtual_parameter_register 0
        bytecode.floor.f64 %dst, %src
        bytecode.ret
    }
    bytecode.ext.func @round_f (i64) -> () {
        %dst = bytecode.virtual_general_register
        %src = bytecode.virtual_parameter_register 0
        bytecode.round.f64 %dst, %src, 2
        bytecode.ret
    }
}

// CHECK-LABEL: bytecode.ext.func @add
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.add.i64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @mul
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.mul.i64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @min
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.min.i64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @max
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.max.i64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @sub
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.sub.i64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @div
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.div.i64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @div_u
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.div.u64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @min_u
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.min.u64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @add_u
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.add.u64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @max_u
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.max.u64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @mul_u
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.mul.u64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @rem_u
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.rem.u64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @sub_u
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.sub.u64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @rem
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.rem.i64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @abs
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[SRC:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:         bytecode.abs.i64 [[DST]], [[SRC]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @add_f
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         bytecode.add.f64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @sub_f
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         bytecode.sub.f64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @mul_f
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         bytecode.mul.f64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @div_f
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         bytecode.div.f64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @rem_f
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         bytecode.rem.f64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @max_f
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         bytecode.max.f64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @min_f
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         bytecode.min.f64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @abs_f
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[SRC:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:         bytecode.abs.f64 [[DST]], [[SRC]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @neg_f
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[SRC:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:         bytecode.neg.f64 [[DST]], [[SRC]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @ceil_f
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[SRC:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:         bytecode.ceil.f64 [[DST]], [[SRC]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @floor_f
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[SRC:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:         bytecode.floor.f64 [[DST]], [[SRC]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @round_f
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[SRC:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:         bytecode.round.f64 [[DST]], [[SRC]], 2
// CHECK:         bytecode.ret
