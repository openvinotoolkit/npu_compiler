//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

bytecode.func_section @function_section {
    bytecode.ext.func @convert_integer (i64) -> () {
        %dst = bytecode.virtual_general_register
        %src = bytecode.virtual_parameter_register 0
        bytecode.convert.i16toi8  %dst, %src
        bytecode.convert.i32toi8  %dst, %src
        bytecode.convert.i32toi16 %dst, %src
        bytecode.convert.i64toi8  %dst, %src
        bytecode.convert.i64toi16 %dst, %src
        bytecode.convert.i64toi32 %dst, %src
        bytecode.ret
    }
}

// CHECK-LABEL: bytecode.ext.func @convert_integer
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[SRC:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:         bytecode.convert.i16toi8  [[DST]], [[SRC]]
// CHECK:         bytecode.convert.i32toi8  [[DST]], [[SRC]]
// CHECK:         bytecode.convert.i32toi16 [[DST]], [[SRC]]
// CHECK:         bytecode.convert.i64toi8  [[DST]], [[SRC]]
// CHECK:         bytecode.convert.i64toi16 [[DST]], [[SRC]]
// CHECK:         bytecode.convert.i64toi32 [[DST]], [[SRC]]
// CHECK:         bytecode.ret

// -----

bytecode.func_section @function_section {
    bytecode.ext.func @convert_int_to_float (i64) -> () {
        %dst = bytecode.virtual_general_register
        %src = bytecode.virtual_parameter_register 0
        bytecode.convert.i8tof32  %dst, %src
        bytecode.convert.i8tof64  %dst, %src
        bytecode.convert.i16tof32 %dst, %src
        bytecode.convert.i16tof64 %dst, %src
        bytecode.convert.i32tof32 %dst, %src
        bytecode.convert.i32tof64 %dst, %src
        bytecode.convert.i64tof32 %dst, %src
        bytecode.convert.i64tof64 %dst, %src
        bytecode.ret
    }
}

// CHECK-LABEL: bytecode.ext.func @convert_int_to_float
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[SRC:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:         bytecode.convert.i8tof32  [[DST]], [[SRC]]
// CHECK:         bytecode.convert.i8tof64  [[DST]], [[SRC]]
// CHECK:         bytecode.convert.i16tof32 [[DST]], [[SRC]]
// CHECK:         bytecode.convert.i16tof64 [[DST]], [[SRC]]
// CHECK:         bytecode.convert.i32tof32 [[DST]], [[SRC]]
// CHECK:         bytecode.convert.i32tof64 [[DST]], [[SRC]]
// CHECK:         bytecode.convert.i64tof32 [[DST]], [[SRC]]
// CHECK:         bytecode.convert.i64tof64 [[DST]], [[SRC]]
// CHECK:         bytecode.ret

// -----

bytecode.func_section @function_section {
    bytecode.ext.func @convert_float_to_int (i64) -> () {
        %dst = bytecode.virtual_general_register
        %src = bytecode.virtual_parameter_register 0
        bytecode.convert.f32toi8  %dst, %src
        bytecode.convert.f32toi16 %dst, %src
        bytecode.convert.f32toi32 %dst, %src
        bytecode.convert.f32toi64 %dst, %src
        bytecode.convert.f64toi8  %dst, %src
        bytecode.convert.f64toi16 %dst, %src
        bytecode.convert.f64toi32 %dst, %src
        bytecode.convert.f64toi64 %dst, %src
        bytecode.ret
    }
}

// CHECK-LABEL: bytecode.ext.func @convert_float_to_int
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[SRC:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:         bytecode.convert.f32toi8  [[DST]], [[SRC]]
// CHECK:         bytecode.convert.f32toi16 [[DST]], [[SRC]]
// CHECK:         bytecode.convert.f32toi32 [[DST]], [[SRC]]
// CHECK:         bytecode.convert.f32toi64 [[DST]], [[SRC]]
// CHECK:         bytecode.convert.f64toi8  [[DST]], [[SRC]]
// CHECK:         bytecode.convert.f64toi16 [[DST]], [[SRC]]
// CHECK:         bytecode.convert.f64toi32 [[DST]], [[SRC]]
// CHECK:         bytecode.convert.f64toi64 [[DST]], [[SRC]]
// CHECK:         bytecode.ret

// -----

bytecode.func_section @function_section {
    bytecode.ext.func @convert_float_to_float (i64) -> () {
        %dst = bytecode.virtual_general_register
        %src = bytecode.virtual_parameter_register 0
        bytecode.convert.f32tof64 %dst, %src
        bytecode.convert.f64tof32 %dst, %src
        bytecode.ret
    }
}

// CHECK-LABEL: bytecode.ext.func @convert_float_to_float
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[SRC:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:         bytecode.convert.f32tof64 [[DST]], [[SRC]]
// CHECK:         bytecode.convert.f64tof32 [[DST]], [[SRC]]
// CHECK:         bytecode.ret
