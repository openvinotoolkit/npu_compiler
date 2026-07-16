//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK: bytecode.type @i64 #bytecode.integer_type<width = 64, is_signed = true>
// CHECK: bytecode.type @i32 #bytecode.integer_type<width = 32, is_signed = true>
// CHECK: bytecode.type @i8 #bytecode.integer_type<width = 8, is_signed = true>
// CHECK: bytecode.type @u64 #bytecode.integer_type<width = 64, is_signed = false>
// CHECK: bytecode.type @u32 #bytecode.integer_type<width = 32, is_signed = false>
// CHECK: bytecode.type @u8 #bytecode.integer_type<width = 8, is_signed = false>
bytecode.type_section @type_section {
    bytecode.type @i64 #bytecode.integer_type<width = 64, is_signed = true>
    bytecode.type @i32 #bytecode.integer_type<width = 32, is_signed = true>
    bytecode.type @i8 #bytecode.integer_type<width = 8, is_signed = true>
    bytecode.type @u64 #bytecode.integer_type<width = 64, is_signed = false>
    bytecode.type @u32 #bytecode.integer_type<width = 32, is_signed = false>
    bytecode.type @u8 #bytecode.integer_type<width = 8, is_signed = false>
}

// -----

// CHECK: bytecode.type @f32 #bytecode.float_type<width = 32, format = IEEE754>
// CHECK: bytecode.type @f64 #bytecode.float_type<width = 64, format = IEEE754>
// CHECK: bytecode.type @bf16 #bytecode.float_type<width = 16, format = BFloat>
bytecode.type_section @type_section {
    bytecode.type @f32 #bytecode.float_type<width = 32, format = IEEE754>
    bytecode.type @f64 #bytecode.float_type<width = 64, format = IEEE754>
    bytecode.type @bf16 #bytecode.float_type<width = 16, format = BFloat>
}

// -----

// CHECK: bytecode.type @opaque8 #bytecode.opaque_type<width = 8>
bytecode.type_section @type_section {
    bytecode.type @opaque8 #bytecode.opaque_type<width = 8>
}

// -----

// CHECK: bytecode.type @i64 #bytecode.integer_type<width = 64, is_signed = true>
// CHECK: bytecode.type @buf #bytecode.buffer_type<element_type = @i64, rank = 4, shape = [1, 16, 32, 32], strides = [16384, 1024, 32, 1]>
bytecode.type_section @type_section {
    bytecode.type @i64 #bytecode.integer_type<width = 64, is_signed = true>
    bytecode.type @buf #bytecode.buffer_type<element_type = @i64, rank = 4, shape = [1, 16, 32, 32], strides = [16384, 1024, 32, 1]>
}

// -----

// CHECK: bytecode.type @i64 #bytecode.integer_type<width = 64, is_signed = true>
// CHECK: bytecode.type @buf #bytecode.buffer_type<element_type = @i64, rank = 2, shape = [4, 8], strides = [8, 1]>
// CHECK: bytecode.type @fn_type #bytecode.function_type<arguments = [@buf], results = [@i64]>
bytecode.type_section @type_section {
    bytecode.type @i64 #bytecode.integer_type<width = 64, is_signed = true>
    bytecode.type @buf #bytecode.buffer_type<element_type = @i64, rank = 2, shape = [4, 8], strides = [8, 1]>
    bytecode.type @fn_type #bytecode.function_type<arguments = [@buf], results = [@i64]>
}
