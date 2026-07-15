//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

bytecode.func_section @function_section {
    bytecode.ext.func @ext_buffer_create_static () -> () {
        %0 = bytecode.virtual_general_register
        bytecode.ext.buffer.create %0, memref<2x3xf32>
        bytecode.ret
    }
}

// CHECK-LABEL: bytecode.ext.func @ext_buffer_create_static
// CHECK:         [[BUF:%.+]] = bytecode.virtual_general_register
// CHECK:         bytecode.ext.buffer.create [[BUF]], memref<2x3xf32>
// CHECK:         bytecode.ret

// -----

bytecode.func_section @function_section {
    bytecode.ext.func @ext_buffer_create_dynamic () -> () {
        %0 = bytecode.virtual_general_register
        %n = bytecode.virtual_general_register
        bytecode.ext.buffer.create %0, memref<?xi8> sizes(%n)
        bytecode.ret
    }
}

// CHECK-LABEL: bytecode.ext.func @ext_buffer_create_dynamic
// CHECK:         [[BUF:%.+]] = bytecode.virtual_general_register
// CHECK:         [[N:%.+]] = bytecode.virtual_general_register
// CHECK:         bytecode.ext.buffer.create [[BUF]], memref<?xi8> sizes([[N]])
// CHECK:         bytecode.ret
