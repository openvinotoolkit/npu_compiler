//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --verify-diagnostics --init-compiler="platform=%platform%" %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

bytecode.func_section @function_section {
    bytecode.ext.func @ext_buffer_create_dynamic_dimension_without_sizes () -> () {
        %0 = bytecode.virtual_general_register
        // expected-error @+1 {{expected 1 dynamic-size operand(s) for the dynamic dimensions of 'memref<?x3xf32>', got 0}}
        bytecode.ext.buffer.create %0, memref<?x3xf32>
        bytecode.ret
    }
}

// -----

bytecode.func_section @function_section {
    bytecode.ext.func @ext_buffer_create_static_with_sizes () -> () {
        %0 = bytecode.virtual_general_register
        %n = bytecode.virtual_general_register
        // expected-error @+1 {{expected 0 dynamic-size operand(s) for the dynamic dimensions of 'memref<2x3xf32>', got 1}}
        bytecode.ext.buffer.create %0, memref<2x3xf32> sizes(%n)
        bytecode.ret
    }
}

// -----

bytecode.func_section @function_section {
    bytecode.ext.func @ext_buffer_create_dynamic_stride () -> () {
        %0 = bytecode.virtual_general_register
        // expected-error @+1 {{memref strides must be fully static}}
        bytecode.ext.buffer.create %0, memref<2x3xf32, strided<[?, 1], offset: 0>>
        bytecode.ret
    }
}

// -----

bytecode.func_section @function_section {
    bytecode.ext.func @ext_buffer_create_non_strided_layout () -> () {
        %0 = bytecode.virtual_general_register
        // expected-error @+1 {{memref layout must be representable as a strided layout}}
        bytecode.ext.buffer.create %0, memref<2x3xf32, {order = affine_map<(d0, d1) -> (d1, d0)>}>
        bytecode.ret
    }
}

// -----

bytecode.func_section @function_section {
    bytecode.ext.func @ext_buffer_create_dynamic_offset () -> () {
        %0 = bytecode.virtual_general_register
        // expected-error @+1 {{memref offset must be static zero, got dynamic offset}}
        bytecode.ext.buffer.create %0, memref<2x3xf32, strided<[3, 1], offset: ?>>
        bytecode.ret
    }
}

// -----

bytecode.func_section @function_section {
    bytecode.ext.func @ext_buffer_create_non_zero_offset () -> () {
        %0 = bytecode.virtual_general_register
        // expected-error @+1 {{memref offset must be static zero, got 4}}
        bytecode.ext.buffer.create %0, memref<2x3xf32, strided<[3, 1], offset: 4>>
        bytecode.ret
    }
}
