//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --verify-diagnostics %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// Test that a valid type reference passes verification
module {
bytecode.string_section @string_section {
    bytecode.string @valid_fn_name "valid_fn"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @valid_fn @string_section::@valid_fn_name @type_section::@fn_type {
        bytecode.ret
    }
}
}

// -----

// Test that a dangling type reference is caught by the verifier
module {
bytecode.string_section @string_section {
    bytecode.string @bad_fn_name "bad_fn"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    // expected-error @+1 {{references undefined symbol @type_section::@nonexistent}}
    bytecode.func @bad_fn @string_section::@bad_fn_name @type_section::@nonexistent {
        bytecode.ret
    }
}
}

// -----

// Test that a function type reference pointing to a non-function type is caught
module {
bytecode.string_section @string_section {
    bytecode.string @bad_fn_name "bad_fn"
}
bytecode.type_section @type_section {
    bytecode.type @i64_type #bytecode.integer_type<width = 64, is_signed = true>
}
bytecode.func_section @func_section {
    // expected-error @+1 {{resolves to a non-function type in the type section}}
    bytecode.func @bad_fn @string_section::@bad_fn_name @type_section::@i64_type {
        bytecode.ret
    }
}
}
