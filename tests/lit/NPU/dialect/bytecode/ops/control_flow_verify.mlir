//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --verify-diagnostics %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// Test that je with falseDest as the next block passes verification
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "je_valid"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @je_valid @fn_name @fn_type {
        %lhs = bytecode.virtual_general_register
        %rhs = bytecode.virtual_general_register
        bytecode.je %lhs, %rhs, ^trueDest, ^falseDest
      ^falseDest:
        bytecode.ret
      ^trueDest:
        bytecode.ret
    }
}
}

// -----

// Test that je with trueDest (not falseDest) as the next block is caught by the verifier
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "je_invalid"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @je_invalid @fn_name @fn_type {
        %lhs = bytecode.virtual_general_register
        %rhs = bytecode.virtual_general_register
        // expected-error @+1 {{falseDest must be the physically next block in the region}}
        bytecode.je %lhs, %rhs, ^trueDest, ^falseDest
      ^trueDest:
        bytecode.ret
      ^falseDest:
        bytecode.ret
    }
}
}

// -----

// Test that jne with falseDest as the next block passes verification
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "jne_valid"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @jne_valid @fn_name @fn_type {
        %lhs = bytecode.virtual_general_register
        %rhs = bytecode.virtual_general_register
        bytecode.jne %lhs, %rhs, ^trueDest, ^falseDest
      ^falseDest:
        bytecode.ret
      ^trueDest:
        bytecode.ret
    }
}
}

// -----

// Test that jne with trueDest (not falseDest) as the next block is caught by the verifier
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "jne_invalid"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @jne_invalid @fn_name @fn_type {
        %lhs = bytecode.virtual_general_register
        %rhs = bytecode.virtual_general_register
        // expected-error @+1 {{falseDest must be the physically next block in the region}}
        bytecode.jne %lhs, %rhs, ^trueDest, ^falseDest
      ^trueDest:
        bytecode.ret
      ^falseDest:
        bytecode.ret
    }
}
}

// -----

// Test that je with trueDest as current block passes when a serializable op precedes it
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "je_self_loop_valid"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @je_self_loop_valid @fn_name @fn_type {
        %lhs = bytecode.virtual_general_register
        %rhs = bytecode.virtual_general_register
        bytecode.jmp ^loop_body
      ^loop_body:
        bytecode.add.i64 %lhs, %lhs, %rhs
        bytecode.je %lhs, %rhs, ^loop_body, ^exit
      ^exit:
        bytecode.ret
    }
}
}

// -----

// Test that je with trueDest as current block and no serializable predecessor is caught by the verifier
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "je_self_loop_invalid"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @je_self_loop_invalid @fn_name @fn_type {
        %lhs = bytecode.virtual_general_register
        %rhs = bytecode.virtual_general_register
        bytecode.jmp ^loop_body
      ^loop_body:
        // expected-error @+1 {{trueDest self-loop with no serializable predecessor produces a zero PC-relative offset}}
        bytecode.je %lhs, %rhs, ^loop_body, ^exit
      ^exit:
        bytecode.ret
    }
}
}
// -----

// Test that jne with trueDest as current block and no serializable predecessor is caught by the verifier
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "jne_self_loop_invalid"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @jne_self_loop_invalid @fn_name @fn_type {
        %lhs = bytecode.virtual_general_register
        %rhs = bytecode.virtual_general_register
        bytecode.jmp ^loop_body
      ^loop_body:
        // expected-error @+1 {{trueDest self-loop with no serializable predecessor produces a zero PC-relative offset}}
        bytecode.jne %lhs, %rhs, ^loop_body, ^exit
      ^exit:
        bytecode.ret
    }
}
}
