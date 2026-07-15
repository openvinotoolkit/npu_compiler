//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: not vpux-opt --split-input-file --init-compiler="platform=%platform%" --allocate-bytecode-registers %s 2>&1 | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// ===--------------------------------------------------------------------=== //
// Param index past the declared arity: signature has 1 argument, body uses slot 1.
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "param_out_of_range"
}
bytecode.type_section @type_section {
    bytecode.type @i64 #bytecode.integer_type<width = 64, is_signed = true>
    bytecode.type @fn_type #bytecode.function_type<arguments = [@i64], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @param_out_of_range @fn_name @fn_type {
        %p1 = bytecode.virtual_parameter_register 1
        bytecode.set_imm %p1, 100
        bytecode.ret
    }
}
}

// CHECK: bytecode.func @param_out_of_range uses parameter register index 1, but the signature only defines 1 parameter(s)

// -----

// ===--------------------------------------------------------------------=== //
// Signature declares no parameters at all: any virtual_parameter_register
// reference is rejected. Pins down the numParams == 0 boundary distinct from
// the case above (where numParams == 1).
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "param_when_no_params_declared"
}
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @param_when_no_params_declared @fn_name @fn_type {
        %p = bytecode.virtual_parameter_register 0
        bytecode.set_imm %p, 5
        bytecode.ret
    }
}
}

// CHECK: bytecode.func @param_when_no_params_declared uses parameter register index 0, but the signature only defines 0 parameter(s)

// -----

// ===--------------------------------------------------------------------=== //
// Negative literal in the assembly parses as an I16Attr, but the generated
// getter returns it as uint16_t, so `-1` wraps to 65535 before validation.
// The upper-bound check catches it; the `paramIndex < 0` branch of the
// validator is therefore unreachable from text IR. Pinning the 65535
// message here also guards against a regression where the getter is
// switched to a signed type (the diagnostic would then report `-1`).
// ===--------------------------------------------------------------------=== //
module {
bytecode.string_section @string_section {
    bytecode.string @fn_name "negative_param_index"
}
bytecode.type_section @type_section {
    bytecode.type @i64 #bytecode.integer_type<width = 64, is_signed = true>
    bytecode.type @fn_type #bytecode.function_type<arguments = [@i64], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @negative_param_index @fn_name @fn_type {
        %p = bytecode.virtual_parameter_register -1
        bytecode.set_imm %p, 7
        bytecode.ret
    }
}
}

// CHECK: bytecode.func @negative_param_index uses parameter register index 65535, but the signature only defines 1 parameter(s)
