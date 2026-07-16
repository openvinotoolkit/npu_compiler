//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-translate --platform=%platform% --export-bytecode %s -o %t
// RUN: bytecode_interpreter --path %t --mode=run --function=vm_buffer | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

module {
bytecode.func_section @function_section {
    bytecode.func @vm_buffer @fn_vm_buffer @fn_type {
        %buf = bytecode.general_register 0
        %view = bytecode.general_register 1
        %dim0 = bytecode.general_register 2
        %dim1 = bytecode.general_register 3
        %stride0 = bytecode.general_register 4
        %stride1 = bytecode.general_register 5
        %offset0 = bytecode.general_register 6
        %offset1 = bytecode.general_register 7
        %src_dim = bytecode.general_register 8
        %view_dim = bytecode.general_register 9

        bytecode.set_imm %dim0, 2
        bytecode.set_imm %dim1, 3
        bytecode.set_imm %stride0, 3
        bytecode.set_imm %stride1, 1
        bytecode.set_imm %offset0, 1
        bytecode.set_imm %offset1, 1

        bytecode.buffer.create %buf, @i64_type shape(%dim0, %dim1) strides(%stride0, %stride1)
        bytecode.buffer.get_dim %src_dim, %buf, %offset1
        bytecode.buffer.subview %view, %buf offsets(%offset0, %offset1) sizes(%offset1, %dim0)
            strides(%offset1, %offset1)
        bytecode.buffer.get_dim %view_dim, %view, %offset1
        bytecode.retv %src_dim, %view_dim
    }
}
bytecode.string_section @string_section {
    bytecode.string @fn_vm_buffer "vm_buffer"
}
bytecode.type_section @type_section {
    bytecode.type @i64_type #bytecode.integer_type<width = 64, is_signed = true>
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = [@i64_type, @i64_type]>
}
}

// CHECK: Result[0] (i64): 3
// CHECK: Result[1] (i64): 2
