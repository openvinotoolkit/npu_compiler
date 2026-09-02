//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-translate --split-input-file --platform=%platform% --export-bytecode %s -o %t
// RUN: bytecode_interpreter --path %t --mode print-full | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// Verifies that PC-relative byte offsets are correctly computed and embedded directly in
// JMP/JE/JNE instructions during serialization (FuncOp::serialize()).
//
// Four functions exercise forward jump, backward jump (loop back-edge), forward conditional
// jump (je), and backward conditional jump (jne do-while), with body sizes 12 B, 28 B, 18 B,
// and 34 B respectively (each jump instruction now encodes a 64-bit offset).

module {
bytecode.func_section @function_section {
    bytecode.func @jmp_forward @string_section::@fn_jmp_forward @type_section::@fn_void_type {
        bytecode.jmp ^bb1
    ^bb1:
        bytecode.ret
    }
    bytecode.func @jmp_backward @string_section::@fn_jmp_backward @type_section::@fn_void_type {
        %r1 = bytecode.general_register 1
        %r2 = bytecode.general_register 2
        bytecode.jmp ^loop
    ^loop:
        bytecode.add.i64 %r1, %r1, %r2
        bytecode.jmp ^loop
    }
    bytecode.func @je_forward @string_section::@fn_je_forward @type_section::@fn_void_type {
        %r1 = bytecode.general_register 1
        %r2 = bytecode.general_register 2
        bytecode.je %r1, %r2, ^taken, ^fallthrough
    ^fallthrough:
        bytecode.ret
    ^taken:
        bytecode.ret
    }
    bytecode.func @jne_backward @string_section::@fn_jne_backward @type_section::@fn_void_type {
        %r1 = bytecode.general_register 1
        %r2 = bytecode.general_register 2
        bytecode.jmp ^loop_body
    ^loop_body:
        bytecode.add.i64 %r1, %r1, %r2
        bytecode.jne %r1, %r2, ^loop_body, ^exit
    ^exit:
        bytecode.ret
    }
}
bytecode.string_section @string_section {
    bytecode.string @fn_jmp_forward "jmp_forward"
    bytecode.string @fn_jmp_backward "jmp_backward"
    bytecode.string @fn_je_forward "je_forward"
    bytecode.string @fn_jne_backward "jne_backward"
}
bytecode.type_section @type_section {
    bytecode.type @fn_void_type #bytecode.function_type<arguments = [], results = []>
}
}

// CHECK:  Magic Number: 4E 50 55 42 79 74 65 00
// CHECK:  Version: 1.0.0
// CHECK:  Section Header Table:
// CHECK:    Number of sections: 3
// CHECK:      Section type: Function, name index: 0, offset: {{[0-9]+}}, size: 92
// CHECK:        Number of functions: 4, entrypoint function index: 0
// CHECK:          Name index: 0, function type index: 0, num general registers: 0, body offset: 0, body size: 12
// CHECK:          Name index: 1, function type index: 0, num general registers: 3, body offset: 12, body size: 28
// CHECK:          Name index: 2, function type index: 0, num general registers: 3, body offset: 40, body size: 18
// CHECK:          Name index: 3, function type index: 0, num general registers: 3, body offset: 58, body size: 34
// CHECK:      Section type: String, name index: 0, offset: {{[0-9]+}}, size: 49
// CHECK:      Section type: Type, name index: 0, offset: {{[0-9]+}}, size: 5
// CHECK:    Function section 0
// CHECK:      Function name: jmp_forward
// CHECK:        jmp 10
// CHECK:        ret
// CHECK:      Function name: jmp_backward
// CHECK:        jmp 10
// CHECK:        add.i64 1, 1, 2
// CHECK:        jmp -8
// CHECK:      Function name: je_forward
// CHECK:        je 16, 1, 2
// CHECK:        ret
// CHECK:        ret
// CHECK:      Function name: jne_backward
// CHECK:        jmp 10
// CHECK:        add.i64 1, 1, 2
// CHECK:        jne -8, 1, 2
// CHECK:        ret
// CHECK:    String section 0
// CHECK:      String 0: jmp_forward\0
// CHECK:      String 1: jmp_backward\0
// CHECK:      String 2: je_forward\0
// CHECK:      String 3: jne_backward\0
// CHECK:    Type section 0
// CHECK:      Type 0: function<params=[], results=[]>
