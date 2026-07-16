//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-translate --split-input-file --platform=%platform% --export-bytecode %s -o %t
// RUN: bytecode_interpreter --path %t --mode print-full | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

module {
bytecode.func_section @function_section {
    bytecode.func @arithmetic @fn_arithmetic @fn_type {
        %dst = bytecode.general_register 0
        %lhs = bytecode.general_register 1
        %rhs = bytecode.general_register 2
        bytecode.abs.i64 %dst, %rhs
        bytecode.add.i64 %dst, %dst, %rhs
        bytecode.div.i64 %dst, %lhs, %rhs
        bytecode.div.u64 %dst, %lhs, %rhs
        bytecode.min.u64 %dst, %lhs, %rhs
        bytecode.add.u64 %dst, %lhs, %rhs
        bytecode.max.u64 %dst, %lhs, %rhs
        bytecode.mul.u64 %dst, %lhs, %rhs
        bytecode.rem.u64 %dst, %lhs, %rhs
        bytecode.sub.u64 %dst, %lhs, %rhs
        bytecode.max.i64 %dst, %lhs, %rhs
        bytecode.min.i64 %dst, %lhs, %rhs
        bytecode.mul.i64 %dst, %lhs, %rhs
        bytecode.rem.i64 %dst, %lhs, %rhs
        bytecode.sub.i64 %dst, %lhs, %rhs
        bytecode.add.f64 %dst, %lhs, %rhs
        bytecode.sub.f64 %dst, %lhs, %rhs
        bytecode.mul.f64 %dst, %lhs, %rhs
        bytecode.div.f64 %dst, %lhs, %rhs
        bytecode.rem.f64 %dst, %lhs, %rhs
        bytecode.max.f64 %dst, %lhs, %rhs
        bytecode.min.f64 %dst, %lhs, %rhs
        bytecode.abs.f64 %dst, %rhs
        bytecode.neg.f64 %dst, %rhs
        bytecode.ceil.f64 %dst, %rhs
        bytecode.floor.f64 %dst, %rhs
        bytecode.round.f64 %dst, %lhs, 2
        bytecode.ret
    }

    bytecode.func @bitwise @fn_bitwise @fn_type {
        %dst = bytecode.general_register 0
        %lhs = bytecode.general_register 1
        %rhs = bytecode.general_register 2
        bytecode.and.64 %dst, %lhs, %rhs
        bytecode.not.64 %dst, %rhs
        bytecode.or.64 %dst, %lhs, %rhs
        bytecode.sll.64 %dst, %lhs, %rhs
        bytecode.srl.64 %dst, %lhs, %rhs
        bytecode.sra.64 %dst, %lhs, %rhs
        bytecode.xor.64 %dst, %lhs, %rhs
        bytecode.ret
    }

    bytecode.func @comparison @fn_comparison @fn_type {
        %dst = bytecode.general_register 0
        %lhs = bytecode.general_register 1
        %rhs = bytecode.general_register 2
        bytecode.cmp.i64 %dst, %lhs, %rhs, 260
        bytecode.cmp.f64 %dst, %lhs, %rhs, 2
        bytecode.ret
    }

    bytecode.func @kernel_submission @fn_kernel_submission @fn_type_no_args_no_results {
        %cmd_list = bytecode.general_register 0
        %kernel = bytecode.general_register 1
        %signal = bytecode.general_register 2
        %wait0 = bytecode.general_register 3
        %wait1 = bytecode.general_register 4
        bytecode.cmd_list.create %cmd_list
        bytecode.cmd_list.add_kernel %cmd_list, %kernel,
                                     (%signal : !bytecode.Register),
                                     (%wait0, %wait1 : !bytecode.Register, !bytecode.Register)
        bytecode.cmd_list.close %cmd_list
        bytecode.cmd_list.exec %cmd_list, 1
        bytecode.ret
    }

    bytecode.func @select @fn_select @fn_type {
        %dst = bytecode.general_register 0
        %cond = bytecode.general_register 1
        %trueVal = bytecode.general_register 2
        %falseVal = bytecode.general_register 3
        bytecode.select %dst, %cond, %trueVal, %falseVal
        bytecode.ret
    }

    bytecode.func @buffer @fn_buffer @fn_type {
        %dst = bytecode.general_register 0
        %src = bytecode.general_register 1
        %dim0 = bytecode.general_register 2
        %dim1 = bytecode.general_register 3
        bytecode.buffer.get_dim %dim1, %src, %dim0
        bytecode.buffer.subview %dst, %src offsets(%dim0) sizes(%dim1) strides(%dim0)
        bytecode.ret
    }

    // Exercises kernel.create end-to-end: takes two f32[16] buffer parameters,
    // takes a rank-1 subview of each (offset=0, size=8, stride=1) using a kernel
    // from the kernel section (index 0)
    bytecode.func @kernel_create @fn_kernel_create @fn_buf_pair_type {
        %in_buf  = bytecode.general_register 0
        %out_buf = bytecode.general_register 1
        %off     = bytecode.general_register 2
        %size    = bytecode.general_register 3
        %stride  = bytecode.general_register 4
        %in_sv   = bytecode.general_register 5
        %out_sv  = bytecode.general_register 6
        %handle  = bytecode.general_register 7
        bytecode.set_imm %off, 0
        bytecode.set_imm %size, 8
        bytecode.set_imm %stride, 1
        bytecode.buffer.subview %in_sv, %in_buf offsets(%off) sizes(%size) strides(%stride)
        bytecode.buffer.subview %out_sv, %out_buf offsets(%off) sizes(%size) strides(%stride)
        bytecode.kernel.create %handle, @my_kernel, inputs(%in_sv), outputs(%out_sv)
        bytecode.ret
    }

    bytecode.func @buffer_create @fn_buffer_create @fn_type_no_args_no_results {
        %dst = bytecode.general_register 0
        %dim0 = bytecode.general_register 1
        %dim1 = bytecode.general_register 2
        bytecode.buffer.create %dst, @i64_type shape(%dim0, %dim1) strides(%dim1, %dim0)
        bytecode.ret
    }

    bytecode.func @buffer_view @fn_buffer_view @fn_type {
        %dst = bytecode.general_register 0
        %src = bytecode.general_register 1
        %byteOff = bytecode.general_register 2
        %dim = bytecode.general_register 3
        bytecode.buffer.view %dst, %src, @i64_type offset(%byteOff) shape(%dim) strides(%dim)
        bytecode.ret
    }

    bytecode.func @conversion @fn_conversion @fn_type {
        %dst = bytecode.general_register 0
        %src = bytecode.general_register 1
        bytecode.convert.i8tof32  %dst, %src
        bytecode.convert.i8tof64  %dst, %src
        bytecode.convert.i16toi8  %dst, %src
        bytecode.convert.i16tof32 %dst, %src
        bytecode.convert.i16tof64 %dst, %src
        bytecode.convert.i32toi8  %dst, %src
        bytecode.convert.i32toi16 %dst, %src
        bytecode.convert.i32tof32 %dst, %src
        bytecode.convert.i32tof64 %dst, %src
        bytecode.convert.i64toi8  %dst, %src
        bytecode.convert.i64toi16 %dst, %src
        bytecode.convert.i64toi32 %dst, %src
        bytecode.convert.i64tof32 %dst, %src
        bytecode.convert.i64tof64 %dst, %src
        bytecode.convert.f32toi8  %dst, %src
        bytecode.convert.f32toi16 %dst, %src
        bytecode.convert.f32toi32 %dst, %src
        bytecode.convert.f32toi64 %dst, %src
        bytecode.convert.f32tof64 %dst, %src
        bytecode.convert.f64toi8  %dst, %src
        bytecode.convert.f64toi16 %dst, %src
        bytecode.convert.f64toi32 %dst, %src
        bytecode.convert.f64toi64 %dst, %src
        bytecode.convert.f64tof32 %dst, %src
        bytecode.ret
    }

}
bytecode.kernel_section @kernel_section {
    bytecode.kernel @my_kernel "\00\01\02\03"
}
bytecode.constant_section @constant_section {
    bytecode.constant @my_constant dense<[42, 100, 50]> : tensor<3xi64>
    bytecode.constant @another_constant dense<[3.14]> : tensor<1xf32>
}
bytecode.string_section @string_section {
    bytecode.string @my_kernel "my_kernel"
    bytecode.string @my_string "Example of a string"
    bytecode.string @another_string "Another example"
    bytecode.string @fn_arithmetic "arithmetic"
    bytecode.string @fn_bitwise "bitwise"
    bytecode.string @fn_comparison "comparison"
    bytecode.string @fn_kernel_submission "kernel_submission"
    bytecode.string @fn_select "select"
    bytecode.string @fn_buffer "buffer"
    bytecode.string @fn_kernel_create "kernel_create"
    bytecode.string @fn_buffer_create "buffer_create"
    bytecode.string @fn_buffer_view "buffer_view"
    bytecode.string @fn_conversion "conversion"
}
bytecode.type_section @type_section {
    bytecode.type @i64_type #bytecode.integer_type<width = 64, is_signed = true>
    bytecode.type @f32_type #bytecode.float_type<width = 32, format = IEEE754>
    bytecode.type @fn_type #bytecode.function_type<arguments = [@i64_type, @i64_type], results = []>
    bytecode.type @fn_type_no_args_no_results #bytecode.function_type<arguments = [], results = []>
    bytecode.type @in_buf_type #bytecode.buffer_type<element_type = @f32_type, rank = 1, shape = [16], strides = [1]>
    bytecode.type @out_buf_type #bytecode.buffer_type<element_type = @f32_type, rank = 1, shape = [16], strides = [1]>
    bytecode.type @fn_buf_pair_type #bytecode.function_type<arguments = [@in_buf_type, @out_buf_type], results = []>
}
bytecode.metadata_section @metadata_section {
    bytecode.network_metadata @my_string 0 1
    bytecode.input_metadata @another_string @f32_type @my_constant index_used_by_driver(0) has_dynamic_strides(false)
    bytecode.output_metadata @fn_arithmetic @f32_type @my_constant index_used_by_driver(0) has_dynamic_strides(false)
}
}


// CHECK:  Magic Number: 4E 50 55 42 79 74 65 00
// CHECK:  Version: 1.0.0
// CHECK:  Section Header Table:
// CHECK:    Number of sections: 6
// CHECK:      Section type: Function, name index: 0, offset: {{[0-9]+}}, size: 618
// CHECK:        Number of functions: 10, entrypoint function index: 0
// CHECK:          Name index: 3, function type index: 2, num general registers: 3, body offset: 0, body size: 208
// CHECK:          Name index: 4, function type index: 2, num general registers: 3, body offset: 208, body size: 56
// CHECK:          Name index: 5, function type index: 2, num general registers: 3, body offset: 264, body size: 22
// CHECK:          Name index: 6, function type index: 3, num general registers: 5, body offset: 286, body size: 32
// CHECK:          Name index: 7, function type index: 2, num general registers: 4, body offset: 318, body size: 12
// CHECK:          Name index: 8, function type index: 2, num general registers: 4, body offset: 330, body size: 24
// CHECK:          Name index: 9, function type index: 6, num general registers: 8, body offset: 354, body size: 82
// CHECK:          Name index: 10, function type index: 3, num general registers: 3, body offset: 436, body size: 18
// CHECK:          Name index: 11, function type index: 2, num general registers: 4, body offset: 454, body size: 18
// CHECK:          Name index: 12, function type index: 2, num general registers: 2, body offset: 472, body size: 146
// CHECK:      Section type: Constant, name index: 0, offset: {{[0-9]+}}, size: 28
// CHECK:        Number of entries: 2
// CHECK:          Entry 0 offset: 0, size: 24
// CHECK:          Entry 1 offset: 24, size: 4
// CHECK:      Section type: Kernel, name index: 0, offset: {{[0-9]+}}, size: 4
// CHECK:        Number of entries: 1
// CHECK:          Entry 0 offset: 0, size: 4
// CHECK:      Section type: String, name index: 0, offset: {{[0-9]+}}, size: 159
// CHECK:        Number of entries: 13
// CHECK:          Entry 0 offset: 0, size: 10
// CHECK:          Entry 1 offset: 10, size: 20
// CHECK:          Entry 2 offset: 30, size: 16
// CHECK:          Entry 3 offset: 46, size: 11
// CHECK:          Entry 4 offset: 57, size: 8
// CHECK:          Entry 5 offset: 65, size: 11
// CHECK:          Entry 6 offset: 76, size: 18
// CHECK:          Entry 7 offset: 94, size: 7
// CHECK:          Entry 8 offset: 101, size: 7
// CHECK:          Entry 9 offset: 108, size: 14
// CHECK:          Entry 10 offset: 122, size: 14
// CHECK:          Entry 11 offset: 136, size: 12
// CHECK:          Entry 12 offset: 148, size: 11
// CHECK:      Section type: Type, name index: 0, offset: {{[0-9]+}}, size: 105
// CHECK:        Number of entries: 7
// CHECK:          Entry 0 offset: 0, size: 3
// CHECK:          Entry 1 offset: 3, size: 3
// CHECK:          Entry 2 offset: 6, size: 21
// CHECK:          Entry 3 offset: 27, size: 5
// CHECK:          Entry 4 offset: 32, size: 26
// CHECK:          Entry 5 offset: 58, size: 26
// CHECK:          Entry 6 offset: 84, size: 21
// CHECK:      Section type: Metadata, name index: 0, offset: {{[0-9]+}}, size: 91
// CHECK:        Number of entries: 3
// CHECK:          Entry 0 offset: 0, size: 25
// CHECK:          Entry 1 offset: 25, size: 33
// CHECK:          Entry 2 offset: 58, size: 33
// CHECK:    Function section 0
// CHECK:      Function name: arithmetic
// CHECK:        abs.i64 0, 2
// CHECK:        add.i64 0, 0, 2
// CHECK:        div.i64 0, 1, 2
// CHECK:        div.u64 0, 1, 2
// CHECK:        min.u64 0, 1, 2
// CHECK:        add.u64 0, 1, 2
// CHECK:        max.u64 0, 1, 2
// CHECK:        mul.u64 0, 1, 2
// CHECK:        rem.u64 0, 1, 2
// CHECK:        sub.u64 0, 1, 2
// CHECK:        max.i64 0, 1, 2
// CHECK:        min.i64 0, 1, 2
// CHECK:        mul.i64 0, 1, 2
// CHECK:        rem.i64 0, 1, 2
// CHECK:        sub.i64 0, 1, 2
// CHECK:        add.f64 0, 1, 2
// CHECK:        sub.f64 0, 1, 2
// CHECK:        mul.f64 0, 1, 2
// CHECK:        div.f64 0, 1, 2
// CHECK:        rem.f64 0, 1, 2
// CHECK:        max.f64 0, 1, 2
// CHECK:        min.f64 0, 1, 2
// CHECK:        abs.f64 0, 2
// CHECK:        neg.f64 0, 2
// CHECK:        ceil.f64 0, 2
// CHECK:        floor.f64 0, 2
// CHECK:        round.f64 0, 1, 2
// CHECK:        ret
// CHECK:      Function name: bitwise
// CHECK:        and.64 0, 1, 2
// CHECK:        not.64 0, 2
// CHECK:        or.64 0, 1, 2
// CHECK:        sll.64 0, 1, 2
// CHECK:        srl.64 0, 1, 2
// CHECK:        sra.64 0, 1, 2
// CHECK:        xor.64 0, 1, 2
// CHECK:        ret
// CHECK:      Function name: comparison
// CHECK:        cmp.i64 0, 1, 2, 260
// CHECK:        cmp.f64 0, 1, 2, 2
// CHECK:        ret
// CHECK:      Function name: kernel_submission
// CHECK:        cmd_list.create 0
// CHECK:        cmd_list.add_kernel 0, 1, 1, 2, 2, 3, 4
// CHECK:        cmd_list.close 0
// CHECK:        cmd_list.exec 0, 1
// CHECK:        ret
// CHECK:      Function name: select
// CHECK:        select 0, 1, 2, 3
// CHECK:        ret
// CHECK:      Function name: buffer
// CHECK:        buffer.get_dim 3, 1, 2
// CHECK:        buffer.subview 0, 1, 1, 2, 3, 2
// CHECK:        ret
// CHECK:      Function name: kernel_create
// CHECK:        set.imm 2, 0
// CHECK:        set.imm 3, 8
// CHECK:        set.imm 4, 1
// CHECK:        buffer.subview 5, 0, 1, 2, 3, 4
// CHECK:        buffer.subview 6, 1, 1, 2, 3, 4
// CHECK:        kernel.create 7, 0, 0, 1, 5, 1, 6
// CHECK:        ret
// CHECK:      Function name: buffer_create
// CHECK:        buffer.create 0, 0, 2, 1, 2, 2, 1
// CHECK:        ret
// CHECK:      Function name: buffer_view
// CHECK:        buffer.view 0, 1, 2, 0, 1, 3, 3
// CHECK:        ret
// CHECK:      Function name: conversion
// CHECK:        convert.i8tof32 0, 1
// CHECK:        convert.i8tof64 0, 1
// CHECK:        convert.i16toi8 0, 1
// CHECK:        convert.i16tof32 0, 1
// CHECK:        convert.i16tof64 0, 1
// CHECK:        convert.i32toi8 0, 1
// CHECK:        convert.i32toi16 0, 1
// CHECK:        convert.i32tof32 0, 1
// CHECK:        convert.i32tof64 0, 1
// CHECK:        convert.i64toi8 0, 1
// CHECK:        convert.i64toi16 0, 1
// CHECK:        convert.i64toi32 0, 1
// CHECK:        convert.i64tof32 0, 1
// CHECK:        convert.i64tof64 0, 1
// CHECK:        convert.f32toi8 0, 1
// CHECK:        convert.f32toi16 0, 1
// CHECK:        convert.f32toi32 0, 1
// CHECK:        convert.f32toi64 0, 1
// CHECK:        convert.f32tof64 0, 1
// CHECK:        convert.f64toi8 0, 1
// CHECK:        convert.f64toi16 0, 1
// CHECK:        convert.f64toi32 0, 1
// CHECK:        convert.f64toi64 0, 1
// CHECK:        convert.f64tof32 0, 1
// CHECK:        ret
// CHECK:    Constant section 0
// CHECK:      Constant 0: 0x2A0000000000000064000000000000003200000000000000
// CHECK:      Constant 1: 0xC3F54840
// CHECK:    Kernel section 0
// CHECK:      Kernel 0: 0x00010203
// CHECK:    String section 0
// CHECK:      String 0: my_kernel\0
// CHECK:      String 1: Example of a string\0
// CHECK:      String 2: Another example\0
// CHECK:      String 3: arithmetic\0
// CHECK:      String 4: bitwise\0
// CHECK:      String 5: comparison\0
// CHECK:      String 6: kernel_submission\0
// CHECK:      String 7: select\0
// CHECK:      String 8: buffer\0
// CHECK:      String 9: kernel_create\0
// CHECK:      String 10: buffer_create\0
// CHECK:      String 11: buffer_view\0
// CHECK:      String 12: conversion\0
// CHECK:    Type section 0
// CHECK:      Type 0: i64
// CHECK:      Type 1: float32 (IEEE754)
// CHECK:      Type 2: function<params=[0,0], results=[]>
// CHECK:      Type 3: function<params=[], results=[]>
// CHECK:      Type 4: buffer<typeIndex=1, rank=1, shape=[16], strides=[1]>
// CHECK:      Type 5: buffer<typeIndex=1, rank=1, shape=[16], strides=[1]>
// CHECK:      Type 6: function<params=[4,5], results=[]>
// CHECK:    Metadata section 0
// CHECK:      Metadata 0: 0x00010000000000000000000000000000000100000000000000
// CHECK:      Metadata 1: 0x010200000000000000010000000000000000000000000000000000000000000000
// CHECK:      Metadata 2: 0x020300000000000000010000000000000000000000000000000000000000000000
