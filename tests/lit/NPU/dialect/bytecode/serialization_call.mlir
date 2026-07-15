//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-translate --split-input-file --platform=%platform% --export-bytecode %s -o %t
// RUN: bytecode_interpreter --path %t --mode print-full | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

module {
bytecode.func_section @function_section {
    bytecode.func @callee_scalar @callee_scalar_name @fn_type_scalar {
        bytecode.ret
    }

    bytecode.func @caller_scalar @caller_scalar_name @fn_type_scalar {
        %func_idx = bytecode.general_register 0
        %arg = bytecode.general_register 1

        bytecode.set_imm %func_idx, 0
        bytecode.call %func_idx, results(), args(%arg : !bytecode.Register)

        bytecode.ret
    }

    bytecode.func @callee_buffer @callee_buffer_name @fn_type_buffer {
        bytecode.ret
    }

    bytecode.func @caller_buffer @caller_buffer_name @fn_type_buffer {
        %func_idx = bytecode.general_register 0
        %buf = bytecode.general_register 1
        %arg = bytecode.general_register 2

        bytecode.set_imm %func_idx, 2
        bytecode.call %func_idx, results(), args(%buf, %arg : !bytecode.Register, !bytecode.Register)

        bytecode.ret
    }

    bytecode.func @callee_retv @callee_retv_name @fn_type_retv {
        %dst = bytecode.general_register 0
        %arg = bytecode.general_register 1

        bytecode.set %dst, %arg
        bytecode.retv %dst
    }

    bytecode.func @caller_retv @caller_retv_name @fn_type_retv {
        %func_idx = bytecode.general_register 0
        %ret_dst = bytecode.general_register 1
        %arg = bytecode.general_register 2

        bytecode.set_imm %func_idx, 4
        bytecode.call %func_idx, results(%ret_dst : !bytecode.Register), args(%arg : !bytecode.Register)
        bytecode.retv %ret_dst
    }
}
bytecode.string_section @string_section {
    bytecode.string @callee_scalar_name "callee_scalar"
    bytecode.string @caller_scalar_name "caller_scalar"
    bytecode.string @callee_buffer_name "callee_buffer"
    bytecode.string @caller_buffer_name "caller_buffer"
    bytecode.string @callee_retv_name "callee_retv"
    bytecode.string @caller_retv_name "caller_retv"
}
bytecode.type_section @type_section {
    bytecode.type @i64_type #bytecode.integer_type<width = 64, is_signed = true>
    bytecode.type @buf #bytecode.buffer_type<element_type = @i64_type, rank = 4, shape = [1, 16, 32, 32], strides = [16384, 1024, 32, 1]>
    bytecode.type @fn_type_scalar #bytecode.function_type<arguments = [@i64_type, @i64_type], results = []>
    bytecode.type @fn_type_buffer #bytecode.function_type<arguments = [@buf, @i64_type], results = []>
    bytecode.type @fn_type_retv #bytecode.function_type<arguments = [@i64_type], results = [@i64_type]>
}
}

// CHECK: Version: 1.0.0
// CHECK: Section Header Table:
// CHECK: Number of sections: 3
// CHECK: Section type: Function, name index: 0
// CHECK: Number of functions: 6, entrypoint function index: 0
// CHECK: Function section 0
// CHECK: Function name: callee_scalar
// CHECK: ret
// CHECK: Function name: caller_scalar
// CHECK: set.imm 0, 0
// CHECK: call 0, 0, 1, 1
// CHECK: ret
// CHECK: Function name: callee_buffer
// CHECK: ret
// CHECK: Function name: caller_buffer
// CHECK: set.imm 0, 2
// CHECK: call 0, 0, 2, 1, 2
// CHECK: ret
// CHECK: Function name: callee_retv
// CHECK: set 0, 1
// CHECK: retv 1, 0
// CHECK: Function name: caller_retv
// CHECK: set.imm 0, 4
// CHECK: call 0, 1, 1, 1, 2
// CHECK: retv 1, 1
// CHECK: String section 0
// CHECK: String 0: callee_scalar\0
// CHECK: String 1: caller_scalar\0
// CHECK: String 2: callee_buffer\0
// CHECK: String 3: caller_buffer\0
// CHECK: String 4: callee_retv\0
// CHECK: String 5: caller_retv\0
// CHECK: Type section 0
// CHECK: Type 0: i64
// CHECK: Type 1: buffer<typeIndex=0, rank=4, shape=[1,16,32,32], strides=[16384,1024,32,1]>
// CHECK: Type 2: function<params=[0,0], results=[]>
// CHECK: Type 3: function<params=[1,0], results=[]>
// CHECK: Type 4: function<params=[0], results=[0]>
