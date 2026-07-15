//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

bytecode.type_section @type_section {
    bytecode.type @i64_type #bytecode.integer_type<width = 64, is_signed = true>
}

bytecode.func_section @function_section {
    bytecode.ext.func @create () -> () {
        %dst = bytecode.virtual_general_register
        %s0 = bytecode.virtual_general_register
        %s1 = bytecode.virtual_general_register
        %st0 = bytecode.virtual_general_register
        %st1 = bytecode.virtual_general_register
        bytecode.buffer.create %dst, @i64_type shape(%s0, %s1) strides(%st0, %st1)
        bytecode.ret
    }
    bytecode.ext.func @subview (i64) -> () {
        %dst = bytecode.virtual_general_register
        %src = bytecode.virtual_parameter_register 0
        %o0 = bytecode.virtual_general_register
        %o1 = bytecode.virtual_general_register
        %sz0 = bytecode.virtual_general_register
        %sz1 = bytecode.virtual_general_register
        %st0 = bytecode.virtual_general_register
        %st1 = bytecode.virtual_general_register
        bytecode.buffer.get_dim %sz0, %src, %o0
        bytecode.buffer.subview %dst, %src offsets(%o0, %o1) sizes(%sz0, %sz1) strides(%st0, %st1)
        bytecode.ret
    }
    bytecode.ext.func @store (i64, i64) -> () {
        %buf = bytecode.virtual_parameter_register 0
        %val = bytecode.virtual_parameter_register 1
        %i0 = bytecode.virtual_general_register
        %i1 = bytecode.virtual_general_register
        bytecode.buffer.store %buf, %val indices(%i0, %i1)
        bytecode.ret
    }
}

// CHECK-LABEL: bytecode.ext.func @create
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[S0:%.+]] = bytecode.virtual_general_register
// CHECK:         [[S1:%.+]] = bytecode.virtual_general_register
// CHECK:         [[ST0:%.+]] = bytecode.virtual_general_register
// CHECK:         [[ST1:%.+]] = bytecode.virtual_general_register
// CHECK:         bytecode.buffer.create [[DST]], @i64_type shape([[S0]], [[S1]]) strides([[ST0]], [[ST1]])
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @subview
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[SRC:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:         [[O0:%.+]] = bytecode.virtual_general_register
// CHECK:         [[O1:%.+]] = bytecode.virtual_general_register
// CHECK:         [[SZ0:%.+]] = bytecode.virtual_general_register
// CHECK:         [[SZ1:%.+]] = bytecode.virtual_general_register
// CHECK:         [[ST0:%.+]] = bytecode.virtual_general_register
// CHECK:         [[ST1:%.+]] = bytecode.virtual_general_register
// CHECK:         bytecode.buffer.get_dim [[SZ0]], [[SRC]], [[O0]]
// CHECK:         bytecode.buffer.subview [[DST]], [[SRC]] offsets([[O0]], [[O1]]) sizes([[SZ0]], [[SZ1]]) strides([[ST0]], [[ST1]])
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @store
// CHECK:         [[BUF:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:         [[VAL:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[I0:%.+]] = bytecode.virtual_general_register
// CHECK:         [[I1:%.+]] = bytecode.virtual_general_register
// CHECK:         bytecode.buffer.store [[BUF]], [[VAL]] indices([[I0]], [[I1]])
// CHECK:         bytecode.ret
