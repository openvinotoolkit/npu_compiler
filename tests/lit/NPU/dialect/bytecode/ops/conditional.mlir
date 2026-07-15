//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

bytecode.func_section @function_section {
    bytecode.ext.func @select (i64, i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %cond = bytecode.virtual_parameter_register 0
        %trueVal = bytecode.virtual_parameter_register 1
        %falseVal = bytecode.virtual_parameter_register 2
        bytecode.select %dst, %cond, %trueVal, %falseVal
        bytecode.ret
    }
}

// CHECK-LABEL: bytecode.ext.func @select
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[COND:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:         [[TRUE:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[FALSE:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.select [[DST]], [[COND]], [[TRUE]], [[FALSE]]
// CHECK:         bytecode.ret
