//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

bytecode.func_section @imm_register_section {
    bytecode.ext.func @imm_register_test () -> () {
        %dst = bytecode.virtual_general_register
        %c42 = bytecode.imm_register 42
        bytecode.set %dst, %c42
        bytecode.ret
    }
}

// CHECK-LABEL: bytecode.ext.func @imm_register_test
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[C42:%.+]] = bytecode.imm_register 42
// CHECK:         bytecode.set [[DST]], [[C42]]
// CHECK:         bytecode.ret
