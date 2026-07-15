//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

bytecode.type_section @type_section {
    bytecode.type @i64_type #bytecode.integer_type<width = 64, is_signed = true>
    bytecode.type @f32_type #bytecode.float_type<width = 32, format = IEEE754>
    // CHECK:  bytecode.type @i64_type #bytecode.integer_type<width = 64, is_signed = true>
    // CHECK:  bytecode.type @f32_type #bytecode.float_type<width = 32, format = IEEE754>
}
