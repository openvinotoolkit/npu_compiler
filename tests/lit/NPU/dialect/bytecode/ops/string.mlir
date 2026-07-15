//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

bytecode.string_section @string_section {
    bytecode.string @my_string "Example of a string"
    bytecode.string @another_string "Another example"
    // CHECK:  bytecode.string @my_string "Example of a string"
    // CHECK:  bytecode.string @another_string "Another example"
}
