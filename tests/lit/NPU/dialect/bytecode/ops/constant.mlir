//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

bytecode.constant_section @constant_section {
    bytecode.constant @my_constant dense<[42, 100, 50]> : tensor<3xi64>
    bytecode.constant @another_constant dense<[3.14]> : tensor<1xf32>
    // CHECK:  bytecode.constant @my_constant dense<[42, 100, 50]> : tensor<3xi64>
    // CHECK:  bytecode.constant @another_constant dense<3.140000e+00> : tensor<1xf32>
}
