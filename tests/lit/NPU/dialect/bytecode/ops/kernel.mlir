//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

bytecode.kernel_section @kernel_section {
    bytecode.kernel @first_kernel "\00\01\02\03"
    bytecode.kernel @second_kernel "\04\05\06\07\08"
    // CHECK:  bytecode.kernel @first_kernel "\00\01\02\03"
    // CHECK:  bytecode.kernel @second_kernel "\04\05\06\07\08"
}
