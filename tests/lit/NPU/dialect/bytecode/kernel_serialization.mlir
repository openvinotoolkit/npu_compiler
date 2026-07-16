//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-translate --split-input-file --platform=%platform% --export-bytecode %s -o %t
// RUN: bytecode_interpreter --path %t --mode print-full | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

module {
bytecode.kernel_section @kernel_section {
    bytecode.kernel @first_kernel "\00\01\02\03"
    bytecode.kernel @second_kernel "\04\05\06\07\08"
}
}

// CHECK:      Section type: Kernel
// CHECK:        Number of entries: 2
// CHECK:    Kernel section 0
// CHECK:      Kernel 0: 0x00010203
// CHECK:      Kernel 1: 0x0405060708
