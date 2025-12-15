//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-translate --vpu-arch=%arch% --weights-separation-path=true --import-IE ./ws_range.xml | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// This test checks that the following construct is correctly folded into a single constant, even in weights separation mode:
// Convert   Convert   Convert
//    \         |         /
//     +------Range------+
//              |
//           Reshape

// CHECK: func.func @main() -> tensor<16xf32> {
// CHECK: [[CST:%.+]] = const.Declare tensor<16xf32>
// CHECK:   return [[CST]] : tensor<16xf32>
