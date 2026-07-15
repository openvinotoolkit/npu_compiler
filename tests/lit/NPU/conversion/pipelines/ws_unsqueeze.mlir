//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-translate --platform=%platform% --weights-separation-path=true --import-IE ./ws_unsqueeze.xml | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// This test checks that the following graph is correctly folded,
// even in weights separation mode because Unsqueeze does not support non-constant axes:
//
//                     Constant
//                         |
// Parameter            Convert
//     |                    |
//    +------Unsqueeze------+
//
//  =>
//
// Parameter    Constant
//     |           |
//    +--Unsqueeze--+
//

// CHECK: func.func @main([[ARG0:%.+]]: tensor<16xf32>) -> tensor<1x16xf32>
// CHECK:   [[CST:%.+]] = const.Declare tensor<1xsi64> = dense<0> : tensor<1xsi64>
// CHECK:   [[UNSQUEEZE:%.+]] = IE.Unsqueeze([[ARG0]], [[CST]]) : tensor<16xf32>, tensor<1xsi64> -> tensor<1x16xf32>
// CHECK:   return [[UNSQUEEZE]] : tensor<1x16xf32>
