//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @UseLeakyRelu
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x16x300x300xf32>)
func.func @UseLeakyRelu(%arg0: tensor<1x16x300x300xf32>) -> tensor<1x16x300x300xf32> {
    %0 = const.Declare tensor<1x16xf32> = dense<1.0> : tensor<1x16xf32>
    %1 = IE.PRelu(%arg0, %0) :
        tensor<1x16x300x300xf32>, tensor<1x16xf32> -> tensor<1x16x300x300xf32>
    return %1 : tensor<1x16x300x300xf32>

    // CHECK:       [[VAL0:%.+]] = IE.LeakyRelu([[ARG_0]])
    // CHECK-SAME:      negative_slope = 1.000000e+00
    // CHECK-NOT:   IE.PRelu
    // CHECK:       return [[VAL0]]
}
