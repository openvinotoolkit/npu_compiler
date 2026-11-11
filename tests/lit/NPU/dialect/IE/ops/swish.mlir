//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform%" --split-input-file --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @ConvertConstToAttr
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x16x300x300xf32>
func.func @ConvertConstToAttr(%arg0: tensor<1x16x300x300xf32>) -> tensor<1x16x300x300xf32> {
    %0 = const.Declare tensor<1x16xf32> = dense<1.0> : tensor<1x16xf32>
    %1 = IE.Swish(%arg0, %0) :
        tensor<1x16x300x300xf32>, tensor<1x16xf32> -> tensor<1x16x300x300xf32>
    return %1 : tensor<1x16x300x300xf32>

    // CHECK:       [[VAL0:%.+]] = IE.Swish([[ARG_0]])
    // CHECK-SAME:      beta_value = 1.000000e+00
    // CHECK:       return [[VAL0]]
}
