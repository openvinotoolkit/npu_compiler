//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @ConstFold
func.func @ConstFold(%arg0: tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32> {
    %prob = IE.LogSoftmax(%arg0) {axisInd = 0} : tensor<1x8x4x4xf32> -> tensor<1x8x4x4xf32>
    return %prob : tensor<1x8x4x4xf32>

    // CHECK-DAG:       [[VAL0:%.+]] = const.Declare tensor<1x8x4x4xf32> = dense<0.000000e+00> : tensor<1x8x4x4xf32>
    // CHECK-NOT:   IE.LogSoftmax
    // CHECK:       return [[VAL0]] : tensor<1x8x4x4xf32>
}
