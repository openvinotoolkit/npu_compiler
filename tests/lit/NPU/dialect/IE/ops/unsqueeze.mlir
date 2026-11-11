//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @Eliminate
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<4x4xf32>
func.func @Eliminate(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = IE.Unsqueeze(%arg0) { axes_value = [] } : tensor<4x4xf32> -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>

    // CHECK-NOT: IE.Unsqueeze
    // CHECK:     return [[ARG_0]]
}

// CHECK-LABEL: @ConstFold
func.func @ConstFold() -> tensor<1x1x4x4xf32> {
    %0 = const.Declare tensor<4x4xf32> = dense<1.0> : tensor<4x4xf32>
    %1 = IE.Unsqueeze(%0) { axes_value = [0, 1] } : tensor<4x4xf32> -> tensor<1x1x4x4xf32>
    return %1 : tensor<1x1x4x4xf32>

    // CHECK-DAG:       [[VAL0:%.+]] = const.Declare tensor<1x1x4x4xf32> =
    // CHECK-SAME:      dense<1.000000e+00> : tensor<4x4xf32>, [#const.Reshape<[1, 1, 4, 4]>]
    // CHECK-NOT:   IE.Unsqueeze
    // CHECK:       return [[VAL0]]
}

// CHECK-LABEL: @FuseWithReshape
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x16xf32>
func.func @FuseWithReshape(%arg0: tensor<1x16xf32>) -> tensor<4x1x4x1xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [4, 4] } : tensor<1x16xf32> -> tensor<4x4xf32>
    %1 = IE.Unsqueeze(%0) { axes_value = [1, 3] } : tensor<4x4xf32> -> tensor<4x1x4x1xf32>
    return %1 : tensor<4x1x4x1xf32>

    // CHECK: [[VAL0:%.+]] = IE.Reshape([[ARG_0]]) {shape_value = [4, 1, 4, 1]} : tensor<1x16xf32> -> tensor<4x1x4x1xf32>
    // CHECK: return [[VAL0]] : tensor<4x1x4x1xf32>
}

// CHECK-LABEL: @ConvertConstToAttr
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<4x4xf32>
func.func @ConvertConstToAttr(%arg0: tensor<4x4xf32>) -> tensor<4x1x4x1xf32> {
    %0 = const.Declare tensor<2xsi64> = dense<[1, 3]> : tensor<2xsi64>
    %1 = IE.Unsqueeze(%arg0, %0) : tensor<4x4xf32>, tensor<2xsi64> -> tensor<4x1x4x1xf32>
    return %1 : tensor<4x1x4x1xf32>

    // CHECK: [[VAL0:%.+]] = IE.Unsqueeze([[ARG_0]]) {axes_value = [1, 3]} : tensor<4x4xf32> -> tensor<4x1x4x1xf32>
    // CHECK: return [[VAL0]] : tensor<4x1x4x1xf32>
}
