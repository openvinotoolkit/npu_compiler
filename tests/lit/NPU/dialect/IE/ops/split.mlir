//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK: func.func @ConvertConstToAttr([[ARG:%.*]]: tensor<2x6xf32>)
func.func @ConvertConstToAttr(%arg: tensor<2x6xf32>) -> (tensor<1x6xf32>, tensor<1x6xf32>) {
    %0 = const.Declare tensor<1xsi64> = dense<0> : tensor<1xsi64>
    %1:2 = IE.Split(%arg, %0) {num_splits = 2} : tensor<2x6xf32>, tensor<1xsi64> -> tensor<1x6xf32>, tensor<1x6xf32>
    return %1#0, %1#1 : tensor<1x6xf32>, tensor<1x6xf32>

    // CHECK-NOT:   const.Declare
    // CHECK:       [[VAL0:%.*]]:2 = IE.Split([[ARG]])
    // CHECK-SAME:      axis_value = 0
    // CHECK-SAME:      num_splits = 2
    // CHECK:       return [[VAL0]]#0, [[VAL0]]#1
}

// CHECK: func.func @ConvertConstToAttrNegativeInd([[ARG:%.*]]: tensor<2x6xf32>)
func.func @ConvertConstToAttrNegativeInd(%arg: tensor<2x6xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
    %0 = const.Declare tensor<1xsi64> = dense<-1> : tensor<1xsi64>
    %1:2 = IE.Split(%arg, %0) {num_splits = 2} : tensor<2x6xf32>, tensor<1xsi64> -> tensor<2x3xf32>, tensor<2x3xf32>
    return %1#0, %1#1 : tensor<2x3xf32>, tensor<2x3xf32>

    // CHECK-NOT:   const.Declare
    // CHECK:       [[VAL0:%.*]]:2 = IE.Split([[ARG]])
    // CHECK-SAME:      axis_value = 1
    // CHECK-SAME:      num_splits = 2
    // CHECK:       return [[VAL0]]#0, [[VAL0]]#1
}
