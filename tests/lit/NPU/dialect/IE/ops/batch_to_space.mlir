//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @ConvertConstToAttr
  func.func @ConvertConstToAttr(%arg0: tensor<24x16x7x8xf32>) -> tensor<1x32x20x31xf32> {
    %cst_shape = const.Declare tensor<4xsi32> = dense<[1, 2, 3, 4]> : tensor<4xsi64>, [#const.CastElemType<si32>]
    %cst_crops_begin = const.Declare tensor<4xsi32> = dense<[0, 0, 0, 1]> : tensor<4xsi64>, [#const.CastElemType<si32>]
    %cst_crops_end = const.Declare tensor<4xsi32> = dense<[0, 0, 1, 0]> : tensor<4xsi64>, [#const.CastElemType<si32>]

    %0 = IE.BatchToSpace(%arg0, %cst_shape, %cst_crops_begin, %cst_crops_end) {operandSegmentSizes = array<i32: 1, 1, 1, 1>} : tensor<24x16x7x8xf32>, tensor<4xsi32>, tensor<4xsi32>, tensor<4xsi32> -> tensor<1x32x20x31xf32>

    return %0 : tensor<1x32x20x31xf32>

    // CHECK-NOT:   const.Declare
    // CHECK: [[VAL0:%.+]] = IE.BatchToSpace(%arg0) {block_shape_value = [1, 2, 3, 4], crops_begin_value = [0, 0, 0, 1], crops_end_value = [0, 0, 1, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<24x16x7x8xf32> -> tensor<1x32x20x31xf32>
    // CHECK: return [[VAL0]] : tensor<1x32x20x31xf32>
}
