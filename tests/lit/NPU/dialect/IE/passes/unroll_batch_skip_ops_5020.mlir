//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --unroll-batch="skip-unroll-batch=true" %s | FileCheck %s
// REQUIRES: platform-NPU5020

// NPU5020 has 1 DPU tile. When batch > numTiles, SplitOverBatch is not viable and the
// UnrollBatch pass must fall back to unrolling even if skip-unroll-batch=true.

// CHECK-LABEL: @FallbackUnrollAvgPoolBatchGreaterThanTiles
// CHECK-SAME:      [[IN1:%.+]]: tensor<3x64x28x28xf16>
func.func @FallbackUnrollAvgPoolBatchGreaterThanTiles(%arg0: tensor<3x64x28x28xf16>) -> tensor<3x64x14x14xf16> {
    %AVG_POOL = IE.AvgPool(%arg0) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [2, 2]
    } : tensor<3x64x28x28xf16> -> tensor<3x64x14x14xf16>

    return %AVG_POOL : tensor<3x64x14x14xf16>

    // CHECK-NOT:   IE.AvgPool([[IN1]])

    // CHECK:   [[SLICE0:%.+]] = IE.Slice [[IN1]] [0, 0, 0, 0] [1, 64, 28, 28] :
    // CHECK-SAME:      tensor<3x64x28x28xf16> to tensor<1x64x28x28xf16>
    // CHECK:   [[POOL0:%.+]] = IE.AvgPool([[SLICE0]])
    // CHECK-SAME:      : tensor<1x64x28x28xf16> -> tensor<1x64x14x14xf16>

    // CHECK:   [[SLICE1:%.+]] = IE.Slice [[IN1]] [1, 0, 0, 0] [1, 64, 28, 28] :
    // CHECK-SAME:      tensor<3x64x28x28xf16> to tensor<1x64x28x28xf16>
    // CHECK:   [[POOL1:%.+]] = IE.AvgPool([[SLICE1]])
    // CHECK-SAME:      : tensor<1x64x28x28xf16> -> tensor<1x64x14x14xf16>

    // CHECK:   [[SLICE2:%.+]] = IE.Slice [[IN1]] [2, 0, 0, 0] [1, 64, 28, 28] :
    // CHECK-SAME:      tensor<3x64x28x28xf16> to tensor<1x64x28x28xf16>
    // CHECK:   [[POOL2:%.+]] = IE.AvgPool([[SLICE2]])
    // CHECK-SAME:      : tensor<1x64x28x28xf16> -> tensor<1x64x14x14xf16>

    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[POOL0]], [[POOL1]], [[POOL2]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 0 : i64>
    // CHECK-SAME:  } : tensor<1x64x14x14xf16>, tensor<1x64x14x14xf16>, tensor<1x64x14x14xf16> -> tensor<3x64x14x14xf16>

    // CHECK:   return [[CONCAT]] : tensor<3x64x14x14xf16>
}

// -----

// CHECK-LABEL: @FallbackUnrollMaxPoolBatchGreaterThanTiles
// CHECK-SAME:      [[IN1:%.+]]: tensor<3x64x28x28xf16>
func.func @FallbackUnrollMaxPoolBatchGreaterThanTiles(%arg0: tensor<3x64x28x28xf16>) -> tensor<3x64x14x14xf16> {
    %MAX_POOL = IE.MaxPool(%arg0) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [2, 2]
    } : tensor<3x64x28x28xf16> -> tensor<3x64x14x14xf16>

    return %MAX_POOL : tensor<3x64x14x14xf16>

    // CHECK-NOT:   IE.MaxPool([[IN1]])

    // CHECK:   [[SLICE0:%.+]] = IE.Slice [[IN1]] [0, 0, 0, 0] [1, 64, 28, 28] :
    // CHECK-SAME:      tensor<3x64x28x28xf16> to tensor<1x64x28x28xf16>
    // CHECK:   [[POOL0:%.+]] = IE.MaxPool([[SLICE0]])
    // CHECK-SAME:      : tensor<1x64x28x28xf16> -> tensor<1x64x14x14xf16>

    // CHECK:   [[SLICE1:%.+]] = IE.Slice [[IN1]] [1, 0, 0, 0] [1, 64, 28, 28] :
    // CHECK-SAME:      tensor<3x64x28x28xf16> to tensor<1x64x28x28xf16>
    // CHECK:   [[POOL1:%.+]] = IE.MaxPool([[SLICE1]])
    // CHECK-SAME:      : tensor<1x64x28x28xf16> -> tensor<1x64x14x14xf16>

    // CHECK:   [[SLICE2:%.+]] = IE.Slice [[IN1]] [2, 0, 0, 0] [1, 64, 28, 28] :
    // CHECK-SAME:      tensor<3x64x28x28xf16> to tensor<1x64x28x28xf16>
    // CHECK:   [[POOL2:%.+]] = IE.MaxPool([[SLICE2]])
    // CHECK-SAME:      : tensor<1x64x28x28xf16> -> tensor<1x64x14x14xf16>

    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[POOL0]], [[POOL1]], [[POOL2]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 0 : i64>
    // CHECK-SAME:  } : tensor<1x64x14x14xf16>, tensor<1x64x14x14xf16>, tensor<1x64x14x14xf16> -> tensor<3x64x14x14xf16>

    // CHECK:   return [[CONCAT]] : tensor<3x64x14x14xf16>
}
