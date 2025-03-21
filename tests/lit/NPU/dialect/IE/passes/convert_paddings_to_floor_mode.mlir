//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-paddings-to-floor-mode %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @MaxPool
func.func @MaxPool(%arg0: tensor<1x512x38x38xf32>) -> tensor<1x512x19x19xf32> {
    %0 = IE.MaxPool(%arg0)
        {
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [2, 2],
            rounding_type = #IE.rounding_type<FLOOR>
        } :
        tensor<1x512x38x38xf32> -> tensor<1x512x19x19xf32>

    return %0 : tensor<1x512x19x19xf32>

    // CHECK:        [[MAX_POOL:%.+]] = IE.MaxPool(%arg0) {
    // CHECK-SAME:       kernel_size = [1, 1],
    // CHECK-SAME:       pads_begin = [0, 0],
    // CHECK-SAME:       pads_end = [0, 0],
    // CHECK-SAME:       rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:       strides = [2, 2]
    // CHECK-SAME:   } :
    // CHECK:        tensor<1x512x38x38xf32> -> tensor<1x512x19x19xf32>
    // CHECK:        return [[MAX_POOL]]  : tensor<1x512x19x19xf32>
}

// -----

// CHECK-LABEL: @AvgPool5D
func.func @AvgPool5D(%arg0: tensor<1x3x38x38x38xf32>) -> tensor<1x3x19x19x19xf32> {
    %0 = IE.AvgPool(%arg0)
        {
            kernel_size = [1, 1, 1],
            pads_begin = [0, 0, 0],
            pads_end = [0, 0, 0],
            strides = [2, 2, 2],
            rounding_type = #IE.rounding_type<FLOOR>
        } :
        tensor<1x3x38x38x38xf32> -> tensor<1x3x19x19x19xf32>

    return %0 : tensor<1x3x19x19x19xf32>

    // CHECK:        [[AVG_POOL:%.+]] = IE.AvgPool(%arg0) {
    // CHECK-SAME:       kernel_size = [1, 1, 1],
    // CHECK-SAME:       pads_begin = [0, 0, 0],
    // CHECK-SAME:       pads_end = [0, 0, 0],
    // CHECK-SAME:       rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:       strides = [2, 2, 2]
    // CHECK-SAME:   } :
    // CHECK:        tensor<1x3x38x38x38xf32> -> tensor<1x3x19x19x19xf32>
    // CHECK:        return [[AVG_POOL]]  : tensor<1x3x19x19x19xf32>
}


// -----

// CHECK-LABEL: @Convolution5D
func.func @Convolution5D(%arg0: tensor<1x3x38x38x38xf32>, %arg1: tensor<16x3x3x3x3xf32>) -> tensor<1x16x20x20x20xf32> {
    %0 = IE.Convolution(%arg0, %arg1)
        {
            pads_begin = [2, 2, 2],
            pads_end = [2, 2, 2],
            strides = [2, 2, 2],
            dilations = [1, 1, 1]
        } :
        tensor<1x3x38x38x38xf32>, tensor<16x3x3x3x3xf32> -> tensor<1x16x20x20x20xf32>

    return %0 : tensor<1x16x20x20x20xf32>

    // CHECK:        [[CONV:%.+]] = IE.Convolution(%arg0, %arg1)
    // CHECK-SAME:       dilations = [1, 1, 1],
    // CHECK-SAME:       pads_begin = [2, 2, 2],
    // CHECK-SAME:       pads_end = [1, 1, 1],
    // CHECK-SAME:       strides = [2, 2, 2]
    // CHECK-SAME:   } :
    // CHECK:        tensor<1x3x38x38x38xf32>, tensor<16x3x3x3x3xf32> -> tensor<1x16x20x20x20xf32>
    // CHECK:        return [[CONV]]  : tensor<1x16x20x20x20xf32>
}

// -----

// CHECK-LABEL: @AvgPoolExcludePadEnabled
func.func @AvgPoolExcludePadEnabled(%arg0: tensor<1x16x30x30xf32>) -> tensor<1x16x15x15xf32> {
    %0 = IE.AvgPool(%arg0)
        {
            kernel_size = [3, 3],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [2, 2],
            rounding_type = #IE.rounding_type<CEIL>
        } :
        tensor<1x16x30x30xf32> -> tensor<1x16x15x15xf32>

    return %0 : tensor<1x16x15x15xf32>

    // CHECK:        [[AVG_POOL:%.+]] = IE.AvgPool(%arg0) {
    // CHECK-SAME:       exclude_pads,
    // CHECK-SAME:       kernel_size = [3, 3],
    // CHECK-SAME:       pads_begin = [0, 0],
    // CHECK-SAME:       pads_end = [1, 1],
    // CHECK-SAME:       rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:       strides = [2, 2]
    // CHECK-SAME:   } :
    // CHECK:        tensor<1x16x30x30xf32> -> tensor<1x16x15x15xf32>
    // CHECK:        return [[AVG_POOL]]  : tensor<1x16x15x15xf32>
}
