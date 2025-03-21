//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --handle-exclude-pad-for-avg-pool %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @HandleExcludePadForAvgPool
func.func @HandleExcludePadForAvgPool(%arg0 : tensor<1x1024x7x7xf16>) -> (tensor<1x1024x7x7xf16>) {
    %0 = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x7xf16>
    return %0 : tensor<1x1024x7x7xf16>

    //CHECK:        [[VAR:%.+]] = IE.AvgPool({{[^:]+}})
    //CHECK-SAME:   {kernel_size = [3, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x7x7xf16> -> tensor<1x1024x5x5xf16>

    //CHECK:        [[VAR0:%.+]] = IE.StridedSlice({{[^:]+}})
    //CHECK-SAME:   begins_attr = [0, 0, 0, 0]
    //CHECK-SAME:   ends_attr = [0, 0, 2, 2]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x2x2xf16>

    //CHECK:        [[VAR1:%.+]] = IE.AvgPool([[VAR0]])
    //CHECK-SAME:   {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x2x2xf16> -> tensor<1x1024x1x1xf16>

    //CHECK:        [[VAR2:%.+]] = IE.StridedSlice({{[^:]+}})
    //CHECK-SAME:   begins_attr = [0, 0, 0, 5]
    //CHECK-SAME:   ends_attr = [0, 0, 2, 7]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x2x2xf16>

    //CHECK:        [[VAR3:%.+]] = IE.AvgPool([[VAR2]])
    //CHECK-SAME:   {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x2x2xf16> -> tensor<1x1024x1x1xf16>

    //CHECK:        [[VAR4:%.+]] = IE.StridedSlice({{[^:]+}})
    //CHECK-SAME:   begins_attr = [0, 0, 5, 5]
    //CHECK-SAME:   ends_attr = [0, 0, 7, 7]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x2x2xf16>

    //CHECK:        [[VAR5:%.+]] = IE.AvgPool([[VAR4]])
    //CHECK-SAME:   {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x2x2xf16> -> tensor<1x1024x1x1xf16>

    //CHECK:        [[VAR6:%.+]] = IE.StridedSlice({{[^:]+}})
    //CHECK-SAME:   begins_attr = [0, 0, 5, 0]
    //CHECK-SAME:   ends_attr = [0, 0, 7, 2]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x2x2xf16>

    //CHECK:        [[VAR7:%.+]] = IE.AvgPool([[VAR6]])
    //CHECK-SAME:   {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x2x2xf16> -> tensor<1x1024x1x1xf16>

    //CHECK:        [[VAR8:%.+]] = IE.StridedSlice({{[^:]+}})
    //CHECK-SAME:   begins_attr = [0, 0, 0, 0]
    //CHECK-SAME:   ends_attr = [0, 0, 7, 2]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x2xf16>

    //CHECK:        [[VAR9:%.+]] = IE.AvgPool([[VAR8]])
    //CHECK-SAME:   {kernel_size = [3, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x7x2xf16> -> tensor<1x1024x5x1xf16>

    //CHECK:        [[VAR10:%.+]] = IE.StridedSlice({{[^:]+}})
    //CHECK-SAME:   begins_attr = [0, 0, 0, 5]
    //CHECK-SAME:   ends_attr = [0, 0, 7, 7]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x2xf16>

    //CHECK:        [[VAR11:%.+]] = IE.AvgPool([[VAR10]])
    //CHECK-SAME:   {kernel_size = [3, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x7x2xf16> -> tensor<1x1024x5x1xf16>

    //CHECK:        [[VAR12:%.+]] = IE.StridedSlice({{[^:]+}})
    //CHECK-SAME:   begins_attr = [0, 0, 0, 0]
    //CHECK-SAME:   ends_attr = [0, 0, 2, 7]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x2x7xf16>

    //CHECK:        [[VAR13:%.+]] = IE.AvgPool([[VAR12]])
    //CHECK-SAME:   {kernel_size = [2, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x2x7xf16> -> tensor<1x1024x1x5xf16>

    //CHECK:        [[VAR14:%.+]] = IE.StridedSlice({{[^:]+}})
    //CHECK-SAME:   begins_attr = [0, 0, 5, 0]
    //CHECK-SAME:   ends_attr = [0, 0, 7, 7]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x2x7xf16>

    //CHECK:        [[VAR15:%.+]] = IE.AvgPool([[VAR14]])
    //CHECK-SAME:   {kernel_size = [2, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x2x7xf16> -> tensor<1x1024x1x5xf16>

    //CHECK:        [[VAR17:%.+]] = IE.Concat([[VAR]], [[VAR1]], [[VAR3]], [[VAR5]], [[VAR7]], [[VAR9]], [[VAR11]], [[VAR13]], [[VAR15]])
    //CHECK-SAME{LITERAL}:      {static_offsets = [[0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 6], [0, 0, 6, 6], [0, 0, 6, 0], [0, 0, 1, 0], [0, 0, 1, 6], [0, 0, 0, 1], [0, 0, 6, 1]]}
    //CHECK-SAME:   : tensor<1x1024x5x5xf16>, tensor<1x1024x1x1xf16>, tensor<1x1024x1x1xf16>, tensor<1x1024x1x1xf16>, tensor<1x1024x1x1xf16>, tensor<1x1024x5x1xf16>, tensor<1x1024x5x1xf16>, tensor<1x1024x1x5xf16>, tensor<1x1024x1x5xf16> -> tensor<1x1024x7x7xf16>

    // CHECK        return [[VAR17]]
}

// -----

// CHECK-LABEL: @HandleExcludePadForAvgPoolCommonCase
func.func @HandleExcludePadForAvgPoolCommonCase(%arg0 : tensor<1x1024x7x7xf16>) -> (tensor<1x1024x8x3xf16>) {
    %0 = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [3, 3],
        pads_begin = [2, 1],
        pads_end = [1, 2],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 3]
    } : tensor<1x1024x7x7xf16> -> tensor<1x1024x8x3xf16>
    return %0 : tensor<1x1024x8x3xf16>

    //CHECK:        [[SLICE0:%.+]] = IE.StridedSlice({{[^:]+}})
    //CHECK-SAME:   begins_attr = [0, 0, 0, 2]
    //CHECK-SAME:   ends_attr = [0, 0, 7, 7]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x5xf16>
    //CHECK:        [[AVGPOOL0:%.+]] = IE.AvgPool([[SLICE0]])
    //CHECK-SAME:   {kernel_size = [3, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 3]} : tensor<1x1024x7x5xf16> -> tensor<1x1024x5x1xf16>

    //CHECK:        [[SLICE1:%.+]] = IE.StridedSlice({{[^:]+}})
    //CHECK-SAME:   begins_attr = [0, 0, 0, 0]
    //CHECK-SAME:   ends_attr = [0, 0, 1, 2]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x1x2xf16>
    //CHECK:        [[AVGPOOL1:%.+]] = IE.AvgPool([[SLICE1]])
    //CHECK-SAME:   {kernel_size = [1, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x1x2xf16> -> tensor<1x1024x1x1xf16>

    //CHECK:        [[SLICE2:%.+]] = IE.StridedSlice({{[^:]+}})
    //CHECK-SAME:   begins_attr = [0, 0, 0, 5]
    //CHECK-SAME:   ends_attr = [0, 0, 1, 7]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x1x2xf16>
    //CHECK:        [[AVGPOOL2:%.+]] = IE.AvgPool([[SLICE2]])
    //CHECK-SAME:   {kernel_size = [1, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x1x2xf16> -> tensor<1x1024x1x1xf16>

    //CHECK:        [[SLICE3:%.+]] = IE.StridedSlice({{[^:]+}})
    //CHECK-SAME:   begins_attr = [0, 0, 5, 5]
    //CHECK-SAME:   ends_attr = [0, 0, 7, 7]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x2x2xf16>
    //CHECK:        [[AVGPOOL3:%.+]] = IE.AvgPool([[SLICE3]])
    //CHECK-SAME:   {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x2x2xf16> -> tensor<1x1024x1x1xf16>

    //CHECK:        [[SLICE4:%.+]] = IE.StridedSlice({{[^:]+}})
    //CHECK-SAME:   begins_attr = [0, 0, 5, 0]
    //CHECK-SAME:   ends_attr = [0, 0, 7, 2]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x2x2xf16>
    //CHECK:        [[AVGPOOL4:%.+]] = IE.AvgPool([[SLICE4]])
    //CHECK-SAME:   {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x2x2xf16> -> tensor<1x1024x1x1xf16>

    //CHECK:        [[SLICE5:%.+]] = IE.StridedSlice({{[^:]+}})
    //CHECK-SAME:   begins_attr = [0, 0, 0, 0]
    //CHECK-SAME:   ends_attr = [0, 0, 7, 2]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x2xf16>
    //CHECK:        [[AVGPOOL5:%.+]] = IE.AvgPool([[SLICE5]])
    //CHECK-SAME:   {kernel_size = [3, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x7x2xf16> -> tensor<1x1024x5x1xf16>


    //CHECK:        [[SLICE6:%.+]] = IE.StridedSlice({{[^:]+}})
    //CHECK-SAME:   begins_attr = [0, 0, 0, 5]
    //CHECK-SAME:   ends_attr = [0, 0, 7, 7]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x2xf16>
    //CHECK:        [[AVGPOOL6:%.+]] = IE.AvgPool([[SLICE6]])
    //CHECK-SAME:   {kernel_size = [3, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x7x2xf16> -> tensor<1x1024x5x1xf16>

    //CHECK:        [[SLICE7:%.+]] = IE.StridedSlice({{[^:]+}})
    //CHECK-SAME:   begins_attr = [0, 0, 0, 2]
    //CHECK-SAME:   ends_attr = [0, 0, 1, 7]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x1x5xf16>
    //CHECK:        [[AVGPOOL7:%.+]] = IE.AvgPool([[SLICE7]])
    //CHECK-SAME:   {kernel_size = [1, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 3]} : tensor<1x1024x1x5xf16> -> tensor<1x1024x1x1xf16>

    //CHECK:        [[SLICE8:%.+]] = IE.StridedSlice({{[^:]+}})
    //CHECK-SAME:   begins_attr = [0, 0, 5, 2]
    //CHECK-SAME:   ends_attr = [0, 0, 7, 7]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x2x5xf16>
    //CHECK:        [[AVGPOOL8:%.+]] = IE.AvgPool([[SLICE8]])
    //CHECK-SAME:   {kernel_size = [2, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 3]} : tensor<1x1024x2x5xf16> -> tensor<1x1024x1x1xf16>

    //CHECK:        [[CONCAT:%.+]] = IE.Concat([[AVGPOOL0]], [[AVGPOOL1]], [[AVGPOOL2]], [[AVGPOOL3]], [[AVGPOOL4]], [[AVGPOOL5]], [[AVGPOOL6]], [[AVGPOOL7]], [[AVGPOOL8]])
    //CHECK-SAME{LITERAL}:      {static_offsets = [[0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 7, 2], [0, 0, 7, 0], [0, 0, 1, 0], [0, 0, 1, 2], [0, 0, 0, 1], [0, 0, 7, 1]]}
    //CHECK-SAME:   : tensor<1x1024x5x1xf16>, tensor<1x1024x1x1xf16>, tensor<1x1024x1x1xf16>, tensor<1x1024x1x1xf16>, tensor<1x1024x1x1xf16>, tensor<1x1024x5x1xf16>, tensor<1x1024x5x1xf16>, tensor<1x1024x1x1xf16>, tensor<1x1024x1x1xf16> -> tensor<1x1024x8x3xf16>
    // CHECK        return [[CONCAT]]

}

// -----

// CHECK-LABEL: @HandleExcludePadForAvgPoolPadWithZeroCase1
func.func @HandleExcludePadForAvgPoolPadWithZeroCase1(%arg0 : tensor<1x128x1x249xf16>) -> (tensor<1x128x1x50xf16>) {
    %0 = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [1, 5],
        pads_begin = [0, 0],
        pads_end = [0, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 5]
    } : tensor<1x128x1x249xf16> -> tensor<1x128x1x50xf16>
    return %0 : tensor<1x128x1x50xf16>

    //CHECK:        [[AVGPOOL0:%.+]] = IE.AvgPool({{[^:]+}})
    //CHECK-SAME:   {kernel_size = [1, 5], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 5]} : tensor<1x128x1x249xf16> -> tensor<1x128x1x49xf16>
    //CHECK:        [[SLICE1:%.+]] = IE.StridedSlice({{[^:]+}})
    //CHECK-SAME:   begins_attr = [0, 0, 0, 245]
    //CHECK-SAME:   ends_attr = [0, 0, 1, 249]
    //CHECK-SAME:   : tensor<1x128x1x249xf16> -> tensor<1x128x1x4xf16>
    //CHECK:        [[AVGPOOL1:%.+]] = IE.AvgPool([[SLICE1]])
    //CHECK-SAME:   {kernel_size = [1, 4], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x128x1x4xf16> -> tensor<1x128x1x1xf16
    //CHECK:        [[CONCAT:%.+]] = IE.Concat([[AVGPOOL0]], [[AVGPOOL1]])
    //CHECK-SAME{LITERAL}:      {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 49]]}
    //CHECK-SAME:   : tensor<1x128x1x49xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x1x50xf16>
    // CHECK        return [[CONCAT]]

}

// -----

// CHECK-LABEL: @HandleExcludePadForAvgPoolPadWithZeroCase2
func.func @HandleExcludePadForAvgPoolPadWithZeroCase2(%arg0 : tensor<1x128x1x10xf16>) -> (tensor<1x128x1x3xf16>) {
    %0 = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [1, 4],
        pads_begin = [0, 0],
        pads_end = [0, 2],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 4]
    } : tensor<1x128x1x10xf16> -> tensor<1x128x1x3xf16>
    return %0 : tensor<1x128x1x3xf16>

    //CHECK:        [[AVGPOOL0:%.+]] = IE.AvgPool({{[^:]+}})
    //CHECK-SAME:   {kernel_size = [1, 4], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 4]} : tensor<1x128x1x10xf16> -> tensor<1x128x1x2xf16>
    //CHECK:        [[SLICE1:%.+]] = IE.StridedSlice({{[^:]+}})
    //CHECK-SAME:   begins_attr = [0, 0, 0, 8]
    //CHECK-SAME:   ends_attr = [0, 0, 1, 10]
    //CHECK-SAME:   : tensor<1x128x1x10xf16> -> tensor<1x128x1x2xf16>
    //CHECK:        [[AVGPOOL1:%.+]] = IE.AvgPool([[SLICE1]])
    //CHECK-SAME:   {kernel_size = [1, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x128x1x2xf16> -> tensor<1x128x1x1xf16>
    //CHECK:        [[CONCAT:%.+]] = IE.Concat([[AVGPOOL0]], [[AVGPOOL1]])
    //CHECK-SAME{LITERAL}:      {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 2]]}
    //CHECK-SAME:   : tensor<1x128x1x2xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x1x3xf16>
    // CHECK        return [[CONCAT]]

}

// -----

// CHECK-LABEL: @HandleExcludePadForAvgPoolPadWithZeroCase3
func.func @HandleExcludePadForAvgPoolPadWithZeroCase3(%arg0 : tensor<1x128x249x1xf16>) -> (tensor<1x128x50x1xf16>) {
    %0 = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [5, 1],
        pads_begin = [0, 0],
        pads_end = [1, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [5, 1]
    } : tensor<1x128x249x1xf16> -> tensor<1x128x50x1xf16>
    return %0 : tensor<1x128x50x1xf16>

    //CHECK:        [[AVGPOOL0:%.+]] = IE.AvgPool({{[^:]+}})
    //CHECK-SAME:   {kernel_size = [5, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [5, 1]} : tensor<1x128x249x1xf16> -> tensor<1x128x49x1xf16>
    //CHECK:        [[SLICE1:%.+]] = IE.StridedSlice({{[^:]+}})
    //CHECK-SAME:   begins_attr = [0, 0, 245, 0]
    //CHECK-SAME:   ends_attr = [0, 0, 249, 1]
    //CHECK-SAME:   : tensor<1x128x249x1xf16> -> tensor<1x128x4x1xf16>
    //CHECK:        [[AVGPOOL1:%.+]] = IE.AvgPool([[SLICE1]])
    //CHECK-SAME:   {kernel_size = [4, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x128x4x1xf16> -> tensor<1x128x1x1xf16>
    //CHECK:        [[CONCAT:%.+]] = IE.Concat([[AVGPOOL0]], [[AVGPOOL1]])
    //CHECK-SAME{LITERAL}:      {static_offsets = [[0, 0, 0, 0], [0, 0, 49, 0]]}
    //CHECK-SAME:   : tensor<1x128x49x1xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x50x1xf16>
    // CHECK        return [[CONCAT]]

}

// -----

// CHECK-LABEL: @HandleExcludePadForAvgPoolPadWithZeroCase4
func.func @HandleExcludePadForAvgPoolPadWithZeroCase4(%arg0 : tensor<1x1024x7x7xf16>) -> (tensor<1x1024x8x3xf16>) {
    %0 = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [3, 3],
        pads_begin = [2, 0],
        pads_end = [1, 3],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 3]
    } : tensor<1x1024x7x7xf16> -> tensor<1x1024x8x3xf16>
    return %0 : tensor<1x1024x8x3xf16>

    //CHECK:        [[AVGPOOL0:%.+]] = IE.AvgPool({{[^:]+}})
    //CHECK-SAME:   {kernel_size = [3, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 3]} : tensor<1x1024x7x7xf16> -> tensor<1x1024x5x2xf16>

    //CHECK:        [[SLICE1:%.+]] = IE.StridedSlice({{[^:]+}})
    //CHECK-SAME:   begins_attr = [0, 0, 0, 6]
    //CHECK-SAME:   ends_attr = [0, 0, 1, 7]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x1x1xf16>
    //CHECK:        [[AVGPOOL1:%.+]] = IE.AvgPool([[SLICE1]])
    //CHECK-SAME:   {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x1x1xf16> -> tensor<1x1024x1x1xf16>

    //CHECK:        [[SLICE2:%.+]] = IE.StridedSlice({{[^:]+}})
    //CHECK-SAME:   begins_attr = [0, 0, 5, 6]
    //CHECK-SAME:   ends_attr = [0, 0, 7, 7]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x2x1xf16>
    //CHECK:        [[AVGPOOL2:%.+]] = IE.AvgPool([[SLICE2]])
    //CHECK-SAME:   {kernel_size = [2, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x2x1xf16> -> tensor<1x1024x1x1xf16>

    //CHECK:        [[SLICE3:%.+]] = IE.StridedSlice({{[^:]+}})
    //CHECK-SAME:   begins_attr = [0, 0, 0, 6]
    //CHECK-SAME:   ends_attr = [0, 0, 7, 7]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x1xf16>
    //CHECK:        [[AVGPOOL3:%.+]] = IE.AvgPool([[SLICE3]])
    //CHECK-SAME:   {kernel_size = [3, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x7x1xf16> -> tensor<1x1024x5x1xf16>

    //CHECK:        [[SLICE4:%.+]] = IE.StridedSlice({{[^:]+}})
    //CHECK-SAME:   begins_attr = [0, 0, 0, 0]
    //CHECK-SAME:   ends_attr = [0, 0, 1, 7]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x1x7xf16>
    //CHECK:        [[AVGPOOL4:%.+]] = IE.AvgPool([[SLICE4]])
    //CHECK-SAME:   {kernel_size = [1, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 3]} : tensor<1x1024x1x7xf16> -> tensor<1x1024x1x2xf16>

    //CHECK:        [[SLICE5:%.+]] = IE.StridedSlice({{[^:]+}})
    //CHECK-SAME:   begins_attr = [0, 0, 5, 0]
    //CHECK-SAME:   ends_attr = [0, 0, 7, 7]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x2x7xf16>
    //CHECK:        [[AVGPOOL5:%.+]] = IE.AvgPool([[SLICE5]])
    //CHECK-SAME:   {kernel_size = [2, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 3]} : tensor<1x1024x2x7xf16> -> tensor<1x1024x1x2xf16>

    //CHECK:        [[CONCAT:%.+]] = IE.Concat([[AVGPOOL0]], [[AVGPOOL1]], [[AVGPOOL2]], [[AVGPOOL3]], [[AVGPOOL4]], [[AVGPOOL5]])
    //CHECK-SAME{LITERAL}:      static_offsets = [[0, 0, 1, 0], [0, 0, 0, 2], [0, 0, 7, 2], [0, 0, 1, 2], [0, 0, 0, 0], [0, 0, 7, 0]]}
    //CHECK-SAME:   : tensor<1x1024x5x2xf16>, tensor<1x1024x1x1xf16>, tensor<1x1024x1x1xf16>, tensor<1x1024x5x1xf16>, tensor<1x1024x1x2xf16>, tensor<1x1024x1x2xf16> -> tensor<1x1024x8x3xf16>

    // CHECK        return [[CONCAT]]
}

// -----

// CHECK-LABEL: @HandleExcludePadForAvgPoolPadWithZeroCase5
func.func @HandleExcludePadForAvgPoolPadWithZeroCase5(%arg0 : tensor<1x256x96x32xf16>) -> (tensor<1x256x48x16xf16>) {
    %0 = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [2, 2],
        pads_begin = [0, 0],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [2, 2]
    } : tensor<1x256x96x32xf16> -> tensor<1x256x48x16xf16>
    return %0 : tensor<1x256x48x16xf16>

    //CHECK:        [[AVGPOOL:%.+]] = IE.AvgPool({{[^:]+}})
    //CHECK-SAME:   {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x256x96x32xf16> -> tensor<1x256x48x16xf16>

    // CHECK        return [[AVGPOOL]]
}
