//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --resolve-strided-slice %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ResolveStridedSliceWithStride
func @ResolveStridedSliceWithStride(%arg0: tensor<1x10x20x30xf16>) -> tensor<1x5x5x5xf16> {
    %0 = IE.StridedSlice(%arg0) {
        begins_attr = [0, 0, 0, 15],
        ends_attr = [1, 5, 10, 20],
        strides_attr = [1, 1, 2, 1],
        begin_mask = [0, 1, 1, 0],
        end_mask = [1, 0, 0, 0],
        new_axis_mask = [0, 0, 0, 0],
        shrink_axis_mask = [0, 0, 0, 0],
        ellipsis_mask = [0, 0, 0, 0],
        operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>
    } : tensor<1x10x20x30xf16> -> tensor<1x5x5x5xf16>

    return %0 : tensor<1x5x5x5xf16>
    // CHECK:       %[[VAL0:.*]] = IE.StridedSlice(%arg0)

    // Only attributes with name *_attr could have values != 0
    // CHECK-SAME:  begin_mask = [0, 0, 0, 0]
    // CHECK-SAME:  begins_attr = [0, 0, 0, 15]
    // CHECK-SAME:  ellipsis_mask = [0, 0, 0, 0]
    // CHECK-SAME:  end_mask = [0, 0, 0, 0]
    // CHECK-SAME:  ends_attr = [1, 5, 10, 20]
    // CHECK-SAME:  new_axis_mask = [0, 0, 0, 0]
    // CHECK-SAME:  shrink_axis_mask = [0, 0, 0, 0]
    // CHECK-SAME:  strides_attr = [1, 1, 2, 1]
}

// CHECK-LABEL: @ResolveStridedSliceWoutStride
func @ResolveStridedSliceWoutStride(%arg0: tensor<1x10x20x30xf16>) -> tensor<1x5x10x5xf16> {
    %0 = IE.StridedSlice(%arg0) {
        begins_attr = [0, 0, 0, 15],
        ends_attr = [1, 5, 10, 20],
        strides_attr = [1, 1, 1, 1],
        begin_mask = [0, 1, 1, 0],
        end_mask = [1, 0, 0, 0],
        new_axis_mask = [0, 0, 0, 0],
        shrink_axis_mask = [0, 0, 0, 0],
        ellipsis_mask = [0, 0, 0, 0],
        operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>
    } : tensor<1x10x20x30xf16> -> tensor<1x5x10x5xf16>

    return %0 : tensor<1x5x10x5xf16>

    // CHECK:       %[[VAL0:.*]] = IE.Slice %arg0 [0, 0, 0, 15] [1, 5, 10, 5]
    // CHECK-NOT:   IE.StridedSlice
}

// CHECK-LABEL: @ResolveStridedSliceNegStride
func @ResolveStridedSliceNegStride(%arg0: tensor<1x10x20x30xf16>) -> tensor<1x5x5x5xf16> {
    %0 = IE.StridedSlice(%arg0) {
        begins_attr = [0, 0, 0, 15],
        ends_attr = [1, 5, 10, 20],
        strides_attr = [1, 1, -2, 1],
        begin_mask = [0, 1, 1, 0],
        end_mask = [1, 0, 0, 0],
        new_axis_mask = [0, 0, 0, 0],
        shrink_axis_mask = [0, 0, 0, 0],
        ellipsis_mask = [0, 0, 0, 0],
        operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>
    } : tensor<1x10x20x30xf16> -> tensor<1x5x5x5xf16>

    return %0 : tensor<1x5x5x5xf16>

    // CHECK:       %[[VAL0:.*]] = IE.StridedSlice(%arg0)

    // Only attributes with name *_attr could have values != 0
    // CHECK-SAME:  begin_mask = [0, 0, 0, 0]
    // CHECK-SAME:  begins_attr = [0, 0, 11, 15]
    // CHECK-SAME:  ellipsis_mask = [0, 0, 0, 0]
    // CHECK-SAME:  end_mask = [0, 0, 0, 0]
    // CHECK-SAME:  ends_attr = [1, 5, 20, 20]
    // CHECK-SAME:  new_axis_mask = [0, 0, 0, 0]
    // CHECK-SAME:  shrink_axis_mask = [0, 0, 0, 0]
    // CHECK-SAME:  strides_attr = [1, 1, 2, 1]
}
