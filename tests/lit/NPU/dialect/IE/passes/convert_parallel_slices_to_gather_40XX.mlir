//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-parallel-slices-to-gather %s | FileCheck %s
// REQUIRES: arch-NPU40XX
// CHECK-LABEL: @NotConvertParallelSliceBranchesToPreventGatherWithTiling
func.func @NotConvertParallelSliceBranchesToPreventGatherWithTiling(
                %arg0: tensor<8x61440x1x1xf16>,
                %arg1 : tensor<4x30720x1x1xf16>) -> tensor<1x12x1x7680xf16> {
    // group 0
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 61440]} : tensor<8x61440x1x1xf16> -> tensor<8x61440xf16>
    %1 = IE.Slice %0 [0, 0] [1, 7680] : tensor<8x61440xf16> to tensor<1x7680xf16>
    %2 = IE.Slice %0 [1, 7680] [1, 7680] : tensor<8x61440xf16> to tensor<1x7680xf16>
    %3 = IE.Slice %0 [2, 15360] [1, 7680] : tensor<8x61440xf16> to tensor<1x7680xf16>
    %4 = IE.Slice %0 [3, 23040] [1, 7680] : tensor<8x61440xf16> to tensor<1x7680xf16>
    %5 = IE.Slice %0 [4, 30720] [1, 7680] : tensor<8x61440xf16> to tensor<1x7680xf16>
    %6 = IE.Slice %0 [5, 38400] [1, 7680] : tensor<8x61440xf16> to tensor<1x7680xf16>
    %7 = IE.Slice %0 [6, 46080] [1, 7680] : tensor<8x61440xf16> to tensor<1x7680xf16>
    %8 = IE.Slice %0 [7, 53760] [1, 7680] : tensor<8x61440xf16> to tensor<1x7680xf16>

    // group 1
    %9 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [1], [1]], shape_value = [4, 30720]} : tensor<4x30720x1x1xf16> -> tensor<4x30720xf16>
    %10 = IE.Slice %9 [0, 0] [1, 7680] : tensor<4x30720xf16> to tensor<1x7680xf16>
    %11 = IE.Slice %9 [1, 7680] [1, 7680] : tensor<4x30720xf16> to tensor<1x7680xf16>
    %12 = IE.Slice %9 [2, 15360] [1, 7680] : tensor<4x30720xf16> to tensor<1x7680xf16>
    %13 = IE.Slice %9 [3, 23040] [1, 7680] : tensor<4x30720xf16> to tensor<1x7680xf16>

    // concat
    %14 = IE.Concat(%1, %2, %3, %4, %5, %6, %7, %8, %10, %11, %12, %13)
            {static_offsets = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0]]
            } : tensor<1x7680xf16>, tensor<1x7680xf16>, tensor<1x7680xf16>, tensor<1x7680xf16>, tensor<1x7680xf16>, tensor<1x7680xf16>, tensor<1x7680xf16>, tensor<1x7680xf16>,
                tensor<1x7680xf16>, tensor<1x7680xf16>, tensor<1x7680xf16>, tensor<1x7680xf16> -> tensor<12x7680xf16>
    %15 = IE.AffineReshape(%14) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 12, 1, 7680]} : tensor<12x7680xf16> -> tensor<1x12x1x7680xf16>

    return %15 : tensor<1x12x1x7680xf16>

    // CHECK-NOT:       IE.Gather
}
