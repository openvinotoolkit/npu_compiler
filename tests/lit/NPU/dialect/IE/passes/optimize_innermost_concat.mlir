//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-innermost-concat %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeInnermostConcatNCHW
// CHECK-SAME: ([[ARG0:%arg[0-9]+]]: tensor<1x256x2048x1xf16>,
// CHECK-SAME:  [[ARG1:%arg[0-9]+]]: tensor<1x256x2048x1xf16>,
// CHECK-SAME:  [[ARG2:%arg[0-9]+]]: tensor<1x256x2048x1xf16>,
// CHECK-SAME:  [[ARG3:%arg[0-9]+]]: tensor<1x256x2048x1xf16>)
func.func @OptimizeInnermostConcatNCHW(%arg0: tensor<1x256x2048x1xf16>, %arg1: tensor<1x256x2048x1xf16>,
                                       %arg2: tensor<1x256x2048x1xf16>, %arg3: tensor<1x256x2048x1xf16>) -> tensor<1x256x2048x4xf16> {
    %0 = IE.Concat(%arg0, %arg1, %arg2, %arg3) {
        static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3]]
    } : tensor<1x256x2048x1xf16>, tensor<1x256x2048x1xf16>, tensor<1x256x2048x1xf16>, tensor<1x256x2048x1xf16> -> tensor<1x256x2048x4xf16>

    return %0 : tensor<1x256x2048x4xf16>

    // CHECK: [[PERMUTE_CAST0:%.+]] = IE.PermuteCast([[ARG0]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x256x2048x1xf16> -> tensor<1x1x256x2048xf16>
    // CHECK: [[PERMUTE_CAST1:%.+]] = IE.PermuteCast([[ARG1]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x256x2048x1xf16> -> tensor<1x1x256x2048xf16>
    // CHECK: [[PERMUTE_CAST2:%.+]] = IE.PermuteCast([[ARG2]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x256x2048x1xf16> -> tensor<1x1x256x2048xf16>
    // CHECK: [[PERMUTE_CAST3:%.+]] = IE.PermuteCast([[ARG3]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x256x2048x1xf16> -> tensor<1x1x256x2048xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[PERMUTE_CAST0]], [[PERMUTE_CAST1]], [[PERMUTE_CAST2]], [[PERMUTE_CAST3]])
    // CHECK-SAME(LITERAL):     {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]]} : tensor<1x1x256x2048xf16>, tensor<1x1x256x2048xf16>, tensor<1x1x256x2048xf16>, tensor<1x1x256x2048xf16> -> tensor<1x4x256x2048xf16>
    // CHECK: [[PERMUTE:%.+]] = IE.MemPermute([[CONCAT]]) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x4x256x2048xf16> -> tensor<1x256x2048x4xf16>

    // CHECK: return [[PERMUTE]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeInnermostConcatNHWC
// CHECK-SAME: ([[ARG0:%arg[0-9]+]]: tensor<1x1x2048x256xf16, {order = #NHWC}>,
// CHECK-SAME:  [[ARG1:%arg[0-9]+]]: tensor<1x1x2048x256xf16, {order = #NHWC}>,
// CHECK-SAME:  [[ARG2:%arg[0-9]+]]: tensor<1x1x2048x256xf16, {order = #NHWC}>,
// CHECK-SAME:  [[ARG3:%arg[0-9]+]]: tensor<1x1x2048x256xf16, {order = #NHWC}>)
func.func @OptimizeInnermostConcatNHWC(%arg0: tensor<1x1x2048x256xf16, {order = #NHWC}>, %arg1: tensor<1x1x2048x256xf16, {order = #NHWC}>,
                                       %arg2: tensor<1x1x2048x256xf16, {order = #NHWC}>, %arg3: tensor<1x1x2048x256xf16, {order = #NHWC}>) -> tensor<1x4x2048x256xf16, {order = #NHWC}> {
    %0 = IE.Concat(%arg0, %arg1, %arg2, %arg3) {
        static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]]
    } : tensor<1x1x2048x256xf16, {order = #NHWC}>, tensor<1x1x2048x256xf16, {order = #NHWC}>, tensor<1x1x2048x256xf16, {order = #NHWC}>, tensor<1x1x2048x256xf16, {order = #NHWC}> -> tensor<1x4x2048x256xf16, {order = #NHWC}>

    return %0 : tensor<1x4x2048x256xf16, {order = #NHWC}>

    // CHECK: [[PERMUTE_CAST0:%.+]] = IE.PermuteCast([[ARG0]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x2048x256xf16, {order = #NHWC}> -> tensor<1x1x2048x256xf16>
    // CHECK: [[PERMUTE_CAST1:%.+]] = IE.PermuteCast([[ARG1]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x2048x256xf16, {order = #NHWC}> -> tensor<1x1x2048x256xf16>
    // CHECK: [[PERMUTE_CAST2:%.+]] = IE.PermuteCast([[ARG2]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x2048x256xf16, {order = #NHWC}> -> tensor<1x1x2048x256xf16>
    // CHECK: [[PERMUTE_CAST3:%.+]] = IE.PermuteCast([[ARG3]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x2048x256xf16, {order = #NHWC}> -> tensor<1x1x2048x256xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[PERMUTE_CAST0]], [[PERMUTE_CAST1]], [[PERMUTE_CAST2]], [[PERMUTE_CAST3]])
    // CHECK-SAME(LITERAL):     {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]]} : tensor<1x1x2048x256xf16>, tensor<1x1x2048x256xf16>, tensor<1x1x2048x256xf16>, tensor<1x1x2048x256xf16> -> tensor<1x4x2048x256xf16>
    // CHECK: [[PERMUTE:%.+]] = IE.MemPermute([[CONCAT]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x2048x256xf16> -> tensor<1x4x2048x256xf16, {order = #NHWC}>

    // CHECK: return [[PERMUTE]]
}

// -----

// CHECK-LABEL: @SkipNonInnermostConcat
// CHECK-SAME: ([[ARG0:%arg[0-9]+]]: tensor<1x256x2048x4xf16>,
// CHECK-SAME:  [[ARG1:%arg[0-9]+]]: tensor<1x256x2048x4xf16>)
func.func @SkipNonInnermostConcat(%arg0: tensor<1x256x2048x4xf16>, %arg1: tensor<1x256x2048x4xf16>) -> tensor<1x512x2048x4xf16> {
    %0 = IE.Concat(%arg0, %arg1) {
        static_offsets = [[0, 0, 0, 0], [0, 256, 0, 0]]
    } : tensor<1x256x2048x4xf16>, tensor<1x256x2048x4xf16> -> tensor<1x512x2048x4xf16>

    return %0 : tensor<1x512x2048x4xf16>

    // CHECK: [[CONCAT:%.+]] = IE.Concat([[ARG0]], [[ARG1]])
    // CHECK: return [[CONCAT]]
}

// -----

// CHECK-LABEL: @SkipNonUnitInnermostDim
// CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x256x2048x2xf16>,
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<1x256x2048x2xf16>)
func.func @SkipNonUnitInnermostDim(%arg0: tensor<1x256x2048x2xf16>, %arg1: tensor<1x256x2048x2xf16>) -> tensor<1x256x2048x4xf16> {
    %0 = IE.Concat(%arg0, %arg1) {
        static_offsets = [[0, 0, 0, 0], [0, 0, 0, 2]]
    } : tensor<1x256x2048x2xf16>, tensor<1x256x2048x2xf16> -> tensor<1x256x2048x4xf16>

    return %0 : tensor<1x256x2048x4xf16>

    // CHECK: [[CONCAT:%.+]] = IE.Concat([[INPUT0]], [[INPUT1]])
    // CHECK: return [[CONCAT]]
}

// -----

// CHECK-LABEL: @SkipConcatAllNonConcatDimsAreUnit
// CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x1x1x1xf16>,
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<1x1x1x1xf16>)
func.func @SkipConcatAllNonConcatDimsAreUnit(%arg0: tensor<1x1x1x1xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x1x1x2xf16> {
    %0 = IE.Concat(%arg0, %arg1) {
        static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1]]
    } : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x2xf16>

    return %0 : tensor<1x1x1x2xf16>

    // All non-concat dimensions (N, C, H) are unit size, optimization should be skipped
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[INPUT0]], [[INPUT1]])
    // CHECK: return [[CONCAT]]
}

// -----

// CHECK-LABEL: @AllowConcatPartialNonConcatDimsAreUnit
// CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x1x256x1xf16>,
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<1x1x256x1xf16>)
func.func @AllowConcatPartialNonConcatDimsAreUnit(%arg0: tensor<1x1x256x1xf16>, %arg1: tensor<1x1x256x1xf16>) -> tensor<1x1x256x2xf16> {
    %0 = IE.Concat(%arg0, %arg1) {
        static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1]]
    } : tensor<1x1x256x1xf16>, tensor<1x1x256x1xf16> -> tensor<1x1x256x2xf16>

    return %0 : tensor<1x1x256x2xf16>

    // H dimension (256) is non-unit, so optimization should proceed
    // CHECK: [[PERMUTE_CAST0:%.+]] = IE.PermuteCast([[INPUT0]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x256x1xf16> -> tensor<1x1x1x256xf16>
    // CHECK: [[PERMUTE_CAST1:%.+]] = IE.PermuteCast([[INPUT1]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x256x1xf16> -> tensor<1x1x1x256xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[PERMUTE_CAST0]], [[PERMUTE_CAST1]])
    // CHECK: [[PERMUTE:%.+]] = IE.MemPermute([[CONCAT]]) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x2x1x256xf16> -> tensor<1x1x256x2xf16>
    // CHECK: return [[PERMUTE]]
}
