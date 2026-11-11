//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @Eliminate
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<4x4xf32>
func.func @Eliminate(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [4, 4] } : tensor<4x4xf32> -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>

    // CHECK-NOT: IE.Reshape
    // CHECK:     return [[ARG_0]]
}

// -----

// CHECK-LABEL: @ConstFold
func.func @ConstFold() -> tensor<4x4xf32> {
    %0 = const.Declare tensor<16xf32> = dense<1.0> : tensor<16xf32>
    %1 = IE.Reshape(%0) { shape_value = [4, 4] } : tensor<16xf32> -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>

    // CHECK-DAG:       [[VAL0:%.+]] = const.Declare tensor<4x4xf32> =
    // CHECK-SAME:      dense<1.000000e+00> : tensor<16xf32>, [#const.Reshape<[4, 4]>]
    // CHECK-NOT:   IE.Reshape
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: @FuseReshapes
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<16x1xf32>
func.func @FuseReshapes(%arg0: tensor<16x1xf32>) -> tensor<4x4xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [1, 1, 4, 4] } : tensor<16x1xf32> -> tensor<1x1x4x4xf32>
    %1 = IE.Reshape(%0) { shape_value = [4, 4] } : tensor<1x1x4x4xf32> -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>

    // CHECK: [[VAL0:%.+]] = IE.Reshape([[ARG_0]]) {shape_value = [4, 4]} : tensor<16x1xf32> -> tensor<4x4xf32>
    // CHECK: return [[VAL0]] : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: @ConvertToAffineReshape
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x1x9x16x2xf32>
func.func @ConvertToAffineReshape(%arg0: tensor<1x1x9x16x2xf32>) -> tensor<1x3x3x32xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [1, 3, 3, 32] } : tensor<1x1x9x16x2xf32> -> tensor<1x3x3x32xf32>
    return %0 : tensor<1x3x3x32xf32>

    // CHECK: [[VAL0:%.+]] = IE.AffineReshape([[ARG_0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1, 2], [3], [3]], shape_value = [1, 3, 3, 32]} : tensor<1x1x9x16x2xf32> -> tensor<1x3x3x32xf32>
    // CHECK: return [[VAL0]] : tensor<1x3x3x32xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @DontConvertToAffineReshape
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x1x1x160000xf16, {order = #NHWC}>
// Because there's no explicit valid layout infered based on the input layout and dim mapping
func.func @DontConvertToAffineReshape(%arg0: tensor<1x1x1x160000xf16, {order = #NHWC}>) -> tensor<1x2500x1x64xf16> {
    %0 = IE.Reshape(%arg0) {shape_value = [1, 2500, 1, 64]} : tensor<1x1x1x160000xf16, {order = #NHWC}> -> tensor<1x2500x1x64xf16>
    return %0 : tensor<1x2500x1x64xf16>

    // CHECK: [[VAL0:%.+]] = IE.Reshape([[ARG_0]])
    // CHECK-SAME{LITERAL}: {shape_value = [1, 2500, 1, 64]} : tensor<1x1x1x160000xf16, {order = #NHWC}> -> tensor<1x2500x1x64xf16>
    // CHECK: return [[VAL0]] : tensor<1x2500x1x64xf16>
}

// -----

// An unbounded dim is associated only with an unbounded dim, so the Reshape becomes an AffineReshape.

// CHECK-LABEL: @ConvertUnboundedToAffineReshape
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x?x64xf16>
func.func @ConvertUnboundedToAffineReshape(%arg0: tensor<1x?x64xf16>) -> tensor<1x1x?x8x8xf16> {
    %0 = IE.Reshape(%arg0) {shape_value = [1, 1, -9223372036854775808, 8, 8]} : tensor<1x?x64xf16> -> tensor<1x1x?x8x8xf16>
    return %0 : tensor<1x1x?x8x8xf16>

    // CHECK: [[VAL0:%.+]] = IE.AffineReshape([[ARG_0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3, 4]], shape_value = [1, 1, -9223372036854775808, 8, 8]} : tensor<1x?x64xf16> -> tensor<1x1x?x8x8xf16>
    // CHECK: return [[VAL0]] : tensor<1x1x?x8x8xf16>
}

// -----

// Merging an unbounded dim with a static one has no affine form, so the Reshape is kept.

// CHECK-LABEL: @DontConvertUnboundedMergeToAffineReshape
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<?x64xf16>
func.func @DontConvertUnboundedMergeToAffineReshape(%arg0: tensor<?x64xf16>) -> tensor<?xf16> {
    %0 = IE.Reshape(%arg0) {shape_value = [-9223372036854775808]} : tensor<?x64xf16> -> tensor<?xf16>
    return %0 : tensor<?xf16>

    // CHECK: [[VAL0:%.+]] = IE.Reshape([[ARG_0]])
    // CHECK-SAME{LITERAL}: {shape_value = [-9223372036854775808]} : tensor<?x64xf16> -> tensor<?xf16>
    // CHECK: return [[VAL0]] : tensor<?xf16>
}

// -----

// A dim of 1 next to an unbounded dim is squeezed into it.

// CHECK-LABEL: @ConvertUnboundedSqueezeToAffineReshape
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x?x64xf16>
func.func @ConvertUnboundedSqueezeToAffineReshape(%arg0: tensor<1x?x64xf16>) -> tensor<?x64xf16> {
    %0 = IE.Reshape(%arg0) {shape_value = [-9223372036854775808, 64]} : tensor<1x?x64xf16> -> tensor<?x64xf16>
    return %0 : tensor<?x64xf16>

    // CHECK: [[VAL0:%.+]] = IE.AffineReshape([[ARG_0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1]], shape_value = [-9223372036854775808, 64]} : tensor<1x?x64xf16> -> tensor<?x64xf16>
    // CHECK: return [[VAL0]] : tensor<?x64xf16>
}

// -----

// A dim of 1 is unsqueezed next to an unbounded dim.

// CHECK-LABEL: @ConvertUnboundedUnsqueezeToAffineReshape
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<?x64xf16>
func.func @ConvertUnboundedUnsqueezeToAffineReshape(%arg0: tensor<?x64xf16>) -> tensor<1x?x64xf16> {
    %0 = IE.Reshape(%arg0) {shape_value = [1, -9223372036854775808, 64]} : tensor<?x64xf16> -> tensor<1x?x64xf16>
    return %0 : tensor<1x?x64xf16>

    // CHECK: [[VAL0:%.+]] = IE.AffineReshape([[ARG_0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2]], shape_value = [1, -9223372036854775808, 64]} : tensor<?x64xf16> -> tensor<1x?x64xf16>
    // CHECK: return [[VAL0]] : tensor<1x?x64xf16>
}
