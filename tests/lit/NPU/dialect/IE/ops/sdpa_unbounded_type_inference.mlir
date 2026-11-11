//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// A sequence dimension without an upper bound does not count as a K/V mismatch, so V keeps its row-major layout.

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @BoundedKeyUnboundedRowMajorValue
func.func @BoundedKeyUnboundedRowMajorValue(
        %query: tensor<1x8x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 32, 64]> : tensor<4xsi64>, order = #NCHW}>,
        %key: tensor<1x8x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 128, 64]> : tensor<4xsi64>, order = #NCHW}>,
        %value: tensor<1x8x?x96xf16>)
            -> tensor<1x8x?x96xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 32, 96]> : tensor<4xsi64>, order = #NCHW}> {
    %0 = IE.SDPA(%query, %key, %value) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>}
        : tensor<1x8x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 32, 64]> : tensor<4xsi64>, order = #NCHW}>,
          tensor<1x8x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 128, 64]> : tensor<4xsi64>, order = #NCHW}>,
          tensor<1x8x?x96xf16>
        -> tensor<1x8x?x96xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 32, 96]> : tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x8x?x96xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 32, 96]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:       IE.SDPA
    // CHECK-SAME:      -> tensor<1x8x?x96xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 32, 96]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

// The unbounded sequence dimension of a pre-transposed V is its last one, so Ev is taken from the one before.

// CHECK-LABEL: @UnboundedTransposedValue
func.func @UnboundedTransposedValue(%query: tensor<1x8x?x64xf16>, %key: tensor<1x8x?x64xf16>, %value: tensor<1x8x96x?xf16>)
        -> tensor<1x8x?x96xf16> {
    %0 = IE.SDPA(%query, %key, %value) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>}
        : tensor<1x8x?x64xf16>, tensor<1x8x?x64xf16>, tensor<1x8x96x?xf16> -> tensor<1x8x?x96xf16>
    return %0 : tensor<1x8x?x96xf16>

    // CHECK:       IE.SDPA
    // CHECK-SAME:      -> tensor<1x8x?x96xf16>
}

// -----

// CHECK-LABEL: @UnboundedRowMajorValue
func.func @UnboundedRowMajorValue(%query: tensor<1x8x?x64xf16>, %key: tensor<1x8x?x64xf16>, %value: tensor<1x8x?x96xf16>)
        -> tensor<1x8x?x96xf16> {
    %0 = IE.SDPA(%query, %key, %value) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>}
        : tensor<1x8x?x64xf16>, tensor<1x8x?x64xf16>, tensor<1x8x?x96xf16> -> tensor<1x8x?x96xf16>
    return %0 : tensor<1x8x?x96xf16>

    // CHECK:       IE.SDPA
    // CHECK-SAME:      -> tensor<1x8x?x96xf16>
}
