//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// An unbounded input makes the output unbounded, even when the other input has bounds.

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @UnboundedAndBoundedInputs
func.func @UnboundedAndBoundedInputs(
        %lhs: tensor<1x16x?x32xf16>,
        %rhs: tensor<1x16x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 32]> : tensor<4xsi64>, order = #NCHW}>)
            -> tensor<1x16x?x32xf16> {
    %0 = IE.Multiply(%lhs, %rhs) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x16x?x32xf16>,
          tensor<1x16x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 32]> : tensor<4xsi64>, order = #NCHW}>
        -> tensor<1x16x?x32xf16>
    return %0 : tensor<1x16x?x32xf16>

    // CHECK:       IE.Multiply
    // CHECK-SAME:      -> tensor<1x16x?x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @BoundedAndUnboundedBroadcastInputs
func.func @BoundedAndUnboundedBroadcastInputs(
        %lhs: tensor<1x16x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 1]> : tensor<4xsi64>, order = #NCHW}>,
        %rhs: tensor<1x16x?x?xf16>)
            -> tensor<1x16x?x?xf16> {
    %0 = IE.Multiply(%lhs, %rhs) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x16x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 1]> : tensor<4xsi64>, order = #NCHW}>,
          tensor<1x16x?x?xf16>
        -> tensor<1x16x?x?xf16>
    return %0 : tensor<1x16x?x?xf16>

    // CHECK:       IE.Multiply
    // CHECK-SAME:      -> tensor<1x16x?x?xf16>
}
