//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt %s --split-input-file --init-compiler="platform=%platform%" --verify-diagnostics
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Positive: dynamic shape with matching bounds is accepted by MemRefAttrLayout::verifyLayout.
func.func @BoundsMatchDynamicShape(
    %arg0: memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>)
        -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}> {
    return %arg0 : memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Negative: dynamic shape without bounds is rejected.
// expected-error@+1 {{requires bounds in MemRefAttr}}
func.func @DynamicShapeRequiresBounds(%arg0: memref<1x?x3x3xf32, {order = #NCHW}>)
        -> memref<1x?x3x3xf32, {order = #NCHW}> {
    return %arg0 : memref<1x?x3x3xf32, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Negative: bounds rank does not match shape rank.
// expected-error@+2 {{Bounds '[1, 18, 3]' do not match with shape}}
func.func @BoundsRankMismatch(
    %arg0: memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3]> : tensor<3xsi64>, order = #NCHW}>)
        -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3]> : tensor<3xsi64>, order = #NCHW}> {
    return %arg0 : memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3]> : tensor<3xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Negative: a static dim disagrees with its bound entry.
// expected-error@+2 {{at dim '1': expected '4', got '18'}}
func.func @BoundContradictsStaticDim(
    %arg0: memref<1x4x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}>)
        -> memref<1x4x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}> {
    return %arg0 : memref<1x4x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Negative: a bound entry is negative.
// expected-error@+2 {{has negative value '-1' at dim '1'}}
func.func @NegativeBoundValue(
    %arg0: memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, -1, 3, 3]> : tensor<4xsi64>, order = #NCHW}>)
        -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, -1, 3, 3]> : tensor<4xsi64>, order = #NCHW}> {
    return %arg0 : memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, -1, 3, 3]> : tensor<4xsi64>, order = #NCHW}>
}
