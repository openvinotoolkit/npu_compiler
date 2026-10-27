//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt %s --split-input-file --init-compiler="platform=%platform%" --verify-diagnostics
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Positive: dynamic shape with matching bounds.
func.func @BoundsMatchDynamicShape(
    %arg0: tensor<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>)
        -> tensor<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}> {
    return %arg0 : tensor<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Positive: a dynamic shape without bounds stays legal. Unlike MemRefAttr, a TensorAttr does not
// require bounds on a dynamic shape - IE::GatherOp infers exactly this type when neither of its
// inputs is bounded.
func.func @DynamicShapeWithoutBounds(%arg0: tensor<1x?x3x3xf32, {order = #NCHW}>)
        -> tensor<1x?x3x3xf32, {order = #NCHW}> {
    return %arg0 : tensor<1x?x3x3xf32, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Positive: the dynamic dims mask is the other dynamic representation. It pairs a static shape with
// a mask and carries no bounds, so the bounds checks must leave it alone.
func.func @StaticShapeWithDynamicDimsMask(
    %arg0: tensor<1x16x48x400xf32, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]> : tensor<4xsi64>, order = #NCHW}>)
        -> tensor<1x16x48x400xf32, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]> : tensor<4xsi64>, order = #NCHW}> {
    return %arg0 : tensor<1x16x48x400xf32, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Positive, and deliberately so, but not because a static shape carrying bounds is right.
// Core::BoundedTensorType::get refuses to build such a type, while its classof only tests that the
// bounds vector is non-empty. So isa<Core::BoundedTensorType> answers true for a type that nothing
// can construct. The rest of the compiler produces that state and depends on it: VPU.Slice narrows a
// bounded dynamic input to a static tile and keeps the bounds, which dialect/VPU/ops/slice.mlir
// @SliceWithBoundedDynamicInput asserts, and the tiling and bufferization suites read the bounds off
// such a type to size a buffer. Rejecting it here breaks 24 tests. Pinned so that a later attempt to
// make the verifier match get() shows up as this test failing, and not as a suite-wide breakage.
// TODO: E#233281 fix the producers so a static result drops its bounds, then make this a negative
// case and let the verifier match get().
func.func @BoundsOnStaticShapeStayLegal(
    %arg0: tensor<1x16x48x400xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 400]> : tensor<4xsi64>, order = #NCHW}>)
        -> tensor<1x16x48x400xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 400]> : tensor<4xsi64>, order = #NCHW}> {
    return %arg0 : tensor<1x16x48x400xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 400]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Positive, and deliberately so, but not because the bound on dim 1 means anything here. A bound is
// only required to have the rank of the shape. pipelines/build_output_shape_func.mlir carries this
// exact type, where the batch bound outlives the dimension the debatcher made static, so an equality
// check would reject IR that the compiler produces today.
// TODO: E#233282 make the debatcher reduce the bound with the dimension it makes static, then make
// this a negative case and require a bound to equal a static dimension.
func.func @BoundExceedsStaticDim(
    %arg0: tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 16, 64]> : tensor<4xsi64>, order = #NCHW}>)
        -> tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 16, 64]> : tensor<4xsi64>, order = #NCHW}> {
    return %arg0 : tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 16, 64]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Negative: bounds rank does not match shape rank.
// expected-error@+2 {{TensorType bounds '[1, 18, 3]' do not match to shape}}
func.func @BoundsRankMismatch(
    %arg0: tensor<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3]> : tensor<3xsi64>, order = #NCHW}>)
        -> tensor<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3]> : tensor<3xsi64>, order = #NCHW}> {
    return %arg0 : tensor<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3]> : tensor<3xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Negative: a bound entry is negative.
// expected-error@+2 {{has negative value '-1' at dim '1'}}
func.func @NegativeBoundValue(
    %arg0: tensor<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, -1, 3, 3]> : tensor<4xsi64>, order = #NCHW}>)
        -> tensor<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, -1, 3, 3]> : tensor<4xsi64>, order = #NCHW}> {
    return %arg0 : tensor<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, -1, 3, 3]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#BadRank = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Negative: an order map with more results than the shape has dimensions. This case also pins the
// attachment point. The check only reaches a tensor type when the interface is attached to
// TensorAttr from a builtin-dialect extension, because this file loads no dialect but func and
// builtin, and an attach from CoreDialect::initialize() never runs on it.
// expected-error@+2 {{doesn't match to shape}}
func.func @OrderRankMismatch(
    %arg0: tensor<1x64xf32, {order = #BadRank}>)
        -> tensor<1x64xf32, {order = #BadRank}> {
    return %arg0 : tensor<1x64xf32, {order = #BadRank}>
}
