//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// These cases cover the VPU side of the rule that the IE dialect's own reduce.mlir covers for
// IE::inferReduceReturnTypeComponents. VPU_LayerOp verifies the declared result type against the
// inferred one, so a wrong type fails when the file is parsed.

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InBoundedType = tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 400]> : tensor<4xsi64>, order = #NCHW}>

// A global mean over the dynamic height and width leaves a static shape, so the result must carry no
// bounds. An implementation that gates the bounds on the input alone fails here.

// CHECK-LABEL: @ReduceMeanOverEveryDynamicAxisHasNoBounds
func.func @ReduceMeanOverEveryDynamicAxisHasNoBounds(%IN: !InBoundedType) -> tensor<1x16x1x1xf16> {
    %REDUCE = VPU.ReduceMean(%IN) {axes_value = [2, 3], keep_dims} : !InBoundedType -> tensor<1x16x1x1xf16>
    return %REDUCE : tensor<1x16x1x1xf16>

    // CHECK:       VPU.ReduceMean
    // CHECK-SAME:  -> tensor<1x16x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InBoundedType = tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 400]> : tensor<4xsi64>, order = #NCHW}>
!OutBoundedType = tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 48, 400]> : tensor<4xsi64>, order = #NCHW}>

// A Reduce that leaves a dynamic dimension keeps its bounds. Without them the dimension has nothing
// behind it, and the next operation that broadcasts compares a real extent against INT64_MIN.

// CHECK-LABEL: @ReduceMeanWithADynamicResultKeepsBounds
func.func @ReduceMeanWithADynamicResultKeepsBounds(%IN: !InBoundedType) -> !OutBoundedType {
    %REDUCE = VPU.ReduceMean(%IN) {axes_value = [1], keep_dims} : !InBoundedType -> !OutBoundedType
    return %REDUCE : !OutBoundedType

    // CHECK:       VPU.ReduceMean
    // CHECK-SAME:  -> tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 48, 400]> : tensor<4xsi64>, order = #NCHW}>
}
