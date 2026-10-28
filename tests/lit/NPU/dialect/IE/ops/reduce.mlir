//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @FoldReduceL1
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceL1(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceL1(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceL1
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @FoldReduceL2
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceL2(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceL2(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceL2
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @FoldReduceLogicalAnd
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceLogicalAnd(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceLogicalAnd(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceLogicalAnd
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @FoldReduceLogicalOr
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceLogicalOr(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceLogicalOr(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceLogicalOr
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @FoldReduceMax
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceMax(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceMax(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceMax
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @FoldReduceMean
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceMean(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceMean(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceMean
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @DoNotFoldPaddedReduceMean
func.func @DoNotFoldPaddedReduceMean(%arg0: tensor<1x16x4x2xf16>) -> tensor<1x16x4x2xf16> {
    %0 = IE.ReduceMean(%arg0) {axes_value = [1], keep_dims, input_padding = [0, 4, 0, 0], output_padding = [0, 15, 0, 0]} : tensor<1x16x4x2xf16> -> tensor<1x16x4x2xf16>
    return %0 : tensor<1x16x4x2xf16>

    // CHECK:   [[REDUCE:%.+]] = IE.ReduceMean
    // CHECK:   return [[REDUCE]]
}

// -----

// CHECK-LABEL: @FoldReduceMin
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceMin(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceMin(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceMin
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @FoldReduceProd
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceProd(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceProd(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceProd
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @FoldReduceSum
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceSum(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceSum
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @DoNotFoldPaddedReduceSum
func.func @DoNotFoldPaddedReduceSum(%arg0: tensor<1x16x4x2xf16>) -> tensor<1x16x4x2xf16> {
    %0 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims, input_padding = [0, 4, 0, 0], output_padding = [0, 15, 0, 0]} : tensor<1x16x4x2xf16> -> tensor<1x16x4x2xf16>
    return %0 : tensor<1x16x4x2xf16>

    // CHECK:   [[REDUCE:%.+]] = IE.ReduceSum
    // CHECK:   return [[REDUCE]]
}

// -----

// The cases below pin the result type of a Reduce whose input has a dimension known only at
// inference time. The rule is in the comment on IE::inferReduceReturnTypeComponents, in
// reduce_infer.cpp.
//
// Each case declares the type the inference has to produce. IE_LayerOp verifies that declared type
// against the inferred one, so a wrong type fails at parse time and no pass has to run.

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InBoundedType = tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 400]> : tensor<4xsi64>, order = #NCHW}>
!OutBoundedType = tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 48, 400]> : tensor<4xsi64>, order = #NCHW}>

// A result that is still dynamic keeps its bounds. All ten IE Reduce operations share the helper, so
// IE.ReduceMean stands for all of them.

// CHECK-LABEL: @ReduceMeanKeepsBounds
func.func @ReduceMeanKeepsBounds(%IN: !InBoundedType) -> !OutBoundedType {
    %REDUCE = IE.ReduceMean(%IN) {axes_value = [1], keep_dims} : !InBoundedType -> !OutBoundedType
    return %REDUCE : !OutBoundedType

    // CHECK:       IE.ReduceMean
    // CHECK-SAME:  -> tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 48, 400]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

!InBoundedType = tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 400]> : tensor<4xsi64>, order = #NCHW}>
!OutBoundedType = tensor<1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 400]> : tensor<3xsi64>, order = #CHW}>

// Without keep_dims the reduced axis leaves the shape, and the bounds have to lose the same entry.

// CHECK-LABEL: @ReduceMeanWithoutKeepDimsKeepsBounds
func.func @ReduceMeanWithoutKeepDimsKeepsBounds(%IN: !InBoundedType) -> !OutBoundedType {
    %REDUCE = IE.ReduceMean(%IN) {axes_value = [1]} : !InBoundedType -> !OutBoundedType
    return %REDUCE : !OutBoundedType

    // CHECK:       IE.ReduceMean
    // CHECK-SAME:  -> tensor<1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 400]> : tensor<3xsi64>, order = #CHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InBoundedType = tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 400]> : tensor<4xsi64>, order = #NCHW}>

// Reducing over every dynamic axis leaves nothing dynamic, so the result must carry no bounds. An
// implementation that gates the bounds on the input alone fails here.

// CHECK-LABEL: @ReduceMeanOverEveryDynamicAxisIsStatic
func.func @ReduceMeanOverEveryDynamicAxisIsStatic(%IN: !InBoundedType) -> tensor<1x16x1x1xf16> {
    %REDUCE = IE.ReduceMean(%IN) {axes_value = [2, 3], keep_dims} : !InBoundedType -> tensor<1x16x1x1xf16>
    return %REDUCE : tensor<1x16x1x1xf16>

    // CHECK:       IE.ReduceMean
    // CHECK-SAME:  -> tensor<1x16x1x1xf16>
}

// -----

// A static Reduce must keep a plain type. tensor<1x1x48x400xf16> and tensor<1x1x48x400xf16, {}> are
// different MLIR types, so writing out an identity order here would move every static Reduce in
// every model.

// CHECK-LABEL: @ReduceMeanStaticInputKeepsAPlainType
func.func @ReduceMeanStaticInputKeepsAPlainType(%IN: tensor<1x16x48x400xf16>) -> tensor<1x1x48x400xf16> {
    %REDUCE = IE.ReduceMean(%IN) {axes_value = [1], keep_dims} : tensor<1x16x48x400xf16> -> tensor<1x1x48x400xf16>
    return %REDUCE : tensor<1x1x48x400xf16>

    // CHECK:       IE.ReduceMean
    // CHECK-SAME:  -> tensor<1x1x48x400xf16>
}
