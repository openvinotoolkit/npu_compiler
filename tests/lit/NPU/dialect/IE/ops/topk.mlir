//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

///--------------------------------------------------------------------------------
/// [Op: IE.TopK — shape inference]
///--------------------------------------------------------------------------------

// -----

///---
/// Case 1: Const K, static input.
/// ConvertConstToAttr folds the const into a k_value attribute; output stays fully static (no bounds).
///---

// CHECK-LABEL: @ConvertConstToAttrForTopK
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x151x513x513xf32>
func.func @ConvertConstToAttrForTopK(%arg0: tensor<1x151x513x513xf32>) -> tensor<1x1x513x513xsi32> {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<1x151x513x513xf32>, tensor<1xsi32> -> tensor<1x1x513x513xf32>, tensor<1x1x513x513xsi32>

    return %target_shape : tensor<1x1x513x513xsi32>

    // CHECK: [[VAL0:%.+]], [[VAL1:%.+]] = IE.TopK([[ARG_0]])
    // CHECK-SAME{LITERAL}: {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>} :
    // CHECK-SAME{LITERAL}: tensor<1x151x513x513xf32> -> tensor<1x1x513x513xf32>, tensor<1x1x513x513xsi32>
    // CHECK: return [[VAL1]] : tensor<1x1x513x513xsi32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#BOUNDS = #const.OpaqueI64Elements<[1, 151, 513, 513]> : tensor<4xsi64>

///---
/// Case 2: Static input, non-constant K.
/// The axis dim becomes dynamic, bounded by the input's own size; other dims are unchanged.
///---

// CHECK-LABEL: @TopKDynamicK
// CHECK-SAME: ([[ARG_0:%.+]]: tensor<1x151x513x513xf32>, [[K:%.+]]: tensor<1xsi32>)
func.func @TopKDynamicK(%arg0: tensor<1x151x513x513xf32>, %k: tensor<1xsi32>) ->
        (tensor<1x?x513x513xf32, {bounds = #BOUNDS, order = #NCHW}>,
         tensor<1x?x513x513xsi32, {bounds = #BOUNDS, order = #NCHW}>) {
    %output_values, %target_shape = IE.TopK(%arg0, %k) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<1x151x513x513xf32>, tensor<1xsi32>
            -> tensor<1x?x513x513xf32, {bounds = #BOUNDS, order = #NCHW}>,
               tensor<1x?x513x513xsi32, {bounds = #BOUNDS, order = #NCHW}>

    return %output_values, %target_shape
        : tensor<1x?x513x513xf32, {bounds = #BOUNDS, order = #NCHW}>,
          tensor<1x?x513x513xsi32, {bounds = #BOUNDS, order = #NCHW}>

    // CHECK: IE.TopK([[ARG_0]], [[K]])
    // CHECK-SAME: -> tensor<1x?x513x513xf32, {bounds = #const.OpaqueI64Elements<[1, 151, 513, 513]> : tensor<4xsi64>, order = #NCHW}>,
    // CHECK-SAME:    tensor<1x?x513x513xsi32, {bounds = #const.OpaqueI64Elements<[1, 151, 513, 513]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#IN_BOUNDS = #const.OpaqueI64Elements<[2, 151, 513, 513]> : tensor<4xsi64>
#OUT_BOUNDS = #const.OpaqueI64Elements<[2, 151, 5, 513]> : tensor<4xsi64>

///---
/// Case 3: Dynamic input (dims 0 and 2, axis = 2), constant K.
/// The axis bound is tightened to k_value; other bounds (incl. dynamic dim 0) are kept.
///---

// CHECK-LABEL: @TopKDynamicInputStaticK
// CHECK-SAME: ([[ARG_0:%.+]]: tensor<?x151x?x513xf32, {{.*}}>)
func.func @TopKDynamicInputStaticK(
        %arg0: tensor<?x151x?x513xf32, {bounds = #IN_BOUNDS, order = #NCHW}>) ->
        (tensor<?x151x5x513xf32, {bounds = #OUT_BOUNDS, order = #NCHW}>,
         tensor<?x151x5x513xsi32, {bounds = #OUT_BOUNDS, order = #NCHW}>) {
    %cst = const.Declare tensor<1xsi32> = dense<5> : tensor<1xsi32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst) {axis = 2 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<?x151x?x513xf32, {bounds = #IN_BOUNDS, order = #NCHW}>, tensor<1xsi32>
            -> tensor<?x151x5x513xf32, {bounds = #OUT_BOUNDS, order = #NCHW}>,
               tensor<?x151x5x513xsi32, {bounds = #OUT_BOUNDS, order = #NCHW}>

    return %output_values, %target_shape
        : tensor<?x151x5x513xf32, {bounds = #OUT_BOUNDS, order = #NCHW}>,
          tensor<?x151x5x513xsi32, {bounds = #OUT_BOUNDS, order = #NCHW}>

    // K is foldable to a constant, so ConvertConstToAttr drops the %cst operand.
    // CHECK: IE.TopK([[ARG_0]])
    // CHECK-SAME: {axis = 2 : i64, element_type = si32, k_value = 5 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>}
    // CHECK-SAME: -> tensor<?x151x5x513xf32, {bounds = #const.OpaqueI64Elements<[2, 151, 5, 513]> : tensor<4xsi64>, order = #NCHW}>,
    // CHECK-SAME:    tensor<?x151x5x513xsi32, {bounds = #const.OpaqueI64Elements<[2, 151, 5, 513]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#BOUNDS = #const.OpaqueI64Elements<[2, 151, 4, 513]> : tensor<4xsi64>

///---
/// Case 4: Dynamic input (dim 0), static axis (axis = 1), non-constant K.
/// The axis dim becomes dynamic with the input's size as the bound; the dynamic dim 0 is kept.
///---

// CHECK-LABEL: @TopKDynamicInputDynamicK
// CHECK-SAME: ([[ARG_0:%.+]]: tensor<?x151x4x513xf32, {{.*}}>, [[K:%.+]]: tensor<1xsi32>)
func.func @TopKDynamicInputDynamicK(
        %arg0: tensor<?x151x4x513xf32, {bounds = #BOUNDS, order = #NCHW}>,
        %k: tensor<1xsi32>) ->
        (tensor<?x?x4x513xf32, {bounds = #BOUNDS, order = #NCHW}>,
         tensor<?x?x4x513xsi32, {bounds = #BOUNDS, order = #NCHW}>) {
    %output_values, %target_shape = IE.TopK(%arg0, %k) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<?x151x4x513xf32, {bounds = #BOUNDS, order = #NCHW}>, tensor<1xsi32>
            -> tensor<?x?x4x513xf32, {bounds = #BOUNDS, order = #NCHW}>,
               tensor<?x?x4x513xsi32, {bounds = #BOUNDS, order = #NCHW}>

    return %output_values, %target_shape
        : tensor<?x?x4x513xf32, {bounds = #BOUNDS, order = #NCHW}>,
          tensor<?x?x4x513xsi32, {bounds = #BOUNDS, order = #NCHW}>

    // CHECK: IE.TopK([[ARG_0]], [[K]])
    // CHECK-SAME: -> tensor<?x?x4x513xf32, {bounds = #const.OpaqueI64Elements<[2, 151, 4, 513]> : tensor<4xsi64>, order = #NCHW}>,
    // CHECK-SAME:    tensor<?x?x4x513xsi32, {bounds = #const.OpaqueI64Elements<[2, 151, 4, 513]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#BOUNDS = #const.OpaqueI64Elements<[1, 151, 513, 513]> : tensor<4xsi64>

///---
/// Case 5: Negative axis (-1), non-constant K — axis is normalized to the last dim.
///---

// CHECK-LABEL: @TopKNegativeAxisDynamicK
// CHECK-SAME: ([[ARG_0:%.+]]: tensor<1x151x513x513xf32>, [[K:%.+]]: tensor<1xsi32>)
func.func @TopKNegativeAxisDynamicK(%arg0: tensor<1x151x513x513xf32>, %k: tensor<1xsi32>) ->
        tensor<1x151x513x?xsi32, {bounds = #BOUNDS, order = #NCHW}> {
    %output_values, %target_shape = IE.TopK(%arg0, %k) {axis = -1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<1x151x513x513xf32>, tensor<1xsi32>
            -> tensor<1x151x513x?xf32, {bounds = #BOUNDS, order = #NCHW}>,
               tensor<1x151x513x?xsi32, {bounds = #BOUNDS, order = #NCHW}>

    return %target_shape : tensor<1x151x513x?xsi32, {bounds = #BOUNDS, order = #NCHW}>

    // CHECK: IE.TopK([[ARG_0]], [[K]])
    // CHECK-SAME: -> tensor<1x151x513x?xf32, {bounds = #const.OpaqueI64Elements<[1, 151, 513, 513]> : tensor<4xsi64>, order = #NCHW}>,
    // CHECK-SAME:    tensor<1x151x513x?xsi32, {bounds = #const.OpaqueI64Elements<[1, 151, 513, 513]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#BOUNDS = #const.OpaqueI64Elements<[1, 151, 513, 513]> : tensor<4xsi64>

///---
/// Case 6: Min mode with a different sort type does not affect shape inference.
///---

// CHECK-LABEL: @TopKMinModeDynamicK
// CHECK-SAME: ([[ARG_0:%.+]]: tensor<1x151x513x513xf32>, [[K:%.+]]: tensor<1xsi32>)
func.func @TopKMinModeDynamicK(%arg0: tensor<1x151x513x513xf32>, %k: tensor<1xsi32>) ->
        tensor<1x?x513x513xsi32, {bounds = #BOUNDS, order = #NCHW}> {
    %output_values, %target_shape = IE.TopK(%arg0, %k) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MIN>, sort = #IE.topk_sort_type<SORT_VALUES>}
            : tensor<1x151x513x513xf32>, tensor<1xsi32>
            -> tensor<1x?x513x513xf32, {bounds = #BOUNDS, order = #NCHW}>,
               tensor<1x?x513x513xsi32, {bounds = #BOUNDS, order = #NCHW}>

    return %target_shape : tensor<1x?x513x513xsi32, {bounds = #BOUNDS, order = #NCHW}>

    // CHECK: IE.TopK([[ARG_0]], [[K]])
    // CHECK-SAME: mode = #IE.topk_mode<MIN>, sort = #IE.topk_sort_type<SORT_VALUES>
    // CHECK-SAME: -> tensor<1x?x513x513xf32, {bounds = #const.OpaqueI64Elements<[1, 151, 513, 513]> : tensor<4xsi64>, order = #NCHW}>,
    // CHECK-SAME:    tensor<1x?x513x513xsi32, {bounds = #const.OpaqueI64Elements<[1, 151, 513, 513]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

///---
/// Case 7: Unbounded axis dim, constant K.
/// K replaces the only unbounded dim, so the output is static.
///---

// CHECK-LABEL: @TopKUnboundedAxisStaticK
// CHECK-SAME: ([[ARG_0:%.+]]: tensor<1x?x513x513xf32>)
func.func @TopKUnboundedAxisStaticK(%arg0: tensor<1x?x513x513xf32>) -> tensor<1x5x513x513xsi32> {
    %cst = const.Declare tensor<1xsi32> = dense<5> : tensor<1xsi32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<1x?x513x513xf32>, tensor<1xsi32> -> tensor<1x5x513x513xf32>, tensor<1x5x513x513xsi32>

    return %target_shape : tensor<1x5x513x513xsi32>

    // CHECK: IE.TopK([[ARG_0]])
    // CHECK-SAME: k_value = 5 : i64
    // CHECK-SAME: -> tensor<1x5x513x513xf32>, tensor<1x5x513x513xsi32>
}

// -----

///---
/// Case 8: Unbounded axis dim, non-constant K.
/// The output stays unbounded: no bounds can be derived.
///---

// CHECK-LABEL: @TopKUnboundedAxisDynamicK
// CHECK-SAME: ([[ARG_0:%.+]]: tensor<1x?x513x513xf32>, [[K:%.+]]: tensor<1xsi32>)
func.func @TopKUnboundedAxisDynamicK(%arg0: tensor<1x?x513x513xf32>, %k: tensor<1xsi32>) -> tensor<1x?x513x513xsi32> {
    %output_values, %target_shape = IE.TopK(%arg0, %k) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<1x?x513x513xf32>, tensor<1xsi32> -> tensor<1x?x513x513xf32>, tensor<1x?x513x513xsi32>

    return %target_shape : tensor<1x?x513x513xsi32>

    // CHECK: IE.TopK([[ARG_0]], [[K]])
    // CHECK-SAME: -> tensor<1x?x513x513xf32>, tensor<1x?x513x513xsi32>
}

// -----

///---
/// Case 9: Unbounded non-axis dim, constant K.
/// The axis dim becomes K; the other unbounded dim keeps the output unbounded.
///---

// CHECK-LABEL: @TopKUnboundedNonAxisStaticK
// CHECK-SAME: ([[ARG_0:%.+]]: tensor<?x151x513x513xf32>)
func.func @TopKUnboundedNonAxisStaticK(%arg0: tensor<?x151x513x513xf32>) -> tensor<?x5x513x513xsi32> {
    %cst = const.Declare tensor<1xsi32> = dense<5> : tensor<1xsi32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<?x151x513x513xf32>, tensor<1xsi32> -> tensor<?x5x513x513xf32>, tensor<?x5x513x513xsi32>

    return %target_shape : tensor<?x5x513x513xsi32>

    // CHECK: IE.TopK([[ARG_0]])
    // CHECK-SAME: k_value = 5 : i64
    // CHECK-SAME: -> tensor<?x5x513x513xf32>, tensor<?x5x513x513xsi32>
}
