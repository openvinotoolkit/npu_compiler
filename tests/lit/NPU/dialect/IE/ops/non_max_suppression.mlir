//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

///--------------------------------------------------------------------------------
/// [Op: IE.NonMaxSuppression — shape inference]
///
/// Output shapes follow input staticness:
///   • Static inputs  → static output tensor<B*C*min(N,K) x 3 x ET>
///   • BoundedTensorType inputs → dynamic output with bounds (BoundedTensorType)
///   • valid_outputs is always tensor<1xsi32>
///
/// Plain dynamic inputs without BoundedTensorType are rejected by the op verifier
/// (there is no upper bound from which to derive the output size).
///
/// where B=batches, C=classes, N=num_boxes, K=max_output_boxes_per_class.
///
///   Test cases (each separated by the split delimiter):
///   1. Static inputs, attrs present — baseline, bounds from shape
///   2. Dynamic inputs, attrs present — bounds from input *bounds*, not shape
///   3. max_output_boxes_per_class > num_boxes — min() clamps to num_boxes
///   4. Float32 scores — selected_scores preserves input element type
///   5. Const inputs for optional scalars — ConvertConstToAttr canonicalization
///   6. Large single-class case (matches smoke_custom_NmsLayerTest parameters)
///   7. max_output_boxes_per_class_value = 0 — "no limit", clamp becomes num_boxes
///--------------------------------------------------------------------------------

// -----

///---
/// Case 1: Static inputs.
/// output[0] = batches(3) * classes(5) * min(num_boxes(100), max_output(20)) = 300
/// Static inputs → static output shape.
///---

// CHECK-LABEL: func.func @nms_static_inputs
// CHECK-SAME:  ([[BOX_COORDS:%.+]]: tensor<3x100x4xf16>, [[BOX_SCORES:%.+]]: tensor<3x5x100xf16>)
func.func @nms_static_inputs(%box_coords: tensor<3x100x4xf16>, %box_scores: tensor<3x5x100xf16>) ->
        (tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>) {
    %sel_idx, %sel_scores, %valid = IE.NonMaxSuppression(%box_coords, %box_scores) {
            box_encoding = #IE.box_encoding_type<CENTER>,
            iou_threshold_value = 5.000000e-01 : f64,
            max_output_boxes_per_class_value = 20 : i64,
            operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>,
            score_threshold_value = 3.000000e-01 : f64,
            soft_nms_sigma_value = 0.000000e+00 : f64
        } : tensor<3x100x4xf16>, tensor<3x5x100xf16>
          -> tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>
    return %sel_idx, %sel_scores, %valid : tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>

    // Static inputs produce a static output shape.
    // CHECK: [[SEL_IDX:%.+]], [[SEL_SCORES:%.+]], [[VALID:%.+]] = IE.NonMaxSuppression([[BOX_COORDS]], [[BOX_SCORES]])
    // CHECK-SAME: -> tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>
    // CHECK: return [[SEL_IDX]], [[SEL_SCORES]], [[VALID]] : tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>
}

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

///---
/// Case 2: Dynamic inputs.
/// Bounds are derived from the input *bounds*, not the dynamic shapes.
/// Input scores bounds: [2, 5, 100] → bounds[0] = 2 * 5 * min(100, 20) = 200
///---

// CHECK-LABEL: func.func @nms_dynamic_inputs
// CHECK-SAME:  ([[BOX_COORDS:%.+]]: tensor<?x?x4xf16,{{.*}}>, [[BOX_SCORES:%.+]]: tensor<?x5x?xf16,{{.*}}>)
func.func @nms_dynamic_inputs(
        %box_coords: tensor<?x?x4xf16, {bounds = #const.OpaqueI64Elements<[2, 100, 4]> : tensor<3xsi64>, order = #CHW}>,
        %box_scores: tensor<?x5x?xf16,  {bounds = #const.OpaqueI64Elements<[2, 5, 100]> : tensor<3xsi64>, order = #CHW}>) ->
        (tensor<?x3xsi32, {bounds = #const.OpaqueI64Elements<[200, 3]> : tensor<2xsi64>, order = #NC}>,
         tensor<?x3xf16,  {bounds = #const.OpaqueI64Elements<[200, 3]> : tensor<2xsi64>, order = #NC}>,
         tensor<1xsi32>) {
    %sel_idx, %sel_scores, %valid = IE.NonMaxSuppression(%box_coords, %box_scores) {
            box_encoding = #IE.box_encoding_type<CENTER>,
            iou_threshold_value = 5.000000e-01 : f64,
            max_output_boxes_per_class_value = 20 : i64,
            operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>,
            score_threshold_value = 3.000000e-01 : f64,
            soft_nms_sigma_value = 0.000000e+00 : f64
        } : tensor<?x?x4xf16, {bounds = #const.OpaqueI64Elements<[2, 100, 4]> : tensor<3xsi64>, order = #CHW}>,
            tensor<?x5x?xf16,  {bounds = #const.OpaqueI64Elements<[2, 5, 100]> : tensor<3xsi64>, order = #CHW}>
          -> tensor<?x3xsi32, {bounds = #const.OpaqueI64Elements<[200, 3]> : tensor<2xsi64>, order = #NC}>,
             tensor<?x3xf16,  {bounds = #const.OpaqueI64Elements<[200, 3]> : tensor<2xsi64>, order = #NC}>,
             tensor<1xsi32>
    return %sel_idx, %sel_scores, %valid
        : tensor<?x3xsi32, {bounds = #const.OpaqueI64Elements<[200, 3]> : tensor<2xsi64>, order = #NC}>,
          tensor<?x3xf16,  {bounds = #const.OpaqueI64Elements<[200, 3]> : tensor<2xsi64>, order = #NC}>,
          tensor<1xsi32>

    // CHECK: IE.NonMaxSuppression([[BOX_COORDS]], [[BOX_SCORES]])
    // CHECK-SAME: -> tensor<?x3xsi32, {bounds = #const.OpaqueI64Elements<[200, 3]> : tensor<2xsi64>, order = #NC}>,
    // CHECK-SAME:    tensor<?x3xf16, {bounds = #const.OpaqueI64Elements<[200, 3]> : tensor<2xsi64>, order = #NC}>,
    // CHECK-SAME:    tensor<1xsi32>
}

// -----

///---
/// Case 3: max_output_boxes_per_class > num_boxes — min() clamps to num_boxes.
/// output[0] = batches(1) * classes(2) * min(num_boxes(100), max_output(200)) = 200
/// Static inputs → static output shape.
///---

// CHECK-LABEL: func.func @nms_max_output_clamped_to_num_boxes
func.func @nms_max_output_clamped_to_num_boxes(
        %box_coords: tensor<1x100x4xf16>,
        %box_scores: tensor<1x2x100xf16>) ->
        (tensor<200x3xsi32>, tensor<200x3xf16>, tensor<1xsi32>) {
    %sel_idx, %sel_scores, %valid = IE.NonMaxSuppression(%box_coords, %box_scores) {
            box_encoding = #IE.box_encoding_type<CENTER>,
            iou_threshold_value = 5.000000e-01 : f64,
            max_output_boxes_per_class_value = 200 : i64,
            operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>,
            score_threshold_value = 3.000000e-01 : f64,
            soft_nms_sigma_value = 0.000000e+00 : f64
        } : tensor<1x100x4xf16>, tensor<1x2x100xf16>
          -> tensor<200x3xsi32>, tensor<200x3xf16>, tensor<1xsi32>
    return %sel_idx, %sel_scores, %valid : tensor<200x3xsi32>, tensor<200x3xf16>, tensor<1xsi32>

    // CHECK: IE.NonMaxSuppression
    // CHECK-SAME: -> tensor<200x3xsi32>, tensor<200x3xf16>, tensor<1xsi32>
}

// -----

///---
/// Case 4: Float32 scores — selected_scores preserves input element type (f32).
/// output[0] = batches(1) * classes(1) * min(num_boxes(50), max_output(10)) = 10
/// Static inputs → static output shape.
///---

// CHECK-LABEL: func.func @nms_scores_elem_type_propagated_f32
func.func @nms_scores_elem_type_propagated_f32(
        %box_coords: tensor<1x50x4xf32>,
        %box_scores: tensor<1x1x50xf32>) ->
        (tensor<10x3xsi32>, tensor<10x3xf32>, tensor<1xsi32>) {
    %sel_idx, %sel_scores, %valid = IE.NonMaxSuppression(%box_coords, %box_scores) {
            box_encoding = #IE.box_encoding_type<CORNER>,
            iou_threshold_value = 5.000000e-01 : f64,
            max_output_boxes_per_class_value = 10 : i64,
            operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>,
            score_threshold_value = 3.000000e-01 : f64,
            soft_nms_sigma_value = 0.000000e+00 : f64
        } : tensor<1x50x4xf32>, tensor<1x1x50xf32>
          -> tensor<10x3xsi32>, tensor<10x3xf32>, tensor<1xsi32>
    return %sel_idx, %sel_scores, %valid : tensor<10x3xsi32>, tensor<10x3xf32>, tensor<1xsi32>

    // selected_scores element type matches input scores (f32, not si32)
    // CHECK: IE.NonMaxSuppression
    // CHECK-SAME: -> tensor<10x3xsi32>, tensor<10x3xf32>, tensor<1xsi32>
}

// -----

///---
/// Case 5: Optional scalar inputs provided as const tensors.
/// ConvertConstToAttr canonicalization folds them into attributes.
/// The output shape calculation must use the folded attribute value.
/// output[0] = batches(1) * classes(3) * min(num_boxes(50), max_output(15)) = 45
///---

// CHECK-LABEL: func.func @nms_const_scalars_canonicalized_to_attrs
func.func @nms_const_scalars_canonicalized_to_attrs(
        %box_coords: tensor<1x50x4xf16>,
        %box_scores: tensor<1x3x50xf16>) ->
        (tensor<45x3xsi32>, tensor<45x3xf16>, tensor<1xsi32>) {
    %max_boxes  = const.Declare tensor<1xsi64> = dense<15> : tensor<1xsi64>
    %iou        = const.Declare tensor<1xf32>  = dense<5.000000e-01> : tensor<1xf32>
    %score_thr  = const.Declare tensor<1xf32>  = dense<3.000000e-01> : tensor<1xf32>
    %soft_sigma = const.Declare tensor<1xf32>  = dense<0.000000e+00> : tensor<1xf32>

    %sel_idx, %sel_scores, %valid = IE.NonMaxSuppression(%box_coords, %box_scores,
            %max_boxes, %iou, %score_thr, %soft_sigma) {
            box_encoding = #IE.box_encoding_type<CENTER>,
            operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>
        } : tensor<1x50x4xf16>, tensor<1x3x50xf16>,
            tensor<1xsi64>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>
          -> tensor<45x3xsi32>, tensor<45x3xf16>, tensor<1xsi32>
    return %sel_idx, %sel_scores, %valid : tensor<45x3xsi32>, tensor<45x3xf16>, tensor<1xsi32>

    // After canonicalization, const scalar operands are folded into attributes.
    // Static inputs → static output shape.
    // CHECK-NOT: const.Declare
    // CHECK: IE.NonMaxSuppression
    // CHECK-SAME: max_output_boxes_per_class_value = 15
    // CHECK-SAME: -> tensor<45x3xsi32>, tensor<45x3xf16>, tensor<1xsi32>
}

// -----

///---
/// Case 6: Large single-class detector (matches smoke_custom_NmsLayerTest parameters).
/// output[0] = 1 * 1 * min(128, 100) = 100 → static output shape.
///---

// CHECK-LABEL: func.func @nms_large_single_class
func.func @nms_large_single_class(
        %box_coords: tensor<1x128x4xf32>,
        %box_scores: tensor<1x1x128xf32>) ->
        (tensor<100x3xsi32>, tensor<100x3xf32>, tensor<1xsi32>) {
    %sel_idx, %sel_scores, %valid = IE.NonMaxSuppression(%box_coords, %box_scores) {
            box_encoding = #IE.box_encoding_type<CORNER>,
            iou_threshold_value = 5.000000e-01 : f64,
            max_output_boxes_per_class_value = 100 : i64,
            operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>,
            score_threshold_value = 3.999023437500e-01 : f64,
            soft_nms_sigma_value = 0.000000e+00 : f64
        } : tensor<1x128x4xf32>, tensor<1x1x128xf32>
          -> tensor<100x3xsi32>, tensor<100x3xf32>, tensor<1xsi32>
    return %sel_idx, %sel_scores, %valid : tensor<100x3xsi32>, tensor<100x3xf32>, tensor<1xsi32>

    // Static inputs: 1 * 1 * min(128, 100) = 100.
    // CHECK: IE.NonMaxSuppression
    // CHECK-SAME: -> tensor<100x3xsi32>, tensor<100x3xf32>, tensor<1xsi32>
}

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

///---
/// Case 7: max_output_boxes_per_class_value = 0.
/// Per OV NMS spec, 0 means "no limit" — the clamp becomes num_boxes.
/// Dynamic inputs with valid bounds → BoundedTensorType output.
/// bounds[0] = 1 * 1 * 1917 = 1917.
///---

// CHECK-LABEL: func.func @nms_max_output_zero_means_no_limit
func.func @nms_max_output_zero_means_no_limit(
        %box_coords: tensor<1x?x4xf16, {bounds = #const.OpaqueI64Elements<[1, 1917, 4]> : tensor<3xsi64>, order = #CHW}>,
        %box_scores: tensor<1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1917]> : tensor<3xsi64>, order = #CHW}>) ->
        (tensor<?x3xsi32, {bounds = #const.OpaqueI64Elements<[1917, 3]> : tensor<2xsi64>, order = #NC}>,
         tensor<?x3xf16,  {bounds = #const.OpaqueI64Elements<[1917, 3]> : tensor<2xsi64>, order = #NC}>,
         tensor<1xsi32>) {
    %sel_idx, %sel_scores, %valid = IE.NonMaxSuppression(%box_coords, %box_scores) {
            box_encoding = #IE.box_encoding_type<CORNER>,
            iou_threshold_value = 6.000000e-01 : f64,
            max_output_boxes_per_class_value = 0 : i64,
            operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>,
            score_threshold_value = 0.000000e+00 : f64,
            soft_nms_sigma_value = 0.000000e+00 : f64
        } : tensor<1x?x4xf16, {bounds = #const.OpaqueI64Elements<[1, 1917, 4]> : tensor<3xsi64>, order = #CHW}>,
            tensor<1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1917]> : tensor<3xsi64>, order = #CHW}>
          -> tensor<?x3xsi32, {bounds = #const.OpaqueI64Elements<[1917, 3]> : tensor<2xsi64>, order = #NC}>,
             tensor<?x3xf16,  {bounds = #const.OpaqueI64Elements<[1917, 3]> : tensor<2xsi64>, order = #NC}>,
             tensor<1xsi32>
    return %sel_idx, %sel_scores, %valid
        : tensor<?x3xsi32, {bounds = #const.OpaqueI64Elements<[1917, 3]> : tensor<2xsi64>, order = #NC}>,
          tensor<?x3xf16,  {bounds = #const.OpaqueI64Elements<[1917, 3]> : tensor<2xsi64>, order = #NC}>,
          tensor<1xsi32>

    // max_output_boxes_per_class = 0 → no limit; bounds = [1 * 1 * 1917, 3].
    // CHECK: IE.NonMaxSuppression
    // CHECK-SAME: -> tensor<?x3xsi32, {bounds = #const.OpaqueI64Elements<[1917, 3]> : tensor<2xsi64>, order = #NC}>,
    // CHECK-SAME:    tensor<?x3xf16, {bounds = #const.OpaqueI64Elements<[1917, 3]> : tensor<2xsi64>, order = #NC}>,
    // CHECK-SAME:    tensor<1xsi32>
}
