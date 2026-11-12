//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

///--------------------------------------------------------------------------------
/// [Op: VPU.TopK — shape inference]
///--------------------------------------------------------------------------------

// -----

///---
/// Case 1: Static input, k_value attribute.
/// Output is fully static (no bounds).
///---

// CHECK-LABEL: @TopKStaticInputKValue
// CHECK-SAME:  ([[ARG_0:%.+]]: tensor<1x151x513x513xf32>, [[AUX:%.+]]: tensor<1x1x1x2416xui8>)
func.func @TopKStaticInputKValue(%arg0: tensor<1x151x513x513xf32>, %aux: tensor<1x1x1x2416xui8>) -> tensor<1x1x513x513xsi32> {
    %output_values, %target_shape = VPU.TopK(%arg0, %aux) {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<1x151x513x513xf32>, tensor<1x1x1x2416xui8> -> tensor<1x1x513x513xf32>, tensor<1x1x513x513xsi32>
    return %target_shape : tensor<1x1x513x513xsi32>

    // CHECK: [[VALUES:%.+]], [[INDICES:%.+]] = VPU.TopK([[ARG_0]], [[AUX]])
    // CHECK-SAME: k_value = 1 : i64
    // CHECK-SAME: : tensor<1x151x513x513xf32>, tensor<1x1x1x2416xui8> -> tensor<1x1x513x513xf32>, tensor<1x1x513x513xsi32>
    // CHECK: return [[INDICES]] : tensor<1x1x513x513xsi32>
}

// -----

///---
/// Case 2: Static input, constant K operand.
/// Output is fully static (no bounds).
///---

// CHECK-LABEL: @TopKStaticInputConstK
// CHECK-SAME:  ([[ARG_0:%.+]]: tensor<1x151x513x513xf32>, [[AUX:%.+]]: tensor<1x1x1x2416xui8>)
func.func @TopKStaticInputConstK(%arg0: tensor<1x151x513x513xf32>, %aux: tensor<1x1x1x2416xui8>) -> tensor<1x5x513x513xsi32> {
    %cst_k = const.Declare tensor<1xsi32> = dense<5> : tensor<1xsi32>
    %output_values, %target_shape = VPU.TopK(%arg0, %cst_k, %aux) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<1x151x513x513xf32>, tensor<1xsi32>, tensor<1x1x1x2416xui8> -> tensor<1x5x513x513xf32>, tensor<1x5x513x513xsi32>
    return %target_shape : tensor<1x5x513x513xsi32>

    // CHECK-DAG: [[CST_K:%.+]] = const.Declare tensor<1xsi32> = dense<5> : tensor<1xsi32>
    // CHECK: [[VALUES:%.+]], [[INDICES:%.+]] = VPU.TopK([[ARG_0]], [[CST_K]], [[AUX]])
    // CHECK-SAME: -> tensor<1x5x513x513xf32>, tensor<1x5x513x513xsi32>
    // CHECK: return [[INDICES]] : tensor<1x5x513x513xsi32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#BOUNDS = #const.OpaqueI64Elements<[1, 151, 513, 513]> : tensor<4xsi64>

///---
/// Case 3: Static input, non-constant K.
/// The axis dim becomes dynamic, bounded by the input's own size; other dims are unchanged.
///---

// CHECK-LABEL: @TopKStaticInputDynamicK
// CHECK-SAME:  ([[ARG_0:%.+]]: tensor<1x151x513x513xf32>, [[K:%.+]]: tensor<1xsi32>, [[AUX:%.+]]: tensor<1x1x1x2416xui8>)
func.func @TopKStaticInputDynamicK(%arg0: tensor<1x151x513x513xf32>, %k: tensor<1xsi32>, %aux: tensor<1x1x1x2416xui8>) ->
        (tensor<1x?x513x513xf32, {bounds = #BOUNDS, order = #NCHW}>,
         tensor<1x?x513x513xsi32, {bounds = #BOUNDS, order = #NCHW}>) {
    %output_values, %target_shape = VPU.TopK(%arg0, %k, %aux) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<1x151x513x513xf32>, tensor<1xsi32>, tensor<1x1x1x2416xui8>
            -> tensor<1x?x513x513xf32, {bounds = #BOUNDS, order = #NCHW}>,
               tensor<1x?x513x513xsi32, {bounds = #BOUNDS, order = #NCHW}>
    return %output_values, %target_shape
        : tensor<1x?x513x513xf32, {bounds = #BOUNDS, order = #NCHW}>,
          tensor<1x?x513x513xsi32, {bounds = #BOUNDS, order = #NCHW}>

    // CHECK: VPU.TopK([[ARG_0]], [[K]], [[AUX]])
    // CHECK-SAME: -> tensor<1x?x513x513xf32, {bounds = #const.OpaqueI64Elements<[1, 151, 513, 513]> : tensor<4xsi64>, order = #NCHW}>,
    // CHECK-SAME:    tensor<1x?x513x513xsi32, {bounds = #const.OpaqueI64Elements<[1, 151, 513, 513]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#IN_BOUNDS = #const.OpaqueI64Elements<[2, 151, 513, 513]> : tensor<4xsi64>
#OUT_BOUNDS = #const.OpaqueI64Elements<[2, 5, 513, 513]> : tensor<4xsi64>

///---
/// Case 4: Dynamic input (dims 0 and 3 dynamic, axis = 1), k_value attribute.
/// The axis bound is tightened to k_value; the axis dim becomes static, the other bounds are kept.
///---

// CHECK-LABEL: @TopKDynamicInputKValue
// CHECK-SAME:  ([[ARG_0:%.+]]: tensor<?x151x513x?xf32, {{.*}}>, [[AUX:%.+]]: tensor<1x1x1x2416xui8>)
func.func @TopKDynamicInputKValue(
        %arg0: tensor<?x151x513x?xf32, {bounds = #IN_BOUNDS, order = #NCHW}>,
        %aux: tensor<1x1x1x2416xui8>) ->
        (tensor<?x5x513x?xf32, {bounds = #OUT_BOUNDS, order = #NCHW}>,
         tensor<?x5x513x?xsi32, {bounds = #OUT_BOUNDS, order = #NCHW}>) {
    %output_values, %target_shape = VPU.TopK(%arg0, %aux) {axis = 1 : i64, element_type = si32, k_value = 5 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<?x151x513x?xf32, {bounds = #IN_BOUNDS, order = #NCHW}>, tensor<1x1x1x2416xui8>
            -> tensor<?x5x513x?xf32, {bounds = #OUT_BOUNDS, order = #NCHW}>,
               tensor<?x5x513x?xsi32, {bounds = #OUT_BOUNDS, order = #NCHW}>
    return %output_values, %target_shape
        : tensor<?x5x513x?xf32, {bounds = #OUT_BOUNDS, order = #NCHW}>,
          tensor<?x5x513x?xsi32, {bounds = #OUT_BOUNDS, order = #NCHW}>

    // CHECK: VPU.TopK([[ARG_0]], [[AUX]])
    // CHECK-SAME: k_value = 5 : i64
    // CHECK-SAME: -> tensor<?x5x513x?xf32, {bounds = #const.OpaqueI64Elements<[2, 5, 513, 513]> : tensor<4xsi64>, order = #NCHW}>,
    // CHECK-SAME:    tensor<?x5x513x?xsi32, {bounds = #const.OpaqueI64Elements<[2, 5, 513, 513]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#BOUNDS = #const.OpaqueI64Elements<[2, 151, 4, 513]> : tensor<4xsi64>

///---
/// Case 5: Dynamic input (dim 0), non-constant K.
/// The axis dim becomes dynamic with the input's size as the bound; the dynamic dim 0 is kept.
///---

// CHECK-LABEL: @TopKDynamicInputDynamicK
// CHECK-SAME:  ([[ARG_0:%.+]]: tensor<?x151x4x513xf32, {{.*}}>, [[K:%.+]]: tensor<1xsi32>, [[AUX:%.+]]: tensor<1x1x1x2416xui8>)
func.func @TopKDynamicInputDynamicK(
        %arg0: tensor<?x151x4x513xf32, {bounds = #BOUNDS, order = #NCHW}>,
        %k: tensor<1xsi32>,
        %aux: tensor<1x1x1x2416xui8>) ->
        (tensor<?x?x4x513xf32, {bounds = #BOUNDS, order = #NCHW}>,
         tensor<?x?x4x513xsi32, {bounds = #BOUNDS, order = #NCHW}>) {
    %output_values, %target_shape = VPU.TopK(%arg0, %k, %aux) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<?x151x4x513xf32, {bounds = #BOUNDS, order = #NCHW}>, tensor<1xsi32>, tensor<1x1x1x2416xui8>
            -> tensor<?x?x4x513xf32, {bounds = #BOUNDS, order = #NCHW}>,
               tensor<?x?x4x513xsi32, {bounds = #BOUNDS, order = #NCHW}>
    return %output_values, %target_shape
        : tensor<?x?x4x513xf32, {bounds = #BOUNDS, order = #NCHW}>,
          tensor<?x?x4x513xsi32, {bounds = #BOUNDS, order = #NCHW}>

    // CHECK: VPU.TopK([[ARG_0]], [[K]], [[AUX]])
    // CHECK-SAME: -> tensor<?x?x4x513xf32, {bounds = #const.OpaqueI64Elements<[2, 151, 4, 513]> : tensor<4xsi64>, order = #NCHW}>,
    // CHECK-SAME:    tensor<?x?x4x513xsi32, {bounds = #const.OpaqueI64Elements<[2, 151, 4, 513]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#BOUNDS = #const.OpaqueI64Elements<[1, 151, 513, 513]> : tensor<4xsi64>

///---
/// Case 6: Negative axis (-1), non-constant K — the axis is normalized to the last dim.
///---

// CHECK-LABEL: @TopKNegativeAxisDynamicK
// CHECK-SAME:  ([[ARG_0:%.+]]: tensor<1x151x513x513xf32>, [[K:%.+]]: tensor<1xsi32>, [[AUX:%.+]]: tensor<1x1x1x8208xui8>)
func.func @TopKNegativeAxisDynamicK(%arg0: tensor<1x151x513x513xf32>, %k: tensor<1xsi32>, %aux: tensor<1x1x1x8208xui8>) ->
        tensor<1x151x513x?xsi32, {bounds = #BOUNDS, order = #NCHW}> {
    %output_values, %target_shape = VPU.TopK(%arg0, %k, %aux) {axis = -1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<1x151x513x513xf32>, tensor<1xsi32>, tensor<1x1x1x8208xui8>
            -> tensor<1x151x513x?xf32, {bounds = #BOUNDS, order = #NCHW}>,
               tensor<1x151x513x?xsi32, {bounds = #BOUNDS, order = #NCHW}>
    return %target_shape : tensor<1x151x513x?xsi32, {bounds = #BOUNDS, order = #NCHW}>

    // CHECK: VPU.TopK([[ARG_0]], [[K]], [[AUX]])
    // CHECK-SAME: axis = -1 : i64
    // CHECK-SAME: -> tensor<1x151x513x?xf32, {bounds = #const.OpaqueI64Elements<[1, 151, 513, 513]> : tensor<4xsi64>, order = #NCHW}>,
    // CHECK-SAME:    tensor<1x151x513x?xsi32, {bounds = #const.OpaqueI64Elements<[1, 151, 513, 513]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#BOUNDS = #const.OpaqueI64Elements<[1, 64, 32, 32]> : tensor<4xsi64>

///---
/// Case 7: Non-default layout (NHWC), non-constant K.
/// The dims order of the input is propagated to the dynamic output.
///---

// CHECK-LABEL: @TopKNHWCDynamicK
// CHECK-SAME:  ([[ARG_0:%.+]]: tensor<1x64x32x32xf16, {order = #NHWC}>, [[K:%.+]]: tensor<1xsi32>, [[AUX:%.+]]: tensor<1x1x1x1024xui8>)
func.func @TopKNHWCDynamicK(%arg0: tensor<1x64x32x32xf16, {order = #NHWC}>, %k: tensor<1xsi32>, %aux: tensor<1x1x1x1024xui8>) ->
        tensor<1x?x32x32xsi32, {bounds = #BOUNDS, order = #NHWC}> {
    %output_values, %target_shape = VPU.TopK(%arg0, %k, %aux) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_VALUES>}
            : tensor<1x64x32x32xf16, {order = #NHWC}>, tensor<1xsi32>, tensor<1x1x1x1024xui8>
            -> tensor<1x?x32x32xf16, {bounds = #BOUNDS, order = #NHWC}>,
               tensor<1x?x32x32xsi32, {bounds = #BOUNDS, order = #NHWC}>
    return %target_shape : tensor<1x?x32x32xsi32, {bounds = #BOUNDS, order = #NHWC}>

    // CHECK: VPU.TopK([[ARG_0]], [[K]], [[AUX]])
    // CHECK-SAME: -> tensor<1x?x32x32xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 32, 32]> : tensor<4xsi64>, order = #NHWC}>,
    // CHECK-SAME:    tensor<1x?x32x32xsi32, {bounds = #const.OpaqueI64Elements<[1, 64, 32, 32]> : tensor<4xsi64>, order = #NHWC}>
}
