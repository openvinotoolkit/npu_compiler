//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// What this checks (E#223694): with `--convert-nms-to-static=true`, the frontend runs
// OpenVINO's `ConvertNMS9ToNMSIEInternal` pattern before import (see frontend/IE.cpp).
//
// Note: `boxes`/`scores` in this IR have no dynamic (`-1`) dims, so `--set-upper-bounds` has
// no effect on them - they stay fully static. The only dynamic dim in the original IR is the
// NMS output's first dim (`-1`), and after `IE.NonMaxSuppression` shape inference it collapses
// to the static number of boxes (3), since selected count is bounded by
// min(max_output_boxes_per_class=100, num_boxes=3). With `--convert-nms-to-static=true`,
// `outputsInfo` gets an extra `originalShape` (a bounded type, with the same bound of 3), and
// the indices go through an extra si32->si64->si32 IE.Convert round-trip (NMSIEInternal
// hardcodes si32 indices, so a Convert is needed to restore the original i64 output_type).
//
// TODO (E#223694): this test only covers `--convert-nms-to-static=true`. A follow-up should add
// coverage for `--convert-nms-to-static=false` with an IR whose `boxes`/`scores` actually have
// dynamic (`-1`) dims, so `--set-upper-bounds` takes effect and the NMS output genuinely stays
// dynamic (`?`) instead of collapsing to a static shape as it does here. Not covered yet.
//
// Run it: cd tests/lit/NPU/pipelines && vpux-translate --platform=NPU5010 --import-IE \
//   --convert-nms-to-static=true --set-upper-bounds="1 1 200" IR/nms_convert_to_static_option.xml

// RUN: vpux-translate --platform=%platform% --import-IE --convert-nms-to-static=true --set-upper-bounds="1 1 200" ./IR/nms_convert_to_static_option.xml | FileCheck %s --check-prefixes=CHECK-STATIC
// REQUIRES: platform-NPU5010

// CHECK-STATIC:       module @nms_convert_to_static_option {
// CHECK-STATIC:         net.NetworkInfo entryPoint : @main
// CHECK-STATIC-SAME:      inputsInfo : {
// CHECK-STATIC:           DataInfo "boxes" tensorNames = ["boxes"] : tensor<1x3x4xf32>
// CHECK-STATIC:           DataInfo "scores" tensorNames = ["scores"] : tensor<1x1x3xf32>
// CHECK-STATIC:         } outputsInfo : {
// CHECK-STATIC:           DataInfo "NonMaxSuppression_0.0" friendlyName = "Result" tensorNames = ["NonMaxSuppression_0.0", "selected_indices"]
// CHECK-STATIC-SAME:        : tensor<3x3xsi32>
// CHECK-STATIC-SAME:        originalShape = tensor<?x3xsi32, {bounds = #const.OpaqueI64Elements<[3, 3]> : tensor<2xsi64>, order = #NC}>
// CHECK-STATIC:         }
// CHECK-STATIC:         func.func @main(%arg0: tensor<1x3x4xf32>, %arg1: tensor<1x1x3xf32>)
// CHECK-STATIC-SAME:        -> tensor<3x3xsi32> {
// CHECK-STATIC:           [[SEL_IDX:%.+]], [[SEL_SCORES:%.+]], [[VALID:%.+]] = IE.NonMaxSuppression(%arg0, %arg1, {{.+}})
// CHECK-STATIC-SAME:        {box_encoding = #IE.box_encoding_type<CORNER>, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}
// CHECK-STATIC-SAME:        : tensor<1x3x4xf32>, tensor<1x1x3xf32>, tensor<1xsi64>, tensor<1xf32>, tensor<1xf32>
// CHECK-STATIC-SAME:        -> tensor<3x3xsi32>, tensor<3x3xf32>, tensor<1xsi32>
// CHECK-STATIC:           [[TO_I64:%.+]] = IE.Convert([[SEL_IDX]]) {dstElemType = si64}
// CHECK-STATIC:           [[TO_I32:%.+]] = IE.Convert([[TO_I64]]) {dstElemType = si32}
// CHECK-STATIC:           return [[TO_I32]]
