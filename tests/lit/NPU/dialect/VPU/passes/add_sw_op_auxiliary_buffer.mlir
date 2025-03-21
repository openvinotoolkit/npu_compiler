//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --add-sw-op-auxiliary-buffer %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: func.func @Proposal
// CHECK-SAME:        [[ARG0:%arg[0-9]]]: tensor<1x2x4x4xf16>
// CHECK-SAME:        [[ARG1:%arg[0-9]]]: tensor<1x4x4x4xf16>
func.func @Proposal(%arg0: tensor<1x2x4x4xf16>, %arg1: tensor<1x4x4x4xf16>) -> (tensor<300x5xf16>, tensor<300xf16>) {
    %cst = const.Declare tensor<3xf16> = dense<[2.250000e+02, 2.250000e+02, 1.000000e+00]> : tensor<3xf16>
    %output, %probs = VPU.Proposal(%arg0, %arg1, %cst) {proposal_attrs = #IE.Proposal<baseSize = 4 : i64, preNmsTopN = 6000 : i64, postNmsTopN = 300 : i64, nmsThresh = 0.69999998807907104 : f64, featStride = 1 : i64, minSize = 4 : i64, ratio = [5.000000e-01], scale = [1.2000000476837158], clipBeforeNms = true, clipAfterNms = false, normalize = true, boxSizeScale = 2.000000e+00 : f64, boxCoordinateScale = 2.000000e+00 : f64, framework = "", inferProbs = true>} : tensor<1x2x4x4xf16>, tensor<1x4x4x4xf16>, tensor<3xf16> -> tensor<300x5xf16>, tensor<300xf16>
    return %output, %probs : tensor<300x5xf16>, tensor<300xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<3xf16> = dense<[2.250000e+02, 2.250000e+02, 1.000000e+00]> : tensor<3xf16>
    // CHECK:       [[CST_0:%.+]] = const.Declare tensor<182xui8> = dense<0> : tensor<182xui8>
    // CHECK:       [[OUTPUT:%.+]], [[PROBS:%.+]] = VPU.Proposal([[ARG0]], [[ARG1]], [[CST]], [[CST_0]]) {proposal_attrs = #IE.Proposal<baseSize = 4 : i64, preNmsTopN = 6000 : i64, postNmsTopN = 300 : i64, nmsThresh = 0.69999998807907104 : f64, featStride = 1 : i64, minSize = 4 : i64, ratio = [5.000000e-01], scale = [1.2000000476837158], clipBeforeNms = true, clipAfterNms = false, normalize = true, boxSizeScale = 2.000000e+00 : f64, boxCoordinateScale = 2.000000e+00 : f64, framework = "", inferProbs = true>} : tensor<1x2x4x4xf16>, tensor<1x4x4x4xf16>, tensor<3xf16>, tensor<182xui8> -> tensor<300x5xf16>, tensor<300xf16>
    // CHECK:       return [[OUTPUT]], [[PROBS]] : tensor<300x5xf16>, tensor<300xf16>
}

// -----

// CHECK-LABEL: @AddTopKBuffer
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x200x32000xf16>
func.func @AddTopKBuffer(%arg0: tensor<1x1x200x32000xf16>) -> (tensor<1x1x200x1xf16>, tensor<1x1x200x1xsi32>) {
    %output_values, %target_shape = VPU.TopK(%arg0)
        {axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, operandSegmentSizes = array<i32: 1, 0, 0>, sort = #IE.topk_sort_type<NONE>}
        : tensor<1x1x200x32000xf16> -> tensor<1x1x200x1xf16>, tensor<1x1x200x1xsi32>
    return %output_values, %target_shape : tensor<1x1x200x1xf16>, tensor<1x1x200x1xsi32>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x1x1x512000xui8> = dense<0> : tensor<1x1x1x512000xui8>
    // CHECK:       [[OUTPUT:%.+]], [[TARGET_SHAPE:%.+]] = VPU.TopK([[ARG0]], [[CST]]) {axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, operandSegmentSizes = array<i32: 1, 0, 1>, sort = #IE.topk_sort_type<NONE>} : tensor<1x1x200x32000xf16>, tensor<1x1x1x512000xui8> -> tensor<1x1x200x1xf16>, tensor<1x1x200x1xsi32>
    // CHECK:       return [[OUTPUT]], [[TARGET_SHAPE]] : tensor<1x1x200x1xf16>, tensor<1x1x200x1xsi32>
}

// -----

// CHECK-LABEL: @AddTopKBufferSmallerReproducer
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x1x1001xf16>
func.func @AddTopKBufferSmallerReproducer(%arg0: tensor<1x1x1x1001xf16>) -> (tensor<1x1x1x1xf16>, tensor<1x1x1x1xsi32>) {
    %output_values, %target_shape = VPU.TopK(%arg0)
        {axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, operandSegmentSizes = array<i32: 1, 0, 0>, sort = #IE.topk_sort_type<SORT_VALUES>}
        : tensor<1x1x1x1001xf16> -> tensor<1x1x1x1xf16>, tensor<1x1x1x1xsi32>
    return %output_values, %target_shape : tensor<1x1x1x1xf16>, tensor<1x1x1x1xsi32>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x1x1x16016xui8> = dense<0> : tensor<1x1x1x16016xui8>
    // CHECK:       [[OUTPUT:%.+]], [[TARGET_SHAPE:%.+]] = VPU.TopK([[ARG0]], [[CST]]) {axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, operandSegmentSizes = array<i32: 1, 0, 1>, sort = #IE.topk_sort_type<SORT_VALUES>} : tensor<1x1x1x1001xf16>, tensor<1x1x1x16016xui8> -> tensor<1x1x1x1xf16>, tensor<1x1x1x1xsi32>
    // CHECK:       return [[OUTPUT]], [[TARGET_SHAPE]] : tensor<1x1x1x1xf16>, tensor<1x1x1x1xsi32>
}

// -----

// CHECK-LABEL: @ExperimentalDetectronROIFeatureExtractor
// CHECK-SAME:  [[INPUT_ROIS:%.+]]: tensor<100x4xf32>
// CHECK-SAME:  [[INPUT_FEATURE0:%.+]]: tensor<1x64x192x320xf32>
// CHECK-SAME:  [[INPUT_FEATURE1:%.+]]: tensor<1x64x96x160xf32>
// CHECK-SAME:  [[INPUT_FEATURE2:%.+]]: tensor<1x64x48x80xf32>
func.func @ExperimentalDetectronROIFeatureExtractor(%arg0: tensor<100x4xf32>, %arg1: tensor<1x64x192x320xf32>, %arg2: tensor<1x64x96x160xf32>, %arg3: tensor<1x64x48x80xf32>) -> (tensor<100x64x14x14xf32>, tensor<100x4xf32>) {
    %output, %outputROIs = VPU.ExperimentalDetectronROIFeatureExtractor(%arg0, %arg1, %arg2, %arg3) {attr = #IE.ExperimentalDetectronROIFeatureExtractor<output_size = 14 : i64, sampling_ratio = 2 : i64, aligned = false, pyramid_scales = [4, 8, 16]>, operandSegmentSizes = array<i32: 4, 0, 0, 0, 0>} : tensor<100x4xf32>, tensor<1x64x192x320xf32>, tensor<1x64x96x160xf32>, tensor<1x64x48x80xf32> -> tensor<100x64x14x14xf32>, tensor<100x4xf32>
    return %output, %outputROIs : tensor<100x64x14x14xf32>, tensor<100x4xf32>

    // CHECK:     [[AUX_REORD:%.+]] = const.Declare tensor<400xf32> = dense<0.000000e+00> : tensor<400xf32>
    // CHECK:     [[AUX_ORIG_MAP:%.+]] = const.Declare tensor<100xui32> = dense<0> : tensor<100xui32>
    // CHECK:     [[AUX_OUTPUT_TEMP:%.+]] = const.Declare tensor<1254400xf32> = dense<0.000000e+00> : tensor<1254400xf32>
    // CHECK:     [[AUX_LEVELS:%.+]] = const.Declare tensor<100xui32> = dense<0> : tensor<100xui32>
    // CHECK:     [[OUTPUT_FEATURES:%.+]], [[OUTPUT_ROIS:%.+]] = VPU.ExperimentalDetectronROIFeatureExtractor([[INPUT_ROIS:%.+]], [[INPUT_FEATURE0:%.+]], [[INPUT_FEATURE1:%.+]], [[INPUT_FEATURE2:%.+]], [[AUX_REORD:%.+]], [[AUX_ORIG_MAP:%.+]], [[AUX_OUTPUT_TEMP:%.+]], [[AUX_LEVELS:%.+]]) {attr = #IE.ExperimentalDetectronROIFeatureExtractor<output_size = 14 : i64, sampling_ratio = 2 : i64, aligned = false, pyramid_scales = [4, 8, 16]>, operandSegmentSizes = array<i32: 4, 1, 1, 1, 1>} : tensor<100x4xf32>, tensor<1x64x192x320xf32>, tensor<1x64x96x160xf32>, tensor<1x64x48x80xf32>, tensor<400xf32>, tensor<100xui32>, tensor<1254400xf32>, tensor<100xui32> -> tensor<100x64x14x14xf32>, tensor<100x4xf32>
    // CHECK:     return [[OUTPUT_FEATURES:%.+]], [[OUTPUT_ROIS:%.+]] : tensor<100x64x14x14xf32>, tensor<100x4xf32>
}
