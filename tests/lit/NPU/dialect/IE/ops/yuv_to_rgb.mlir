//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ConvertToMultiInputs
func.func @ConvertToMultiInputs(%arg0: tensor<1x552x432x1xf16>) ->  tensor<1x368x432x3xf16>  {
    %0 = IE.YuvToRgb(%arg0) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 0, 0>, outFmt = #IE.color_fmt<RGB>} : tensor<1x552x432x1xf16> -> tensor<1x368x432x3xf16>
    return %0 : tensor<1x368x432x3xf16>

    //CHECK: [[LUMA:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 368, 432, 1] : tensor<1x552x432x1xf16> to tensor<1x368x432x1xf16>
    //CHECK: [[CHROMA:%.*]] = IE.Slice %arg0 [0, 368, 0, 0] [1, 184, 432, 1] : tensor<1x552x432x1xf16> to tensor<1x184x432x1xf16>
    //CHECK: [[RESHAPE:%.*]] = IE.Reshape([[CHROMA]]) {shape_value = [1, 184, 216, 2]} : tensor<1x184x432x1xf16> -> tensor<1x184x216x2xf16>
    //CHECK: [[YUV_TO_RGB:%.*]] = IE.YuvToRgb([[LUMA]], [[RESHAPE]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>} : tensor<1x368x432x1xf16>, tensor<1x184x216x2xf16> -> tensor<1x368x432x3xf16>
    //CHECK: return [[YUV_TO_RGB]] : tensor<1x368x432x3xf16>
}

