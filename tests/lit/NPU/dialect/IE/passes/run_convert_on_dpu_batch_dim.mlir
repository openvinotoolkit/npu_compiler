//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --run-f16-to-f32-convert-on-dpu %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// Tests for Roll and Pad batch-dimension guard.
//
// NCEInvariant::isSupported dispatches Roll/Pad to their SEOpInterface model, which does not check
// the N (batch) dimension.  A Roll/Pad with batch != 1 therefore passes the DPU eligibility check
// even though it will not execute on DPU (the SEP/NCE path requires batch == 1).  Fusing the
// FP16->FP32 Convert into such an op would rewrite its result type to FP32 while
// inferReturnTypeComponents derives FP16 from the FP16 data input, causing a type mismatch
// during verification.  The pass must skip fusion when batch != 1.
//
// Note: Roll/Pad with batch == 1 are lowered to the SEP/DPU path by earlier passes and therefore
// do not appear as IE.Roll/IE.Pad ops at the point where this pass runs, so only the batch != 1
// guard is exercised here.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotFoldConvertIntoRollBatchNot1
// CHECK-SAME: ([[INPUT:%.+]]: tensor<2x16x8x8xf16, {order = #NHWC}>)
func.func @NotFoldConvertIntoRollBatchNot1(%arg0: tensor<2x16x8x8xf16, {order = #NHWC}>) -> tensor<2x16x8x8xf32, {order = #NHWC}> {
    %shift = const.Declare tensor<1x1x1x2xsi32> = dense<[[[[0, 3]]]]> : tensor<1x1x1x2xsi32>
    %axes  = const.Declare tensor<1x1x1x2xsi32> = dense<[[[[2, 3]]]]> : tensor<1x1x1x2xsi32>
    %0 = IE.Roll(%arg0, %shift, %axes)
            : tensor<2x16x8x8xf16, {order = #NHWC}>, tensor<1x1x1x2xsi32>, tensor<1x1x1x2xsi32>
           -> tensor<2x16x8x8xf16, {order = #NHWC}>
    %1 = IE.Convert(%0) {dstElemType = f32}
            : tensor<2x16x8x8xf16, {order = #NHWC}> -> tensor<2x16x8x8xf32, {order = #NHWC}>
    return %1 : tensor<2x16x8x8xf32, {order = #NHWC}>

    // Roll passes NCEInvariant::isSupported (batch is not checked there) but batch=2 means it
    // will not run on DPU, so fusing the Convert would cause a type mismatch. Fusion is skipped.
    // CHECK:       [[ROLL:%.+]] = IE.Roll([[INPUT]], {{%.+}}, {{%.+}})
    // CHECK-SAME:    -> tensor<2x16x8x8xf16, {order = #NHWC}>
    // CHECK-NEXT:  [[CONVERT:%.+]] = IE.Convert([[ROLL]]) {dstElemType = f32}
    // CHECK-SAME:    -> tensor<2x16x8x8xf32, {order = #NHWC}>
    // CHECK:       return [[CONVERT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotFoldConvertIntoPadBatchNot1
// CHECK-SAME: ([[INPUT:%.+]]: tensor<2x64x6x6xf16, {order = #NHWC}>)
func.func @NotFoldConvertIntoPadBatchNot1(%arg0: tensor<2x64x6x6xf16, {order = #NHWC}>) -> tensor<2x64x8x8xf32, {order = #NHWC}> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64,
                         pads_begin_attr = [0, 0, 1, 1], pads_end_attr = [0, 0, 1, 1]}
            : tensor<2x64x6x6xf16, {order = #NHWC}> -> tensor<2x64x8x8xf16, {order = #NHWC}>
    %1 = IE.Convert(%0) {dstElemType = f32}
            : tensor<2x64x8x8xf16, {order = #NHWC}> -> tensor<2x64x8x8xf32, {order = #NHWC}>
    return %1 : tensor<2x64x8x8xf32, {order = #NHWC}>

    // Pad passes NCEInvariant::isSupported (batch is not checked there) but batch=2 means it
    // will not run on DPU, so fusing the Convert would cause a type mismatch. Fusion is skipped.
    // CHECK:       [[PAD:%.+]] = IE.Pad([[INPUT]])
    // CHECK-SAME:    -> tensor<2x64x8x8xf16, {order = #NHWC}>
    // CHECK-NEXT:  [[CONVERT:%.+]] = IE.Convert([[PAD]]) {dstElemType = f32}
    // CHECK-SAME:    -> tensor<2x64x8x8xf32, {order = #NHWC}>
    // CHECK:       return [[CONVERT]]
}
