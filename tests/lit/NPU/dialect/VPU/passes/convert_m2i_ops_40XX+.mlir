//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-m2i-ops --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @convertM2iColorConvert
func.func @convertM2iColorConvert(%arg0: tensor<1x360x320x1xui8>) -> tensor<1x240x320x3xui8> {
   %0 = VPU.M2I.ColorConvert(%arg0) {inFmt = #IE.color_fmt<I420>, outFmt = #IE.color_fmt<RGB>} -> tensor<1x240x320x3xui8>
   return %0 : tensor<1x240x320x3xui8>

   //CHECK: [[VAL0:%.*]] = VPU.M2I.Task(%arg0) {do_csc = true, do_norm = false, inFmt = #VPU.m2i_color_fmt<PL_YUV420_8>, outFmt = #VPU.m2i_color_fmt<IL_RGB888>} -> tensor<1x240x320x3xui8>
   //CHECK: return [[VAL0]]
}

// -----

// CHECK-LABEL: @convertM2iResizePL
func.func @convertM2iResizePL(%arg0: tensor<1x3x256x256xf16>) -> tensor<1x3x224x224xf16> {
   %0 = VPU.M2I.Resize(%arg0) {axes = [2, 3], interp = #VPU.m2i_interp<BILINEAR>, sizes = [224, 224]} -> tensor<1x3x224x224xf16>
   return %0 : tensor<1x3x224x224xf16>

   //CHECK: [[VAL0:%.*]] = VPU.M2I.Task(%arg0) {axes = [2, 3], do_csc = false, do_norm = false, inFmt = #VPU.m2i_color_fmt<PL_FP16_RGB>, interp = #VPU.m2i_interp<BILINEAR>, outFmt = #VPU.m2i_color_fmt<PL_FP16_RGB>, sizes = [224, 224]} -> tensor<1x3x224x224xf16>
   //CHECK: return [[VAL0]]
}

// -----

// CHECK-LABEL: @convertM2iResizeIL
func.func @convertM2iResizeIL(%arg0: tensor<1x256x256x3xui8>) -> tensor<1x224x224x3xui8> {
   %0 = VPU.M2I.Resize(%arg0) {axes = [1, 2], interp = #VPU.m2i_interp<BILINEAR>, sizes = [224, 224]} -> tensor<1x224x224x3xui8>
   return %0 : tensor<1x224x224x3xui8>

   //CHECK: [[VAL0:%.*]] = VPU.M2I.Task(%arg0) {axes = [1, 2], do_csc = false, do_norm = false, inFmt = #VPU.m2i_color_fmt<IL_RGB888>, interp = #VPU.m2i_interp<BILINEAR>, outFmt = #VPU.m2i_color_fmt<IL_RGB888>, sizes = [224, 224]} -> tensor<1x224x224x3xui8>
   //CHECK: return [[VAL0]]
}

// -----

// CHECK-LABEL: @convertM2iNorm
func.func @convertM2iNorm(%arg0: tensor<1x3x256x256xf16>) -> tensor<1x3x256x256xf16> {
   %0 = VPU.M2I.Norm(%arg0) {beta_value = [0.000000e+00, 0.4169921875, 1.000000e+00], eps = 1.000000e-03 : f64, gamma_value = [0.000000e+00, 0.4169921875, 1.000000e+00], mean_value = [0.000000e+00, 0.4169921875, 1.000000e+00], variance_value = [7.826089859008789E-5, 1.3154296875, 7.5546875]} -> tensor<1x3x256x256xf16>
   return %0 : tensor<1x3x256x256xf16>

   //CHECK: [[VAL0:%.*]] = VPU.M2I.Task(%arg0) {do_csc = false, do_norm = true, inFmt = #VPU.m2i_color_fmt<PL_FP16_RGB>, norm = [0.000000e+00, 0.000000e+00, 0.032836883204562642, 0.000000e+00, 0.4169921875, 0.4169921875, 1.1473576981482279, 0.4169921875, 1.000000e+00, 1.000000e+00, 2.748761084561552, 1.000000e+00], outFmt = #VPU.m2i_color_fmt<PL_FP16_RGB>} -> tensor<1x3x256x256xf16>
   //CHECK: return [[VAL0]]
}
