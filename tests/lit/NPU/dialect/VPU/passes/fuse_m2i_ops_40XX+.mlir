//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-m2i-ops %s | FileCheck %s
// REQUIRES: arch-NPU40XX
// CHECK-LABEL: @fuseCscConvertInterpPerm
func.func @fuseCscConvertInterpPerm(%arg0: tensor<1x288x256x1xui8>) -> tensor<1x3x168x224xf16> {
   %0 = VPU.M2I.ColorConvert(%arg0) {inFmt = #IE.color_fmt<NV12>, outFmt = #IE.color_fmt<RGB>} -> tensor<1x192x256x3xui8>
   %1 = VPU.Convert(%0) {dstElemType = f16} : tensor<1x192x256x3xui8> -> tensor<1x192x256x3xf16>
   %2 = VPU.Interpolate(%1) {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [1, 2], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00], sizes_attr = [168, 224]} : tensor<1x192x256x3xf16> -> tensor<1x168x224x3xf16>
   %3 = VPU.MemPermute(%2) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x168x224x3xf16> -> tensor<1x3x168x224xf16>
   return %3 : tensor<1x3x168x224xf16>

   //CHECK: [[VAL0:%.*]] = VPU.M2I.Task(%arg0) {axes = [1, 2], do_csc = true, do_norm = false, inFmt = #VPU.m2i_color_fmt<SP_NV12_8>, outFmt = #VPU.m2i_color_fmt<PL_FP16_RGB>, sizes = [168, 224]} -> tensor<1x3x168x224xf16>
   //CHECK: return [[VAL0]]
}

// -----

// CHECK-LABEL: @fuseCscInterpConvertPerm
func.func @fuseCscInterpConvertPerm(%arg0: tensor<1x288x256x1xui8>) -> tensor<1x3x168x224xf16> {
   %csc = VPU.M2I.ColorConvert(%arg0) {inFmt = #IE.color_fmt<NV12>, outFmt = #IE.color_fmt<RGB>} -> tensor<1x192x256x3xui8>
   %interp = VPU.Interpolate(%csc) {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [1, 2], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00], sizes_attr = [168, 224]} : tensor<1x192x256x3xui8> -> tensor<1x168x224x3xui8>
   %conv = VPU.Convert(%interp) {dstElemType = f16} : tensor<1x168x224x3xui8> -> tensor<1x168x224x3xf16>
   %perm = VPU.MemPermute(%conv) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x168x224x3xf16> -> tensor<1x3x168x224xf16>
   return %perm : tensor<1x3x168x224xf16>

   //CHECK: [[VAL0:%.*]] = VPU.M2I.Task(%arg0) {axes = [1, 2], do_csc = true, do_norm = false, inFmt = #VPU.m2i_color_fmt<SP_NV12_8>, outFmt = #VPU.m2i_color_fmt<PL_FP16_RGB>, sizes = [168, 224]} -> tensor<1x3x168x224xf16>
   //CHECK: return [[VAL0]]
}

// -----

// CHECK-LABEL: @fuseCscResizeConvertPerm
func.func @fuseCscResizeConvertPerm(%arg0: tensor<1x288x256x1xui8>) -> tensor<1x3x168x224xf16> {
   %csc = VPU.M2I.ColorConvert(%arg0) {inFmt = #IE.color_fmt<NV12>, outFmt = #IE.color_fmt<RGB>} -> tensor<1x192x256x3xui8>
   %resize = VPU.M2I.Resize(%csc) {axes = [1, 2], interp = #VPU.m2i_interp<BILINEAR>, sizes = [168, 224]} -> tensor<1x168x224x3xui8>
   %conv = VPU.Convert(%resize) {dstElemType = f16} : tensor<1x168x224x3xui8> -> tensor<1x168x224x3xf16>
   %perm = VPU.MemPermute(%conv) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x168x224x3xf16> -> tensor<1x3x168x224xf16>
   return %perm : tensor<1x3x168x224xf16>

   //CHECK: [[VAL0:%.*]] = VPU.M2I.Task(%arg0) {axes = [1, 2], do_csc = true, do_norm = false, inFmt = #VPU.m2i_color_fmt<SP_NV12_8>, interp = #VPU.m2i_interp<BILINEAR>, outFmt = #VPU.m2i_color_fmt<PL_FP16_RGB>, sizes = [168, 224]} -> tensor<1x3x168x224xf16>
   //CHECK: return [[VAL0]]
}

// -----

// CHECK-LABEL: @fuseCscResizePerm
func.func @fuseCscResizePerm(%arg0: tensor<1x288x256x1xui8>) -> tensor<1x3x168x224xui8> {
  %0 = VPU.M2I.ColorConvert(%arg0) {inFmt = #IE.color_fmt<I420>, outFmt = #IE.color_fmt<BGR>} -> tensor<1x192x256x3xui8>
  %1 = VPU.M2I.Resize(%0) {axes = [1, 2], interp = #VPU.m2i_interp<BILINEAR>, sizes = [168, 224]} -> tensor<1x168x224x3xui8>
  %2 = VPU.MemPermute(%1) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x168x224x3xui8> -> tensor<1x3x168x224xui8>
  return %2 : tensor<1x3x168x224xui8>

  //CHECK: [[VAL0:%.*]] = VPU.M2I.Task(%arg0) {axes = [1, 2], chroma_out_reverse_channels, do_csc = true, do_norm = false, inFmt = #VPU.m2i_color_fmt<PL_YUV420_8>, interp = #VPU.m2i_interp<BILINEAR>, outFmt = #VPU.m2i_color_fmt<PL_RGB24>, sizes = [168, 224]} -> tensor<1x3x168x224xui8>
  //CHECK: return [[VAL0]]
}

// -----

// CHECK-LABEL: @fuseConvertResizePerm
func.func @fuseConvertResizePerm(%arg0: tensor<1x192x256x3xui8>) -> tensor<1x3x168x224xf16> {
  %convert = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x192x256x3xui8> -> tensor<1x192x256x3xf16>
  %resize = VPU.M2I.Resize(%convert) {axes = [1, 2], interp = #VPU.m2i_interp<BILINEAR>, sizes = [168, 224]} -> tensor<1x168x224x3xf16>
  %perm = VPU.MemPermute(%resize) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x168x224x3xf16> -> tensor<1x3x168x224xf16>
  return %perm : tensor<1x3x168x224xf16>

  //CHECK: [[VAL0:%.*]] = VPU.M2I.Task(%arg0) {axes = [1, 2], do_csc = false, do_norm = false, inFmt = #VPU.m2i_color_fmt<IL_RGB888>, interp = #VPU.m2i_interp<BILINEAR>, outFmt = #VPU.m2i_color_fmt<PL_FP16_RGB>, sizes = [168, 224]} -> tensor<1x3x168x224xf16>
  //CHECK: return [[VAL0]]
}

// -----

// CHECK-LABEL: @fuseResizePerm
func.func @fuseResizePerm(%arg0: tensor<1x192x256x3xui8>) -> tensor<1x3x168x224xui8> {
  %resize = VPU.M2I.Resize(%arg0) {axes = [1, 2], interp = #VPU.m2i_interp<BILINEAR>, sizes = [168, 224]} -> tensor<1x168x224x3xui8>
  %perm = VPU.MemPermute(%resize) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x168x224x3xui8> -> tensor<1x3x168x224xui8>
  return %perm : tensor<1x3x168x224xui8>

  //CHECK: [[VAL0:%.*]] = VPU.M2I.Task(%arg0) {axes = [1, 2], do_csc = false, do_norm = false, inFmt = #VPU.m2i_color_fmt<IL_RGB888>, interp = #VPU.m2i_interp<BILINEAR>, outFmt = #VPU.m2i_color_fmt<PL_RGB24>, sizes = [168, 224]} -> tensor<1x3x168x224xui8>
  //CHECK: return [[VAL0]]
}

// -----

// CHECK-LABEL: @fuseCscResize
func.func @fuseCscResize(%arg0: tensor<1x288x256x1xui8>) -> tensor<1x168x224x3xui8> {
  %0 = VPU.M2I.ColorConvert(%arg0) {inFmt = #IE.color_fmt<NV12>, outFmt = #IE.color_fmt<RGB>} -> tensor<1x192x256x3xui8>
  %1 = VPU.M2I.Resize(%0) {axes = [1, 2], interp = #VPU.m2i_interp<NEAREST>, sizes = [168, 224]} -> tensor<1x168x224x3xui8>
  return %1 : tensor<1x168x224x3xui8>

  //CHECK: [[VAL0:%.*]] = VPU.M2I.Task(%arg0) {axes = [1, 2], do_csc = true, do_norm = false, inFmt = #VPU.m2i_color_fmt<SP_NV12_8>, outFmt = #VPU.m2i_color_fmt<IL_RGB888>, sizes = [168, 224]} -> tensor<1x168x224x3xui8>
  //CHECK: return [[VAL0]]
}

// -----

// CHECK-LABEL: @fuseCscPermute
func.func @fuseCscPermute(%arg0: tensor<1x252x224x1xui8>) -> tensor<1x3x168x224xui8> {
  %0 = VPU.M2I.ColorConvert(%arg0) {inFmt = #IE.color_fmt<NV12>, outFmt = #IE.color_fmt<RGB>} -> tensor<1x168x224x3xui8>
  %1 = VPU.MemPermute(%0) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x168x224x3xui8> -> tensor<1x3x168x224xui8>
  return %1 : tensor<1x3x168x224xui8>

  //CHECK: [[VAL0:%.*]] = VPU.M2I.Task(%arg0) {do_csc = true, do_norm = false, inFmt = #VPU.m2i_color_fmt<SP_NV12_8>, outFmt = #VPU.m2i_color_fmt<PL_RGB24>} -> tensor<1x3x168x224xui8>
  //CHECK: return [[VAL0]]
}

// -----

// CHECK-LABEL: @fuseCscConvertPerm
func.func @fuseCscConvertPerm(%arg0: tensor<1x252x224x1xui8>) -> tensor<1x3x168x224xf16> {
  %0 = VPU.M2I.ColorConvert(%arg0) {inFmt = #IE.color_fmt<I420>, outFmt = #IE.color_fmt<RGB>} -> tensor<1x168x224x3xui8>
  %1 = VPU.Convert(%0) {dstElemType = f16} : tensor<1x168x224x3xui8> -> tensor<1x168x224x3xf16>
  %2 = VPU.MemPermute(%1) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x168x224x3xf16> -> tensor<1x3x168x224xf16>
  return %2 : tensor<1x3x168x224xf16>

  //CHECK: [[VAL0:%.*]] = VPU.M2I.Task(%arg0) {do_csc = true, do_norm = false, inFmt = #VPU.m2i_color_fmt<PL_YUV420_8>, outFmt = #VPU.m2i_color_fmt<PL_FP16_RGB>} -> tensor<1x3x168x224xf16>
  //CHECK: return [[VAL0]]
}

// -----

// CHECK-LABEL: @fuseResizeNorm
func.func @fuseResizeNorm(%arg0: tensor<1x3x240x240xf16>) -> tensor<1x3x168x224xf16> {
  %resize = VPU.M2I.Resize(%arg0) {axes = [2, 3], interp = #VPU.m2i_interp<BILINEAR>, sizes = [168, 224]} : tensor<1x3x240x240xf16> -> tensor<1x3x168x224xf16>
  %norm = VPU.M2I.Norm(%resize) {beta_value = [0.000000e+00, 0.4169921875, 1.000000e+00], eps = 1.000000e-03 : f64, gamma_value = [0.000000e+00, 0.4169921875, 1.000000e+00], mean_value = [0.000000e+00, 0.4169921875, 1.000000e+00], variance_value = [7.826089859008789E-5, 1.3154296875, 7.5546875]} : tensor<1x3x168x224xf16> -> tensor<1x3x168x224xf16>
  return %norm : tensor<1x3x168x224xf16>

  //CHECK: [[TASK:%.+]] = VPU.M2I.Task(%arg0) {axes = [2, 3], do_csc = false, do_norm = true, inFmt = #VPU.m2i_color_fmt<PL_FP16_RGB>, interp = #VPU.m2i_interp<BILINEAR>, norm = [0.000000e+00, 0.000000e+00, 0.032836883204562642, 0.000000e+00, 0.4169921875, 0.4169921875, 1.1473576981482279, 0.4169921875, 1.000000e+00, 1.000000e+00, 2.748761084561552, 1.000000e+00], outFmt = #VPU.m2i_color_fmt<PL_FP16_RGB>, sizes = [168, 224]} -> tensor<1x3x168x224xf16>
  //CHECK: return [[TASK]]
}

// -----

// CHECK-LABEL: @fuseNormResize
func.func @fuseNormResize(%arg0: tensor<1x3x240x240xf16>) -> tensor<1x3x168x224xf16> {
  %norm = VPU.M2I.Norm(%arg0) {beta_value = [0.000000e+00, 0.4169921875, 1.000000e+00], eps = 1.000000e-03 : f64, gamma_value = [0.000000e+00, 0.4169921875, 1.000000e+00], mean_value = [0.000000e+00, 0.4169921875, 1.000000e+00], variance_value = [7.826089859008789E-5, 1.3154296875, 7.5546875]} : tensor<1x3x240x240xf16> -> tensor<1x3x240x240xf16>
  %resize = VPU.M2I.Resize(%norm) {axes = [2, 3], interp = #VPU.m2i_interp<BILINEAR>, sizes = [168, 224]} : tensor<1x3x240x240xf16> -> tensor<1x3x168x224xf16>
  return %resize : tensor<1x3x168x224xf16>

  //CHECK: [[TASK:%.+]] = VPU.M2I.Task(%arg0) {axes = [2, 3], do_csc = false, do_norm = true, inFmt = #VPU.m2i_color_fmt<PL_FP16_RGB>, interp = #VPU.m2i_interp<BILINEAR>, norm = [0.000000e+00, 0.000000e+00, 0.032836883204562642, 0.000000e+00, 0.4169921875, 0.4169921875, 1.1473576981482279, 0.4169921875, 1.000000e+00, 1.000000e+00, 2.748761084561552, 1.000000e+00], outFmt = #VPU.m2i_color_fmt<PL_FP16_RGB>, sizes = [168, 224]} -> tensor<1x3x168x224xf16>
  //CHECK: return [[TASK]]
}

// -----

// CHECK-LABEL: @fuseInterpNorm
func.func @fuseInterpNorm(%arg0: tensor<1x3x240x240xf16>) -> tensor<1x3x168x224xf16> {
  %resize = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00], sizes_attr = [168, 224]} : tensor<1x3x240x240xf16> -> tensor<1x3x168x224xf16>
  %norm = VPU.M2I.Norm(%resize) {beta_value = [0.000000e+00, 0.4169921875, 1.000000e+00], eps = 1.000000e-03 : f64, gamma_value = [0.000000e+00, 0.4169921875, 1.000000e+00], mean_value = [0.000000e+00, 0.4169921875, 1.000000e+00], variance_value = [7.826089859008789E-5, 1.3154296875, 7.5546875]} : tensor<1x3x168x224xf16> -> tensor<1x3x168x224xf16>
  return %norm : tensor<1x3x168x224xf16>

  //CHECK: [[TASK:%.+]] = VPU.M2I.Task(%arg0) {axes = [2, 3], do_csc = false, do_norm = true, inFmt = #VPU.m2i_color_fmt<PL_FP16_RGB>, interp = #VPU.m2i_interp<BILINEAR>, norm = [0.000000e+00, 0.000000e+00, 0.032836883204562642, 0.000000e+00, 0.4169921875, 0.4169921875, 1.1473576981482279, 0.4169921875, 1.000000e+00, 1.000000e+00, 2.748761084561552, 1.000000e+00], outFmt = #VPU.m2i_color_fmt<PL_FP16_RGB>, sizes = [168, 224]} -> tensor<1x3x168x224xf16>
  //CHECK: return [[TASK]]
}

// -----

// CHECK-LABEL: @fuseNormInterp
func.func @fuseNormInterp(%arg0: tensor<1x3x240x240xf16>) -> tensor<1x3x168x224xf16> {
  %norm = VPU.M2I.Norm(%arg0) {beta_value = [0.000000e+00, 0.4169921875, 1.000000e+00], eps = 1.000000e-03 : f64, gamma_value = [0.000000e+00, 0.4169921875, 1.000000e+00], mean_value = [0.000000e+00, 0.4169921875, 1.000000e+00], variance_value = [7.826089859008789E-5, 1.3154296875, 7.5546875]} : tensor<1x3x240x240xf16> -> tensor<1x3x240x240xf16>
  %resize = VPU.Interpolate(%norm) {attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00], sizes_attr = [168, 224]} : tensor<1x3x240x240xf16> -> tensor<1x3x168x224xf16>
  return %resize : tensor<1x3x168x224xf16>

  //CHECK: [[TASK:%.+]] = VPU.M2I.Task(%arg0) {axes = [2, 3], do_csc = false, do_norm = true, inFmt = #VPU.m2i_color_fmt<PL_FP16_RGB>, interp = #VPU.m2i_interp<BILINEAR>, norm = [0.000000e+00, 0.000000e+00, 0.032836883204562642, 0.000000e+00, 0.4169921875, 0.4169921875, 1.1473576981482279, 0.4169921875, 1.000000e+00, 1.000000e+00, 2.748761084561552, 1.000000e+00], outFmt = #VPU.m2i_color_fmt<PL_FP16_RGB>, sizes = [168, 224]} -> tensor<1x3x168x224xf16>
  //CHECK: return [[TASK]]
}

// -----

// CHECK-LABEL: @fuseM2ITaskNorm
func.func @fuseM2ITaskNorm(%arg0: tensor<1x288x256x1xui8>) -> tensor<1x3x168x224xf16> {
  %m2itask = VPU.M2I.Task(%arg0) {axes = [2, 3], do_csc = true, do_norm = false, inFmt = #VPU.m2i_color_fmt<PL_YUV420_8>, interp = #VPU.m2i_interp<BILINEAR>, outFmt = #VPU.m2i_color_fmt<PL_FP16_RGB>, sizes = [168, 224]} -> tensor<1x3x168x224xf16>
  %norm = VPU.M2I.Norm(%m2itask) {beta_value = [0.000000e+00, 0.4169921875, 1.000000e+00], eps = 1.000000e-03 : f64, gamma_value = [0.000000e+00, 0.4169921875, 1.000000e+00], mean_value = [0.000000e+00, 0.4169921875, 1.000000e+00], variance_value = [7.826089859008789E-5, 1.3154296875, 7.5546875]} -> tensor<1x3x168x224xf16>
  return %norm : tensor<1x3x168x224xf16>

  //CHECK: [[TASK:%.+]] = VPU.M2I.Task(%arg0) {axes = [2, 3], do_csc = true, do_norm = true, inFmt = #VPU.m2i_color_fmt<PL_YUV420_8>, interp = #VPU.m2i_interp<BILINEAR>, norm = [0.000000e+00, 0.000000e+00, 0.032836883204562642, 0.000000e+00, 0.4169921875, 0.4169921875, 1.1473576981482279, 0.4169921875, 1.000000e+00, 1.000000e+00, 2.748761084561552, 1.000000e+00], outFmt = #VPU.m2i_color_fmt<PL_FP16_RGB>, sizes = [168, 224]} -> tensor<1x3x168x224xf16>
  //CHECK: return [[TASK]]
}

// -----

// CHECK-LABEL: @fuseNormM2ITask
func.func @fuseNormM2ITask(%arg0: tensor<1x3x168x224xf16>) -> tensor<1x288x256x1xui8> {
  %norm = VPU.M2I.Norm(%arg0) {beta_value = [0.000000e+00, 0.4169921875, 1.000000e+00], eps = 1.000000e-03 : f64, gamma_value = [0.000000e+00, 0.4169921875, 1.000000e+00], mean_value = [0.000000e+00, 0.4169921875, 1.000000e+00], variance_value = [7.826089859008789E-5, 1.3154296875, 7.5546875]} -> tensor<1x3x168x224xf16>
  %m2itask = VPU.M2I.Task(%norm) {axes = [1, 2], do_csc = true, do_norm = false, inFmt = #VPU.m2i_color_fmt<PL_FP16_RGB>, interp = #VPU.m2i_interp<BILINEAR>, outFmt = #VPU.m2i_color_fmt<PL_YUV420_8>, sizes = [192, 256]} -> tensor<1x288x256x1xui8>
  return %m2itask : tensor<1x288x256x1xui8>

  //CHECK: [[TASK:%.+]] = VPU.M2I.Task(%arg0) {axes = [1, 2], do_csc = true, do_norm = true, inFmt = #VPU.m2i_color_fmt<PL_FP16_RGB>, interp = #VPU.m2i_interp<BILINEAR>, norm = [0.000000e+00, 0.000000e+00, 0.032836883204562642, 0.000000e+00, 0.4169921875, 0.4169921875, 1.1473576981482279, 0.4169921875, 1.000000e+00, 1.000000e+00, 2.748761084561552, 1.000000e+00], outFmt = #VPU.m2i_color_fmt<PL_YUV420_8>, sizes = [192, 256]} -> tensor<1x288x256x1xui8>
  //CHECK: return [[TASK]]
}

// -----

// CHECK-LABEL: @fuseCscConvertResizePermNorm
func.func @fuseCscConvertResizePermNorm(%arg0: tensor<1x288x256x1xui8>) -> tensor<1x3x168x224xf16> {
   %csc = VPU.M2I.ColorConvert(%arg0) {inFmt = #IE.color_fmt<NV12>, outFmt = #IE.color_fmt<RGB>} -> tensor<1x192x256x3xui8>
   %conv = VPU.Convert(%csc) {dstElemType = f16} : tensor<1x192x256x3xui8> -> tensor<1x192x256x3xf16>
   %interp = VPU.Interpolate(%conv) {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [1, 2], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00], sizes_attr = [168, 224]} : tensor<1x192x256x3xf16> -> tensor<1x168x224x3xf16>
   %perm = VPU.MemPermute(%interp) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x168x224x3xf16> -> tensor<1x3x168x224xf16>
   %norm = VPU.M2I.Norm(%perm) {beta_value = [0.000000e+00, 0.4169921875, 1.000000e+00], eps = 1.000000e-03 : f64, gamma_value = [0.000000e+00, 0.4169921875, 1.000000e+00], mean_value = [0.000000e+00, 0.4169921875, 1.000000e+00], variance_value = [7.826089859008789E-5, 1.3154296875, 7.5546875]} -> tensor<1x3x168x224xf16>
   return %norm : tensor<1x3x168x224xf16>

   //CHECK: [[VAL0:%.*]] = VPU.M2I.Task(%arg0) {axes = [1, 2], do_csc = true, do_norm = true, inFmt = #VPU.m2i_color_fmt<SP_NV12_8>, norm = [0.000000e+00, 0.000000e+00, 0.032836883204562642, 0.000000e+00, 0.4169921875, 0.4169921875, 1.1473576981482279, 0.4169921875, 1.000000e+00, 1.000000e+00, 2.748761084561552, 1.000000e+00], outFmt = #VPU.m2i_color_fmt<PL_FP16_RGB>, sizes = [168, 224]} -> tensor<1x3x168x224xf16>
   //CHECK: return [[VAL0]]
}

// -----

// CHECK-LABEL: @fuseCscResizeConvertPermNorm
func.func @fuseCscResizeConvertPermNorm(%arg0: tensor<1x288x256x1xui8>) -> tensor<1x3x168x224xf16> {
   %csc = VPU.M2I.ColorConvert(%arg0) {inFmt = #IE.color_fmt<NV12>, outFmt = #IE.color_fmt<RGB>} -> tensor<1x192x256x3xui8>
   %resize = VPU.M2I.Resize(%csc) {axes = [1, 2], interp = #VPU.m2i_interp<BILINEAR>, sizes = [168, 224]} -> tensor<1x168x224x3xui8>
   %conv = VPU.Convert(%resize) {dstElemType = f16} : tensor<1x168x224x3xui8> -> tensor<1x168x224x3xf16>
   %perm = VPU.MemPermute(%conv) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x168x224x3xf16> -> tensor<1x3x168x224xf16>
   %norm = VPU.M2I.Norm(%perm) {beta_value = [0.000000e+00, 0.4169921875, 1.000000e+00], eps = 1.000000e-03 : f64, gamma_value = [0.000000e+00, 0.4169921875, 1.000000e+00], mean_value = [0.000000e+00, 0.4169921875, 1.000000e+00], variance_value = [7.826089859008789E-5, 1.3154296875, 7.5546875]} -> tensor<1x3x168x224xf16>
   return %norm : tensor<1x3x168x224xf16>

   //CHECK: [[VAL0:%.*]] = VPU.M2I.Task(%arg0) {axes = [1, 2], do_csc = true, do_norm = true, inFmt = #VPU.m2i_color_fmt<SP_NV12_8>, interp = #VPU.m2i_interp<BILINEAR>, norm = [0.000000e+00, 0.000000e+00, 0.032836883204562642, 0.000000e+00, 0.4169921875, 0.4169921875, 1.1473576981482279, 0.4169921875, 1.000000e+00, 1.000000e+00, 2.748761084561552, 1.000000e+00], outFmt = #VPU.m2i_color_fmt<PL_FP16_RGB>, sizes = [168, 224]} -> tensor<1x3x168x224xf16>
   //CHECK: return [[VAL0]]
}

// -----

// CHECK-LABEL: @fuseTaskConvertPermNorm
func.func @fuseTaskConvertPermNorm(%arg0: tensor<1x768x512x1xui8>) -> tensor<1x3x224x224xf16> {
   %task = VPU.M2I.Task(%arg0) {axes = [1, 2], chroma_out_reverse_channels, do_csc = true, do_norm = false, inFmt = #VPU.m2i_color_fmt<SP_NV12_8>, interp = #VPU.m2i_interp<BILINEAR>, outFmt = #VPU.m2i_color_fmt<IL_RGB888>, sizes = [224, 224]} -> tensor<1x224x224x3xui8>
   %conv = VPU.Convert(%task) {dstElemType = f16} : tensor<1x224x224x3xui8> -> tensor<1x224x224x3xf16>
   %perm = VPU.MemPermute(%conv) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x224x224x3xf16> -> tensor<1x3x224x224xf16>
   %norm = VPU.M2I.Norm(%perm) {beta_value = [-1.8046875, -2.03515625, -2.1171875], eps = 0.000000e+00 : f64, gamma_value = [0.017425537109375, 0.0175018310546875, 0.017120361328125], mean_value = [0.000000e+00, 0.000000e+00, 0.000000e+00], variance_value = [1.000000e+00, 1.000000e+00, 1.000000e+00]} -> tensor<1x3x224x224xf16>
   return %norm : tensor<1x3x224x224xf16>

   //CHECK: [[VAL0:%.*]] = VPU.M2I.Task(%arg0) {axes = [1, 2], chroma_out_reverse_channels, do_csc = true, do_norm = true, inFmt = #VPU.m2i_color_fmt<SP_NV12_8>, interp = #VPU.m2i_interp<BILINEAR>, norm = [0.017425537109375, 0.000000e+00, 1.000000e+00, -1.8046875, 0.0175018310546875, 0.000000e+00, 1.000000e+00, -2.03515625, 0.017120361328125, 0.000000e+00, 1.000000e+00, -2.1171875], outFmt = #VPU.m2i_color_fmt<PL_FP16_RGB>, sizes = [224, 224]} -> tensor<1x3x224x224xf16>
   //CHECK: return [[VAL0]]
}
