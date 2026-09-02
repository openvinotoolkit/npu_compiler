//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --mvn-fusion --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @FuseMVNInsideSqrt
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x1500x512xf32>) -> tensor<1500x512xf32>
func.func @FuseMVNInsideSqrt(%arg0: tensor<1x1500x512xf32>) -> tensor<1500x512xf32> {
    %1 = IE.Reshape(%arg0) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>

    %mean1 = IE.ReduceMean(%1) {axes_value = [1], keep_dims} : tensor<1500x512xf32> -> tensor<1500x1xf32>

    %sub1 = IE.Subtract(%1, %mean1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1500x512xf32>, tensor<1500x1xf32> -> tensor<1500x512xf32>
    %mul1 = IE.Multiply(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1500x512xf32>, tensor<1500x512xf32> -> tensor<1500x512xf32>

    %mean2 = IE.ReduceMean(%mul1) {axes_value = [1], keep_dims} : tensor<1500x512xf32> -> tensor<1500x1xf32>

    %mul2 = IE.Multiply(%mean1, %mean1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1500x1xf32>, tensor<1500x1xf32> -> tensor<1500x1xf32>
    %sub2 = IE.Subtract(%mean2, %mul2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1500x1xf32>, tensor<1500x1xf32> -> tensor<1500x1xf32>

    %eps = const.Declare tensor<1xf32> = dense<0.000001> : tensor<1xf32>
    %insideAdd = IE.Add(%sub2, %eps) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1500x1xf32>, tensor<1xf32> -> tensor<1500x1xf32>

    %sqrt = IE.Sqrt(%insideAdd) : tensor<1500x1xf32> -> tensor<1500x1xf32>
    %div = IE.Divide(%sub1, %sqrt) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1500x512xf32>, tensor<1500x1xf32> -> tensor<1500x512xf32>
    return %div : tensor<1500x512xf32>

    // CHECK-NOT: IE.Multiply
    // CHECK-NOT: IE.Add
    // CHECK-NOT: IE.Subtract
    // CHECK-NOT: IE.ReduceMean
    // CHECK-NOT: IE.Divide
    // CHECK-NOT: IE.Sqrt
    // CHECK:  [[PRE_RESHAPE:%.+]] = IE.AffineReshape([[ARG_0]])
    // CHECK:        tensor<1x1500x512xf32> -> tensor<1x1500x512x1xf32>
    // CHECK:  [[MVN:%.+]] = IE.MVN([[PRE_RESHAPE]])
    // CHECK-SAME:    across_channels = false,
    // CHECK:         eps
    // CHECK-SAME:    normalize_variance = true} : tensor<1x1500x512x1xf32> -> tensor<1x1500x512x1xf32>
    // CHECK:  [[POST_RESHAPE:%.+]] = IE.AffineReshape([[MVN]])
    // CHECK:       tensor<1x1500x512x1xf32> -> tensor<1500x512xf32>
    // CHECK:  return [[POST_RESHAPE]] : tensor<1500x512xf32>
}

// -----

// CHECK-LABEL: @FuseMVNOutsideSqrt
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1500x512xf32>) -> tensor<1500x512xf32>
func.func @FuseMVNOutsideSqrt(%arg0: tensor<1500x512xf32>) -> tensor<1500x512xf32> {
    %mean1 = IE.ReduceMean(%arg0) {axes_value = [1], keep_dims} : tensor<1500x512xf32> -> tensor<1500x1xf32>

    %sub1 = IE.Subtract(%arg0, %mean1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1500x512xf32>, tensor<1500x1xf32> -> tensor<1500x512xf32>
    %mul1 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1500x512xf32>, tensor<1500x512xf32> -> tensor<1500x512xf32>

    %mean2 = IE.ReduceMean(%mul1) {axes_value = [1], keep_dims} : tensor<1500x512xf32> -> tensor<1500x1xf32>

    %mul2 = IE.Multiply(%mean1, %mean1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1500x1xf32>, tensor<1500x1xf32> -> tensor<1500x1xf32>
    %sub2 = IE.Subtract(%mean2, %mul2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1500x1xf32>, tensor<1500x1xf32> -> tensor<1500x1xf32>

    %sqrt = IE.Sqrt(%sub2) : tensor<1500x1xf32> -> tensor<1500x1xf32>
    %eps = const.Declare tensor<1xf32> = dense<0.000001> : tensor<1xf32>
    %outsideAdd = IE.Add(%sqrt, %eps) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1500x1xf32>, tensor<1xf32> -> tensor<1500x1xf32>

    %div = IE.Divide(%sub1, %outsideAdd) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1500x512xf32>, tensor<1500x1xf32> -> tensor<1500x512xf32>
    return %div : tensor<1500x512xf32>

    // CHECK-NOT: IE.Multiply
    // CHECK-NOT: IE.Add
    // CHECK-NOT: IE.Subtract
    // CHECK-NOT: IE.ReduceMean
    // CHECK-NOT: IE.Divide
    // CHECK-NOT: IE.Sqrt
    // CHECK:  [[PRE_RESHAPE:%.+]] = IE.AffineReshape([[ARG_0]])
    // CHECK:        tensor<1500x512xf32> -> tensor<1x1500x512x1xf32>
    // CHECK:  [[MVN:%.+]] = IE.MVN([[PRE_RESHAPE]])
    // CHECK-SAME:    across_channels = false,
    // CHECK:         eps
    // CHECK-SAME:    normalize_variance = true} : tensor<1x1500x512x1xf32> -> tensor<1x1500x512x1xf32>
    // CHECK:  [[POST_RESHAPE:%.+]] = IE.AffineReshape([[MVN]])
    // CHECK:       tensor<1x1500x512x1xf32> -> tensor<1500x512xf32>
    // CHECK:  return [[POST_RESHAPE]] : tensor<1500x512xf32>
}

// -----

// CHECK-LABEL: @FuseMVNAxes2D
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<16x1500x512xf32>) -> tensor<16x1500x512xf32>
func.func @FuseMVNAxes2D(%arg0: tensor<16x1500x512xf32>) -> tensor<16x1500x512xf32> {
    %mean1 = IE.ReduceMean(%arg0) {axes_value = [1,2], keep_dims} : tensor<16x1500x512xf32> -> tensor<16x1x1xf32>

    %sub1 = IE.Subtract(%arg0, %mean1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1500x512xf32>, tensor<16x1x1xf32> -> tensor<16x1500x512xf32>
    %mul1 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1500x512xf32>, tensor<16x1500x512xf32> -> tensor<16x1500x512xf32>

    %mean2 = IE.ReduceMean(%mul1) {axes_value = [1,2], keep_dims} : tensor<16x1500x512xf32> -> tensor<16x1x1xf32>

    %mul2 = IE.Multiply(%mean1, %mean1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1x1xf32>, tensor<16x1x1xf32> -> tensor<16x1x1xf32>
    %sub2 = IE.Subtract(%mean2, %mul2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1x1xf32>, tensor<16x1x1xf32> -> tensor<16x1x1xf32>

    %sqrt = IE.Sqrt(%sub2) : tensor<16x1x1xf32> -> tensor<16x1x1xf32>
    %eps = const.Declare tensor<1xf32> = dense<0.000001> : tensor<1xf32>
    %outsideAdd = IE.Add(%sqrt, %eps) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1x1xf32>, tensor<1xf32> -> tensor<16x1x1xf32>

    %div = IE.Divide(%sub1, %outsideAdd) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1500x512xf32>, tensor<16x1x1xf32> -> tensor<16x1500x512xf32>
    return %div : tensor<16x1500x512xf32>

    // CHECK-NOT: IE.Multiply
    // CHECK-NOT: IE.Add
    // CHECK-NOT: IE.Subtract
    // CHECK-NOT: IE.ReduceMean
    // CHECK-NOT: IE.Divide
    // CHECK-NOT: IE.Sqrt
    // CHECK:  [[PRE_RESHAPE:%.+]] = IE.AffineReshape([[ARG_0]])
    // CHECK:        tensor<16x1500x512xf32> -> tensor<1x16x1500x512xf32>
    // CHECK:  [[MVN:%.+]] = IE.MVN([[PRE_RESHAPE]])
    // CHECK-SAME:    across_channels = false,
    // CHECK:         eps
    // CHECK-SAME:    normalize_variance = true} : tensor<1x16x1500x512xf32> -> tensor<1x16x1500x512xf32>
    // CHECK:  [[POST_RESHAPE:%.+]] = IE.AffineReshape([[MVN]])
    // CHECK:       tensor<1x16x1500x512xf32> -> tensor<16x1500x512xf32>
    // CHECK:  return [[POST_RESHAPE]] : tensor<16x1500x512xf32>
}

// -----

// CHECK-LABEL: @FuseMVNAcrossChannel
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x1500x512xf32>) -> tensor<1x16x1500x512xf32>
func.func @FuseMVNAcrossChannel(%arg0: tensor<1x16x1500x512xf32>) -> tensor<1x16x1500x512xf32> {
    %mean1 = IE.ReduceMean(%arg0) {axes_value = [1,2,3], keep_dims} : tensor<1x16x1500x512xf32> -> tensor<1x1x1x1xf32>

    %sub1 = IE.Subtract(%arg0, %mean1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1500x512xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x1500x512xf32>
    %mul1 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1500x512xf32>, tensor<1x16x1500x512xf32> -> tensor<1x16x1500x512xf32>

    %mean2 = IE.ReduceMean(%mul1) {axes_value = [1,2,3], keep_dims} : tensor<1x16x1500x512xf32> -> tensor<1x1x1x1xf32>

    %mul2 = IE.Multiply(%mean1, %mean1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32>
    %sub2 = IE.Subtract(%mean2, %mul2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32>

    %sqrt = IE.Sqrt(%sub2) : tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32>
    %eps = const.Declare tensor<1xf32> = dense<0.000001> : tensor<1xf32>
    %outsideAdd = IE.Add(%sqrt, %eps) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1xf32> -> tensor<1x1x1x1xf32>

    %div = IE.Divide(%sub1, %outsideAdd) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1500x512xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x1500x512xf32>
    return %div : tensor<1x16x1500x512xf32>

    // CHECK-NOT: IE.Multiply
    // CHECK-NOT: IE.Add
    // CHECK-NOT: IE.Subtract
    // CHECK-NOT: IE.ReduceMean
    // CHECK-NOT: IE.Divide
    // CHECK-NOT: IE.Sqrt
    // CHECK:  [[MVN:%.+]] = IE.MVN([[ARG_0]])
    // CHECK-SAME:    across_channels = true,
    // CHECK:         eps
    // CHECK-SAME:    normalize_variance = true} : tensor<1x16x1500x512xf32> -> tensor<1x16x1500x512xf32>
    // CHECK:  return [[MVN]] : tensor<1x16x1500x512xf32>
}

// -----

// CHECK-LABEL: @FuseOvRefMvnPow
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x32x15x64xf16>
func.func @FuseOvRefMvnPow(%arg0: tensor<1x32x15x64xf16>) -> tensor<1x32x15x64xf16> {
    %cst_two = const.Declare tensor<1xf16> = dense<2.0>  : tensor<1xf16>
    %cst_eps = const.Declare tensor<1xf16> = dense<0.00002>  : tensor<1xf16>

    %0 = IE.ReduceMean(%arg0) {axes_value = [2, 3], keep_dims} : tensor<1x32x15x64xf16> -> tensor<1x32x1x1xf16>
    %1 = IE.Subtract(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x15x64xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x15x64xf16>
    %2 = IE.Power(%1, %cst_two) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x15x64xf16>, tensor<1xf16> -> tensor<1x32x15x64xf16>
    %3 = IE.ReduceMean(%2) {axes_value = [2, 3], keep_dims} : tensor<1x32x15x64xf16> -> tensor<1x32x1x1xf16>
    %4 = IE.Add(%3, %cst_eps) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1x1xf16>, tensor<1xf16> -> tensor<1x32x1x1xf16>
    %5 = IE.Sqrt(%4) : tensor<1x32x1x1xf16> -> tensor<1x32x1x1xf16>
    %6 = IE.Divide(%1, %5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x15x64xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x15x64xf16>

    return %6 : tensor<1x32x15x64xf16>

    // CHECK:      [[MVN:%.+]] = IE.MVN([[INPUT]]) {across_channels = false, eps = 2.002716064453125E-5 : f64, normalize_variance = true}
    // CHECK-SAME:                : tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16>
    // CHECK:      return [[MVN]] : tensor<1x32x15x64xf16>
}

// -----

// CHECK-LABEL: @FuseOvRefMvnMul
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x32x15x64xf16>
func.func @FuseOvRefMvnMul(%arg0: tensor<1x32x15x64xf16>) -> tensor<1x32x15x64xf16> {
    %cst_eps = const.Declare tensor<1xf16> = dense<0.00002>  : tensor<1xf16>

    %0 = IE.ReduceMean(%arg0) {axes_value = [1,2,3], keep_dims} : tensor<1x32x15x64xf16> -> tensor<1x1x1x1xf16>
    %1 = IE.Subtract(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x15x64xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x15x64xf16>
    %2 = IE.Multiply(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x15x64xf16>, tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16>
    %3 = IE.ReduceMean(%2) {axes_value = [1,2,3], keep_dims} : tensor<1x32x15x64xf16> -> tensor<1x1x1x1xf16>
    %4 = IE.Add(%3, %cst_eps) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf16>, tensor<1xf16> -> tensor<1x1x1x1xf16>
    %5 = IE.Sqrt(%4) : tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
    %6 = IE.Divide(%1, %5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x15x64xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x15x64xf16>

    return %6 : tensor<1x32x15x64xf16>

    // CHECK:      [[MVN:%.+]] = IE.MVN([[INPUT]]) {across_channels = true, eps = 2.002716064453125E-5 : f64, normalize_variance = true}
    // CHECK-SAME:                : tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16>
    // CHECK:      return [[MVN]] : tensor<1x32x15x64xf16>
}

// -----

// CHECK-LABEL: @FuseOvRefMvn3D
// CHECK-SAME:  [[INPUT:%.+]]: tensor<151x1x768xf32>
func.func @FuseOvRefMvn3D(%arg0: tensor<151x1x768xf32>) -> tensor<151x1x768xf32> {
    %cst_two = const.Declare tensor<1x1x1xf32> = dense<2.0>  : tensor<1x1x1xf32>
    %cst_eps = const.Declare tensor<1xf32> = dense<0.00002>  : tensor<1xf32>

    %0 = IE.ReduceMean(%arg0) {axes_value = [2], keep_dims} : tensor<151x1x768xf32> -> tensor<151x1x1xf32>
    %1 = IE.Subtract(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<151x1x768xf32>, tensor<151x1x1xf32> -> tensor<151x1x768xf32>
    %2 = IE.Power(%1, %cst_two) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<151x1x768xf32>, tensor<1x1x1xf32> -> tensor<151x1x768xf32>
    %3 = IE.ReduceMean(%2) {axes_value = [2], keep_dims} : tensor<151x1x768xf32> -> tensor<151x1x1xf32>
    %4 = IE.Add(%3, %cst_eps) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<151x1x1xf32>, tensor<1xf32> -> tensor<151x1x1xf32>
    %5 = IE.Sqrt(%4) : tensor<151x1x1xf32> -> tensor<151x1x1xf32>
    %6 = IE.Divide(%1, %5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<151x1x768xf32>, tensor<151x1x1xf32> -> tensor<151x1x768xf32>

    return %6 : tensor<151x1x768xf32>

    // CHECK:      [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME:                      tensor<151x1x768xf32> -> tensor<151x1x768x1xf32>
    // CHECK:      [[MVN:%.+]] = IE.MVN([[RESHAPE_IN]]) {across_channels = false, eps = 1.9999999494757503E-5 : f64, normalize_variance = true}
    // CHECK-SAME:                    : tensor<151x1x768x1xf32> -> tensor<151x1x768x1xf32>
    // CHECK:      [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[MVN]])
    // CHECK-SAME:                      tensor<151x1x768x1xf32> -> tensor<151x1x768xf32>
    // CHECK:        return [[RESHAPE_OUT]] : tensor<151x1x768xf32>
}

// -----

// CHECK-LABEL: @FuseMVNWithSquaredDiff
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x8x64xf32>
func.func @FuseMVNWithSquaredDiff(%arg0: tensor<1x8x64xf32>) -> tensor<1x8x64xf32> {
    // TFLite-style LayerNorm: variance computed via SquaredDifference(x, mean).
    // Pattern: out = x*rsqrt + Reshape(0 - mean*rsqrt) = (x - mean) / sqrt(var + eps).
    %cst_neg_half = const.Declare tensor<f32>   = dense<-5.000000e-01>  : tensor<f32>
    %cst_eps     = const.Declare tensor<f32>    = dense<1.000000e-06>   : tensor<f32>
    %cst_zero    = const.Declare tensor<f32>    = dense<0.000000e+00>   : tensor<f32>

    %mean    = IE.ReduceMean(%arg0) {axes_value = [2]}
                   : tensor<1x8x64xf32> -> tensor<1x8xf32>
    %mean_r  = IE.Reshape(%mean) {shape_value = [1, 8, 1]}
                   : tensor<1x8xf32> -> tensor<1x8x1xf32>
    %sq_diff = IE.SquaredDiff(%arg0, %mean_r) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                   : tensor<1x8x64xf32>, tensor<1x8x1xf32> -> tensor<1x8x64xf32>
    %var     = IE.ReduceMean(%sq_diff) {axes_value = [2]}
                   : tensor<1x8x64xf32> -> tensor<1x8xf32>
    %var_eps = IE.Add(%var, %cst_eps) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                   : tensor<1x8xf32>, tensor<f32> -> tensor<1x8xf32>
    %rsqrt   = IE.Power(%var_eps, %cst_neg_half) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                   : tensor<1x8xf32>, tensor<f32> -> tensor<1x8xf32>
    %rsqrt_r = IE.Reshape(%rsqrt) {shape_value = [1, 8, 1]}
                   : tensor<1x8xf32> -> tensor<1x8x1xf32>
    %x_mul   = IE.Multiply(%arg0, %rsqrt_r) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                   : tensor<1x8x64xf32>, tensor<1x8x1xf32> -> tensor<1x8x64xf32>
    %neg_mul = IE.Multiply(%mean, %rsqrt) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                   : tensor<1x8xf32>, tensor<1x8xf32> -> tensor<1x8xf32>
    %neg     = IE.Subtract(%cst_zero, %neg_mul) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                   : tensor<f32>, tensor<1x8xf32> -> tensor<1x8xf32>
    %neg_r   = IE.Reshape(%neg) {shape_value = [1, 8, 1]}
                   : tensor<1x8xf32> -> tensor<1x8x1xf32>
    %result  = IE.Add(%x_mul, %neg_r) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                   : tensor<1x8x64xf32>, tensor<1x8x1xf32> -> tensor<1x8x64xf32>
    return %result : tensor<1x8x64xf32>

    // CHECK-NOT: IE.SquaredDiff
    // CHECK-NOT: IE.Power
    // CHECK-NOT: IE.ReduceMean
    // CHECK:     [[PRE_RESHAPE:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME:    tensor<1x8x64xf32> -> tensor<1x8x64x1xf32>
    // CHECK:     [[MVN:%.+]] = IE.MVN([[PRE_RESHAPE]])
    // CHECK-SAME:    across_channels = false
    // CHECK-SAME:    eps = 9.9999999747524271E-7 : f64
    // CHECK-SAME:    normalize_variance = true
    // CHECK-SAME:    tensor<1x8x64x1xf32> -> tensor<1x8x64x1xf32>
    // CHECK:     [[POST_RESHAPE:%.+]] = IE.AffineReshape([[MVN]])
    // CHECK-SAME:    tensor<1x8x64x1xf32> -> tensor<1x8x64xf32>
    // CHECK:     return [[POST_RESHAPE]] : tensor<1x8x64xf32>
}

// -----

// CHECK-LABEL: @FuseMVNWithSquaredDiffTranspose
// CHECK-SAME:  [[INPUT:%.+]]: tensor<8x64x49xf32>
func.func @FuseMVNWithSquaredDiffTranspose(%arg0: tensor<8x64x49xf32>) -> tensor<8x64x49xf32> {
    // TFLite-style LayerNorm with axes [0, 1] on a 8x64x49 tensor.
    // canConvertToMVN1 cannot handle this case with a pure reshape because the
    // values to be normalized (8*64=512 per W-position) are not contiguous in
    // the original layout. getMVN1Mapping applies a Transpose [2,0,1] to
    // rearrange 8x64x49 -> 49x8x64, then a Reshape to 1x49x8x64, applies MVN,
    // and inverts via Reshape + Transpose [1,2,0].
    %cst_neg_half = const.Declare tensor<f32>    = dense<-5.000000e-01>   : tensor<f32>
    %cst_eps      = const.Declare tensor<f32>    = dense<1.000000e-06>    : tensor<f32>
    %cst_zero     = const.Declare tensor<f32>    = dense<0.000000e+00>    : tensor<f32>

    // mean: reduce axes [0,1] from 8x64x49 -> 49 (no keep_dims)
    %mean    = IE.ReduceMean(%arg0) {axes_value = [0, 1]}
                   : tensor<8x64x49xf32> -> tensor<49xf32>
    %mean_r  = IE.Reshape(%mean) {shape_value = [1, 1, 49]}
                   : tensor<49xf32> -> tensor<1x1x49xf32>
    %sq_diff = IE.SquaredDiff(%arg0, %mean_r) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                   : tensor<8x64x49xf32>, tensor<1x1x49xf32> -> tensor<8x64x49xf32>
    %var     = IE.ReduceMean(%sq_diff) {axes_value = [0, 1]}
                   : tensor<8x64x49xf32> -> tensor<49xf32>
    %var_eps = IE.Add(%var, %cst_eps) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                   : tensor<49xf32>, tensor<f32> -> tensor<49xf32>
    %rsqrt   = IE.Power(%var_eps, %cst_neg_half) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                   : tensor<49xf32>, tensor<f32> -> tensor<49xf32>
    %rsqrt_r = IE.Reshape(%rsqrt) {shape_value = [1, 1, 49]}
                   : tensor<49xf32> -> tensor<1x1x49xf32>
    %x_mul   = IE.Multiply(%arg0, %rsqrt_r) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                   : tensor<8x64x49xf32>, tensor<1x1x49xf32> -> tensor<8x64x49xf32>
    %neg_mul = IE.Multiply(%mean, %rsqrt) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                   : tensor<49xf32>, tensor<49xf32> -> tensor<49xf32>
    %neg     = IE.Subtract(%cst_zero, %neg_mul) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                   : tensor<f32>, tensor<49xf32> -> tensor<49xf32>
    %neg_r   = IE.Reshape(%neg) {shape_value = [1, 1, 49]}
                   : tensor<49xf32> -> tensor<1x1x49xf32>
    %result  = IE.Add(%x_mul, %neg_r) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                   : tensor<8x64x49xf32>, tensor<1x1x49xf32> -> tensor<8x64x49xf32>
    return %result : tensor<8x64x49xf32>

    // CHECK-NOT: IE.SquaredDiff
    // CHECK-NOT: IE.Power
    // CHECK-NOT: IE.ReduceMean
    // Pre-Transpose [2,0,1]: 8x64x49 -> 49x8x64
    // CHECK:     [[PRE_TRANSPOSE:%.+]] = IE.Transpose([[INPUT]])
    // CHECK-SAME:    tensor<8x64x49xf32> -> tensor<49x8x64xf32>
    // Reshape: 49x8x64 -> 1x49x8x64
    // CHECK:     [[PRE_RESHAPE:%.+]] = IE.AffineReshape([[PRE_TRANSPOSE]])
    // CHECK-SAME:    tensor<49x8x64xf32> -> tensor<1x49x8x64xf32>
    // MVN normalizes over H=8,W=64 for each of the 49 channels (= axes [0,1])
    // CHECK:     [[MVN:%.+]] = IE.MVN([[PRE_RESHAPE]])
    // CHECK-SAME:    across_channels = false
    // CHECK-SAME:    eps = 9.9999999747524271E-7 : f64
    // CHECK-SAME:    normalize_variance = true
    // CHECK-SAME:    tensor<1x49x8x64xf32> -> tensor<1x49x8x64xf32>
    // Reshape: 1x49x8x64 -> 49x8x64
    // CHECK:     [[POST_RESHAPE:%.+]] = IE.AffineReshape([[MVN]])
    // CHECK-SAME:    tensor<1x49x8x64xf32> -> tensor<49x8x64xf32>
    // Post-Transpose [1,2,0]: 49x8x64 -> 8x64x49
    // CHECK:     [[POST_TRANSPOSE:%.+]] = IE.Transpose([[POST_RESHAPE]])
    // CHECK-SAME:    tensor<49x8x64xf32> -> tensor<8x64x49xf32>
    // CHECK:     return [[POST_TRANSPOSE]] : tensor<8x64x49xf32>
}

// -----

// CHECK-LABEL: @FuseMVNWithSquaredDiffAxes02
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x8x64xf32>
func.func @FuseMVNWithSquaredDiffAxes02(%arg0: tensor<1x8x64xf32>) -> tensor<1x8x64xf32> {
    // TFLite-style LayerNorm with axes [0, 2] on a batch-1 tensor.
    // Reducing over dim-0 (batch=1) is trivial; canConvertToMVN1 treats [0,2] as [2].
    %cst_neg_half = const.Declare tensor<f32>   = dense<-5.000000e-01>  : tensor<f32>
    %cst_eps     = const.Declare tensor<f32>    = dense<1.000000e-06>   : tensor<f32>
    %cst_zero    = const.Declare tensor<f32>    = dense<0.000000e+00>   : tensor<f32>

    %mean    = IE.ReduceMean(%arg0) {axes_value = [0, 2]}
                   : tensor<1x8x64xf32> -> tensor<8xf32>
    %mean_r  = IE.Reshape(%mean) {shape_value = [1, 8, 1]}
                   : tensor<8xf32> -> tensor<1x8x1xf32>
    %sq_diff = IE.SquaredDiff(%arg0, %mean_r) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                   : tensor<1x8x64xf32>, tensor<1x8x1xf32> -> tensor<1x8x64xf32>
    %var     = IE.ReduceMean(%sq_diff) {axes_value = [0, 2]}
                   : tensor<1x8x64xf32> -> tensor<8xf32>
    %var_eps = IE.Add(%var, %cst_eps) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                   : tensor<8xf32>, tensor<f32> -> tensor<8xf32>
    %rsqrt   = IE.Power(%var_eps, %cst_neg_half) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                   : tensor<8xf32>, tensor<f32> -> tensor<8xf32>
    %rsqrt_r = IE.Reshape(%rsqrt) {shape_value = [1, 8, 1]}
                   : tensor<8xf32> -> tensor<1x8x1xf32>
    %x_mul   = IE.Multiply(%arg0, %rsqrt_r) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                   : tensor<1x8x64xf32>, tensor<1x8x1xf32> -> tensor<1x8x64xf32>
    %neg_mul = IE.Multiply(%mean, %rsqrt) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                   : tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
    %neg     = IE.Subtract(%cst_zero, %neg_mul) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                   : tensor<f32>, tensor<8xf32> -> tensor<8xf32>
    %neg_r   = IE.Reshape(%neg) {shape_value = [1, 8, 1]}
                   : tensor<8xf32> -> tensor<1x8x1xf32>
    %result  = IE.Add(%x_mul, %neg_r) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                   : tensor<1x8x64xf32>, tensor<1x8x1xf32> -> tensor<1x8x64xf32>
    return %result : tensor<1x8x64xf32>

    // CHECK-NOT: IE.SquaredDiff
    // CHECK-NOT: IE.Power
    // CHECK-NOT: IE.ReduceMean
    // CHECK:     [[PRE_RESHAPE:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME:    tensor<1x8x64xf32> -> tensor<1x8x1x64xf32>
    // CHECK:     [[MVN:%.+]] = IE.MVN([[PRE_RESHAPE]])
    // CHECK-SAME:    across_channels = false
    // CHECK-SAME:    eps = 9.9999999747524271E-7 : f64
    // CHECK-SAME:    normalize_variance = true
    // CHECK-SAME:    tensor<1x8x1x64xf32> -> tensor<1x8x1x64xf32>
    // CHECK:     [[POST_RESHAPE:%.+]] = IE.AffineReshape([[MVN]])
    // CHECK-SAME:    tensor<1x8x1x64xf32> -> tensor<1x8x64xf32>
    // CHECK:     return [[POST_RESHAPE]] : tensor<1x8x64xf32>
}

// -----

// CHECK-LABEL: @FuseOvRefMvnIndependentReduceMean
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x32x15x64xf16>
func.func @FuseOvRefMvnIndependentReduceMean(%arg0: tensor<1x32x15x64xf16>) -> tensor<1x32x15x64xf16> {
    // Two independent ReduceMean ops on the same input with the same axes,
    // producing two independent SubtractOps (Sub1 for numerator, Sub2 for variance).
    %cst_eps = const.Declare tensor<1xf16> = dense<0.00002> : tensor<1xf16>

    // mean1 -> Sub1 (numerator path)
    %mean1 = IE.ReduceMean(%arg0) {axes_value = [2, 3], keep_dims} : tensor<1x32x15x64xf16> -> tensor<1x32x1x1xf16>
    %sub1 = IE.Subtract(%arg0, %mean1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x15x64xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x15x64xf16>

    // mean2 -> Sub2 (variance path, independent but semantically equivalent)
    %mean2 = IE.ReduceMean(%arg0) {axes_value = [2, 3], keep_dims} : tensor<1x32x15x64xf16> -> tensor<1x32x1x1xf16>
    %sub2 = IE.Subtract(%arg0, %mean2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x15x64xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x15x64xf16>

    // variance = ReduceMean(Sub2 * Sub2)
    %sq = IE.Multiply(%sub2, %sub2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x15x64xf16>, tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16>
    %var = IE.ReduceMean(%sq) {axes_value = [2, 3], keep_dims} : tensor<1x32x15x64xf16> -> tensor<1x32x1x1xf16>
    %add_eps = IE.Add(%var, %cst_eps) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1x1xf16>, tensor<1xf16> -> tensor<1x32x1x1xf16>
    %sqrt = IE.Sqrt(%add_eps) : tensor<1x32x1x1xf16> -> tensor<1x32x1x1xf16>

    // out = Sub1 / sqrt(var + eps)
    %result = IE.Divide(%sub1, %sqrt) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x15x64xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x15x64xf16>

    return %result : tensor<1x32x15x64xf16>

    // CHECK-NOT: IE.Subtract
    // CHECK-NOT: IE.Multiply
    // CHECK-NOT: IE.ReduceMean
    // CHECK:     [[MVN:%.+]] = IE.MVN([[INPUT]])
    // CHECK-SAME:    across_channels = false
    // CHECK-SAME:    normalize_variance = true
    // CHECK-SAME:    tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16>
    // CHECK:     return [[MVN]] : tensor<1x32x15x64xf16>
}

// -----

// CHECK-LABEL: @FuseMVNWithAffineNHWCLastAxis
// CHECK-SAME: [[INPUT:%.+]]: tensor<2x2x2x4xf32>
func.func @FuseMVNWithAffineNHWCLastAxis(%arg0: tensor<2x2x2x4xf32>) -> tensor<2x2x2x4xf32> {
    %one = const.Declare tensor<f32> = dense<1.0> : tensor<f32>
    %eps = const.Declare tensor<f32> = dense<1.0e-6> : tensor<f32>
    %gamma = const.Declare tensor<1x1x1x4xf32> = dense<[[[[1.0, 1.25, 0.75, 2.0]]]]> : tensor<1x1x1x4xf32>
    %beta = const.Declare tensor<1x1x1x4xf32> = dense<[[[[0.5, -0.25, 0.0, 1.0]]]]> : tensor<1x1x1x4xf32>

    %mean = IE.ReduceMean(%arg0) {axes_value = [3], keep_dims} : tensor<2x2x2x4xf32> -> tensor<2x2x2x1xf32>
    %centered = IE.Subtract(%arg0, %mean) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<2x2x2x4xf32>, tensor<2x2x2x1xf32> -> tensor<2x2x2x4xf32>
    %square = IE.Multiply(%centered, %centered) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<2x2x2x4xf32>, tensor<2x2x2x4xf32> -> tensor<2x2x2x4xf32>
    %variance = IE.ReduceMean(%square) {axes_value = [3], keep_dims} :
        tensor<2x2x2x4xf32> -> tensor<2x2x2x1xf32>
    %variance_eps = IE.Add(%variance, %eps) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<2x2x2x1xf32>, tensor<f32> -> tensor<2x2x2x1xf32>
    %sqrt = IE.Sqrt(%variance_eps) : tensor<2x2x2x1xf32> -> tensor<2x2x2x1xf32>
    %rsqrt = IE.Divide(%one, %sqrt) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<f32>, tensor<2x2x2x1xf32> -> tensor<2x2x2x1xf32>
    %scale = IE.Multiply(%rsqrt, %gamma) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<2x2x2x1xf32>, tensor<1x1x1x4xf32> -> tensor<2x2x2x4xf32>
    %x_scale = IE.Multiply(%arg0, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<2x2x2x4xf32>, tensor<2x2x2x4xf32> -> tensor<2x2x2x4xf32>
    %mean_scale = IE.Multiply(%mean, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<2x2x2x1xf32>, tensor<2x2x2x4xf32> -> tensor<2x2x2x4xf32>
    %beta_minus_mean_scale = IE.Subtract(%beta, %mean_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x1x1x4xf32>, tensor<2x2x2x4xf32> -> tensor<2x2x2x4xf32>
    %result = IE.Add(%x_scale, %beta_minus_mean_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<2x2x2x4xf32>, tensor<2x2x2x4xf32> -> tensor<2x2x2x4xf32>
    return %result : tensor<2x2x2x4xf32>

    // CHECK-NOT: IE.ReduceMean
    // CHECK-NOT: IE.Sqrt
    // CHECK-NOT: IE.Divide
    // CHECK-NOT: IE.Subtract
    // CHECK: [[MVN:%.+]] = IE.MVN
    // CHECK: [[POST:%.+]] = IE.AffineReshape([[MVN]])
    // CHECK: IE.Multiply([[POST]]
    // CHECK: IE.Add
    // CHECK: return
}

// -----

// CHECK-LABEL: @FuseMVNWithAffineExpandedLayerNorm
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x8x64xf32>
func.func @FuseMVNWithAffineExpandedLayerNorm(%arg0: tensor<1x8x64xf32>) -> tensor<1x8x64xf32> {
    %one = const.Declare tensor<f32> = dense<1.0> : tensor<f32>
    %eps = const.Declare tensor<f32> = dense<1.0e-6> : tensor<f32>
    %gamma = const.Declare tensor<1x1x64xf32> = dense<1.25> : tensor<1x1x64xf32>
    %beta = const.Declare tensor<1x1x64xf32> = dense<0.5> : tensor<1x1x64xf32>

    %mean = IE.ReduceMean(%arg0) {axes_value = [2], keep_dims} : tensor<1x8x64xf32> -> tensor<1x8x1xf32>
    %centered = IE.Subtract(%arg0, %mean) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x8x64xf32>, tensor<1x8x1xf32> -> tensor<1x8x64xf32>
    %square = IE.Multiply(%centered, %centered) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x8x64xf32>, tensor<1x8x64xf32> -> tensor<1x8x64xf32>
    %variance = IE.ReduceMean(%square) {axes_value = [2], keep_dims} :
        tensor<1x8x64xf32> -> tensor<1x8x1xf32>
    %variance_eps = IE.Add(%variance, %eps) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x8x1xf32>, tensor<f32> -> tensor<1x8x1xf32>
    %sqrt = IE.Sqrt(%variance_eps) : tensor<1x8x1xf32> -> tensor<1x8x1xf32>
    %rsqrt = IE.Divide(%one, %sqrt) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<f32>, tensor<1x8x1xf32> -> tensor<1x8x1xf32>
    %scale = IE.Multiply(%rsqrt, %gamma) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x8x1xf32>, tensor<1x1x64xf32> -> tensor<1x8x64xf32>
    %x_scale = IE.Multiply(%arg0, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x8x64xf32>, tensor<1x8x64xf32> -> tensor<1x8x64xf32>
    %mean_scale = IE.Multiply(%mean, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x8x1xf32>, tensor<1x8x64xf32> -> tensor<1x8x64xf32>
    %beta_minus_mean_scale = IE.Subtract(%beta, %mean_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x1x64xf32>, tensor<1x8x64xf32> -> tensor<1x8x64xf32>
    %result = IE.Add(%x_scale, %beta_minus_mean_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x8x64xf32>, tensor<1x8x64xf32> -> tensor<1x8x64xf32>
    return %result : tensor<1x8x64xf32>

    // CHECK-NOT: IE.ReduceMean
    // CHECK-NOT: IE.Sqrt
    // CHECK-NOT: IE.Divide
    // CHECK-NOT: IE.Subtract
    // CHECK: [[MVN:%.+]] = IE.MVN
    // CHECK: [[POST:%.+]] = IE.AffineReshape([[MVN]])
    // CHECK: IE.Multiply([[POST]]
    // CHECK: IE.Add
    // CHECK: return
}
