//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --expand-attention --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU6010

// CHECK-LABEL: @ExpandAttentionSharedRoot
// CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x1500x512xf32>, [[INPUT1:%.+]]: tensor<1x1500x512xf32>) -> (tensor<1x1500x512xf32>, tensor<1x1500x8x64xf32>)
func.func @ExpandAttentionSharedRoot(%input0: tensor<1x1500x512xf32>, %input1: tensor<1x1500x512xf32>) -> (tensor<1x1500x512xf32>, tensor<1x1500x8x64xf32>) {
  %scale_ln = const.Declare tensor<1x1x512xf32> = dense<2.000000e+00> : tensor<1x1x512xf32>
  %bias_ln = const.Declare tensor<1x1x512xf32> = dense<3.000000e+00> : tensor<1x1x512xf32>
  %wq = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %wk = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %wv = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %wq_input_low = const.Declare tensor<1x1xf32> = dense<0.000000e+00> : tensor<1x1xf32>
  %wq_input_high = const.Declare tensor<1x1xf32> = dense<1.000000e+00> : tensor<1x1xf32>
  %wq_output_low = const.Declare tensor<512x1xf32> = dense<0.000000e+00> : tensor<512x1xf32>
  %wq_output_high = const.Declare tensor<512x1xf32> = dense<1.000000e+00> : tensor<512x1xf32>
  %v_bias = const.Declare tensor<1x1x512xf32> = dense<0.000000e+00> : tensor<1x1x512xf32>
  %q_scale = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
  %q_bias = const.Declare tensor<1x1x512xf32> = dense<0.000000e+00> : tensor<1x1x512xf32>
  %attention_scale = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>

  %root = IE.Add(%input0, %input1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>
  %mvn = IE.MVN6(%root) {axes_value = [2], eps = 9.9999997473787516E-6 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>

  %ln_mul = IE.Multiply(%mvn, %scale_ln) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>
  %ln_add = IE.Add(%ln_mul, %bias_ln) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>

  %wq_fq = IE.FakeQuantize(%wq, %wq_input_low, %wq_input_high, %wq_output_low, %wq_output_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<512x512xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<512x1xf32>, tensor<512x1xf32> -> tensor<512x512xf32>
  %q_reshape_in = IE.Reshape(%ln_add) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %q_fc = IE.FullyConnected(%q_reshape_in, %wq_fq) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %q_reshape_out = IE.Reshape(%q_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %q_mul = IE.Multiply(%q_reshape_out, %q_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x1xf32> -> tensor<1x1500x512xf32>
  %q_add = IE.Add(%q_mul, %q_bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>
  %q_affine = IE.AffineReshape(%q_add) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %q_transpose = IE.Transpose(%q_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %k_reshape_in = IE.Reshape(%ln_add) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %k_fc = IE.FullyConnected(%k_reshape_in, %wk) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %k_reshape_out = IE.Reshape(%k_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %k_affine = IE.AffineReshape(%k_reshape_out) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %k_transpose = IE.Transpose(%k_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %v_reshape_in = IE.Reshape(%ln_add) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %v_fc = IE.FullyConnected(%v_reshape_in, %wv) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %v_reshape_out = IE.Reshape(%v_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %v_add = IE.Add(%v_reshape_out, %v_bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>
  %v_affine = IE.AffineReshape(%v_add) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %v_transpose = IE.Transpose(%v_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %attention = IE.Attention(%q_transpose, %k_transpose, %v_transpose, %attention_scale) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x8x1500x64xf32>, tensor<1x8x1500x64xf32>, tensor<1x8x1500x64xf32>, tensor<1xf32> -> tensor<1x8x1500x64xf32>
  %out = IE.Transpose(%attention) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x8x1500x64xf32> -> tensor<1x1500x8x64xf32>
  return %root, %out : tensor<1x1500x512xf32>, tensor<1x1500x8x64xf32>

  // CHECK: [[ROOT_PAD:%.+]] = const.Declare tensor<1x4x512xf32> = dense<0.000000e+00> : tensor<1x4x512xf32>
  // CHECK: [[ROOT:%.+]] = IE.Add([[INPUT0]], [[INPUT1]])
  // CHECK-SAME: tensor<1x1500x512xf32>, tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>
  // CHECK: [[PADDED_ROOT:%.+]] = IE.Concat([[ROOT]], [[ROOT_PAD]])
  // CHECK-SAME: tensor<1x1500x512xf32>, tensor<1x4x512xf32> -> tensor<1x1504x512xf32>
  // CHECK: [[PADDED_MVN:%.+]] = IE.MVN6([[PADDED_ROOT]])
  // CHECK: [[PADDED_MUL:%.+]] = IE.Multiply([[PADDED_MVN]]
  // CHECK: [[PADDED_ADD:%.+]] = IE.Add([[PADDED_MUL]]

  // CHECK: [[Q_RESHAPE_IN:%.+]] = IE.AffineReshape([[PADDED_ADD]])
  // CHECK-SAME: shape_value = [1504, 512]
  // CHECK-SAME: tensor<1x1504x512xf32> -> tensor<1504x512xf32>
  // CHECK: [[Q_FC:%.+]] = IE.FullyConnected([[Q_RESHAPE_IN]],
  // CHECK-SAME: tensor<1504x512xf32>, tensor<512x512xf32> -> tensor<1504x512xf32>
  // CHECK: [[Q_AFFINE:%.+]] = IE.AffineReshape([[Q_FC]])
  // CHECK-SAME: shape_value = [1, 1504, 8, 64]
  // CHECK-SAME: tensor<1504x512xf32> -> tensor<1x1504x8x64xf32>
  // CHECK: [[Q_TRANSPOSE:%.+]] = IE.Transpose([[Q_AFFINE]])
  // CHECK-SAME: tensor<1x1504x8x64xf32> -> tensor<1x8x1504x64xf32>

  // CHECK: [[K_RESHAPE_IN:%.+]] = IE.AffineReshape([[PADDED_ADD]])
  // CHECK-SAME: shape_value = [1504, 512]
  // CHECK-SAME: tensor<1x1504x512xf32> -> tensor<1504x512xf32>
  // CHECK: [[K_FC:%.+]] = IE.FullyConnected([[K_RESHAPE_IN]],
  // CHECK-SAME: tensor<1504x512xf32>, tensor<512x512xf32> -> tensor<1504x512xf32>
  // CHECK: [[K_AFFINE:%.+]] = IE.AffineReshape([[K_FC]])
  // CHECK-SAME: shape_value = [1, 1504, 8, 64]
  // CHECK-SAME: tensor<1504x512xf32> -> tensor<1x1504x8x64xf32>
  // CHECK: [[K_TRANSPOSE:%.+]] = IE.Transpose([[K_AFFINE]])
  // CHECK-SAME: tensor<1x1504x8x64xf32> -> tensor<1x8x1504x64xf32>

  // CHECK: [[V_RESHAPE_IN:%.+]] = IE.AffineReshape([[PADDED_ADD]])
  // CHECK-SAME: shape_value = [1504, 512]
  // CHECK-SAME: tensor<1x1504x512xf32> -> tensor<1504x512xf32>
  // CHECK: [[V_FC:%.+]] = IE.FullyConnected([[V_RESHAPE_IN]],
  // CHECK-SAME: tensor<1504x512xf32>, tensor<512x512xf32> -> tensor<1504x512xf32>
  // CHECK: [[V_AFFINE:%.+]] = IE.AffineReshape([[V_FC]])
  // CHECK-SAME: shape_value = [1, 1504, 8, 64]
  // CHECK-SAME: tensor<1504x512xf32> -> tensor<1x1504x8x64xf32>
  // CHECK: [[V_TRANSPOSE:%.+]] = IE.Transpose([[V_AFFINE]])
  // CHECK-SAME: tensor<1x1504x8x64xf32> -> tensor<1x8x1504x64xf32>

  // CHECK: [[EXPANDED_ATTENTION:%.+]] = IE.Attention([[Q_TRANSPOSE]], [[K_TRANSPOSE]], [[V_TRANSPOSE]],
  // CHECK-SAME: padSizeS = 4 : i64
  // CHECK-SAME: tensor<1x8x1504x64xf32>, tensor<1x8x1504x64xf32>, tensor<1x8x1504x64xf32>, tensor<1xf32> -> tensor<1x8x1504x64xf32>
  // CHECK: [[EXPANDED_TRANSPOSE:%.+]] = IE.Transpose([[EXPANDED_ATTENTION]])
  // CHECK-SAME: tensor<1x8x1504x64xf32> -> tensor<1x1504x8x64xf32>
  // CHECK: [[SLICE:%.+]] = IE.Slice [[EXPANDED_TRANSPOSE]] [0, 0, 0, 0] [1, 1500, 8, 64] : tensor<1x1504x8x64xf32> to tensor<1x1500x8x64xf32>
  // CHECK: return [[ROOT]], [[SLICE]]
}

// -----

// CHECK-LABEL: @ExpandAttentionWithMultiUsesRootOps
// CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x1500x512xf32>, [[INPUT1:%.+]]: tensor<1x1x512xf32>) -> (tensor<1x1500x512xf32>, tensor<1x1500x8x64xf32>)
func.func @ExpandAttentionWithMultiUsesRootOps(%input0: tensor<1x1500x512xf32>, %input1: tensor<1x1x512xf32>) -> (tensor<1x1500x512xf32>, tensor<1x1500x8x64xf32>) {
  %scale_ln = const.Declare tensor<1x1x512xf32> = dense<2.000000e+00> : tensor<1x1x512xf32>
  %bias_ln = const.Declare tensor<1x1x512xf32> = dense<3.000000e+00> : tensor<1x1x512xf32>
  %wq = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %wk = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %wv = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %v_bias = const.Declare tensor<1x1x512xf32> = dense<0.000000e+00> : tensor<1x1x512xf32>
  %q_scale = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
  %q_bias = const.Declare tensor<1x1x512xf32> = dense<0.000000e+00> : tensor<1x1x512xf32>
  %attention_scale = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>

  %mul = IE.Multiply(%input0, %input1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>
  %mvn = IE.MVN6(%mul) {axes_value = [2], eps = 9.9999997473787516E-6 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>

  %ln_mul = IE.Multiply(%mvn, %scale_ln) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>
  %ln_add = IE.Add(%ln_mul, %bias_ln) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>

  %q_reshape_in = IE.Reshape(%ln_add) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %q_fc = IE.FullyConnected(%q_reshape_in, %wq) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %q_reshape_out = IE.Reshape(%q_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %q_mul = IE.Multiply(%q_reshape_out, %q_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x1xf32> -> tensor<1x1500x512xf32>
  %q_add = IE.Add(%q_mul, %q_bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>
  %q_affine = IE.AffineReshape(%q_add) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %q_transpose = IE.Transpose(%q_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %k_reshape_in = IE.Reshape(%ln_add) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %k_fc = IE.FullyConnected(%k_reshape_in, %wk) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %k_reshape_out = IE.Reshape(%k_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %k_affine = IE.AffineReshape(%k_reshape_out) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %k_transpose = IE.Transpose(%k_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %v_reshape_in = IE.Reshape(%ln_add) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %v_fc = IE.FullyConnected(%v_reshape_in, %wv) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %v_reshape_out = IE.Reshape(%v_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %v_add = IE.Add(%v_reshape_out, %v_bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>
  %v_affine = IE.AffineReshape(%v_add) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %v_transpose = IE.Transpose(%v_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %attention = IE.Attention(%q_transpose, %k_transpose, %v_transpose, %attention_scale) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x8x1500x64xf32>, tensor<1x8x1500x64xf32>, tensor<1x8x1500x64xf32>, tensor<1xf32> -> tensor<1x8x1500x64xf32>
  %out = IE.Transpose(%attention) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x8x1500x64xf32> -> tensor<1x1500x8x64xf32>
  return %mvn, %out : tensor<1x1500x512xf32>, tensor<1x1500x8x64xf32>

  // CHECK: [[ROOT_PAD:%.+]] = const.Declare tensor<1x4x512xf32> = dense<0.000000e+00> : tensor<1x4x512xf32>
  // CHECK: [[SCALE_LN:%.+]] = const.Declare tensor<1x1x512xf32> = dense<2.000000e+00> : tensor<1x1x512xf32>
  // CHECK: [[BIAS_LN:%.+]] = const.Declare tensor<1x1x512xf32> = dense<3.000000e+00> : tensor<1x1x512xf32>
  // CHECK: [[WEIGHT:%.+]] = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  // CHECK: [[ATTENTION_SCALE:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>
  // CHECK: [[MUL:%.+]] = IE.Multiply([[INPUT0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK-SAME: tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>
  // CHECK: [[MVN:%.+]] = IE.MVN6([[MUL]])
  // CHECK-SAME: tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>
  // CHECK: [[PADDED_ROOT:%.+]] = IE.Concat([[MVN]], [[ROOT_PAD]])
  // CHECK-SAME: tensor<1x1500x512xf32>, tensor<1x4x512xf32> -> tensor<1x1504x512xf32>
  // CHECK: [[PADDED_MUL:%.+]] = IE.Multiply([[PADDED_ROOT]], [[SCALE_LN]])
  // CHECK-SAME: tensor<1x1504x512xf32>, tensor<1x1x512xf32> -> tensor<1x1504x512xf32>
  // CHECK: [[PADDED_ADD:%.+]] = IE.Add([[PADDED_MUL]], [[BIAS_LN]])
  // CHECK-SAME: tensor<1x1504x512xf32>, tensor<1x1x512xf32> -> tensor<1x1504x512xf32>

  // CHECK: [[Q_RESHAPE_IN:%.+]] = IE.AffineReshape([[PADDED_ADD]])
  // CHECK-SAME: shape_value = [1504, 512]
  // CHECK-SAME: tensor<1x1504x512xf32> -> tensor<1504x512xf32>
  // CHECK: [[Q_FC:%.+]] = IE.FullyConnected([[Q_RESHAPE_IN]], [[WEIGHT]])
  // CHECK-SAME: tensor<1504x512xf32>, tensor<512x512xf32> -> tensor<1504x512xf32>
  // CHECK: [[Q_AFFINE:%.+]] = IE.AffineReshape([[Q_FC]])
  // CHECK-SAME: shape_value = [1, 1504, 8, 64]
  // CHECK-SAME: tensor<1504x512xf32> -> tensor<1x1504x8x64xf32>
  // CHECK: [[Q_TRANSPOSE:%.+]] = IE.Transpose([[Q_AFFINE]])
  // CHECK-SAME: tensor<1x1504x8x64xf32> -> tensor<1x8x1504x64xf32>

  // CHECK: [[K_RESHAPE_IN:%.+]] = IE.AffineReshape([[PADDED_ADD]])
  // CHECK-SAME: shape_value = [1504, 512]
  // CHECK-SAME: tensor<1x1504x512xf32> -> tensor<1504x512xf32>
  // CHECK: [[K_FC:%.+]] = IE.FullyConnected([[K_RESHAPE_IN]], [[WEIGHT]])
  // CHECK-SAME: tensor<1504x512xf32>, tensor<512x512xf32> -> tensor<1504x512xf32>
  // CHECK: [[K_AFFINE:%.+]] = IE.AffineReshape([[K_FC]])
  // CHECK-SAME: shape_value = [1, 1504, 8, 64]
  // CHECK-SAME: tensor<1504x512xf32> -> tensor<1x1504x8x64xf32>
  // CHECK: [[K_TRANSPOSE:%.+]] = IE.Transpose([[K_AFFINE]])
  // CHECK-SAME: tensor<1x1504x8x64xf32> -> tensor<1x8x1504x64xf32>

  // CHECK: [[V_RESHAPE_IN:%.+]] = IE.AffineReshape([[PADDED_ADD]])
  // CHECK-SAME: shape_value = [1504, 512]
  // CHECK-SAME: tensor<1x1504x512xf32> -> tensor<1504x512xf32>
  // CHECK: [[V_FC:%.+]] = IE.FullyConnected([[V_RESHAPE_IN]], [[WEIGHT]])
  // CHECK-SAME: tensor<1504x512xf32>, tensor<512x512xf32> -> tensor<1504x512xf32>
  // CHECK: [[V_AFFINE:%.+]] = IE.AffineReshape([[V_FC]])
  // CHECK-SAME: shape_value = [1, 1504, 8, 64]
  // CHECK-SAME: tensor<1504x512xf32> -> tensor<1x1504x8x64xf32>
  // CHECK: [[V_TRANSPOSE:%.+]] = IE.Transpose([[V_AFFINE]])
  // CHECK-SAME: tensor<1x1504x8x64xf32> -> tensor<1x8x1504x64xf32>

  // CHECK: [[EXPANDED_ATTENTION:%.+]] = IE.Attention([[Q_TRANSPOSE]], [[K_TRANSPOSE]], [[V_TRANSPOSE]], [[ATTENTION_SCALE]])
  // CHECK-SAME: padSizeS = 4 : i64
  // CHECK-SAME: tensor<1x8x1504x64xf32>, tensor<1x8x1504x64xf32>, tensor<1x8x1504x64xf32>, tensor<1xf32> -> tensor<1x8x1504x64xf32>
  // CHECK: [[EXPANDED_TRANSPOSE:%.+]] = IE.Transpose([[EXPANDED_ATTENTION]])
  // CHECK-SAME: tensor<1x8x1504x64xf32> -> tensor<1x1504x8x64xf32>
  // CHECK: [[SLICE:%.+]] = IE.Slice [[EXPANDED_TRANSPOSE]] [0, 0, 0, 0] [1, 1500, 8, 64] : tensor<1x1504x8x64xf32> to tensor<1x1500x8x64xf32>
  // CHECK: return [[MVN]], [[SLICE]]
}

// -----

// CHECK-LABEL: @ExpandAttentionSharedAffineReshapeRoot
// CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x1500x512xf32>, [[INPUT1:%.+]]: tensor<1x1500x512xf32>) -> (tensor<1x1500x512xf32>, tensor<1x8x1500x64xf32>)
func.func @ExpandAttentionSharedAffineReshapeRoot(%input0: tensor<1x1500x512xf32>, %input1: tensor<1x1500x512xf32>) -> (tensor<1x1500x512xf32>, tensor<1x8x1500x64xf32>) {
  %scale_ln = const.Declare tensor<1x1x512xf32> = dense<2.000000e+00> : tensor<1x1x512xf32>
  %bias_ln = const.Declare tensor<1x1x512xf32> = dense<3.000000e+00> : tensor<1x1x512xf32>
  %q_scale = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
  %q_bias = const.Declare tensor<1x1x512xf32> = dense<0.000000e+00> : tensor<1x1x512xf32>
  %attention_scale = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>
  %wq = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %wk = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %wv = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %v_bias = const.Declare tensor<1x1x512xf32> = dense<0.000000e+00> : tensor<1x1x512xf32>

  %root = IE.Add(%input0, %input1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>
  %mvn = IE.MVN6(%root) {axes_value = [2], eps = 9.9999997473787516E-6 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>

  %ln_mul = IE.Multiply(%mvn, %scale_ln) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>
  %ln_add = IE.Add(%ln_mul, %bias_ln) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>
  %shared_reshape = IE.AffineReshape(%ln_add) {dim_mapping = [[0], [0], [1]], shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>

  %q_fc = IE.FullyConnected(%shared_reshape, %wq) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %q_reshape_out = IE.AffineReshape(%q_fc) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %q_mul = IE.Multiply(%q_reshape_out, %q_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x1xf32> -> tensor<1x1500x512xf32>
  %q_add = IE.Add(%q_mul, %q_bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>
  %q_affine = IE.AffineReshape(%q_add) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %q_transpose = IE.Transpose(%q_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %k_fc = IE.FullyConnected(%shared_reshape, %wk) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %k_affine = IE.AffineReshape(%k_fc) {dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1500x512xf32> -> tensor<1x1500x8x64xf32>
  %k_transpose = IE.Transpose(%k_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %v_fc = IE.FullyConnected(%shared_reshape, %wv) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %v_reshape_out = IE.AffineReshape(%v_fc) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %v_add = IE.Add(%v_reshape_out, %v_bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>
  %v_affine = IE.AffineReshape(%v_add) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %v_transpose = IE.Transpose(%v_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x64x1500xf32>

  %attention = IE.Attention(%q_transpose, %k_transpose, %v_transpose, %attention_scale) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x8x1500x64xf32>, tensor<1x8x1500x64xf32>, tensor<1x8x64x1500xf32>, tensor<1xf32> -> tensor<1x8x1500x64xf32>
  return %root, %attention : tensor<1x1500x512xf32>, tensor<1x8x1500x64xf32>

  // CHECK: [[ROOT_PAD:%.+]] = const.Declare tensor<1x4x512xf32> = dense<0.000000e+00> : tensor<1x4x512xf32>
  // CHECK: [[PADDED_ROOT:%.+]] = IE.Concat(
  // CHECK-SAME: [[ROOT_PAD]])
  // CHECK-SAME: tensor<1x1500x512xf32>, tensor<1x4x512xf32> -> tensor<1x1504x512xf32>
  // CHECK: [[PADDED_MVN:%.+]] = IE.MVN6([[PADDED_ROOT]])
  // CHECK: [[PADDED_MUL:%.+]] = IE.Multiply([[PADDED_MVN]]
  // CHECK-SAME: tensor<1x1504x512xf32>, tensor<1x1x512xf32> -> tensor<1x1504x512xf32>
  // CHECK: [[PADDED_ADD:%.+]] = IE.Add([[PADDED_MUL]]
  // CHECK-SAME: tensor<1x1504x512xf32>, tensor<1x1x512xf32> -> tensor<1x1504x512xf32>

  // CHECK: [[SHARED_RESHAPE:%.+]] = IE.AffineReshape([[PADDED_ADD]])
  // CHECK: [[Q_FC:%.+]] = IE.FullyConnected([[SHARED_RESHAPE]],
  // CHECK: [[Q_AFFINE:%.+]] = IE.AffineReshape([[Q_FC]])
  // CHECK: [[Q_TRANSPOSE:%.+]] = IE.Transpose([[Q_AFFINE]])

  // CHECK: [[K_FC:%.+]] = IE.FullyConnected([[SHARED_RESHAPE]],
  // CHECK: [[K_AFFINE:%.+]] = IE.AffineReshape([[K_FC]])
  // CHECK: [[K_TRANSPOSE:%.+]] = IE.Transpose([[K_AFFINE]])

  // CHECK: [[V_FC:%.+]] = IE.FullyConnected([[SHARED_RESHAPE]],
  // CHECK: [[V_AFFINE:%.+]] = IE.AffineReshape([[V_FC]])
  // CHECK: [[V_TRANSPOSE:%.+]] = IE.Transpose([[V_AFFINE]])

  // CHECK: [[EXPANDED_ATTENTION:%.+]] = IE.Attention([[Q_TRANSPOSE]], [[K_TRANSPOSE]], [[V_TRANSPOSE]],
  // CHECK-SAME: padSizeS = 4 : i64
  // CHECK-SAME: tensor<1x8x1504x64xf32>, tensor<1x8x1504x64xf32>, tensor<1x8x64x1504xf32>, tensor<1xf32> -> tensor<1x8x1504x64xf32>
  // CHECK: [[SLICE:%.+]] = IE.Slice [[EXPANDED_ATTENTION]] [0, 0, 0, 0] [1, 8, 1500, 64] : tensor<1x8x1504x64xf32> to tensor<1x8x1500x64xf32>
  // CHECK: return {{%.+}}, [[SLICE]]
}

// -----

// CHECK-LABEL: @ExpandAttentionSharedRootTransposedV
// CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x1500x512xf32>, [[INPUT1:%.+]]: tensor<1x1500x512xf32>) -> (tensor<1x1500x512xf32>, tensor<1x1500x8x64xf32>)
func.func @ExpandAttentionSharedRootTransposedV(%input0: tensor<1x1500x512xf32>, %input1: tensor<1x1500x512xf32>) -> (tensor<1x1500x512xf32>, tensor<1x1500x8x64xf32>) {
  %scale_ln = const.Declare tensor<1x1x512xf32> = dense<2.000000e+00> : tensor<1x1x512xf32>
  %bias_ln = const.Declare tensor<1x1x512xf32> = dense<3.000000e+00> : tensor<1x1x512xf32>
  %wq = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %wk = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %wv = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %v_bias = const.Declare tensor<1x1x512xf32> = dense<0.000000e+00> : tensor<1x1x512xf32>
  %q_scale = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
  %q_bias = const.Declare tensor<1x1x512xf32> = dense<0.000000e+00> : tensor<1x1x512xf32>
  %attention_scale = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>

  %root = IE.Add(%input0, %input1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>
  %mvn = IE.MVN6(%root) {axes_value = [2], eps = 9.9999997473787516E-6 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>

  %ln_mul = IE.Multiply(%mvn, %scale_ln) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>
  %ln_add = IE.Add(%ln_mul, %bias_ln) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>

  %q_reshape_in = IE.Reshape(%ln_add) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %q_fc = IE.FullyConnected(%q_reshape_in, %wq) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %q_reshape_out = IE.Reshape(%q_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %q_mul = IE.Multiply(%q_reshape_out, %q_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x1xf32> -> tensor<1x1500x512xf32>
  %q_add = IE.Add(%q_mul, %q_bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>
  %q_affine = IE.AffineReshape(%q_add) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %q_transpose = IE.Transpose(%q_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %k_reshape_in = IE.Reshape(%ln_add) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %k_fc = IE.FullyConnected(%k_reshape_in, %wk) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %k_reshape_out = IE.Reshape(%k_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %k_affine = IE.AffineReshape(%k_reshape_out) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %k_transpose = IE.Transpose(%k_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %v_reshape_in = IE.Reshape(%ln_add) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %v_fc = IE.FullyConnected(%v_reshape_in, %wv) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %v_reshape_out = IE.Reshape(%v_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %v_add = IE.Add(%v_reshape_out, %v_bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>
  %v_affine = IE.AffineReshape(%v_add) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %v_transpose = IE.Transpose(%v_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>
  %v_transposed = IE.Transpose(%v_transpose) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x8x1500x64xf32> -> tensor<1x8x64x1500xf32>

  %attention = IE.Attention(%q_transpose, %k_transpose, %v_transposed, %attention_scale) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x8x1500x64xf32>, tensor<1x8x1500x64xf32>, tensor<1x8x64x1500xf32>, tensor<1xf32> -> tensor<1x8x1500x64xf32>
  %out = IE.Transpose(%attention) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x8x1500x64xf32> -> tensor<1x1500x8x64xf32>
  return %root, %out : tensor<1x1500x512xf32>, tensor<1x1500x8x64xf32>

  // CHECK: [[ROOT_PAD:%.+]] = const.Declare tensor<1x4x512xf32> = dense<0.000000e+00> : tensor<1x4x512xf32>
  // CHECK: [[PADDED_ROOT:%.+]] = IE.Concat(
  // CHECK-SAME: [[ROOT_PAD]])
  // CHECK-SAME: tensor<1x1500x512xf32>, tensor<1x4x512xf32> -> tensor<1x1504x512xf32>
  // CHECK: [[PADDED_MVN:%.+]] = IE.MVN6([[PADDED_ROOT]])
  // CHECK: [[PADDED_MUL:%.+]] = IE.Multiply([[PADDED_MVN]]
  // CHECK: [[PADDED_ADD:%.+]] = IE.Add([[PADDED_MUL]]

  // CHECK: [[Q_RESHAPE_IN:%.+]] = IE.AffineReshape([[PADDED_ADD]])
  // CHECK-SAME: shape_value = [1504, 512]
  // CHECK-SAME: tensor<1x1504x512xf32> -> tensor<1504x512xf32>
  // CHECK: [[Q_FC:%.+]] = IE.FullyConnected([[Q_RESHAPE_IN]],
  // CHECK-SAME: tensor<1504x512xf32>, tensor<512x512xf32> -> tensor<1504x512xf32>
  // CHECK: [[Q_AFFINE:%.+]] = IE.AffineReshape([[Q_FC]])
  // CHECK-SAME: shape_value = [1, 1504, 8, 64]
  // CHECK-SAME: tensor<1504x512xf32> -> tensor<1x1504x8x64xf32>
  // CHECK: [[Q_TRANSPOSE:%.+]] = IE.Transpose([[Q_AFFINE]])
  // CHECK-SAME: tensor<1x1504x8x64xf32> -> tensor<1x8x1504x64xf32>

  // CHECK: [[K_RESHAPE_IN:%.+]] = IE.AffineReshape([[PADDED_ADD]])
  // CHECK-SAME: shape_value = [1504, 512]
  // CHECK-SAME: tensor<1x1504x512xf32> -> tensor<1504x512xf32>
  // CHECK: [[K_FC:%.+]] = IE.FullyConnected([[K_RESHAPE_IN]],
  // CHECK-SAME: tensor<1504x512xf32>, tensor<512x512xf32> -> tensor<1504x512xf32>
  // CHECK: [[K_AFFINE:%.+]] = IE.AffineReshape([[K_FC]])
  // CHECK-SAME: shape_value = [1, 1504, 8, 64]
  // CHECK-SAME: tensor<1504x512xf32> -> tensor<1x1504x8x64xf32>
  // CHECK: [[K_TRANSPOSE:%.+]] = IE.Transpose([[K_AFFINE]])
  // CHECK-SAME: tensor<1x1504x8x64xf32> -> tensor<1x8x1504x64xf32>

  // CHECK: [[V_RESHAPE_IN:%.+]] = IE.AffineReshape([[PADDED_ADD]])
  // CHECK-SAME: shape_value = [1504, 512]
  // CHECK-SAME: tensor<1x1504x512xf32> -> tensor<1504x512xf32>
  // CHECK: [[V_FC:%.+]] = IE.FullyConnected([[V_RESHAPE_IN]],
  // CHECK-SAME: tensor<1504x512xf32>, tensor<512x512xf32> -> tensor<1504x512xf32>
  // CHECK: [[V_AFFINE:%.+]] = IE.AffineReshape([[V_FC]])
  // CHECK-SAME: shape_value = [1, 1504, 8, 64]
  // CHECK-SAME: tensor<1504x512xf32> -> tensor<1x1504x8x64xf32>
  // CHECK: [[V_TRANSPOSED:%.+]] = IE.Transpose([[V_AFFINE]])
  // CHECK-SAME: tensor<1x1504x8x64xf32> -> tensor<1x8x64x1504xf32>

  // CHECK: [[EXPANDED_ATTENTION:%.+]] = IE.Attention([[Q_TRANSPOSE]], [[K_TRANSPOSE]], [[V_TRANSPOSED]],
  // CHECK-SAME: padSizeS = 4 : i64
  // CHECK-SAME: tensor<1x8x1504x64xf32>, tensor<1x8x1504x64xf32>, tensor<1x8x64x1504xf32>, tensor<1xf32> -> tensor<1x8x1504x64xf32>
  // CHECK: [[EXPANDED_TRANSPOSE:%.+]] = IE.Transpose([[EXPANDED_ATTENTION]])
  // CHECK-SAME: tensor<1x8x1504x64xf32> -> tensor<1x1504x8x64xf32>
  // CHECK: [[SLICE:%.+]] = IE.Slice [[EXPANDED_TRANSPOSE]] [0, 0, 0, 0] [1, 1500, 8, 64] : tensor<1x1504x8x64xf32> to tensor<1x1500x8x64xf32>
  // CHECK: return {{%.+}}, [[SLICE]]
}

// -----

// CHECK-LABEL: @NotExpandAttentionWithAttentionMask
// CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x1500x512xf32>, [[INPUT1:%.+]]: tensor<1x1500x512xf32>) -> (tensor<1x1500x512xf32>, tensor<1x8x1500x64xf32>)
func.func @NotExpandAttentionWithAttentionMask(%input0: tensor<1x1500x512xf32>, %input1: tensor<1x1500x512xf32>) -> (tensor<1x1500x512xf32>, tensor<1x8x1500x64xf32>) {
  %wq = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %wk = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %wv = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %mask = const.Declare tensor<1x1x1500x1500xf32> = dense<0.000000e+00> : tensor<1x1x1500x1500xf32>
  %root = IE.Add(%input0, %input1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>
  %mvn = IE.MVN6(%root) {axes_value = [2], eps = 9.9999997473787516E-6 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>

  %q_reshape_in = IE.Reshape(%mvn) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %q_fc = IE.FullyConnected(%q_reshape_in, %wq) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %q_reshape_out = IE.Reshape(%q_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %q_affine = IE.AffineReshape(%q_reshape_out) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %q_transpose = IE.Transpose(%q_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %k_reshape_in = IE.Reshape(%mvn) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %k_fc = IE.FullyConnected(%k_reshape_in, %wk) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %k_reshape_out = IE.Reshape(%k_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %k_affine = IE.AffineReshape(%k_reshape_out) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %k_transpose = IE.Transpose(%k_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %v_reshape_in = IE.Reshape(%mvn) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %v_fc = IE.FullyConnected(%v_reshape_in, %wv) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %v_reshape_out = IE.Reshape(%v_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %v_affine = IE.AffineReshape(%v_reshape_out) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %v_transpose = IE.Transpose(%v_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %attention = IE.Attention(%q_transpose, %k_transpose, %v_transpose, %mask) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0>} : tensor<1x8x1500x64xf32>, tensor<1x8x1500x64xf32>, tensor<1x8x1500x64xf32>, tensor<1x1x1500x1500xf32> -> tensor<1x8x1500x64xf32>
  return %root, %attention : tensor<1x1500x512xf32>, tensor<1x8x1500x64xf32>

  // CHECK-NOT: IE.Concat
  // CHECK: [[MASK:%.+]] = const.Declare tensor<1x1x1500x1500xf32>
  // CHECK: [[ROOT:%.+]] = IE.Add([[INPUT0]], [[INPUT1]])
  // CHECK-SAME: tensor<1x1500x512xf32>, tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>
  // CHECK: [[MVN:%.+]] = IE.MVN6([[ROOT]])
  // CHECK: [[Q_RESHAPE_IN:%.+]] = IE.AffineReshape([[MVN]])
  // CHECK-SAME: shape_value = [1500, 512]
  // CHECK: [[Q_FC:%.+]] = IE.FullyConnected([[Q_RESHAPE_IN]],
  // CHECK: [[Q_AFFINE:%.+]] = IE.AffineReshape([[Q_FC]])
  // CHECK-SAME: shape_value = [1, 1500, 8, 64]
  // CHECK: [[Q_TRANSPOSE:%.+]] = IE.Transpose([[Q_AFFINE]])
  // CHECK: [[ATTENTION:%.+]] = IE.Attention([[Q_TRANSPOSE]],
  // CHECK-SAME: [[MASK]])
  // CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0>
  // CHECK-NOT: IE.Slice
  // CHECK: return [[ROOT]], [[ATTENTION]]
}

// -----

// CHECK-LABEL: @NotExpandAttentionWithNonSplatScale
// CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x1500x512xf32>, [[INPUT1:%.+]]: tensor<1x1500x512xf32>) -> (tensor<1x1500x512xf32>, tensor<1x8x1500x64xf32>)
func.func @NotExpandAttentionWithNonSplatScale(%input0: tensor<1x1500x512xf32>, %input1: tensor<1x1500x512xf32>) -> (tensor<1x1500x512xf32>, tensor<1x8x1500x64xf32>) {
  %wq = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %wk = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %wv = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %attention_scale = const.Declare tensor<1x8xf32> = dense<[[3.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]]> : tensor<1x8xf32>
  %root = IE.Add(%input0, %input1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>
  %mvn = IE.MVN6(%root) {axes_value = [2], eps = 9.9999997473787516E-6 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>

  %q_reshape_in = IE.Reshape(%mvn) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %q_fc = IE.FullyConnected(%q_reshape_in, %wq) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %q_reshape_out = IE.Reshape(%q_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %q_affine = IE.AffineReshape(%q_reshape_out) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %q_transpose = IE.Transpose(%q_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %k_reshape_in = IE.Reshape(%mvn) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %k_fc = IE.FullyConnected(%k_reshape_in, %wk) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %k_reshape_out = IE.Reshape(%k_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %k_affine = IE.AffineReshape(%k_reshape_out) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %k_transpose = IE.Transpose(%k_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %v_reshape_in = IE.Reshape(%mvn) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %v_fc = IE.FullyConnected(%v_reshape_in, %wv) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %v_reshape_out = IE.Reshape(%v_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %v_affine = IE.AffineReshape(%v_reshape_out) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %v_transpose = IE.Transpose(%v_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %attention = IE.Attention(%q_transpose, %k_transpose, %v_transpose, %attention_scale) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x8x1500x64xf32>, tensor<1x8x1500x64xf32>, tensor<1x8x1500x64xf32>, tensor<1x8xf32> -> tensor<1x8x1500x64xf32>
  return %root, %attention : tensor<1x1500x512xf32>, tensor<1x8x1500x64xf32>

  // CHECK: [[SCALE:%.+]] = const.Declare tensor<1x8xf32>
  // CHECK-NOT: IE.Concat
  // CHECK: [[ROOT:%.+]] = IE.Add([[INPUT0]], [[INPUT1]])
  // CHECK-SAME: tensor<1x1500x512xf32>, tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>
  // CHECK: [[MVN:%.+]] = IE.MVN6([[ROOT]])
  // CHECK: [[Q_RESHAPE_IN:%.+]] = IE.AffineReshape([[MVN]])
  // CHECK-SAME: shape_value = [1500, 512]
  // CHECK: [[Q_FC:%.+]] = IE.FullyConnected([[Q_RESHAPE_IN]],
  // CHECK: [[Q_AFFINE:%.+]] = IE.AffineReshape([[Q_FC]])
  // CHECK-SAME: shape_value = [1, 1500, 8, 64]
  // CHECK: [[Q_TRANSPOSE:%.+]] = IE.Transpose([[Q_AFFINE]])

  // CHECK: [[K_RESHAPE_IN:%.+]] = IE.AffineReshape([[MVN]])
  // CHECK-SAME: shape_value = [1500, 512]
  // CHECK: [[K_FC:%.+]] = IE.FullyConnected([[K_RESHAPE_IN]],
  // CHECK: [[K_AFFINE:%.+]] = IE.AffineReshape([[K_FC]])
  // CHECK-SAME: shape_value = [1, 1500, 8, 64]
  // CHECK: [[K_TRANSPOSE:%.+]] = IE.Transpose([[K_AFFINE]])

  // CHECK: [[V_RESHAPE_IN:%.+]] = IE.AffineReshape([[MVN]])
  // CHECK-SAME: shape_value = [1500, 512]
  // CHECK: [[V_FC:%.+]] = IE.FullyConnected([[V_RESHAPE_IN]],
  // CHECK: [[V_AFFINE:%.+]] = IE.AffineReshape([[V_FC]])
  // CHECK-SAME: shape_value = [1, 1500, 8, 64]
  // CHECK: [[V_TRANSPOSE:%.+]] = IE.Transpose([[V_AFFINE]])

  // CHECK: [[ATTENTION:%.+]] = IE.Attention([[Q_TRANSPOSE]], [[K_TRANSPOSE]], [[V_TRANSPOSE]], [[SCALE]])
  // CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>
  // CHECK-NOT: IE.Slice
  // CHECK: return [[ROOT]], [[ATTENTION]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: func.func @ExpandAttentionWithFakeQuantize
// CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x577x768xf32>, [[INPUT1:%.+]]: tensor<1x577x768xf32>) -> (tensor<1x577x768xf32>, tensor<1x12x577x64xf32>)
func.func @ExpandAttentionWithFakeQuantize(%arg0: tensor<1x577x768xf32>, %arg1: tensor<1x577x768xf32>) -> (tensor<1x577x768xf32>, tensor<1x12x577x64xf32>) {
  %norm_mul = const.Declare tensor<1x1x768xf32> = dense<1.000000e+00> : tensor<1x1x768xf32>
  %norm_add = const.Declare tensor<1x1x768xf32> = dense<0.000000e+00> : tensor<1x1x768xf32>
  %cst = const.Declare tensor<1x1x768xf32> = dense<2.000000e+00> : tensor<1x1x2304xf32>, [#const.SubView<[0, 0, 0], [1, 1, 768]>]
  %cst_0 = const.Declare tensor<768x1xf32> = dense<1.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[0, 0], [768, 1]>]
  %cst_1 = const.Declare tensor<768x1xf32> = dense<0.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[0, 0], [768, 1]>]
  %cst_2 = const.Declare tensor<768x768xf32> = dense<1.000000e+00> : tensor<2304x768xf32>, [#const.SubView<[0, 0], [768, 768]>]
  %cst_3 = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1]>]
  %cst_4 = const.Declare tensor<1xf32> = dense<0.000000e+00> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1]>]
  %cst_5 = const.Declare tensor<1x1x768xf32> = dense<2.000000e+00> : tensor<1x1x2304xf32>, [#const.SubView<[0, 0, 768], [1, 1, 768]>]
  %cst_6 = const.Declare tensor<768x1xf32> = dense<1.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[768, 0], [768, 1]>]
  %cst_7 = const.Declare tensor<768x1xf32> = dense<0.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[768, 0], [768, 1]>]
  %cst_8 = const.Declare tensor<768x768xf32> = dense<1.000000e+00> : tensor<2304x768xf32>, [#const.SubView<[768, 0], [768, 768]>]
  %cst_9 = const.Declare tensor<1x1x768xf32> = dense<2.000000e+00> : tensor<1x1x2304xf32>, [#const.SubView<[0, 0, 1536], [1, 1, 768]>]
  %cst_10 = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>
  %cst_11 = const.Declare tensor<768x768xf32> = dense<1.000000e+00> : tensor<2304x768xf32>, [#const.SubView<[1536, 0], [768, 768]>]
  %cst_12 = const.Declare tensor<1xf32> = dense<0.000000e+00> : tensor<1x1xf32>, [#const.Reshape<[1]>]
  %cst_13 = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1x1xf32>, [#const.Reshape<[1]>]
  %cst_14 = const.Declare tensor<768x1xf32> = dense<0.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[1536, 0], [768, 1]>]
  %cst_15 = const.Declare tensor<768x1xf32> = dense<1.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[1536, 0], [768, 1]>]
  %root = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x577x768xf32>, tensor<1x577x768xf32> -> tensor<1x577x768xf32>
  %mvn = IE.MVN6(%root) {axes_value = [2], eps = 9.9999997473787516E-6 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<1x577x768xf32> -> tensor<1x577x768xf32>
  %0 = IE.Multiply(%mvn, %norm_mul) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x577x768xf32>, tensor<1x1x768xf32> -> tensor<1x577x768xf32>
  %1 = IE.Add(%0, %norm_add) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x577x768xf32>, tensor<1x1x768xf32> -> tensor<1x577x768xf32>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [1]], shape_value = [577, 768]} : tensor<1x577x768xf32> -> tensor<577x768xf32>
  %3 = IE.FakeQuantize(%cst_11, %cst_12, %cst_13, %cst_14, %cst_15) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<768x768xf32>, tensor<1xf32>, tensor<1xf32>, tensor<768x1xf32>, tensor<768x1xf32> -> tensor<768x768xf32>
  %4 = IE.FullyConnected(%2, %3) : tensor<577x768xf32>, tensor<768x768xf32> -> tensor<577x768xf32>
  %5 = IE.AffineReshape(%4) {dim_mapping = [[0, 1], [2]], shape_value = [1, 577, 768]} : tensor<577x768xf32> -> tensor<1x577x768xf32>
  %6 = IE.Add(%5, %cst_9) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x577x768xf32>, tensor<1x1x768xf32> -> tensor<1x577x768xf32>
  %7 = IE.AffineReshape(%6) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 577, 12, 64]} : tensor<1x577x768xf32> -> tensor<1x577x12x64xf32>
  %8 = IE.Transpose(%7) {order_value = #NHWC} : tensor<1x577x12x64xf32> -> tensor<1x12x64x577xf32>
  %9 = IE.FakeQuantize(%cst_8, %cst_12, %cst_13, %cst_7, %cst_6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<768x768xf32>, tensor<1xf32>, tensor<1xf32>, tensor<768x1xf32>, tensor<768x1xf32> -> tensor<768x768xf32>
  %10 = IE.FullyConnected(%2, %9) : tensor<577x768xf32>, tensor<768x768xf32> -> tensor<577x768xf32>
  %11 = IE.AffineReshape(%10) {dim_mapping = [[0, 1], [2]], shape_value = [1, 577, 768]} : tensor<577x768xf32> -> tensor<1x577x768xf32>
  %12 = IE.Add(%11, %cst_5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x577x768xf32>, tensor<1x1x768xf32> -> tensor<1x577x768xf32>
  %13 = IE.AffineReshape(%12) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 577, 12, 64]} : tensor<1x577x768xf32> -> tensor<1x577x12x64xf32>
  %14 = IE.FakeQuantize(%13, %cst_4, %cst_3, %cst_4, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<1x577x12x64xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<1x577x12x64xf32>
  %15 = IE.Transpose(%14) {order_value = #NHCW} : tensor<1x577x12x64xf32> -> tensor<1x12x577x64xf32>
  %16 = IE.FakeQuantize(%cst_2, %cst_12, %cst_13, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<768x768xf32>, tensor<1xf32>, tensor<1xf32>, tensor<768x1xf32>, tensor<768x1xf32> -> tensor<768x768xf32>
  %17 = IE.FullyConnected(%2, %16) : tensor<577x768xf32>, tensor<768x768xf32> -> tensor<577x768xf32>
  %18 = IE.AffineReshape(%17) {dim_mapping = [[0, 1], [2]], shape_value = [1, 577, 768]} : tensor<577x768xf32> -> tensor<1x577x768xf32>
  %19 = IE.Add(%18, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x577x768xf32>, tensor<1x1x768xf32> -> tensor<1x577x768xf32>
  %20 = IE.AffineReshape(%19) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 577, 12, 64]} : tensor<1x577x768xf32> -> tensor<1x577x12x64xf32>
  %21 = IE.FakeQuantize(%20, %cst_4, %cst_3, %cst_4, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<1x577x12x64xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<1x577x12x64xf32>
  %22 = IE.Transpose(%21) {order_value = #NHCW} : tensor<1x577x12x64xf32> -> tensor<1x12x577x64xf32>
  %23 = IE.Attention(%22, %15, %8, %cst_10) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x12x577x64xf32>, tensor<1x12x577x64xf32>, tensor<1x12x64x577xf32>, tensor<1xf32> -> tensor<1x12x577x64xf32>

  return %root, %23 : tensor<1x577x768xf32>, tensor<1x12x577x64xf32>

// CHECK: [[ROOT_PAD:%.+]] = const.Declare tensor<1x15x768xf32> = dense<0.000000e+00> : tensor<1x15x768xf32>
// CHECK: [[Q_BIAS:%.+]] = const.Declare tensor<1x1x768xf32> = dense<2.000000e+00> : tensor<1x1x2304xf32>, [#const.SubView<[0, 0, 0], [1, 1, 768]>]
// CHECK: [[Q_OUT_HIGH:%.+]] = const.Declare tensor<768x1xf32> = dense<1.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[0, 0], [768, 1]>]
// CHECK: [[Q_OUT_LOW:%.+]] = const.Declare tensor<768x1xf32> = dense<0.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[0, 0], [768, 1]>]
// CHECK: [[Q_WEIGHT:%.+]] = const.Declare tensor<768x768xf32> = dense<1.000000e+00> : tensor<2304x768xf32>, [#const.SubView<[0, 0], [768, 768]>]
// CHECK: [[ACT_HIGH:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1]>]
// CHECK: [[ACT_LOW:%.+]] = const.Declare tensor<1xf32> = dense<0.000000e+00> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1]>]
// CHECK: [[K_BIAS:%.+]] = const.Declare tensor<1x1x768xf32> = dense<2.000000e+00> : tensor<1x1x2304xf32>, [#const.SubView<[0, 0, 768], [1, 1, 768]>]
// CHECK: [[K_OUT_HIGH:%.+]] = const.Declare tensor<768x1xf32> = dense<1.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[768, 0], [768, 1]>]
// CHECK: [[K_OUT_LOW:%.+]] = const.Declare tensor<768x1xf32> = dense<0.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[768, 0], [768, 1]>]
// CHECK: [[K_WEIGHT:%.+]] = const.Declare tensor<768x768xf32> = dense<1.000000e+00> : tensor<2304x768xf32>, [#const.SubView<[768, 0], [768, 768]>]
// CHECK: [[V_BIAS:%.+]] = const.Declare tensor<1x1x768xf32> = dense<2.000000e+00> : tensor<1x1x2304xf32>, [#const.SubView<[0, 0, 1536], [1, 1, 768]>]
// CHECK: [[SCALE:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>
// CHECK: [[V_WEIGHT:%.+]] = const.Declare tensor<768x768xf32> = dense<1.000000e+00> : tensor<2304x768xf32>, [#const.SubView<[1536, 0], [768, 768]>]
// CHECK: [[WEIGHT_LOW:%.+]] = const.Declare tensor<1xf32> = dense<0.000000e+00> : tensor<1x1xf32>, [#const.Reshape<[1]>]
// CHECK: [[WEIGHT_HIGH:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1x1xf32>, [#const.Reshape<[1]>]
// CHECK: [[V_OUT_LOW:%.+]] = const.Declare tensor<768x1xf32> = dense<0.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[1536, 0], [768, 1]>]
// CHECK: [[V_OUT_HIGH:%.+]] = const.Declare tensor<768x1xf32> = dense<1.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[1536, 0], [768, 1]>]
// CHECK: [[ROOT:%.+]] = IE.Add([[INPUT0]], [[INPUT1]])
// CHECK: [[PADDED_ROOT:%.+]] = IE.Concat([[ROOT]], [[ROOT_PAD]])
// CHECK-SAME: tensor<1x577x768xf32>, tensor<1x15x768xf32> -> tensor<1x592x768xf32>
// CHECK: [[PADDED_MVN:%.+]] = IE.MVN6([[PADDED_ROOT]])
// CHECK: [[QKV_RESHAPE_IN:%.+]] = IE.AffineReshape([[PADDED_MVN]])
// CHECK-SAME: shape_value = [592, 768]
// CHECK-SAME: tensor<1x592x768xf32> -> tensor<592x768xf32>
// CHECK: [[V_WEIGHT_FQ:%.+]] = IE.FakeQuantize([[V_WEIGHT]], [[WEIGHT_LOW]], [[WEIGHT_HIGH]], [[V_OUT_LOW]], [[V_OUT_HIGH]])
// CHECK: [[V_FC:%.+]] = IE.FullyConnected([[QKV_RESHAPE_IN]], [[V_WEIGHT_FQ]]) : tensor<592x768xf32>, tensor<768x768xf32> -> tensor<592x768xf32>
// CHECK: [[V_RESHAPE_IN:%.+]] = IE.AffineReshape([[V_FC]])
// CHECK-SAME: shape_value = [1, 592, 768]
// CHECK: [[V_ADD:%.+]] = IE.Add([[V_RESHAPE_IN]], [[V_BIAS]])
// CHECK: [[V_RESHAPE_OUT:%.+]] = IE.AffineReshape([[V_ADD]])
// CHECK-SAME: shape_value = [1, 592, 12, 64]
// CHECK: [[V_TRANSPOSE:%.+]] = IE.Transpose([[V_RESHAPE_OUT]]) {order_value = #NHWC} : tensor<1x592x12x64xf32> -> tensor<1x12x64x592xf32>
// CHECK: [[K_WEIGHT_FQ:%.+]] = IE.FakeQuantize([[K_WEIGHT]], [[WEIGHT_LOW]], [[WEIGHT_HIGH]], [[K_OUT_LOW]], [[K_OUT_HIGH]])
// CHECK: [[K_FC:%.+]] = IE.FullyConnected([[QKV_RESHAPE_IN]], [[K_WEIGHT_FQ]]) : tensor<592x768xf32>, tensor<768x768xf32> -> tensor<592x768xf32>
// CHECK: [[K_RESHAPE_IN:%.+]] = IE.AffineReshape([[K_FC]])
// CHECK-SAME: shape_value = [1, 592, 768]
// CHECK: [[K_ADD:%.+]] = IE.Add([[K_RESHAPE_IN]], [[K_BIAS]])
// CHECK: [[K_RESHAPE_OUT:%.+]] = IE.AffineReshape([[K_ADD]])
// CHECK-SAME: shape_value = [1, 592, 12, 64]
// CHECK: [[K_FQ:%.+]] = IE.FakeQuantize([[K_RESHAPE_OUT]], [[ACT_LOW]], [[ACT_HIGH]], [[ACT_LOW]], [[ACT_HIGH]])
// CHECK: [[K_TRANSPOSE:%.+]] = IE.Transpose([[K_FQ]]) {order_value = #NHCW} : tensor<1x592x12x64xf32> -> tensor<1x12x592x64xf32>
// CHECK: [[Q_WEIGHT_FQ:%.+]] = IE.FakeQuantize([[Q_WEIGHT]], [[WEIGHT_LOW]], [[WEIGHT_HIGH]], [[Q_OUT_LOW]], [[Q_OUT_HIGH]])
// CHECK: [[Q_FC:%.+]] = IE.FullyConnected([[QKV_RESHAPE_IN]], [[Q_WEIGHT_FQ]]) : tensor<592x768xf32>, tensor<768x768xf32> -> tensor<592x768xf32>
// CHECK: [[Q_RESHAPE_IN:%.+]] = IE.AffineReshape([[Q_FC]])
// CHECK-SAME: shape_value = [1, 592, 768]
// CHECK: [[Q_ADD:%.+]] = IE.Add([[Q_RESHAPE_IN]], [[Q_BIAS]])
// CHECK: [[Q_RESHAPE_OUT:%.+]] = IE.AffineReshape([[Q_ADD]])
// CHECK-SAME: shape_value = [1, 592, 12, 64]
// CHECK: [[Q_FQ:%.+]] = IE.FakeQuantize([[Q_RESHAPE_OUT]], [[ACT_LOW]], [[ACT_HIGH]], [[ACT_LOW]], [[ACT_HIGH]])
// CHECK: [[Q_TRANSPOSE:%.+]] = IE.Transpose([[Q_FQ]]) {order_value = #NHCW} : tensor<1x592x12x64xf32> -> tensor<1x12x592x64xf32>
// CHECK: [[ATTENTION:%.+]] = IE.Attention([[Q_TRANSPOSE]], [[K_TRANSPOSE]], [[V_TRANSPOSE]], [[SCALE]])
// CHECK-SAME: padSizeS = 15 : i64
// CHECK-SAME: tensor<1x12x592x64xf32>, tensor<1x12x592x64xf32>, tensor<1x12x64x592xf32>, tensor<1xf32> -> tensor<1x12x592x64xf32>
// CHECK: [[SLICE:%.+]] = IE.Slice [[ATTENTION]] [0, 0, 0, 0] [1, 12, 577, 64] : tensor<1x12x592x64xf32> to tensor<1x12x577x64xf32>
// CHECK: return [[ROOT]], [[SLICE]] : tensor<1x577x768xf32>, tensor<1x12x577x64xf32>
}


// -----

// CHECK-LABEL: @NotExpandAttentionWithNonZeroBias
// CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x1500x512xf32>, [[INPUT1:%.+]]: tensor<1x1500x512xf32>) -> (tensor<1x1500x512xf32>, tensor<1x8x1500x64xf32>)
func.func @NotExpandAttentionWithNonZeroBias(%input0: tensor<1x1500x512xf32>, %input1: tensor<1x1500x512xf32>) -> (tensor<1x1500x512xf32>, tensor<1x8x1500x64xf32>) {
  %wq = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %wk = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %wv = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %attention_scale = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>
  %attention_bias = const.Declare tensor<1x1x1500x1500xf32> = dense<1.000000e+00> : tensor<1x1x1500x1500xf32>
  %root = IE.Add(%input0, %input1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>
  %mvn = IE.MVN6(%root) {axes_value = [2], eps = 9.9999997473787516E-6 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>

  %q_reshape_in = IE.Reshape(%mvn) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %q_fc = IE.FullyConnected(%q_reshape_in, %wq) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %q_reshape_out = IE.Reshape(%q_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %q_affine = IE.AffineReshape(%q_reshape_out) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %q_transpose = IE.Transpose(%q_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %k_reshape_in = IE.Reshape(%mvn) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %k_fc = IE.FullyConnected(%k_reshape_in, %wk) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %k_reshape_out = IE.Reshape(%k_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %k_affine = IE.AffineReshape(%k_reshape_out) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %k_transpose = IE.Transpose(%k_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %v_reshape_in = IE.Reshape(%mvn) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %v_fc = IE.FullyConnected(%v_reshape_in, %wv) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %v_reshape_out = IE.Reshape(%v_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %v_affine = IE.AffineReshape(%v_reshape_out) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %v_transpose = IE.Transpose(%v_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %attention = IE.Attention(%q_transpose, %k_transpose, %v_transpose, %attention_scale, %attention_bias) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 1>} : tensor<1x8x1500x64xf32>, tensor<1x8x1500x64xf32>, tensor<1x8x1500x64xf32>, tensor<1xf32>, tensor<1x1x1500x1500xf32> -> tensor<1x8x1500x64xf32>
  return %root, %attention : tensor<1x1500x512xf32>, tensor<1x8x1500x64xf32>

  // CHECK-NOT: IE.Concat
  // CHECK: [[SCALE:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>
  // CHECK: [[BIAS:%.+]] = const.Declare tensor<1x1x1500x1500xf32> = dense<1.000000e+00> : tensor<1x1x1500x1500xf32>
  // CHECK: [[ROOT:%.+]] = IE.Add([[INPUT0]], [[INPUT1]])
  // CHECK-SAME: tensor<1x1500x512xf32>, tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>
  // CHECK: [[MVN:%.+]] = IE.MVN6([[ROOT]])
  // CHECK: [[Q_RESHAPE_IN:%.+]] = IE.AffineReshape([[MVN]])
  // CHECK-SAME: shape_value = [1500, 512]
  // CHECK: [[Q_FC:%.+]] = IE.FullyConnected([[Q_RESHAPE_IN]],
  // CHECK: [[Q_AFFINE:%.+]] = IE.AffineReshape([[Q_FC]])
  // CHECK-SAME: shape_value = [1, 1500, 8, 64]
  // CHECK: [[Q_TRANSPOSE:%.+]] = IE.Transpose([[Q_AFFINE]])
  // CHECK: [[ATTENTION:%.+]] = IE.Attention([[Q_TRANSPOSE]],
  // CHECK-SAME: [[SCALE]], [[BIAS]])
  // CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 1>
  // CHECK-NOT: IE.Slice
  // CHECK: return [[ROOT]], [[ATTENTION]]
}

// -----

// CHECK-LABEL: @NotExpandAttentionIfNoCommonRootOp
// CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x1500x512xf32>, [[INPUT1:%.+]]: tensor<1x1500x512xf32>) -> (tensor<1x1500x512xf32>, tensor<1x8x1500x64xf32>)
func.func @NotExpandAttentionIfNoCommonRootOp(%input0: tensor<1x1500x512xf32>, %input1: tensor<1x1500x512xf32>) -> (tensor<1x1500x512xf32>, tensor<1x8x1500x64xf32>) {
  %root_bias = const.Declare tensor<1x1x512xf32> = dense<2.000000e+00> : tensor<1x1x512xf32>
  %root_scale = const.Declare tensor<1x1x512xf32> = dense<3.000000e+00> : tensor<1x1x512xf32>
  %wq = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %wk = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %wv = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %attention_scale = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>
  %root = IE.Add(%input0, %input1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>
  %mvn = IE.MVN6(%root) {axes_value = [2], eps = 9.9999997473787516E-6 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>
  %q_root = IE.Add(%mvn, %root_bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>
  %kv_root = IE.Multiply(%mvn, %root_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>

  %q_reshape_in = IE.Reshape(%q_root) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %q_fc = IE.FullyConnected(%q_reshape_in, %wq) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %q_reshape_out = IE.Reshape(%q_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %q_affine = IE.AffineReshape(%q_reshape_out) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %q_transpose = IE.Transpose(%q_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %k_reshape_in = IE.Reshape(%kv_root) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %k_fc = IE.FullyConnected(%k_reshape_in, %wk) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %k_reshape_out = IE.Reshape(%k_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %k_affine = IE.AffineReshape(%k_reshape_out) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %k_transpose = IE.Transpose(%k_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %v_reshape_in = IE.Reshape(%kv_root) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %v_fc = IE.FullyConnected(%v_reshape_in, %wv) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %v_reshape_out = IE.Reshape(%v_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %v_affine = IE.AffineReshape(%v_reshape_out) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %v_transpose = IE.Transpose(%v_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %attention = IE.Attention(%q_transpose, %k_transpose, %v_transpose, %attention_scale) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x8x1500x64xf32>, tensor<1x8x1500x64xf32>, tensor<1x8x1500x64xf32>, tensor<1xf32> -> tensor<1x8x1500x64xf32>
  return %root, %attention : tensor<1x1500x512xf32>, tensor<1x8x1500x64xf32>

  // CHECK-NOT: IE.Concat
  // CHECK: [[ROOT:%.+]] = IE.Add([[INPUT0]], [[INPUT1]])
  // CHECK-SAME: tensor<1x1500x512xf32>, tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>
  // CHECK: [[MVN:%.+]] = IE.MVN6([[ROOT]])
  // CHECK: [[Q_ROOT:%.+]] = IE.Add([[MVN]],
  // CHECK-SAME: tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>
  // CHECK: [[KV_ROOT:%.+]] = IE.Multiply([[MVN]],
  // CHECK-SAME: tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>
  // CHECK: [[Q_RESHAPE_IN:%.+]] = IE.AffineReshape([[Q_ROOT]])
  // CHECK-SAME: shape_value = [1500, 512]
  // CHECK: [[Q_FC:%.+]] = IE.FullyConnected([[Q_RESHAPE_IN]],
  // CHECK: [[Q_AFFINE:%.+]] = IE.AffineReshape([[Q_FC]])
  // CHECK-SAME: shape_value = [1, 1500, 8, 64]
  // CHECK: [[Q_TRANSPOSE:%.+]] = IE.Transpose([[Q_AFFINE]])
  // CHECK: [[K_RESHAPE_IN:%.+]] = IE.AffineReshape([[KV_ROOT]])
  // CHECK-SAME: shape_value = [1500, 512]
  // CHECK: [[K_FC:%.+]] = IE.FullyConnected([[K_RESHAPE_IN]],
  // CHECK: [[K_AFFINE:%.+]] = IE.AffineReshape([[K_FC]])
  // CHECK-SAME: shape_value = [1, 1500, 8, 64]
  // CHECK: [[K_TRANSPOSE:%.+]] = IE.Transpose([[K_AFFINE]])
  // CHECK: [[V_RESHAPE_IN:%.+]] = IE.AffineReshape([[KV_ROOT]])
  // CHECK-SAME: shape_value = [1500, 512]
  // CHECK: [[V_FC:%.+]] = IE.FullyConnected([[V_RESHAPE_IN]],
  // CHECK: [[V_AFFINE:%.+]] = IE.AffineReshape([[V_FC]])
  // CHECK-SAME: shape_value = [1, 1500, 8, 64]
  // CHECK: [[V_TRANSPOSE:%.+]] = IE.Transpose([[V_AFFINE]])
  // CHECK: [[ATTENTION:%.+]] = IE.Attention([[Q_TRANSPOSE]], [[K_TRANSPOSE]], [[V_TRANSPOSE]],
  // CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>
  // CHECK-NOT: IE.Slice
  // CHECK: return [[ROOT]], [[ATTENTION]]
}

// -----

// CHECK-LABEL: @NotExpandAttentionWithNonConstWeights
// CHECK-SAME: ([[INPUT0:%arg[0-9]+]]: tensor<1x1500x512xf32>, [[INPUT1:%arg[0-9]+]]: tensor<1x1500x512xf32>, [[WQ:%arg[0-9]+]]: tensor<512x512xf32>)
func.func @NotExpandAttentionWithNonConstWeights(%input0: tensor<1x1500x512xf32>, %input1: tensor<1x1500x512xf32>, %wq: tensor<512x512xf32>) -> (tensor<1x1500x512xf32>, tensor<1x8x1500x64xf32>) {
  %wk = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %wv = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %attention_scale = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>
  %root = IE.Add(%input0, %input1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>
  %mvn = IE.MVN6(%root) {axes_value = [2], eps = 9.9999997473787516E-6 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>

  %q_reshape_in = IE.Reshape(%mvn) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %q_fc = IE.FullyConnected(%q_reshape_in, %wq) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %q_reshape_out = IE.Reshape(%q_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %q_affine = IE.AffineReshape(%q_reshape_out) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %q_transpose = IE.Transpose(%q_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %k_reshape_in = IE.Reshape(%mvn) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %k_fc = IE.FullyConnected(%k_reshape_in, %wk) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %k_reshape_out = IE.Reshape(%k_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %k_affine = IE.AffineReshape(%k_reshape_out) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %k_transpose = IE.Transpose(%k_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %v_reshape_in = IE.Reshape(%mvn) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %v_fc = IE.FullyConnected(%v_reshape_in, %wv) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %v_reshape_out = IE.Reshape(%v_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %v_affine = IE.AffineReshape(%v_reshape_out) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %v_transpose = IE.Transpose(%v_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %attention = IE.Attention(%q_transpose, %k_transpose, %v_transpose, %attention_scale) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x8x1500x64xf32>, tensor<1x8x1500x64xf32>, tensor<1x8x1500x64xf32>, tensor<1xf32> -> tensor<1x8x1500x64xf32>
  return %root, %attention : tensor<1x1500x512xf32>, tensor<1x8x1500x64xf32>

  // CHECK-NOT: IE.Concat
  // CHECK: [[ROOT:%.+]] = IE.Add([[INPUT0]], [[INPUT1]])
  // CHECK-SAME: tensor<1x1500x512xf32>, tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>
  // CHECK: [[MVN:%.+]] = IE.MVN6([[ROOT]])
  // CHECK: [[Q_RESHAPE_IN:%.+]] = IE.AffineReshape([[MVN]])
  // CHECK-SAME: shape_value = [1500, 512]
  // CHECK: [[Q_FC:%.+]] = IE.FullyConnected([[Q_RESHAPE_IN]], [[WQ]])
  // CHECK: [[Q_AFFINE:%.+]] = IE.AffineReshape([[Q_FC]])
  // CHECK-SAME: shape_value = [1, 1500, 8, 64]
  // CHECK: [[Q_TRANSPOSE:%.+]] = IE.Transpose([[Q_AFFINE]])
  // CHECK: [[ATTENTION:%.+]] = IE.Attention([[Q_TRANSPOSE]],
  // CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>
  // CHECK-NOT: IE.Slice
  // CHECK: return [[ROOT]], [[ATTENTION]]
}

// -----

// CHECK-LABEL: @NotExpandAttentionIfRootHasOneUse
// CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x1500x512xf32>, [[INPUT1:%.+]]: tensor<1x1500x512xf32>) -> tensor<1x1500x8x64xf32>
func.func @NotExpandAttentionIfRootHasOneUse(%input0: tensor<1x1500x512xf32>, %input1: tensor<1x1500x512xf32>) -> tensor<1x1500x8x64xf32> {
  %scale_ln = const.Declare tensor<1x1x512xf32> = dense<2.000000e+00> : tensor<1x1x512xf32>
  %bias_ln = const.Declare tensor<1x1x512xf32> = dense<3.000000e+00> : tensor<1x1x512xf32>
  %wq = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %wk = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %wv = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %wq_input_low = const.Declare tensor<1x1xf32> = dense<0.000000e+00> : tensor<1x1xf32>
  %wq_input_high = const.Declare tensor<1x1xf32> = dense<1.000000e+00> : tensor<1x1xf32>
  %wq_output_low = const.Declare tensor<512x1xf32> = dense<0.000000e+00> : tensor<512x1xf32>
  %wq_output_high = const.Declare tensor<512x1xf32> = dense<1.000000e+00> : tensor<512x1xf32>
  %v_bias = const.Declare tensor<1x1x512xf32> = dense<0.000000e+00> : tensor<1x1x512xf32>
  %q_scale = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
  %q_bias = const.Declare tensor<1x1x512xf32> = dense<0.000000e+00> : tensor<1x1x512xf32>
  %attention_scale = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>

  %root = IE.Add(%input0, %input1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>
  %mvn = IE.MVN6(%root) {axes_value = [2], eps = 9.9999997473787516E-6 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>

  %ln_mul = IE.Multiply(%mvn, %scale_ln) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>
  %ln_add = IE.Add(%ln_mul, %bias_ln) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>

  %wq_fq = IE.FakeQuantize(%wq, %wq_input_low, %wq_input_high, %wq_output_low, %wq_output_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<512x512xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<512x1xf32>, tensor<512x1xf32> -> tensor<512x512xf32>
  %q_reshape_in = IE.Reshape(%ln_add) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %q_fc = IE.FullyConnected(%q_reshape_in, %wq_fq) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %q_reshape_out = IE.Reshape(%q_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %q_mul = IE.Multiply(%q_reshape_out, %q_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x1xf32> -> tensor<1x1500x512xf32>
  %q_add = IE.Add(%q_mul, %q_bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>
  %q_affine = IE.AffineReshape(%q_add) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %q_transpose = IE.Transpose(%q_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %k_reshape_in = IE.Reshape(%ln_add) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %k_fc = IE.FullyConnected(%k_reshape_in, %wk) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %k_reshape_out = IE.Reshape(%k_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %k_affine = IE.AffineReshape(%k_reshape_out) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %k_transpose = IE.Transpose(%k_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %v_reshape_in = IE.Reshape(%ln_add) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  %v_fc = IE.FullyConnected(%v_reshape_in, %wv) : tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  %v_reshape_out = IE.Reshape(%v_fc) {shape_value = [1, 1500, 512]} : tensor<1500x512xf32> -> tensor<1x1500x512xf32>
  %v_add = IE.Add(%v_reshape_out, %v_bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1500x512xf32>, tensor<1x1x512xf32> -> tensor<1x1500x512xf32>
  %v_affine = IE.AffineReshape(%v_add) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1500, 8, 64]} : tensor<1x1500x512xf32> -> tensor<1x1500x8x64xf32>
  %v_transpose = IE.Transpose(%v_affine) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  %attention = IE.Attention(%q_transpose, %k_transpose, %v_transpose, %attention_scale) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x8x1500x64xf32>, tensor<1x8x1500x64xf32>, tensor<1x8x1500x64xf32>, tensor<1xf32> -> tensor<1x8x1500x64xf32>
  %out = IE.Transpose(%attention) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x8x1500x64xf32> -> tensor<1x1500x8x64xf32>
  return %out : tensor<1x1500x8x64xf32>

  // CHECK-NOT: IE.Concat
  // CHECK: [[ROOT:%.+]] = IE.Add([[INPUT0]], [[INPUT1]])
  // CHECK-SAME: tensor<1x1500x512xf32>, tensor<1x1500x512xf32> -> tensor<1x1500x512xf32>
  // CHECK: [[MVN:%.+]] = IE.MVN6([[ROOT]])
  // CHECK: [[MUL:%.+]] = IE.Multiply([[MVN]]
  // CHECK: [[ADD:%.+]] = IE.Add([[MUL]]

  // CHECK: [[Q_RESHAPE_IN:%.+]] = IE.AffineReshape([[ADD]])
  // CHECK-SAME: shape_value = [1500, 512]
  // CHECK-SAME: tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  // CHECK: [[Q_FC:%.+]] = IE.FullyConnected([[Q_RESHAPE_IN]],
  // CHECK-SAME: tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  // CHECK: [[Q_AFFINE:%.+]] = IE.AffineReshape([[Q_FC]])
  // CHECK-SAME: shape_value = [1, 1500, 8, 64]
  // CHECK-SAME: tensor<1500x512xf32> -> tensor<1x1500x8x64xf32>
  // CHECK: [[Q_TRANSPOSE:%.+]] = IE.Transpose([[Q_AFFINE]])
  // CHECK-SAME: tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  // CHECK: [[K_RESHAPE_IN:%.+]] = IE.AffineReshape([[ADD]])
  // CHECK-SAME: shape_value = [1500, 512]
  // CHECK-SAME: tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  // CHECK: [[K_FC:%.+]] = IE.FullyConnected([[K_RESHAPE_IN]],
  // CHECK-SAME: tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  // CHECK: [[K_AFFINE:%.+]] = IE.AffineReshape([[K_FC]])
  // CHECK-SAME: shape_value = [1, 1500, 8, 64]
  // CHECK-SAME: tensor<1500x512xf32> -> tensor<1x1500x8x64xf32>
  // CHECK: [[K_TRANSPOSE:%.+]] = IE.Transpose([[K_AFFINE]])
  // CHECK-SAME: tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  // CHECK: [[V_RESHAPE_IN:%.+]] = IE.AffineReshape([[ADD]])
  // CHECK-SAME: shape_value = [1500, 512]
  // CHECK-SAME: tensor<1x1500x512xf32> -> tensor<1500x512xf32>
  // CHECK: [[V_FC:%.+]] = IE.FullyConnected([[V_RESHAPE_IN]],
  // CHECK-SAME: tensor<1500x512xf32>, tensor<512x512xf32> -> tensor<1500x512xf32>
  // CHECK: [[V_AFFINE:%.+]] = IE.AffineReshape([[V_FC]])
  // CHECK-SAME: shape_value = [1, 1500, 8, 64]
  // CHECK-SAME: tensor<1500x512xf32> -> tensor<1x1500x8x64xf32>
  // CHECK: [[V_TRANSPOSE:%.+]] = IE.Transpose([[V_AFFINE]])
  // CHECK-SAME: tensor<1x1500x8x64xf32> -> tensor<1x8x1500x64xf32>

  // CHECK: [[ATTENTION:%.+]] = IE.Attention([[Q_TRANSPOSE]], [[K_TRANSPOSE]], [[V_TRANSPOSE]],
  // CHECK-SAME: tensor<1x8x1500x64xf32>, tensor<1x8x1500x64xf32>, tensor<1x8x1500x64xf32>, tensor<1xf32> -> tensor<1x8x1500x64xf32>
  // CHECK: [[OUT:%.+]] = IE.Transpose([[ATTENTION]])
  // CHECK-SAME: tensor<1x8x1500x64xf32> -> tensor<1x1500x8x64xf32>
  // CHECK-NOT: IE.Slice
  // CHECK: return [[OUT]]
}
