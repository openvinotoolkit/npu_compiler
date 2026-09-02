//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --adjust-qkv-transpose-for-unroll-sdpa %s | FileCheck %s
// REQUIRES: platform-NPU5010

#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d0, d3, d1, d4)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @AdjustQKVTransposeForUnrollSDPA
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x577x768xf32>)
func.func @AdjustQKVTransposeForUnrollSDPA(%input: tensor<1x577x768xf32>) -> tensor<1x12x577x64xf32> {
  %norm_mul = const.Declare tensor<1x1x768xf32> = dense<1.0> : tensor<1x1x768xf32>
  %norm_add = const.Declare tensor<1x1x768xf32> = dense<3.0> : tensor<1x1x768xf32>
  %input_fq_low = const.Declare tensor<1x1x1xf32> = dense<0.0> : tensor<1x1x1xf32>
  %input_fq_high = const.Declare tensor<1x1x1xf32> = dense<1.0> : tensor<1x1x1xf32>
  %3 = const.Declare tensor<2304x768xf32> = dense<1.0> : tensor<2304x768xf32>
  %cst_44 = const.Declare tensor<1x1x2304xf32> = dense<2.0> : tensor<1x1x2304xf32>
  %cst_358 = const.Declare tensor<1x1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1x1xf32>
  %cst_359 = const.Declare tensor<1x1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1x1xf32>
  %cst_360 = const.Declare tensor<1x1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1x1xf32>
  %cst_361 = const.Declare tensor<1x1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1x1xf32>
  %cst_62 = const.Declare tensor<1xsi64> = dense<0> : tensor<1xsi64>
  %cst_63 = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
  %cst_64 = const.Declare tensor<1xsi64> = dense<2> : tensor<1xsi64>
  %cst_362 = const.Declare tensor<1xf32> = dense<1.0> : tensor<1xf32>

  %508 = IE.Multiply(%input, %norm_mul) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x577x768xf32>, tensor<1x1x768xf32> -> tensor<1x577x768xf32>
  %509 = IE.Add(%508, %norm_add) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x577x768xf32>, tensor<1x1x768xf32> -> tensor<1x577x768xf32>
  %510 = IE.FakeQuantize(%509, %input_fq_low, %input_fq_high, %input_fq_low, %input_fq_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x577x768xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x577x768xf32>
  %511 = IE.AffineReshape(%510) {dim_mapping = [[0], [0], [1]], shape_value = [577, 768]} : tensor<1x577x768xf32> -> tensor<577x768xf32>
  %512 = IE.FullyConnected(%511, %3) : tensor<577x768xf32>, tensor<2304x768xf32> -> tensor<577x2304xf32>
  %513 = IE.AffineReshape(%512) {dim_mapping = [[0, 1], [2]], shape_value = [1, 577, 2304]} : tensor<577x2304xf32> -> tensor<1x577x2304xf32>
  %514 = IE.Add(%513, %cst_44) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x577x2304xf32>, tensor<1x1x2304xf32> -> tensor<1x577x2304xf32>
  %515 = IE.AffineReshape(%514) {dim_mapping = [[0], [1], [2, 3, 4]], shape_value = [1, 577, 3, 12, 64]} : tensor<1x577x2304xf32> -> tensor<1x577x3x12x64xf32>
  %516 = IE.Transpose(%515) {order_value = #map1} : tensor<1x577x3x12x64xf32> -> tensor<3x1x12x577x64xf32>
  %517 = IE.FakeQuantize(%516, %cst_358, %cst_359, %cst_360, %cst_361) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<3x1x12x577x64xf32>, tensor<1x1x1x1x1xf32>, tensor<1x1x1x1x1xf32>, tensor<1x1x1x1x1xf32>, tensor<1x1x1x1x1xf32> -> tensor<3x1x12x577x64xf32>
  %518 = IE.Gather(%517, %cst_62) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 0 : i64} : tensor<3x1x12x577x64xf32>, tensor<1xsi64> -> tensor<1x12x577x64xf32>
  %519 = IE.Gather(%517, %cst_63) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 0 : i64} : tensor<3x1x12x577x64xf32>, tensor<1xsi64> -> tensor<1x12x577x64xf32>
  %520 = IE.Gather(%516, %cst_64) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 0 : i64} : tensor<3x1x12x577x64xf32>, tensor<1xsi64> -> tensor<1x12x577x64xf32>
  %521 = IE.Transpose(%520) {order_value = #NCWH} : tensor<1x12x577x64xf32> -> tensor<1x12x64x577xf32>
  %522 = IE.Attention(%518, %519, %521, %cst_362) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x12x577x64xf32>, tensor<1x12x577x64xf32>, tensor<1x12x64x577xf32>, tensor<1xf32> -> tensor<1x12x577x64xf32>
  return %522 : tensor<1x12x577x64xf32>

  // CHECK-DAG: [[Q_WEIGHT:%.+]] = const.Declare tensor<768x768xf32> = dense<1.000000e+00> : tensor<2304x768xf32>, [#const.SubView<[0, 0], [768, 768]>]
  // CHECK-DAG: [[K_WEIGHT:%.+]] = const.Declare tensor<768x768xf32> = dense<1.000000e+00> : tensor<2304x768xf32>, [#const.SubView<[768, 0], [768, 768]>]
  // CHECK-DAG: [[V_WEIGHT:%.+]] = const.Declare tensor<768x768xf32> = dense<1.000000e+00> : tensor<2304x768xf32>, [#const.SubView<[1536, 0], [768, 768]>]
  // CHECK-DAG: [[Q_BIAS:%.+]] = const.Declare tensor<1x1x768xf32> = dense<2.000000e+00> : tensor<1x1x2304xf32>, [#const.SubView<[0, 0, 0], [1, 1, 768]>]
  // CHECK-DAG: [[K_BIAS:%.+]] = const.Declare tensor<1x1x768xf32> = dense<2.000000e+00> : tensor<1x1x2304xf32>, [#const.SubView<[0, 0, 768], [1, 1, 768]>]
  // CHECK-DAG: [[V_BIAS:%.+]] = const.Declare tensor<1x1x768xf32> = dense<2.000000e+00> : tensor<1x1x2304xf32>, [#const.SubView<[0, 0, 1536], [1, 1, 768]>]
  // CHECK-DAG: [[INPUT_ADD_CST:%.+]] = const.Declare tensor<1x1x768xf32> = dense<3.000000e+00> : tensor<1x1x768xf32>
  // CHECK-DAG: [[INPUT_FQ_LOW:%.+]] = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
  // CHECK-DAG: [[INPUT_FQ_HIGH:%.+]] = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
  // CHECK-DAG: [[FQ_LOW:%.+]] = const.Declare tensor<1xf32> = dense<0.000000e+00> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1]>]
  // CHECK-DAG: [[FQ_HIGH:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1]>]
  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>

  // CHECK: [[INPUT_ADD:%.+]] = IE.Add([[INPUT]], [[INPUT_ADD_CST]])
  // CHECK: [[INPUT_FQ:%.+]] = IE.FakeQuantize([[INPUT_ADD]], [[INPUT_FQ_LOW]], [[INPUT_FQ_HIGH]], [[INPUT_FQ_LOW]], [[INPUT_FQ_HIGH]])
  // CHECK: [[FC_INPUT:%.+]] = IE.AffineReshape([[INPUT_FQ]]) {dim_mapping = {{\[\[}}0], [0], [1]], shape_value = [577, 768]} : tensor<1x577x768xf32> -> tensor<577x768xf32>

  // CHECK: [[Q_FC:%.+]] = IE.FullyConnected([[FC_INPUT]], [[Q_WEIGHT]]) : tensor<577x768xf32>, tensor<768x768xf32> -> tensor<577x768xf32>
  // CHECK: [[Q_RESHAPE_FC:%.+]] = IE.AffineReshape([[Q_FC]]) {dim_mapping = {{\[\[}}0, 1], [2]], shape_value = [1, 577, 768]} : tensor<577x768xf32> -> tensor<1x577x768xf32>
  // CHECK: [[Q_ADD:%.+]] = IE.Add([[Q_RESHAPE_FC]], [[Q_BIAS]])
  // CHECK: [[Q_RESHAPE:%.+]] = IE.AffineReshape([[Q_ADD]]) {dim_mapping = {{\[\[}}0], [1], [2, 3]], shape_value = [1, 577, 12, 64]} : tensor<1x577x768xf32> -> tensor<1x577x12x64xf32>
  // CHECK: [[Q_FQ:%.+]] = IE.FakeQuantize([[Q_RESHAPE]], [[FQ_LOW]], [[FQ_HIGH]], [[FQ_LOW]], [[FQ_HIGH]])
  // CHECK: [[Q_TRANSPOSE:%.+]] = IE.Transpose([[Q_FQ]]) {order_value = #NHCW} : tensor<1x577x12x64xf32> -> tensor<1x12x577x64xf32>

  // CHECK: [[K_FC:%.+]] = IE.FullyConnected([[FC_INPUT]], [[K_WEIGHT]]) : tensor<577x768xf32>, tensor<768x768xf32> -> tensor<577x768xf32>
  // CHECK: [[K_RESHAPE_FC:%.+]] = IE.AffineReshape([[K_FC]]) {dim_mapping = {{\[\[}}0, 1], [2]], shape_value = [1, 577, 768]} : tensor<577x768xf32> -> tensor<1x577x768xf32>
  // CHECK: [[K_ADD:%.+]] = IE.Add([[K_RESHAPE_FC]], [[K_BIAS]])
  // CHECK: [[K_RESHAPE:%.+]] = IE.AffineReshape([[K_ADD]]) {dim_mapping = {{\[\[}}0], [1], [2, 3]], shape_value = [1, 577, 12, 64]} : tensor<1x577x768xf32> -> tensor<1x577x12x64xf32>
  // CHECK: [[K_FQ:%.+]] = IE.FakeQuantize([[K_RESHAPE]], [[FQ_LOW]], [[FQ_HIGH]], [[FQ_LOW]], [[FQ_HIGH]])
  // CHECK: [[K_TRANSPOSE:%.+]] = IE.Transpose([[K_FQ]]) {order_value = #NHCW} : tensor<1x577x12x64xf32> -> tensor<1x12x577x64xf32>

  // CHECK: [[V_FC:%.+]] = IE.FullyConnected([[FC_INPUT]], [[V_WEIGHT]]) : tensor<577x768xf32>, tensor<768x768xf32> -> tensor<577x768xf32>
  // CHECK: [[V_RESHAPE_FC:%.+]] = IE.AffineReshape([[V_FC]]) {dim_mapping = {{\[\[}}0, 1], [2]], shape_value = [1, 577, 768]} : tensor<577x768xf32> -> tensor<1x577x768xf32>
  // CHECK: [[V_ADD:%.+]] = IE.Add([[V_RESHAPE_FC]], [[V_BIAS]])
  // CHECK: [[V_RESHAPE:%.+]] = IE.AffineReshape([[V_ADD]]) {dim_mapping = {{\[\[}}0], [1], [2, 3]], shape_value = [1, 577, 12, 64]} : tensor<1x577x768xf32> -> tensor<1x577x12x64xf32>
  // CHECK: [[V_TRANSPOSE:%.+]] = IE.Transpose([[V_RESHAPE]]) {order_value = #NHCW} : tensor<1x577x12x64xf32> -> tensor<1x12x577x64xf32>
  // CHECK: [[VT:%.+]] = IE.Transpose([[V_TRANSPOSE]]) {order_value = #NCWH} : tensor<1x12x577x64xf32> -> tensor<1x12x64x577xf32>

  // CHECK-NOT: IE.Gather
  // CHECK: [[ATTENTION:%.+]] = IE.Attention([[Q_TRANSPOSE]], [[K_TRANSPOSE]], [[VT]], [[SCALE:%.+]])
  // CHECK: return [[ATTENTION]]
}

// -----

#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d0, d3, d1, d4)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @AdjustQKVTransposeForUnrollSDPAWithFQWeights
// CHECK-SAME: ([[INPUT:%.+]]: tensor<577x768xf32>)
func.func @AdjustQKVTransposeForUnrollSDPAWithFQWeights(%input: tensor<577x768xf32>) -> tensor<1x12x577x64xf32> {
  %weights = const.Declare tensor<2304x768xf32> = dense<1.0> : tensor<2304x768xf32>
  %weight_input_low = const.Declare tensor<1x1xf32> = dense<0.0> : tensor<1x1xf32>
  %weight_input_high = const.Declare tensor<1x1xf32> = dense<1.0> : tensor<1x1xf32>
  %weight_output_low = const.Declare tensor<2304x1xf32> = dense<0.0> : tensor<2304x1xf32>
  %weight_output_high = const.Declare tensor<2304x1xf32> = dense<1.0> : tensor<2304x1xf32>
  %bias = const.Declare tensor<1x1x2304xf32> = dense<2.0> : tensor<1x1x2304xf32>
  %fq_low = const.Declare tensor<1x1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1x1xf32>
  %fq_high = const.Declare tensor<1x1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1x1xf32>
  %q_idx = const.Declare tensor<1xsi64> = dense<0> : tensor<1xsi64>
  %k_idx = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
  %v_idx = const.Declare tensor<1xsi64> = dense<2> : tensor<1xsi64>
  %scale = const.Declare tensor<1xf32> = dense<1.0> : tensor<1xf32>

  %weight_fq = IE.FakeQuantize(%weights, %weight_input_low, %weight_input_high, %weight_output_low, %weight_output_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<2304x768xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<2304x1xf32>, tensor<2304x1xf32> -> tensor<2304x768xf32>
  %fc = IE.FullyConnected(%input, %weight_fq) : tensor<577x768xf32>, tensor<2304x768xf32> -> tensor<577x2304xf32>
  %reshape_fc = IE.AffineReshape(%fc) {dim_mapping = [[0, 1], [2]], shape_value = [1, 577, 2304]} : tensor<577x2304xf32> -> tensor<1x577x2304xf32>
  %add = IE.Add(%reshape_fc, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x577x2304xf32>, tensor<1x1x2304xf32> -> tensor<1x577x2304xf32>
  %reshape = IE.AffineReshape(%add) {dim_mapping = [[0], [1], [2, 3, 4]], shape_value = [1, 577, 3, 12, 64]} : tensor<1x577x2304xf32> -> tensor<1x577x3x12x64xf32>
  %transpose = IE.Transpose(%reshape) {order_value = #map1} : tensor<1x577x3x12x64xf32> -> tensor<3x1x12x577x64xf32>
  %fq = IE.FakeQuantize(%transpose, %fq_low, %fq_high, %fq_low, %fq_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<3x1x12x577x64xf32>, tensor<1x1x1x1x1xf32>, tensor<1x1x1x1x1xf32>, tensor<1x1x1x1x1xf32>, tensor<1x1x1x1x1xf32> -> tensor<3x1x12x577x64xf32>
  %q = IE.Gather(%fq, %q_idx) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 0 : i64} : tensor<3x1x12x577x64xf32>, tensor<1xsi64> -> tensor<1x12x577x64xf32>
  %k = IE.Gather(%fq, %k_idx) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 0 : i64} : tensor<3x1x12x577x64xf32>, tensor<1xsi64> -> tensor<1x12x577x64xf32>
  %v = IE.Gather(%transpose, %v_idx) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 0 : i64} : tensor<3x1x12x577x64xf32>, tensor<1xsi64> -> tensor<1x12x577x64xf32>
  %vt = IE.Transpose(%v) {order_value = #NCWH} : tensor<1x12x577x64xf32> -> tensor<1x12x64x577xf32>
  %attention = IE.Attention(%q, %k, %vt, %scale) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x12x577x64xf32>, tensor<1x12x577x64xf32>, tensor<1x12x64x577xf32>, tensor<1xf32> -> tensor<1x12x577x64xf32>
  return %attention : tensor<1x12x577x64xf32>

  // CHECK-DAG: [[Q_WEIGHT:%.+]] = const.Declare tensor<768x768xf32> = dense<1.000000e+00> : tensor<2304x768xf32>, [#const.SubView<[0, 0], [768, 768]>]
  // CHECK-DAG: [[K_WEIGHT:%.+]] = const.Declare tensor<768x768xf32> = dense<1.000000e+00> : tensor<2304x768xf32>, [#const.SubView<[768, 0], [768, 768]>]
  // CHECK-DAG: [[V_WEIGHT:%.+]] = const.Declare tensor<768x768xf32> = dense<1.000000e+00> : tensor<2304x768xf32>, [#const.SubView<[1536, 0], [768, 768]>]
  // CHECK-DAG: [[Q_BIAS:%.+]] = const.Declare tensor<1x1x768xf32> = dense<2.000000e+00> : tensor<1x1x2304xf32>, [#const.SubView<[0, 0, 0], [1, 1, 768]>]
  // CHECK-DAG: [[K_BIAS:%.+]] = const.Declare tensor<1x1x768xf32> = dense<2.000000e+00> : tensor<1x1x2304xf32>, [#const.SubView<[0, 0, 768], [1, 1, 768]>]
  // CHECK-DAG: [[V_BIAS:%.+]] = const.Declare tensor<1x1x768xf32> = dense<2.000000e+00> : tensor<1x1x2304xf32>, [#const.SubView<[0, 0, 1536], [1, 1, 768]>]
  // CHECK-DAG: [[Q_OUTPUT_LOW:%.+]] = const.Declare tensor<768x1xf32> = dense<0.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[0, 0], [768, 1]>]
  // CHECK-DAG: [[Q_OUTPUT_HIGH:%.+]] = const.Declare tensor<768x1xf32> = dense<1.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[0, 0], [768, 1]>]
  // CHECK-DAG: [[K_OUTPUT_LOW:%.+]] = const.Declare tensor<768x1xf32> = dense<0.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[768, 0], [768, 1]>]
  // CHECK-DAG: [[K_OUTPUT_HIGH:%.+]] = const.Declare tensor<768x1xf32> = dense<1.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[768, 0], [768, 1]>]
  // CHECK-DAG: [[V_OUTPUT_LOW:%.+]] = const.Declare tensor<768x1xf32> = dense<0.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[1536, 0], [768, 1]>]
  // CHECK-DAG: [[V_OUTPUT_HIGH:%.+]] = const.Declare tensor<768x1xf32> = dense<1.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[1536, 0], [768, 1]>]
  // CHECK-DAG: [[WEIGHT_INPUT_LOW:%.+]] = const.Declare tensor<1xf32> = dense<0.000000e+00> : tensor<1x1xf32>, [#const.Reshape<[1]>]
  // CHECK-DAG: [[WEIGHT_INPUT_HIGH:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1x1xf32>, [#const.Reshape<[1]>]
  // CHECK-DAG: [[FQ_LOW:%.+]] = const.Declare tensor<1xf32> = dense<0.000000e+00> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1]>]
  // CHECK-DAG: [[FQ_HIGH:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1]>]
  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>

  // CHECK: [[Q_WEIGHT_FQ:%.+]] = IE.FakeQuantize([[Q_WEIGHT]], [[WEIGHT_INPUT_LOW]], [[WEIGHT_INPUT_HIGH]], [[Q_OUTPUT_LOW]], [[Q_OUTPUT_HIGH]])
  // CHECK: [[Q_FC:%.+]] = IE.FullyConnected([[INPUT]], [[Q_WEIGHT_FQ]]) : tensor<577x768xf32>, tensor<768x768xf32> -> tensor<577x768xf32>
  // CHECK: [[Q_RESHAPE_FC:%.+]] = IE.AffineReshape([[Q_FC]]) {dim_mapping = {{\[\[}}0, 1], [2]], shape_value = [1, 577, 768]} : tensor<577x768xf32> -> tensor<1x577x768xf32>
  // CHECK: [[Q_ADD:%.+]] = IE.Add([[Q_RESHAPE_FC]], [[Q_BIAS]])
  // CHECK: [[Q_RESHAPE:%.+]] = IE.AffineReshape([[Q_ADD]]) {dim_mapping = {{\[\[}}0], [1], [2, 3]], shape_value = [1, 577, 12, 64]} : tensor<1x577x768xf32> -> tensor<1x577x12x64xf32>
  // CHECK: [[Q_FQ:%.+]] = IE.FakeQuantize([[Q_RESHAPE]], [[FQ_LOW]], [[FQ_HIGH]], [[FQ_LOW]], [[FQ_HIGH]])
  // CHECK: [[Q_TRANSPOSE:%.+]] = IE.Transpose([[Q_FQ]]) {order_value = #NHCW} : tensor<1x577x12x64xf32> -> tensor<1x12x577x64xf32>

  // CHECK: [[K_WEIGHT_FQ:%.+]] = IE.FakeQuantize([[K_WEIGHT]], [[WEIGHT_INPUT_LOW]], [[WEIGHT_INPUT_HIGH]], [[K_OUTPUT_LOW]], [[K_OUTPUT_HIGH]])
  // CHECK: [[K_FC:%.+]] = IE.FullyConnected([[INPUT]], [[K_WEIGHT_FQ]]) : tensor<577x768xf32>, tensor<768x768xf32> -> tensor<577x768xf32>
  // CHECK: [[K_RESHAPE_FC:%.+]] = IE.AffineReshape([[K_FC]]) {dim_mapping = {{\[\[}}0, 1], [2]], shape_value = [1, 577, 768]} : tensor<577x768xf32> -> tensor<1x577x768xf32>
  // CHECK: [[K_ADD:%.+]] = IE.Add([[K_RESHAPE_FC]], [[K_BIAS]])
  // CHECK: [[K_RESHAPE:%.+]] = IE.AffineReshape([[K_ADD]]) {dim_mapping = {{\[\[}}0], [1], [2, 3]], shape_value = [1, 577, 12, 64]} : tensor<1x577x768xf32> -> tensor<1x577x12x64xf32>
  // CHECK: [[K_FQ:%.+]] = IE.FakeQuantize([[K_RESHAPE]], [[FQ_LOW]], [[FQ_HIGH]], [[FQ_LOW]], [[FQ_HIGH]])
  // CHECK: [[K_TRANSPOSE:%.+]] = IE.Transpose([[K_FQ]]) {order_value = #NHCW} : tensor<1x577x12x64xf32> -> tensor<1x12x577x64xf32>

  // CHECK: [[V_WEIGHT_FQ:%.+]] = IE.FakeQuantize([[V_WEIGHT]], [[WEIGHT_INPUT_LOW]], [[WEIGHT_INPUT_HIGH]], [[V_OUTPUT_LOW]], [[V_OUTPUT_HIGH]])
  // CHECK: [[V_FC:%.+]] = IE.FullyConnected([[INPUT]], [[V_WEIGHT_FQ]]) : tensor<577x768xf32>, tensor<768x768xf32> -> tensor<577x768xf32>
  // CHECK: [[V_RESHAPE_FC:%.+]] = IE.AffineReshape([[V_FC]]) {dim_mapping = {{\[\[}}0, 1], [2]], shape_value = [1, 577, 768]} : tensor<577x768xf32> -> tensor<1x577x768xf32>
  // CHECK: [[V_ADD:%.+]] = IE.Add([[V_RESHAPE_FC]], [[V_BIAS]])
  // CHECK: [[V_RESHAPE:%.+]] = IE.AffineReshape([[V_ADD]]) {dim_mapping = {{\[\[}}0], [1], [2, 3]], shape_value = [1, 577, 12, 64]} : tensor<1x577x768xf32> -> tensor<1x577x12x64xf32>
  // CHECK: [[V_TRANSPOSE:%.+]] = IE.Transpose([[V_RESHAPE]]) {order_value = #NHCW} : tensor<1x577x12x64xf32> -> tensor<1x12x577x64xf32>
  // CHECK: [[VT:%.+]] = IE.Transpose([[V_TRANSPOSE]]) {order_value = #NCWH} : tensor<1x12x577x64xf32> -> tensor<1x12x64x577xf32>
  // CHECK-NOT: IE.Gather
  // CHECK: [[ATTENTION:%.+]] = IE.Attention([[Q_TRANSPOSE]], [[K_TRANSPOSE]], [[VT]], [[SCALE]])
  // CHECK: return [[ATTENTION]]
}

// -----

#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d0, d3, d1, d4)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @DontAdjustQKVTransposeWithNonConstWeights
// CHECK-SAME: ([[INPUT:%.+]]: tensor<5x8xf32>, [[WEIGHTS:%.+]]: tensor<24x8xf32>)
func.func @DontAdjustQKVTransposeWithNonConstWeights(%input: tensor<5x8xf32>, %weights: tensor<24x8xf32>) -> tensor<1x2x5x4xf32> {
  %bias = const.Declare tensor<1x1x24xf32> = dense<2.0> : tensor<1x1x24xf32>
  %fq_low = const.Declare tensor<1x1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1x1xf32>
  %fq_high = const.Declare tensor<1x1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1x1xf32>
  %q_idx = const.Declare tensor<1xsi64> = dense<0> : tensor<1xsi64>
  %k_idx = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
  %v_idx = const.Declare tensor<1xsi64> = dense<2> : tensor<1xsi64>
  %scale = const.Declare tensor<1xf32> = dense<1.0> : tensor<1xf32>

  %fc = IE.FullyConnected(%input, %weights) : tensor<5x8xf32>, tensor<24x8xf32> -> tensor<5x24xf32>
  %reshape_fc = IE.AffineReshape(%fc) {dim_mapping = [[0, 1], [2]], shape_value = [1, 5, 24]} : tensor<5x24xf32> -> tensor<1x5x24xf32>
  %add = IE.Add(%reshape_fc, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x5x24xf32>, tensor<1x1x24xf32> -> tensor<1x5x24xf32>
  %reshape = IE.AffineReshape(%add) {dim_mapping = [[0], [1], [2, 3, 4]], shape_value = [1, 5, 3, 2, 4]} : tensor<1x5x24xf32> -> tensor<1x5x3x2x4xf32>
  %transpose = IE.Transpose(%reshape) {order_value = #map1} : tensor<1x5x3x2x4xf32> -> tensor<3x1x2x5x4xf32>
  %fq = IE.FakeQuantize(%transpose, %fq_low, %fq_high, %fq_low, %fq_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<3x1x2x5x4xf32>, tensor<1x1x1x1x1xf32>, tensor<1x1x1x1x1xf32>, tensor<1x1x1x1x1xf32>, tensor<1x1x1x1x1xf32> -> tensor<3x1x2x5x4xf32>
  %q = IE.Gather(%fq, %q_idx) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 0 : i64} : tensor<3x1x2x5x4xf32>, tensor<1xsi64> -> tensor<1x2x5x4xf32>
  %k = IE.Gather(%fq, %k_idx) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 0 : i64} : tensor<3x1x2x5x4xf32>, tensor<1xsi64> -> tensor<1x2x5x4xf32>
  %v = IE.Gather(%transpose, %v_idx) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 0 : i64} : tensor<3x1x2x5x4xf32>, tensor<1xsi64> -> tensor<1x2x5x4xf32>
  %vt = IE.Transpose(%v) {order_value = #NCWH} : tensor<1x2x5x4xf32> -> tensor<1x2x4x5xf32>
  %attention = IE.Attention(%q, %k, %vt, %scale) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x2x5x4xf32>, tensor<1x2x5x4xf32>, tensor<1x2x4x5xf32>, tensor<1xf32> -> tensor<1x2x5x4xf32>
  return %attention : tensor<1x2x5x4xf32>

  // CHECK: [[FC:%.+]] = IE.FullyConnected([[INPUT]], [[WEIGHTS]]) : tensor<5x8xf32>, tensor<24x8xf32> -> tensor<5x24xf32>
  // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose
  // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[TRANSPOSE]]
  // CHECK: [[Q:%.+]] = IE.Gather([[FQ]]
  // CHECK: [[K:%.+]] = IE.Gather([[FQ]]
  // CHECK: [[V:%.+]] = IE.Gather([[TRANSPOSE]]
  // CHECK: [[VT:%.+]] = IE.Transpose([[V]])
  // CHECK: [[ATTENTION:%.+]] = IE.Attention([[Q]], [[K]], [[VT]],
  // CHECK: return [[ATTENTION]]
}

// -----

#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d0, d3, d1, d4)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @DontAdjustQKVTransposeWithGatherWithNonConstParam
// CHECK-SAME: ([[INPUT:%.+]]: tensor<5x8xf32>, [[Q_IDX:%.+]]: tensor<1xsi64>, [[K_IDX:%.+]]: tensor<1xsi64>, [[V_IDX:%.+]]: tensor<1xsi64>)
func.func @DontAdjustQKVTransposeWithGatherWithNonConstParam(%input: tensor<5x8xf32>, %q_idx: tensor<1xsi64>, %k_idx: tensor<1xsi64>, %v_idx: tensor<1xsi64>) -> tensor<1x2x5x4xf32> {
  %weights = const.Declare tensor<24x8xf32> = dense<1.0> : tensor<24x8xf32>
  %bias = const.Declare tensor<1x1x24xf32> = dense<2.0> : tensor<1x1x24xf32>
  %fq_low = const.Declare tensor<1x1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1x1xf32>
  %fq_high = const.Declare tensor<1x1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1x1xf32>
  %scale = const.Declare tensor<1xf32> = dense<1.0> : tensor<1xf32>

  %fc = IE.FullyConnected(%input, %weights) : tensor<5x8xf32>, tensor<24x8xf32> -> tensor<5x24xf32>
  %reshape_fc = IE.AffineReshape(%fc) {dim_mapping = [[0, 1], [2]], shape_value = [1, 5, 24]} : tensor<5x24xf32> -> tensor<1x5x24xf32>
  %add = IE.Add(%reshape_fc, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x5x24xf32>, tensor<1x1x24xf32> -> tensor<1x5x24xf32>
  %reshape = IE.AffineReshape(%add) {dim_mapping = [[0], [1], [2, 3, 4]], shape_value = [1, 5, 3, 2, 4]} : tensor<1x5x24xf32> -> tensor<1x5x3x2x4xf32>
  %transpose = IE.Transpose(%reshape) {order_value = #map1} : tensor<1x5x3x2x4xf32> -> tensor<3x1x2x5x4xf32>
  %fq = IE.FakeQuantize(%transpose, %fq_low, %fq_high, %fq_low, %fq_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<3x1x2x5x4xf32>, tensor<1x1x1x1x1xf32>, tensor<1x1x1x1x1xf32>, tensor<1x1x1x1x1xf32>, tensor<1x1x1x1x1xf32> -> tensor<3x1x2x5x4xf32>
  %q = IE.Gather(%fq, %q_idx) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 0 : i64} : tensor<3x1x2x5x4xf32>, tensor<1xsi64> -> tensor<1x2x5x4xf32>
  %k = IE.Gather(%fq, %k_idx) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 0 : i64} : tensor<3x1x2x5x4xf32>, tensor<1xsi64> -> tensor<1x2x5x4xf32>
  %v = IE.Gather(%transpose, %v_idx) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 0 : i64} : tensor<3x1x2x5x4xf32>, tensor<1xsi64> -> tensor<1x2x5x4xf32>
  %vt = IE.Transpose(%v) {order_value = #NCWH} : tensor<1x2x5x4xf32> -> tensor<1x2x4x5xf32>
  %attention = IE.Attention(%q, %k, %vt, %scale) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x2x5x4xf32>, tensor<1x2x5x4xf32>, tensor<1x2x4x5xf32>, tensor<1xf32> -> tensor<1x2x5x4xf32>
  return %attention : tensor<1x2x5x4xf32>

  // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose
  // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[TRANSPOSE]]
  // CHECK: [[Q:%.+]] = IE.Gather([[FQ]], [[Q_IDX]])
  // CHECK: [[K:%.+]] = IE.Gather([[FQ]], [[K_IDX]])
  // CHECK: [[V:%.+]] = IE.Gather([[TRANSPOSE]], [[V_IDX]])
  // CHECK: [[VT:%.+]] = IE.Transpose([[V]])
  // CHECK: [[ATTENTION:%.+]] = IE.Attention([[Q]], [[K]], [[VT]],
  // CHECK: return [[ATTENTION]]
}

// -----

#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d0, d3, d1, d4)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @DontAdjustQKVTransposeWithNonPerTensorBranchFQ
// CHECK-SAME: ([[INPUT:%.+]]: tensor<5x8xf32>)
func.func @DontAdjustQKVTransposeWithNonPerTensorBranchFQ(%input: tensor<5x8xf32>) -> tensor<1x2x5x4xf32> {
  %weights = const.Declare tensor<24x8xf32> = dense<1.0> : tensor<24x8xf32>
  %bias = const.Declare tensor<1x1x24xf32> = dense<2.0> : tensor<1x1x24xf32>
  %fq_low = const.Declare tensor<3x1x1x1x1xf32> = dense<0.0> : tensor<3x1x1x1x1xf32>
  %fq_high = const.Declare tensor<3x1x1x1x1xf32> = dense<1.0> : tensor<3x1x1x1x1xf32>
  %q_idx = const.Declare tensor<1xsi64> = dense<0> : tensor<1xsi64>
  %k_idx = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
  %v_idx = const.Declare tensor<1xsi64> = dense<2> : tensor<1xsi64>
  %scale = const.Declare tensor<1xf32> = dense<1.0> : tensor<1xf32>

  %fc = IE.FullyConnected(%input, %weights) : tensor<5x8xf32>, tensor<24x8xf32> -> tensor<5x24xf32>
  %reshape_fc = IE.AffineReshape(%fc) {dim_mapping = [[0, 1], [2]], shape_value = [1, 5, 24]} : tensor<5x24xf32> -> tensor<1x5x24xf32>
  %add = IE.Add(%reshape_fc, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x5x24xf32>, tensor<1x1x24xf32> -> tensor<1x5x24xf32>
  %reshape = IE.AffineReshape(%add) {dim_mapping = [[0], [1], [2, 3, 4]], shape_value = [1, 5, 3, 2, 4]} : tensor<1x5x24xf32> -> tensor<1x5x3x2x4xf32>
  %transpose = IE.Transpose(%reshape) {order_value = #map1} : tensor<1x5x3x2x4xf32> -> tensor<3x1x2x5x4xf32>
  %fq = IE.FakeQuantize(%transpose, %fq_low, %fq_high, %fq_low, %fq_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<3x1x2x5x4xf32>, tensor<3x1x1x1x1xf32>, tensor<3x1x1x1x1xf32>, tensor<3x1x1x1x1xf32>, tensor<3x1x1x1x1xf32> -> tensor<3x1x2x5x4xf32>
  %q = IE.Gather(%fq, %q_idx) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 0 : i64} : tensor<3x1x2x5x4xf32>, tensor<1xsi64> -> tensor<1x2x5x4xf32>
  %k = IE.Gather(%fq, %k_idx) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 0 : i64} : tensor<3x1x2x5x4xf32>, tensor<1xsi64> -> tensor<1x2x5x4xf32>
  %v = IE.Gather(%transpose, %v_idx) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 0 : i64} : tensor<3x1x2x5x4xf32>, tensor<1xsi64> -> tensor<1x2x5x4xf32>
  %vt = IE.Transpose(%v) {order_value = #NCWH} : tensor<1x2x5x4xf32> -> tensor<1x2x4x5xf32>
  %attention = IE.Attention(%q, %k, %vt, %scale) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x2x5x4xf32>, tensor<1x2x5x4xf32>, tensor<1x2x4x5xf32>, tensor<1xf32> -> tensor<1x2x5x4xf32>
  return %attention : tensor<1x2x5x4xf32>

  // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose
  // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[TRANSPOSE]]
  // CHECK: [[Q:%.+]] = IE.Gather([[FQ]]
  // CHECK: [[K:%.+]] = IE.Gather([[FQ]]
  // CHECK: [[V:%.+]] = IE.Gather([[TRANSPOSE]]
  // CHECK: [[VT:%.+]] = IE.Transpose([[V]])
  // CHECK: [[ATTENTION:%.+]] = IE.Attention([[Q]], [[K]], [[VT]],
  // CHECK: return [[ATTENTION]]
}
