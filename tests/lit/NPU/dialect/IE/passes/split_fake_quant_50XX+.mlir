//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --split-fake-quant --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010
// COM: F8 is only supported on NPU50+, no need to run these tests on all platforms.

// CHECK:  !qElemType = !quant.uniform<f8E4M3FN:f32, 1.000000e+00>

// CHECK-LABEL: @SingleQuantParamsF8E4M3FN
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x3x30x30xf32>
func.func @SingleQuantParamsF8E4M3FN(%input: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %low = const.Declare tensor<1x1x1x1xf32> = dense<-4.480000e+02> : tensor<1x1x1x1xf32>
    %high = const.Declare tensor<1x1x1x1xf32> = dense<4.480000e+02> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%input, %low, %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK-NOT:    IE.FakeQuantize

    // CHECK:    [[QUAN:%.+]] = IE.Quantize([[INPUT]]) {dstElemType = !qElemType} : tensor<1x3x30x30xf32> -> tensor<1x3x30x30x!qElemType>
    // CHECK:    [[DEQUAN:%.+]] = IE.Dequantize([[QUAN]]) {dstElemType = f32} : tensor<1x3x30x30x!qElemType> -> tensor<1x3x30x30xf32>

    // CHECK:    return [[DEQUAN]] : tensor<1x3x30x30xf32>
}

// -----

// CHECK-LABEL: @DontSingleQuantParamsF8E4M3FNNonZeroZeroPoint
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x3x30x30xf32>
func.func @DontSingleQuantParamsF8E4M3FNNonZeroZeroPoint(%input: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %low = const.Declare tensor<1x1x1x1xf32> = dense<-4.480000e+02> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<4.480000e+02> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%input, %low, %input_high, %low, %output_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-4.480000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:    [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<4.480000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>

    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[LOW]], [[IN_HIGH]], [[LOW]], [[OUT_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    // CHECK:    return [[FQ]] : tensor<1x3x30x30xf32>
}

// -----

// CHECK:  !qElemType = !quant.uniform<f8E5M2:f16, 1.000000e+00>
// CHECK:  !qElemType1 = !quant.uniform<f8E5M2:f16:1, {7.812500e-03,1.562500e-02,0.006173270089285714,3.906250e-03}>

// CHECK-LABEL: @PerChannelQuantOutputF8E5M2
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x4x15x15xf16>
func.func @PerChannelQuantOutputF8E5M2(%input: tensor<1x4x15x15xf16>) -> tensor<1x4x15x15xf16> {
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<-5.734400e+04> : tensor<1x1x1x1xf16>
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<5.734400e+04> : tensor<1x1x1x1xf16>
    %output_low = const.Declare tensor<1x4x1x1xf16> = dense<[[[[-448.0]], [[-896.0]], [[-354.0]], [[-224.0]]]]> : tensor<1x4x1x1xf16>
    %output_high =  const.Declare tensor<1x4x1x1xf16> = dense<[[[[448.0]], [[896.0]], [[354.0]], [[224.0]]]]> : tensor<1x4x1x1xf16>

    %1 = IE.FakeQuantize(%input, %input_low, %input_high, %output_low, %output_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2
    } : tensor<1x4x15x15xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x4x1x1xf16>, tensor<1x4x1x1xf16> -> tensor<1x4x15x15xf16>

    return %1 : tensor<1x4x15x15xf16>

    // CHECK-NOT:    IE.FakeQuantize

    // CHECK:    [[QUAN:%.+]] = IE.Quantize([[INPUT]]) {dstElemType = !qElemType} : tensor<1x4x15x15xf16> -> tensor<1x4x15x15x!qElemType>
    // CHECK:    [[Q_CAST:%.+]] = IE.QuantizeCast([[QUAN]]) {dstElemType = !qElemType1} : tensor<1x4x15x15x!qElemType> -> tensor<1x4x15x15x!qElemType1>
    // CHECK:    [[DEQUAN:%.+]] = IE.Dequantize([[Q_CAST]]) {dstElemType = f16} : tensor<1x4x15x15x!qElemType1> -> tensor<1x4x15x15xf16>

    // CHECK:    return [[DEQUAN]] : tensor<1x4x15x15xf16>
}

// -----


// CHECK-LABEL: @ConstantsSplitFakeQuantF8E4M3FN
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x16x16xf16>
func.func @ConstantsSplitFakeQuantF8E4M3FN(%input: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
    %weights = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
    %low = const.Declare tensor<1x1x1x1xf16> = dense<-4.480000e+02> : tensor<1x1x1x1xf16>
    %high = const.Declare tensor<1x1x1x1xf16> = dense<4.480000e+02> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%weights, %low, %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<16x16x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<16x16x1x1xf16>

    %1 = IE.Convolution(%input, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

    return %1 : tensor<1x16x16x16xf16>

    // CHECK-NOT:    IE.FakeQuantize

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf8E4M3FN> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<f8E4M3FN>]
    // CHECK: [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:    [[DEQUAN:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]]) {dstElemType = f16, vpux.synthetic_dyn_dequant} : tensor<16x16x1x1xf8E4M3FN>, tensor<1x1x1x1xf32> -> tensor<16x16x1x1xf16>
    // CHECK:    [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUAN]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

    // CHECK:    return [[CONV]] : tensor<1x16x16x16xf16>
}


// -----

!qElemType = !quant.uniform<i8:f16, 0.0078431372549019607:-1>

// CHECK-LABEL: @SplitFakeQuantU4ConstWeightsAsymmetric
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x512x1x1xf16>)
func.func @SplitFakeQuantU4ConstWeightsAsymmetric(%arg0: tensor<1x512x1x1xf16>) -> tensor<1x256x1x1xf16> {

    %cst_wgt = const.Declare tensor<1x256x1x512xf16> = dense<3> : tensor<256x512xui4>, [#const.Reshape<[1, 256, 1, 512]>, #const.ConvertElemType<ui8>, #const.CastElemType<f16>]
    %cst_wgt_low  = const.Declare tensor<1x1x1x1xf16> = dense<0.0>  : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_wgt_high = const.Declare tensor<1x1x1x1xf16> = dense<15.0> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_wgt_out_low  = const.Declare tensor<1x256x1x1xf16> = dense<-2.0> : tensor<256x1xf32>, [#const.Reshape<[1, 256, 1, 1]>, #const.CastElemType<f16>]
    %cst_wgt_out_high = const.Declare tensor<1x256x1x1xf16> = dense<2.0>  : tensor<256x1xf32>, [#const.Reshape<[1, 256, 1, 1]>, #const.CastElemType<f16>]

    %cst_act_low  = const.Declare tensor<1x1x1x1xf16> = dense<-1.0> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_act_high = const.Declare tensor<1x1x1x1xf16> = dense<1.0>  : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    %fq_wgt = IE.FakeQuantize(%cst_wgt, %cst_wgt_low, %cst_wgt_high, %cst_wgt_out_low, %cst_wgt_out_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<1x256x1x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x256x1x1xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x1x512xf16>
    %fq_act = IE.FakeQuantize(%arg0, %cst_act_low, %cst_act_high, %cst_act_low, %cst_act_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x512x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x512x1x1xf16>
    %wgt_reshape = IE.AffineReshape(%fq_wgt) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [256, 512, 1, 1]} : tensor<1x256x1x512xf16> -> tensor<256x512x1x1xf16>
    %result = IE.Convolution(%fq_act, %wgt_reshape) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x1x1xf16>, tensor<256x512x1x1xf16> -> tensor<1x256x1x1xf16>
    return %result : tensor<1x256x1x1xf16>

    // CHECK-NOT: IE.FakeQuantize
    // CHECK:     [[WGT:%.+]] = const.Declare tensor<1x256x1x512xui4>
    // CHECK: [[SCALE:%.+]] = const.Declare tensor<1x256x1x1xf32> = dense<0.266666681> : tensor<1x256x1x1xf32>
    // CHECK: [[ZP:%.+]] = const.Declare tensor<1x256x1x1xui4> = dense<8> : tensor<1x256x1x1xui4>
    // CHECK:     [[DQWGT:%.+]] = IE.DynamicDequantize([[WGT]], [[SCALE]], [[ZP]]) {dstElemType = f16, vpux.synthetic_dyn_dequant} : tensor<1x256x1x512xui4>, tensor<1x256x1x1xf32>, tensor<1x256x1x1xui4> -> tensor<1x256x1x512xf16>
    // CHECK:     [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType} : tensor<1x512x1x1xf16> -> tensor<1x512x1x1x!qElemType>
    // CHECK:     [[DQACT:%.+]] = IE.Dequantize([[QUANT]]) {dstElemType = f16} : tensor<1x512x1x1x!qElemType> -> tensor<1x512x1x1xf16>
    // CHECK:     [[RESHAPE:%.+]] = IE.AffineReshape([[DQWGT]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [256, 512, 1, 1]} : tensor<1x256x1x512xf16> -> tensor<256x512x1x1xf16>
    // CHECK:     [[CONV:%.+]] = IE.Convolution([[DQACT]], [[RESHAPE]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x1x1xf16>, tensor<256x512x1x1xf16> -> tensor<1x256x1x1xf16>
    // CHECK:     return [[CONV]]
}


// -----


!qElemType = !quant.uniform<i8:f16, 0.0078431372549019607:-1>

// CHECK-LABEL: @SplitFakeQuantI4ConstWeightsSigned
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x512x1x1xf16>)
func.func @SplitFakeQuantI4ConstWeightsSigned(%arg0: tensor<1x512x1x1xf16>) -> tensor<1x256x1x1xf16> {
    // i4 signed weights (low=-8, high=7, symmetric), activation uses i8 because weights are SI
    %cst_wgt = const.Declare tensor<1x256x1x512xf16> = dense<3> : tensor<256x512xi4>, [#const.Reshape<[1, 256, 1, 512]>, #const.ConvertElemType<ui8>, #const.CastElemType<f16>]
    %cst_wgt_low  = const.Declare tensor<1x1x1x1xf16> = dense<-8.0> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_wgt_high = const.Declare tensor<1x1x1x1xf16> = dense<7.0>  : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_act_low  = const.Declare tensor<1x1x1x1xf16> = dense<-1.0> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_act_high = const.Declare tensor<1x1x1x1xf16> = dense<1.0>  : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    %fq_wgt = IE.FakeQuantize(%cst_wgt, %cst_wgt_low, %cst_wgt_high, %cst_wgt_low, %cst_wgt_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<1x256x1x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x256x1x512xf16>
    %fq_act = IE.FakeQuantize(%arg0, %cst_act_low, %cst_act_high, %cst_act_low, %cst_act_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x512x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x512x1x1xf16>
    %wgt_reshape = IE.AffineReshape(%fq_wgt) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [256, 512, 1, 1]} : tensor<1x256x1x512xf16> -> tensor<256x512x1x1xf16>
    %result = IE.Convolution(%fq_act, %wgt_reshape) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x1x1xf16>, tensor<256x512x1x1xf16> -> tensor<1x256x1x1xf16>
    return %result : tensor<1x256x1x1xf16>

    // CHECK-NOT: IE.FakeQuantize
    // CHECK:     [[WGT:%.+]] = const.Declare tensor<1x256x1x512xsi4>
    // CHECK: [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:     [[DQWGT:%.+]] = IE.DynamicDequantize([[WGT]], [[SCALE]]) {dstElemType = f16, vpux.synthetic_dyn_dequant} : tensor<1x256x1x512xsi4>, tensor<1x1x1x1xf32> -> tensor<1x256x1x512xf16>
    // CHECK:     [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType} : tensor<1x512x1x1xf16> -> tensor<1x512x1x1x!qElemType>
    // CHECK:     [[DQACT:%.+]] = IE.Dequantize([[QUANT]]) {dstElemType = f16} : tensor<1x512x1x1x!qElemType> -> tensor<1x512x1x1xf16>
    // CHECK:     [[RESHAPE:%.+]] = IE.AffineReshape([[DQWGT]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [256, 512, 1, 1]} : tensor<1x256x1x512xf16> -> tensor<256x512x1x1xf16>
    // CHECK:     [[CONV:%.+]] = IE.Convolution([[DQACT]], [[RESHAPE]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x1x1xf16>, tensor<256x512x1x1xf16> -> tensor<1x256x1x1xf16>
    // CHECK:     return [[CONV]]
}


// -----

!qElemType = !quant.uniform<u8:f16, 0.0078431372549019607:128>

// CHECK-LABEL: @SplitFakeQuantI4ConstWeightsAsymmetric
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x512x1x1xf16>)
func.func @SplitFakeQuantI4ConstWeightsAsymmetric(%arg0: tensor<1x512x1x1xf16>) -> tensor<1x256x1x1xf16> {
    // i4 signed weights (storage range [-8, 7]) with an asymmetric output range: the reconstructed
    // zero point must stay packed in i4 storage, not get upcast to i8/f16/f32.
    %cst_wgt = const.Declare tensor<1x256x1x512xf16> = dense<3> : tensor<256x512xi4>, [#const.Reshape<[1, 256, 1, 512]>, #const.ConvertElemType<ui8>, #const.CastElemType<f16>]
    %cst_wgt_low  = const.Declare tensor<1x1x1x1xf16> = dense<-8.0> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_wgt_high = const.Declare tensor<1x1x1x1xf16> = dense<7.0>  : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_wgt_out_low  = const.Declare tensor<1x256x1x1xf16> = dense<-6.0> : tensor<256x1xf32>, [#const.Reshape<[1, 256, 1, 1]>, #const.CastElemType<f16>]
    %cst_wgt_out_high = const.Declare tensor<1x256x1x1xf16> = dense<8.0>  : tensor<256x1xf32>, [#const.Reshape<[1, 256, 1, 1]>, #const.CastElemType<f16>]

    %cst_act_low  = const.Declare tensor<1x1x1x1xf16> = dense<-1.0> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_act_high = const.Declare tensor<1x1x1x1xf16> = dense<1.0>  : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    %fq_wgt = IE.FakeQuantize(%cst_wgt, %cst_wgt_low, %cst_wgt_high, %cst_wgt_out_low, %cst_wgt_out_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<1x256x1x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x256x1x1xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x1x512xf16>
    %fq_act = IE.FakeQuantize(%arg0, %cst_act_low, %cst_act_high, %cst_act_low, %cst_act_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x512x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x512x1x1xf16>
    %wgt_reshape = IE.AffineReshape(%fq_wgt) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [256, 512, 1, 1]} : tensor<1x256x1x512xf16> -> tensor<256x512x1x1xf16>
    %result = IE.Convolution(%fq_act, %wgt_reshape) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x1x1xf16>, tensor<256x512x1x1xf16> -> tensor<1x256x1x1xf16>
    return %result : tensor<1x256x1x1xf16>

    // CHECK-NOT: IE.FakeQuantize
    // CHECK:     [[WGT:%.+]] = const.Declare tensor<1x256x1x512xsi4> = dense<3> : tensor<256x512xi4>, [#const.Reshape<[1, 256, 1, 512]>, #const.ConvertElemType<ui8>, #const.CastElemType<si4>]
    // CHECK:     [[SCALE:%.+]] = const.Declare tensor<1x256x1x1xf32> = dense<0.933333337> : tensor<1x256x1x1xf32>
    // Zero point is reconstructed from the FQ and stays packed in i4 storage.
    // CHECK:     [[ZP:%.+]] = const.Declare tensor<1x256x1x1xsi4> = dense<-2> : tensor<1x256x1x1xsi4>
    // CHECK:     [[DQWGT:%.+]] = IE.DynamicDequantize([[WGT]], [[SCALE]], [[ZP]]) {dstElemType = f16, vpux.synthetic_dyn_dequant} : tensor<1x256x1x512xsi4>, tensor<1x256x1x1xf32>, tensor<1x256x1x1xsi4> -> tensor<1x256x1x512xf16>
    // CHECK:     [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType} : tensor<1x512x1x1xf16> -> tensor<1x512x1x1x!qElemType>
    // CHECK:     [[DQACT:%.+]] = IE.Dequantize([[QUANT]]) {dstElemType = f16} : tensor<1x512x1x1x!qElemType> -> tensor<1x512x1x1xf16>
    // CHECK:     [[RESHAPE:%.+]] = IE.AffineReshape([[DQWGT]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [256, 512, 1, 1]} : tensor<1x256x1x512xf16> -> tensor<256x512x1x1xf16>
    // CHECK:     [[CONV:%.+]] = IE.Convolution([[DQACT]], [[RESHAPE]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x1x1xf16>, tensor<256x512x1x1xf16> -> tensor<1x256x1x1xf16>
    // CHECK:     return [[CONV]]
}


// -----

!qElemType = !quant.uniform<i8:f16, 0.011764705882352941:42>

// CHECK-LABEL: @SplitFakeQuantI2ConstWeightsSigned
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x512x1x1xf16>)
func.func @SplitFakeQuantI2ConstWeightsSigned(%arg0: tensor<1x512x1x1xf16>) -> tensor<1x256x1x1xf16> {
    // i2 signed weights (low=-2, high=1, symmetric), activation uses i8 because weights are SI
    %cst_wgt = const.Declare tensor<1x256x1x512xf16> = dense<1> : tensor<256x512xi2>, [#const.Reshape<[1, 256, 1, 512]>, #const.ConvertElemType<ui8>, #const.CastElemType<f16>]
    %cst_wgt_low  = const.Declare tensor<1x1x1x1xf16> = dense<-2.0> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_wgt_high = const.Declare tensor<1x1x1x1xf16> = dense<1.0>  : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_act_low  = const.Declare tensor<1x1x1x1xf16> = dense<-2.0> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_act_high = const.Declare tensor<1x1x1x1xf16> = dense<1.0>  : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    %fq_wgt = IE.FakeQuantize(%cst_wgt, %cst_wgt_low, %cst_wgt_high, %cst_wgt_low, %cst_wgt_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 4 : i64} : tensor<1x256x1x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x256x1x512xf16>
    %fq_act = IE.FakeQuantize(%arg0, %cst_act_low, %cst_act_high, %cst_act_low, %cst_act_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x512x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x512x1x1xf16>
    %wgt_reshape = IE.AffineReshape(%fq_wgt) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [256, 512, 1, 1]} : tensor<1x256x1x512xf16> -> tensor<256x512x1x1xf16>
    %result = IE.Convolution(%fq_act, %wgt_reshape) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x1x1xf16>, tensor<256x512x1x1xf16> -> tensor<1x256x1x1xf16>
    return %result : tensor<1x256x1x1xf16>

    // CHECK-NOT: IE.FakeQuantize
    // CHECK:     [[WGT:%.+]] = const.Declare tensor<1x256x1x512xsi2>
    // CHECK: [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:     [[DQWGT:%.+]] = IE.DynamicDequantize([[WGT]], [[SCALE]]) {dstElemType = f16, vpux.synthetic_dyn_dequant} : tensor<1x256x1x512xsi2>, tensor<1x1x1x1xf32> -> tensor<1x256x1x512xf16>
    // CHECK:     [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType} : tensor<1x512x1x1xf16> -> tensor<1x512x1x1x!qElemType>
    // CHECK:     [[DQACT:%.+]] = IE.Dequantize([[QUANT]]) {dstElemType = f16} : tensor<1x512x1x1x!qElemType> -> tensor<1x512x1x1xf16>
    // CHECK:     [[RESHAPE:%.+]] = IE.AffineReshape([[DQWGT]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [256, 512, 1, 1]} : tensor<1x256x1x512xf16> -> tensor<256x512x1x1xf16>
    // CHECK:     [[CONV:%.+]] = IE.Convolution([[DQACT]], [[RESHAPE]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x1x1xf16>, tensor<256x512x1x1xf16> -> tensor<1x256x1x1xf16>
    // CHECK:     return [[CONV]]
}
