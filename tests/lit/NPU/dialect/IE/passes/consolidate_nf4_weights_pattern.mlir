//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --consolidate-nf4-weights-pattern %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

!quantileFloatType = !QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.71435546875,-0.53564453125,-0.392822265625,-0.28564453125,-0.1785888671875,-0.08929443359375,0.000000e+00,0.080322265625,0.1607666015625,2.500000e-01,0.321533203125,0.428466796875,0.5712890625,0.71435546875,1.000000e+00}>

// CHECK-LABEL: @ConsolidateFP16ActNF4WeightsPattern
// CHECK-SAME:    [[INPUT:%.+]]: tensor<4096x4096xui4>
func.func @ConsolidateFP16ActNF4WeightsPattern(%input: tensor<4096x4096xui4>) -> tensor<4096x4096xf16> {
    %scale = const.Declare tensor<4096x1xf16> = dense<0.1> : tensor<4096x1xf16>
    %lut = const.Declare tensor<16xf16> = dense_resource<blob> : tensor<16xf16>
    %convert = IE.Convert(%input) {dstElemType = ui8} : tensor<4096x4096xui4> -> tensor<4096x4096xui8>
    %gather = IE.Gather(%lut, %convert) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<16xf16>, tensor<4096x4096xui8> -> tensor<4096x4096xf16>
    %res = IE.Multiply(%gather, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4096x4096xf16>, tensor<4096x1xf16> -> tensor<4096x4096xf16>
    return %res : tensor<4096x4096xf16>

    // CHECK: [[SCALE:%.+]] = const.Declare tensor<4096x1xf16> = dense<9.997550e-02> : tensor<4096x1xf16>
    // CHECK: [[QUANTCAST:%.+]] = IE.QuantizeCast([[INPUT]]) {dstElemType = !QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.71435546875,-0.53564453125,-0.392822265625,-0.28564453125,-0.1785888671875,-0.08929443359375,0.000000e+00,0.080322265625,0.1607666015625,2.500000e-01,0.321533203125,0.428466796875,0.5712890625,0.71435546875,1.000000e+00}>} : tensor<4096x4096xui4> -> tensor<4096x4096x!QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.71435546875,-0.53564453125,-0.392822265625,-0.28564453125,-0.1785888671875,-0.08929443359375,0.000000e+00,0.080322265625,0.1607666015625,2.500000e-01,0.321533203125,0.428466796875,0.5712890625,0.71435546875,1.000000e+00}>>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[QUANTCAST]]) {dstElemType = f16} : tensor<4096x4096x!QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.71435546875,-0.53564453125,-0.392822265625,-0.28564453125,-0.1785888671875,-0.08929443359375,0.000000e+00,0.080322265625,0.1607666015625,2.500000e-01,0.321533203125,0.428466796875,0.5712890625,0.71435546875,1.000000e+00}>> -> tensor<4096x4096xf16>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[CONVERT]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4096x4096xf16>, tensor<4096x1xf16> -> tensor<4096x4096xf16>
    // CHECK: return [[MULTIPLY]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x1000000000BCB7B949B849B692B4B7B1B7AD0000242D253100342535DB369238B739003C"
    }
  }
#-}

// -----

!quantileFloatType = !QuantileFloat.quantileFloat<ui4:f8E4M3FN, {-1.000000e+00,-6.875000e-01,-5.625000e-01,-4.062500e-01,-2.812500e-01,-1.718750e-01,-8.593750e-02,0.000000e+00,7.812500e-02,1.562500e-01,2.500000e-01,3.125000e-01,4.375000e-01,5.625000e-01,6.875000e-01,1.000000e+00}>

// CHECK-LABEL: @ConsolidateF8E4M3FNActNF4WeightsPattern
// CHECK-SAME:    [[INPUT:%.+]]: tensor<4096x4096xui4>
func.func @ConsolidateF8E4M3FNActNF4WeightsPattern(%input: tensor<4096x4096xui4>) -> tensor<4096x4096xf16> {
    %scale = const.Declare tensor<4096x1xf16> = dense<0.1> : tensor<4096x1xf16>
    %lut = const.Declare tensor<16xf16> = dense_resource<blob> : tensor<16xf8E4M3FN>, [#const.CastElemType<f16>]
    %convert = IE.Convert(%input) {dstElemType = ui8} : tensor<4096x4096xui4> -> tensor<4096x4096xui8>
    %gather = IE.Gather(%lut, %convert) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<16xf16>, tensor<4096x4096xui8> -> tensor<4096x4096xf16>
    %res = IE.Multiply(%gather, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4096x4096xf16>, tensor<4096x1xf16> -> tensor<4096x4096xf16>
    return %res : tensor<4096x4096xf16>

    // CHECK: [[SCALE:%.+]] = const.Declare tensor<4096x1xf16> = dense<9.997550e-02> : tensor<4096x1xf16>
    // CHECK: [[QUANTCAST:%.+]] = IE.QuantizeCast([[INPUT]]) {dstElemType = !QuantileFloat.quantileFloat<ui4:f8E4M3FN, {-1.000000e+00,-6.875000e-01,-5.625000e-01,-4.062500e-01,-2.812500e-01,-1.718750e-01,-8.593750e-02,0.000000e+00,7.812500e-02,1.562500e-01,2.500000e-01,3.125000e-01,4.375000e-01,5.625000e-01,6.875000e-01,1.000000e+00}>} : tensor<4096x4096xui4> -> tensor<4096x4096x!QuantileFloat.quantileFloat<ui4:f8E4M3FN, {-1.000000e+00,-6.875000e-01,-5.625000e-01,-4.062500e-01,-2.812500e-01,-1.718750e-01,-8.593750e-02,0.000000e+00,7.812500e-02,1.562500e-01,2.500000e-01,3.125000e-01,4.375000e-01,5.625000e-01,6.875000e-01,1.000000e+00}>>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[QUANTCAST]]) {dstElemType = f16} : tensor<4096x4096x!QuantileFloat.quantileFloat<ui4:f8E4M3FN, {-1.000000e+00,-6.875000e-01,-5.625000e-01,-4.062500e-01,-2.812500e-01,-1.718750e-01,-8.593750e-02,0.000000e+00,7.812500e-02,1.562500e-01,2.500000e-01,3.125000e-01,4.375000e-01,5.625000e-01,6.875000e-01,1.000000e+00}>> -> tensor<4096x4096xf16>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[CONVERT]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4096x4096xf16>, tensor<4096x1xf16> -> tensor<4096x4096xf16>
    // CHECK: return [[MULTIPLY]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x10000000B8B3B1ADA9A39B001A22282A2E313338"
    }
  }
#-}

// -----

!quantileFloatType = !QuantileFloat.quantileFloat<ui4:f8E4M3FN, {-1.000000e+00,-6.875000e-01,-5.625000e-01,-4.062500e-01,-2.812500e-01,-1.718750e-01,-8.593750e-02,0.000000e+00,7.812500e-02,1.562500e-01,2.500000e-01,3.125000e-01,4.375000e-01,5.625000e-01,6.875000e-01,1.000000e+00}>

// CHECK-LABEL: @ConsolidateF8E4M3FNActNF4WeightsPatternPostConvert
// CHECK-SAME:    [[INPUT:%.+]]: tensor<4096x4096xui4>
func.func @ConsolidateF8E4M3FNActNF4WeightsPatternPostConvert(%input: tensor<4096x4096xui4>) -> tensor<4096x4096xf16> {
    %scale = const.Declare tensor<4096x1xf16> = dense<0.1> : tensor<4096x1xf16>
    %lut = const.Declare tensor<16xf8E4M3FN> = dense_resource<blob> : tensor<16xf8E4M3FN>
    %convert = IE.Convert(%input) {dstElemType = ui8} : tensor<4096x4096xui4> -> tensor<4096x4096xui8>
    %gather = IE.Gather(%lut, %convert) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<16xf8E4M3FN>, tensor<4096x4096xui8> -> tensor<4096x4096xf8E4M3FN>
    %post_convert = IE.Convert(%gather) {dstElemType = f16} : tensor<4096x4096xf8E4M3FN> -> tensor<4096x4096xf16>
    %res = IE.Multiply(%post_convert, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4096x4096xf16>, tensor<4096x1xf16> -> tensor<4096x4096xf16>
    return %res : tensor<4096x4096xf16>

    // CHECK: [[SCALE:%.+]] = const.Declare tensor<4096x1xf16> = dense<9.997550e-02> : tensor<4096x1xf16>
    // CHECK: [[QUANTCAST:%.+]] = IE.QuantizeCast([[INPUT]]) {dstElemType = !QuantileFloat.quantileFloat<ui4:f8E4M3FN, {-1.000000e+00,-6.875000e-01,-5.625000e-01,-4.062500e-01,-2.812500e-01,-1.718750e-01,-8.593750e-02,0.000000e+00,7.812500e-02,1.562500e-01,2.500000e-01,3.125000e-01,4.375000e-01,5.625000e-01,6.875000e-01,1.000000e+00}>} : tensor<4096x4096xui4> -> tensor<4096x4096x!QuantileFloat.quantileFloat<ui4:f8E4M3FN, {-1.000000e+00,-6.875000e-01,-5.625000e-01,-4.062500e-01,-2.812500e-01,-1.718750e-01,-8.593750e-02,0.000000e+00,7.812500e-02,1.562500e-01,2.500000e-01,3.125000e-01,4.375000e-01,5.625000e-01,6.875000e-01,1.000000e+00}>>
    // CHECK: [[CONVERT_1:%.+]] = IE.Convert([[QUANTCAST]]) {dstElemType = f8E4M3FN} : tensor<4096x4096x!QuantileFloat.quantileFloat<ui4:f8E4M3FN, {-1.000000e+00,-6.875000e-01,-5.625000e-01,-4.062500e-01,-2.812500e-01,-1.718750e-01,-8.593750e-02,0.000000e+00,7.812500e-02,1.562500e-01,2.500000e-01,3.125000e-01,4.375000e-01,5.625000e-01,6.875000e-01,1.000000e+00}>> -> tensor<4096x4096xf8E4M3FN>
    // CHECK: [[CONVERT_2:%.+]] = IE.Convert([[CONVERT_1]]) {dstElemType = f16} : tensor<4096x4096xf8E4M3FN> -> tensor<4096x4096xf16>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[CONVERT_2]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4096x4096xf16>, tensor<4096x1xf16> -> tensor<4096x4096xf16>
    // CHECK: return [[MULTIPLY]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x10000000B8B3B1ADA9A39B001A22282A2E313338"
    }
  }
#-}

// -----

!quantileFloatType = !QuantileFloat.quantileFloat<ui4:f8E5M2, {-5.000000e-01,-2.187500e-01,-1.562500e-01,-7.812500e-02,-3.906250e-02,-0.013671875,-0.00341796875,0.000000e+00,0.0029296875,0.01171875,3.125000e-02,4.687500e-02,9.375000e-02,1.562500e-01,2.187500e-01,5.000000e-01}>

// CHECK-LABEL: @ConsolidateF8E5M2ActNF4WeightsPattern
// CHECK-SAME:    [[INPUT:%.+]]: tensor<4096x4096xui4>
func.func @ConsolidateF8E5M2ActNF4WeightsPattern(%input: tensor<4096x4096xui4>) -> tensor<4096x4096xf16> {
    %scale = const.Declare tensor<4096x1xf16> = dense<0.1> : tensor<4096x1xf16>
    %lut = const.Declare tensor<16xf16> = dense_resource<blob> : tensor<16xf8E5M2>, [#const.CastElemType<f16>]
    %convert = IE.Convert(%input) {dstElemType = ui8} : tensor<4096x4096xui4> -> tensor<4096x4096xui8>
    %gather = IE.Gather(%lut, %convert) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<16xf16>, tensor<4096x4096xui8> -> tensor<4096x4096xf16>
    %res = IE.Multiply(%gather, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4096x4096xf16>, tensor<4096x1xf16> -> tensor<4096x4096xf16>
    return %res : tensor<4096x4096xf16>

    // CHECK: [[SCALE:%.+]] = const.Declare tensor<4096x1xf16> = dense<9.997550e-02> : tensor<4096x1xf16>
    // CHECK: [[QUANTCAST:%.+]] = IE.QuantizeCast([[INPUT]]) {dstElemType = !QuantileFloat.quantileFloat<ui4:f8E5M2, {-5.000000e-01,-2.187500e-01,-1.562500e-01,-7.812500e-02,-3.906250e-02,-0.013671875,-0.00341796875,0.000000e+00,0.0029296875,0.01171875,3.125000e-02,4.687500e-02,9.375000e-02,1.562500e-01,2.187500e-01,5.000000e-01}>} : tensor<4096x4096xui4> -> tensor<4096x4096x!QuantileFloat.quantileFloat<ui4:f8E5M2, {-5.000000e-01,-2.187500e-01,-1.562500e-01,-7.812500e-02,-3.906250e-02,-0.013671875,-0.00341796875,0.000000e+00,0.0029296875,0.01171875,3.125000e-02,4.687500e-02,9.375000e-02,1.562500e-01,2.187500e-01,5.000000e-01}>>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[QUANTCAST]]) {dstElemType = f16} : tensor<4096x4096x!QuantileFloat.quantileFloat<ui4:f8E5M2, {-5.000000e-01,-2.187500e-01,-1.562500e-01,-7.812500e-02,-3.906250e-02,-0.013671875,-0.00341796875,0.000000e+00,0.0029296875,0.01171875,3.125000e-02,4.687500e-02,9.375000e-02,1.562500e-01,2.187500e-01,5.000000e-01}>> -> tensor<4096x4096xf16>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[CONVERT]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4096x4096xf16>, tensor<4096x1xf16> -> tensor<4096x4096xf16>
    // CHECK: return [[MULTIPLY]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x10000000B8B3B1ADA9A39B001A22282A2E313338"
    }
  }
#-}

// -----

!quantileFloatType = !QuantileFloat.quantileFloat<ui4:f8E5M2, {-5.000000e-01,-2.187500e-01,-1.562500e-01,-7.812500e-02,-3.906250e-02,-0.013671875,-0.00341796875,0.000000e+00,0.0029296875,0.01171875,3.125000e-02,4.687500e-02,9.375000e-02,1.562500e-01,2.187500e-01,5.000000e-01}>

// CHECK-LABEL: @ConsolidateF8E5M2ActNF4WeightsPatternPostConvert
// CHECK-SAME:    [[INPUT:%.+]]: tensor<4096x4096xui4>
func.func @ConsolidateF8E5M2ActNF4WeightsPatternPostConvert(%input: tensor<4096x4096xui4>) -> tensor<4096x4096xf16> {
    %scale = const.Declare tensor<4096x1xf16> = dense<0.1> : tensor<4096x1xf16>
    %lut = const.Declare tensor<16xf8E5M2> = dense_resource<blob> : tensor<16xf8E5M2>
    %convert = IE.Convert(%input) {dstElemType = ui8} : tensor<4096x4096xui4> -> tensor<4096x4096xui8>
    %gather = IE.Gather(%lut, %convert) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<16xf8E5M2>, tensor<4096x4096xui8> -> tensor<4096x4096xf8E5M2>
    %post_convert = IE.Convert(%gather) {dstElemType = f16} : tensor<4096x4096xf8E5M2> -> tensor<4096x4096xf16>
    %res = IE.Multiply(%post_convert, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4096x4096xf16>, tensor<4096x1xf16> -> tensor<4096x4096xf16>
    return %res : tensor<4096x4096xf16>

    // CHECK: [[SCALE:%.+]] = const.Declare tensor<4096x1xf16> = dense<9.997550e-02> : tensor<4096x1xf16>
    // CHECK: [[QUANTCAST:%.+]] = IE.QuantizeCast([[INPUT]]) {dstElemType = !QuantileFloat.quantileFloat<ui4:f8E5M2, {-5.000000e-01,-2.187500e-01,-1.562500e-01,-7.812500e-02,-3.906250e-02,-0.013671875,-0.00341796875,0.000000e+00,0.0029296875,0.01171875,3.125000e-02,4.687500e-02,9.375000e-02,1.562500e-01,2.187500e-01,5.000000e-01}>} : tensor<4096x4096xui4> -> tensor<4096x4096x!QuantileFloat.quantileFloat<ui4:f8E5M2, {-5.000000e-01,-2.187500e-01,-1.562500e-01,-7.812500e-02,-3.906250e-02,-0.013671875,-0.00341796875,0.000000e+00,0.0029296875,0.01171875,3.125000e-02,4.687500e-02,9.375000e-02,1.562500e-01,2.187500e-01,5.000000e-01}>>
    // CHECK: [[CONVERT_1:%.+]] = IE.Convert([[QUANTCAST]]) {dstElemType = f8E5M2} : tensor<4096x4096x!QuantileFloat.quantileFloat<ui4:f8E5M2, {-5.000000e-01,-2.187500e-01,-1.562500e-01,-7.812500e-02,-3.906250e-02,-0.013671875,-0.00341796875,0.000000e+00,0.0029296875,0.01171875,3.125000e-02,4.687500e-02,9.375000e-02,1.562500e-01,2.187500e-01,5.000000e-01}>> -> tensor<4096x4096xf8E5M2>
    // CHECK: [[CONVERT_2:%.+]] = IE.Convert([[CONVERT_1]]) {dstElemType = f16} : tensor<4096x4096xf8E5M2> -> tensor<4096x4096xf16>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[CONVERT_2]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4096x4096xf16>, tensor<4096x1xf16> -> tensor<4096x4096xf16>
    // CHECK: return [[MULTIPLY]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x10000000B8B3B1ADA9A39B001A22282A2E313338"
    }
  }
#-}

// -----

// CHECK-LABEL: @NotConsolidateNF4WeightsPatternIllegalGather
// CHECK-SAME:    [[INPUT:%.+]]: tensor<4096x4096xui4>
func.func @NotConsolidateNF4WeightsPatternIllegalGather(%input: tensor<4096x4096xui4>) -> tensor<16x4096x4096xf16> {
    %scale = const.Declare tensor<1xf16> = dense<0.1> : tensor<1xf16>
    %lut = const.Declare tensor<16xf16> = dense_resource<blob> : tensor<16xf8E4M3FN>, [#const.CastElemType<f16>]
    %convert = IE.Convert(%input) {dstElemType = ui8} : tensor<4096x4096xui4> -> tensor<4096x4096xui8>
    %gather = IE.Gather(%lut, %convert) {axis_value = 1 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<16xf16>, tensor<4096x4096xui8> -> tensor<16x4096x4096xf16>
    %res = IE.Multiply(%gather, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x4096x4096xf16>, tensor<1xf16> -> tensor<16x4096x4096xf16>
    return %res : tensor<16x4096x4096xf16>

    // CHECK: [[SCALE:%.+]] = const.Declare tensor<1xf16> = dense<9.997550e-02> : tensor<1xf16>
    // CHECK: [[LUT:%.+]] = const.Declare tensor<16xf16> = dense_resource<blob> : tensor<16xf8E4M3FN>, [#const.CastElemType<f16>]
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = ui8} : tensor<4096x4096xui4> -> tensor<4096x4096xui8>
    // CHECK: [[GATHER:%.+]] = IE.Gather([[LUT]], [[CONVERT]]) {axis_value = 1 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<16xf16>, tensor<4096x4096xui8> -> tensor<16x4096x4096xf16>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[GATHER]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x4096x4096xf16>, tensor<1xf16> -> tensor<16x4096x4096xf16>
    // CHECK: return [[MULTIPLY]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x10000000B8B3B1ADA9A39B001A22282A2E313338"
    }
  }
#-}

// -----

!quantileFloatType = !QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.71435546875,-0.53564453125,-0.392822265625,-0.28564453125,-0.1785888671875,-0.08929443359375,0.000000e+00,0.080322265625,0.1607666015625,2.500000e-01,0.321533203125,0.428466796875,0.5712890625,0.71435546875,1.000000e+00}>

// CHECK-LABEL: @NotConsolidateNF4WeightsPatternDynamicGather
// CHECK-SAME:    [[INPUT1:%.+]]: tensor<?xf16, {bounds = #const.OpaqueI64Elements<[16]> : tensor<1xsi64>}>
// CHECK-SAME:    [[INPUT:%.+]]: tensor<4096x4096xui4>
func.func @NotConsolidateNF4WeightsPatternDynamicGather(%input1: tensor<?xf16, {bounds = #const.OpaqueI64Elements<[16]> : tensor<1xsi64>}>, %input: tensor<4096x4096xui4>) -> tensor<4096x4096xf16> {
    %scale = const.Declare tensor<1xf16> = dense<0.1> : tensor<1xf16>
    %convert = IE.Convert(%input) {dstElemType = ui8} : tensor<4096x4096xui4> -> tensor<4096x4096xui8>
    %gather = IE.Gather(%input1, %convert) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<?xf16, {bounds = #const.OpaqueI64Elements<[16]> : tensor<1xsi64>}>, tensor<4096x4096xui8> -> tensor<4096x4096xf16>
    %res = IE.Multiply(%gather, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4096x4096xf16>, tensor<1xf16> -> tensor<4096x4096xf16>
    return %res : tensor<4096x4096xf16>

    // CHECK: [[SCALE:%.+]] = const.Declare tensor<1xf16> = dense<{{.+}}> : tensor<1xf16>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = ui8} : tensor<4096x4096xui4> -> tensor<4096x4096xui8>
    // CHECK: [[GATHER:%.+]] = IE.Gather([[INPUT1]], [[CONVERT]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<?xf16, {bounds = #const.OpaqueI64Elements<[16]> : tensor<1xsi64>}>, tensor<4096x4096xui8> -> tensor<4096x4096xf16>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[GATHER]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4096x4096xf16>, tensor<1xf16> -> tensor<4096x4096xf16>
    // CHECK: return [[MULTIPLY]]
}
