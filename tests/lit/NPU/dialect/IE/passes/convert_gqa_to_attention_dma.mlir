//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-gqa-to-attention-dma %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @ConvertGQAToAttentionDMA
// CHECK-SAME:  ([[Q:%.+]]: tensor<1x8x1x64xf16>, [[K:%.+]]: tensor<1x2x1x64xf16>, [[V:%.+]]: tensor<1x2x1x64xf16>,
// CHECK-SAME:   [[PK:%.+]]: tensor<1x2x512x64xf16>, [[PV:%.+]]: tensor<1x2x512x64xf16>, [[SEQ:%.+]]: tensor<1xsi32>)
func.func @ConvertGQAToAttentionDMA(%q: tensor<1x8x1x64xf16>, %k: tensor<1x2x1x64xf16>, %v: tensor<1x2x1x64xf16>,
                                    %pk: tensor<1x2x512x64xf16>, %pv: tensor<1x2x512x64xf16>, %seq: tensor<1xsi32>)
        -> (tensor<1x1x512xf16>, tensor<1x2x512x64xf16>, tensor<1x2x512x64xf16>) {
    %gqa:3 = IE.GroupQueryAttention(%q, %k, %v, %pk, %pv, %seq) {
        causal, kQuantType = #IE.gqa_quant_type<NONE>, kvCacheBitWidth = 0 : i64, kvNumHeads = 2 : i64,
        localWindowSize = -1 : i64, numHeads = 8 : i64,
        operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0>,
        scale = 0.000000e+00 : f32, smoothSoftmax, vQuantType = #IE.gqa_quant_type<NONE>
    } : tensor<1x8x1x64xf16>, tensor<1x2x1x64xf16>, tensor<1x2x1x64xf16>, tensor<1x2x512x64xf16>,
        tensor<1x2x512x64xf16>, tensor<1xsi32>
        -> tensor<1x1x512xf16>, tensor<1x2x512x64xf16>, tensor<1x2x512x64xf16>
    return %gqa#0, %gqa#1, %gqa#2 : tensor<1x1x512xf16>, tensor<1x2x512x64xf16>, tensor<1x2x512x64xf16>

    // CHECK-NOT:   IE.GroupQueryAttention
    // CHECK:       [[SCALE:%.+]] = const.Declare tensor<1xf16> = dense<1.250000e-01>
    // CHECK:       [[ATT_OUT:%.+]], [[PRESENT_K:%.+]], [[PRESENT_V:%.+]] = IE.AttentionDMA([[Q]], [[K]], [[V]], [[PK]], [[PV]], [[SCALE]], [[SEQ]])
    // CHECK-SAME:      {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0>, smoothSoftmax}
    // CHECK-SAME:      -> tensor<1x8x1x64xf16>, tensor<1x2x512x64xf16>, tensor<1x2x512x64xf16>
    // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[ATT_OUT]])
    // CHECK-SAME:      tensor<1x8x1x64xf16> -> tensor<1x1x8x64xf16>
    // CHECK:       [[OUT:%.+]] = IE.Reshape([[TRANSPOSE]]) {shape_value = [1, 1, 512]}
    // CHECK:       return [[OUT]], [[PRESENT_K]], [[PRESENT_V]]
}

// -----

// CHECK-LABEL: @ConvertGQAToAttentionDMARotary
// CHECK-SAME:  ([[Q:%.+]]: tensor<1x8x4x64xf16>, [[K:%.+]]: tensor<1x2x4x64xf16>, [[V:%.+]]: tensor<1x2x4x64xf16>,
// CHECK-SAME:   [[PK:%.+]]: tensor<1x2x512x64xf16>, [[PV:%.+]]: tensor<1x2x512x64xf16>, [[SEQ:%.+]]: tensor<1xsi64>,
// CHECK-SAME:   [[COS:%.+]]: tensor<512x32xf16>, [[SIN:%.+]]: tensor<512x32xf16>)
func.func @ConvertGQAToAttentionDMARotary(%q: tensor<1x8x4x64xf16>, %k: tensor<1x2x4x64xf16>,
                                          %v: tensor<1x2x4x64xf16>, %pk: tensor<1x2x512x64xf16>,
                                          %pv: tensor<1x2x512x64xf16>, %seq: tensor<1xsi64>,
                                          %cos: tensor<512x32xf16>, %sin: tensor<512x32xf16>)
        -> (tensor<1x4x512xf16>, tensor<1x2x512x64xf16>, tensor<1x2x512x64xf16>) {
    %gqa:3 = IE.GroupQueryAttention(%q, %k, %v, %pk, %pv, %seq, %cos, %sin) {
        causal, doRotary, kQuantType = #IE.gqa_quant_type<NONE>, kvCacheBitWidth = 0 : i64, kvNumHeads = 2 : i64,
        localWindowSize = -1 : i64, numHeads = 8 : i64,
        operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0>,
        rotaryInterleaved, scale = 5.000000e-01 : f32, vQuantType = #IE.gqa_quant_type<NONE>
    } : tensor<1x8x4x64xf16>, tensor<1x2x4x64xf16>, tensor<1x2x4x64xf16>, tensor<1x2x512x64xf16>,
        tensor<1x2x512x64xf16>, tensor<1xsi64>, tensor<512x32xf16>, tensor<512x32xf16>
        -> tensor<1x4x512xf16>, tensor<1x2x512x64xf16>, tensor<1x2x512x64xf16>
    return %gqa#0, %gqa#1, %gqa#2 : tensor<1x4x512xf16>, tensor<1x2x512x64xf16>, tensor<1x2x512x64xf16>

    // CHECK:       [[SEQ32:%.+]] = IE.Convert([[SEQ]]) {dstElemType = si32}
    // CHECK:       [[PAST:%.+]] = IE.Reshape([[SEQ32]]) {shape_value = [1]}
    // CHECK:       [[OFFSETS:%.+]] = const.Declare tensor<4xsi32> = dense<[-3, -2, -1, 0]>
    // CHECK:       [[POS:%.+]] = IE.Add([[PAST]], [[OFFSETS]])
    // CHECK:       [[COS_G:%.+]] = IE.Gather([[COS]], [[POS]])
    // CHECK:       [[COS_R:%.+]] = IE.Reshape([[COS_G]]) {shape_value = [1, 1, 4, 32]}
    // CHECK:       [[SIN_G:%.+]] = IE.Gather([[SIN]], [[POS]])
    // CHECK:       [[SIN_R:%.+]] = IE.Reshape([[SIN_G]]) {shape_value = [1, 1, 4, 32]}
    // CHECK:       [[ROPE_Q:%.+]] = IE.RoPE([[Q]], [[COS_R]], [[SIN_R]]) {mode = #IE.rope_mode<PAIRWISE_INTERLEAVED>}
    // CHECK:       [[ROPE_K:%.+]] = IE.RoPE([[K]], [[COS_R]], [[SIN_R]]) {mode = #IE.rope_mode<PAIRWISE_INTERLEAVED>}
    // CHECK:       [[SCALE:%.+]] = const.Declare tensor<1xf16> = dense<5.000000e-01>
    // CHECK:       [[SEQ_K:%.+]] = IE.Convert([[SEQ]]) {dstElemType = si32}
    // CHECK:       [[ATT_OUT:%.+]], [[PRESENT_K:%.+]], [[PRESENT_V:%.+]] = IE.AttentionDMA([[ROPE_Q]], [[ROPE_K]], [[V]], [[PK]], [[PV]], [[SCALE]], [[SEQ_K]])
    // CHECK-SAME:      {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0>}
    // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[ATT_OUT]])
    // CHECK-SAME:      tensor<1x8x4x64xf16> -> tensor<1x4x8x64xf16>
    // CHECK:       [[OUT:%.+]] = IE.Reshape([[TRANSPOSE]]) {shape_value = [1, 4, 512]}
    // CHECK:       return [[OUT]], [[PRESENT_K]], [[PRESENT_V]]
}

// -----

// CHECK-LABEL: @ConvertGQAToAttentionDMAQuantizedWindowSink
// CHECK-SAME:  ([[Q:%.+]]: tensor<1x8x1x64xf16>, [[K:%.+]]: tensor<1x2x1x64xf16>, [[V:%.+]]: tensor<1x2x1x64xf16>,
// CHECK-SAME:   [[PK:%.+]]: tensor<1x2x1024x64xsi8>, [[PV:%.+]]: tensor<1x2x1024x64xsi8>, [[SEQ:%.+]]: tensor<1xsi32>,
// CHECK-SAME:   [[SINK:%.+]]: tensor<8xf16>, [[KS:%.+]]: tensor<2x64xf16>, [[VS:%.+]]: tensor<2x64xf16>)
func.func @ConvertGQAToAttentionDMAQuantizedWindowSink(%q: tensor<1x8x1x64xf16>, %k: tensor<1x2x1x64xf16>,
                                                       %v: tensor<1x2x1x64xf16>, %pk: tensor<1x2x1024x64xsi8>,
                                                       %pv: tensor<1x2x1024x64xsi8>, %seq: tensor<1xsi32>,
                                                       %sink: tensor<8xf16>, %ks: tensor<2x64xf16>,
                                                       %vs: tensor<2x64xf16>)
        -> (tensor<1x1x512xf16>, tensor<1x2x1024x64xsi8>, tensor<1x2x1024x64xsi8>) {
    %gqa:3 = IE.GroupQueryAttention(%q, %k, %v, %pk, %pv, %seq, %sink, %ks, %vs) {
        causal, kQuantType = #IE.gqa_quant_type<PER_CHANNEL>, kvCacheBitWidth = 8 : i64, kvNumHeads = 2 : i64,
        localWindowSize = 128 : i64, numHeads = 8 : i64,
        operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1>,
        scale = 0.000000e+00 : f32, slidingWindowCache, smoothSoftmax, vQuantType = #IE.gqa_quant_type<PER_CHANNEL>
    } : tensor<1x8x1x64xf16>, tensor<1x2x1x64xf16>, tensor<1x2x1x64xf16>, tensor<1x2x1024x64xsi8>,
        tensor<1x2x1024x64xsi8>, tensor<1xsi32>, tensor<8xf16>, tensor<2x64xf16>, tensor<2x64xf16>
        -> tensor<1x1x512xf16>, tensor<1x2x1024x64xsi8>, tensor<1x2x1024x64xsi8>
    return %gqa#0, %gqa#1, %gqa#2 : tensor<1x1x512xf16>, tensor<1x2x1024x64xsi8>, tensor<1x2x1024x64xsi8>

    // CHECK:       [[SCALE:%.+]] = const.Declare tensor<1xf16> = dense<1.250000e-01>
    // CHECK:       [[K_SCALE:%.+]] = IE.Reshape([[KS]]) {shape_value = [1, 2, 1, 64]}
    // CHECK:       [[V_SCALE:%.+]] = IE.Reshape([[VS]]) {shape_value = [1, 2, 1, 64]}
    // CHECK:       [[HEAD_SINK:%.+]] = IE.Reshape([[SINK]]) {shape_value = [1, 8, 1, 1]}
    // CHECK:       [[ATT_OUT:%.+]], [[PRESENT_K:%.+]], [[PRESENT_V:%.+]] = IE.AttentionDMA([[Q]], [[K]], [[V]], [[PK]], [[PV]], [[SCALE]], [[HEAD_SINK]], [[SEQ]], [[K_SCALE]], [[V_SCALE]])
    // CHECK-SAME:      {localWindowSize = 128 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1>, slidingWindowCache}
    // CHECK-SAME:      -> tensor<1x8x1x64xf16>, tensor<1x2x1024x64xsi8>, tensor<1x2x1024x64xsi8>
    // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[ATT_OUT]])
    // CHECK:       [[OUT:%.+]] = IE.Reshape([[TRANSPOSE]]) {shape_value = [1, 1, 512]}
    // CHECK:       return [[OUT]], [[PRESENT_K]], [[PRESENT_V]]
}
