//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --verify-diagnostics %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @NCEReduceMeanInvalidPadding(%arg0: tensor<1x16x30x30xf16, {order = #NHWC}>) -> tensor<1x1x30x30xf16, {order = #NHWC}> {
    // expected-error@+2 {{'VPU.NCE.Reduce' op inferred type(s) 'tensor<1x16x30x30xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>' are incompatible with return type(s) of operation 'tensor<1x1x30x30xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>'}}
    // expected-error@+1 {{'VPU.NCE.Reduce' op failed to infer returned types}}
    %0 = VPU.NCE.Reduce(%arg0) {axes = [1], op_type = #VPU.reduce_type<MEAN>, output_padding = [0, 15, 0, 0], ppe = #VPU.PPEStub<>} -> tensor<1x1x30x30xf16, {order = #NHWC}>
    return %0 : tensor<1x1x30x30xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @NCEReduceMeanInvalidPaddingRank(%arg0: tensor<1x16x30x30xf16, {order = #NHWC}>) -> tensor<1x1x30x30xf16, {order = #NHWC}> {
    // expected-error@+1 {{Output padding [0, 15] incompatible with output type tensor<1x1x30x30xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>}}
    %0 = VPU.NCE.Reduce(%arg0) {axes = [1], op_type = #VPU.reduce_type<MEAN>, output_padding = [0, 15], ppe = #VPU.PPEStub<>} -> tensor<1x1x30x30xf16, {order = #NHWC}>
    return %0 : tensor<1x1x30x30xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @NCEReduceSumInvalidPadding(%arg0: tensor<1x16x30x30xf16, {order = #NHWC}>) -> tensor<1x1x30x30xf16, {order = #NHWC}> {
    // expected-error@+2 {{'VPU.NCE.Reduce' op inferred type(s) 'tensor<1x16x30x30xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>' are incompatible with return type(s) of operation 'tensor<1x1x30x30xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>'}}
    // expected-error@+1 {{'VPU.NCE.Reduce' op failed to infer returned types}}
    %0 = VPU.NCE.Reduce(%arg0) {axes = [1], op_type = #VPU.reduce_type<SUM>, output_padding = [0, 15, 0, 0], ppe = #VPU.PPEStub<>} -> tensor<1x1x30x30xf16, {order = #NHWC}>
    return %0 : tensor<1x1x30x30xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @NCEReduceSumInvalidPaddingRank(%arg0: tensor<1x16x30x30xf16, {order = #NHWC}>) -> tensor<1x1x30x30xf16, {order = #NHWC}> {
    // expected-error@+1 {{Output padding [0, 15] incompatible with output type tensor<1x1x30x30xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>}}
    %0 = VPU.NCE.Reduce(%arg0) {axes = [1], op_type = #VPU.reduce_type<SUM>, output_padding = [0, 15], ppe = #VPU.PPEStub<>} -> tensor<1x1x30x30xf16, {order = #NHWC}>
    return %0 : tensor<1x1x30x30xf16, {order = #NHWC}>
}
