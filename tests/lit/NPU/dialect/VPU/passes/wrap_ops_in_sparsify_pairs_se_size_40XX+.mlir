//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --wrap-ops-in-sparsify-pairs="enable-activation-sparsity-mode=true sparsity-profile=S0" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @WrapSingleOpChannelsNotPow2
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x48x48x16xf16, {order = #NHWC}>, [[WEIGHTS:%.+]]: tensor<48x48x1x1xf16, {order = #NHWC}>
// CHECK-SAME:)
func.func @WrapSingleOpChannelsNotPow2(
        %input: tensor<1x48x48x16xf16, {order = #NHWC}>, %weights: tensor<48x48x1x1xf16, {order = #NHWC}>
    ) -> tensor<1x48x48x16xf16, {order = #NHWC}> {
    %conv = VPU.NCE.Convolution(%input, %weights) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [48, 48, 1, 1],
            strides = [1, 1]
        } : tensor<1x48x48x16xf16, {order = #NHWC}>, tensor<48x48x1x1xf16, {order = #NHWC}> -> tensor<1x48x48x16xf16, {order = #NHWC}>

    return %conv : tensor<1x48x48x16xf16, {order = #NHWC}>

    // CHECK:       [[IN_SPARSIFY:%.+]] = VPU.Sparsify([[INPUT]])
    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[IN_SPARSIFY]], [[WEIGHTS]])
    // CHECK-NOT:       -> !VPU.SparseTensor
    // CHECK-SAME:      -> tensor<1x48x48x16xf16, {order = #NHWC}>
    // CHECK:       [[OUT_SPARSIFY:%.+]] = VPU.Sparsify([[CONV]])
    // CHECK:       [[OUT_DESPARSIFY:%.+]] = VPU.Desparsify([[OUT_SPARSIFY]])
    // CHECK:       return [[OUT_DESPARSIFY]]
}
