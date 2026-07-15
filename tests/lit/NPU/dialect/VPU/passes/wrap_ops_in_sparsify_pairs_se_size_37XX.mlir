//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --wrap-ops-in-sparsify-pairs="enable-activation-sparsity-mode=true sparsity-profile=S0" %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @DoNotWrapSingleOpChannelsNotPow2
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x48x48x16xf16, {order = #NHWC}>, [[WEIGHTS:%.+]]: tensor<48x48x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:     [[WEIGHTS_TABLE:%.+]]: tensor<48x1x1x4xsi32>)
func.func @DoNotWrapSingleOpChannelsNotPow2(
        %input: tensor<1x48x48x16xf16, {order = #NHWC}>, %weights: tensor<48x48x1x1xf16, {order = #NHWC}>, %weights_table: tensor<48x1x1x4xsi32>
    ) -> tensor<1x48x48x16xf16, {order = #NHWC}> {
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) rawFilterShape [48, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            
            strides = [1, 1]
        } : tensor<1x48x48x16xf16, {order = #NHWC}>, tensor<48x48x1x1xf16, {order = #NHWC}>, tensor<48x1x1x4xsi32> -> tensor<1x48x48x16xf16, {order = #NHWC}>

    return %conv : tensor<1x48x48x16xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.Sparsify
    // CHECK-NOT:   VPU.Desparsify
    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-NOT:       -> !VPU.SparseTensor
    // CHECK-SAME:      -> tensor<1x48x48x16xf16, {order = #NHWC}>
    // CHECK-NOT:   VPU.Sparsify
    // CHECK-NOT:   VPU.Desparsify
    // CHECK:       return [[CONV]]
}
