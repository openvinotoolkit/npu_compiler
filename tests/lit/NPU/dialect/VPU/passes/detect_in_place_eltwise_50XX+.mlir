//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --detect-in-place-eltwise %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InplaceEltwiseWithSprLUT
func.func @InplaceEltwiseWithSprLUT(%input1: tensor<1x256x56x56xf16, {order = #NHWC}>,
                                    %input2: tensor<1x256x56x56xf16, {order = #NHWC}>) -> tensor<1x256x56x56xf16, {order = #NHWC}> {
    %avg_pool = VPU.NCE.AveragePool(%input1) {
        kernel_size = [1, 1],
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1]
    } -> tensor<1x256x56x56xf16, {order = #NHWC}>

    %eltwise = VPU.NCE.Eltwise(%avg_pool, %input2) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEFp<mode = <SWISH>,
                         clamp_low = -3.4028234663852886E+38 : f64,
                         clamp_high = 3.4028234663852886E+38 : f64,
                         scale = 1.000000e+00 : f64,
                         prelu_alpha = [1.000000e+00],
                         bias = 0.000000e+00 : f64,
                         adder = 0.000000e+00 : f64,
                         sprlut = dense_resource<__elided__> : tensor<580xui16>>
    } -> tensor<1x256x56x56xf16, {order = #NHWC}>

    return %eltwise : tensor<1x256x56x56xf16, {order = #NHWC}>

    //CHECK-NOT: is_inplace = true
}
