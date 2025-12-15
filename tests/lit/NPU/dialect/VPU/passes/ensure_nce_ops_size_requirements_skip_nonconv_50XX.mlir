//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --ensure-nce-ops-size-requirements="enable-output-ensurance=true skip-non-conv-oc=true" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @DontSplitNonConvOC
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<1x16384x1x1xf16, {order = #NHWC}>
// CHECK-SAME:    [[INPUT1:%.+]]: tensor<16384x16x1x1xf16, {order = #NHWC}>
// CHECK-SAME:    [[INPUT2:%.+]]: tensor<16384x1x1x4xsi32, {order = #NCHW}>
func.func @DontSplitNonConvOC(%arg0: tensor<1x16384x1x1xf16, {order = #NHWC}>,
                                %arg1: tensor<16384x16x1x1xf16, {order = #NHWC}>,
                                %arg2: tensor<16384x1x1x4xsi32, {order = #NCHW}>) -> tensor<1x16384x1x1xf16, {order = #NHWC}> {
    %0 = VPU.NCE.DepthConvolution(%arg0, %arg1, %arg2) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
        rawFilterShape = [16384, 1, 1, 1],
        strides = [1, 1]} -> tensor<1x16384x1x1xf16, {order = #NHWC}>

    return %0 : tensor<1x16384x1x1xf16, {order = #NHWC}>

    // OC is bigger than limitation 8k but non-conv op is not split when skip-non-conv-oc is set true
    // CHECK:      [[DWCONV:%.+]] = VPU.NCE.DepthConvolution
    // CHECK-SAME:         -> tensor<1x16384x1x1xf16, {order = #NHWC}>
}
