//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --ensure-nce-ops-size-requirements="enable-output-ensurance=true skip-eltwise-oc=SKIP_LARGE_SPATIAL" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitEltwiseOCWithSmallSpatialDim
func.func @SplitEltwiseOCWithSmallSpatialDim(%arg0: tensor<1x16384x1x1xf16, {order = #NHWC}>,
                                        %arg1: tensor<1x16384x1x1xf16, {order = #NHWC}>)
                                        -> tensor<1x16384x1x1xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>
    } -> tensor<1x16384x1x1xf16, {order = #NHWC}>

    return %0 : tensor<1x16384x1x1xf16, {order = #NHWC}>

    // CHECK:                   VPU.Concat
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 8192, 0, 0]]}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @DontSplitEltwiseOCWithLargeSpatialDim
func.func @DontSplitEltwiseOCWithLargeSpatialDim(%arg0: tensor<1x16384x8x8xf16, {order = #NHWC}>,
                                                %arg1: tensor<1x16384x8x8xf16, {order = #NHWC}>)
                                                -> tensor<1x16384x8x8xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>
    } -> tensor<1x16384x8x8xf16, {order = #NHWC}>

    return %0 : tensor<1x16384x8x8xf16, {order = #NHWC}>

    // CHECK:      VPU.NCE.Eltwise
    // CHECK-SAME:     -> tensor<1x16384x8x8xf16, {order = #NHWC}>
}
