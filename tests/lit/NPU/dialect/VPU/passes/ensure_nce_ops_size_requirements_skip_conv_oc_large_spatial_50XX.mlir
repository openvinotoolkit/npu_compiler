//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --ensure-nce-ops-size-requirements="enable-output-ensurance=true skip-conv-oc=SKIP_LARGE_SPATIAL" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitConvOCWithSmallSpatialDim
func.func @SplitConvOCWithSmallSpatialDim(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>,
                                    %arg1: tensor<16384x16x1x1xf16, {order = #NHWC}>)
                                    -> tensor<1x16384x1x1xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Convolution(%arg0, %arg1) rawFilterShape [16384, 16, 1, 1] {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
        
        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        strides = [1, 1]
    } : tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<16384x16x1x1xf16, {order = #NHWC}> -> tensor<1x16384x1x1xf16, {order = #NHWC}>

    return %0 : tensor<1x16384x1x1xf16, {order = #NHWC}>

    // CHECK:                   VPU.Concat
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 8192, 0, 0]]}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @DontSplitConvOCWithLargeSpatialDim
func.func @DontSplitConvOCWithLargeSpatialDim(%arg0: tensor<1x16x8x8xf16, {order = #NHWC}>,
                                            %arg1: tensor<16384x16x1x1xf16, {order = #NHWC}>)
                                            -> tensor<1x16384x8x8xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Convolution(%arg0, %arg1) rawFilterShape [16384, 16, 1, 1] {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
        
        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        strides = [1, 1]
    } : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16384x16x1x1xf16, {order = #NHWC}> -> tensor<1x16384x8x8xf16, {order = #NHWC}>

    return %0 : tensor<1x16384x8x8xf16, {order = #NHWC}>

    // CHECK:      VPU.NCE.Convolution
    // CHECK-SAME:     -> tensor<1x16384x8x8xf16, {order = #NHWC}>
}
