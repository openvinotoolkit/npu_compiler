//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @InterpolateDMALargeInputCapsReturnTypeBound
// CHECK-SAME:    [[INPUT:%[^:]+]]: tensor<1x3x1081x1920xf16>,
// CHECK-SAME:    [[SCALES:%[^:]+]]: tensor<2xf16>,
// CHECK-SAME:    [[AUX:%[^:]+]]: tensor<1x1x1x1024xui8>
func.func @InterpolateDMALargeInputCapsReturnTypeBound(
        %input: tensor<1x3x1081x1920xf16>,
        %scales: tensor<2xf16>,
        %aux: tensor<1x1x1x1024xui8>)
            -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2160, 3838]> : tensor<4xsi64>, order = #NCHW}> {
    %0 = VPU.InterpolateDMA(%input, %scales, %aux) {
        attr = #IE.Interpolate<mode = <LINEAR>,
                               shape_calc_mode = <SCALES>,
                               coord_mode = <HALF_PIXEL>,
                               nearest_mode = <FLOOR>,
                               antialias = false,
                               pads_begin = [0, 0, 0, 0],
                               pads_end = [0, 0, 0, 0],
                               cube_coeff = -7.500000e-01 : f64>,
        axes_attr = [2, 3]
    } : tensor<1x3x1081x1920xf16>, tensor<2xf16>, tensor<1x1x1x1024xui8>
            -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2160, 3838]> : tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2160, 3838]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:       [[INTERP:%.+]] = VPU.InterpolateDMA([[INPUT]], [[SCALES]], [[AUX]])
    // CHECK-SAME:      axes_attr = [2, 3]
    // CHECK-SAME:      : tensor<1x3x1081x1920xf16>, tensor<2xf16>, tensor<1x1x1x1024xui8> -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2160, 3838]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       return [[INTERP]]
}
