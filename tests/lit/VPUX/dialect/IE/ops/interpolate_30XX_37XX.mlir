//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @OvPreprocessingInterpolateU8
func @OvPreprocessingInterpolateU8(%arg0: tensor<1x10x30x30xui8>) -> tensor<1x10x40x40xui8> {
    %cst = const.Declare tensor<2xsi64> = dense<40> : tensor<2xsi64>
    %cst_0 = const.Declare tensor<2xf32> = dense<1.333330e+00> : tensor<2xf32>
    %cst_1 = const.Declare tensor<2xsi64> = dense<[2, 3]> : tensor<2xsi64>
    %0 = IE.Interpolate(%arg0, %cst, %cst_0, %cst_1) {attr = {antialias = false, coord_mode = "HALF_PIXEL", cube_coeff = -7.500000e-01 : f64, mode = "NEAREST", nearest_mode = "ROUND_PREFER_FLOOR", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SIZES"}, operand_segment_sizes = dense<1> : vector<4xi32>} : tensor<1x10x30x30xui8>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x10x40x40xui8>

    return %0 : tensor<1x10x40x40xui8>
    // CHECK:       [[VAL0:%.*]] = IE.Convert
    // CHECK-SAME:      dstElemType = f16
    // CHECK:       [[VAL1:%.*]] = IE.Interpolate([[VAL0]]
    // CHECK:       IE.Convert([[VAL1]]
    // CHECK-SAME:      dstElemType = ui8
}
