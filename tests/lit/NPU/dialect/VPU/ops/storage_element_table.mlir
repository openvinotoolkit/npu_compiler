//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CanonicalizeSETableSlice
func.func @CanonicalizeSETableSlice()
        -> tensor<1x1x5x9xi32, {order = #NHWC}> {
    %se_table = VPU.StorageElementTable {
                    dataElemType = f16,
                    dataShape = [1, 64, 4, 4],
                    seAttr = #VPU.SEInterpolate<mode = <BILINEAR>,
                                                coordinate_transformation_mode = <ASYMMETRIC>,
                                                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                                offsets = [0, 0, 0, 0],
                                                sizes = [1, 64, 9, 9]>,
                    seDepth = 2 : i64, seSize = 32 : i64}
                -> tensor<1x2x9x9xi32, {order = #NHWC}>
    // Tile over H and C
    %se_table_slice = VPU.Slice %se_table [0, 0, 0, 0] [1, 1, 5, 9] : tensor<1x2x9x9xi32, {order = #NHWC}> to tensor<1x1x5x9xi32, {order = #NHWC}>

    return %se_table_slice : tensor<1x1x5x9xi32, {order = #NHWC}>

    // CHECK:       [[SE_TABLE:%.*]] = VPU.StorageElementTable {
    // CHECK-SAME:    dataElemType = f16,
    // CHECK-SAME:    dataShape = [1, 32, 3, 4],
    // CHECK-SAME:    seAttr = #VPU.SEInterpolate<mode = <BILINEAR>,
    // CHECK-SAME:                                coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                                offsets = [0, 0, 0, 0],
    // CHECK-SAME:                                sizes = [1, 32, 5, 9]>,
    // CHECK-SAME:    seDepth = 1 : i64,
    // CHECK-SAME:    seSize = 32 : i64}
    // CHECK-SAME:  -> tensor<1x1x5x9xi32, {order = #NHWC}>

    // CHECK: return [[SE_TABLE]] : tensor<1x1x5x9xi32, {order = #NHWC}>
}
