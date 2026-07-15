//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --enable-weights-sparsity="weights-sparsity-heuristic=RATIO" %s | FileCheck %s
// REQUIRES: platform-NPU3720
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotSparsifyCompressedConv
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x4x16x16xf16, {order = #NHWC}>, [[ARG_1:%[^:]+]]: tensor<16x1x1x4xsi32>)
func.func @DoNotSparsifyCompressedConv(%arg0: tensor<1x4x16x16xf16, {order = #NHWC}>, %arg1: tensor<16x1x1x4xsi32>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x1x1x16xf16, {order = #NHWC}> = dense<1.0> : tensor<16x3x1x1xf16>, [
        #const.Reorder<#NHWC>, #const.Reshape<[16, 1, 1, 3]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 13]>]
    %1 = VPU.NCE.CompressConvolution(%arg0, %weights, %arg1) rawFilterShape [16, 3, 1, 1] {
            cm_sp_pattern = 7 : i64,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            
            strides = [1, 1],
            ppe = #VPU.PPEStub<>
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-NOT:   const.Sparsify
    // CHECK-NOT:   const.GetSparsityMap
    // CHECK-NOT:   VPU.GroupSparseTensor
    // CHECK-DAG:       [[weights:%.+]] = const.Declare tensor<16x1x1x16xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:    : tensor<16x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[16, 1, 1, 3]>,
    // CHECK-SAME:                              #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 13]>]
    // CHECK:       VPU.NCE.CompressConvolution([[ARG_0]], [[weights]], [[ARG_1]])
}
