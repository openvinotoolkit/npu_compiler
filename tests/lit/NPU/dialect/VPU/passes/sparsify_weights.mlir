//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --sparsify-weights %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotSparsifyFullyDense
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x32x16x16xf16, {order = #NHWC}>, [[WEIGHT_TABLE:%.+]]: tensor<32x1x1x4xsi32>)
func.func @DoNotSparsifyFullyDense(%input: tensor<1x32x16x16xf16, {order = #NHWC}>, %weight_table: tensor<32x1x1x4xsi32>) -> tensor<1x32x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %nce = VPU.NCE.Convolution(%input, %weights, %weight_table) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [32, 32, 1, 1],
            strides = [1, 1]
        } : tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x16x16xf16, {order = #NHWC}>

    return %nce : tensor<1x32x16x16xf16, {order = #NHWC}>

    // CHECK-NOT:  const.Sparsify
    // CHECK-NOT:  const.GetSparsityMap
    // CHECK-NOT:  VPU.GroupSparseTensor
    // CHECK-DAG:  [[WEIGHTS:%.+]] = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK:      VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHT_TABLE]])
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotSparsifyFullySparse
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x32x16x16xf16, {order = #NHWC}>, [[WEIGHT_TABLE:%.+]]: tensor<32x1x1x4xsi32>)
func.func @DoNotSparsifyFullySparse(%input: tensor<1x32x16x16xf16, {order = #NHWC}>, %weight_table: tensor<32x1x1x4xsi32>) -> tensor<1x32x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<0.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %nce = VPU.NCE.Convolution(%input, %weights, %weight_table) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [32, 32, 1, 1],
            strides = [1, 1]
        } : tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x16x16xf16, {order = #NHWC}>

    return %nce : tensor<1x32x16x16xf16, {order = #NHWC}>

    // CHECK-NOT:  const.Sparsify
    // CHECK-NOT:  const.GetSparsityMap
    // CHECK-NOT:  VPU.GroupSparseTensor
    // CHECK-DAG:  [[WEIGHTS:%.+]] = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK:      VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHT_TABLE]])
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SparsifyHalfDense
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x32x16x16xf16, {order = #NHWC}>, [[WEIGHT_TABLE:%.+]]: tensor<32x1x1x4xsi32>)
func.func @SparsifyHalfDense(%input: tensor<1x32x16x16xf16, {order = #NHWC}>, %weight_table: tensor<32x1x1x4xsi32>) -> tensor<1x32x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x16x1x1xf16>, [
        #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 16, 0, 0]>]
    %nce = VPU.NCE.Convolution(%input, %weights, %weight_table) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [32, 32, 1, 1],
            strides = [1, 1]
        } : tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x16x16xf16, {order = #NHWC}>

    return %nce : tensor<1x32x16x16xf16, {order = #NHWC}>

    // CHECK:  [[DATA:%.+]] = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = {{.*}} [#const.Sparsify<false>]
    // CHECK:  [[DATA_SM:%.+]] = const.Declare tensor<32x1x1x128xi1> = {{.*}} [#const.GetSparsityMap]
    // CHECK:  [[SPARSE:%.+]] = VPU.GroupSparseTensor([[DATA]], [[DATA_SM]])
    // CHECK:  VPU.NCE.Convolution([[INPUT]], [[SPARSE]], [[WEIGHT_TABLE]])
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SparsifyWithMultiUsers
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x32x16x16xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<32x1x1x4xsi32>,
// CHECK-SAME: [[ARG2:%.+]]: tensor<1x32x16x16xf16, {order = #NHWC}>, [[ARG3:%.+]]: tensor<32x1x1x4xsi32>)
// CHECK-SAME: -> (tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<1x32x16x16xf16, {order = #NHWC}>)
func.func @SparsifyWithMultiUsers(%arg0: tensor<1x32x16x16xf16, {order = #NHWC}>, %arg1: tensor<32x1x1x4xsi32>,
                                  %arg2: tensor<1x32x16x16xf16, {order = #NHWC}>, %arg3: tensor<32x1x1x4xsi32>)
          -> (tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<1x32x16x16xf16, {order = #NHWC}>) {
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x22x1x1xf16>, [
        #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 10, 0, 0]>]
    %1 = VPU.NCE.Convolution(%arg0, %weights, %arg1) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [32, 32, 1, 1],
            strides = [1, 1]
        } : tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x16x16xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%arg2, %weights, %arg3) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [32, 32, 1, 1],
            strides = [1, 1]
        } : tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x16x16xf16, {order = #NHWC}>

    return %1, %2: tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<1x32x16x16xf16, {order = #NHWC}>

    // CHECK: [[DATA:%.+]] = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = {{.*}} [#const.Sparsify<false>]
    // CHECK: [[DATA_SM:%.+]] = const.Declare tensor<32x1x1x128xi1> = {{.*}} [#const.GetSparsityMap]
    // CHECK: [[SPARSE:%.+]] = VPU.GroupSparseTensor([[DATA]], [[DATA_SM]])

    // CHECK: [[RES0:%.+]] = VPU.NCE.Convolution([[ARG0]], [[SPARSE]], [[ARG1]])
    // CHECK: [[RES1:%.+]] = VPU.NCE.Convolution([[ARG2]], [[SPARSE]], [[ARG3]])

    // CHECK: return [[RES0]], [[RES1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  func.func @DoNotSparsifyForFragmentation
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x128x32x1xf16, {order = #NHWC}>)
// CHECK-SAME:  -> tensor<1x128x8x4xf16, {order = #NHWC}> {
func.func @DoNotSparsifyForFragmentation(%arg0:tensor<1x128x32x1xf16, {order = #NHWC}>)-> tensor<1x128x8x4xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<128x1x1x4xsi32> = dense<4> : tensor<128x1x1x4xsi32>
    %cst_1 = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<128x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 128, 8, 4]} : tensor<1x128x32x1xf16, {order = #NHWC}> -> tensor<1x128x8x4xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst_1, %cst_0)
        {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <NOOP>,
        clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        rawFilterShape = [128, 128, 1, 1], strides = [1, 1]}
        : tensor<1x128x8x4xf16, {order = #NHWC}>, tensor<128x128x1x1xf16, {order = #NHWC}>, tensor<128x1x1x4xsi32> -> tensor<1x128x8x4xf16, {order = #NHWC}>
    return %1 : tensor<1x128x8x4xf16, {order = #NHWC}>

    // CHECK:    [[CST:%.+]] = const.Declare tensor<128x1x1x4xsi32> = dense<4> : tensor<128x1x1x4xsi32>
    // CHECK:    [[CST_0:%.+]] = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x128x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-NOT: VPU.GroupSparseTensor
    // CHECK:    [[RESHAPE:%.+]] = VPU.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 128, 8, 4]}
    // CHECK-SAME:   tensor<1x128x32x1xf16, {order = #NHWC}> -> tensor<1x128x8x4xf16, {order = #NHWC}>
    // CHECK:    [[CONV:%.+]] = VPU.NCE.Convolution([[RESHAPE]], [[CST_0]], [[CST]])
    // CHECK-SAME:  {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:  ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
    // CHECK-SAME:  lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:  rawFilterShape = [128, 128, 1, 1],
    // CHECK-SAME:  strides = [1, 1]}
    // CHECK-SAME:  -> tensor<1x128x8x4xf16, {order = #NHWC}>
    // CHECK:    return [[CONV]] : tensor<1x128x8x4xf16, {order = #NHWC}>
}
