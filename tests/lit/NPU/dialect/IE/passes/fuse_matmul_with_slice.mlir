//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-matmul-with-slice %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @FuseWithOneSliceUser
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<32x16xf16>
func.func @FuseWithOneSliceUser(%arg0: tensor<32x16xf16>) -> tensor<32x32xf16> {
    %weights = const.Declare tensor<64x16xf16> = dense<1.0> : tensor<64x16xf16>
    %0 = IE.FullyConnected(%arg0, %weights) : tensor<32x16xf16>, tensor<64x16xf16> -> tensor<32x64xf16>
    %1 = IE.Slice %0 [0, 0] [32, 32] : tensor<32x64xf16> to tensor<32x32xf16>
    return %1 : tensor<32x32xf16>

    // CHECK-DAG:  [[WEIGHTS:%.+]] = const.Declare tensor<32x16xf16> = dense<1.000000e+00> : tensor<64x16xf16>, [#const.SubView<[0, 0], [32, 16]>]
    // CHECK:      [[FC:%.+]] = IE.FullyConnected([[ARG_0]], [[WEIGHTS]])
    // CHECK:      return      [[FC]]
}

// -----

// CHECK-LABEL: @FuseWithOneSliceUserAnd1DBias
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<32x16xf16>
func.func @FuseWithOneSliceUserAnd1DBias(%arg0: tensor<32x16xf16>) -> tensor<32x32xf16> {
    %weights = const.Declare tensor<64x16xf16> = dense<1.0> : tensor<64x16xf16>
    %bias = const.Declare tensor<64xf16> = dense<1.0> : tensor<64xf16>
    %0 = IE.FullyConnected(%arg0, %weights, %bias) : tensor<32x16xf16>, tensor<64x16xf16>, tensor<64xf16> -> tensor<32x64xf16>
    %1 = IE.Slice %0 [0, 0] [32, 32] : tensor<32x64xf16> to tensor<32x32xf16>
    return %1 : tensor<32x32xf16>

    // CHECK-DAG:  [[WEIGHTS:%.+]] = const.Declare tensor<32x16xf16> = dense<1.000000e+00> : tensor<64x16xf16>, [#const.SubView<[0, 0], [32, 16]>]
    // CHECK-DAG:  [[BIAS:%.+]] = const.Declare tensor<32xf16> = dense<1.000000e+00> : tensor<64xf16>, [#const.SubView<[0], [32]>]
    // CHECK:      [[FC:%.+]] = IE.FullyConnected([[ARG_0]], [[WEIGHTS]], [[BIAS]])
    // CHECK:      return      [[FC]]
}

// -----

// CHECK-LABEL: @NotFuseDataBoundOp
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<15x16xf16>
func.func @NotFuseDataBoundOp(%arg0: tensor<15x16xf16>) -> tensor<15x32xf16> {
    %weights = const.Declare tensor<64x16xf16> = dense<1.0> : tensor<64x16xf16>
    %0 = IE.FullyConnected(%arg0, %weights) : tensor<15x16xf16>, tensor<64x16xf16> -> tensor<15x64xf16>
    %1 = IE.Slice %0 [0, 0] [15, 32] : tensor<15x64xf16> to tensor<15x32xf16>
    return %1 : tensor<15x32xf16>

    // CHECK-DAG:  [[WEIGHTS:%.+]] = const.Declare tensor<64x16xf16> = dense<1.000000e+00> : tensor<64x16xf16>
    // CHECK:      [[FC:%.+]] = IE.FullyConnected([[ARG_0]], [[WEIGHTS]])
    // CHECK:      [[SLICE:%.+]] = IE.Slice [[FC]]
    // CHECK:      return [[SLICE]] : tensor<15x32xf16>
}

// -----

// CHECK-LABEL: @NotFuseWithNonChannelSlice
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<32x16xf16>
func.func @NotFuseWithNonChannelSlice(%arg0: tensor<32x16xf16>) -> tensor<16x64xf16> {
    %weights = const.Declare tensor<64x16xf16> = dense<1.0> : tensor<64x16xf16>
    %bias = const.Declare tensor<1x64xf16> = dense<1.0> : tensor<1x64xf16>
    %0 = IE.FullyConnected(%arg0, %weights, %bias) : tensor<32x16xf16>, tensor<64x16xf16>, tensor<1x64xf16> -> tensor<32x64xf16>
    %1 = IE.Slice %0 [0, 0] [16, 64] : tensor<32x64xf16> to tensor<16x64xf16>
    return %1 : tensor<16x64xf16>

    // CHECK-DAG:  [[WEIGHTS:%.+]] = const.Declare tensor<64x16xf16> = dense<1.000000e+00> : tensor<64x16xf16>
    // CHECK-DAG:  [[BIAS:%.+]] = const.Declare tensor<1x64xf16> = dense<1.000000e+00> : tensor<1x64xf16>
    // CHECK:      [[FC:%.+]] = IE.FullyConnected([[ARG_0]], [[WEIGHTS]], [[BIAS]])
    // CHECK:      [[SLICE:%.+]] = IE.Slice [[FC]]
    // CHECK:      return      [[SLICE]]
}

// -----

// CHECK-LABEL: @NotFuseWithSliceOverlap
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<32x16xf16>
func.func @NotFuseWithSliceOverlap(%arg0: tensor<32x16xf16>) -> (tensor<32x16xf16>, tensor<32x32xf16>) {
    %weights = const.Declare tensor<64x16xf16> = dense<1.0> : tensor<64x16xf16>
    %0 = IE.FullyConnected(%arg0, %weights) : tensor<32x16xf16>, tensor<64x16xf16> -> tensor<32x64xf16>
    %1 = IE.Slice %0 [0, 31] [32, 16] : tensor<32x64xf16> to tensor<32x16xf16>
    %2 = IE.Slice %0 [0, 0] [32, 32] : tensor<32x64xf16> to tensor<32x32xf16>
    %3 = IE.ReLU(%1) : tensor<32x16xf16> -> tensor<32x16xf16>
    return %3, %2 : tensor<32x16xf16>, tensor<32x32xf16>

    // CHECK-DAG:  [[WEIGHTS:%.+]] = const.Declare tensor<64x16xf16> = dense<1.000000e+00> : tensor<64x16xf16>
    // CHECK:      [[FC0:%.+]] = IE.FullyConnected([[ARG_0]], [[WEIGHTS]])
    // CHECK:      [[SLICE0:%.+]] = IE.Slice [[FC0]] [0, 31] [32, 16] : tensor<32x64xf16> to tensor<32x16xf16>
    // CHECK:      [[SLICE1:%.+]] = IE.Slice [[FC0]] [0, 0] [32, 32] : tensor<32x64xf16> to tensor<32x32xf16>
    // CHECK:      [[RELU:%.+]] = IE.ReLU([[SLICE0]])

    // CHECK:      return      [[RELU]],  [[SLICE1]]
}

// -----

// CHECK-LABEL: @NotFuseWithNonSliceUser
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<32x16xf16>
func.func @NotFuseWithNonSliceUser(%arg0: tensor<32x16xf16>) -> (tensor<32x64xf16>, tensor<32x32xf16>) {
    %weights = const.Declare tensor<64x16xf16> = dense<1.0> : tensor<64x16xf16>
    %0 = IE.FullyConnected(%arg0, %weights) : tensor<32x16xf16>, tensor<64x16xf16> -> tensor<32x64xf16>
    %1 = IE.ReLU(%0) : tensor<32x64xf16> -> tensor<32x64xf16>
    %2 = IE.Slice %0 [0, 0] [32, 32] : tensor<32x64xf16> to tensor<32x32xf16>
    return %1, %2 : tensor<32x64xf16>, tensor<32x32xf16>

    // CHECK-DAG:  [[WEIGHTS:%.+]] = const.Declare tensor<64x16xf16> = dense<1.000000e+00> : tensor<64x16xf16>
    // CHECK:      [[FC:%.+]] = IE.FullyConnected([[ARG_0]], [[WEIGHTS]])
    // CHECK:      [[RELU:%.+]] = IE.ReLU([[FC]])
    // CHECK:      [[SLICE:%.+]] = IE.Slice [[FC]]
    // CHECK:      return      [[RELU]],  [[SLICE]]
}

// -----

// CHECK-LABEL: @FuseFullyConnectedWithSliceThroughViewOpChain
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<32x16xf16>
func.func @FuseFullyConnectedWithSliceThroughViewOpChain(%arg0: tensor<32x16xf16>) -> (tensor<32x512x1x1xf16>, tensor<32x512x1x1xf16>) {
    %weights = const.Declare tensor<1024x16xf16> = dense<1.0> : tensor<1024x16xf16>
    %0 = IE.FullyConnected(%arg0, %weights) : tensor<32x16xf16>, tensor<1024x16xf16> -> tensor<32x1024xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1, 2, 3]], shape_value = [32, 1024, 1, 1]} : tensor<32x1024xf16> -> tensor<32x1024x1x1xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [32, 512, 1, 1] : tensor<32x1024x1x1xf16> to tensor<32x512x1x1xf16>
    %3 = IE.Slice %1 [0, 512, 0, 0] [32, 512, 1, 1] : tensor<32x1024x1x1xf16> to tensor<32x512x1x1xf16>
    %4 = IE.ReLU(%3) : tensor<32x512x1x1xf16> -> tensor<32x512x1x1xf16>
    return %2, %4 : tensor<32x512x1x1xf16>, tensor<32x512x1x1xf16>

    // CHECK:   [[W_SLICE0:%.+]] = const.Declare tensor<512x16xf16> = dense<1.000000e+00> : tensor<1024x16xf16>, [#const.SubView<[512, 0], [512, 16]>]
    // CHECK:   [[FC0:%.+]] = IE.FullyConnected([[ARG_0]], [[W_SLICE0]]) : tensor<32x16xf16>, tensor<512x16xf16> -> tensor<32x512xf16>
    // CHECK:   [[RESHAPE0:%.+]] = IE.AffineReshape([[FC0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2, 3]], shape_value = [32, 512, 1, 1]} : tensor<32x512xf16> -> tensor<32x512x1x1xf16>

    // CHECK:   [[W_SLICE1:%.+]] = const.Declare tensor<512x16xf16> = dense<1.000000e+00> : tensor<1024x16xf16>, [#const.SubView<[0, 0], [512, 16]>]
    // CHECK:   [[FC1:%.+]] = IE.FullyConnected([[ARG_0]], [[W_SLICE1]]) : tensor<32x16xf16>, tensor<512x16xf16> -> tensor<32x512xf16>
    // CHECK:   [[RESHAPE1:%.+]] = IE.AffineReshape([[FC1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2, 3]], shape_value = [32, 512, 1, 1]} : tensor<32x512xf16> -> tensor<32x512x1x1xf16>

    // CHECK:       [[ACTIVATION:%.+]] = IE.ReLU([[RESHAPE0]])

    // CHECK:       return [[RESHAPE1]], [[ACTIVATION]]
}

// -----

// CHECK-LABEL: @NotFuseFullyConnectedThroughViewOpChainWithUnsupportedOp
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<32x16xf16>
func.func @NotFuseFullyConnectedThroughViewOpChainWithUnsupportedOp(%arg0: tensor<32x16xf16>) -> (tensor<32x512x1x1xf16>, tensor<32x512x1x1xf16>) {
    %weights = const.Declare tensor<1024x16xf16> = dense<1.0> : tensor<1024x16xf16>
    %0 = IE.FullyConnected(%arg0, %weights) : tensor<32x16xf16>, tensor<1024x16xf16> -> tensor<32x1024xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1, 2, 3]], shape_value = [32, 1024, 1, 1]} : tensor<32x1024xf16> -> tensor<32x1024x1x1xf16>
    %2 = IE.Add(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<32x1024x1x1xf16>, tensor<32x1024x1x1xf16> -> tensor<32x1024x1x1xf16>
    %3 = IE.Slice %2 [0, 0, 0, 0] [32, 512, 1, 1] : tensor<32x1024x1x1xf16> to tensor<32x512x1x1xf16>
    %4 = IE.Slice %2 [0, 512, 0, 0] [32, 512, 1, 1] : tensor<32x1024x1x1xf16> to tensor<32x512x1x1xf16>
    return %3, %4 : tensor<32x512x1x1xf16>, tensor<32x512x1x1xf16>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<1024x16xf16> = dense<1.000000e+00> : tensor<1024x16xf16>
    // CHECK:       [[FC:%.+]] = IE.FullyConnected([[ARG_0]], [[WEIGHTS]]) : tensor<32x16xf16>, tensor<1024x16xf16> -> tensor<32x1024xf16>
    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[FC]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2, 3]], shape_value = [32, 1024, 1, 1]} : tensor<32x1024xf16> -> tensor<32x1024x1x1xf16>
    // CHECK:       [[ADD:%.+]] = IE.Add([[RESHAPE]], [[RESHAPE]])
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[ADD]] [0, 0, 0, 0] [32, 512, 1, 1] : tensor<32x1024x1x1xf16> to tensor<32x512x1x1xf16>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[ADD]] [0, 512, 0, 0] [32, 512, 1, 1] : tensor<32x1024x1x1xf16> to tensor<32x512x1x1xf16>

    // CHECK:       return [[SLICE0]], [[SLICE1]]
}

// -----

// CHECK-LABEL: @FuseMatMulWithSliceThroughViewOpChain
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x6x32x16xf16>
func.func @FuseMatMulWithSliceThroughViewOpChain(%arg0: tensor<1x6x32x16xf16>) -> (tensor<192x512x1x1xf16>, tensor<192x512x1x1xf16>) {
    %weights = const.Declare tensor<1x6x1024x16xf16> = dense<1.0> : tensor<1x6x1024x16xf16>
    %0 = IE.MatMul(%arg0, %weights) {transpose_b} : tensor<1x6x32x16xf16>, tensor<1x6x1024x16xf16> -> tensor<1x6x32x1024xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [192, 1024, 1, 1]} : tensor<1x6x32x1024xf16> -> tensor<192x1024x1x1xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [192, 512, 1, 1] : tensor<192x1024x1x1xf16> to tensor<192x512x1x1xf16>
    %3 = IE.Slice %1 [0, 512, 0, 0] [192, 512, 1, 1] : tensor<192x1024x1x1xf16> to tensor<192x512x1x1xf16>
    %4 = IE.ReLU(%2) : tensor<192x512x1x1xf16> -> tensor<192x512x1x1xf16>
    return %4, %3 : tensor<192x512x1x1xf16>, tensor<192x512x1x1xf16>

    // CHECK:   [[W_SLICE0:%.+]] = const.Declare tensor<1x6x512x16xf16> = dense<1.000000e+00> : tensor<1x6x1024x16xf16>, [#const.SubView<[0, 0, 512, 0], [1, 6, 512, 16]>]
    // CHECK:       [[MATMUL0:%.+]] = IE.MatMul([[ARG_0]], [[W_SLICE0]]) {transpose_b} : tensor<1x6x32x16xf16>, tensor<1x6x512x16xf16> -> tensor<1x6x32x512xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[MATMUL0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [192, 512, 1, 1]} : tensor<1x6x32x512xf16> -> tensor<192x512x1x1xf16>

    // CHECK:   [[W_SLICE1:%.+]] = const.Declare tensor<1x6x512x16xf16> = dense<1.000000e+00> : tensor<1x6x1024x16xf16>, [#const.SubView<[0, 0, 0, 0], [1, 6, 512, 16]>]
    // CHECK:       [[MATMUL1:%.+]] = IE.MatMul([[ARG_0]], [[W_SLICE1]]) {transpose_b} : tensor<1x6x32x16xf16>, tensor<1x6x512x16xf16> -> tensor<1x6x32x512xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[MATMUL1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [192, 512, 1, 1]} : tensor<1x6x32x512xf16> -> tensor<192x512x1x1xf16>

    // CHECK:       [[ACTIVATION:%.+]] = IE.ReLU([[RESHAPE1]])

    // CHECK:       return [[ACTIVATION]], [[RESHAPE0]]
}

// -----

// CHECK-LABEL: @FuseMatMulWithSliceTransposeA
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<16x32xf16>
func.func @FuseMatMulWithSliceTransposeA(%arg0: tensor<16x32xf16>) -> tensor<32x32xf16> {
    %weights = const.Declare tensor<16x64xf16> = dense<1.0> : tensor<16x64xf16>
    %0 = IE.MatMul(%arg0, %weights) {transpose_a} : tensor<16x32xf16>, tensor<16x64xf16> -> tensor<32x64xf16>
    %1 = IE.Slice %0 [0, 0] [32, 32] : tensor<32x64xf16> to tensor<32x32xf16>
    return %1 : tensor<32x32xf16>

    // CHECK-DAG:  [[WEIGHTS:%.+]] = const.Declare tensor<16x32xf16> = dense<1.000000e+00> : tensor<16x64xf16>, [#const.SubView<[0, 0], [16, 32]>]
    // CHECK:      [[MATMUL:%.+]] = IE.MatMul([[ARG_0]], [[WEIGHTS]]) {transpose_a} : tensor<16x32xf16>, tensor<16x32xf16> -> tensor<32x32xf16>
    // CHECK:      return      [[MATMUL]]
}

// -----

// CHECK-LABEL: @ViewOpsNoFinalSlice
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<8x16xf16>
func.func @ViewOpsNoFinalSlice(%arg0: tensor<8x16xf16>) -> tensor<8x1024x1x1xf16> {
    %weights = const.Declare tensor<1024x16xf16> = dense<1.0> : tensor<1024x16xf16>
    %0 = IE.FullyConnected(%arg0, %weights) : tensor<8x16xf16>, tensor<1024x16xf16> -> tensor<8x1024xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1, 2, 3]], shape_value = [8, 1024, 1, 1]} : tensor<8x1024xf16> -> tensor<8x1024x1x1xf16>
    return %1 : tensor<8x1024x1x1xf16>

    // CHECK:   [[WEIGHTS:%.+]] = const.Declare tensor<1024x16xf16> = dense<1.000000e+00> : tensor<1024x16xf16>
    // CHECK:   [[FC:%.+]] = IE.FullyConnected([[ARG_0]], [[WEIGHTS]])
    // CHECK-SAME:     : tensor<8x16xf16>, tensor<1024x16xf16> -> tensor<8x1024xf16>
    // CHECK:   [[RESHAPE:%.+]] = IE.AffineReshape([[FC]])

    // CHECK:   return [[RESHAPE]]
}

// -----

// CHECK-LABEL: @FuseFullyConnectedWithSliceNoChainOnOneBranch
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<32x16xf16>
func.func @FuseFullyConnectedWithSliceNoChainOnOneBranch(%arg0: tensor<32x16xf16>) -> (tensor<32x512xf16>, tensor<32x512x1x1xf16>) {
    %weights = const.Declare tensor<1024x16xf16> = dense<1.0> : tensor<1024x16xf16>
    %0 = IE.FullyConnected(%arg0, %weights) : tensor<32x16xf16>, tensor<1024x16xf16> -> tensor<32x1024xf16>
    %1 = IE.Slice %0 [0, 0] [32, 512] : tensor<32x1024xf16> to tensor<32x512xf16>
    %2 = IE.AffineReshape(%0) {dim_mapping = [[0], [1, 2, 3]], shape_value = [32, 1024, 1, 1]} : tensor<32x1024xf16> -> tensor<32x1024x1x1xf16>
    %3 = IE.Slice %2 [0, 512, 0, 0] [32, 512, 1, 1] : tensor<32x1024x1x1xf16> to tensor<32x512x1x1xf16>
    %4 = IE.ReLU(%3) : tensor<32x512x1x1xf16> -> tensor<32x512x1x1xf16>
    return %1, %4 : tensor<32x512xf16>, tensor<32x512x1x1xf16>

    // CHECK:   [[W_SLICE1:%.+]] = const.Declare tensor<512x16xf16> = dense<1.000000e+00> : tensor<1024x16xf16>, [#const.SubView<[512, 0], [512, 16]>]
    // CHECK:   [[FC1:%.+]] = IE.FullyConnected([[ARG_0]], [[W_SLICE1]]) : tensor<32x16xf16>, tensor<512x16xf16> -> tensor<32x512xf16>
    // CHECK:   [[RESHAPE1:%.+]] = IE.AffineReshape([[FC1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2, 3]], shape_value = [32, 512, 1, 1]} : tensor<32x512xf16> -> tensor<32x512x1x1xf16>

    // CHECK:   [[W_SLICE0:%.+]] = const.Declare tensor<512x16xf16> = dense<1.000000e+00> : tensor<1024x16xf16>, [#const.SubView<[0, 0], [512, 16]>]
    // CHECK:   [[FC0:%.+]] = IE.FullyConnected([[ARG_0]], [[W_SLICE0]]) : tensor<32x16xf16>, tensor<512x16xf16> -> tensor<32x512xf16>

    // CHECK:   [[ACTIVATION:%.+]] = IE.ReLU([[RESHAPE1]])

    // CHECK:   return [[FC0]], [[ACTIVATION]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseMatMulWithSliceMismatchedViewChainTypes
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x6x32x16xf16>
func.func @FuseMatMulWithSliceMismatchedViewChainTypes(%arg0: tensor<1x6x32x16xf16>) -> (tensor<192x512x1x1xf16>, tensor<1x512x6x32xf16, {order = #NHWC}>) {
    %weights = const.Declare tensor<1x6x1024x16xf16> = dense<1.0> : tensor<1x6x1024x16xf16>
    %0 = IE.MatMul(%arg0, %weights) {transpose_b} : tensor<1x6x32x16xf16>, tensor<1x6x1024x16xf16> -> tensor<1x6x32x1024xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [192, 1024, 1, 1]} : tensor<1x6x32x1024xf16> -> tensor<192x1024x1x1xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [192, 512, 1, 1] : tensor<192x1024x1x1xf16> to tensor<192x512x1x1xf16>
    %3 = IE.PermuteCast(%0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x6x32x1024xf16> -> tensor<1x1024x6x32xf16, {order = #NHWC}>
    %4 = IE.Slice %3 [0, 512, 0, 0] [1, 512, 6, 32] : tensor<1x1024x6x32xf16, {order = #NHWC}> to tensor<1x512x6x32xf16, {order = #NHWC}>
    %5 = IE.ReLU(%2) : tensor<192x512x1x1xf16> -> tensor<192x512x1x1xf16>
    return %5, %4 : tensor<192x512x1x1xf16>, tensor<1x512x6x32xf16, {order = #NHWC}>

    // CHECK:   [[W_SLICE_PC:%.+]] = const.Declare tensor<1x6x512x16xf16> = dense<1.000000e+00> : tensor<1x6x1024x16xf16>, [#const.SubView<[0, 0, 512, 0], [1, 6, 512, 16]>]
    // CHECK:   [[MATMUL_PC:%.+]] = IE.MatMul([[ARG_0]], [[W_SLICE_PC]]) {transpose_b} : tensor<1x6x32x16xf16>, tensor<1x6x512x16xf16> -> tensor<1x6x32x512xf16>
    // CHECK:   [[PERMUTECAST:%.+]] = IE.PermuteCast([[MATMUL_PC]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x6x32x512xf16> -> tensor<1x512x6x32xf16, {order = #NHWC}>

    // CHECK:   [[W_SLICE_AR:%.+]] = const.Declare tensor<1x6x512x16xf16> = dense<1.000000e+00> : tensor<1x6x1024x16xf16>, [#const.SubView<[0, 0, 0, 0], [1, 6, 512, 16]>]
    // CHECK:   [[MATMUL_AR:%.+]] = IE.MatMul([[ARG_0]], [[W_SLICE_AR]]) {transpose_b} : tensor<1x6x32x16xf16>, tensor<1x6x512x16xf16> -> tensor<1x6x32x512xf16>
    // CHECK:   [[RESHAPE_AR:%.+]] = IE.AffineReshape([[MATMUL_AR]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [192, 512, 1, 1]} : tensor<1x6x32x512xf16> -> tensor<192x512x1x1xf16>

    // CHECK:   [[ACTIVATION:%.+]] = IE.ReLU([[RESHAPE_AR]])

    // CHECK:   return [[ACTIVATION]], [[PERMUTECAST]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotFuseWithOverlapAcrossDifferentViewChains
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x6x32x16xf16>
func.func @NotFuseWithOverlapAcrossDifferentViewChains(%arg0: tensor<1x6x32x16xf16>) -> (tensor<192x512x1x1xf16>, tensor<1x512x6x32xf16, {order = #NHWC}>) {
    %weights = const.Declare tensor<1x6x1024x16xf16> = dense<1.0> : tensor<1x6x1024x16xf16>
    %0 = IE.MatMul(%arg0, %weights) {transpose_b} : tensor<1x6x32x16xf16>, tensor<1x6x1024x16xf16> -> tensor<1x6x32x1024xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [192, 1024, 1, 1]} : tensor<1x6x32x1024xf16> -> tensor<192x1024x1x1xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [192, 512, 1, 1] : tensor<192x1024x1x1xf16> to tensor<192x512x1x1xf16>
    %3 = IE.PermuteCast(%0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x6x32x1024xf16> -> tensor<1x1024x6x32xf16, {order = #NHWC}>
    %4 = IE.Slice %3 [0, 256, 0, 0] [1, 512, 6, 32] : tensor<1x1024x6x32xf16, {order = #NHWC}> to tensor<1x512x6x32xf16, {order = #NHWC}>
    %5 = IE.ReLU(%2) : tensor<192x512x1x1xf16> -> tensor<192x512x1x1xf16>
    return %5, %4 : tensor<192x512x1x1xf16>, tensor<1x512x6x32xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<1x6x1024x16xf16> = dense<1.000000e+00> : tensor<1x6x1024x16xf16>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[ARG_0]], [[WEIGHTS]]) {transpose_b} : tensor<1x6x32x16xf16>, tensor<1x6x1024x16xf16> -> tensor<1x6x32x1024xf16>
    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[MATMUL]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [192, 1024, 1, 1]} : tensor<1x6x32x1024xf16> -> tensor<192x1024x1x1xf16>
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[RESHAPE]] [0, 0, 0, 0] [192, 512, 1, 1] : tensor<192x1024x1x1xf16> to tensor<192x512x1x1xf16>
    // CHECK:       [[PERMUTECAST:%.+]] = IE.PermuteCast([[MATMUL]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x6x32x1024xf16> -> tensor<1x1024x6x32xf16, {order = #NHWC}>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[PERMUTECAST]] [0, 256, 0, 0] [1, 512, 6, 32] : tensor<1x1024x6x32xf16, {order = #NHWC}> to tensor<1x512x6x32xf16, {order = #NHWC}>
    // CHECK:       [[ACTIVATION:%.+]] = IE.ReLU([[SLICE0]])

    // CHECK:       return [[ACTIVATION]], [[SLICE1]]
}

// -----

// CHECK-LABEL: @FuseFullyConnectedWithSliceAndDynamicDequantizeWeights
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<32x16xf16>
// CHECK-SAME:      [[WEIGHTS:%[^:]+]]: tensor<1024x16xsi4>
// CHECK-SAME:      [[SCALE:%[^:]+]]: tensor<1024x1xf16>
func.func @FuseFullyConnectedWithSliceAndDynamicDequantizeWeights(%arg0: tensor<32x16xf16>, %weights: tensor<1024x16xsi4>, %scale: tensor<1024x1xf16>)
            -> (tensor<32x512xf16>, tensor<32x512xf16>) {
    %0 = IE.DynamicDequantize(%weights, %scale) {dstElemType = f16} : tensor<1024x16xsi4>, tensor<1024x1xf16> -> tensor<1024x16xf16>
    %1 = IE.FullyConnected(%arg0, %0) : tensor<32x16xf16>, tensor<1024x16xf16> -> tensor<32x1024xf16>
    %2 = IE.Slice %1 [0, 0] [32, 512] : tensor<32x1024xf16> to tensor<32x512xf16>
    %3 = IE.Slice %1 [0, 512] [32, 512] : tensor<32x1024xf16> to tensor<32x512xf16>
    %4 = IE.ReLU(%2) : tensor<32x512xf16> -> tensor<32x512xf16>
    return %4, %3 : tensor<32x512xf16>, tensor<32x512xf16>

    // CHECK-DAG:   [[W_SLICE0:%.+]] = IE.Slice [[WEIGHTS]] [0, 0] [512, 16] : tensor<1024x16xsi4> to tensor<512x16xsi4>
    // CHECK-DAG:   [[SCALE_SLICE0:%.+]] = IE.Slice [[SCALE]] [0, 0] [512, 1] : tensor<1024x1xf16> to tensor<512x1xf16>
    // CHECK:       [[DEQUANT0:%.+]] = IE.DynamicDequantize([[W_SLICE0]], [[SCALE_SLICE0]]) {dstElemType = f16}
    // CHECK-SAME:              : tensor<512x16xsi4>, tensor<512x1xf16> -> tensor<512x16xf16>

    // CHECK-DAG:   [[W_SLICE1:%.+]] = IE.Slice [[WEIGHTS]] [512, 0] [512, 16] : tensor<1024x16xsi4> to tensor<512x16xsi4>
    // CHECK-DAG:   [[SCALE_SLICE1:%.+]] = IE.Slice [[SCALE]] [512, 0] [512, 1] : tensor<1024x1xf16> to tensor<512x1xf16>
    // CHECK:       [[DEQUANT1:%.+]] = IE.DynamicDequantize([[W_SLICE1]], [[SCALE_SLICE1]]) {dstElemType = f16}
    // CHECK-SAME:              : tensor<512x16xsi4>, tensor<512x1xf16> -> tensor<512x16xf16>

    // CHECK:       [[FC1:%.+]] = IE.FullyConnected([[ARG_0]], [[DEQUANT1]]) : tensor<32x16xf16>, tensor<512x16xf16> -> tensor<32x512xf16>
    // CHECK:       [[FC0:%.+]] = IE.FullyConnected([[ARG_0]], [[DEQUANT0]]) : tensor<32x16xf16>, tensor<512x16xf16> -> tensor<32x512xf16>

    // CHECK:       [[ACTIVATION:%.+]] = IE.ReLU([[FC0]])

    // CHECK:       return [[ACTIVATION]], [[FC1]]
}

// -----

// CHECK-LABEL: @FuseFullyConnectedWithSliceAndDynamicDequantizeWeightsWithZeroPoint
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<32x16xf16>
// CHECK-SAME:      [[WEIGHTS:%[^:]+]]: tensor<1024x16xsi4>
// CHECK-SAME:      [[SCALE:%[^:]+]]: tensor<1024x1xf16>
// CHECK-SAME:      [[ZP:%[^:]+]]: tensor<1024x1xsi4>
func.func @FuseFullyConnectedWithSliceAndDynamicDequantizeWeightsWithZeroPoint(%arg0: tensor<32x16xf16>, %weights: tensor<1024x16xsi4>, %scale: tensor<1024x1xf16>, %zp: tensor<1024x1xsi4>)
            -> (tensor<32x512xf16>, tensor<32x512xf16>) {
    %0 = IE.DynamicDequantize(%weights, %scale, %zp) {dstElemType = f16} : tensor<1024x16xsi4>, tensor<1024x1xf16>, tensor<1024x1xsi4> -> tensor<1024x16xf16>
    %1 = IE.FullyConnected(%arg0, %0) : tensor<32x16xf16>, tensor<1024x16xf16> -> tensor<32x1024xf16>
    %2 = IE.Slice %1 [0, 0] [32, 512] : tensor<32x1024xf16> to tensor<32x512xf16>
    %3 = IE.Slice %1 [0, 512] [32, 512] : tensor<32x1024xf16> to tensor<32x512xf16>
    %4 = IE.ReLU(%2) : tensor<32x512xf16> -> tensor<32x512xf16>
    return %4, %3 : tensor<32x512xf16>, tensor<32x512xf16>

    // CHECK-DAG:   [[W_SLICE0:%.+]] = IE.Slice [[WEIGHTS]] [0, 0] [512, 16] : tensor<1024x16xsi4> to tensor<512x16xsi4>
    // CHECK-DAG:   [[SCALE_SLICE0:%.+]] = IE.Slice [[SCALE]] [0, 0] [512, 1] : tensor<1024x1xf16> to tensor<512x1xf16>
    // CHECK-DAG:   [[ZP_SLICE0:%.+]] = IE.Slice [[ZP]] [0, 0] [512, 1] : tensor<1024x1xsi4> to tensor<512x1xsi4>
    // CHECK:       [[DEQUANT0:%.+]] = IE.DynamicDequantize([[W_SLICE0]], [[SCALE_SLICE0]], [[ZP_SLICE0]]) {dstElemType = f16}
    // CHECK-SAME:              : tensor<512x16xsi4>, tensor<512x1xf16>, tensor<512x1xsi4> -> tensor<512x16xf16>

    // CHECK-DAG:   [[W_SLICE1:%.+]] = IE.Slice [[WEIGHTS]] [512, 0] [512, 16] : tensor<1024x16xsi4> to tensor<512x16xsi4>
    // CHECK-DAG:   [[SCALE_SLICE1:%.+]] = IE.Slice [[SCALE]] [512, 0] [512, 1] : tensor<1024x1xf16> to tensor<512x1xf16>
    // CHECK-DAG:   [[ZP_SLICE1:%.+]] = IE.Slice [[ZP]] [512, 0] [512, 1] : tensor<1024x1xsi4> to tensor<512x1xsi4>
    // CHECK:       [[DEQUANT1:%.+]] = IE.DynamicDequantize([[W_SLICE1]], [[SCALE_SLICE1]], [[ZP_SLICE1]]) {dstElemType = f16}
    // CHECK-SAME:              : tensor<512x16xsi4>, tensor<512x1xf16>, tensor<512x1xsi4> -> tensor<512x16xf16>

    // CHECK:       [[FC1:%.+]] = IE.FullyConnected([[ARG_0]], [[DEQUANT1]]) : tensor<32x16xf16>, tensor<512x16xf16> -> tensor<32x512xf16>
    // CHECK:       [[FC0:%.+]] = IE.FullyConnected([[ARG_0]], [[DEQUANT0]]) : tensor<32x16xf16>, tensor<512x16xf16> -> tensor<32x512xf16>

    // CHECK:       [[ACTIVATION:%.+]] = IE.ReLU([[FC0]])

    // CHECK:       return [[ACTIVATION]], [[FC1]]
}

// -----

// CHECK-LABEL: @FuseFullyConnectedWithSliceAndDynamicDequantizeConstWeights
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<32x16xf16>
func.func @FuseFullyConnectedWithSliceAndDynamicDequantizeConstWeights(%arg0: tensor<32x16xf16>)
            -> (tensor<32x512xf16>, tensor<32x512xf16>) {
    %weights = const.Declare tensor<1024x16xsi4> = dense<1> : tensor<1024x16xsi4>
    %scale = const.Declare tensor<1024x1xf16> = dense<1.0> : tensor<1024x1xf16>
    %0 = IE.DynamicDequantize(%weights, %scale) {dstElemType = f16} : tensor<1024x16xsi4>, tensor<1024x1xf16> -> tensor<1024x16xf16>
    %1 = IE.FullyConnected(%arg0, %0) : tensor<32x16xf16>, tensor<1024x16xf16> -> tensor<32x1024xf16>
    %2 = IE.Slice %1 [0, 0] [32, 512] : tensor<32x1024xf16> to tensor<32x512xf16>
    %3 = IE.Slice %1 [0, 512] [32, 512] : tensor<32x1024xf16> to tensor<32x512xf16>
    %4 = IE.ReLU(%2) : tensor<32x512xf16> -> tensor<32x512xf16>
    return %4, %3 : tensor<32x512xf16>, tensor<32x512xf16>

    // CHECK-DAG:   [[W_SLICE0:%.+]] = const.Declare tensor<512x16xsi4> = dense<1> : tensor<1024x16xsi4>, [#const.SubView<[0, 0], [512, 16]>]
    // CHECK-DAG:   [[SCALE_SLICE0:%.+]] = const.Declare tensor<512x1xf16> = dense<1.000000e+00> : tensor<1024x1xf16>, [#const.SubView<[0, 0], [512, 1]>]
    // CHECK:       [[DEQUANT0:%.+]] = IE.DynamicDequantize([[W_SLICE0]], [[SCALE_SLICE0]]) {dstElemType = f16}
    // CHECK-SAME:              : tensor<512x16xsi4>, tensor<512x1xf16> -> tensor<512x16xf16>

    // CHECK-DAG:   [[SCALE_SLICE1:%.+]] = const.Declare tensor<512x1xf16> = dense<1.000000e+00> : tensor<1024x1xf16>, [#const.SubView<[512, 0], [512, 1]>]
    // CHECK-DAG:   [[W_SLICE1:%.+]] = const.Declare tensor<512x16xsi4> = dense<1> : tensor<1024x16xsi4>, [#const.SubView<[512, 0], [512, 16]>]
    // CHECK:       [[DEQUANT1:%.+]] = IE.DynamicDequantize([[W_SLICE1]], [[SCALE_SLICE1]]) {dstElemType = f16}
    // CHECK-SAME:              : tensor<512x16xsi4>, tensor<512x1xf16> -> tensor<512x16xf16>

    // CHECK:       [[FC1:%.+]] = IE.FullyConnected([[ARG_0]], [[DEQUANT1]]) : tensor<32x16xf16>, tensor<512x16xf16> -> tensor<32x512xf16>
    // CHECK:       [[FC0:%.+]] = IE.FullyConnected([[ARG_0]], [[DEQUANT0]]) : tensor<32x16xf16>, tensor<512x16xf16> -> tensor<32x512xf16>

    // CHECK:       [[ACTIVATION:%.+]] = IE.ReLU([[FC0]])

    // CHECK:       return [[ACTIVATION]], [[FC1]]
}

// -----

// CHECK-LABEL: @FuseThreewaySliceWithBeneficialPostOpFusableAmount
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<32x16xf16>
func.func @FuseThreewaySliceWithBeneficialPostOpFusableAmount(%arg0: tensor<32x16xf16>)
            -> (tensor<32x256xf16>, tensor<32x256xf16>,tensor<32x512xf16>) {
    %weights = const.Declare tensor<1024x16xsi4> = dense<1> : tensor<1024x16xsi4>
    %scale = const.Declare tensor<1024x1xf16> = dense<1.0> : tensor<1024x1xf16>
    %0 = IE.DynamicDequantize(%weights, %scale) {dstElemType = f16} : tensor<1024x16xsi4>, tensor<1024x1xf16> -> tensor<1024x16xf16>
    %1 = IE.FullyConnected(%arg0, %0) : tensor<32x16xf16>, tensor<1024x16xf16> -> tensor<32x1024xf16>
    %2 = IE.Slice %1 [0, 0] [32, 512] : tensor<32x1024xf16> to tensor<32x512xf16>
    %3 = IE.Slice %1 [0, 512] [32, 256] : tensor<32x1024xf16> to tensor<32x256xf16>
    %4 = IE.Slice %1 [0, 768] [32, 256] : tensor<32x1024xf16> to tensor<32x256xf16>
    %5 = IE.ReLU(%2) : tensor<32x512xf16> -> tensor<32x512xf16>
    return %3, %4, %5 : tensor<32x256xf16>, tensor<32x256xf16>,tensor<32x512xf16>

    // CHECK-DAG:   [[W_SLICE0:%.+]] = const.Declare tensor<512x16xsi4> = dense<1> : tensor<1024x16xsi4>, [#const.SubView<[0, 0], [512, 16]>]
    // CHECK-DAG:   [[SCALE_SLICE0:%.+]] = const.Declare tensor<512x1xf16> = dense<1.000000e+00> : tensor<1024x1xf16>, [#const.SubView<[0, 0], [512, 1]>]
    // CHECK:       [[DEQUANT0:%.+]] = IE.DynamicDequantize([[W_SLICE0]], [[SCALE_SLICE0]]) {dstElemType = f16}
    // CHECK-SAME:              : tensor<512x16xsi4>, tensor<512x1xf16> -> tensor<512x16xf16>

    // CHECK-DAG:   [[SCALE_SLICE1:%.+]] = const.Declare tensor<256x1xf16> = dense<1.000000e+00> : tensor<1024x1xf16>, [#const.SubView<[512, 0], [256, 1]>]
    // CHECK-DAG:   [[W_SLICE1:%.+]] = const.Declare tensor<256x16xsi4> = dense<1> : tensor<1024x16xsi4>, [#const.SubView<[512, 0], [256, 16]>]
    // CHECK:       [[DEQUANT1:%.+]] = IE.DynamicDequantize([[W_SLICE1]], [[SCALE_SLICE1]]) {dstElemType = f16}
    // CHECK-SAME:              : tensor<256x16xsi4>, tensor<256x1xf16> -> tensor<256x16xf16>

    // CHECK-DAG:   [[SCALE_SLICE2:%.+]] = const.Declare tensor<256x1xf16> = dense<1.000000e+00> : tensor<1024x1xf16>, [#const.SubView<[768, 0], [256, 1]>]
    // CHECK-DAG:   [[W_SLICE2:%.+]] = const.Declare tensor<256x16xsi4> = dense<1> : tensor<1024x16xsi4>, [#const.SubView<[768, 0], [256, 16]>]
    // CHECK:       [[DEQUANT2:%.+]] = IE.DynamicDequantize([[W_SLICE2]], [[SCALE_SLICE2]]) {dstElemType = f16}
    // CHECK-SAME:              : tensor<256x16xsi4>, tensor<256x1xf16> -> tensor<256x16xf16>

    // CHECK:       [[FC2:%.+]] = IE.FullyConnected([[ARG_0]], [[DEQUANT2]]) : tensor<32x16xf16>, tensor<256x16xf16> -> tensor<32x256xf16>
    // CHECK:       [[FC1:%.+]] = IE.FullyConnected([[ARG_0]], [[DEQUANT1]]) : tensor<32x16xf16>, tensor<256x16xf16> -> tensor<32x256xf16>
    // CHECK:       [[FC0:%.+]] = IE.FullyConnected([[ARG_0]], [[DEQUANT0]]) : tensor<32x16xf16>, tensor<512x16xf16> -> tensor<32x512xf16>

    // CHECK:       [[ACTIVATION:%.+]] = IE.ReLU([[FC0]])

    // CHECK:       return [[FC1]], [[FC2]], [[ACTIVATION]]
}

// -----

// CHECK-LABEL: @NotFusePostOpFusableAmountTooSmall
func.func @NotFusePostOpFusableAmountTooSmall(%arg0: tensor<32x16xf16>)
            -> (tensor<32x256xf16>, tensor<32x768xf16>) {
    %weights = const.Declare tensor<1024x16xsi4> = dense<1> : tensor<1024x16xsi4>
    %scale = const.Declare tensor<1024x1xf16> = dense<1.0> : tensor<1024x1xf16>
    %0 = IE.DynamicDequantize(%weights, %scale) {dstElemType = f16} : tensor<1024x16xsi4>, tensor<1024x1xf16> -> tensor<1024x16xf16>
    %1 = IE.FullyConnected(%arg0, %0) : tensor<32x16xf16>, tensor<1024x16xf16> -> tensor<32x1024xf16>
    %2 = IE.Slice %1 [0, 0] [32, 256] : tensor<32x1024xf16> to tensor<32x256xf16>
    %3 = IE.Slice %1 [0, 256] [32, 768] : tensor<32x1024xf16> to tensor<32x768xf16>
    %4 = IE.ReLU(%2) : tensor<32x256xf16> -> tensor<32x256xf16>
    return %4, %3 : tensor<32x256xf16>, tensor<32x768xf16>

    // CHECK:       [[FC:%.+]] = IE.FullyConnected
    // CHECK-SAME:                  -> tensor<32x1024xf16>
    // CHECK-DAG:   [[SLICE0:%.+]] = IE.Slice [[FC]] [0, 0] [32, 256] : tensor<32x1024xf16> to tensor<32x256xf16>
    // CHECK-DAG:   [[SLICE1:%.+]] = IE.Slice [[FC]] [0, 256] [32, 768] : tensor<32x1024xf16> to tensor<32x768xf16>

    // CHECK:       [[ACTIVATION:%.+]] = IE.ReLU([[SLICE0]])

    // CHECK:       return [[ACTIVATION]], [[SLICE1]]
}
