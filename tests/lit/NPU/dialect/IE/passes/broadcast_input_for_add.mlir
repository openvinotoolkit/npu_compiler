//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --broadcast-input-for-add  %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @BroadcastTensorInput
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x16x16x32xf16>
// CHECK-SAME:    [[ARG_1:%[^:]+]]: tensor<1x16x16x1xf16>
func.func @BroadcastTensorInput(%arg0: tensor<1x16x16x32xf16>, %arg1: tensor<1x16x16x1xf16>) -> tensor<1x16x16x32xf16> {
    %0 = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x32xf16>, tensor<1x16x16x1xf16> -> tensor<1x16x16x32xf16>

    return %0 : tensor<1x16x16x32xf16>

    // CHECK-DAG:   [[TARGET_SHAPE:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 16, 16, 32]> : tensor<4xsi64>, [#const.CastElemType<si32>]
    // CHECK:       [[BROADCAST:%.+]] = IE.Broadcast([[ARG_1]], [[TARGET_SHAPE]])
    // CHECK-SAME:      {mode = #IE.broadcast_type<NUMPY>} : tensor<1x16x16x1xf16>, tensor<4xsi32> -> tensor<1x16x16x32xf16>
    // CHECK:       [[ADD_RES:%.+]] = IE.Add([[ARG_0]], [[BROADCAST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x32xf16>, tensor<1x16x16x32xf16> -> tensor<1x16x16x32xf16>

    // CHECK:       return [[ADD_RES]]
}

// -----

// CHECK-LABEL: @BroadcastSingleNonTrivialDimInput
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x16x16x32xf16>
// CHECK-SAME:    [[ARG_1:%[^:]+]]: tensor<1x16x1x1xf16>
func.func @BroadcastSingleNonTrivialDimInput(%arg0: tensor<1x16x16x32xf16>, %arg1: tensor<1x16x1x1xf16>) -> tensor<1x16x16x32xf16> {
    %0 = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x32xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x32xf16>

    return %0 : tensor<1x16x16x32xf16>

    // CHECK-DAG:   [[TARGET_SHAPE:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 16, 16, 32]> : tensor<4xsi64>, [#const.CastElemType<si32>]
    // CHECK:       [[BROADCAST:%.+]] = IE.Broadcast([[ARG_1]], [[TARGET_SHAPE]])
    // CHECK-SAME:      {mode = #IE.broadcast_type<NUMPY>} : tensor<1x16x1x1xf16>, tensor<4xsi32> -> tensor<1x16x16x32xf16>
    // CHECK:       [[ADD_RES:%.+]] = IE.Add([[ARG_0]], [[BROADCAST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x32xf16>, tensor<1x16x16x32xf16> -> tensor<1x16x16x32xf16>

    // CHECK:       return [[ADD_RES]]
}

// -----

// CHECK-LABEL: @BroadcastTwoInputs
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x16x1x32xf16>
// CHECK-SAME:    [[ARG_1:%[^:]+]]: tensor<1x16x16x1xf16>
func.func @BroadcastTwoInputs(%arg0: tensor<1x16x1x32xf16>, %arg1: tensor<1x16x16x1xf16>) -> tensor<1x16x16x32xf16> {
    %0 = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x32xf16>, tensor<1x16x16x1xf16> -> tensor<1x16x16x32xf16>

    return %0 : tensor<1x16x16x32xf16>

    // CHECK-DAG:   [[TARGET_SHAPE0:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 16, 16, 32]> : tensor<4xsi64>, [#const.CastElemType<si32>]
    // CHECK:       [[BROADCAST0:%.+]] = IE.Broadcast([[ARG_0]], [[TARGET_SHAPE0]])
    // CHECK-SAME:      {mode = #IE.broadcast_type<NUMPY>} : tensor<1x16x1x32xf16>, tensor<4xsi32> -> tensor<1x16x16x32xf16>
    // CHECK-DAG:   [[TARGET_SHAPE1:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 16, 16, 32]> : tensor<4xsi64>, [#const.CastElemType<si32>]
    // CHECK:       [[BROADCAST1:%.+]] = IE.Broadcast([[ARG_1]], [[TARGET_SHAPE1]])
    // CHECK-SAME:      {mode = #IE.broadcast_type<NUMPY>} : tensor<1x16x16x1xf16>, tensor<4xsi32> -> tensor<1x16x16x32xf16>
    // CHECK:       [[ADD_RES:%.+]] = IE.Add([[BROADCAST0]], [[BROADCAST1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x32xf16>, tensor<1x16x16x32xf16> -> tensor<1x16x16x32xf16>

    // CHECK:       return [[ADD_RES]]
}

// -----

// CHECK-LABEL: @BroadcastFlatConstantInput
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x16x16x32xf16>
func.func @BroadcastFlatConstantInput(%arg0: tensor<1x16x16x32xf16>) -> tensor<1x16x16x32xf16> {
    %0 = const.Declare tensor<1x16x16x1xf16> = dense<1.0> : tensor<1x16x16x1xf16>, [#const.CastElemType<f16>]
    %1 = IE.Add(%arg0, %0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x32xf16>, tensor<1x16x16x1xf16> -> tensor<1x16x16x32xf16>

    return %1 : tensor<1x16x16x32xf16>

    // CHECK-DAG:   [[BROADCAST:%.+]] = const.Declare tensor<1x16x16x32xf16> = dense<1.000000e+00> : tensor<1x16x16x1xf16>,
    // CHECK-SAME:                            [#const.CastElemType<f16>, #const.Broadcast<3 : i64, 32 : i64>]
    // CHECK:       [[ADD_RES:%.+]] = IE.Add([[ARG_0]], [[BROADCAST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x32xf16>, tensor<1x16x16x32xf16> -> tensor<1x16x16x32xf16>

    // CHECK:       return [[ADD_RES]]
}

// -----

// CHECK-LABEL: @BroadcastNonFlatConstantInput
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x16x4x4xf16>
func.func @BroadcastNonFlatConstantInput(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16> {
    %0 = const.Declare tensor<1x1x1x4xf16> = dense<[[[[1.0, 2.0, 3.0, 4.0]]]]> : tensor<1x1x1x4xf16>, [#const.CastElemType<f16>]
    %1 = IE.Add(%arg0, %0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x4x4xf16>, tensor<1x1x1x4xf16> -> tensor<1x16x4x4xf16>

    return %1 : tensor<1x16x4x4xf16>

    // CHECK:       [[BROADCAST:%.+]] = const.Declare tensor<1x16x4x4xf16>
    // CHECK-SAME{LITERAL}:      dense<[[[[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]]]> : tensor<1x1x1x4xf16>,
    // CHECK-SAME:               [#const.CastElemType<f16>, #const.Broadcast<1 : i64, 16 : i64>, #const.Broadcast<2 : i64, 4 : i64>]
    // CHECK:       [[ADD_RES:%.+]] = IE.Add([[ARG_0]], [[BROADCAST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>

    // CHECK:       return [[ADD_RES]]
}

// -----

// CHECK-LABEL: @BroadcastFQAddConstInput
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x3x30x30xf16>
func.func @BroadcastFQAddConstInput(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x30x30xf16> {
    %CST = const.Declare tensor<1x1x30x30xf16> = dense<1.0> : tensor<1x1x30x30xf16>, [#const.CastElemType<f16>]
    %val_low = const.Declare tensor<1x1x1x1xf16> = dense<4.0> : tensor<1x1x1x1xf16>
    %val_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>

    %relu = IE.ReLU(%arg0) : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    %fq_1 = IE.FakeQuantize(%relu, %val_low, %val_high, %val_low, %val_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>

    %add = IE.Add(%fq_1, %CST)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x3x30x30xf16>

    return %add : tensor<1x3x30x30xf16>

    // CHECK:       [[VAL_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:       [[VAL_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf16>
    // CHECK:       [[RELU:%.+]] = IE.ReLU([[ARG_0]]) : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       [[FQ_1:%.+]] = IE.FakeQuantize([[RELU]], [[VAL_LOW]], [[VAL_HIGH]], [[VAL_LOW]], [[VAL_HIGH]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:       tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x3x30x30xf16> = dense<1.000000e+00> : tensor<1x1x30x30xf16>, [#const.CastElemType<f16>, #const.Broadcast<1 : i64, 3 : i64>]
    // CHECK:       [[ADD:%.+]] = IE.Add([[FQ_1]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x30x30xf16>, tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       return [[ADD]] : tensor<1x3x30x30xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!dynType = tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 1, 128]> : tensor<4xsi64>, order = #NCHW}>

// CHECK-LABEL: func.func @LhsDynamicBroadcast
// CHECK-SAME:  [[INPUT_0:%.+]]: tensor<1x1x1x128xf16>
// CHECK-SAME:  [[INPUT_1:%.+]]: tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 1, 128]> : tensor<4xsi64>, order = #NCHW}>
func.func @LhsDynamicBroadcast(%arg0: tensor<1x1x1x128xf16>, %arg1: !dynType) -> !dynType {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x128xf16>, !dynType -> !dynType
    return %0 : !dynType

    // CHECK:       [[SHAPE:%.+]] = IE.ShapeOf([[INPUT_1]]) {dstElemType = si32} :
    // CHECK-SAME:    tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 1, 128]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi32>
    // CHECK:       [[BCAST:%.+]] = IE.DynamicBroadcast([[INPUT_0]], [[SHAPE]]) {
    // CHECK-SAME:    mode = #IE.broadcast_type<NUMPY>, output_bounds = [1, 500, 1, 128], output_shape = [1, -9223372036854775808, 1, 128]} : tensor<1x1x1x128xf16>, tensor<4xsi32> -> tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 1, 128]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[BCAST]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME:    tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 1, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 1, 128]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 1, 128]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:  return [[ADD]] : tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 1, 128]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!dynType1 = tensor<1x?x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 16, 128]> : tensor<4xsi64>, order = #NCHW}>
!dynType2 = tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 1, 128]> : tensor<4xsi64>, order = #NCHW}>

// CHECK-LABEL: func.func @RhsDynamicBroadcast
// CHECK-SAME:  [[INPUT_0:%.+]]: tensor<1x?x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 16, 128]> : tensor<4xsi64>, order = #NCHW}>
// CHECK-SAME:  [[INPUT_1:%.+]]: tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 1, 128]> : tensor<4xsi64>, order = #NCHW}>
func.func @RhsDynamicBroadcast(%arg0: !dynType1, %arg1: !dynType2) -> !dynType1 {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : !dynType1, !dynType2 -> !dynType1
    return %0 : !dynType1

    // CHECK:       [[SHAPE:%.+]] = IE.ShapeOf([[INPUT_0]]) {dstElemType = si32} :
    // CHECK-SAME:    tensor<1x?x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 16, 128]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi32>
    // CHECK:       [[BCAST:%.+]] = IE.DynamicBroadcast([[INPUT_1]], [[SHAPE]]) {
    // CHECK-SAME:    mode = #IE.broadcast_type<NUMPY>, output_bounds = [1, 500, 16, 128], output_shape = [1, -9223372036854775808, -9223372036854775808, 128]} : tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 1, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi32> -> tensor<1x?x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 16, 128]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT_0]], [[BCAST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME:    tensor<1x?x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 16, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x?x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 16, 128]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x?x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 16, 128]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:  return [[ADD]] : tensor<1x?x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 16, 128]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

// CHECK-LABEL: @NotBroadcastWithSameInputShape
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x16x16x32xf16>
// CHECK-SAME:    [[ARG_1:%[^:]+]]: tensor<1x16x16x32xf16>
func.func @NotBroadcastWithSameInputShape(%arg0: tensor<1x16x16x32xf16>, %arg1: tensor<1x16x16x32xf16>) -> tensor<1x16x16x32xf16> {
    %0 = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x32xf16>, tensor<1x16x16x32xf16> -> tensor<1x16x16x32xf16>

    return %0 : tensor<1x16x16x32xf16>

    // CHECK:       [[ADD:%.+]] = IE.Add([[ARG_0]], [[ARG_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x32xf16>, tensor<1x16x16x32xf16> -> tensor<1x16x16x32xf16>
    // CHECK:       return [[ADD]]
}

// -----

// CHECK-LABEL: @NotBroadcastWithNon4DTensor
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x3x16xf16>
func.func @NotBroadcastWithNon4DTensor(%arg0: tensor<1x3x16xf16>) -> tensor<1x3x16xf16> {
    %0 = const.Declare tensor<1x1x16xf16> = dense<1.0> : tensor<1x1x16xf16>, [#const.CastElemType<f16>]
    %1 = IE.Add(%arg0, %0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x16xf16>, tensor<1x1x16xf16> -> tensor<1x3x16xf16>

    return %1 : tensor<1x3x16xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x16xf16> = dense<1.000000e+00> : tensor<1x1x16xf16>, [#const.CastElemType<f16>]
    // CHECK:       [[ADD:%.+]] =  IE.Add([[ARG_0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x16xf16>, tensor<1x1x16xf16> -> tensor<1x3x16xf16>
    // CHECK:       return [[ADD]]
}

// -----

// CHECK-LABEL: @NotBroadcastWithTrivialInput
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x3x16x16xf16>
func.func @NotBroadcastWithTrivialInput(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
    %0 = const.Declare tensor<1x1x1x1xf16> = dense<1.0> : tensor<1x1x1x1xf16>, [#const.CastElemType<f16>]
    %1 = IE.Add(%arg0, %0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x16x16xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x16x16xf16>

    return %1 : tensor<1x3x16x16xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.CastElemType<f16>]
    // CHECK:       [[ADD:%.+]] =  IE.Add([[ARG_0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x16x16xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x16x16xf16>
    // CHECK:       return [[ADD]]
}

// -----

// Single-input attention bias: a runtime input broadcasts over an already
// full-size operand, so it remains materialized for DPU lowering.
// CHECK-LABEL: @SingleInputAttentionBiasBroadcast
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x12x256x256xf16>
// CHECK-SAME:    [[ARG_1:%[^:]+]]: tensor<1x1x1x256xf16>
func.func @SingleInputAttentionBiasBroadcast(%arg0: tensor<1x12x256x256xf16>, %arg1: tensor<1x1x1x256xf16>) -> tensor<1x12x256x256xf16> {
    %0 = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x256x256xf16>, tensor<1x1x1x256xf16> -> tensor<1x12x256x256xf16>

    return %0 : tensor<1x12x256x256xf16>

    // CHECK-DAG:   [[TARGET_SHAPE:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 12, 256, 256]> : tensor<4xsi64>, [#const.CastElemType<si32>]
    // CHECK:       [[BROADCAST:%.+]] = IE.Broadcast([[ARG_1]], [[TARGET_SHAPE]])
    // CHECK-SAME:      {mode = #IE.broadcast_type<NUMPY>} : tensor<1x1x1x256xf16>, tensor<4xsi32> -> tensor<1x12x256x256xf16>
    // CHECK:       [[ADD:%.+]] = IE.Add([[ARG_0]], [[BROADCAST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x256x256xf16>, tensor<1x12x256x256xf16> -> tensor<1x12x256x256xf16>
    // CHECK:       return [[ADD]]
}

// -----

// Cross-input reading-order broadcast: both runtime inputs expand to the
// output shape, so the CMX heuristic leaves the Add for SW/SHAVE lowering.
// CHECK-LABEL: @CrossInputReadingOrderBroadcast
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x300x1x256xf16>
// CHECK-SAME:    [[ARG_1:%[^:]+]]: tensor<1x1x300x256xf16>
func.func @CrossInputReadingOrderBroadcast(%arg0: tensor<1x300x1x256xf16>, %arg1: tensor<1x1x300x256xf16>) -> tensor<1x300x300x256xf16> {
    %0 = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x300x1x256xf16>, tensor<1x1x300x256xf16> -> tensor<1x300x300x256xf16>

    return %0 : tensor<1x300x300x256xf16>

    // CHECK-NOT:   IE.Broadcast
    // CHECK:       [[ADD:%.+]] = IE.Add([[ARG_0]], [[ARG_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x300x1x256xf16>, tensor<1x1x300x256xf16> -> tensor<1x300x300x256xf16>
    // CHECK:       return [[ADD]]
}

// -----

// A large single-input broadcast remains materialized for DPU lowering.
// CHECK-LABEL: @BroadcastLargeVolumeTensorInput
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x256x300x300xf16>
// CHECK-SAME:    [[ARG_1:%[^:]+]]: tensor<1x256x1x1xf16>
func.func @BroadcastLargeVolumeTensorInput(%arg0: tensor<1x256x300x300xf16>, %arg1: tensor<1x256x1x1xf16>) -> tensor<1x256x300x300xf16> {
    %0 = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x300x300xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x300x300xf16>

    return %0 : tensor<1x256x300x300xf16>

    // CHECK-DAG:   [[TARGET_SHAPE:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 256, 300, 300]> : tensor<4xsi64>, [#const.CastElemType<si32>]
    // CHECK:       [[BROADCAST:%.+]] = IE.Broadcast([[ARG_1]], [[TARGET_SHAPE]])
    // CHECK-SAME:      {mode = #IE.broadcast_type<NUMPY>} : tensor<1x256x1x1xf16>, tensor<4xsi32> -> tensor<1x256x300x300xf16>
    // CHECK:       [[ADD:%.+]] = IE.Add([[ARG_0]], [[BROADCAST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x300x300xf16>, tensor<1x256x300x300xf16> -> tensor<1x256x300x300xf16>
    // CHECK:       return [[ADD]]
}

// -----

// Both operands need broadcasting to a large volume - neither should be materialized.
// CHECK-LABEL: @NotBroadcastLargeVolumeTwoTensorInputs
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x256x1x300xf16>
// CHECK-SAME:    [[ARG_1:%[^:]+]]: tensor<1x256x300x1xf16>
func.func @NotBroadcastLargeVolumeTwoTensorInputs(%arg0: tensor<1x256x1x300xf16>, %arg1: tensor<1x256x300x1xf16>) -> tensor<1x256x300x300xf16> {
    %0 = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x1x300xf16>, tensor<1x256x300x1xf16> -> tensor<1x256x300x300xf16>

    return %0 : tensor<1x256x300x300xf16>

    // CHECK-NOT:   IE.Broadcast
    // CHECK:       [[ADD:%.+]] = IE.Add([[ARG_0]], [[ARG_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x1x300xf16>, tensor<1x256x300x1xf16> -> tensor<1x256x300x300xf16>
    // CHECK:       return [[ADD]]
}

// -----

// Constant broadcasts still fold at compile time regardless of volume - unaffected. Uses a
// non-trivial bias shape (H=300, matching output) so it isn't shortcut to ScaleShift and instead
// exercises the constant-folded broadcast path.
// CHECK-LABEL: @BroadcastLargeVolumeConstantInput
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x256x300x300xf16>
func.func @BroadcastLargeVolumeConstantInput(%arg0: tensor<1x256x300x300xf16>) -> tensor<1x256x300x300xf16> {
    %0 = const.Declare tensor<1x256x300x1xf16> = dense<1.0> : tensor<1x256x300x1xf16>, [#const.CastElemType<f16>]
    %1 = IE.Add(%arg0, %0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x300x300xf16>, tensor<1x256x300x1xf16> -> tensor<1x256x300x300xf16>

    return %1 : tensor<1x256x300x300xf16>

    // CHECK:       [[BROADCAST:%.+]] = const.Declare tensor<1x256x300x300xf16>
    // CHECK-SAME:      dense<1.000000e+00> : tensor<1x256x300x1xf16>,
    // CHECK-SAME:      [#const.CastElemType<f16>, #const.Broadcast<3 : i64, 300 : i64>]
    // CHECK:       [[ADD_RES:%.+]] = IE.Add([[ARG_0]], [[BROADCAST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x300x300xf16>, tensor<1x256x300x300xf16> -> tensor<1x256x300x300xf16>

    // CHECK:       return [[ADD_RES]]
}

// -----

// CHECK-LABEL: @BroadcastModerateVolumeTensorInput
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x12x256x240xf16>
// CHECK-SAME:    [[ARG_1:%[^:]+]]: tensor<1x1x1x240xf16>
func.func @BroadcastModerateVolumeTensorInput(%arg0: tensor<1x12x256x240xf16>, %arg1: tensor<1x1x1x240xf16>) -> tensor<1x12x256x240xf16> {
    %0 = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x256x240xf16>, tensor<1x1x1x240xf16> -> tensor<1x12x256x240xf16>

    return %0 : tensor<1x12x256x240xf16>

    // CHECK-DAG:   [[TARGET_SHAPE:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 12, 256, 240]> : tensor<4xsi64>, [#const.CastElemType<si32>]
    // CHECK:       [[BROADCAST:%.+]] = IE.Broadcast([[ARG_1]], [[TARGET_SHAPE]])
    // CHECK-SAME:      {mode = #IE.broadcast_type<NUMPY>} : tensor<1x1x1x240xf16>, tensor<4xsi32> -> tensor<1x12x256x240xf16>
    // CHECK:       [[ADD_RES:%.+]] = IE.Add([[ARG_0]], [[BROADCAST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x256x240xf16>, tensor<1x12x256x240xf16> -> tensor<1x12x256x240xf16>

    // CHECK:       return [[ADD_RES]]
}
