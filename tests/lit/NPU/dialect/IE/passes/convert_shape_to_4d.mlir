//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-shape-to-4d --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK:       func.func @main(
// CHECK-SAME:      [[VAL_0:%.+]]: tensor<1x1000xf32>
// CHECK-SAME:      [[VAL_1:%.+]]: tensor<1x224x224xf32>
// CHECK-SAME:      [[VAL_2:%.+]]: tensor<1x512xf32>
// CHECK-SAME:      [[VAL_3:%.+]]: tensor<8x1024xf32>
func.func @main(%arg0: tensor<1x1000xf32>, %arg1: tensor<1x224x224xf32>, %arg2: tensor<1x512xf32>, %arg3: tensor<8x1024xf32>) ->
        (tensor<1x1000xf32>, tensor<1x224x224xf32>, tensor<1x512xf32>, tensor<8x1024xf32>) {
    %0 = IE.Clamp(%arg0) {min = 1.0, max = 3.0} : tensor<1x1000xf32> -> tensor<1x1000xf32>
    %1 = IE.Sigmoid(%arg1) : tensor<1x224x224xf32> -> tensor<1x224x224xf32>
    %2 = IE.Elu(%1) {x = 1.0} : tensor<1x224x224xf32> -> tensor<1x224x224xf32>

    %input_low = const.Declare tensor<1x1xf32> = dense<0.0> : tensor<1x1xf32>
    %input_high = const.Declare tensor<1x1xf32> = dense<255.0> : tensor<1x1xf32>
    %output_low = const.Declare tensor<1x1xf32> = dense<0.0> : tensor<1x1xf32>
    %output_high = const.Declare tensor<1x1xf32> = dense<255.0> : tensor<1x1xf32>
    %3 = IE.FakeQuantize(%arg2, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x512xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<1x512xf32>

    %4 = const.Declare tensor<1xf32> = dense<6.0> : tensor<1xf32>
    %5 = const.Declare tensor<1xf32> = dense<2.0> : tensor<1xf32>
    %6 = IE.Subtract(%arg3, %4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<8x1024xf32>, tensor<1xf32> -> tensor<8x1024xf32>
    %7 = IE.Add(%6, %5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<8x1024xf32>, tensor<1xf32> -> tensor<8x1024xf32>

    return %0, %2, %3, %7 : tensor<1x1000xf32>, tensor<1x224x224xf32>, tensor<1x512xf32>, tensor<8x1024xf32>

    // CHECK-DAG: [[VAL_4:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1xf32>, [#const.Reshape<[1, 1, 1, 1]>]
    // CHECK-DAG: [[VAL_5:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<6.000000e+00> : tensor<1xf32>, [#const.Reshape<[1, 1, 1, 1]>]
    // CHECK-DAG: [[VAL_6:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>]
    // CHECK-DAG: [[VAL_7:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>]

    // CHECK-DAG:   [[VAL_3_4D:%.+]] = IE.AffineReshape([[VAL_3]]) {
    // CHECK-SAME:      shape_value = [1, 1, 8, 1024]} : tensor<8x1024xf32> -> tensor<1x1x8x1024xf32>
    // CHECK-DAG:   [[VAL_1_4D:%.+]] = IE.AffineReshape([[VAL_1]]) {
    // CHECK-SAME:      shape_value = [1, 1, 224, 224]} : tensor<1x224x224xf32> -> tensor<1x1x224x224xf32>
    // CHECK-DAG:   [[VAL_0_4D:%.+]] = IE.AffineReshape([[VAL_0]]) {
    // CHECK-SAME:      shape_value = [1, 1, 1, 1000]} : tensor<1x1000xf32> -> tensor<1x1x1x1000xf32>

    // CHECK:   [[VAL_8:%.+]] = IE.Clamp([[VAL_0_4D]])
    // CHECK:   [[VAL_8_2D:%.+]] = IE.AffineReshape([[VAL_8]]) {
    // CHECK-SAME:      shape_value = [1, 1000]} : tensor<1x1x1x1000xf32> -> tensor<1x1000xf32>

    // CHECK:   [[VAL_9:%.+]] = IE.Sigmoid([[VAL_1_4D]])
    // CHECK:   [[VAL_10:%.+]] = IE.Elu([[VAL_9]])
    // CHECK-DAG:   [[VAL_10_3D:%.+]] = IE.AffineReshape([[VAL_10]]) {
    // CHECK-SAME:      shape_value = [1, 224, 224]} : tensor<1x1x224x224xf32> -> tensor<1x224x224xf32>
    // CHECK-DAG:   [[VAL_2_4D:%.+]] = IE.AffineReshape([[VAL_2]]) {
    // CHECK-SAME:      shape_value = [1, 1, 1, 512]} : tensor<1x512xf32> -> tensor<1x1x1x512xf32>

    // CHECK:   [[VAL_11:%.+]] = IE.FakeQuantize([[VAL_2_4D]], [[VAL_7]], [[VAL_6]], [[VAL_7]], [[VAL_6]])
    // CHECK:   [[VAL_11_2D:%.+]] = IE.AffineReshape([[VAL_11]]) {
    // CHECK-SAME:      shape_value = [1, 512]} : tensor<1x1x1x512xf32> -> tensor<1x512xf32>

    // CHECK:   [[VAL_12:%.+]] = IE.Subtract([[VAL_3_4D]], [[VAL_5]])
    // CHECK:   [[VAL_13:%.+]] = IE.Add([[VAL_12]], [[VAL_4]])
    // CHECK:   [[VAL_13_2D:%.+]] = IE.AffineReshape([[VAL_13]]) {
    // CHECK-SAME:      shape_value = [8, 1024]} : tensor<1x1x8x1024xf32> -> tensor<8x1024xf32>

    // CHECK:   return [[VAL_8_2D]], [[VAL_10_3D]], [[VAL_11_2D]], [[VAL_13_2D]]
}

// -----

// CHECK-LABEL: func.func @FakeQuantizePerChannel5D(
// CHECK-SAME:      %[[VAL_0:.+]]: tensor<2x3x4x512x64xf32>
func.func @FakeQuantizePerChannel5D(%arg0: tensor<2x3x4x512x64xf32>) -> (tensor<2x3x4x512x64xf32>) {
    %input_low = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %input_high = const.Declare tensor<f32> = dense<255.0> : tensor<f32>
    %output_low = const.Declare tensor<1x1x1x512x1xf32> = dense<10.0> : tensor<1x1x1x512x1xf32>
    %output_high = const.Declare tensor<1x1x1x512x1xf32> = dense<205.0> : tensor<1x1x1x512x1xf32>
    %3 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32 } :
        tensor<2x3x4x512x64xf32>, tensor<f32>, tensor<f32>, tensor<1x1x1x512x1xf32>, tensor<1x1x1x512x1xf32> -> tensor<2x3x4x512x64xf32>

    return %3 : tensor<2x3x4x512x64xf32>

    // CHECK-DAG: %[[VAL_4:.+]] = const.Declare tensor<1x1x512x1xf32> = dense<2.050000e+02>
    // CHECK-DAG: %[[VAL_3:.+]] = const.Declare tensor<1x1x512x1xf32> = dense<1.000000e+01>
    // CHECK-DAG: %[[VAL_2:.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02>
    // CHECK-DAG: %[[VAL_1:.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00>

    // CHECK:   %[[RESHAPE_BEFORE:.+]] = IE.Reshape(%[[VAL_0]]) {
    // CHECK-SAME:      shape_value = [1, 24, 512, 64]} : tensor<2x3x4x512x64xf32> -> tensor<1x24x512x64xf32>
    // CHECK:   %[[FQ:.+]] = IE.FakeQuantize(%[[RESHAPE_BEFORE]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]])
    // CHECK:   %[[RESHAPE_AFTER:.+]] = IE.Reshape(%[[FQ]]) {
    // CHECK-SAME:      shape_value = [2, 3, 4, 512, 64]} : tensor<1x24x512x64xf32> -> tensor<2x3x4x512x64xf32>
    // CHECK:   return %[[RESHAPE_AFTER]]
}

// -----

// CHECK-LABEL: func.func @FakeQuantizePerChannel5DWithDifferentInputOutput(
// CHECK-SAME:      %[[VAL_0:.+]]: tensor<2x3x4x512x64xf32>
func.func @FakeQuantizePerChannel5DWithDifferentInputOutput(%arg0: tensor<2x3x4x512x64xf32>) -> (tensor<2x3x4x512x64xf32>) {
    %input_low = const.Declare tensor<1x1x4x1x1xf32> = dense<0.0> : tensor<1x1x4x1x1xf32>
    %input_high = const.Declare tensor<1x1x4x1x1xf32> = dense<255.0> : tensor<1x1x4x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x512x1xf32> = dense<10.0> : tensor<1x1x1x512x1xf32>
    %output_high = const.Declare tensor<1x1x1x512x1xf32> = dense<205.0> : tensor<1x1x1x512x1xf32>
    %3 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32 } :
        tensor<2x3x4x512x64xf32>, tensor<1x1x4x1x1xf32>, tensor<1x1x4x1x1xf32>, tensor<1x1x1x512x1xf32>, tensor<1x1x1x512x1xf32> -> tensor<2x3x4x512x64xf32>

    return %3 : tensor<2x3x4x512x64xf32>

    // CHECK-DAG: %[[VAL_4:.+]] = const.Declare tensor<1x1x512x1xf32> = dense<2.050000e+02> : tensor<1x1x1x512x1xf32>, [#const.Reshape<[1, 1, 512, 1]>]
    // CHECK-DAG: %[[VAL_3:.+]] = const.Declare tensor<1x1x512x1xf32> = dense<1.000000e+01> : tensor<1x1x1x512x1xf32>, [#const.Reshape<[1, 1, 512, 1]>]
    // CHECK-DAG: %[[VAL_2:.+]] = const.Declare tensor<1x4x1x1xf32> = dense<2.550000e+02> : tensor<1x1x4x1x1xf32>, [#const.Reshape<[1, 4, 1, 1]>]
    // CHECK-DAG: %[[VAL_1:.+]] = const.Declare tensor<1x4x1x1xf32> = dense<0.000000e+00> : tensor<1x1x4x1x1xf32>, [#const.Reshape<[1, 4, 1, 1]>]

    // CHECK:   %[[RESHAPE_BEFORE:.+]] = IE.AffineReshape(%[[VAL_0]]) {
    // CHECK-SAME:      shape_value = [6, 4, 512, 64]} : tensor<2x3x4x512x64xf32> -> tensor<6x4x512x64xf32>
    // CHECK:   %[[FQ:.+]] = IE.FakeQuantize(%[[RESHAPE_BEFORE]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]])
    // CHECK:   %[[RESHAPE_AFTER:.+]] = IE.AffineReshape(%[[FQ]]) {
    // CHECK-SAME:      shape_value = [2, 3, 4, 512, 64]} : tensor<6x4x512x64xf32> -> tensor<2x3x4x512x64xf32>
    // CHECK:   return %[[RESHAPE_AFTER]]
}

// -----

// CHECK-LABEL: func.func @FakeQuantizePerChannel3D(
// CHECK-SAME:      %[[VAL_0:.+]]: tensor<1x512x64xf32>
func.func @FakeQuantizePerChannel3D(%arg0: tensor<1x512x64xf32>) -> (tensor<1x512x64xf32>) {
    %input_low = const.Declare tensor<1x512x1xf32> = dense<0.0> : tensor<1x512x1xf32>
    %input_high = const.Declare tensor<1x512x1xf32> = dense<255.0> : tensor<1x512x1xf32>
    %output_low = const.Declare tensor<1x512x1xf32> = dense<10.0> : tensor<1x512x1xf32>
    %output_high = const.Declare tensor<1x512x1xf32> = dense<205.0> : tensor<1x512x1xf32>
    %3 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32 } :
        tensor<1x512x64xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32> -> tensor<1x512x64xf32>

    return %3 : tensor<1x512x64xf32>

    // CHECK-DAG: %[[VAL_4:.+]] = const.Declare tensor<1x512x1x1xf32> = dense<2.050000e+02> : tensor<1x512x1xf32>, [#const.Reshape<[1, 512, 1, 1]>]
    // CHECK-DAG: %[[VAL_3:.+]] = const.Declare tensor<1x512x1x1xf32> = dense<1.000000e+01> : tensor<1x512x1xf32>, [#const.Reshape<[1, 512, 1, 1]>]
    // CHECK-DAG: %[[VAL_2:.+]] = const.Declare tensor<1x512x1x1xf32> = dense<2.550000e+02> : tensor<1x512x1xf32>, [#const.Reshape<[1, 512, 1, 1]>]
    // CHECK-DAG: %[[VAL_1:.+]] = const.Declare tensor<1x512x1x1xf32> = dense<0.000000e+00> : tensor<1x512x1xf32>, [#const.Reshape<[1, 512, 1, 1]>]

    // CHECK:   %[[RESHAPE_BEFORE:.+]] = IE.AffineReshape(%[[VAL_0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 512, 1, 64]} : tensor<1x512x64xf32> -> tensor<1x512x1x64xf32>
    // CHECK:   %[[FQ:.+]] = IE.FakeQuantize(%[[RESHAPE_BEFORE]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]])
    // CHECK:   %[[RESHAPE_AFTER:.+]] = IE.AffineReshape(%[[FQ]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2]], shape_value = [1, 512, 64]} : tensor<1x512x1x64xf32> -> tensor<1x512x64xf32>
    // CHECK:   return %[[RESHAPE_AFTER]]
}

// -----

// CHECK-LABEL: func.func @FakeQuantizePerChannel2D(
// CHECK-SAME:      %[[VAL_0:.+]]: tensor<512x64xf32>
func.func @FakeQuantizePerChannel2D(%arg0: tensor<512x64xf32>) -> (tensor<512x64xf32>) {
    %input_low = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %input_high = const.Declare tensor<f32> = dense<255.0> : tensor<f32>
    %output_low = const.Declare tensor<512x1xf32> = dense<10.0> : tensor<512x1xf32>
    %output_high = const.Declare tensor<512x1xf32> = dense<205.0> : tensor<512x1xf32>
    %3 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32 } :
        tensor<512x64xf32>, tensor<f32>, tensor<f32>, tensor<512x1xf32>, tensor<512x1xf32> -> tensor<512x64xf32>

    return %3 : tensor<512x64xf32>

    // CHECK-DAG: %[[VAL_4:.+]] = const.Declare tensor<1x512x1x1xf32> = dense<2.050000e+02>
    // CHECK-DAG: %[[VAL_3:.+]] = const.Declare tensor<1x512x1x1xf32> = dense<1.000000e+01>
    // CHECK-DAG: %[[VAL_2:.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02>
    // CHECK-DAG: %[[VAL_1:.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00>

    // CHECK:   %[[RESHAPE_BEFORE:.+]] = IE.AffineReshape(%[[VAL_0]]) {
    // CHECK-SAME:      shape_value = [1, 512, 1, 64]} : tensor<512x64xf32> -> tensor<1x512x1x64xf32>
    // CHECK:   %[[FQ:.+]] = IE.FakeQuantize(%[[RESHAPE_BEFORE]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]])
    // CHECK:   %[[RESHAPE_AFTER:.+]] = IE.AffineReshape(%[[FQ]]) {
    // CHECK-SAME:      shape_value = [512, 64]} : tensor<1x512x1x64xf32> -> tensor<512x64xf32>
    // CHECK:   return %[[RESHAPE_AFTER]]
}

// -----

// CHECK-LABEL: func.func @FakeQuantizePerTensor(
// CHECK-SAME:      %[[VAL_0:.+]]: tensor<512x64xf32>
// CHECK-SAME:  -> tensor<512x64xf32>
func.func @FakeQuantizePerTensor(%arg0: tensor<512x64xf32>) -> tensor<512x64xf32> {
    %input_low = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %input_high = const.Declare tensor<f32> = dense<255.0> : tensor<f32>
    %output_low = const.Declare tensor<f32> = dense<10.0> : tensor<f32>
    %output_high = const.Declare tensor<f32> = dense<205.0> : tensor<f32>
    %3 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32 } :
        tensor<512x64xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<512x64xf32>

    return %3 : tensor<512x64xf32>

    // CHECK-DAG: %[[VAL_1:.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00>
    // CHECK-DAG: %[[VAL_2:.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02>
    // CHECK-DAG: %[[VAL_3:.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+01>
    // CHECK-DAG: %[[VAL_4:.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.050000e+02>

    // CHECK:   %[[RESHAPE_BEFORE:.+]] = IE.AffineReshape(%[[VAL_0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 512, 64]} : tensor<512x64xf32> -> tensor<1x1x512x64xf32>
    // CHECK:   %[[FQ:.+]] = IE.FakeQuantize(%[[RESHAPE_BEFORE]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]])
    // CHECK:   %[[RESHAPE_AFTER:.+]] = IE.AffineReshape(%[[FQ]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [512, 64]} : tensor<1x1x512x64xf32> -> tensor<512x64xf32>
    // CHECK:   return %[[RESHAPE_AFTER]]
}

// -----

// CHECK-LABEL: func.func @FakeQuantizePerTensor5Dto4D(
// CHECK-SAME:      %[[VAL_0:.+]]: tensor<1x3x13x13x6xf16>
func.func @FakeQuantizePerTensor5Dto4D(%arg0: tensor<1x3x13x13x6xf16>) -> (tensor<1x3x13x13x6xf16>) {
    %input_low = const.Declare tensor<1x1x1x1x1xf16> = dense<0.950494349> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %input_high = const.Declare tensor<1x1x1x1x1xf16> = dense<60.8316383> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %output_low = const.Declare tensor<1x1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %output_high = const.Declare tensor<1x1x1x1x1xf16> = dense<62.8316383> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %3 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32 } :
        tensor<1x3x13x13x6xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x3x13x13x6xf16>

    return %3 : tensor<1x3x13x13x6xf16>

    // CHECK-DAG: %[[VAL_1:.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.950494349> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG: %[[VAL_2:.+]] = const.Declare tensor<1x1x1x1xf16> = dense<60.8316383> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG: %[[VAL_3:.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG: %[[VAL_4:.+]] = const.Declare tensor<1x1x1x1xf16> = dense<62.8316383> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:   %[[RESHAPE_BEFORE:.+]] = IE.AffineReshape(%[[VAL_0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [3], [3]], shape_value = [1, 3, 13, 78]} : tensor<1x3x13x13x6xf16> -> tensor<1x3x13x78xf16>
    // CHECK:   %[[FQ:.+]] = IE.FakeQuantize(%[[RESHAPE_BEFORE]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]])
    // CHECK:   %[[RESHAPE_AFTER:.+]] = IE.AffineReshape(%[[FQ]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [3, 4]], shape_value = [1, 3, 13, 13, 6]} : tensor<1x3x13x78xf16> -> tensor<1x3x13x13x6xf16>
    // CHECK:   return %[[RESHAPE_AFTER]]
}

// -----

// CHECK-LABEL: func.func @FakeQuantizeDifferentInputAndOutput_NoOp(
// CHECK-SAME:      %[[VAL_0:.+]]: tensor<48x3x3x3xf32>
// CHECK-SAME:  -> tensor<48x3x3x3xf32>
func.func @FakeQuantizeDifferentInputAndOutput_NoOp(%arg0: tensor<48x3x3x3xf32>) -> tensor<48x3x3x3xf32> {
    %input_low = const.Declare tensor<1xf32> = dense<0.000000e+00> : tensor<1xf32>
    %input_high = const.Declare tensor<1xf32> = dense<2.540000e+02> : tensor<1xf32>
    %output_low = const.Declare tensor<48x1x1x1xf32> = dense<-1.000000e+00> : tensor<48x1x1x1xf32>
    %output_high = const.Declare tensor<48x1x1x1xf32> = dense<1.000000e+00> : tensor<48x1x1x1xf32>
    %fq = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} :
        tensor<48x3x3x3xf32>, tensor<1xf32>, tensor<1xf32>, tensor<48x1x1x1xf32>, tensor<48x1x1x1xf32> -> tensor<48x3x3x3xf32>
    return %fq : tensor<48x3x3x3xf32>

    // CHECK-DAG: %[[VAL_1:.+]] = const.Declare tensor<1xf32> = dense<0.000000e+00> : tensor<1xf32>
    // CHECK-DAG: %[[VAL_2:.+]] = const.Declare tensor<1xf32> = dense<2.540000e+02> : tensor<1xf32>
    // CHECK-DAG: %[[VAL_3:.+]] = const.Declare tensor<48x1x1x1xf32> = dense<-1.000000e+00> : tensor<48x1x1x1xf32>
    // CHECK-DAG: %[[VAL_4:.+]] = const.Declare tensor<48x1x1x1xf32> = dense<1.000000e+00> : tensor<48x1x1x1xf32>

    // CHECK:   %[[FQ:.+]] = IE.FakeQuantize(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]])
    // CHECK:   return %[[FQ]]
}

// -----

// CHECK-LABEL: func.func @FakeQuantizeDifferentInputAndOutput3Dto4D(
// CHECK-SAME:      %[[VAL_0:.+]]: tensor<1x32x64xf16>
// CHECK-SAME:  -> tensor<1x32x64xf16>
func.func @FakeQuantizeDifferentInputAndOutput3Dto4D(%arg0: tensor<1x32x64xf16>) -> tensor<1x32x64xf16> {
    %input_low = const.Declare tensor<1xf16> = dense<0.000000e+00> : tensor<1xf16>
    %input_high = const.Declare tensor<1xf16> = dense<2.540000e+02> : tensor<1xf16>
    %output_low = const.Declare tensor<1x1x64xf16> = dense<-1.000000e+00> : tensor<1x1x64xf16>
    %output_high = const.Declare tensor<1x1x64xf16> = dense<1.000000e+00> : tensor<1x1x64xf16>

    %fq = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
        : tensor<1x32x64xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1x1x64xf16>, tensor<1x1x64xf16>
        -> tensor<1x32x64xf16>

    return %fq : tensor<1x32x64xf16>

    // CHECK-DAG: %[[VAL_1:.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00>
    // CHECK-DAG: %[[VAL_2:.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.540000e+02>
    // CHECK-DAG: %[[VAL_3:.+]] = const.Declare tensor<1x1x1x64xf16> = dense<-1.000000e+00>
    // CHECK-DAG: %[[VAL_4:.+]] = const.Declare tensor<1x1x1x64xf16> = dense<1.000000e+00>

    // CHECK: %[[RESHAPE_BEFORE:.+]] = IE.AffineReshape(%[[VAL_0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 32, 1, 64]}
    // CHECK-SAME: -> tensor<1x32x1x64xf16>

    // CHECK:   %[[FQ:.+]] = IE.FakeQuantize(%[[RESHAPE_BEFORE]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]])

    // CHECK: %[[RESHAPE_AFTER:.+]] = IE.AffineReshape(%[[FQ]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2]], shape_value = [1, 32, 64]}

    // CHECK:   return %[[RESHAPE_AFTER]]
}

// -----

func.func @main(%arg0: tensor<1x256x32xf32>) -> tensor<1x256x32xf32> {
    %0 = const.Declare tensor<1x256x1xf32> = dense<6.0> : tensor<1x256x1xf32>
    %1 = IE.ScaleShift(%arg0, %0) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<1x256x32xf32>, tensor<1x256x1xf32> -> tensor<1x256x32xf32>
    %2 = IE.Clamp(%1) {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64} : tensor<1x256x32xf32> -> tensor<1x256x32xf32>

    return %2 : tensor<1x256x32xf32>

    // CHECK:       [[VAL_0:%.+]] = const.Declare tensor<1x256x1x1xf32> = dense<6.000000e+00> : tensor<1x256x1xf32>, [
    // CHECK-SAME:      #const.Reshape<[1, 256, 1, 1]>]
    // CHECK:       [[VAL_0_4D:%.+]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 256, 1, 32]} : tensor<1x256x32xf32> -> tensor<1x256x1x32xf32>
    // CHECK:       [[VAL_1:%.+]] = IE.ScaleShift([[VAL_0_4D]], [[VAL_0]]) {operandSegmentSizes = array<i32: 1, 0, 1>}
    // CHECK-SAME:      tensor<1x256x1x32xf32>, tensor<1x256x1x1xf32> -> tensor<1x256x1x32xf32>
    // CHECK:       [[VAL_Reshape:%.+]]  = IE.AffineReshape(%1) {
    // CHECK-SAME:      shape_value = [1, 1, 256, 32]} : tensor<1x256x1x32xf32> -> tensor<1x1x256x32xf32>
    // CHECK:       %[[VAL_2:.+]] = IE.Clamp([[VAL_Reshape]]) {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64} : tensor<1x1x256x32xf32> -> tensor<1x1x256x32xf32>
    // CHECK:       %[[VAL_1_4D:.+]] = IE.AffineReshape(%[[VAL_2]]) {
    // CHECK-SAME:      shape_value = [1, 256, 32]} : tensor<1x1x256x32xf32> -> tensor<1x256x32xf32>

    // CHECK:   return %[[VAL_1_4D]]
}

// -----

// CHECK-LABEL: func.func @AddOpInput3D
func.func @AddOpInput3D(%arg0: tensor<1x1x64xf16>, %arg1: tensor<1x1x64xf16>) -> tensor<1x1x64xf16> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x64xf16>, tensor<1x1x64xf16> -> tensor<1x1x64xf16>
    return %0 : tensor<1x1x64xf16>

    // CHECK:    %[[Reshape_1:.+]] = IE.AffineReshape(%arg1) {
    // CHECK-SAME:      shape_value = [1, 1, 1, 64]} : tensor<1x1x64xf16> -> tensor<1x1x1x64xf16>
    // CHECK:    %[[Reshape_0:.+]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 1, 1, 64]} : tensor<1x1x64xf16> -> tensor<1x1x1x64xf16>
    // CHECK:    %[[Add:.+]] = IE.Add(%[[Reshape_0]], %[[Reshape_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>
    // CHECK:    %[[Reshape_out:.+]] = IE.AffineReshape(%[[Add]]) {
    // CHECK-SAME:      shape_value = [1, 1, 64]} : tensor<1x1x1x64xf16> -> tensor<1x1x64xf16>
    // CHECK:    return %[[Reshape_out]]
}

// -----

// CHECK-LABEL: func.func @AddOpInput3DWithBroadcastNoOpt
func.func @AddOpInput3DWithBroadcastNoOpt(%arg0: tensor<1x1x1xf16>, %arg1: tensor<1x1x64xf16>) -> tensor<1x1x64xf16> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf16>, tensor<1x1x64xf16> -> tensor<1x1x64xf16>
    return %0 : tensor<1x1x64xf16>

    // CHECK:    %[[Reshape_1:.+]] = IE.AffineReshape(%arg1) {
    // CHECK-SAME:      shape_value = [1, 1, 1, 64]} : tensor<1x1x64xf16> -> tensor<1x1x1x64xf16>
    // CHECK:    %[[Reshape_0:.+]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 1, 1, 1]} : tensor<1x1x1xf16> -> tensor<1x1x1x1xf16>
    // CHECK:    %[[Add:.+]] = IE.Add(%[[Reshape_0]], %[[Reshape_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf16>, tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>
    // CHECK:    %[[Reshape_out:.+]] = IE.AffineReshape(%[[Add]]) {
    // CHECK-SAME:      shape_value = [1, 1, 64]} : tensor<1x1x1x64xf16> -> tensor<1x1x64xf16>
    // CHECK:    return %[[Reshape_out]]
}

// -----

// CHECK-LABEL: func.func @AddOpInput2DNoOpt
func.func @AddOpInput2DNoOpt(%arg0: tensor<3x16xf16>, %arg1: tensor<3x16xf16>) -> tensor<3x16xf16> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3x16xf16>, tensor<3x16xf16> -> tensor<3x16xf16>
    return %0 : tensor<3x16xf16>

    // CHECK:    %[[Reshape_1:.+]] = IE.AffineReshape(%arg1) {
    // CHECK-SAME:      shape_value = [1, 1, 3, 16]} : tensor<3x16xf16> -> tensor<1x1x3x16xf16>
    // CHECK:    %[[Reshape_0:.+]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 1, 3, 16]} : tensor<3x16xf16> -> tensor<1x1x3x16xf16>
    // CHECK:    %[[Add:.+]] = IE.Add(%[[Reshape_0]], %[[Reshape_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3x16xf16>, tensor<1x1x3x16xf16> -> tensor<1x1x3x16xf16>
    // CHECK:    %[[Reshape_out:.+]] = IE.AffineReshape(%[[Add]]) {
    // CHECK-SAME:      shape_value = [3, 16]} : tensor<1x1x3x16xf16> -> tensor<3x16xf16>
    // CHECK:    return %[[Reshape_out]]
}

// -----

// CHECK-LABEL: @Convert3dAddWithLastDim
func.func @Convert3dAddWithLastDim(%arg0: tensor<1x1x80xf16>) -> tensor<1x1x80xf16> {
    %ADD_WEIGHTS = const.Declare tensor<1x1x80xf16> = dense<2.000000e+00> : tensor<1x1x80xf16>

    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x1x80xf16>, tensor<1x1x80xf16> -> tensor<1x1x80xf16>

    return %ADD : tensor<1x1x80xf16>

    // CHECK-DAG:   [[ADD_WEIGHTS:%.+]] = const.Declare tensor<1x1x1x80xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<1x1x80xf16>, [#const.Reshape<[1, 1, 1, 80]>]

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 1, 1, 80]
    // CHECK-SAME:  } : tensor<1x1x80xf16> -> tensor<1x1x1x80xf16>

    // CHECK:   [[ADD:%.+]] = IE.Add([[RESHAPE_INPUT]], [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x1x1x80xf16>, tensor<1x1x1x80xf16> -> tensor<1x1x1x80xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[ADD]]) {
    // CHECK-SAME:      shape_value = [1, 1, 80]
    // CHECK-SAME:  } : tensor<1x1x1x80xf16> -> tensor<1x1x80xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x1x80xf16>
}

// -----

// CHECK-LABEL: @Convert3dMulWithLastDim
func.func @Convert3dMulWithLastDim(%arg0: tensor<1x1x80xf16>) -> tensor<1x1x80xf16> {
    %MUL_WEIGHTS = const.Declare tensor<1x1x80xf16> = dense<2.000000e+00> : tensor<1x1x80xf16>

    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x1x80xf16>, tensor<1x1x80xf16> -> tensor<1x1x80xf16>

    return %MUL : tensor<1x1x80xf16>

    // CHECK-DAG:   [[MUL_WEIGHTS:%.+]] = const.Declare tensor<1x80x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<1x1x80xf16>

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 80, 1, 1]
    // CHECK-SAME:  } : tensor<1x1x80xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[MUL:%.+]] = IE.Multiply([[RESHAPE_INPUT]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[MUL]]) {
    // CHECK-SAME:      shape_value = [1, 1, 80]
    // CHECK-SAME:  } : tensor<1x80x1x1xf16> -> tensor<1x1x80xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x1x80xf16>
}

// -----

// CHECK-LABEL: @Convert3dAddWithSecondDim
func.func @Convert3dAddWithSecondDim(%arg0: tensor<1x80x1xf16>) -> tensor<1x80x1xf16> {
    %ADD_WEIGHTS = const.Declare tensor<1x80x1xf16> = dense<2.000000e+00> : tensor<1x80x1xf16>

    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x80x1xf16>, tensor<1x80x1xf16> -> tensor<1x80x1xf16>

    return %ADD : tensor<1x80x1xf16>

    // CHECK-DAG:   [[ADD_WEIGHTS:%.+]] = const.Declare tensor<1x80x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<1x80x1xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME{LITERAL}: dim_mapping = [[0], [1], [2, 3]],
    // CHECK-SAME:          shape_value = [1, 80, 1, 1]
    // CHECK-SAME:  } : tensor<1x80x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[ADD:%.+]] = IE.Add([[RESHAPE_INPUT]], [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[ADD]]) {
    // CHECK-SAME{LITERAL}: dim_mapping = [[0], [1], [2], [2]],
    // CHECK-SAME:          shape_value = [1, 80, 1]
    // CHECK-SAME:  } : tensor<1x80x1x1xf16> -> tensor<1x80x1xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x80x1xf16>
}

// -----

// CHECK-LABEL: @Convert3dMulWithLastDim
func.func @Convert3dMulWithLastDim(%arg0: tensor<1x80x1xf16>) -> tensor<1x80x1xf16> {
    %MUL_WEIGHTS = const.Declare tensor<1x80x1xf16> = dense<2.000000e+00> : tensor<1x80x1xf16>

    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x80x1xf16>, tensor<1x80x1xf16> -> tensor<1x80x1xf16>

    return %MUL : tensor<1x80x1xf16>

    // CHECK-DAG:   [[MUL_WEIGHTS:%.+]] = const.Declare tensor<1x80x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<1x80x1xf16>

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 80, 1, 1]
    // CHECK-SAME:  } : tensor<1x80x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[MUL:%.+]] = IE.Multiply([[RESHAPE_INPUT]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[MUL]]) {
    // CHECK-SAME:      shape_value = [1, 80, 1]
    // CHECK-SAME:  } : tensor<1x80x1x1xf16> -> tensor<1x80x1xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x80x1xf16>
}

// -----

// CHECK-LABEL: @Convert3dAddWithFirstDim
func.func @Convert3dAddWithFirstDim(%arg0: tensor<80x1x1xf16>) -> tensor<80x1x1xf16> {
    %ADD_WEIGHTS = const.Declare tensor<80x1x1xf16> = dense<2.000000e+00> : tensor<80x1x1xf16>

    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<80x1x1xf16>, tensor<80x1x1xf16> -> tensor<80x1x1xf16>

    return %ADD : tensor<80x1x1xf16>

    // CHECK-DAG:   [[ADD_WEIGHTS:%.+]] = const.Declare tensor<1x80x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<80x1x1xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 80, 1, 1]
    // CHECK-SAME:  } : tensor<80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[ADD:%.+]] = IE.Add([[RESHAPE_INPUT]], [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[ADD]]) {
    // CHECK-SAME:      shape_value = [80, 1, 1]
    // CHECK-SAME:  } : tensor<1x80x1x1xf16> -> tensor<80x1x1xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<80x1x1xf16>
}

// -----

// CHECK-LABEL: func.func @AddOpInputWith4Dand1D
func.func @AddOpInputWith4Dand1D(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1xf16>) -> tensor<1x10x256x256xf16> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1xf16> -> tensor<1x10x256x256xf16>
    return %0 : tensor<1x10x256x256xf16>

    // CHECK:    [[Reshape_0:%.+]] = IE.AffineReshape(%arg1)
    // CHECK-SAME:      shape_value = [1, 1, 1, 1]} : tensor<1xf16> -> tensor<1x1x1x1xf16>
    // CHECK:    [[Result:%.+]] = IE.Add(%arg0, [[Reshape_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x10x256x256xf16>
    // CHECK:    return [[Result]]
}

// -----

// CHECK-LABEL: @Convert3dMulWithFirstDim
func.func @Convert3dMulWithFirstDim(%arg0: tensor<80x1x1xf16>) -> tensor<80x1x1xf16> {
    %MUL_WEIGHTS = const.Declare tensor<80x1x1xf16> = dense<2.000000e+00> : tensor<80x1x1xf16>

    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<80x1x1xf16>, tensor<80x1x1xf16> -> tensor<80x1x1xf16>

    return %MUL : tensor<80x1x1xf16>

    // CHECK-DAG:   [[MUL_WEIGHTS:%.+]] = const.Declare tensor<1x80x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<80x1x1xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 80, 1, 1]
    // CHECK-SAME:  } : tensor<80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[MUL:%.+]] = IE.Multiply([[RESHAPE_INPUT]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[MUL]]) {
    // CHECK-SAME:      shape_value = [80, 1, 1]
    // CHECK-SAME:  } : tensor<1x80x1x1xf16> -> tensor<80x1x1xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<80x1x1xf16>
}

// -----

// CHECK-LABEL: @Convert3dMulWithDifferentDim
func.func @Convert3dMulWithDifferentDim(%arg0: tensor<1x1x256xf16>) -> tensor<1x256x256xf16> {
    %MUL_WEIGHTS = const.Declare tensor<1x256x1xf16> = dense<2.000000e+00> : tensor<1x256x1xf16>

    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x1x256xf16>, tensor<1x256x1xf16> -> tensor<1x256x256xf16>

    return %MUL : tensor<1x256x256xf16>

    // CHECK-DAG:   [[MUL_WEIGHTS:%.+]] = const.Declare tensor<1x256x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> :  tensor<1x256x1xf16>, [#const.Reshape<[1, 256, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 1, 1, 256]
    // CHECK-SAME:  } : tensor<1x1x256xf16> -> tensor<1x1x1x256xf16>

    // CHECK:   [[MUL:%.+]] = IE.Multiply([[RESHAPE_INPUT]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x1x1x256xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x1x256xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[MUL]]) {
    // CHECK-SAME:      shape_value = [1, 256, 256]
    // CHECK-SAME:  } : tensor<1x256x1x256xf16> -> tensor<1x256x256xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x256x256xf16>
}

// -----

// CHECK-LABEL: @Convert2dAddWithLastDim
func.func @Convert2dAddWithLastDim(%arg0: tensor<1x80xf16>) -> tensor<1x80xf16> {
    %ADD_WEIGHTS = const.Declare tensor<1x80xf16> = dense<2.000000e+00> : tensor<1x80xf16>

    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x80xf16>, tensor<1x80xf16> -> tensor<1x80xf16>

    return %ADD : tensor<1x80xf16>

    // CHECK-DAG:   [[ADD_WEIGHTS:%.+]] = const.Declare tensor<1x80x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<1x80xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME{LITERAL}: dim_mapping = [[0], [1, 2, 3]],
    // CHECK-SAME:          shape_value = [1, 80, 1, 1]
    // CHECK-SAME:  } : tensor<1x80xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[ADD:%.+]] = IE.Add([[RESHAPE_INPUT]], [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[ADD]]) {
    // CHECK-SAME{LITERAL}: dim_mapping = [[0], [1], [1], [1]],
    // CHECK-SAME:          shape_value = [1, 80]
    // CHECK-SAME:  } : tensor<1x80x1x1xf16> -> tensor<1x80xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x80xf16>

}

// -----

// CHECK-LABEL: @Convert2dMulWithLastDim
func.func @Convert2dMulWithLastDim(%arg0: tensor<1x80xf16>) -> tensor<1x80xf16> {
    %MUL_WEIGHTS = const.Declare tensor<1x80xf16> = dense<2.000000e+00> : tensor<1x80xf16>

    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x80xf16>, tensor<1x80xf16> -> tensor<1x80xf16>

    return %MUL : tensor<1x80xf16>

    // CHECK-DAG:   [[MUL_WEIGHTS:%.+]] = const.Declare tensor<1x80x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<1x80xf16>

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 80, 1, 1]
    // CHECK-SAME:  } : tensor<1x80xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[MUL:%.+]] = IE.Multiply([[RESHAPE_INPUT]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[MUL]]) {
    // CHECK-SAME:      shape_value = [1, 80]
    // CHECK-SAME:  } : tensor<1x80x1x1xf16> -> tensor<1x80xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x80xf16>
}

// -----

// CHECK-LABEL: @Convert2dAddWithFirstDim
func.func @Convert2dAddWithFirstDim(%arg0: tensor<80x1xf16>) -> tensor<80x1xf16> {
    %ADD_WEIGHTS = const.Declare tensor<80x1xf16> = dense<2.000000e+00> : tensor<80x1xf16>

    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<80x1xf16>, tensor<80x1xf16> -> tensor<80x1xf16>

    return %ADD : tensor<80x1xf16>

    // CHECK-DAG:   [[ADD_WEIGHTS:%.+]] = const.Declare tensor<1x1x80x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<80x1xf16>, [#const.Reshape<[1, 1, 80, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 1, 80, 1]
    // CHECK-SAME:  } : tensor<80x1xf16> -> tensor<1x1x80x1xf16>

    // CHECK:   [[ADD:%.+]] = IE.Add([[RESHAPE_INPUT]], [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x1x80x1xf16>, tensor<1x1x80x1xf16> -> tensor<1x1x80x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[ADD]]) {
    // CHECK-SAME:      shape_value = [80, 1]
    // CHECK-SAME:  } : tensor<1x1x80x1xf16> -> tensor<80x1xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<80x1xf16>
}

// -----

// CHECK-LABEL: @Convert2dMulWithFirstDim
func.func @Convert2dMulWithFirstDim(%arg0: tensor<80x1xf16>) -> tensor<80x1xf16> {
    %MUL_WEIGHTS = const.Declare tensor<80x1xf16> = dense<2.000000e+00> : tensor<80x1xf16>

    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<80x1xf16>, tensor<80x1xf16> -> tensor<80x1xf16>

    return %MUL : tensor<80x1xf16>

    // CHECK-DAG:   [[MUL_WEIGHTS:%.+]] = const.Declare tensor<1x80x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<80x1xf16>

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 80, 1, 1]
    // CHECK-SAME:  } : tensor<80x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[MUL:%.+]] = IE.Multiply([[RESHAPE_INPUT]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[MUL]]) {
    // CHECK-SAME:      shape_value = [80, 1]
    // CHECK-SAME:  } : tensor<1x80x1x1xf16> -> tensor<80x1xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<80x1xf16>
}

// -----

// CHECK-LABEL: @Convert3DMulWithFirstDimLargeOne
func.func @Convert3DMulWithFirstDimLargeOne(%arg0: tensor<16x256x32xf32>) -> tensor<16x256x32xf32> {
    %0 = const.Declare tensor<16x256x1xf32> = dense<6.0> : tensor<16x256x1xf32>
    %1 = IE.Multiply(%arg0, %0) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<16x256x32xf32>, tensor<16x256x1xf32> -> tensor<16x256x32xf32>

    return %1 : tensor<16x256x32xf32>

    // CHECK-DAG:   %[[VAL_0:.+]] = const.Declare tensor<1x16x256x1xf32> = dense<6.000000e+00> : tensor<16x256x1xf32>, [#const.Reshape<[1, 16, 256, 1]>]
    // CHECK:       %[[VAL_0_4D:.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 16, 256, 32]} : tensor<16x256x32xf32> -> tensor<1x16x256x32xf32>
    // CHECK:       %[[VAL_1:.+]] = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x32xf32>, tensor<1x16x256x1xf32> -> tensor<1x16x256x32xf32>
    // CHECK:       %[[VAL_1_4D:.+]] = IE.AffineReshape(%[[VAL_1]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [2]], shape_value = [16, 256, 32]} : tensor<1x16x256x32xf32> -> tensor<16x256x32xf32>

    // CHECK:   return %[[VAL_1_4D]]
}

// -----

// CHECK-LABEL: @Convert7DMulWithDifferentDimTo4D
func.func @Convert7DMulWithDifferentDimTo4D(%arg0: tensor<1x1x9x8x8x30x54xf16>) -> tensor<1x2x9x8x8x30x54xf16> {
    %MUL_WEIGHTS = const.Declare tensor<1x2x9x1x1x30x54xf16> = dense<2.000000e+00> : tensor<1x2x9x1x1x30x54xf16>

    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x1x9x8x8x30x54xf16>, tensor<1x2x9x1x1x30x54xf16> -> tensor<1x2x9x8x8x30x54xf16>

    return %MUL : tensor<1x2x9x8x8x30x54xf16>

    // CHECK-DAG:   [[MUL_WEIGHTS:%.+]] = const.Declare tensor<2x9x1x1620xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<1x2x9x1x1x30x54xf16>, [#const.Reshape<[2, 9, 1, 1620]>]

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [0], [1], [2], [2], [3], [3]]
    // CHECK-SAME:              shape_value = [1, 9, 64, 1620]
    // CHECK-SAME:  } : tensor<1x1x9x8x8x30x54xf16> -> tensor<1x9x64x1620xf16>

    // CHECK:   [[MULTIPLY:%.+]] = IE.Multiply([[RESHAPE_INPUT]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:     auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x9x64x1620xf16>, tensor<2x9x1x1620xf16> -> tensor<2x9x64x1620xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[MULTIPLY]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1], [2], [3, 4], [5, 6]]
    // CHECK-SAME:              shape_value = [1, 2, 9, 8, 8, 30, 54]
    // CHECK-SAME:  } : tensor<2x9x64x1620xf16> -> tensor<1x2x9x8x8x30x54xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x2x9x8x8x30x54xf16>
}

// -----

// CHECK-LABEL: @Convert3DSubtractWithFirstDimLargeOne
func.func @Convert3DSubtractWithFirstDimLargeOne(%arg0: tensor<64x64x100xf32>, %arg1: tensor<64x1x100xf32>) -> tensor<64x64x100xf32> {
    %1 = IE.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<64x64x100xf32>, tensor<64x1x100xf32> -> tensor<64x64x100xf32>

    return %1 : tensor<64x64x100xf32>

    // CHECK:       %[[VAL_1:.+]] = IE.AffineReshape(%arg1)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 64, 1, 100]} : tensor<64x1x100xf32> -> tensor<1x64x1x100xf32>
    // CHECK:       %[[VAL_0:.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 64, 64, 100]} : tensor<64x64x100xf32> -> tensor<1x64x64x100xf32>
    // CHECK:       %[[SUBSTRACT:.+]] = IE.Subtract(%[[VAL_0]], %[[VAL_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x64x100xf32>, tensor<1x64x1x100xf32> -> tensor<1x64x64x100xf32>
    // CHECK:       %[[VAL_2:.+]] = IE.AffineReshape(%[[SUBSTRACT]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [2]], shape_value = [64, 64, 100]} : tensor<1x64x64x100xf32> -> tensor<64x64x100xf32>

    // CHECK:   return %[[VAL_2]]
}

// -----

// CHECK-LABEL: @Convert3DAddWithFirstDimLargeOne
func.func @Convert3DAddWithFirstDimLargeOne(%arg0: tensor<16x32x32xf16>) -> tensor<16x32x32xf16> {
    %ADD_WEIGHTS = const.Declare tensor<16x1x1xf16> = dense<2.000000e+00> : tensor<16x1x1xf16>

    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<16x32x32xf16>, tensor<16x1x1xf16> -> tensor<16x32x32xf16>

    return %ADD : tensor<16x32x32xf16>

    // CHECK-DAG:       [[VAL_0:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<2.000000e+00> : tensor<16x1x1xf16>, [#const.Reshape<[1, 16, 1, 1]>]
    // CHECK:       [[VAL_1:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 16, 32, 32]} : tensor<16x32x32xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       [[ADD:%.+]] = IE.Add([[VAL_1]], [[VAL_0]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x32xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       [[VAL_2:%.+]] = IE.AffineReshape([[ADD]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [2]], shape_value = [16, 32, 32]} : tensor<1x16x32x32xf16> -> tensor<16x32x32xf16>

    // CHECK:   return [[VAL_2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.956:128>

// CHECK: [[Q_ELEM_TYPE:!.+]] = !quant.uniform<u8:f16

func.func @Add3dMixPrecision(%arg0: tensor<12x77x64x!qElemType>, %arg1: tensor<12x77x64x!qElemType>) -> tensor<12x77x64x!qElemType> {
    %ADD = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<12x77x64x!qElemType>, tensor<12x77x64x!qElemType> -> tensor<12x77x64xf16>
    %QUANT = IE.Quantize(%ADD) {dstElemType = !qElemType} : tensor<12x77x64xf16> -> tensor<12x77x64x!qElemType>
    return %QUANT : tensor<12x77x64x!qElemType>

    // CHECK:       @Add3dMixPrecision([[ARG0:%.+]]: tensor<{{.+}}>, [[ARG1:%.+]]: tensor<{{.+}}>)
    // CHECK-DAG:       [[Reshape_1:%.+]] = IE.AffineReshape([[ARG1]]) {dim_mapping = {{\[\[}}0, 1], [2], [3]], shape_value = [1, 12, 77, 64]}
    // CHECK-DAG:       [[Reshape_0:%.+]] = IE.AffineReshape([[ARG0]]) {dim_mapping = {{\[\[}}0, 1], [2], [3]], shape_value = [1, 12, 77, 64]}
    // CHECK:           [[Add:%.+]] = IE.Add([[Reshape_0]], [[Reshape_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x77x64x[[Q_ELEM_TYPE]]>, tensor<1x12x77x64x[[Q_ELEM_TYPE]]> -> tensor<1x12x77x64xf16>
    // CHECK:           [[Reshape_out:%.+]] = IE.AffineReshape([[Add]]) {dim_mapping = {{\[\[}}0], [0], [1], [2]], shape_value = [12, 77, 64]}
    // CHECK:           [[Quant:%.+]] = IE.Quantize([[Reshape_out]])
    // CHECK:           return [[Quant]]
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @Convert3DTransposeWithFirstDimLargeOne
func.func @Convert3DTransposeWithFirstDimLargeOne(%arg0: tensor<512x4096x1xf16>) -> tensor<4096x512x1xf16> {
    %0 = IE.Transpose(%arg0) {order_value = affine_map<(d0, d1, d2) -> (d1, d0, d2)>} : tensor<512x4096x1xf16> -> tensor<4096x512x1xf16>

    return %0 : tensor<4096x512x1xf16>

    // CHECK:       [[VAL_0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1, 2], [3], [3]], shape_value = [1, 512, 1, 4096]} : tensor<512x4096x1xf16> -> tensor<1x512x1x4096xf16>
    // CHECK:       [[TRANS:%.+]] = IE.Transpose([[VAL_0]])
    // CHECK-SAME:      {order_value = #NWHC} : tensor<1x512x1x4096xf16> -> tensor<1x4096x1x512xf16>
    // CHECK:       [[VAL_1:%.+]] = IE.AffineReshape([[TRANS]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [0], [1, 2]], shape_value = [4096, 512, 1]} : tensor<1x4096x1x512xf16> -> tensor<4096x512x1xf16>

    // CHECK:   return [[VAL_1]]
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @Convert3DTransposeWithLast
func.func @Convert3DTransposeWithLast(%arg0: tensor<1x512x4096xf16>) -> tensor<1x4096x512xf16> {
    %0 = IE.Transpose(%arg0) {order_value = affine_map<(d0, d1, d2) -> (d0, d2, d1)>} : tensor<1x512x4096xf16> -> tensor<1x4096x512xf16>

    return %0 : tensor<1x4096x512xf16>

    // CHECK:       [[VAL_0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 512, 1, 4096]} : tensor<1x512x4096xf16> -> tensor<1x512x1x4096xf16>
    // CHECK:       [[TRANS:%.+]] = IE.Transpose([[VAL_0]])
    // CHECK-SAME:      {order_value = #NWHC} : tensor<1x512x1x4096xf16> -> tensor<1x4096x1x512xf16>
    // CHECK:       [[VAL_1:%.+]] = IE.AffineReshape([[TRANS]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [1], [1], [2]], shape_value = [1, 4096, 512]} : tensor<1x4096x1x512xf16> -> tensor<1x4096x512xf16>

    // CHECK:   return [[VAL_1]]
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @Convert2DTranspose
func.func @Convert2DTranspose(%arg0: tensor<4096x512xf16>) -> tensor<512x4096xf16> {
    %0 = IE.Transpose(%arg0) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<4096x512xf16> -> tensor<512x4096xf16>

    return %0 : tensor<512x4096xf16>

    // CHECK:       [[VAL_0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 4096, 1, 512]} : tensor<4096x512xf16> -> tensor<1x4096x1x512xf16>
    // CHECK:       [[TRANS:%.+]] = IE.Transpose([[VAL_0]])
    // CHECK-SAME:      {order_value = #NWHC} : tensor<1x4096x1x512xf16> -> tensor<1x512x1x4096xf16>
    // CHECK:       [[VAL_1:%.+]] = IE.AffineReshape([[TRANS]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [0], [1]], shape_value = [512, 4096]} : tensor<1x512x1x4096xf16> -> tensor<512x4096xf16>

    // CHECK:   return [[VAL_1]]
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @Convert5DTranspose
func.func @Convert5DTranspose(%arg0: tensor<1x3x1439x9x16xf16>) -> tensor<1x3x9x16x1439xf16> {
    %0 = IE.Transpose(%arg0) {order_value = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>} : tensor<1x3x1439x9x16xf16> -> tensor<1x3x9x16x1439xf16>

    return %0 : tensor<1x3x9x16x1439xf16>

    // CHECK:       [[VAL_0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [1], [2], [3], [3]], shape_value = [1, 3, 1439, 144]} : tensor<1x3x1439x9x16xf16> -> tensor<1x3x1439x144xf16
    // CHECK:       [[TRANS:%.+]] = IE.Transpose([[VAL_0]])
    // CHECK-SAME:      {order_value = #NCWH} : tensor<1x3x1439x144xf16> -> tensor<1x3x144x1439xf16>
    // CHECK:       [[VAL_1:%.+]] = IE.AffineReshape([[TRANS]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 3, 9, 16, 1439]} : tensor<1x3x144x1439xf16> -> tensor<1x3x9x16x1439xf16>

    // CHECK:   return [[VAL_1]]
}

// -----

// CHECK-LABEL: func.func @ConvertShapeTo4DStridedSlice
func.func @ConvertShapeTo4DStridedSlice(%arg0: tensor<4004x320xf16>) -> (tensor<4004x160xf16>) {
    %0 = IE.StridedSlice(%arg0) {begin_mask = [0, 1], begins_attr = [0, 0], ellipsis_mask = [0, 0], end_mask = [1, 0], ends_attr = [4004, 320], new_axis_mask = [0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0], strides_attr = [1, 2]} : tensor<4004x320xf16> -> tensor<4004x160xf16>
    return %0 : tensor<4004x160xf16>

    // CHECK:       [[Reshape_0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [1, 2, 3]], shape_value = [4004, 320, 1, 1]} : tensor<4004x320xf16> -> tensor<4004x320x1x1xf16>
    // CHECK:       %[[STRIDEDSLICE:.+]] = IE.StridedSlice(%0) {begin_mask = [0, 1, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [1, 0, 0, 0], ends_attr = [4004, 320, 1, 1], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 2, 1, 1]} : tensor<4004x320x1x1xf16> -> tensor<4004x160x1x1xf16>
    // CHECK:       %[[Reshape_1:.+]] = IE.AffineReshape(%[[STRIDEDSLICE]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [1], [1], [1]], shape_value = [4004, 160]} : tensor<4004x160x1x1xf16> -> tensor<4004x160xf16>
    // CHECK:    return %[[Reshape_1]]
}

// -----

// CHECK-LABEL: func.func @ConvertShapeTo4DFrom5DStridedSlice
func.func @ConvertShapeTo4DFrom5DStridedSlice(%arg0: tensor<1x5x20x32x32xf16>) -> (tensor<1x5x20x32x16xf16>) {
    %0 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0, 0], begins_attr = [0, 0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0, 0], end_mask = [0, 0, 0, 0, 0], ends_attr = [1, 5, 20, 32, 32], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1, 2]} : tensor<1x5x20x32x32xf16> -> tensor<1x5x20x32x16xf16>
    return %0 : tensor<1x5x20x32x16xf16>

    // CHECK:       [[Reshape_0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [5, 20, 32, 32]} : tensor<1x5x20x32x32xf16> -> tensor<5x20x32x32xf16>
    // CHECK:       %[[STRIDEDSLICE:.+]] = IE.StridedSlice(%0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [5, 20, 32, 32], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<5x20x32x32xf16> -> tensor<5x20x32x16xf16>
    // CHECK:       %[[Reshape_1:.+]] = IE.AffineReshape(%[[STRIDEDSLICE]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 5, 20, 32, 16]} : tensor<5x20x32x16xf16> -> tensor<1x5x20x32x16xf16>
    // CHECK:    return %[[Reshape_1]]
}

// -----

// CHECK-LABEL: func.func @ConvertShapeTo4DFrom6DStridedSlice
func.func @ConvertShapeTo4DFrom6DStridedSlice(%arg0: tensor<1x1x5x20x32x32xf16>) -> (tensor<1x1x5x20x32x16xf16>) {
    %0 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0, 0, 0], begins_attr = [0, 0, 0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0, 0, 0], end_mask = [0, 0, 0, 0, 0, 0], ends_attr = [1, 1, 5, 20, 32, 32], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1, 1, 2]} : tensor<1x1x5x20x32x32xf16> -> tensor<1x1x5x20x32x16xf16>
    return %0 : tensor<1x1x5x20x32x16xf16>

    // CHECK:       [[Reshape_0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [0], [1], [2], [3]], shape_value = [5, 20, 32, 32]} : tensor<1x1x5x20x32x32xf16> -> tensor<5x20x32x32xf16>
    // CHECK:       %[[STRIDEDSLICE:.+]] = IE.StridedSlice(%0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [5, 20, 32, 32], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<5x20x32x32xf16> -> tensor<5x20x32x16xf16>
    // CHECK:       %[[Reshape_1:.+]] = IE.AffineReshape(%[[STRIDEDSLICE]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1, 2], [3], [4], [5]], shape_value = [1, 1, 5, 20, 32, 16]} : tensor<5x20x32x16xf16> -> tensor<1x1x5x20x32x16xf16>
    // CHECK:    return %[[Reshape_1]]
}

// -----

// CHECK-LABEL: func.func @ConvertConcat6DTo4D
func.func @ConvertConcat6DTo4D(%arg0: tensor<1x2x3x4x5x6xf16>, %arg1: tensor<1x2x3x4x5x6xf16>, %arg2: tensor<1x2x3x4x5x6xf16>) -> tensor<1x2x3x12x5x6xf16> {
    %0 = IE.Concat(%arg2, %arg1, %arg0) {static_offsets = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 4, 0, 0], [0, 0, 0, 8, 0, 0]]} : tensor<1x2x3x4x5x6xf16>, tensor<1x2x3x4x5x6xf16>, tensor<1x2x3x4x5x6xf16> -> tensor<1x2x3x12x5x6xf16>
    return %0 : tensor<1x2x3x12x5x6xf16>

    // CHECK:             [[AFFINERESHAPE0:%.+]] = IE.AffineReshape({{[^:]+}})
    // CHECK{LITERAL}:    {dim_mapping = [[0], [1], [1], [2], [3], [3]], shape_value = [1, 6, 4, 30]} : tensor<1x2x3x4x5x6xf16> -> tensor<1x6x4x30xf16>
    // CHECK:             [[AFFINERESHAPE1:%.+]] = IE.AffineReshape({{[^:]+}})
    // CHECK{LITERAL}:    {dim_mapping = [[0], [1], [1], [2], [3], [3]], shape_value = [1, 6, 4, 30]} : tensor<1x2x3x4x5x6xf16> -> tensor<1x6x4x30xf16>
    // CHECK:             [[AFFINERESHAPE2:%.+]] = IE.AffineReshape({{[^:]+}})
    // CHECK{LITERAL}:    {dim_mapping = [[0], [1], [1], [2], [3], [3]], shape_value = [1, 6, 4, 30]} : tensor<1x2x3x4x5x6xf16> -> tensor<1x6x4x30xf16>
    // CHECK:             [[CONCAT0:%.+]] = IE.Concat([[AFFINERESHAPE0]], [[AFFINERESHAPE1]], [[AFFINERESHAPE2]])
    // CHECK{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0]]} : tensor<1x6x4x30xf16>, tensor<1x6x4x30xf16>, tensor<1x6x4x30xf16> -> tensor<1x6x12x30xf16>
    // CHECK:             [[AFFINERESHAPE3:%.+]] = IE.AffineReshape([[CONCAT0]])
    // CHECK{LITERAL}:    {dim_mapping = [[0], [1, 2], [3], [4, 5]], shape_value = [1, 2, 3, 12, 5, 6]} : tensor<1x6x12x30xf16> -> tensor<1x2x3x12x5x6xf16>

    // CHECK:              return [[AFFINERESHAPE3]] : tensor<1x2x3x12x5x6xf16>

}

// -----

// CHECK-LABEL: func.func @ConvertConcat5DTo4D
func.func @ConvertConcat5DTo4D(%arg0: tensor<1x2x3x4x5xf16>, %arg1: tensor<1x2x3x4x5xf16>, %arg2: tensor<1x2x3x4x5xf16>) -> tensor<1x2x9x4x5xf16> {
    %0 = IE.Concat(%arg2, %arg1, %arg0) {static_offsets = [[0, 0, 0, 0, 0], [0, 0, 3, 0, 0], [0, 0, 6, 0, 0]]} : tensor<1x2x3x4x5xf16>, tensor<1x2x3x4x5xf16>, tensor<1x2x3x4x5xf16> -> tensor<1x2x9x4x5xf16>
    return %0 : tensor<1x2x9x4x5xf16>

    // CHECK:             [[AFFINERESHAPE0:%.+]] = IE.AffineReshape({{[^:]+}})
    // CHECK{LITERAL}:    {dim_mapping = [[0], [1], [2], [3], [3]], shape_value = [1, 2, 3, 20]} : tensor<1x2x3x4x5xf16> -> tensor<1x2x3x20xf16>
    // CHECK:             [[AFFINERESHAPE1:%.+]] = IE.AffineReshape({{[^:]+}})
    // CHECK{LITERAL}:    {dim_mapping = [[0], [1], [2], [3], [3]], shape_value = [1, 2, 3, 20]} : tensor<1x2x3x4x5xf16> -> tensor<1x2x3x20xf16>
    // CHECK:             [[AFFINERESHAPE2:%.+]] = IE.AffineReshape({{[^:]+}})
    // CHECK{LITERAL}:    {dim_mapping = [[0], [1], [2], [3], [3]], shape_value = [1, 2, 3, 20]} : tensor<1x2x3x4x5xf16> -> tensor<1x2x3x20xf16>
    // CHECK:             [[CONCAT0:%.+]] = IE.Concat([[AFFINERESHAPE0]], [[AFFINERESHAPE1]], [[AFFINERESHAPE2]])
    // CHECK{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0]]} : tensor<1x2x3x20xf16>, tensor<1x2x3x20xf16>, tensor<1x2x3x20xf16> -> tensor<1x2x9x20xf16>
    // CHECK:             [[AFFINERESHAPE3:%.+]] = IE.AffineReshape([[CONCAT0]])
    // CHECK{LITERAL}:    {dim_mapping = [[0], [1], [2], [3, 4]], shape_value = [1, 2, 9, 4, 5]} : tensor<1x2x9x20xf16> -> tensor<1x2x9x4x5xf16>

    // CHECK:              return [[AFFINERESHAPE3]] : tensor<1x2x9x4x5xf16>

}

// -----

// CHECK-LABEL: func.func @ConvertConcat1DTo4D
func.func @ConvertConcat1DTo4D(%arg0: tensor<1xf16>, %arg1: tensor<1xf16>, %arg2: tensor<1xf16>) -> tensor<3xf16> {
    %0 = IE.Concat(%arg2, %arg1, %arg0) {static_offsets = [[0], [1], [2]]} : tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<3xf16>
    return %0 : tensor<3xf16>

    // CHECK:             [[AFFINERESHAPE0:%.+]] = IE.AffineReshape({{[^:]+}})
    // CHECK{LITERAL}:    {dim_mapping = [[0, 1, 2, 3]], shape_value = [1, 1, 1, 1]} : tensor<1xf16> -> tensor<1x1x1x1xf16>
    // CHECK:             [[AFFINERESHAPE1:%.+]] = IE.AffineReshape({{[^:]+}})
    // CHECK{LITERAL}:    {dim_mapping = [[0, 1, 2, 3]], shape_value = [1, 1, 1, 1]} : tensor<1xf16> -> tensor<1x1x1x1xf16>
    // CHECK:             [[AFFINERESHAPE2:%.+]] = IE.AffineReshape({{[^:]+}})
    // CHECK{LITERAL}:    {dim_mapping = [[0, 1, 2, 3]], shape_value = [1, 1, 1, 1]} : tensor<1xf16> -> tensor<1x1x1x1xf16>
    // CHECK:             [[CONCAT0:%.+]] = IE.Concat([[AFFINERESHAPE0]], [[AFFINERESHAPE1]], [[AFFINERESHAPE2]])
    // CHECK{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0]]} : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x3x1xf16>
    // CHECK:             [[AFFINERESHAPE3:%.+]] = IE.AffineReshape([[CONCAT0]])
    // CHECK{LITERAL}:    {dim_mapping = [[0], [0], [0], [0]], shape_value = [3]} : tensor<1x1x3x1xf16> -> tensor<3xf16>

    // CHECK:              return [[AFFINERESHAPE3]] : tensor<3xf16>

}

// -----

// CHECK-LABEL: func.func @ConvertConcatTo4DWithoutOffset
func.func @ConvertConcatTo4DWithoutOffset(%arg0: tensor<1x3x40x40x5xf16>, %arg1: tensor<1x3x40x40x5xf16>, %arg2: tensor<1x3x40x40x5xf16>) -> tensor<1x3x40x40x15xf16> {
    %0 = IE.Concat(%arg0, %arg1, %arg2) {per_axis = #IE.Concat<axis = 4 : i64, offset = 1 : i64, stride = 3 : i64>} : tensor<1x3x40x40x5xf16>, tensor<1x3x40x40x5xf16>, tensor<1x3x40x40x5xf16> -> tensor<1x3x40x40x15xf16>
    return %0 : tensor<1x3x40x40x15xf16>

    // CHECK:             [[AFFINERESHAPE0:%.+]] = IE.AffineReshape({{[^:]+}})
    // CHECK{LITERAL}:    {dim_mapping = [[0], [1], [1], [1], [2, 3]], shape_value = [1, 4800, 5, 1]} : tensor<1x3x40x40x5xf16> -> tensor<1x4800x5x1xf16>
    // CHECK:             [[AFFINERESHAPE1:%.+]] = IE.AffineReshape({{[^:]+}})
    // CHECK{LITERAL}:    {dim_mapping = [[0], [1], [1], [1], [2, 3]], shape_value = [1, 4800, 5, 1]} : tensor<1x3x40x40x5xf16> -> tensor<1x4800x5x1xf16>
    // CHECK:             [[AFFINERESHAPE2:%.+]] = IE.AffineReshape({{[^:]+}})
    // CHECK{LITERAL}:    {dim_mapping = [[0], [1], [1], [1], [2, 3]], shape_value = [1, 4800, 5, 1]} : tensor<1x3x40x40x5xf16> -> tensor<1x4800x5x1xf16>
    // CHECK:             [[CONCAT0:%.+]] = IE.Concat([[AFFINERESHAPE0]], [[AFFINERESHAPE1]], [[AFFINERESHAPE2]])
    // CHECK{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0]]} : tensor<1x4800x5x1xf16>, tensor<1x4800x5x1xf16>, tensor<1x4800x5x1xf16> -> tensor<1x4800x15x1xf16>
    // CHECK:             [[AFFINERESHAPE3:%.+]] = IE.AffineReshape([[CONCAT0]])
    // CHECK{LITERAL}:    {dim_mapping = [[0], [1, 2, 3], [4], [4]], shape_value = [1, 3, 40, 40, 15]} : tensor<1x4800x15x1xf16> -> tensor<1x3x40x40x15xf16>

    // CHECK:              return [[AFFINERESHAPE3]] : tensor<1x3x40x40x15xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertConcatTo4DWithoutOffsetAndNegativeAxis
func.func @ConvertConcatTo4DWithoutOffsetAndNegativeAxis(%arg0: tensor<1x3x40x40x5xf16>, %arg1: tensor<1x3x40x40x5xf16>, %arg2: tensor<1x3x40x40x5xf16>) -> tensor<1x3x40x40x15xf16> {
    %0 = IE.Concat(%arg0, %arg1, %arg2) {per_axis = #IE.Concat<axis = -1 : i64, offset = 1 : i64, stride = 3 : i64>} : tensor<1x3x40x40x5xf16>, tensor<1x3x40x40x5xf16>, tensor<1x3x40x40x5xf16> -> tensor<1x3x40x40x15xf16>
    return %0 : tensor<1x3x40x40x15xf16>

    // CHECK:             [[AFFINERESHAPE0:%.+]] = IE.AffineReshape({{[^:]+}})
    // CHECK{LITERAL}:    {dim_mapping = [[0], [1], [1], [1], [2, 3]], shape_value = [1, 4800, 5, 1]} : tensor<1x3x40x40x5xf16> -> tensor<1x4800x5x1xf16>
    // CHECK:             [[AFFINERESHAPE1:%.+]] = IE.AffineReshape({{[^:]+}})
    // CHECK{LITERAL}:    {dim_mapping = [[0], [1], [1], [1], [2, 3]], shape_value = [1, 4800, 5, 1]} : tensor<1x3x40x40x5xf16> -> tensor<1x4800x5x1xf16>
    // CHECK:             [[AFFINERESHAPE2:%.+]] = IE.AffineReshape({{[^:]+}})
    // CHECK{LITERAL}:    {dim_mapping = [[0], [1], [1], [1], [2, 3]], shape_value = [1, 4800, 5, 1]} : tensor<1x3x40x40x5xf16> -> tensor<1x4800x5x1xf16>
    // CHECK:             [[CONCAT0:%.+]] = IE.Concat([[AFFINERESHAPE0]], [[AFFINERESHAPE1]], [[AFFINERESHAPE2]])
    // CHECK{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0]]} : tensor<1x4800x5x1xf16>, tensor<1x4800x5x1xf16>, tensor<1x4800x5x1xf16> -> tensor<1x4800x15x1xf16>
    // CHECK:             [[AFFINERESHAPE3:%.+]] = IE.AffineReshape([[CONCAT0]])
    // CHECK{LITERAL}:    {dim_mapping = [[0], [1, 2, 3], [4], [4]], shape_value = [1, 3, 40, 40, 15]} : tensor<1x4800x15x1xf16> -> tensor<1x3x40x40x15xf16>

    // CHECK:              return [[AFFINERESHAPE3]] : tensor<1x3x40x40x15xf16>
}

// -----

// CHECK-LABEL: @Convert2DSoftmax
func.func @Convert2DSoftmax(%arg0: tensor<4096x512xf16>) -> tensor<4096x512xf16> {
    %0 = IE.SoftMax(%arg0) {axisInd = 1} : tensor<4096x512xf16> -> tensor<4096x512xf16>

    return %0 : tensor<4096x512xf16>

    // CHECK:       [[VAL_0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 4096, 512]} : tensor<4096x512xf16> -> tensor<1x1x4096x512xf16>
    // CHECK:       [[Softmax:%.+]] = IE.SoftMax([[VAL_0]])
    // CHECK-SAME:      {axisInd = 3 : i64} : tensor<1x1x4096x512xf16> -> tensor<1x1x4096x512xf16>
    // CHECK:       [[VAL_1:%.+]] = IE.AffineReshape([[Softmax]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [0], [1]], shape_value = [4096, 512]} : tensor<1x1x4096x512xf16> -> tensor<4096x512xf16>

    // CHECK:   return [[VAL_1]]
}

// -----

// CHECK-LABEL: @Convert3DSoftmax
func.func @Convert3DSoftmax(%arg0: tensor<8x4096x512xf16>) -> tensor<8x4096x512xf16> {
    %0 = IE.SoftMax(%arg0) {axisInd = 2} : tensor<8x4096x512xf16> -> tensor<8x4096x512xf16>

    return %0 : tensor<8x4096x512xf16>

    // CHECK:       [[VAL_0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 8, 4096, 512]} : tensor<8x4096x512xf16> -> tensor<1x8x4096x512xf16>
    // CHECK:       [[SoftMax:%.+]] = IE.SoftMax([[VAL_0]])
    // CHECK-SAME:      {axisInd = 3 : i64} : tensor<1x8x4096x512xf16> -> tensor<1x8x4096x512xf16>
    // CHECK:       [[VAL_1:%.+]] = IE.AffineReshape([[SoftMax]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [2]], shape_value = [8, 4096, 512]} : tensor<1x8x4096x512xf16> -> tensor<8x4096x512xf16>

    // CHECK:   return [[VAL_1]]
}

// -----

// CHECK-LABEL: @Convert7DSoftmax
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x9x8x8x30x54xf16>) -> tensor<1x1x9x8x8x30x54xf16> {
func.func @Convert7DSoftmax(%arg0: tensor<1x1x9x8x8x30x54xf16>) -> tensor<1x1x9x8x8x30x54xf16> {
    %0 = IE.SoftMax(%arg0) {axisInd = 2} : tensor<1x1x9x8x8x30x54xf16> -> tensor<1x1x9x8x8x30x54xf16>

    return %0 : tensor<1x1x9x8x8x30x54xf16>

    // CHECK:       [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [2], [2], [2], [2, 3]], shape_value = [1, 9, 103680, 1]} : tensor<1x1x9x8x8x30x54xf16> -> tensor<1x9x103680x1xf16>

    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[AFFINE_RESHAPE]])
    // CHECK-SAME:      {axisInd = 1 : i64} : tensor<1x9x103680x1xf16> -> tensor<1x9x103680x1xf16>

    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[SOFTMAX]])
    // CHECK-SAME{LITERAL}:     {shape_value = [1, 1, 9, 8, 8, 30, 54]} : tensor<1x9x103680x1xf16> -> tensor<1x1x9x8x8x30x54xf16>

    // CHECK:   return [[RESHAPE]]
}

// -----

// CHECK-LABEL: @Convert7DSoftmaxWithAxisOnLastDim
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x9x8x8x30x54xf16>) -> tensor<1x1x9x8x8x30x54xf16> {
func.func @Convert7DSoftmaxWithAxisOnLastDim(%arg0: tensor<1x1x9x8x8x30x54xf16>) -> tensor<1x1x9x8x8x30x54xf16> {
    %0 = IE.SoftMax(%arg0) {axisInd = 6} : tensor<1x1x9x8x8x30x54xf16> -> tensor<1x1x9x8x8x30x54xf16>

    return %0 : tensor<1x1x9x8x8x30x54xf16>

    // CHECK:       [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [1], [1], [1], [2, 3]], shape_value = [1, 17280, 54, 1]} : tensor<1x1x9x8x8x30x54xf16> -> tensor<1x17280x54x1xf16>

    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[AFFINE_RESHAPE]])
    // CHECK-SAME:               {axisInd = 2 : i64} : tensor<1x17280x54x1xf16> -> tensor<1x17280x54x1xf16>

    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[SOFTMAX]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2, 3, 4, 5], [6], [6]], shape_value = [1, 1, 9, 8, 8, 30, 54]} : tensor<1x17280x54x1xf16> -> tensor<1x1x9x8x8x30x54xf16>

    // CHECK:   return [[RESHAPE]]
}

// -----

// CHECK-LABEL: @Convert7DSoftmaxWithAxisOnFirstDim
// CHECK-SAME:      [[INPUT:%.+]]: tensor<8x1x9x8x1x30x54xf16>) -> tensor<8x1x9x8x1x30x54xf16> {
func.func @Convert7DSoftmaxWithAxisOnFirstDim(%arg0: tensor<8x1x9x8x1x30x54xf16>) -> tensor<8x1x9x8x1x30x54xf16> {
    %0 = IE.SoftMax(%arg0) {axisInd = 0} : tensor<8x1x9x8x1x30x54xf16> -> tensor<8x1x9x8x1x30x54xf16>

    return %0 : tensor<8x1x9x8x1x30x54xf16>

    // CHECK:       [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [2], [2], [2], [2], [2, 3]], shape_value = [1, 8, 116640, 1]} : tensor<8x1x9x8x1x30x54xf16> -> tensor<1x8x116640x1xf16>

    // CHECK:         [[SOFTMAX:%.+]] = IE.SoftMax([[AFFINE_RESHAPE]])
    // CHECK-SAME:               {axisInd = 1 : i64} : tensor<1x8x116640x1xf16> -> tensor<1x8x116640x1xf16>

    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[SOFTMAX]])
    // CHECK-SAME{LITERAL}:      {shape_value = [8, 1, 9, 8, 1, 30, 54]} : tensor<1x8x116640x1xf16> -> tensor<8x1x9x8x1x30x54xf16>

    // CHECK:         return [[RESHAPE]]
}

// -----

// CHECK-LABEL: func.func @Convert2DReduceL1KeepDims(
// CHECK-SAME:      %[[VAL_0:.+]]: tensor<7x7xf32>) -> tensor<1x1xf32> {
func.func @Convert2DReduceL1KeepDims(%arg0: tensor<7x7xf32>) -> tensor<1x1xf32> {
  %0 = IE.ReduceL1(%arg0) {axes_value = [0, 1], keep_dims} : tensor<7x7xf32> -> tensor<1x1xf32>
  return %0 : tensor<1x1xf32>

    // CHECK:   %[[VAL_1:.+]] = IE.AffineReshape(%[[VAL_0]]) {dim_mapping = {{\[\[}}0, 1, 2], [3]], shape_value = [1, 1, 7, 7]} : tensor<7x7xf32> -> tensor<1x1x7x7xf32>
    // CHECK:   %[[VAL_2:.+]] = IE.ReduceL1(%[[VAL_1]]) {axes_value = [2, 3], keep_dims} : tensor<1x1x7x7xf32> -> tensor<1x1x1x1xf32>
    // CHECK:   %[[VAL_3:.+]] = IE.AffineReshape(%[[VAL_2]]) {dim_mapping = {{\[\[}}0], [1], [1], [1]], shape_value = [1, 1]} : tensor<1x1x1x1xf32> -> tensor<1x1xf32>
    // CHECK:   return %[[VAL_3]] : tensor<1x1xf32>
}

// -----

// CHECK-LABEL: func.func @Convert2DReduceL2(
// CHECK-SAME:      %[[VAL_0:.+]]: tensor<7x7xf32>) -> tensor<1xf32> {
func.func @Convert2DReduceL2(%arg0: tensor<7x7xf32>) -> tensor<1xf32> {
  %0 = IE.ReduceL2(%arg0) {axes_value = [0, 1]} : tensor<7x7xf32> -> tensor<1xf32>
  return %0 : tensor<1xf32>

// CHECK:   %[[VAL_1:.+]] = IE.AffineReshape(%[[VAL_0]]) {dim_mapping = {{\[\[}}0, 1, 2], [3]], shape_value = [1, 1, 7, 7]} : tensor<7x7xf32> -> tensor<1x1x7x7xf32>
// CHECK:   %[[VAL_2:.+]] = IE.ReduceL2(%[[VAL_1]]) {axes_value = [2, 3], keep_dims} : tensor<1x1x7x7xf32> -> tensor<1x1x1x1xf32>
// CHECK:   %[[VAL_3:.+]] = IE.AffineReshape(%[[VAL_2]]) {dim_mapping = {{\[\[}}0], [0], [0], [0]], shape_value = [1]} : tensor<1x1x1x1xf32> -> tensor<1xf32>
// CHECK:   return %[[VAL_3]] : tensor<1xf32>
}

// -----

// CHECK-LABEL: func.func @Convert4DReduceLogicalAnd(
// CHECK-SAME:      %[[VAL_0:.+]]: tensor<1x1024x7x7xf32>) -> tensor<1x1024xf32> {
func.func @Convert4DReduceLogicalAnd(%arg0: tensor<1x1024x7x7xf32>) -> tensor<1x1024xf32> {
  %0 = IE.ReduceLogicalAnd(%arg0) {axes_value = [2, 3]} : tensor<1x1024x7x7xf32> -> tensor<1x1024xf32>
  return %0 : tensor<1x1024xf32>

// CHECK:   %[[VAL_1:.+]] = IE.ReduceLogicalAnd(%[[VAL_0]]) {axes_value = [2, 3], keep_dims} : tensor<1x1024x7x7xf32> -> tensor<1x1024x1x1xf32>
// CHECK:   %[[VAL_2:.+]] = IE.AffineReshape(%[[VAL_1]]) {dim_mapping = {{\[\[}}0], [1], [1], [1]], shape_value = [1, 1024]} : tensor<1x1024x1x1xf32> -> tensor<1x1024xf32>
// CHECK:   return %[[VAL_2]] : tensor<1x1024xf32>
}

// -----

// CHECK-LABEL: func.func @Convert3DReduceLogicalOr(
// CHECK-SAME:      %[[VAL_0:.+]]: tensor<1024x7x7xf32>) -> tensor<1024xf32> {
func.func @Convert3DReduceLogicalOr(%arg0: tensor<1024x7x7xf32>) -> tensor<1024xf32> {
  %0 = IE.ReduceLogicalOr(%arg0) {axes_value = [1, 2]} : tensor<1024x7x7xf32> -> tensor<1024xf32>
  return %0 : tensor<1024xf32>

// CHECK:   %[[VAL_1:.+]] = IE.AffineReshape(%[[VAL_0]]) {dim_mapping = {{\[\[}}0, 1], [2], [3]], shape_value = [1, 1024, 7, 7]} : tensor<1024x7x7xf32> -> tensor<1x1024x7x7xf32>
// CHECK:   %[[VAL_2:.+]] = IE.ReduceLogicalOr(%[[VAL_1]]) {axes_value = [2, 3], keep_dims} : tensor<1x1024x7x7xf32> -> tensor<1x1024x1x1xf32>
// CHECK:   %[[VAL_3:.+]] = IE.AffineReshape(%[[VAL_2]]) {dim_mapping = {{\[\[}}0], [0], [0], [0]], shape_value = [1024]} : tensor<1x1024x1x1xf32> -> tensor<1024xf32>
// CHECK:   return %[[VAL_3]] : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @Convert3DReduceMax(
// CHECK-SAME:      %[[VAL_0:.+]]: tensor<1024x7x7xf32>) -> tensor<1024xf32> {
func.func @Convert3DReduceMax(%arg0: tensor<1024x7x7xf32>) -> tensor<1024xf32> {
  %0 = IE.ReduceMax(%arg0) {axes_value = [1, 2]} : tensor<1024x7x7xf32> -> tensor<1024xf32>
  return %0 : tensor<1024xf32>

// CHECK:   %[[VAL_1:.+]] = IE.AffineReshape(%[[VAL_0]]) {dim_mapping = {{\[\[}}0, 1], [2], [3]], shape_value = [1, 1024, 7, 7]} : tensor<1024x7x7xf32> -> tensor<1x1024x7x7xf32>
// CHECK:   %[[VAL_2:.+]] = IE.ReduceMax(%[[VAL_1]]) {axes_value = [2, 3], keep_dims} : tensor<1x1024x7x7xf32> -> tensor<1x1024x1x1xf32>
// CHECK:   %[[VAL_3:.+]] = IE.AffineReshape(%[[VAL_2]]) {dim_mapping = {{\[\[}}0], [0], [0], [0]], shape_value = [1024]} : tensor<1x1024x1x1xf32> -> tensor<1024xf32>
// CHECK:   return %[[VAL_3]] : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @Convert3DReduceMean(
// CHECK-SAME:      %[[VAL_0:.+]]: tensor<1024x7x7xf32>) -> tensor<1024xf32> {
func.func @Convert3DReduceMean(%arg0: tensor<1024x7x7xf32>) -> tensor<1024xf32> {
  %0 = IE.ReduceMean(%arg0) {axes_value = [1, 2]} : tensor<1024x7x7xf32> -> tensor<1024xf32>
  return %0 : tensor<1024xf32>

// CHECK:   %[[VAL_1:.+]] = IE.AffineReshape(%[[VAL_0]]) {dim_mapping = {{\[\[}}0, 1], [2], [3]], shape_value = [1, 1024, 7, 7]} : tensor<1024x7x7xf32> -> tensor<1x1024x7x7xf32>
// CHECK:   %[[VAL_2:.+]] = IE.ReduceMean(%[[VAL_1]]) {axes_value = [2, 3], keep_dims} : tensor<1x1024x7x7xf32> -> tensor<1x1024x1x1xf32>
// CHECK:   %[[VAL_3:.+]] = IE.AffineReshape(%[[VAL_2]]) {dim_mapping = {{\[\[}}0], [0], [0], [0]], shape_value = [1024]} : tensor<1x1024x1x1xf32> -> tensor<1024xf32>
// CHECK:   return %[[VAL_3]] : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @Convert3DReduceMin(
// CHECK-SAME:      %[[VAL_0:.+]]: tensor<1024x7x7xf32>) -> tensor<1024xf32> {
func.func @Convert3DReduceMin(%arg0: tensor<1024x7x7xf32>) -> tensor<1024xf32> {
  %0 = IE.ReduceMin(%arg0) {axes_value = [1, 2]} : tensor<1024x7x7xf32> -> tensor<1024xf32>
  return %0 : tensor<1024xf32>

// CHECK:   %[[VAL_1:.+]] = IE.AffineReshape(%[[VAL_0]]) {dim_mapping = {{\[\[}}0, 1], [2], [3]], shape_value = [1, 1024, 7, 7]} : tensor<1024x7x7xf32> -> tensor<1x1024x7x7xf32>
// CHECK:   %[[VAL_2:.+]] = IE.ReduceMin(%[[VAL_1]]) {axes_value = [2, 3], keep_dims} : tensor<1x1024x7x7xf32> -> tensor<1x1024x1x1xf32>
// CHECK:   %[[VAL_3:.+]] = IE.AffineReshape(%[[VAL_2]]) {dim_mapping = {{\[\[}}0], [0], [0], [0]], shape_value = [1024]} : tensor<1x1024x1x1xf32> -> tensor<1024xf32>
// CHECK:   return %[[VAL_3]] : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @Convert5DReduceProd(
// CHECK-SAME:      %[[VAL_0:.+]]: tensor<2x1024x7x7x3xf32>) -> tensor<2x1024x3xf32> {
func.func @Convert5DReduceProd(%arg0: tensor<2x1024x7x7x3xf32>) -> tensor<2x1024x3xf32> {
  %0 = IE.ReduceProd(%arg0) {axes_value = [2, 3]} : tensor<2x1024x7x7x3xf32> -> tensor<2x1024x3xf32>
  return %0 : tensor<2x1024x3xf32>

// CHECK:   %[[VAL_1:.+]] = IE.Reshape(%[[VAL_0]]) {shape_value = [1, 2048, 49, 3]} : tensor<2x1024x7x7x3xf32> -> tensor<1x2048x49x3xf32>
// CHECK:   %[[VAL_2:.+]] = IE.ReduceProd(%[[VAL_1]]) {axes_value = [2], keep_dims} : tensor<1x2048x49x3xf32> -> tensor<1x2048x1x3xf32>
// CHECK:   %[[VAL_3:.+]] = IE.Reshape(%[[VAL_2]]) {shape_value = [2, 1024, 3]} : tensor<1x2048x1x3xf32> -> tensor<2x1024x3xf32>
// CHECK:   return %[[VAL_3]] : tensor<2x1024x3xf32>
}

// -----

// CHECK-LABEL: func.func @Convert5DReduceSumKeepDims(
// CHECK-SAME:      %[[VAL_0:.+]]: tensor<2x1024x7x7x3xf32>) -> tensor<1x1x7x1x1xf32> {
func.func @Convert5DReduceSumKeepDims(%arg0: tensor<2x1024x7x7x3xf32>) -> tensor<1x1x7x1x1xf32> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [0, 1, 3, 4], keep_dims} : tensor<2x1024x7x7x3xf32> -> tensor<1x1x7x1x1xf32>
  return %0 : tensor<1x1x7x1x1xf32>

// CHECK:   %[[VAL_1:.+]] = IE.Reshape(%[[VAL_0]]) {shape_value = [1, 2048, 7, 21]} : tensor<2x1024x7x7x3xf32> -> tensor<1x2048x7x21xf32>
// CHECK:   %[[VAL_2:.+]] = IE.ReduceSum(%[[VAL_1]]) {axes_value = [1, 3], keep_dims} : tensor<1x2048x7x21xf32> -> tensor<1x1x7x1xf32>
// CHECK:   %[[VAL_3:.+]] = IE.AffineReshape(%[[VAL_2]]) {dim_mapping = {{\[\[}}0], [1], [2], [3, 4]], shape_value = [1, 1, 7, 1, 1]} : tensor<1x1x7x1xf32> -> tensor<1x1x7x1x1xf32>
}

// -----

// CHECK-LABEL: func.func @DoNotConvert5DReduceSum(
// CHECK-SAME:      %[[VAL_0:.+]]: tensor<2x1024x7x7x3xf32>) -> tensor<2x7x3xf32> {
func.func @DoNotConvert5DReduceSum(%arg0: tensor<2x1024x7x7x3xf32>) -> tensor<2x7x3xf32> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [1, 3]} : tensor<2x1024x7x7x3xf32> -> tensor<2x7x3xf32>
  return %0 : tensor<2x7x3xf32>

// CHECK:   %[[VAL_1:.+]] = IE.ReduceSum(%[[VAL_0]]) {axes_value = [1, 3]} : tensor<2x1024x7x7x3xf32> -> tensor<2x7x3xf32>
// CHECK:   return %[[VAL_1]] : tensor<2x7x3xf32>
}

// -----

// CHECK-LABEL: @ReshapeMvn6Scale1DBias2D
// CHECK-SAME:   [[INPUT:%.+]]: tensor<151x1x768xf16>,
// CHECK-SAME:   [[SCALE:%.+]]: tensor<768xf16>,
// CHECK-SAME:   [[BIAS:%.+]]: tensor<1x768xf16>
func.func @ReshapeMvn6Scale1DBias2D(%input: tensor<151x1x768xf16>, %scale: tensor<768xf16>, %bias: tensor<1x768xf16>) -> tensor<151x1x768xf16> {
    %add = IE.MVN6(%input, %scale, %bias)
            {axes_value = [2], eps = 5.00E-7 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true,
             operandSegmentSizes = array<i32: 1, 1, 1, 0>}
            : tensor<151x1x768xf16>, tensor<768xf16>, tensor<1x768xf16> -> tensor<151x1x768xf16>
    return %add : tensor<151x1x768xf16>

    // CHECK:       [[DATA_RESHAPE:%.+]]  = IE.AffineReshape([[INPUT]]) {dim_mapping = {{\[\[}}0, 1], [2], [3]], shape_value = [1, 151, 1, 768]}
    // CHECK-SAME:                          : tensor<151x1x768xf16> -> tensor<1x151x1x768xf16>

    // CHECK:       [[SCALE_RESHAPE:%.+]] = IE.AffineReshape([[SCALE]]) {dim_mapping = {{\[\[}}0, 1, 2, 3]], shape_value = [1, 1, 1, 768]}
    // CHECK-SAME:                          : tensor<768xf16> -> tensor<1x1x1x768xf16>

    // CHECK:       [[BIAS_RESHAPE:%.+]]  = IE.AffineReshape([[BIAS]]) {dim_mapping = {{\[\[}}0, 1, 2], [3]], shape_value = [1, 1, 1, 768]}
    // CHECK-SAME:                          : tensor<1x768xf16> -> tensor<1x1x1x768xf16>

    // CHECK:       [[MVN:%.+]] = IE.MVN6([[DATA_RESHAPE]], [[SCALE_RESHAPE]], [[BIAS_RESHAPE]])
    // CHECK-SAME:                 {axes_value = [3], eps = 5.000000e-07 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true,
    // CHECK-SAME:                  operandSegmentSizes = array<i32: 1, 1, 1, 0>}
    // CHECK-SAME:                : tensor<1x151x1x768xf16>, tensor<1x1x1x768xf16>, tensor<1x1x1x768xf16> -> tensor<1x151x1x768xf16>

    // CHECK:       [[OUT_RESHAPE:%.+]]  = IE.AffineReshape([[MVN]]) {dim_mapping = {{\[\[}}0], [0], [1], [2]], shape_value = [151, 1, 768]}
    // CHECK-SAME:                          : tensor<1x151x1x768xf16> -> tensor<151x1x768xf16>

    // CHECK:       return [[OUT_RESHAPE]] : tensor<151x1x768xf16>
}

// -----

// CHECK-LABEL: @ReshapeMvn6Scale4D
// CHECK-SAME:   [[INPUT:%.+]]: tensor<151x1x768xf16>,
// CHECK-SAME:   [[SCALE:%.+]]: tensor<1x1x1x768xf16>
func.func @ReshapeMvn6Scale4D(%input: tensor<151x1x768xf16>, %scale: tensor<1x1x1x768xf16>) -> tensor<151x1x768xf16> {
    %add = IE.MVN6(%input, %scale)
            {axes_value = [2], eps = 5.00E-7 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true,
             operandSegmentSizes = array<i32: 1, 1, 0, 0>}
            : tensor<151x1x768xf16>, tensor<1x1x1x768xf16> -> tensor<151x1x768xf16>
    return %add : tensor<151x1x768xf16>

    // CHECK:       [[DATA_RESHAPE:%.+]]  = IE.AffineReshape([[INPUT]]) {dim_mapping = {{\[\[}}0, 1], [2], [3]], shape_value = [1, 151, 1, 768]}
    // CHECK-SAME:                          : tensor<151x1x768xf16> -> tensor<1x151x1x768xf16>

    // CHECK:       [[MVN:%.+]] = IE.MVN6([[DATA_RESHAPE]], [[SCALE]])
    // CHECK-SAME:                {axes_value = [3], eps = 5.000000e-07 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>,
    // CHECK-SAME:                 normalize_variance = true, operandSegmentSizes = array<i32: 1, 1, 0, 0>}
    // CHECK-SAME:                : tensor<1x151x1x768xf16>, tensor<1x1x1x768xf16> -> tensor<1x151x1x768xf16>

    // CHECK:       [[OUT_RESHAPE:%.+]] = IE.AffineReshape([[MVN]]) {dim_mapping = {{\[\[}}0], [0], [1], [2]], shape_value = [151, 1, 768]}
    // CHECK-SAME:                        : tensor<1x151x1x768xf16> -> tensor<151x1x768xf16>

    // CHECK:       return [[OUT_RESHAPE]] : tensor<151x1x768xf16>
}

// -----

// CHECK-LABEL: @Convert5DInterpolate
func.func @Convert5DInterpolate(%arg0: tensor<1x2x2x2x2xf16>) -> tensor<1x2x4x4x4xf16> {
    %0 = IE.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <SIMPLE>, antialias = false, pads_begin = [0, 0, 0, 0, 0], pads_end = [0, 0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3, 4], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00, 2.000000e+00], sizes_attr = [4, 4, 4]} : tensor<1x2x2x2x2xf16> -> tensor<1x2x4x4x4xf16>
    return %0 : tensor<1x2x4x4x4xf16>

    // CHECK: [[INPUT_RESHAPE:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:           {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [2, 2, 2, 2]} : tensor<1x2x2x2x2xf16> -> tensor<2x2x2x2xf16>
    // CHECK: [[Interpolate:%.+]] = IE.Interpolate([[INPUT_RESHAPE]])
    // CHECK:   {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <SIMPLE>,
    // CHECK-SAME:       antialias = false,
    // CHECK-SAME:       pads_begin = [0, 0, 0, 0, 0], pads_end = [0, 0, 0, 0, 0],
    // CHECK:            cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:       axes_attr = [1, 2, 3],
    // CHECK:            operandSegmentSizes = array<i32: 1, 0, 0, 0>,
    // CHECK:            scales_attr = [2.000000e+00, 2.000000e+00, 2.000000e+00], sizes_attr = [4, 4, 4]} :
    // CHECK:       tensor<2x2x2x2xf16> -> tensor<2x4x4x4xf16>

    // CHECK: [[OUTPUT_RESHAPE:%.+]] = IE.AffineReshape([[Interpolate]])
    // CHECK-SAME{LITERAL}:            {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 2, 4, 4, 4]} : tensor<2x4x4x4xf16> -> tensor<1x2x4x4x4xf16>
    // CHECK: return [[OUTPUT_RESHAPE]] : tensor<1x2x4x4x4xf16>

}

// -----

// CHECK-LABEL: @Convert5DInterpolateNDifferentThen1
func.func @Convert5DInterpolateNDifferentThen1(%arg0: tensor<2x2x2x2x2xf16>) -> tensor<2x2x4x4x4xf16> {
    %0 = IE.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <SIMPLE>, antialias = false, pads_begin = [0, 0, 0, 0, 0], pads_end = [0, 0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3, 4], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00, 2.000000e+00], sizes_attr = [4, 4, 4]} : tensor<2x2x2x2x2xf16> -> tensor<2x2x4x4x4xf16>
    return %0 : tensor<2x2x4x4x4xf16>

    // CHECK: [[INPUT_RESHAPE:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:           {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [4, 2, 2, 2]} : tensor<2x2x2x2x2xf16> -> tensor<4x2x2x2xf16>
    // CHECK: [[Interpolate:%.+]] = IE.Interpolate([[INPUT_RESHAPE]])
    // CHECK:   {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <SIMPLE>,
    // CHECK-SAME:       antialias = false,
    // CHECK-SAME:       pads_begin = [0, 0, 0, 0, 0], pads_end = [0, 0, 0, 0, 0],
    // CHECK:            cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:       axes_attr = [1, 2, 3],
    // CHECK:            operandSegmentSizes = array<i32: 1, 0, 0, 0>,
    // CHECK:            scales_attr = [2.000000e+00, 2.000000e+00, 2.000000e+00], sizes_attr = [4, 4, 4]} :
    // CHECK:       tensor<4x2x2x2xf16> -> tensor<4x4x4x4xf16>

    // CHECK: [[OUTPUT_RESHAPE:%.+]] = IE.AffineReshape([[Interpolate]])
    // CHECK-SAME{LITERAL}:            {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [2, 2, 4, 4, 4]} : tensor<4x4x4x4xf16> -> tensor<2x2x4x4x4xf16>
    // CHECK: return [[OUTPUT_RESHAPE]] : tensor<2x2x4x4x4xf16>

}

// -----

// CHECK-LABEL: @Convert5DInterpolateTwoAxes
func.func @Convert5DInterpolateTwoAxes(%arg0: tensor<1x2x2x2x2xf16>) -> tensor<1x2x4x4x2xf16> {
    %0 = IE.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <SIMPLE>, antialias = false, pads_begin = [0, 0, 0, 0, 0], pads_end = [0, 0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [4, 4]} : tensor<1x2x2x2x2xf16> -> tensor<1x2x4x4x2xf16>
    return %0 : tensor<1x2x4x4x2xf16>

    // CHECK: [[INPUT_RESHAPE:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:           {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [2, 2, 2, 2]} : tensor<1x2x2x2x2xf16> -> tensor<2x2x2x2xf16>
    // CHECK: [[Interpolate:%.+]] = IE.Interpolate([[INPUT_RESHAPE]])
    // CHECK:   {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <SIMPLE>,
    // CHECK-SAME:       antialias = false,
    // CHECK-SAME:       pads_begin = [0, 0, 0, 0, 0], pads_end = [0, 0, 0, 0, 0],
    // CHECK:            cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:       axes_attr = [1, 2],
    // CHECK:            operandSegmentSizes = array<i32: 1, 0, 0, 0>,
    // CHECK:            scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [4, 4]} :
    // CHECK:       tensor<2x2x2x2xf16> -> tensor<2x4x4x2xf16>

    // CHECK: [[OUTPUT_RESHAPE:%.+]] = IE.AffineReshape([[Interpolate]])
    // CHECK-SAME{LITERAL}:            {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 2, 4, 4, 2]} : tensor<2x4x4x2xf16> -> tensor<1x2x4x4x2xf16>
    // CHECK: return [[OUTPUT_RESHAPE]] : tensor<1x2x4x4x2xf16>

}

// -----

// CHECK-LABEL: @Convert3DInterpolate
func.func @Convert3DInterpolate(%arg0: tensor<8x64x1xf16>) -> tensor<8x64x2xf16> {
    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>,
                nearest_mode = <FLOOR>, pads_begin = [0, 0, 0], pads_end = [0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [0, 1, 2], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [1.000000e+00, 1.000000e+00, 2.000000e+00],
        sizes_attr = [8, 64, 4]} : tensor<8x64x1xf16> -> tensor<8x64x2xf16>

    return %0 : tensor<8x64x2xf16>

    // CHECK: [[INPUT_RESHAPE:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:           {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 8, 64, 1]} : tensor<8x64x1xf16> -> tensor<1x8x64x1xf16>
    // CHECK: [[Interpolate:%.+]] = IE.Interpolate([[INPUT_RESHAPE]])
    // CHECK:   {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
    // CHECK-SAME:       antialias = false,
    // CHECK-SAME:       pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0],
    // CHECK:            cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:       axes_attr = [0, 1, 2, 3],
    // CHECK:            operandSegmentSizes = array<i32: 1, 0, 0, 0>,
    // CHECK:            scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 2.000000e+00], sizes_attr = [1, 8, 64, 4]} :
    // CHECK:       tensor<1x8x64x1xf16> -> tensor<1x8x64x2xf16>

    // CHECK: [[OUTPUT_RESHAPE:%.+]] = IE.AffineReshape([[Interpolate]])
    // CHECK-SAME{LITERAL}:            {dim_mapping = [[0], [0], [1], [2]], shape_value = [8, 64, 2]} : tensor<1x8x64x2xf16> -> tensor<8x64x2xf16>
    // CHECK: return [[OUTPUT_RESHAPE]] : tensor<8x64x2xf16>

}

// -----

// CHECK-LABEL: @Convert2DInterpolate
func.func @Convert2DInterpolate(%arg0: tensor<8x64xf16>) -> tensor<17x64xf16> {
    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>,
                nearest_mode = <FLOOR>, pads_begin = [0, 0], pads_end = [1, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [0, 1], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [2.000000e+00, 1.000000e+00],
        sizes_attr = [8, 64]} : tensor<8x64xf16> -> tensor<17x64xf16>

    return %0 : tensor<17x64xf16>

    // CHECK: [[INPUT_RESHAPE:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:           {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 8, 64]} : tensor<8x64xf16> -> tensor<1x1x8x64xf16>
    // CHECK: [[Interpolate:%.+]] = IE.Interpolate([[INPUT_RESHAPE]])
    // CHECK:   {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
    // CHECK-SAME:       antialias = false,
    // CHECK-SAME:       pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 1, 0],
    // CHECK:            cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:       axes_attr = [0, 1, 2, 3],
    // CHECK:            operandSegmentSizes = array<i32: 1, 0, 0, 0>,
    // CHECK:            scales_attr = [1.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00], sizes_attr = [1, 1, 8, 64]} :
    // CHECK:       tensor<1x1x8x64xf16> -> tensor<1x1x17x64xf16>

    // CHECK: [[OUTPUT_RESHAPE:%.+]] = IE.AffineReshape([[Interpolate]])
    // CHECK-SAME{LITERAL}:            {dim_mapping = [[0], [0], [0], [1]], shape_value = [17, 64]} : tensor<1x1x17x64xf16> -> tensor<17x64xf16>
    // CHECK: return [[OUTPUT_RESHAPE]] : tensor<17x64xf16>

}

// -----

// CHECK-LABEL: @ConvertFloor
func.func @ConvertFloor(%arg0: tensor<8x4096x512xf16>) -> tensor<8x4096x512xf16> {
    %0 = IE.Floor(%arg0) : tensor<8x4096x512xf16> -> tensor<8x4096x512xf16>

    return %0 : tensor<8x4096x512xf16>

    // CHECK:       [[VAL_0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 8, 4096, 512]} : tensor<8x4096x512xf16> -> tensor<1x8x4096x512xf16>
    // CHECK:       [[Floor:%.+]] = IE.Floor([[VAL_0]]) : tensor<1x8x4096x512xf16> -> tensor<1x8x4096x512xf16>
    // CHECK:       [[VAL_1:%.+]] = IE.AffineReshape([[Floor]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [2]], shape_value = [8, 4096, 512]} : tensor<1x8x4096x512xf16> -> tensor<8x4096x512xf16>

    // CHECK:   return [[VAL_1]]
}

// -----

// CHECK-LABEL: @ConvertSelect
func.func @ConvertSelect(%arg0: tensor<1x10x40x40xf16>, %arg1: tensor<1x10x40x40xf16>) -> tensor<1x10x40x40xf16> {
    %cst = const.Declare tensor<1xf16> = dense<0.000000e+00> : tensor<f16>, [#const.Reshape<[1]>]
    %0 = IE.Select(%arg0, %cst, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x40x40xf16>, tensor<1xf16>, tensor<1x10x40x40xf16> -> tensor<1x10x40x40xf16>

    return %0 : tensor<1x10x40x40xf16>

    // CHECK:       [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<f16>, [#const.Reshape<[1, 1, 1, 1]>]
    // CHECK:       [[Select:%.+]] = IE.Select(%arg0, [[CST_0]], %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x40x40xf16>, tensor<1x1x1x1xf16>, tensor<1x10x40x40xf16> -> tensor<1x10x40x40xf16>
    // CHECK:   return [[Select]]
}

// -----

// CHECK-LABEL: @ConvertSelectThirdInputIsNot4D
func.func @ConvertSelectThirdInputIsNot4D(%arg0: tensor<1x10x40x40xf16>, %arg1: tensor<1x10x40x40xf16>) -> tensor<1x10x40x40xf16> {
    %cst = const.Declare tensor<1xf16> = dense<0.000000e+00> : tensor<f16>, [#const.Reshape<[1]>]
    %0 = IE.Select(%arg0, %arg1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x40x40xf16>, tensor<1x10x40x40xf16>, tensor<1xf16> -> tensor<1x10x40x40xf16>

    return %0 : tensor<1x10x40x40xf16>

    // CHECK:       [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<f16>, [#const.Reshape<[1, 1, 1, 1]>]
    // CHECK:       [[Select:%.+]] = IE.Select(%arg0, %arg1, [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x40x40xf16>, tensor<1x10x40x40xf16>, tensor<1x1x1x1xf16> -> tensor<1x10x40x40xf16>
    // CHECK:   return [[Select]]
}

// -----

// CHECK-LABEL: @ConvertSelect5D
func.func @ConvertSelect5D(%arg0: tensor<1x1x10x40x40xf16>, %arg1: tensor<1x1x10x40x40xf16>) -> tensor<1x1x10x40x40xf16> {
    %cst = const.Declare tensor<1xf16> = dense<0.000000e+00> : tensor<f16>, [#const.Reshape<[1]>]
    %0 = IE.Select(%arg0, %arg1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x10x40x40xf16>, tensor<1x1x10x40x40xf16>, tensor<1xf16> -> tensor<1x1x10x40x40xf16>

    return %0 : tensor<1x1x10x40x40xf16>

    // CHECK:       [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<f16>, [#const.Reshape<[1, 1, 1, 1]>]
    // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [1, 10, 40, 40]} : tensor<1x1x10x40x40xf16> -> tensor<1x10x40x40xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape(%arg1)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [1, 10, 40, 40]} : tensor<1x1x10x40x40xf16> -> tensor<1x10x40x40xf16>
    // CHECK:       [[Select:%.+]] = IE.Select([[RESHAPE_0]], [[RESHAPE_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x40x40xf16>, tensor<1x10x40x40xf16>, tensor<1x1x1x1xf16> -> tensor<1x10x40x40xf16>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.AffineReshape([[Select]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 1, 10, 40, 40]} : tensor<1x10x40x40xf16> -> tensor<1x1x10x40x40xf16>
    // CHECK:   return [[RESHAPE_2]]
}

// -----

// CHECK-LABEL: @ConvertConvertOpTo4dFrom5D
func.func @ConvertConvertOpTo4dFrom5D(%arg0: tensor<1x2x8x4096x512xf32>) -> tensor<1x2x8x4096x512xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x2x8x4096x512xf32> -> tensor<1x2x8x4096x512xf16>

    return %0 : tensor<1x2x8x4096x512xf16>

    // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [1], [2], [2], [3]], shape_value = [1, 2, 32768, 512]} : tensor<1x2x8x4096x512xf32> -> tensor<1x2x32768x512xf32>
    // CHECK:       [[CONVERTED:%.+]] = IE.Convert([[RESHAPE_0]]) {dstElemType = f16} : tensor<1x2x32768x512xf32> -> tensor<1x2x32768x512xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[CONVERTED]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 2, 8, 4096, 512]} : tensor<1x2x32768x512xf16> -> tensor<1x2x8x4096x512xf16>
    // CHECK:    return [[RESHAPE_1]]
}

// -----

// CHECK-LABEL: @ConvertAccumulate
func.func @ConvertAccumulate(%LHS: tensor<16x96xf32>, %RHS: tensor<16x96xf32>) -> tensor<16x96xf32> {
    // CHECK:   ([[LHS:%.+]]: tensor<16x96xf32>, [[RHS:%.+]]: tensor<16x96xf32>)
    %ADD = IE.Accumulate(%LHS, %RHS) {
        operandSegmentSizes = array<i32: 1, 1, 0, 0>
    } : tensor<16x96xf32>, tensor<16x96xf32> -> tensor<16x96xf32>
    // CHECK:   [[RESHAPE_RHS:%.+]] = IE.AffineReshape([[RHS]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1, 2], [3]],
    // CHECK-SAME:              shape_value = [1, 1, 16, 96]
    // CHECK-SAME:  } : tensor<16x96xf32> -> tensor<1x1x16x96xf32>

    // CHECK:   [[RESHAPE_LHS:%.+]] = IE.AffineReshape([[LHS]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1, 2], [3]],
    // CHECK-SAME:              shape_value = [1, 1, 16, 96]
    // CHECK-SAME:  } : tensor<16x96xf32> -> tensor<1x1x16x96xf32>

    // CHECK:   [[TRANSPOSE_LHS:%.+]] = IE.Transpose([[RESHAPE_LHS]]) {
    // CHECK-SAME:      order_value = #NWHC
    // CHECK-SAME:  } : tensor<1x1x16x96xf32> -> tensor<1x96x16x1xf32>

    // CHECK:   [[TRANSPOSE_RHS:%.+]] = IE.Transpose([[RESHAPE_RHS]]) {
    // CHECK-SAME:      order_value = #NWHC
    // CHECK-SAME:  } : tensor<1x1x16x96xf32> -> tensor<1x96x16x1xf32>

    // CHECK:   [[ADD:%.+]] = IE.Add([[TRANSPOSE_LHS]], [[TRANSPOSE_RHS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
    // CHECK-SAME:  } : tensor<1x96x16x1xf32>, tensor<1x96x16x1xf32> -> tensor<1x96x16x1xf32>

    // CHECK:   [[TRANSPOSE_OUT:%.+]] = IE.Transpose([[ADD]]) {
    // CHECK-SAME:      order_value = #NWHC
    // CHECK-SAME:  } : tensor<1x96x16x1xf32> -> tensor<1x1x16x96xf32>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[TRANSPOSE_OUT]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [0], [0], [1]],
    // CHECK-SAME:              shape_value = [16, 96]
    // CHECK-SAME:  } : tensor<1x1x16x96xf32> -> tensor<16x96xf32>

    return %ADD : tensor<16x96xf32>
    // CHECK:   return [[RESHAPE_OUT]] : tensor<16x96xf32>
}

// -----

// CHECK-LABEL: @ConvertAccumulateWithScales
func.func @ConvertAccumulateWithScales(%LHS: tensor<16x96xf32>,
                                       %RHS: tensor<16x96xf32>,
                                       %LHS_SCALE: tensor<96xf32>,
                                       %RHS_SCALE: tensor<96xf32>) -> tensor<16x96xf32> {
    // CHECK:   ([[LHS:%.+]]: tensor<16x96xf32>, [[RHS:%.+]]: tensor<16x96xf32>
    // CHECK-SAME:  [[LHS_SCALE:%.+]]: tensor<96xf32>, [[RHS_SCALE:%.+]]: tensor<96xf32>)
    %ADD = IE.Accumulate(%LHS, %RHS, %LHS_SCALE, %RHS_SCALE) {
        operandSegmentSizes = array<i32: 1, 1, 1, 1>
    } : tensor<16x96xf32>, tensor<16x96xf32>, tensor<96xf32>, tensor<96xf32> -> tensor<16x96xf32>
    // CHECK:   [[RESHAPE_RHS:%.+]] = IE.AffineReshape([[RHS]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1, 2], [3]],
    // CHECK-SAME:              shape_value = [1, 1, 16, 96]
    // CHECK-SAME:  } : tensor<16x96xf32> -> tensor<1x1x16x96xf32>

    // CHECK:   [[RESHAPE_LHS:%.+]] = IE.AffineReshape([[LHS]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1, 2], [3]],
    // CHECK-SAME:              shape_value = [1, 1, 16, 96]
    // CHECK-SAME:  } : tensor<16x96xf32> -> tensor<1x1x16x96xf32>

    // CHECK:   [[TRANSPOSE_LHS:%.+]] = IE.Transpose([[RESHAPE_LHS]]) {
    // CHECK-SAME:      order_value = #NWHC
    // CHECK-SAME:  } : tensor<1x1x16x96xf32> -> tensor<1x96x16x1xf32>

    // CHECK:   [[TRANSPOSE_RHS:%.+]] = IE.Transpose([[RESHAPE_RHS]]) {
    // CHECK-SAME:      order_value = #NWHC
    // CHECK-SAME:  } : tensor<1x1x16x96xf32> -> tensor<1x96x16x1xf32>

    // CHECK:   [[RESHAPE_LHS_SCALES:%.+]] = IE.AffineReshape([[LHS_SCALE]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1, 2, 3]],
    // CHECK-SAME:              shape_value = [1, 96, 1, 1]
    // CHECK-SAME:  } : tensor<96xf32> -> tensor<1x96x1x1xf32>

    // CHECK:   [[RESHAPE_RHS_SCALES:%.+]] = IE.AffineReshape([[RHS_SCALE]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1, 2, 3]],
    // CHECK-SAME:              shape_value = [1, 96, 1, 1]
    // CHECK-SAME:  } : tensor<96xf32> -> tensor<1x96x1x1xf32>

    // CHECK:   [[ADD:%.+]] = IE.Accumulate([[TRANSPOSE_LHS]], [[TRANSPOSE_RHS]], [[RESHAPE_LHS_SCALES]], [[RESHAPE_RHS_SCALES]]) {
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 1, 1>
    // CHECK-SAME:  } : tensor<1x96x16x1xf32>, tensor<1x96x16x1xf32>, tensor<1x96x1x1xf32>, tensor<1x96x1x1xf32> -> tensor<1x96x16x1xf32>

    // CHECK:   [[TRANSPOSE_OUT:%.+]] = IE.Transpose([[ADD]]) {
    // CHECK-SAME:      order_value = #NWHC
    // CHECK-SAME:  } : tensor<1x96x16x1xf32> -> tensor<1x1x16x96xf32>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[TRANSPOSE_OUT]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [0], [0], [1]],
    // CHECK-SAME:              shape_value = [16, 96]
    // CHECK-SAME:  } : tensor<1x1x16x96xf32> -> tensor<16x96xf32>

    return %ADD : tensor<16x96xf32>
    // CHECK:   return [[RESHAPE_OUT]] : tensor<16x96xf32>
}

// -----

// CHECK-LABEL: func.func @ConvertBroadcastSubgraph(
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x1024x64xf16>
func.func @ConvertBroadcastSubgraph(%input: tensor<1x4x1024x64xf16>) -> tensor<1x32x1024x64xf16> {
    %cst = const.Declare tensor<5xsi32> = dense<[1, 4, 8, 1024, 64]> : tensor<5xsi32>

    %in_reshape = IE.AffineReshape(%input) {
        dim_mapping = [[0], [1, 2], [3], [4]],
        shape_value = [1, 4, 1, 1024, 64]
    } : tensor<1x4x1024x64xf16> -> tensor<1x4x1x1024x64xf16>

    %broadcast = IE.Broadcast(%in_reshape, %cst) {
        mode = #IE.broadcast_type<BIDIRECTIONAL>
    } : tensor<1x4x1x1024x64xf16>, tensor<5xsi32> -> tensor<1x4x8x1024x64xf16>

    %out_reshape = IE.AffineReshape(%broadcast) {
        dim_mapping = [[0], [1], [1], [2], [3]],
        shape_value = [1, 32, 1024, 64]
    } : tensor<1x4x8x1024x64xf16> -> tensor<1x32x1024x64xf16>

    return %out_reshape : tensor<1x32x1024x64xf16>

    // CHECK:   [[CST:%.+]] = const.Declare tensor<4xsi32> = dense<[4, 8, 1024, 64]>
    // CHECK-SAME:     : tensor<4xsi64>, [#const.CastElemType<si32>]

    // CHECK:   [[IN_RESHAPE:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [4, 1, 1024, 64]}
    // CHECK-SAME:     : tensor<1x4x1024x64xf16> -> tensor<4x1x1024x64xf16>

    // CHECK:   [[BROADCAST:%.+]] = IE.Broadcast([[IN_RESHAPE]], [[CST]])
    // CHECK-SAME:     {mode = #IE.broadcast_type<BIDIRECTIONAL>}
    // CHECK-SAME:     : tensor<4x1x1024x64xf16>, tensor<4xsi32> -> tensor<4x8x1024x64xf16>

    // CHECK:   [[OUT_RESHAPE:%.+]] = IE.Reshape([[BROADCAST]]) {shape_value = [1, 32, 1024, 64]}
    // CHECK-SAME:     : tensor<4x8x1024x64xf16> -> tensor<1x32x1024x64xf16>

    // CHECK:   return [[OUT_RESHAPE]] : tensor<1x32x1024x64xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertBroadcast(
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x1x1024x64xf16>
func.func @ConvertBroadcast(%input: tensor<1x4x1x1024x64xf16>) -> tensor<1x4x16x1024x64xf16> {
    %cst = const.Declare tensor<5xsi32> = dense<[1, 4, 16, 1024, 64]> : tensor<5xsi32>

    %broadcast = IE.Broadcast(%input, %cst) {
        mode = #IE.broadcast_type<BIDIRECTIONAL>
    } : tensor<1x4x1x1024x64xf16>, tensor<5xsi32> -> tensor<1x4x16x1024x64xf16>

    return %broadcast : tensor<1x4x16x1024x64xf16>

    // CHECK:   [[CST:%.+]] = const.Declare tensor<4xsi32> = dense<[4, 16, 1024, 64]>
    // CHECK-SAME:      : tensor<4xsi64>, [#const.CastElemType<si32>]

    // CHECK:   [[IN_RESHAPE:%.+]] = IE.AffineReshape([[INPUT]]) {
    // CHECK-SAME{LITERAL}: dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [4, 1, 1024, 64]}
    // CHECK-SAME:      : tensor<1x4x1x1024x64xf16> -> tensor<4x1x1024x64xf16>

    // CHECK:   [[BROADCAST:%.+]] = IE.Broadcast([[IN_RESHAPE]], [[CST]])
    // CHECK-SAME:      {mode = #IE.broadcast_type<BIDIRECTIONAL>}
    // CHECK-SAME:      : tensor<4x1x1024x64xf16>, tensor<4xsi32> -> tensor<4x16x1024x64xf16>

    // CHECK:   [[OUT_RESHAPE:%.+]] = IE.AffineReshape([[BROADCAST]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 4, 16, 1024, 64]}
    // CHECK-SAME:      : tensor<4x16x1024x64xf16> -> tensor<1x4x16x1024x64xf16>

    // CHECK:   return [[OUT_RESHAPE]] : tensor<1x4x16x1024x64xf16>
}

// -----

// CHECK-LABEL: func.func @DoNotConvertBroadcast(
// CHECK-SAME:      [[INPUT:%.+]]: tensor<3x4x1x1024x64xf16>
func.func @DoNotConvertBroadcast(%input: tensor<3x4x1x1024x64xf16>) -> tensor<3x4x16x1024x64xf16> {
    %cst = const.Declare tensor<5xsi32> = dense<[3, 4, 16, 1024, 64]> : tensor<5xsi32>

    %broadcast = IE.Broadcast(%input, %cst) {
        mode = #IE.broadcast_type<BIDIRECTIONAL>
    } : tensor<3x4x1x1024x64xf16>, tensor<5xsi32> -> tensor<3x4x16x1024x64xf16>

    return %broadcast : tensor<3x4x16x1024x64xf16>

    // CHECK:   [[CST:%.+]] = const.Declare tensor<5xsi32> = dense<[3, 4, 16, 1024, 64]> : tensor<5xsi32>

    // CHECK:   [[BROADCAST:%.+]] = IE.Broadcast([[INPUT]], [[CST]])
    // CHECK-SAME:      {mode = #IE.broadcast_type<BIDIRECTIONAL>}
    // CHECK-SAME:      : tensor<3x4x1x1024x64xf16>, tensor<5xsi32> -> tensor<3x4x16x1024x64xf16>

    // CHECK:   return [[BROADCAST]] : tensor<3x4x16x1024x64xf16>
}

// -----

// CHECK-LABEL: func.func @Convert3DConcatTo4DWithAxis
func.func @Convert3DConcatTo4DWithAxis(%arg0: tensor<1x128x249xf16>, %arg1: tensor<1x32x249xf16>) -> tensor<1x160x249xf16> {
    %0 = IE.Concat(%arg0, %arg1) {per_axis = #IE.Concat<axis = 1 : i64, offset = 1 : i64, stride = 1 : i64>} : tensor<1x128x249xf16>, tensor<1x32x249xf16> -> tensor<1x160x249xf16>
    return %0 : tensor<1x160x249xf16>

    // CHECK:             [[AFFINERESHAPE0:%.+]] = IE.AffineReshape({{[^:]+}})
    // CHECK{LITERAL}:    {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 128, 1, 249]} : tensor<1x128x249xf16> -> tensor<1x128x1x249xf16>
    // CHECK:             [[AFFINERESHAPE1:%.+]] = IE.AffineReshape({{[^:]+}})
    // CHECK{LITERAL}:    {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 32, 1, 249]} : tensor<1x32x249xf16> -> tensor<1x32x1x249xf16>

    // CHECK:             [[CONCAT0:%.+]] = IE.Concat([[AFFINERESHAPE0]], [[AFFINERESHAPE1]])
    // CHECK{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]} : tensor<1x128x1x249xf16>, tensor<1x32x1x249xf16> -> tensor<1x160x1x249xf16>
    // CHECK:             [[AFFINERESHAPE2:%.+]] = IE.AffineReshape([[CONCAT0]])
    // CHECK{LITERAL}:    dim_mapping = [[0], [1], [1], [2]], shape_value = [1, 160, 249]} : tensor<1x160x1x249xf16> -> tensor<1x160x249xf16>

    // CHECK:              return [[AFFINERESHAPE2]] : tensor<1x160x249xf16>

}

// -----

// CHECK-LABEL: func.func @Convert2DConcatTo4DWithAxis
func.func @Convert2DConcatTo4DWithAxis(%arg0: tensor<1x128xf16>, %arg1: tensor<1x32xf16>) -> tensor<1x160xf16> {
    %0 = IE.Concat(%arg0, %arg1) {per_axis = #IE.Concat<axis = 1 : i64, offset = 1 : i64, stride = 1 : i64>} : tensor<1x128xf16>, tensor<1x32xf16> -> tensor<1x160xf16>
    return %0 : tensor<1x160xf16>

    // CHECK:             [[AFFINERESHAPE0:%.+]] = IE.AffineReshape({{[^:]+}})
    // CHECK{LITERAL}:    {dim_mapping = [[0], [1, 2, 3]], shape_value = [1, 128, 1, 1]} : tensor<1x128xf16> -> tensor<1x128x1x1xf16>
    // CHECK:             [[AFFINERESHAPE1:%.+]] = IE.AffineReshape({{[^:]+}})
    // CHECK{LITERAL}:    {dim_mapping = [[0], [1, 2, 3]], shape_value = [1, 32, 1, 1]} : tensor<1x32xf16> -> tensor<1x32x1x1xf16>

    // CHECK:             [[CONCAT0:%.+]] = IE.Concat([[AFFINERESHAPE0]], [[AFFINERESHAPE1]])
    // CHECK{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]} : tensor<1x128x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x160x1x1xf16>
    // CHECK:             [[AFFINERESHAPE2:%.+]] = IE.AffineReshape([[CONCAT0]])
    // CHECK{LITERAL}:    {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 160]} : tensor<1x160x1x1xf16> -> tensor<1x160xf16>

    // CHECK:              return [[AFFINERESHAPE2]] : tensor<1x160xf16>
}

// -----

// CHECK-LABEL: func.func @Convert1DConcatTo4DWithAxis
func.func @Convert1DConcatTo4DWithAxis(%arg0: tensor<128xf16>, %arg1: tensor<32xf16>) -> tensor<160xf16> {
    %0 = IE.Concat(%arg0, %arg1) {per_axis = #IE.Concat<axis = 0 : i64, offset = 1 : i64, stride = 1 : i64>} : tensor<128xf16>, tensor<32xf16> -> tensor<160xf16>
    return %0 : tensor<160xf16>

    // CHECK:             [[AFFINERESHAPE0:%.+]] = IE.AffineReshape({{[^:]+}})
    // CHECK{LITERAL}:    {dim_mapping = [[0, 1, 2, 3]], shape_value = [1, 1, 128, 1]} : tensor<128xf16> -> tensor<1x1x128x1xf16>
    // CHECK:             [[AFFINERESHAPE1:%.+]] = IE.AffineReshape({{[^:]+}})
    // CHECK{LITERAL}:    {dim_mapping = [[0, 1, 2, 3]], shape_value = [1, 1, 32, 1]} : tensor<32xf16> -> tensor<1x1x32x1xf16>

    // CHECK:             [[CONCAT0:%.+]] = IE.Concat([[AFFINERESHAPE0]], [[AFFINERESHAPE1]])
    // CHECK{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 0, 128, 0]]} : tensor<1x1x128x1xf16>, tensor<1x1x32x1xf16> -> tensor<1x1x160x1xf16>
    // CHECK:             [[AFFINERESHAPE2:%.+]] = IE.AffineReshape([[CONCAT0]])
    // CHECK{LITERAL}:    {dim_mapping = [[0], [0], [0], [0]], shape_value = [160]} : tensor<1x1x160x1xf16> -> tensor<160xf16>

    // CHECK:              return [[AFFINERESHAPE2]] : tensor<160xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertTileWith3DInput3DRepeats
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1536x1xf16>
func.func @ConvertTileWith3DInput3DRepeats(%arg0: tensor<1x1536x1xf16>) -> (tensor<1x1536x801xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 801]} : tensor<1x1536x1xf16> -> tensor<1x1536x801xf16>
    return %0 : tensor<1x1536x801xf16>

    // CHECK:           [[AFFINERESHAPE0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK{LITERAL}:  {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 1536, 1]} : tensor<1x1536x1xf16> -> tensor<1x1x1536x1xf16>
    // CHECK:           [[TILE:%.+]] = IE.Tile([[AFFINERESHAPE0]])
    // CHECK{LITERAL}:  {repeats_values = [1, 1, 1, 801]} : tensor<1x1x1536x1xf16> -> tensor<1x1x1536x801xf16>
    // CHECK:           [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[TILE]])
    // CHECK{LITERAL}:  {dim_mapping = [[0], [0], [1], [2]], shape_value = [1, 1536, 801]} : tensor<1x1x1536x801xf16> -> tensor<1x1536x801xf16>

    // CHECK:           return [[AFFINERESHAPE1]] : tensor<1x1536x801xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertTileWith3DInput5DRepeats
// CHECK-SAME:      [[INPUT:%.+]]: tensor<4x1536x1xf16>
func.func @ConvertTileWith3DInput5DRepeats(%arg0: tensor<4x1536x1xf16>) -> (tensor<1x1x4x1536x801xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1, 1, 801]} : tensor<4x1536x1xf16> -> tensor<1x1x4x1536x801xf16>
    return %0 : tensor<1x1x4x1536x801xf16>

    // CHECK:           [[AFFINERESHAPE0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK{LITERAL}:  {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 4, 1536, 1]} : tensor<4x1536x1xf16> -> tensor<1x4x1536x1xf16>
    // CHECK:           [[TILE:%.+]] = IE.Tile([[AFFINERESHAPE0]])
    // CHECK{LITERAL}:  {repeats_values = [1, 1, 1, 801]} : tensor<1x4x1536x1xf16> -> tensor<1x4x1536x801xf16>
    // CHECK:           [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[TILE]])
    // CHECK{LITERAL}:  {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 1, 4, 1536, 801]} : tensor<1x4x1536x801xf16> -> tensor<1x1x4x1536x801xf16>

    // CHECK:           return [[AFFINERESHAPE1]] : tensor<1x1x4x1536x801xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertTileWith5DInput3DRepeats
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x2x4x1536x1xf16>
func.func @ConvertTileWith5DInput3DRepeats(%arg0: tensor<1x2x4x1536x1xf16>) -> (tensor<1x2x4x1536x801xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 801]} : tensor<1x2x4x1536x1xf16> -> tensor<1x2x4x1536x801xf16>
    return %0 : tensor<1x2x4x1536x801xf16>

    // CHECK:           [[AFFINERESHAPE0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK{LITERAL}:  {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [2, 4, 1536, 1]} : tensor<1x2x4x1536x1xf16> -> tensor<2x4x1536x1xf16>
    // CHECK:           [[TILE:%.+]] = IE.Tile([[AFFINERESHAPE0]])
    // CHECK{LITERAL}:  {repeats_values = [1, 1, 1, 801]} : tensor<2x4x1536x1xf16> -> tensor<2x4x1536x801xf16>
    // CHECK:           [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[TILE]])
    // CHECK{LITERAL}:  {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 2, 4, 1536, 801]} : tensor<2x4x1536x801xf16> -> tensor<1x2x4x1536x801xf16>

    // CHECK:           return [[AFFINERESHAPE1]] : tensor<1x2x4x1536x801xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertTileWith5DInput5DRepeats
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x2x4x1536x1xf16>
func.func @ConvertTileWith5DInput5DRepeats(%arg0: tensor<1x2x4x1536x1xf16>) -> (tensor<1x2x4x1536x801xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1, 1, 801]} : tensor<1x2x4x1536x1xf16> -> tensor<1x2x4x1536x801xf16>
    return %0 : tensor<1x2x4x1536x801xf16>

    // CHECK:           [[AFFINERESHAPE0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK{LITERAL}:  {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [2, 4, 1536, 1]} : tensor<1x2x4x1536x1xf16> -> tensor<2x4x1536x1xf16>
    // CHECK:           [[TILE:%.+]] = IE.Tile([[AFFINERESHAPE0]])
    // CHECK{LITERAL}:  {repeats_values = [1, 1, 1, 801]} : tensor<2x4x1536x1xf16> -> tensor<2x4x1536x801xf16>
    // CHECK:           [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[TILE]])
    // CHECK{LITERAL}:  {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 2, 4, 1536, 801]} : tensor<2x4x1536x801xf16> -> tensor<1x2x4x1536x801xf16>

    // CHECK:           return [[AFFINERESHAPE1]] : tensor<1x2x4x1536x801xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertTileWith6DInput6DRepeats
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x19947x1x4x1x2xf16>
func.func @ConvertTileWith6DInput6DRepeats(%arg0: tensor<1x19947x1x4x1x2xf16>) -> (tensor<1x19947x8x4x4x2xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 8, 1, 4, 1]} : tensor<1x19947x1x4x1x2xf16> -> tensor<1x19947x8x4x4x2xf16>
    return %0 : tensor<1x19947x8x4x4x2xf16>

    // CHECK:           [[AFFINERESHAPE0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK{LITERAL}:  {dim_mapping = [[0], [0], [1], [2], [3], [4]], shape_value = [19947, 1, 4, 1, 2]} : tensor<1x19947x1x4x1x2xf16> -> tensor<19947x1x4x1x2xf16>
    // CHECK:           [[TILE:%.+]] = IE.Tile([[AFFINERESHAPE0]])
    // CHECK{LITERAL}:  {repeats_values = [1, 8, 1, 4, 1]} : tensor<19947x1x4x1x2xf16> -> tensor<19947x8x4x4x2xf16>
    // CHECK:           [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[TILE]])
    // CHECK{LITERAL}:  {dim_mapping = [[0, 1], [2], [3], [4], [5]], shape_value = [1, 19947, 8, 4, 4, 2]} : tensor<19947x8x4x4x2xf16> -> tensor<1x19947x8x4x4x2xf16>

    // CHECK:           return [[AFFINERESHAPE1]] : tensor<1x19947x8x4x4x2xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertSequeezedTileWith5DInput5DRepeats
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1024x1x1x1x128xf16>
func.func @ConvertSequeezedTileWith5DInput5DRepeats(%arg0: tensor<1024x1x1x1x128xf16>) -> (tensor<1024x1x1x16x128xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1, 16, 1]} : tensor<1024x1x1x1x128xf16> -> tensor<1024x1x1x16x128xf16>
    return %0 : tensor<1024x1x1x16x128xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [1], [2], [2], [3]], shape_value = [1024, 1, 1, 128]} : tensor<1024x1x1x1x128xf16> -> tensor<1024x1x1x128xf16>
    // CHECK:       [[TILE:%.+]] = IE.Tile([[RESHAPE_IN]]) {repeats_values = [1, 1, 16, 1]} : tensor<1024x1x1x128xf16> -> tensor<1024x1x16x128xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[TILE]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [1, 2], [3], [4]], shape_value = [1024, 1, 1, 16, 128]} : tensor<1024x1x16x128xf16> -> tensor<1024x1x1x16x128xf16>

    // CHECK:       return [[RESHAPE_OUT]] : tensor<1024x1x1x16x128xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertTileWithRepeatInHighestDim
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x2x3x4x5x6xf16>
func.func @ConvertTileWithRepeatInHighestDim(%arg0: tensor<1x2x3x4x5x6xf16>) -> (tensor<4x2x3x16x5x6xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [4, 1, 1, 4, 1, 1]} : tensor<1x2x3x4x5x6xf16> -> tensor<4x2x3x16x5x6xf16>
    return %0 : tensor<4x2x3x16x5x6xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [1], [1], [2], [3], [3]], shape_value = [1, 6, 4, 30]} : tensor<1x2x3x4x5x6xf16> -> tensor<1x6x4x30xf16>
    // CHECK:       [[TILE:%.+]] = IE.Tile([[RESHAPE_IN]]) {repeats_values = [4, 1, 4, 1]} : tensor<1x6x4x30xf16> -> tensor<4x6x16x30xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[TILE]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [1, 2], [3], [4, 5]], shape_value = [4, 2, 3, 16, 5, 6]} : tensor<4x6x16x30xf16> -> tensor<4x2x3x16x5x6xf16>

    // CHECK:       return [[RESHAPE_OUT]] : tensor<4x2x3x16x5x6xf16>
}


// -----

// CHECK-LABEL: func.func @NotConvertTileCouldNotAdjustShape
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x2x3x4x5x6xf16>
func.func @NotConvertTileCouldNotAdjustShape(%arg0: tensor<1x2x3x4x5x6xf16>) -> (tensor<2x4x6x8x10x12xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [2, 2, 2, 2, 2, 2]} : tensor<1x2x3x4x5x6xf16> -> tensor<2x4x6x8x10x12xf16>
    return %0 : tensor<2x4x6x8x10x12xf16>

    // CHECK:       [[TILE:%.+]] = IE.Tile([[INPUT]]) {repeats_values = [2, 2, 2, 2, 2, 2]} : tensor<1x2x3x4x5x6xf16> -> tensor<2x4x6x8x10x12xf16>
    // CHECK:       return [[TILE]] : tensor<2x4x6x8x10x12xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertTileWith5DInput5DRepeatsWith3TrivialDims
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x1x1x128xf16>
func.func @ConvertTileWith5DInput5DRepeatsWith3TrivialDims(%arg0: tensor<1x1x1x1x128xf16>) -> (tensor<1x1x1x16x128xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1, 16, 1]} : tensor<1x1x1x1x128xf16> -> tensor<1x1x1x16x128xf16>
    return %0 : tensor<1x1x1x16x128xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [1], [2], [2], [3]], shape_value = [1, 1, 1, 128]} : tensor<1x1x1x1x128xf16> -> tensor<1x1x1x128xf16>
    // CHECK:       [[TILE:%.+]] = IE.Tile([[RESHAPE_IN]]) {repeats_values = [1, 1, 16, 1]} : tensor<1x1x1x128xf16> -> tensor<1x1x16x128xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[TILE]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [1, 2], [3], [4]], shape_value = [1, 1, 1, 16, 128]} : tensor<1x1x16x128xf16> -> tensor<1x1x1x16x128xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x1x1x16x128xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertTileWith5DInput2DRepeats
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x2x4x1536x1xf16>
func.func @ConvertTileWith5DInput2DRepeats(%arg0: tensor<1x2x4x1536x1xf16>) -> (tensor<1x2x4x1536x801xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 801]} : tensor<1x2x4x1536x1xf16> -> tensor<1x2x4x1536x801xf16>
    return %0 : tensor<1x2x4x1536x801xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [2, 4, 1536, 1]} : tensor<1x2x4x1536x1xf16> -> tensor<2x4x1536x1xf16>
    // CHECK:       [[TILE:%.+]] = IE.Tile([[RESHAPE_IN]]) {repeats_values = [1, 1, 1, 801]} : tensor<2x4x1536x1xf16> -> tensor<2x4x1536x801xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[TILE]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 2, 4, 1536, 801]} : tensor<2x4x1536x801xf16> -> tensor<1x2x4x1536x801xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x2x4x1536x801xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertTileWithDifferentInputOutputSize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<4x1536x16xf16>
func.func @ConvertTileWithDifferentInputOutputSize(%arg0: tensor<4x1536x16xf16>) -> (tensor<16x4x1536x16xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [16, 1, 1, 1]} : tensor<4x1536x16xf16> -> tensor<16x4x1536x16xf16>
    return %0 : tensor<16x4x1536x16xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK{LITERAL}:  {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 4, 1536, 16]} : tensor<4x1536x16xf16> -> tensor<1x4x1536x16xf16>
    // CHECK:       [[TILE:%.+]] = IE.Tile([[RESHAPE_IN]]) {repeats_values = [16, 1, 1, 1]} : tensor<1x4x1536x16xf16> -> tensor<16x4x1536x16xf16>

    // CHECK:       return [[TILE]] : tensor<16x4x1536x16xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertTileWith3DInput5DRepeats
// CHECK-SAME:      [[INPUT:%.+]]: tensor<2x3x4xf16>
func.func @ConvertTileWith3DInput5DRepeats(%arg0: tensor<2x3x4xf16>) -> (tensor<2x1x2x6x8xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [2, 1, 1, 2, 2]} : tensor<2x3x4xf16> -> tensor<2x1x2x6x8xf16>
    return %0 : tensor<2x1x2x6x8xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK{LITERAL}:  {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 2, 3, 4]} : tensor<2x3x4xf16> -> tensor<1x2x3x4xf16>
    // CHECK:       [[TILE:%.+]] = IE.Tile([[RESHAPE_IN]]) {repeats_values = [2, 1, 2, 2]} : tensor<1x2x3x4xf16> -> tensor<2x2x6x8xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[TILE]])
    // CHECK{LITERAL}:  {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [2, 1, 2, 6, 8]} : tensor<2x2x6x8xf16> -> tensor<2x1x2x6x8xf16>

    // CHECK:       return [[RESHAPE_OUT]] : tensor<2x1x2x6x8xf16>
}

// -----

// CHECK-LABEL: func.func @Convert4DConcatWithoutAxis
func.func @Convert4DConcatWithoutAxis(%arg0: tensor<1x320x14x14x10xf16>, %arg1: tensor<1x320x14x14x10xf16>) -> tensor<1x640x14x14x10xf16> {
  %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0, 0], [0, 320, 0, 0, 0]]} : tensor<1x320x14x14x10xf16>, tensor<1x320x14x14x10xf16> -> tensor<1x640x14x14x10xf16>
  return %0 : tensor<1x640x14x14x10xf16>

    // CHECK:             [[AFFINERESHAPE0:%.+]] = IE.AffineReshape({{[^:]+}})
    // CHECK{LITERAL}:    {dim_mapping = [[0, 1], [2], [3], [3], [3]], shape_value = [1, 1, 320, 1960]} : tensor<1x320x14x14x10xf16> -> tensor<1x1x320x1960xf16>
    // CHECK:             [[AFFINERESHAPE1:%.+]] = IE.AffineReshape({{[^:]+}})
    // CHECK{LITERAL}:    {dim_mapping = [[0, 1], [2], [3], [3], [3]], shape_value = [1, 1, 320, 1960]} : tensor<1x320x14x14x10xf16> -> tensor<1x1x320x1960xf16>

    // CHECK:             [[CONCAT0:%.+]] = IE.Concat([[AFFINERESHAPE0]], [[AFFINERESHAPE1]])
    // CHECK{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 0, 320, 0]]} : tensor<1x1x320x1960xf16>, tensor<1x1x320x1960xf16> -> tensor<1x1x640x1960xf16>
    // CHECK:             [[AFFINERESHAPE2:%.+]] = IE.AffineReshape([[CONCAT0]])
    // CHECK{LITERAL}:    {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [1, 640, 14, 14, 10]} : tensor<1x1x640x1960xf16> -> tensor<1x640x14x14x10xf16>

    // CHECK:              return [[AFFINERESHAPE2]] : tensor<1x640x14x14x10xf16>
}

// -----

// CHECK-LABEL: @Convert3DTrivialAddExtendNWithOne
// CHECK-SAME:  [[INPUT_0:%.+]]: tensor<1x512x28xf16>, [[INPUT_1:%.+]]: tensor<1x512x28xf16>
func.func @Convert3DTrivialAddExtendNWithOne(%arg0: tensor<1x512x28xf16>, %arg1: tensor<1x512x28xf16>) -> tensor<1x512x28xf16> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x28xf16>, tensor<1x512x28xf16> -> tensor<1x512x28xf16>
    return %0 : tensor<1x512x28xf16>

    // CHECK:       [[VAL_1:%.+]] = IE.AffineReshape([[INPUT_1]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 512, 28]} : tensor<1x512x28xf16> -> tensor<1x1x512x28xf16>
    // CHECK:       [[VAL_0:%.+]] = IE.AffineReshape([[INPUT_0]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 512, 28]} : tensor<1x512x28xf16> -> tensor<1x1x512x28xf16>
    // CHECK:       [[ADD:%.+]] = IE.Add([[VAL_0]], [[VAL_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x28xf16>, tensor<1x1x512x28xf16> -> tensor<1x1x512x28xf16>
    // CHECK:       [[VAL_2:%.+]] = IE.AffineReshape([[ADD]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [2]], shape_value = [1, 512, 28]} : tensor<1x1x512x28xf16> -> tensor<1x512x28xf16>
    // CHECK:   return [[VAL_2]]
}

// -----

// CHECK-LABEL: func.func @ConvertLSTMSequence(
// CHECK-SAME:      [[VAL_0:%.+]]: tensor<4x640x64xf16>,
// CHECK-SAME:      [[VAL_1:%.+]]: tensor<4x1x128xf16>,
// CHECK-SAME:      [[VAL_2:%.+]]: tensor<4x1x128xf16>) -> (tensor<4x1x640x128xf16>, tensor<4x1x128xf16>, tensor<4x1x128xf16>) {
func.func @ConvertLSTMSequence(%arg0: tensor<4x640x64xf16>, %arg1: tensor<4x1x128xf16>, %arg2: tensor<4x1x128xf16>) -> (tensor<4x1x640x128xf16>, tensor<4x1x128xf16>, tensor<4x1x128xf16>) {
  %cst = const.Declare tensor<1x512x64xf16> = dense<1.0> : tensor<1x512x64xf16>
  %cst_0 = const.Declare tensor<1x512x128xf16> = dense<2.0> : tensor<1x512x128xf16>
  %cst_1 = const.Declare tensor<1x512xf16> = dense<3.0> : tensor<1x512xf16>
  %outputHiddenValues, %outputHiddenState, %outputCellState = IE.LSTMSequence(%arg0, %arg1, %arg2, %cst, %cst_0, %cst_1) {direction = #IE.rnn_seq_direction<REVERSE>, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>, sequenceLength = 640 : i64} : tensor<4x640x64xf16>, tensor<4x1x128xf16>, tensor<4x1x128xf16>, tensor<1x512x64xf16>, tensor<1x512x128xf16>, tensor<1x512xf16> -> tensor<4x1x640x128xf16>, tensor<4x1x128xf16>, tensor<4x1x128xf16>
  return %outputHiddenValues, %outputHiddenState, %outputCellState : tensor<4x1x640x128xf16>, tensor<4x1x128xf16>, tensor<4x1x128xf16>

// CHECK-DAG:   [[VAL_4:%.+]] = const.Declare tensor<1x4x128x128xf16> = dense<2.000000e+00> : tensor<1x512x128xf16>, [#const.Reshape<[1, 4, 128, 128]>]
// CHECK-DAG:   [[VAL_3:%.+]] = const.Declare tensor<1x1x4x128xf16> = dense<3.000000e+00> : tensor<1x512xf16>, [#const.Reshape<[1, 1, 4, 128]>]
// CHECK-DAG:   [[VAL_5:%.+]] = const.Declare tensor<1x1x512x64xf16> = dense<1.000000e+00> : tensor<1x512x64xf16>, [#const.Reshape<[1, 1, 512, 64]>]
// CHECK:       [[VAL_6:%.+]] = IE.AffineReshape([[VAL_0]])
// CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3]], shape_value = [4, 1, 640, 64]} : tensor<4x640x64xf16> -> tensor<4x1x640x64xf16>
// CHECK:       [[VAL_7:%.+]] = IE.AffineReshape([[VAL_1]])
// CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2], [3]], shape_value = [4, 1, 1, 128]} : tensor<4x1x128xf16> -> tensor<4x1x1x128xf16>
// CHECK:       [[VAL_8:%.+]] = IE.AffineReshape([[VAL_2]])
// CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2], [3]], shape_value = [4, 1, 1, 128]} : tensor<4x1x128xf16> -> tensor<4x1x1x128xf16>
// CHECK:       [[VAL_9:%.+]], [[VAL_10:.*]], [[VAL_11:.*]] = IE.LSTMSequence([[VAL_6]], [[VAL_7]], [[VAL_8]], [[VAL_5]], [[VAL_4]], [[VAL_3]]) {direction = #IE.rnn_seq_direction<REVERSE>, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>, sequenceLength = 640 : i64} : tensor<4x1x640x64xf16>, tensor<4x1x1x128xf16>, tensor<4x1x1x128xf16>, tensor<1x1x512x64xf16>, tensor<1x4x128x128xf16>, tensor<1x1x4x128xf16> -> tensor<4x1x640x128xf16>, tensor<4x1x1x128xf16>, tensor<4x1x1x128xf16>
// CHECK:       [[VAL_12:%.+]] = IE.AffineReshape([[VAL_10]])
// CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2]], shape_value = [4, 1, 128]} : tensor<4x1x1x128xf16> -> tensor<4x1x128xf16>
// CHECK:       [[VAL_13:%.+]] = IE.AffineReshape([[VAL_11]])
// CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2]], shape_value = [4, 1, 128]} : tensor<4x1x1x128xf16> -> tensor<4x1x128xf16>
// CHECK:       return [[VAL_9]], [[VAL_12]], [[VAL_13]] : tensor<4x1x640x128xf16>, tensor<4x1x128xf16>, tensor<4x1x128xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertLSTMSequenceWithMissingWeights(
// CHECK-SAME:      [[VAL_0:%.+]]: tensor<4x1x640x512xf16>,
// CHECK-SAME:      [[VAL_1:%.+]]: tensor<4x1x128xf16>,
// CHECK-SAME:      [[VAL_2:%.+]]: tensor<4x1x128xf16>) -> (tensor<4x1x640x128xf16>, tensor<4x1x128xf16>, tensor<4x1x128xf16>) {
func.func @ConvertLSTMSequenceWithMissingWeights(%arg0: tensor<4x1x640x512xf16>, %arg1: tensor<4x1x128xf16>, %arg2: tensor<4x1x128xf16>) -> (tensor<4x1x640x128xf16>, tensor<4x1x128xf16>, tensor<4x1x128xf16>) {
  %cst = const.Declare tensor<1x512x128xf16> = dense<1.0> : tensor<1x512x128xf16>

  %outputHiddenValues, %outputHiddenState, %outputCellState = IE.LSTMSequence(%arg0, %arg1, %arg2, %cst) {direction = #IE.rnn_seq_direction<REVERSE>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>, sequenceLength = 640 : i64} : tensor<4x1x640x512xf16>, tensor<4x1x128xf16>, tensor<4x1x128xf16>, tensor<1x512x128xf16> -> tensor<4x1x640x128xf16>, tensor<4x1x128xf16>, tensor<4x1x128xf16>
  return %outputHiddenValues, %outputHiddenState, %outputCellState : tensor<4x1x640x128xf16>, tensor<4x1x128xf16>, tensor<4x1x128xf16>

//  CHECK-DAG:  [[VAL_3:%.+]] = const.Declare tensor<1x4x128x128xf16> = dense<1.000000e+00> : tensor<1x512x128xf16>, [#const.Reshape<[1, 4, 128, 128]>]
//  CHECK:      [[VAL_4:%.+]] = IE.AffineReshape([[VAL_1]])
// CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2], [3]], shape_value = [4, 1, 1, 128]} : tensor<4x1x128xf16> -> tensor<4x1x1x128xf16>
//  CHECK:      [[VAL_5:%.+]] = IE.AffineReshape([[VAL_2]])
// CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2], [3]], shape_value = [4, 1, 1, 128]} : tensor<4x1x128xf16> -> tensor<4x1x1x128xf16>
// CHECK:       [[VAL_6:%.+]], [[VAL_7:.*]], [[VAL_8:.*]] = IE.LSTMSequence([[VAL_0]], [[VAL_4]], [[VAL_5]], [[VAL_3]]) {direction = #IE.rnn_seq_direction<REVERSE>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>, sequenceLength = 640 : i64} : tensor<4x1x640x512xf16>, tensor<4x1x1x128xf16>, tensor<4x1x1x128xf16>, tensor<1x4x128x128xf16> -> tensor<4x1x640x128xf16>, tensor<4x1x1x128xf16>, tensor<4x1x1x128xf16>
// CHECK:       [[VAL_9:%.+]] = IE.AffineReshape([[VAL_7]])
// CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2]], shape_value = [4, 1, 128]} : tensor<4x1x1x128xf16> -> tensor<4x1x128xf16>
// CHECK:       [[VAL_10:%.+]] = IE.AffineReshape([[VAL_8]])
// CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2]], shape_value = [4, 1, 128]} : tensor<4x1x1x128xf16> -> tensor<4x1x128xf16>
// CHECK:       return [[VAL_6]], [[VAL_9]], [[VAL_10]] : tensor<4x1x640x128xf16>, tensor<4x1x128xf16>, tensor<4x1x128xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertLSTMGatesWith2DInput
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<100x2048xf16>, [[INPUT1:%.+]]: tensor<100x512xf16>
func.func @ConvertLSTMGatesWith2DInput(%arg0: tensor<100x2048xf16>, %arg1: tensor<100x512xf16>) -> (tensor<100x512xf16>, tensor<100x512xf16>) {
    %0, %1 = IE.LSTMGates(%arg0, %arg1) : tensor<100x2048xf16>, tensor<100x512xf16> -> tensor<100x512xf16>, tensor<100x512xf16>
    return %0, %1 : tensor<100x512xf16>, tensor<100x512xf16>

    // CHECK:           [[AFFINERESHAPE0:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 100, 2048]} : tensor<100x2048xf16> -> tensor<1x1x100x2048xf16>
    // CHECK:           [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[INPUT1]])
    // CHECK{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 100, 512]} : tensor<100x512xf16> -> tensor<1x1x100x512xf16>
    // CHECK:           [[LSTMGATES_0:%.+]], [[LSTMGATES_1:%.+]] = IE.LSTMGates([[AFFINERESHAPE0]], [[AFFINERESHAPE1]])
    // CHECK-SAME:      -> tensor<1x1x100x512xf16>, tensor<1x1x100x512xf16>
    // CHECK:           [[AFFINERESHAPE2:%.+]] = IE.AffineReshape([[LSTMGATES_0]])
    // CHECK{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [100, 512]} : tensor<1x1x100x512xf16> -> tensor<100x512xf16>
    // CHECK:           [[AFFINERESHAPE3:%.+]] = IE.AffineReshape([[LSTMGATES_1]])
    // CHECK{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [100, 512]} : tensor<1x1x100x512xf16> -> tensor<100x512xf16>

    // CHECK:           return [[AFFINERESHAPE2]], [[AFFINERESHAPE3]] : tensor<100x512xf16>, tensor<100x512xf16>
}

// -----

// CHECK-LABEL: func.func @RoundOpWithInput3D
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<25x196x160xf16>
func.func @RoundOpWithInput3D(%arg0: tensor<25x196x160xf16>) -> tensor<25x196x160xf16> {
    %0 = IE.Round(%arg0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<25x196x160xf16> -> tensor<25x196x160xf16>
    return %0 : tensor<25x196x160xf16>

    // CHECK:           [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK{LITERAL}:  {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 25, 196, 160]} : tensor<25x196x160xf16> -> tensor<1x25x196x160xf16>
    // CHECK:           [[ROUND:%.+]] = IE.Round([[RESHAPE_IN]]) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x25x196x160xf16> -> tensor<1x25x196x160xf16>
    // CHECK:           [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[ROUND]])
    // CHECK{LITERAL}:  {dim_mapping = [[0], [0], [1], [2]], shape_value = [25, 196, 160]} : tensor<1x25x196x160xf16> -> tensor<25x196x160xf16>
    // CHECK:           return [[RESHAPE_OUT]] : tensor<25x196x160xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertLogTo4DFrom5D
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x6x80x38x38xf32>
func.func @ConvertLogTo4DFrom5D(%arg0: tensor<1x6x80x38x38xf32>) -> tensor<1x6x80x38x38xf32> {
  %0 = IE.Log(%arg0) : tensor<1x6x80x38x38xf32> -> tensor<1x6x80x38x38xf32>
  return %0 : tensor<1x6x80x38x38xf32>
    //CHECK:            [[RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT0]])
    //CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [3], [3]], shape_value = [1, 6, 80, 1444]} : tensor<1x6x80x38x38xf32> -> tensor<1x6x80x1444xf32>
    //CHECK:            [[LOG_0:%.+]] = IE.Log([[RESHAPE_0]]) : tensor<1x6x80x1444xf32> -> tensor<1x6x80x1444xf32>
    //CHECK:            [[RESHAPE_1:%.+]] = IE.AffineReshape([[LOG_0]])
    //CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [3, 4]], shape_value = [1, 6, 80, 38, 38]} : tensor<1x6x80x1444xf32> -> tensor<1x6x80x38x38xf32>
    //CHECK:            return [[RESHAPE_1:%.+]] : tensor<1x6x80x38x38xf32>
}

// -----

// CHECK-LABEL: func.func @ConvertLogTo4DFrom2D
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x50xf32>
func.func @ConvertLogTo4DFrom2D(%arg0: tensor<1x50xf32>) -> tensor<1x50xf32> {
  %0 = IE.Log(%arg0) : tensor<1x50xf32> -> tensor<1x50xf32>
  return %0 : tensor<1x50xf32>
    //CHECK:            [[RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT0]])
    //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 50]} : tensor<1x50xf32> -> tensor<1x1x1x50xf32>
    //CHECK:            [[LOG_0:%.+]] = IE.Log([[RESHAPE_0]]) : tensor<1x1x1x50xf32> -> tensor<1x1x1x50xf32>
    //CHECK:            [[RESHAPE_1:%.+]] = IE.AffineReshape([[LOG_0]])
    //CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 50]} : tensor<1x1x1x50xf32> -> tensor<1x50xf32>
    //CHECK:            return [[RESHAPE_1:%.+]] : tensor<1x50xf32>
}

// -----

// CHECK-LABEL: func.func @Convert2DLogSoftmax
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<4096x512xf16>
func.func @Convert2DLogSoftmax(%arg0: tensor<4096x512xf16>) -> tensor<4096x512xf16> {
    %0 = IE.LogSoftmax(%arg0) {axisInd = 1} : tensor<4096x512xf16> -> tensor<4096x512xf16>

    return %0 : tensor<4096x512xf16>

    // CHECK:       [[VAL_0:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 4096, 512]} : tensor<4096x512xf16> -> tensor<1x1x4096x512xf16>
    // CHECK:       [[LOG_SOFTMAX:%.+]] = IE.LogSoftmax([[VAL_0]])
    // CHECK-SAME:      {axisInd = 3 : i64} : tensor<1x1x4096x512xf16> -> tensor<1x1x4096x512xf16>
    // CHECK:       [[VAL_1:%.+]] = IE.AffineReshape([[LOG_SOFTMAX]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [0], [1]], shape_value = [4096, 512]} : tensor<1x1x4096x512xf16> -> tensor<4096x512xf16>

    // CHECK:   return [[VAL_1]]
}

// -----

// CHECK-LABEL: func.func @Convert3DLogSoftmax
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<8x4096x512xf16>
func.func @Convert3DLogSoftmax(%arg0: tensor<8x4096x512xf16>) -> tensor<8x4096x512xf16> {
    %0 = IE.LogSoftmax(%arg0) {axisInd = 2} : tensor<8x4096x512xf16> -> tensor<8x4096x512xf16>

    return %0 : tensor<8x4096x512xf16>

    // CHECK:       [[VAL_0:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 8, 4096, 512]} : tensor<8x4096x512xf16> -> tensor<1x8x4096x512xf16>
    // CHECK:       [[LOG_SOFTMAX:%.+]] = IE.LogSoftmax([[VAL_0]])
    // CHECK-SAME:      {axisInd = 3 : i64} : tensor<1x8x4096x512xf16> -> tensor<1x8x4096x512xf16>
    // CHECK:       [[VAL_1:%.+]] = IE.AffineReshape([[LOG_SOFTMAX]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [2]], shape_value = [8, 4096, 512]} : tensor<1x8x4096x512xf16> -> tensor<8x4096x512xf16>

    // CHECK:   return [[VAL_1]]
}

// -----

// CHECK-LABEL: func.func @ConvertLSTMCellto4D
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x64xf16>, [[INPUT1:%.+]]: tensor<1x128xf16>, [[INPUT2:%.+]]: tensor<1x128xf16>
func.func @ConvertLSTMCellto4D(%arg0: tensor<1x64xf16>, %arg1: tensor<1x128xf16>, %arg2: tensor<1x128xf16>) -> (tensor<1x128xf16>, tensor<1x128xf16>) {
    %cst = const.Declare tensor<512xf16> = dense<1.000000e+00> : tensor<512xf16>
    %cst_0 = const.Declare tensor<512x128xf16> = dense<1.000000e+00> : tensor<512x128xf16>
    %cst_1 = const.Declare tensor<512x64xf16> = dense<1.000000e+00> : tensor<512x64xf16>
    %outputHiddenState, %outputCellState = IE.LSTMCell(%arg0, %arg1, %arg2, %cst_1, %cst_0, %cst) {hiddenSize = 128 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>} : tensor<1x64xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<512x64xf16>, tensor<512x128xf16>, tensor<512xf16> -> tensor<1x128xf16>, tensor<1x128xf16>
    return %outputHiddenState, %outputCellState : tensor<1x128xf16>, tensor<1x128xf16>

    // CHECK:           [[BIAS:%.+]] = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+00> : tensor<512xf16>,
    // CHECK{LITERAL}:  [#const.Reshape<[1, 1, 1, 512]>]
    // CHECK:           [[WHIDDEN:%.+]] = const.Declare tensor<1x1x512x128xf16> = dense<1.000000e+00> : tensor<512x128xf16>,
    // CHECK{LITERAL}:  [#const.Reshape<[1, 1, 512, 128]>]
    // CHECK:           [[WDATA:%.+]] = const.Declare tensor<1x1x512x64xf16> = dense<1.000000e+00> : tensor<512x64xf16>,
    // CHECK{LITERAL}:  [#const.Reshape<[1, 1, 512, 64]>]
    // CHECK:           [[INDATA:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 64]} : tensor<1x64xf16> -> tensor<1x1x1x64xf16>
    // CHECK:           [[HIDDEN:%.+]] = IE.AffineReshape([[INPUT1]])
    // CHECK{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 128]} : tensor<1x128xf16> -> tensor<1x1x1x128xf16>
    // CHECK:           [[CELL:%.+]] = IE.AffineReshape([[INPUT2]])
    // CHECK{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 128]} : tensor<1x128xf16> -> tensor<1x1x1x128xf16>
    // CHECK:           [[OUTHIDDEN:%.+]], [[OUTCELL:%.+]] = IE.LSTMCell([[INDATA]], [[HIDDEN]], [[CELL]], [[WDATA]], [[WHIDDEN]], [[BIAS]]) {hiddenSize = 128 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>} : tensor<1x1x1x64xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x512x64xf16>, tensor<1x1x512x128xf16>, tensor<1x1x1x512xf16> -> tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
    // CHECK:           [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[OUTHIDDEN]])
    // CHECK{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 128]} : tensor<1x1x1x128xf16> -> tensor<1x128xf16>
    // CHECK:           [[AFFINERESHAPE2:%.+]] = IE.AffineReshape([[OUTCELL]])
    // CHECK{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 128]} : tensor<1x1x1x128xf16> -> tensor<1x128xf16>
    // CHECK:           return [[AFFINERESHAPE1]], [[AFFINERESHAPE2]] : tensor<1x128xf16>, tensor<1x128xf16>

}

// -----

// CHECK-LABEL: func.func @Convert3DErfto4D
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x512x1024xf16>
func.func @Convert3DErfto4D(%arg0: tensor<1x512x1024xf16>) -> tensor<1x512x1024xf16> {
    %0 = IE.Erf(%arg0) : tensor<1x512x1024xf16> -> tensor<1x512x1024xf16>
    return %0 : tensor<1x512x1024xf16>

    // CHECK:       [[AffineReshape0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 512, 1024]} : tensor<1x512x1024xf16> -> tensor<1x1x512x1024xf16>
    // CHECK:       [[ERF:%.+]]  = IE.Erf([[AffineReshape0]]) : tensor<1x1x512x1024xf16> -> tensor<1x1x512x1024xf16>
    // CHECK:       [[AffineReshape1:%.+]] = IE.AffineReshape([[ERF]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [2]], shape_value = [1, 512, 1024]} : tensor<1x1x512x1024xf16> -> tensor<1x512x1024xf16>

    // CHECK:   return [[AffineReshape1]] : tensor<1x512x1024xf16>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: func.func @Convert3DDynamicDequantizeto4D
// CHECK-SAME:      [[INPUT:%.+]]: tensor<28x4608x128x!qElemType>,
// CHECK-SAME:      [[SCALE:%.+]]: tensor<28x4608x1xf16>
func.func @Convert3DDynamicDequantizeto4D(%arg0: tensor<28x4608x128x!quant.uniform<i4:f16, 1.000000e+00>>, %arg1: tensor<28x4608x1xf16>) -> tensor<28x4608x128xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<28x4608x128x!quant.uniform<i4:f16, 1.000000e+00>>, tensor<28x4608x1xf16> -> tensor<28x4608x128xf16>
    return %0 : tensor<28x4608x128xf16>


    // CHECK:       [[AffineReshape1:%.+]] = IE.AffineReshape([[SCALE]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 28, 4608, 1]} : tensor<28x4608x1xf16> -> tensor<1x28x4608x1xf16>
    // CHECK:       [[AffineReshape0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 28, 4608, 128]} : tensor<28x4608x128x!qElemType> -> tensor<1x28x4608x128x!qElemType>
    // CHECK:       [[DynamicDequantize:%.+]]  = IE.DynamicDequantize([[AffineReshape0]], [[AffineReshape1]]) {dstElemType = f16} : tensor<1x28x4608x128x!qElemType>, tensor<1x28x4608x1xf16> -> tensor<1x28x4608x128xf16>
    // CHECK:       [[AffineReshape2:%.+]] = IE.AffineReshape([[DynamicDequantize]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1], [2]], shape_value = [28, 4608, 128]} : tensor<1x28x4608x128xf16> -> tensor<28x4608x128xf16>

    // CHECK:       return [[AffineReshape2]] : tensor<28x4608x128xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertScaleShiftWith3DInputs
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x512x4096xf32>)
func.func @ConvertScaleShiftWith3DInputs(%arg0: tensor<1x512x4096xf32>) -> tensor<1x512x4096xf32> {
    %weights = const.Declare tensor<1x512x1xf32> = dense<6.0> : tensor<1x512x1xf32>
    %bias = const.Declare tensor<1x512x1xf32> = dense<4.0> : tensor<1x512x1xf32>
    %0 = IE.ScaleShift(%arg0, %weights, %bias) {operandSegmentSizes = array<i32: 1, 1, 1>} : tensor<1x512x4096xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32> -> tensor<1x512x4096xf32>

    return %0 : tensor<1x512x4096xf32>

    // CHECK:       [[BIAS:%.+]] = const.Declare tensor<1x512x1x1xf32> = dense<4.000000e+00> : tensor<1x512x1xf32>, [
    // CHECK-SAME:      #const.Reshape<[1, 512, 1, 1]>]
    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<1x512x1x1xf32> = dense<6.000000e+00> : tensor<1x512x1xf32>, [
    // CHECK-SAME:      #const.Reshape<[1, 512, 1, 1]>]
    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK{LITERAL}:  {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 512, 1, 4096]} : tensor<1x512x4096xf32> -> tensor<1x512x1x4096xf32>
    // CHECK:       [[SCALE_SHIFT:%.+]] = IE.ScaleShift([[RESHAPE_IN]], [[WEIGHTS]], [[BIAS]]) {operandSegmentSizes = array<i32: 1, 1, 1>}
    // CHECK-SAME:      tensor<1x512x1x4096xf32>, tensor<1x512x1x1xf32>, tensor<1x512x1x1xf32> -> tensor<1x512x1x4096xf32>
    // CHECK:       [[RESHAPE_OUT:%.+]]  = IE.AffineReshape([[SCALE_SHIFT]])
    // CHECK{LITERAL}:  {dim_mapping = [[0], [1], [1], [2]], shape_value = [1, 512, 4096]} : tensor<1x512x1x4096xf32> -> tensor<1x512x4096xf32>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x512x4096xf32>
}

// -----

// CHECK-LABEL: func.func @Convert3DScaleShiftWithSoftMax
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4096x4096xf16>)
func.func @Convert3DScaleShiftWithSoftMax(%arg0: tensor<4096x4096xf16>) -> tensor<1x4096x4096xf16> {
    %bias = const.Declare tensor<1x1x1xf16> = dense<-0.000000e+00> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    %weights = const.Declare tensor<1x1x1xf16> = dense<0.0452205911> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]

    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2]], shape_value = [1, 4096, 4096]} : tensor<4096x4096xf16> -> tensor<1x4096x4096xf16>
    %1 = IE.ScaleShift(%0, %weights, %bias) {operandSegmentSizes = array<i32: 1, 1, 1>} : tensor<1x4096x4096xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16> -> tensor<1x4096x4096xf16>
    %2 = IE.SoftMax(%1) {axisInd = 2 : i64} : tensor<1x4096x4096xf16> -> tensor<1x4096x4096xf16>

    return %2 : tensor<1x4096x4096xf16>

    // CHECK:       [[BIAS:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-0.000000e+00> : tensor<1x1x1xf32>, [
    // CHECK-SAME:      #const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.0452205911> : tensor<1x1x1xf32>, [
    // CHECK-SAME:      #const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 4096, 1, 4096]} : tensor<4096x4096xf16> -> tensor<1x4096x1x4096xf16>
    // CHECK:       [[SCALE_SHIFT:%.+]] = IE.ScaleShift([[RESHAPE_IN]], [[WEIGHTS]], [[BIAS]]) {operandSegmentSizes = array<i32: 1, 1, 1>}
    // CHECK-SAME:      tensor<1x4096x1x4096xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x4096x1x4096xf16>
    // CHECK:       [[RESHAPE_OUT_SCALE_SHIFT:%.+]]  = IE.AffineReshape([[SCALE_SHIFT]])
    // CHECK{LITERAL}:  {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 4096, 4096, 1]} : tensor<1x4096x1x4096xf16> -> tensor<1x4096x4096x1xf16>
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[RESHAPE_OUT_SCALE_SHIFT]]) {axisInd = 2 : i64}
    // CHECK-SAME:      tensor<1x4096x4096x1xf16> -> tensor<1x4096x4096x1xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]]  = IE.AffineReshape([[SOFTMAX]])
    // CHECK{LITERAL}:  {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 4096, 4096]} : tensor<1x4096x4096x1xf16> -> tensor<1x4096x4096xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x4096x4096xf16>
}

// -----

// CHECK-LABEL: func.func @Convert3DGatherElements
// CHECK-SAME: ([[ARG0:%.+]]: tensor<12x64x512xf16>
func.func @Convert3DGatherElements(%arg0: tensor<12x64x512xf16>) -> tensor<12x64x64xf16> {
    %cst = const.Declare tensor<12x64x64xsi32> = dense<0> : tensor<12x64x64xsi64>, [#const.CastElemType<si32>]
    %0 = IE.GatherElements(%arg0, %cst) {axis = 2 : i64} : tensor<12x64x512xf16>, tensor<12x64x64xsi32> -> tensor<12x64x64xf16>
    return %0 : tensor<12x64x64xf16>

    // CHECK: [[INDICES:%.+]] = const.Declare tensor<1x768x64x1xsi32>
    // CHECK: [[RESHAPE_IN:%.+]] = IE.Reshape([[ARG0]]) {shape_value = [1, 768, 512, 1]} : tensor<12x64x512xf16> -> tensor<1x768x512x1xf16>
    // CHECK: [[GATHER_ELEMENTS:%.+]] = IE.GatherElements([[RESHAPE_IN]], [[INDICES]]) {axis = 2 : i64} : tensor<1x768x512x1xf16>, tensor<1x768x64x1xsi32> -> tensor<1x768x64x1xf16>
    // CHECK: [[RESHAPE_OUT:%.+]] = IE.Reshape([[GATHER_ELEMENTS]]) {shape_value = [12, 64, 64]} : tensor<1x768x64x1xf16> -> tensor<12x64x64xf16>
    // CHECK:   return [[RESHAPE_OUT]] : tensor<12x64x64xf16>
}

// -----

// CHECK-LABEL: func.func @Convert5DGatherElements
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x12x64x512x2xf16>
func.func @Convert5DGatherElements(%arg0: tensor<1x12x64x512x2xf16>) -> tensor<1x12x64x64x2xf16> {
    %cst = const.Declare tensor<1x12x64x64x2xsi32> = dense<0> : tensor<1x12x64x64x2xsi64>, [#const.CastElemType<si32>]
    %0 = IE.GatherElements(%arg0, %cst) {axis = 3 : i64} : tensor<1x12x64x512x2xf16>, tensor<1x12x64x64x2xsi32> -> tensor<1x12x64x64x2xf16>
    return %0 : tensor<1x12x64x64x2xf16>

    // CHECK: [[INDICES:%.+]] = const.Declare tensor<1x768x64x2xsi32>
    // CHECK: [[RESHAPE_IN:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 768, 512, 2]} : tensor<1x12x64x512x2xf16> -> tensor<1x768x512x2xf16>
    // CHECK: [[GATHER_ELEMENTS:%.+]] = IE.GatherElements([[RESHAPE_IN]], [[INDICES]]) {axis = 2 : i64} : tensor<1x768x512x2xf16>, tensor<1x768x64x2xsi32> -> tensor<1x768x64x2xf16>
    // CHECK: [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[GATHER_ELEMENTS]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1, 2], [3], [4]], shape_value = [1, 12, 64, 64, 2]} : tensor<1x768x64x2xf16> -> tensor<1x12x64x64x2xf16>
    // CHECK:  return [[RESHAPE_OUT]] : tensor<1x12x64x64x2xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertRandomUniformWith1DInput
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1xf32>, [[INPUT1:%.+]]: tensor<1xf32>
func.func @ConvertRandomUniformWith1DInput(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1x1x512x1024xf32> {
    %0 = IE.RandomUniform(%arg0, %arg1) {global_seed = 0 : i64, op_seed = 0 : i64, outputType = f32, output_shape = [1, 1, 512, 1024]} : tensor<1xf32>, tensor<1xf32> -> tensor<1x1x512x1024xf32>
    return %0 : tensor<1x1x512x1024xf32>

    // CHECK:           [[AFFINERESHAPE0:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK{LITERAL}:  {dim_mapping = [[0, 1, 2, 3]], shape_value = [1, 1, 1, 1]} : tensor<1xf32> -> tensor<1x1x1x1xf32>
    // CHECK:           [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[INPUT1]])
    // CHECK{LITERAL}:  {dim_mapping = [[0, 1, 2, 3]], shape_value = [1, 1, 1, 1]} : tensor<1xf32> -> tensor<1x1x1x1xf32>
    // CHECK:           [[RANDOMUNIFORM:%.+]] = IE.RandomUniform([[AFFINERESHAPE0]], [[AFFINERESHAPE1]])
    // CHECK-SAME:      -> tensor<1x1x512x1024xf32>

    // CHECK:           return [[RANDOMUNIFORM]] : tensor<1x1x512x1024xf32>
}

// -----

// CHECK-LABEL: func.func @ConvertRandomUniformWith2DOutputAttr
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1xf32>, [[INPUT1:%.+]]: tensor<1xf32>
func.func @ConvertRandomUniformWith2DOutputAttr(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1x1024xf32> {
    %0 = IE.RandomUniform(%arg0, %arg1) {global_seed = 0 : i64, op_seed = 0 : i64, outputType = f32, output_shape = [1, 1024]} : tensor<1xf32>, tensor<1xf32> -> tensor<1x1024xf32>
    return %0 : tensor<1x1024xf32>

    // CHECK:           [[AFFINERESHAPE0:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK{LITERAL}:  {dim_mapping = [[0, 1, 2, 3]], shape_value = [1, 1, 1, 1]} : tensor<1xf32> -> tensor<1x1x1x1xf32>
    // CHECK:           [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[INPUT1]])
    // CHECK{LITERAL}:  {dim_mapping = [[0, 1, 2, 3]], shape_value = [1, 1, 1, 1]} : tensor<1xf32> -> tensor<1x1x1x1xf32>
    // CHECK:           [[RANDOMUNIFORM:%.+]] = IE.RandomUniform([[AFFINERESHAPE0]], [[AFFINERESHAPE1]]) {global_seed = 0 : i64, op_seed = 0 : i64, outputType = f32, output_shape = [1, 1, 1, 1024]} : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x1024xf32>
    // CHECK:           [[AFFINERESHAPEOUT:%.+]] = IE.AffineReshape([[RANDOMUNIFORM]])
    // CHECK{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 1024]} : tensor<1x1x1x1024xf32> -> tensor<1x1024xf32>

    // CHECK:           return [[AFFINERESHAPEOUT]] : tensor<1x1024xf32>
}

// -----

// CHECK-LABEL: func.func @ConvertGatherNDTo4DWithBatchDimsIsZero
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1000x256x10x15xf16>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<25x125x3xsi32>
func.func @ConvertGatherNDTo4DWithBatchDimsIsZero(%arg0: tensor<1000x256x10x15xf16>, %arg1: tensor<25x125x3xsi32>) -> tensor<25x125x15xf16> {
    %0 = IE.GatherND(%arg0, %arg1) {batch_dims = 0 : i64}
            : tensor<1000x256x10x15xf16>, tensor<25x125x3xsi32> -> tensor<25x125x15xf16>

    return %0 : tensor<25x125x15xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = IE.Reshape([[INPUT_0]])
    // CHECK-SAME{LITERAL}:     {shape_value = [1, 1, 2560000, 15]} : tensor<1000x256x10x15xf16> -> tensor<1x1x2560000x15xf16>
    // CHECK:       [[RESHAPE_INDICES:%.+]] = IE.Reshape([[INPUT_1]])
    // CHECK-SAME{LITERAL}:     {shape_value = [1, 1, 3125, 3]} : tensor<25x125x3xsi32> -> tensor<1x1x3125x3xsi32>
    // CHECK:       [[GATHERND:%.+]] = IE.GatherND([[RESHAPE_IN]], [[RESHAPE_INDICES]]) {
    // CHECK-SAME:                  batch_dims = 2 : i64, original_shape = [1, 1, 1000, 256, 10, 15]
    // CHECK-SAME:              } : tensor<1x1x2560000x15xf16>, tensor<1x1x3125x3xsi32> -> tensor<1x1x3125x15xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.Reshape([[GATHERND]])
    // CHECK-SAME{LITERAL}:     {shape_value = [25, 125, 15]} : tensor<1x1x3125x15xf16> -> tensor<25x125x15xf16>

    // CHECK:       return [[RESHAPE_OUT]] : tensor<25x125x15xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertGatherNDTo4DWithBatchDimsIsOne
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<5x6x7xf16>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<5x3x1xsi32>
func.func @ConvertGatherNDTo4DWithBatchDimsIsOne(%arg0: tensor<5x6x7xf16>, %arg1: tensor<5x3x1xsi32>) -> tensor<5x3x7xf16> {
    %0 = IE.GatherND(%arg0, %arg1) {batch_dims = 1 : i64}
            : tensor<5x6x7xf16>, tensor<5x3x1xsi32> -> tensor<5x3x7xf16>

    return %0 : tensor<5x3x7xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT_0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 5, 6, 7]} : tensor<5x6x7xf16> -> tensor<1x5x6x7xf16>
    // CHECK:       [[RESHAPE_INDICES:%.+]] = IE.AffineReshape([[INPUT_1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 5, 3, 1]} : tensor<5x3x1xsi32> -> tensor<1x5x3x1xsi32>
    // CHECK:       [[GATHERND:%.+]] = IE.GatherND([[RESHAPE_IN]], [[RESHAPE_INDICES]]) {
    // CHECK-SAME:                  batch_dims = 2 : i64, original_shape = [1, 5, 6, 7]
    // CHECK-SAME:              } : tensor<1x5x6x7xf16>, tensor<1x5x3x1xsi32> -> tensor<1x5x3x7xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[GATHERND]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1], [2]], shape_value = [5, 3, 7]} : tensor<1x5x3x7xf16> -> tensor<5x3x7xf16>

    // CHECK:       return [[RESHAPE_OUT]] : tensor<5x3x7xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertGatherNDTo4DWithBatchDimsIsTwo
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x16x32x56x16xf16>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x16x14580x2xsi32>
func.func @ConvertGatherNDTo4DWithBatchDimsIsTwo(%arg0: tensor<1x16x32x56x16xf16>, %arg1: tensor<1x16x14580x2xsi32>) -> tensor<1x16x14580x16xf16> {
    %0 = IE.GatherND(%arg0, %arg1) {batch_dims = 2 : i64}
            : tensor<1x16x32x56x16xf16>, tensor<1x16x14580x2xsi32> -> tensor<1x16x14580x16xf16>

    return %0 : tensor<1x16x14580x16xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT_0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [2], [2], [3]], shape_value = [1, 16, 1792, 16]} : tensor<1x16x32x56x16xf16> -> tensor<1x16x1792x16xf16>
    // CHECK:       [[GATHERND:%.+]] = IE.GatherND([[RESHAPE_IN]], [[INPUT_1]]) {
    // CHECK-SAME:                  batch_dims = 2 : i64, original_shape = [1, 16, 32, 56, 16]
    // CHECK-SAME:              } : tensor<1x16x1792x16xf16>, tensor<1x16x14580x2xsi32> -> tensor<1x16x14580x16xf16>

    // CHECK:       return [[GATHERND]] : tensor<1x16x14580x16xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertGatherNDTo4DWithBatchDimsIsTwoAndOut3D
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x16x32x56x16xf16>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x16x14580x3xsi32>
func.func @ConvertGatherNDTo4DWithBatchDimsIsTwoAndOut3D(%arg0: tensor<1x16x32x56x16xf16>, %arg1: tensor<1x16x14580x3xsi32>) -> tensor<1x16x14580xf16> {
    %0 = IE.GatherND(%arg0, %arg1) {batch_dims = 2 : i64}
            : tensor<1x16x32x56x16xf16>, tensor<1x16x14580x3xsi32> -> tensor<1x16x14580xf16>

    return %0 : tensor<1x16x14580xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT_0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [2], [2], [2, 3]], shape_value = [1, 16, 28672, 1]} : tensor<1x16x32x56x16xf16> -> tensor<1x16x28672x1xf16>
    // CHECK:       [[GATHERND:%.+]] = IE.GatherND([[RESHAPE_IN]], [[INPUT_1]]) {
    // CHECK-SAME:                  batch_dims = 2 : i64, original_shape = [1, 16, 32, 56, 16, 1]
    // CHECK-SAME:              } : tensor<1x16x28672x1xf16>, tensor<1x16x14580x3xsi32> -> tensor<1x16x14580x1xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[GATHERND]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 16, 14580]} : tensor<1x16x14580x1xf16> -> tensor<1x16x14580xf16>

    // CHECK:       return [[RESHAPE_OUT]] : tensor<1x16x14580xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertGatherNDTo4DWithLargeTensorRank
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<2x3x4x5x6x7xf16>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<2x3x10x8x6x3xsi32>
func.func @ConvertGatherNDTo4DWithLargeTensorRank(%arg0: tensor<2x3x4x5x6x7xf16>, %arg1: tensor<2x3x10x8x6x3xsi32>) -> tensor<2x3x10x8x6x7xf16> {
    %0 = IE.GatherND(%arg0, %arg1) {batch_dims = 2 : i64}
            : tensor<2x3x4x5x6x7xf16>, tensor<2x3x10x8x6x3xsi32> -> tensor<2x3x10x8x6x7xf16>

    return %0 : tensor<2x3x10x8x6x7xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = IE.Reshape([[INPUT_0]])
    // CHECK-SAME{LITERAL}:     {shape_value = [1, 6, 120, 7]} : tensor<2x3x4x5x6x7xf16> -> tensor<1x6x120x7xf16>
    // CHECK:       [[RESHAPE_INDICES:%.+]] = IE.Reshape([[INPUT_1]])
    // CHECK-SAME{LITERAL}:     {shape_value = [1, 6, 480, 3]} : tensor<2x3x10x8x6x3xsi32> -> tensor<1x6x480x3xsi32>
    // CHECK:       [[GATHERND:%.+]] = IE.GatherND([[RESHAPE_IN]], [[RESHAPE_INDICES]]) {
    // CHECK-SAME:                  batch_dims = 2 : i64, original_shape = [1, 6, 4, 5, 6, 7]
    // CHECK-SAME:              } : tensor<1x6x120x7xf16>, tensor<1x6x480x3xsi32> -> tensor<1x6x480x7xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.Reshape([[GATHERND]])
    // CHECK-SAME{LITERAL}:     {shape_value = [2, 3, 10, 8, 6, 7]} : tensor<1x6x480x7xf16> -> tensor<2x3x10x8x6x7xf16>

    // CHECK:       return [[RESHAPE_OUT]] : tensor<2x3x10x8x6x7xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertGatherNDTo4DWithMultiIndicesDim
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x2x3x4xf16>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x5x6x2xsi32>
func.func @ConvertGatherNDTo4DWithMultiIndicesDim(%arg0: tensor<1x2x3x4xf16>, %arg1: tensor<1x5x6x2xsi32>) -> tensor<1x5x6x4xf16> {
    %0 = IE.GatherND(%arg0, %arg1) {batch_dims = 1 : i64}
            : tensor<1x2x3x4xf16>, tensor<1x5x6x2xsi32> -> tensor<1x5x6x4xf16>

    return %0 : tensor<1x5x6x4xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT_0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 6, 4]} : tensor<1x2x3x4xf16> -> tensor<1x1x6x4xf16>
    // CHECK:       [[RESHAPE_INDICES:%.+]] = IE.AffineReshape([[INPUT_1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 30, 2]} : tensor<1x5x6x2xsi32> -> tensor<1x1x30x2xsi32>
    // CHECK:       [[GATHERND:%.+]] = IE.GatherND([[RESHAPE_IN]], [[RESHAPE_INDICES]]) {
    // CHECK-SAME:                  batch_dims = 2 : i64, original_shape = [1, 1, 2, 3, 4]
    // CHECK-SAME:              } : tensor<1x1x6x4xf16>, tensor<1x1x30x2xsi32> -> tensor<1x1x30x4xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[GATHERND]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [1, 5, 6, 4]} : tensor<1x1x30x4xf16> -> tensor<1x5x6x4xf16>

    // CHECK:       return [[RESHAPE_OUT]] : tensor<1x5x6x4xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertGatherNDTo4DWithMultiBatch
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<2x3x4x5xf16>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<2x3x7x1xsi32>
func.func @ConvertGatherNDTo4DWithMultiBatch(%arg0: tensor<2x3x4x5xf16>, %arg1: tensor<2x3x7x1xsi32>) -> tensor<2x3x7x5xf16> {
    %0 = IE.GatherND(%arg0, %arg1) {batch_dims = 2 : i64}
            : tensor<2x3x4x5xf16>, tensor<2x3x7x1xsi32> -> tensor<2x3x7x5xf16>

    return %0 : tensor<2x3x7x5xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = IE.Reshape([[INPUT_0]])
    // CHECK-SAME{LITERAL}:     {shape_value = [1, 6, 4, 5]} : tensor<2x3x4x5xf16> -> tensor<1x6x4x5xf16>
    // CHECK:       [[RESHAPE_INDICES:%.+]] = IE.Reshape([[INPUT_1]])
    // CHECK-SAME{LITERAL}:     {shape_value = [1, 6, 7, 1]} : tensor<2x3x7x1xsi32> -> tensor<1x6x7x1xsi32>
    // CHECK:       [[GATHERND:%.+]] = IE.GatherND([[RESHAPE_IN]], [[RESHAPE_INDICES]]) {
    // CHECK-SAME:                  batch_dims = 2 : i64, original_shape = [1, 6, 4, 5]
    // CHECK-SAME:              } : tensor<1x6x4x5xf16>, tensor<1x6x7x1xsi32> -> tensor<1x6x7x5xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.Reshape([[GATHERND]])
    // CHECK-SAME{LITERAL}:     {shape_value = [2, 3, 7, 5]} : tensor<1x6x7x5xf16> -> tensor<2x3x7x5xf16>

    // CHECK:       return [[RESHAPE_OUT]] : tensor<2x3x7x5xf16>
}

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>
#N = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @NotConvertGatherNDTo4DWithDynamicShape
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x88xsi32>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<?x2xsi32, {bounds = #const.OpaqueI64Elements<[88, 2]> : tensor<2xsi64>, order = #NC}>
func.func @NotConvertGatherNDTo4DWithDynamicShape(%arg0: tensor<1x88xsi32>, %arg1: tensor<?x2xsi32, {bounds = #const.OpaqueI64Elements<[88, 2]> : tensor<2xsi64>, order = #NC}>)
                                    -> tensor<?xsi32, {bounds = #const.OpaqueI64Elements<[88]> : tensor<1xsi64>, order = #N}> {
    %0 = IE.GatherND(%arg0, %arg1) {batch_dims = 0 : i64}
            : tensor<1x88xsi32>, tensor<?x2xsi32, {bounds = #const.OpaqueI64Elements<[88, 2]> : tensor<2xsi64>, order = #NC}> -> tensor<?xsi32, {bounds = #const.OpaqueI64Elements<[88]> : tensor<1xsi64>, order = #N}>

    return %0 : tensor<?xsi32, {bounds = #const.OpaqueI64Elements<[88]> : tensor<1xsi64>, order = #N}>

    // CHECK:       [[GATHERND:%.+]] = IE.GatherND([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:                  batch_dims = 0 : i64
    // CHECK-SAME:              } : tensor<1x88xsi32>, tensor<?x2xsi32, {bounds = #const.OpaqueI64Elements<[88, 2]> : tensor<2xsi64>, order = #NC}> -> tensor<?xsi32, {bounds = #const.OpaqueI64Elements<[88]> : tensor<1xsi64>, order = #C}>

    // CHECK:       return [[GATHERND]] : tensor<?xsi32, {bounds = #const.OpaqueI64Elements<[88]> : tensor<1xsi64>, order = #C}>
}
// -----

// CHECK-LABEL:   func.func @ConvertGRUGatesWith2DInput(
// CHECK-SAME:                                          %[[VAL_0:.*]]: tensor<2x768xf16>, %[[VAL_1:.*]]: tensor<2x256xf16>,
// CHECK-SAME:                                          %[[VAL_2:.*]]: tensor<2x768xf16>,
// CHECK-SAME:                                          %[[VAL_3:.*]]: tensor<1024xf16>) -> tensor<2x256xf16> {
func.func @ConvertGRUGatesWith2DInput(%arg0: tensor<2x768xf16>, %arg1: tensor<2x256xf16>, %arg2: tensor<2x768xf16>, %arg3: tensor<1024xf16>) -> (tensor<2x256xf16>) {
    %0 = IE.GRUGates(%arg0, %arg1, %arg2, %arg3) : tensor<2x768xf16>, tensor<2x256xf16>, tensor<2x768xf16>, tensor<1024xf16> -> tensor<2x256xf16>
    return %0 : tensor<2x256xf16>
// CHECK:           %[[VAL_4:.*]] = IE.AffineReshape(%[[VAL_0]]) {dim_mapping = {{\[\[}}0, 1, 2], [3]], shape_value = [1, 1, 2, 768]} : tensor<2x768xf16> -> tensor<1x1x2x768xf16>
// CHECK:           %[[VAL_5:.*]] = IE.AffineReshape(%[[VAL_1]]) {dim_mapping = {{\[\[}}0, 1, 2], [3]], shape_value = [1, 1, 2, 256]} : tensor<2x256xf16> -> tensor<1x1x2x256xf16>
// CHECK:           %[[VAL_6:.*]] = IE.AffineReshape(%[[VAL_2]]) {dim_mapping = {{\[\[}}0, 1, 2], [3]], shape_value = [1, 1, 2, 768]} : tensor<2x768xf16> -> tensor<1x1x2x768xf16>
// CHECK:           %[[VAL_7:.*]] = IE.AffineReshape(%[[VAL_3]]) {dim_mapping = {{\[\[}}0, 1, 2, 3]], shape_value = [1, 1, 1, 1024]} : tensor<1024xf16> -> tensor<1x1x1x1024xf16>
// CHECK:           %[[VAL_8:.*]] = IE.GRUGates(%[[VAL_4]], %[[VAL_5]], %[[VAL_6]], %[[VAL_7]]) : tensor<1x1x2x768xf16>, tensor<1x1x2x256xf16>, tensor<1x1x2x768xf16>, tensor<1x1x1x1024xf16> -> tensor<1x1x2x256xf16>
// CHECK:           %[[VAL_9:.*]] = IE.AffineReshape(%[[VAL_8]]) {dim_mapping = {{\[\[}}0], [0], [0], [1]], shape_value = [2, 256]} : tensor<1x1x2x256xf16> -> tensor<2x256xf16>
// CHECK:           return %[[VAL_9]] : tensor<2x256xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertSplitOpTo4DAtFirstDim
// CHECK-SAME: [[INPUT:%.+]]: tensor<3x1x16x1575x72xf16>
func.func @ConvertSplitOpTo4DAtFirstDim(%arg0: tensor<3x1x16x1575x72xf16>)
                                -> (tensor<1x1x16x1575x72xf16>, tensor<1x1x16x1575x72xf16>, tensor<1x1x16x1575x72xf16>) {
    %0:3 = IE.Split(%arg0) {axis_value = 0 : i64, num_splits = 3 : i64
                } : tensor<3x1x16x1575x72xf16> -> tensor<1x1x16x1575x72xf16>, tensor<1x1x16x1575x72xf16>, tensor<1x1x16x1575x72xf16>

    return %0#0, %0#1, %0#2 : tensor<1x1x16x1575x72xf16>, tensor<1x1x16x1575x72xf16>, tensor<1x1x16x1575x72xf16>

    // CHECK:       [[Reshape_IN:%.+]] = IE.AffineReshape([[INPUT]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [3, 16, 1575, 72]} : tensor<3x1x16x1575x72xf16> -> tensor<3x16x1575x72xf16>
    // CHECK:       [[SPLIT:%.+]]:3 = IE.Split([[Reshape_IN]]) {axis_value = 0 : i64, num_splits = 3 : i64
    // CHECK-SAME:              } : tensor<3x16x1575x72xf16> -> tensor<1x16x1575x72xf16>, tensor<1x16x1575x72xf16>, tensor<1x16x1575x72xf16>
    // CHECK:       [[Reshape_OUT_0:%.+]] = IE.AffineReshape([[SPLIT]]#0) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 1, 16, 1575, 72]} : tensor<1x16x1575x72xf16> -> tensor<1x1x16x1575x72xf16>
    // CHECK:       [[Reshape_OUT_1:%.+]] = IE.AffineReshape([[SPLIT]]#1) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 1, 16, 1575, 72]} : tensor<1x16x1575x72xf16> -> tensor<1x1x16x1575x72xf16>
    // CHECK:       [[Reshape_OUT_2:%.+]] = IE.AffineReshape([[SPLIT]]#2) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 1, 16, 1575, 72]} : tensor<1x16x1575x72xf16> -> tensor<1x1x16x1575x72xf16>

    // CHECK:       return [[Reshape_OUT_0]], [[Reshape_OUT_1]], [[Reshape_OUT_2]] : tensor<1x1x16x1575x72xf16>, tensor<1x1x16x1575x72xf16>, tensor<1x1x16x1575x72xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertSplitOpTo4DAtSecondDim
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x3x16x1575x72xf16>
func.func @ConvertSplitOpTo4DAtSecondDim(%arg0: tensor<1x3x16x1575x72xf16>)
                                -> (tensor<1x1x16x1575x72xf16>, tensor<1x1x16x1575x72xf16>, tensor<1x1x16x1575x72xf16>) {
    %0:3 = IE.Split(%arg0) {axis_value = 1 : i64, num_splits = 3 : i64
                } : tensor<1x3x16x1575x72xf16> -> tensor<1x1x16x1575x72xf16>, tensor<1x1x16x1575x72xf16>, tensor<1x1x16x1575x72xf16>

    return %0#0, %0#1, %0#2 : tensor<1x1x16x1575x72xf16>, tensor<1x1x16x1575x72xf16>, tensor<1x1x16x1575x72xf16>

    // CHECK:       [[Reshape_IN:%.+]] = IE.AffineReshape([[INPUT]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [3, 16, 1575, 72]} : tensor<1x3x16x1575x72xf16> -> tensor<3x16x1575x72xf16>
    // CHECK:       [[SPLIT:%.+]]:3 = IE.Split([[Reshape_IN]]) {axis_value = 0 : i64, num_splits = 3 : i64
    // CHECK-SAME:              } : tensor<3x16x1575x72xf16> -> tensor<1x16x1575x72xf16>, tensor<1x16x1575x72xf16>, tensor<1x16x1575x72xf16>
    // CHECK:       [[Reshape_OUT_0:%.+]] = IE.AffineReshape([[SPLIT]]#0) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 1, 16, 1575, 72]} : tensor<1x16x1575x72xf16> -> tensor<1x1x16x1575x72xf16>
    // CHECK:       [[Reshape_OUT_1:%.+]] = IE.AffineReshape([[SPLIT]]#1) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 1, 16, 1575, 72]} : tensor<1x16x1575x72xf16> -> tensor<1x1x16x1575x72xf16>
    // CHECK:       [[Reshape_OUT_2:%.+]] = IE.AffineReshape([[SPLIT]]#2) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 1, 16, 1575, 72]} : tensor<1x16x1575x72xf16> -> tensor<1x1x16x1575x72xf16>

    // CHECK:       return [[Reshape_OUT_0]], [[Reshape_OUT_1]], [[Reshape_OUT_2]] : tensor<1x1x16x1575x72xf16>, tensor<1x1x16x1575x72xf16>, tensor<1x1x16x1575x72xf16>
}

// -----

// CHECK-LABEL: func.func @NotConvertSplitOpTo4DWithoutDimSizeOne
// CHECK-SAME: [[INPUT:%.+]]: tensor<4x3x16x1575x72xf16>
func.func @NotConvertSplitOpTo4DWithoutDimSizeOne(%arg0: tensor<4x3x16x1575x72xf16>)
                                -> (tensor<4x1x16x1575x72xf16>, tensor<4x1x16x1575x72xf16>, tensor<4x1x16x1575x72xf16>) {
    %0:3 = IE.Split(%arg0) {axis_value = 1 : i64, num_splits = 3 : i64
                } : tensor<4x3x16x1575x72xf16> -> tensor<4x1x16x1575x72xf16>, tensor<4x1x16x1575x72xf16>, tensor<4x1x16x1575x72xf16>

    return %0#0, %0#1, %0#2 : tensor<4x1x16x1575x72xf16>, tensor<4x1x16x1575x72xf16>, tensor<4x1x16x1575x72xf16>

    // CHECK:       [[SPLIT:%.+]]:3 = IE.Split([[INPUT]]) {axis_value = 1 : i64, num_splits = 3 : i64
    // CHECK-SAME:              } : tensor<4x3x16x1575x72xf16> -> tensor<4x1x16x1575x72xf16>, tensor<4x1x16x1575x72xf16>, tensor<4x1x16x1575x72xf16>
    // CHECK:       return [[SPLIT:%.+]]#0, [[SPLIT:%.+]]#1, [[SPLIT:%.+]]#2 : tensor<4x1x16x1575x72xf16>, tensor<4x1x16x1575x72xf16>, tensor<4x1x16x1575x72xf16>
}

// -----

// CHECK-LABEL: func.func @NotConvertSplitOpTo4DWithSmallTensorRank
// CHECK-SAME: [[INPUT:%.+]]: tensor<3x1575x72xf16>
func.func @NotConvertSplitOpTo4DWithSmallTensorRank(%arg0: tensor<3x1575x72xf16>)
                                -> (tensor<1x1575x72xf16>, tensor<1x1575x72xf16>, tensor<1x1575x72xf16>) {
    %0:3 = IE.Split(%arg0) {axis_value = 0 : i64, num_splits = 3 : i64
                } : tensor<3x1575x72xf16> -> tensor<1x1575x72xf16>, tensor<1x1575x72xf16>, tensor<1x1575x72xf16>

    return %0#0, %0#1, %0#2 : tensor<1x1575x72xf16>, tensor<1x1575x72xf16>, tensor<1x1575x72xf16>

    // CHECK:       [[SPLIT:%.+]]:3 = IE.Split([[INPUT]]) {axis_value = 0 : i64, num_splits = 3 : i64
    // CHECK-SAME:              } : tensor<3x1575x72xf16> -> tensor<1x1575x72xf16>, tensor<1x1575x72xf16>, tensor<1x1575x72xf16>
    // CHECK:       return [[SPLIT:%.+]]#0, [[SPLIT:%.+]]#1, [[SPLIT:%.+]]#2 : tensor<1x1575x72xf16>, tensor<1x1575x72xf16>, tensor<1x1575x72xf16>
}

// -----

// CHECK-LABEL: @DynamicQuantize
// CHECK-SAME:  [[INPUT0:%.+]]: tensor<1x6000x400xf32>, [[INPUT1:%.+]]: tensor<1xf32>, [[INPUT2:%.+]]: tensor<1xf32>
func.func @DynamicQuantize(%arg0: tensor<1x6000x400xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>)
-> (tensor<1x6000x400xui8>, tensor<1xf32>, tensor<1xui8>) {
    %output, %scale, %zero_point = IE.DynamicQuantize(%arg0, %arg1, %arg2) {operandSegmentSizes = array<i32: 1, 1, 1>} : tensor<1x6000x400xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<1x6000x400xui8>, tensor<1xf32>, tensor<1xui8>
    return %output, %scale, %zero_point : tensor<1x6000x400xui8>, tensor<1xf32>, tensor<1xui8>

    // CHECK: [[RESHAPE2:%.+]] = IE.AffineReshape([[INPUT2]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2, 3]], shape_value = [1, 1, 1, 1]}
    // CHECK:               tensor<1xf32> -> tensor<1x1x1x1xf32>
    // CHECK: [[RESHAPE1:%.+]] = IE.AffineReshape([[INPUT1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2, 3]], shape_value = [1, 1, 1, 1]}
    // CHECK:               tensor<1xf32> -> tensor<1x1x1x1xf32>
    // CHECK: [[RESHAPE0:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 6000, 400]}
    // CHECK:               tensor<1x6000x400xf32> -> tensor<1x1x6000x400xf32>

    // CHECK: [[OUTPUT:%.+]], [[SCALE:%.+]], [[ZP:%.+]] = IE.DynamicQuantize([[RESHAPE0]], [[RESHAPE1]], [[RESHAPE2]])
    // CHECK: {operandSegmentSizes = array<i32: 1, 1, 1>} : tensor<1x1x6000x400xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>
    // CHECK: -> tensor<1x1x6000x400xui8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xui8>

    // CHECK: [[OUTRESHAPE0:%.+]] = IE.AffineReshape([[OUTPUT]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1], [2]], shape_value = [1, 6000, 400]}
    // CHECK:               tensor<1x1x6000x400xui8> -> tensor<1x6000x400xui8>
    // CHECK: [[OUTRESHAPE1:%.+]] = IE.AffineReshape([[SCALE]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [0]], shape_value = [1]}
    // CHECK:               tensor<1x1x1x1xf32> -> tensor<1xf32>
    // CHECK: [[OUTRESHAPE2:%.+]] = IE.AffineReshape([[ZP]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [0]], shape_value = [1]}
    // CHECK:               tensor<1x1x1x1xui8> -> tensor<1xui8>
    // CHECK: return [[OUTRESHAPE0]], [[OUTRESHAPE1]], [[OUTRESHAPE2]] : tensor<1x6000x400xui8>, tensor<1xf32>, tensor<1xui8>
}

// -----

// CHECK-LABEL: @SDPA
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<32x1x96xf16>, [[ARG1:%.+]]: tensor<32x1024x96xf16>, [[ARG2:%.+]]: tensor<32x96x1024xf16>, [[ARG3:%.+]]: tensor<1x1x1024xf16>)
func.func @SDPA(%arg0: tensor<32x1x96xf16>, %arg1: tensor<32x1024x96xf16>, %arg2: tensor<32x96x1024xf16>, %arg3: tensor<1x1x1024xf16>) -> (tensor<32x1x96xf16>){
    %0 = IE.SDPA(%arg0, %arg1, %arg2, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>} : tensor<32x1x96xf16>, tensor<32x1024x96xf16>, tensor<32x96x1024xf16>, tensor<1x1x1024xf16> -> tensor<32x1x96xf16>
    return %0 : tensor<32x1x96xf16>

// CHECK:               [[AR0:%.+]] = IE.AffineReshape([[ARG0]])
// CHECK-SAME{LITERAL}:     shape_value = [1, 32, 1, 96]} : tensor<32x1x96xf16> -> tensor<1x32x1x96xf16>
// CHECK:               [[AR1:%.+]] = IE.AffineReshape([[ARG1]])
// CHECK-SAME{LITERAL}:     shape_value = [1, 32, 1024, 96]} : tensor<32x1024x96xf16> -> tensor<1x32x1024x96xf16>
// CHECK:               [[AR2:%.+]] = IE.AffineReshape([[ARG2]])
// CHECK-SAME{LITERAL}:     shape_value = [1, 32, 96, 1024]} : tensor<32x96x1024xf16> -> tensor<1x32x96x1024xf16>
// CHECK:               [[AR3:%.+]] = IE.AffineReshape([[ARG3]])
// CHECK-SAME{LITERAL}:     shape_value = [1, 1, 1, 1024]} : tensor<1x1x1024xf16> -> tensor<1x1x1x1024xf16>
// CHECK:               [[SDPA:%.+]] = IE.SDPA([[AR0]], [[AR1]], [[AR2]], [[AR3]]) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>} : tensor<1x32x1x96xf16>, tensor<1x32x1024x96xf16>, tensor<1x32x96x1024xf16>, tensor<1x1x1x1024xf16> -> tensor<1x32x1x96xf16>
// CHECK:               [[AR4:%.+]] = IE.AffineReshape([[SDPA]])
// CHECK-SAME{LITERAL}:     shape_value = [32, 1, 96]} : tensor<1x32x1x96xf16> -> tensor<32x1x96xf16>
// CHECK:               return [[AR4]] : tensor<32x1x96xf16>
}

// -----

// CHECK-LABEL: func.func @Convert5DAddWithInterleavedInputTo4D
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<18360x1x4x1x2xf16>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<18360x8x4x4x2xf16>
func.func @Convert5DAddWithInterleavedInputTo4D(%arg0: tensor<18360x1x4x1x2xf16>, %arg1: tensor<18360x8x4x4x2xf16>) -> tensor<18360x8x4x4x2xf16> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<18360x1x4x1x2xf16>, tensor<18360x8x4x4x2xf16> -> tensor<18360x8x4x4x2xf16>
    return %0 : tensor<18360x8x4x4x2xf16>

    // CHECK:               [[BCAST_SHAPE:%.+]] = const.Declare tensor<5xsi32> =
    // CHECK-SAME{LITERAL}:     dense<[18360, 8, 4, 4, 2]> : tensor<5xsi64>, [#const.CastElemType<si32>]

    // CHECK:               [[AR_IN_1:%.+]] = IE.AffineReshape([[INPUT_1]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1], [2], [2], [3], [3]], shape_value = [1, 18360, 32, 8]} : tensor<18360x8x4x4x2xf16> -> tensor<1x18360x32x8xf16>

    // CHECK:               [[BCAST:%.+]] = IE.Broadcast([[INPUT_0]], [[BCAST_SHAPE]]) {mode = #IE.broadcast_type<NUMPY>} : tensor<18360x1x4x1x2xf16>, tensor<5xsi32> -> tensor<18360x8x4x4x2xf16>
    // CHECK:               [[AR_IN_0:%.+]] = IE.AffineReshape([[BCAST]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1], [2], [2], [3], [3]], shape_value = [1, 18360, 32, 8]} : tensor<18360x8x4x4x2xf16> -> tensor<1x18360x32x8xf16>

    // CHECK:               [[ADD:%.+]] = IE.Add([[AR_IN_0]], [[AR_IN_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x18360x32x8xf16>, tensor<1x18360x32x8xf16> -> tensor<1x18360x32x8xf16>
    // CHECK:               [[AR_OUT:%.+]] = IE.AffineReshape([[ADD]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [0], [1, 2], [3, 4]], shape_value = [18360, 8, 4, 4, 2]} : tensor<1x18360x32x8xf16> -> tensor<18360x8x4x4x2xf16>

    // CHECK:         return [[AR_OUT]] : tensor<18360x8x4x4x2xf16>
}

// -----

// CHECK-LABEL: func.func @Convert5DMultiplyWithInterleavedInputsTo4D
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<18360x1x4x1x2xf16>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x8x1x4x1xf16>
func.func @Convert5DMultiplyWithInterleavedInputsTo4D(%arg0: tensor<18360x1x4x1x2xf16>, %arg1: tensor<1x8x1x4x1xf16>) -> tensor<18360x8x4x4x2xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<18360x1x4x1x2xf16>, tensor<1x8x1x4x1xf16> -> tensor<18360x8x4x4x2xf16>
    return %0 : tensor<18360x8x4x4x2xf16>

    // CHECK:               [[BCAST_SHAPE:%.+]] = const.Declare tensor<5xsi32> =
    // CHECK-SAME{LITERAL}:     dense<[18360, 8, 4, 4, 2]> : tensor<5xsi64>, [#const.CastElemType<si32>]

    // CHECK:               [[BCAST_0:%.+]] = IE.Broadcast([[INPUT_0]], [[BCAST_SHAPE]]) {mode = #IE.broadcast_type<NUMPY>} : tensor<18360x1x4x1x2xf16>, tensor<5xsi32> -> tensor<18360x8x4x4x2xf16>
    // CHECK:               [[AR_IN_0:%.+]] = IE.AffineReshape([[BCAST_0]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1], [2], [2], [3], [3]], shape_value = [1, 18360, 32, 8]} : tensor<18360x8x4x4x2xf16> -> tensor<1x18360x32x8xf16>

    // CHECK:               [[BCAST_1:%.+]] = IE.Broadcast([[INPUT_1]], [[BCAST_SHAPE]]) {mode = #IE.broadcast_type<NUMPY>} : tensor<1x8x1x4x1xf16>, tensor<5xsi32> -> tensor<18360x8x4x4x2xf16>
    // CHECK:               [[AR_IN_1:%.+]] = IE.AffineReshape([[BCAST_1]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1], [2], [2], [3], [3]], shape_value = [1, 18360, 32, 8]} : tensor<18360x8x4x4x2xf16> -> tensor<1x18360x32x8xf16>

    // CHECK:               [[MUL:%.+]] = IE.Multiply([[AR_IN_0]], [[AR_IN_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x18360x32x8xf16>, tensor<1x18360x32x8xf16> -> tensor<1x18360x32x8xf16>
    // CHECK:               [[AR_OUT:%.+]] = IE.AffineReshape([[MUL]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [0], [1, 2], [3, 4]], shape_value = [18360, 8, 4, 4, 2]} : tensor<1x18360x32x8xf16> -> tensor<18360x8x4x4x2xf16>

    // CHECK:         return [[AR_OUT]] : tensor<18360x8x4x4x2xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertRollOpWith4DInputData
// CHECK-SAME:      [[DATA:%.+]]: tensor<1x1x1x16xf16>
func.func @ConvertRollOpWith4DInputData(%arg0: tensor<1x1x1x16xf16>) -> tensor<1x1x1x16xf16> {
    %shift = const.Declare tensor<1xsi64> = dense<5> : tensor<1xsi64>
    %axes =  const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    %1 = IE.Roll(%arg0, %shift, %axes) : tensor<1x1x1x16xf16>, tensor<1xsi64>, tensor<1xsi64> -> tensor<1x1x1x16xf16>
    return %1 : tensor<1x1x1x16xf16>

    // CHECK:    [[SHIFT:%.+]] = const.Declare tensor<1x1x1x2xsi32>
    // CHECK-SAME{LITERAL}: dense<[[[[0, 5]]]]> : tensor<1x1x1x2xsi32>
    // CHECK:    [[AXES:%.+]] = const.Declare tensor<1x1x1x2xsi32>
    // CHECK-SAME{LITERAL}:  dense<[[[[2, 3]]]]> : tensor<1x1x1x2xsi32>

    // CHECK:    [[ROLL:%.+]] = IE.Roll([[DATA]], [[SHIFT]], [[AXES]]) : tensor<1x1x1x16xf16>, tensor<1x1x1x2xsi32>, tensor<1x1x1x2xsi32> -> tensor<1x1x1x16xf16>

    // CHECK:    return [[ROLL]] : tensor<1x1x1x16xf16>

}

// -----

// CHECK-LABEL: func.func @Convert5DAddTo4D
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x1x8x1024x2048xf16>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x1x1024x2048xf16>
func.func @Convert5DAddTo4D(%arg0: tensor<1x1x8x1024x2048xf16>, %arg1: tensor<1x1x1x1024x2048xf16>) -> tensor<1x1x8x1024x2048xf16> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x8x1024x2048xf16>, tensor<1x1x1x1024x2048xf16> -> tensor<1x1x8x1024x2048xf16>
    return %0 : tensor<1x1x8x1024x2048xf16>

    // CHECK:               [[RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT_0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [1, 8, 1024, 2048]} : tensor<1x1x8x1024x2048xf16> -> tensor<1x8x1024x2048xf16>

    // CHECK:               [[RESHAPE_1:%.+]] = IE.AffineReshape([[INPUT_1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 1, 1024, 2048]} : tensor<1x1x1x1024x2048xf16> -> tensor<1x1x1024x2048xf16>

    // CHECK:               [[ADD:%.+]] = IE.Add([[RESHAPE_0]], [[RESHAPE_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x1024x2048xf16>, tensor<1x1x1024x2048xf16> -> tensor<1x8x1024x2048xf16>

    // CHECK:               [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[ADD]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 1, 8, 1024, 2048]} : tensor<1x8x1024x2048xf16> -> tensor<1x1x8x1024x2048xf16>

    // CHECK:               return [[RESHAPE_OUT]] : tensor<1x1x8x1024x2048xf16>
}

// -----

// CHECK-LABEL: func.func @Convert3DSignOpTo4D
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1024x2048xf16>
func.func @Convert3DSignOpTo4D(%arg0: tensor<1x1024x2048xf16>) -> tensor<1x1024x2048xf16> {
    %0 = IE.Sign(%arg0) : tensor<1x1024x2048xf16> -> tensor<1x1024x2048xf16>
    return %0 : tensor<1x1024x2048xf16>

    // CHECK:               [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 1024, 2048]} : tensor<1x1024x2048xf16> -> tensor<1x1x1024x2048xf16>

    // CHECK:               [[SIGN:%.+]] = IE.Sign([[RESHAPE_IN]]) : tensor<1x1x1024x2048xf16> -> tensor<1x1x1024x2048xf16>

    // CHECK:               [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[SIGN]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1], [2]], shape_value = [1, 1024, 2048]} : tensor<1x1x1024x2048xf16> -> tensor<1x1024x2048xf16>

    // CHECK:               return [[RESHAPE_OUT]] : tensor<1x1024x2048xf16>
}

// -----

// CHECK-LABEL: @Convert5DSoftmax
func.func @Convert5DSoftmax(%arg0: tensor<1x1x8x1024x2048xf16>) -> tensor<1x1x8x1024x2048xf16> {
    %0 = IE.SoftMax(%arg0) {axisInd = 4 : i64} : tensor<1x1x8x1024x2048xf16> -> tensor<1x1x8x1024x2048xf16>

    return %0 : tensor<1x1x8x1024x2048xf16>

    // CHECK:       [[VAL_0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [1], [2, 3]], shape_value = [1, 8192, 2048, 1]} : tensor<1x1x8x1024x2048xf16> -> tensor<1x8192x2048x1xf16>
    // CHECK:       [[SoftMax:%.+]] = IE.SoftMax([[VAL_0]])
    // CHECK-SAME:      {axisInd = 2 : i64} : tensor<1x8192x2048x1xf16> -> tensor<1x8192x2048x1xf16>
    // CHECK:       [[VAL_1:%.+]] = IE.AffineReshape([[SoftMax]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2, 3], [4], [4]], shape_value = [1, 1, 8, 1024, 2048]} : tensor<1x8192x2048x1xf16> -> tensor<1x1x8x1024x2048xf16>

    // CHECK:   return [[VAL_1]]
}

// -----

// CHECK-LABEL: @Convert2DReverse
// CHECK-SAME:      [[INPUT:%.+]]: tensor<768x7xf16>
func.func @Convert2DReverse(%arg0: tensor<768x7xf16>) -> tensor<768x7xf16> {
    %0 = IE.Reverse(%arg0) {axis_value = [1], mode = #IE.reverse_mode<INDEX>} : tensor<768x7xf16> -> tensor<768x7xf16>

    return %0 : tensor<768x7xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [1, 2, 3]], shape_value = [768, 7, 1, 1]} : tensor<768x7xf16> -> tensor<768x7x1x1xf16>
    // CHECK:       [[REVERSE:%.+]] = IE.Reverse([[RESHAPE_IN]]) {axis_value = [1], mode = #IE.reverse_mode<INDEX>} : tensor<768x7x1x1xf16> -> tensor<768x7x1x1xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[REVERSE]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [1], [1], [1]], shape_value = [768, 7]} : tensor<768x7x1x1xf16> -> tensor<768x7xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<768x7xf16>
}

// -----

// CHECK-LABEL: @Convert3DReverse
// CHECK-SAME:      [[INPUT:%.+]]: tensor<384x768x7xf16>
func.func @Convert3DReverse(%arg0: tensor<384x768x7xf16>) -> tensor<384x768x7xf16> {
    %0 = IE.Reverse(%arg0) {axis_value = [2], mode = #IE.reverse_mode<INDEX>} : tensor<384x768x7xf16> -> tensor<384x768x7xf16>

    return %0 : tensor<384x768x7xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [1], [2, 3]], shape_value = [384, 768, 7, 1]} : tensor<384x768x7xf16> -> tensor<384x768x7x1xf16>
    // CHECK:       [[REVERSE:%.+]] = IE.Reverse([[RESHAPE_IN]]) {axis_value = [2], mode = #IE.reverse_mode<INDEX>} : tensor<384x768x7x1xf16> -> tensor<384x768x7x1xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[REVERSE]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [1], [2], [2]], shape_value = [384, 768, 7]} : tensor<384x768x7x1xf16> -> tensor<384x768x7xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<384x768x7xf16>
}

// -----

// CHECK-LABEL: @Convert5DReverse
// CHECK-SAME:      [[INPUT:%.+]]: tensor<384x768x7x7x7xf16>
func.func @Convert5DReverse(%arg0: tensor<384x768x7x7x7xf16>) -> tensor<384x768x7x7x7xf16> {
    %0 = IE.Reverse(%arg0) {axis_value = [2, 3, 4], mode = #IE.reverse_mode<INDEX>} : tensor<384x768x7x7x7xf16> -> tensor<384x768x7x7x7xf16>

    return %0 : tensor<384x768x7x7x7xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [294912, 7, 7, 7]} : tensor<384x768x7x7x7xf16> -> tensor<294912x7x7x7xf16>
    // CHECK:       [[REVERSE:%.+]] = IE.Reverse([[RESHAPE_IN]]) {axis_value = [1, 2, 3], mode = #IE.reverse_mode<INDEX>} : tensor<294912x7x7x7xf16> -> tensor<294912x7x7x7xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[REVERSE]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [384, 768, 7, 7, 7]} : tensor<294912x7x7x7xf16> -> tensor<384x768x7x7x7xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<384x768x7x7x7xf16>
}

// -----

// CHECK-LABEL: @Convert5DReverseWith2Axes
// CHECK-SAME:      [[INPUT:%.+]]: tensor<384x768x7x7x7xf16>
func.func @Convert5DReverseWith2Axes(%arg0: tensor<384x768x7x7x7xf16>) -> tensor<384x768x7x7x7xf16> {
    %0 = IE.Reverse(%arg0) {axis_value = [1, 4], mode = #IE.reverse_mode<INDEX>} : tensor<384x768x7x7x7xf16> -> tensor<384x768x7x7x7xf16>

    return %0 : tensor<384x768x7x7x7xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [1], [2], [2], [3]], shape_value = [384, 768, 49, 7]} : tensor<384x768x7x7x7xf16> -> tensor<384x768x49x7xf16>
    // CHECK:       [[REVERSE:%.+]] = IE.Reverse([[RESHAPE_IN]]) {axis_value = [1, 3], mode = #IE.reverse_mode<INDEX>} : tensor<384x768x49x7xf16> -> tensor<384x768x49x7xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[REVERSE]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [384, 768, 7, 7, 7]} : tensor<384x768x49x7xf16> -> tensor<384x768x7x7x7xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<384x768x7x7x7xf16>
}

// -----

// CHECK-LABEL: @NotConvert5DReverseWith2Axes
// CHECK-SAME:      [[INPUT:%.+]]: tensor<384x768x7x7x7xf16>
func.func @NotConvert5DReverseWith2Axes(%arg0: tensor<384x768x7x7x7xf16>) -> tensor<384x768x7x7x7xf16> {
    %0 = IE.Reverse(%arg0) {axis_value = [1, 3], mode = #IE.reverse_mode<INDEX>} : tensor<384x768x7x7x7xf16> -> tensor<384x768x7x7x7xf16>

    return %0 : tensor<384x768x7x7x7xf16>

    // CHECK:       [[REVERSE:%.+]] = IE.Reverse([[INPUT]]) {axis_value = [1, 3], mode = #IE.reverse_mode<INDEX>} : tensor<384x768x7x7x7xf16> -> tensor<384x768x7x7x7xf16>

    // CHECK:   return [[REVERSE]] : tensor<384x768x7x7x7xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:0, {0.1:128, 0.2:128}>

// Note: Axis changed from 0 to 2.
// CHECK: [[Q_ELEM_TYPE:!.+]] = !quant.uniform<u8:f16:2, {1.000000e-01:128,2.000000e-01:128}>

func.func @QuantizedAddWithQDQ(%arg0: tensor<2x4xf16>) -> tensor<2x4xf16> {
    %quant = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<2x4xf16> -> tensor<2x4x!qElemType>
    %add = IE.Add(%quant, %quant) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x4x!qElemType>, tensor<2x4x!qElemType> -> tensor<2x4x!qElemType>
    %dequant = IE.Dequantize(%add) {dstElemType = f16} : tensor<2x4x!qElemType> -> tensor<2x4xf16>
    return %dequant : tensor<2x4xf16>

    // Note: The IE.AffineReshapes are inserted by the canonicalizer. By default, they are IE.Reshapes.
    // CHECK:   [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 2, 4]} : tensor<2x4xf16> -> tensor<1x1x2x4xf16>
    // CHECK:   [[QUANT:%.+]] = IE.Quantize([[AFFINE_RESHAPE]]) {dstElemType = [[Q_ELEM_TYPE]]} : tensor<1x1x2x4xf16> -> tensor<1x1x2x4x[[Q_ELEM_TYPE]]>
    // CHECK:   [[MAX_POOL:%.+]] = IE.Add([[QUANT]], [[QUANT]])
    // CHECK:   [[DEQUANT:%.+]] = IE.Dequantize([[MAX_POOL]]) {dstElemType = f16} : tensor<1x1x2x4x[[Q_ELEM_TYPE]]> -> tensor<1x1x2x4xf16>
    // CHECK:   [[RESULT:%.+]] = IE.AffineReshape([[DEQUANT]])
    // CHECK:   return [[RESULT]] : tensor<2x4xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:0, {0.1:128, 0.2:128}>

// Note: Axis changed from 0 to 2.
// CHECK: [[Q_ELEM_TYPE:!.+]] = !quant.uniform<u8:f16:2, {1.000000e-01:128,2.000000e-01:128}>

func.func @QuantizedQDQ(%arg0: tensor<2x4xf16>) -> tensor<2x4xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<2x4xf16> -> tensor<2x4x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<2x4x!qElemType> -> tensor<2x4xf16>
    return %1 : tensor<2x4xf16>

    // Note: The IE.AffineReshapes are inserted by the canonicalizer. By default, they are IE.Reshapes.
    // CHECK:   [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 2, 4]} : tensor<2x4xf16> -> tensor<1x1x2x4xf16>
    // CHECK:   [[QUANT:%.+]] = IE.Quantize([[AFFINE_RESHAPE]]) {dstElemType = [[Q_ELEM_TYPE]]} : tensor<1x1x2x4xf16> -> tensor<1x1x2x4x[[Q_ELEM_TYPE]]>
    // CHECK:   [[DEQUANT:%.+]] = IE.Dequantize([[QUANT]]) {dstElemType = f16} : tensor<1x1x2x4x[[Q_ELEM_TYPE]]> -> tensor<1x1x2x4xf16>
    // CHECK:   [[RESULT:%.+]] = IE.AffineReshape([[DEQUANT]])
    // CHECK:   return [[RESULT]] : tensor<2x4xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:0, {0.1:128, 0.2:128}>

// Note: Axis changed from 0 to 2.
// CHECK: [[Q_ELEM_TYPE:!.+]] = !quant.uniform<u8:f16:2, {1.000000e-01:128,2.000000e-01:128}>

func.func @QuantizedAddMulWithQDQ(%arg0: tensor<2x4xf16>) -> tensor<2x4xf16> {
    %quant = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<2x4xf16> -> tensor<2x4x!qElemType>
    %add = IE.Add(%quant, %quant) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x4x!qElemType>, tensor<2x4x!qElemType> -> tensor<2x4x!qElemType>
    %dequant = IE.Dequantize(%add) {dstElemType = f16} : tensor<2x4x!qElemType> -> tensor<2x4xf16>
    %mul = IE.Multiply(%dequant, %dequant) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x4xf16>, tensor<2x4xf16> -> tensor<2x4xf16>
    return %mul : tensor<2x4xf16>

    // Note: The IE.AffineReshapes are inserted by the canonicalizer. By default, they are IE.Reshapes.
    // CHECK:   [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 2, 4]} : tensor<2x4xf16> -> tensor<1x1x2x4xf16>
    // CHECK:   [[QUANT:%.+]] = IE.Quantize([[AFFINE_RESHAPE]]) {dstElemType = [[Q_ELEM_TYPE]]} : tensor<1x1x2x4xf16> -> tensor<1x1x2x4x[[Q_ELEM_TYPE]]>
    // CHECK:   [[ADD:%.+]] = IE.Add([[QUANT]], [[QUANT]])
    // CHECK:   [[DEQUANT:%.+]] = IE.Dequantize([[ADD]]) {dstElemType = f16} : tensor<1x1x2x4x[[Q_ELEM_TYPE]]> -> tensor<1x1x2x4xf16>
    // CHECK:   [[MUL:%.+]] = IE.Multiply([[DEQUANT]], [[DEQUANT]])
    // CHECK:   [[RESULT:%.+]] = IE.AffineReshape([[MUL]])
    // CHECK:   return [[RESULT]] : tensor<2x4xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:0, {0.1:128, 0.2:128}>

// CHECK: [[Q_ELEM_TYPE_0:!.+]] = !quant.uniform<u8:f16:0, {1.000000e-01:128,2.000000e-01:128}>
// CHECK: [[Q_ELEM_TYPE_1:!.+]] = !quant.uniform<u8:f16:2, {1.000000e-01:128,2.000000e-01:128}>

func.func @QuantizedAdd(%arg0: tensor<2x4x!qElemType>) -> tensor<2x4x!qElemType> {
    %add = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x4x!qElemType>, tensor<2x4x!qElemType> -> tensor<2x4x!qElemType>
    return %add : tensor<2x4x!qElemType>

    // Note: Here IE.AffineReshape is inserted by ConvertShapeTo4D directly.
    // CHECK:   [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape
    // CHECK-SAME{LITERAL}:                  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 2, 4]}
    // CHECK-SAME:                           tensor<2x4x[[Q_ELEM_TYPE_0]]> -> tensor<1x1x2x4x[[Q_ELEM_TYPE_1]]>
    // CHECK:   [[ADD:%.+]] = IE.Add([[AFFINE_RESHAPE]], [[AFFINE_RESHAPE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:   [[RESULT:%.+]] = IE.AffineReshape([[ADD]])
    // CHECK-SAME{LITERAL}:          {dim_mapping = [[0], [0], [0], [1]], shape_value = [2, 4]}
    // CHECK-SAME:                   tensor<1x1x2x4x[[Q_ELEM_TYPE_1]]> -> tensor<2x4x[[Q_ELEM_TYPE_0]]>
    // CHECK:   return [[RESULT]] : tensor<2x4x[[Q_ELEM_TYPE_0]]>
}

// -----

!dynType = tensor<1x1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 1, 128]> : tensor<5xsi64>, order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>}>

// CHECK-LABEL: @SemiDynamicAdd5D
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 1, 128]> : tensor<5xsi64>, order = #NCDHW}>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x1x1x128xf16>
func.func @SemiDynamicAdd5D(%arg0: !dynType, %arg1: tensor<1x1x1x1x128xf16>) -> !dynType {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : !dynType, tensor<1x1x1x1x128xf16> -> !dynType
    return %0 : !dynType

    // CHECK:    [[SHAPE_OUT:%.+]] = const.Declare tensor<5xsi32> = dense<[1, 1, -1, 1, 128]> : tensor<5xsi32>
    // CHECK:    [[SHAPE_IN:%.+]] = const.Declare tensor<4xsi32> = dense<[1, -1, 1, 128]> : tensor<4xsi32>

    // CHECK:    [[RESHAPE_IN_0:%.+]] = IE.DynamicReshape([[INPUT_0]], [[SHAPE_IN]]) {output_bounds = [1, 500, 1, 128], output_shape = [1, -9223372036854775808, 1, 128]} : tensor<1x1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 1, 128]> : tensor<5xsi64>, order = #NCDHW}>, tensor<4xsi32> -> tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 1, 128]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:    [[RESHAPE_IN_1:%.+]] = IE.AffineReshape([[INPUT_1]]) {
    // CHECK-SAME{LITERAL}:                 dim_mapping = [[0], [1], [2], [2], [3]], shape_value = [1, 1, 1, 128]} : tensor<1x1x1x1x128xf16> -> tensor<1x1x1x128xf16>
    // CHECK:    [[ADD:%.+]] = IE.Add([[RESHAPE_IN_0]], [[RESHAPE_IN_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 1, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16> -> tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 1, 128]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:    [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[ADD]], [[SHAPE_OUT]]) {only_set_shape, output_bounds = [1, 1, 500, 1, 128], output_shape = [1, 1, -9223372036854775808, 1, 128]} : tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 1, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<5xsi32> -> tensor<1x1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 1, 128]> : tensor<5xsi64>, order = #NCDHW}>

    // CHECK:    return [[RESHAPE_OUT]] : tensor<1x1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 1, 128]> : tensor<5xsi64>, order = #NCDHW}>
}

// -----

!dynType = tensor<?x?x?x?x128xf16, {bounds = #const.OpaqueI64Elements<[2, 3, 4, 5, 128]> : tensor<5xsi64>, order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>}>

// CHECK-LABEL: @SemiMultiaxisDynamicAdd5D
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<?x?x?x?x128xf16, {bounds = #const.OpaqueI64Elements<[2, 3, 4, 5, 128]> : tensor<5xsi64>, order = #NCDHW}>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x1x1x128xf16>
func.func @SemiMultiaxisDynamicAdd5D(%arg0: !dynType, %arg1: tensor<1x1x1x1x128xf16>) -> !dynType {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : !dynType, tensor<1x1x1x1x128xf16> -> !dynType
    return %0 : !dynType

    // CHECK:    [[SHAPE_OUT:%.+]] = const.Declare tensor<5xsi32> = dense<[-1, -1, -1, -1, 128]> : tensor<5xsi32>
    // CHECK:    [[SHAPE_IN:%.+]] = const.Declare tensor<4xsi32> = dense<[1, -1, -1, 128]> : tensor<4xsi32>

    // CHECK:    [[RESHAPE_IN_0:%.+]] = IE.DynamicReshape([[INPUT_0]], [[SHAPE_IN]]) {output_bounds = [1, 6, 20, 128], output_shape = [1, -9223372036854775808, -9223372036854775808, 128]} : tensor<?x?x?x?x128xf16, {bounds = #const.OpaqueI64Elements<[2, 3, 4, 5, 128]> : tensor<5xsi64>, order = #NCDHW}>, tensor<4xsi32> -> tensor<1x?x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 6, 20, 128]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:    [[RESHAPE_IN_1:%.+]] = IE.AffineReshape([[INPUT_1]]) {
    // CHECK-SAME{LITERAL}:                 dim_mapping = [[0], [1], [2], [2], [3]], shape_value = [1, 1, 1, 128]} : tensor<1x1x1x1x128xf16> -> tensor<1x1x1x128xf16>
    // CHECK:    [[ADD:%.+]] = IE.Add([[RESHAPE_IN_0]], [[RESHAPE_IN_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x?x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 6, 20, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16> -> tensor<1x?x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 6, 20, 128]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:    [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[ADD]], [[SHAPE_OUT]]) {only_set_shape, output_bounds = [2, 3, 4, 5, 128], output_shape = [-9223372036854775808, -9223372036854775808, -9223372036854775808, -9223372036854775808, 128]} : tensor<1x?x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 6, 20, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<5xsi32> -> tensor<?x?x?x?x128xf16, {bounds = #const.OpaqueI64Elements<[2, 3, 4, 5, 128]> : tensor<5xsi64>, order = #NCDHW}>

    // CHECK:    return [[RESHAPE_OUT]] : tensor<?x?x?x?x128xf16, {bounds = #const.OpaqueI64Elements<[2, 3, 4, 5, 128]> : tensor<5xsi64>, order = #NCDHW}>
}

// -----

!dynType1 = tensor<1x1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 1, 128]> : tensor<5xsi64>, order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>}>
!dynType2 = tensor<1x1x?x1x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 1, 1]> : tensor<5xsi64>, order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>}>

// CHECK-LABEL: @FullyDynamicAdd5D
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 1, 128]> : tensor<5xsi64>, order = #NCDHW}>
// CHECK-SAME:      [[INPUT_1:%.+]]:  tensor<1x1x?x1x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 1, 1]> : tensor<5xsi64>, order = #NCDHW}>
func.func @FullyDynamicAdd5D(%arg0: !dynType1, %arg1: !dynType2) -> !dynType1 {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : !dynType1, !dynType2 -> !dynType1
    return %0 : !dynType1

    // CHECK:    [[SHAPE_IN_1:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 1, -1, 1]> : tensor<4xsi32>
    // CHECK:    [[SHAPE_OUT:%.+]] = const.Declare tensor<5xsi32> = dense<[1, 1, -1, 1, 128]> : tensor<5xsi32>
    // CHECK:    [[SHAPE_IN_0:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 1, -1, 128]> : tensor<4xsi32>

    // CHECK:    [[RESHAPE_IN_0:%.+]] = IE.DynamicReshape([[INPUT_0]], [[SHAPE_IN_0]]) {output_bounds = [1, 1, 500, 128], output_shape = [1, 1, -9223372036854775808, 128]} : tensor<1x1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 1, 128]> : tensor<5xsi64>, order = #NCDHW}>, tensor<4xsi32> -> tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 128]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:    [[RESHAPE_IN_1:%.+]] = IE.DynamicReshape([[INPUT_1]], [[SHAPE_IN_1]]) {output_bounds = [1, 1, 500, 1], output_shape = [1, 1, -9223372036854775808, 1]} : tensor<1x1x?x1x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 1, 1]> : tensor<5xsi64>, order = #NCDHW}>, tensor<4xsi32> -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 1]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:    [[ADD:%.+]] = IE.Add([[RESHAPE_IN_0]], [[RESHAPE_IN_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 1]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 128]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:    [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[ADD]], [[SHAPE_OUT]]) {only_set_shape, output_bounds = [1, 1, 500, 1, 128], output_shape = [1, 1, -9223372036854775808, 1, 128]} : tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<5xsi32> -> tensor<1x1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 1, 128]> : tensor<5xsi64>, order = #NCDHW}>

    // CHECK:    return [[RESHAPE_OUT]] : tensor<1x1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 1, 128]> : tensor<5xsi64>, order = #NCDHW}>
}

// -----

!dynType = tensor<?x?x768xf16, {bounds = #const.OpaqueI64Elements<[10, 10, 768]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>

// CHECK-LABEL: @SemiDynamicAdd3D
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<?x?x768xf16, {bounds = #const.OpaqueI64Elements<[10, 10, 768]> : tensor<3xsi64>, order = #CHW}>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x768xf16>
func.func @SemiDynamicAdd3D(%arg0: !dynType, %arg1: tensor<1x1x768xf16>) -> !dynType {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : !dynType, tensor<1x1x768xf16> -> !dynType
    return %0 : !dynType

    // CHECK:    [[SHAPE_OUT:%.+]] = const.Declare tensor<3xsi32> = dense<[-1, -1, 768]> : tensor<3xsi32>
    // CHECK:    [[SHAPE_IN:%.+]] = const.Declare tensor<4xsi32> = dense<[1, -1, -1, 768]> : tensor<4xsi32>

    // CHECK:    [[RESHAPE_IN_1:%.+]] = IE.AffineReshape([[INPUT_1]]) {
    // CHECK-SAME{LITERAL}:                 dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1, 1, 768]} : tensor<1x1x768xf16> -> tensor<1x1x1x768xf16>
    // CHECK:    [[RESHAPE_IN_0:%.+]] = IE.DynamicReshape([[INPUT_0]], [[SHAPE_IN]]) {output_bounds = [1, 10, 10, 768], output_shape = [1, -9223372036854775808, -9223372036854775808, 768]} : tensor<?x?x768xf16, {bounds = #const.OpaqueI64Elements<[10, 10, 768]> : tensor<3xsi64>, order = #CHW}>, tensor<4xsi32> -> tensor<1x?x?x768xf16, {bounds = #const.OpaqueI64Elements<[1, 10, 10, 768]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:    [[ADD:%.+]] = IE.Add([[RESHAPE_IN_0]], [[RESHAPE_IN_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x?x?x768xf16, {bounds = #const.OpaqueI64Elements<[1, 10, 10, 768]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x768xf16> -> tensor<1x?x?x768xf16, {bounds = #const.OpaqueI64Elements<[1, 10, 10, 768]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:    [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[ADD]], [[SHAPE_OUT]]) {only_set_shape, output_bounds = [10, 10, 768], output_shape = [-9223372036854775808, -9223372036854775808, 768]} : tensor<1x?x?x768xf16, {bounds = #const.OpaqueI64Elements<[1, 10, 10, 768]> : tensor<4xsi64>, order = #NCHW}>, tensor<3xsi32> -> tensor<?x?x768xf16, {bounds = #const.OpaqueI64Elements<[10, 10, 768]> : tensor<3xsi64>, order = #CHW}>

    // CHECK:    return [[RESHAPE_OUT]] : tensor<?x?x768xf16, {bounds = #const.OpaqueI64Elements<[10, 10, 768]> : tensor<3xsi64>, order = #CHW}>
}

// -----

!dynType = tensor<?x1x1x2x64xf16, {bounds = #const.OpaqueI64Elements<[500, 1, 1, 2, 64]> : tensor<5xsi64>, order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>}>

// CHECK-LABEL: @DynamicFakeQuantize5D
// CHECK-SAME:      [[INPUT:%.+]]: tensor<?x1x1x2x64xf16, {bounds = #const.OpaqueI64Elements<[500, 1, 1, 2, 64]> : tensor<5xsi64>, order = #NCDHW}>
func.func @DynamicFakeQuantize5D(%arg0: !dynType) -> !dynType {
    %cst = const.Declare tensor<1x1x1x1x1xf16> = dense<1.0> : tensor<1x1x1x1x1xf16>
    %0 = IE.FakeQuantize(%arg0, %cst, %cst, %cst, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : !dynType, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> !dynType
    return %0 : !dynType

    // CHECK:    [[SHAPE_OUT:%.+]] = const.Declare tensor<5xsi32> = dense<[-1, 1, 1, 2, 64]> : tensor<5xsi32>
    // CHECK:    [[FQ_CONST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>]
    // CHECK:    [[SHAPE_IN:%.+]] = const.Declare tensor<4xsi32> = dense<[1, -1, 2, 64]> : tensor<4xsi32>

    // CHECK:    [[RESHAPE_IN:%.+]] = IE.DynamicReshape([[INPUT]], [[SHAPE_IN]]) {only_set_shape, output_bounds = [1, 500, 2, 64], output_shape = [1, -9223372036854775808, 2, 64]} : tensor<?x1x1x2x64xf16, {bounds = #const.OpaqueI64Elements<[500, 1, 1, 2, 64]> : tensor<5xsi64>, order = #NCDHW}>, tensor<4xsi32> -> tensor<1x?x2x64xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 2, 64]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[RESHAPE_IN]], [[FQ_CONST]], [[FQ_CONST]], [[FQ_CONST]], [[FQ_CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x?x2x64xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 2, 64]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x?x2x64xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 2, 64]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:    [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[FQ]], [[SHAPE_OUT]]) {only_set_shape, output_bounds = [500, 1, 1, 2, 64], output_shape = [-9223372036854775808, 1, 1, 2, 64]} : tensor<1x?x2x64xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 2, 64]> : tensor<4xsi64>, order = #NCHW}>, tensor<5xsi32> -> tensor<?x1x1x2x64xf16, {bounds = #const.OpaqueI64Elements<[500, 1, 1, 2, 64]> : tensor<5xsi64>, order = #NCDHW}>

    // CHECK:    return [[RESHAPE_OUT]] : tensor<?x1x1x2x64xf16, {bounds = #const.OpaqueI64Elements<[500, 1, 1, 2, 64]> : tensor<5xsi64>, order = #NCDHW}>
}

// -----

!dynType = tensor<1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 2]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>

// CHECK-LABEL: @DynamicFakeQuantize3D
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 2]> : tensor<3xsi64>, order = #CHW}>
func.func @DynamicFakeQuantize3D(%arg0: !dynType) -> !dynType {
    %cst = const.Declare tensor<1x1x1xf16> = dense<1.0> : tensor<1x1x1xf16>
    %0 = IE.FakeQuantize(%arg0, %cst, %cst, %cst, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : !dynType, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16> -> !dynType
    return %0 : !dynType

    // CHECK:    [[SHAPE_OUT:%.+]] = const.Declare tensor<3xsi32> = dense<[1, -1, 2]> : tensor<3xsi32>
    // CHECK:    [[FQ_CONST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>]
    // CHECK:    [[SHAPE_IN:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 1, -1, 2]> : tensor<4xsi32>

    // CHECK:    [[RESHAPE_IN:%.+]] = IE.DynamicReshape([[INPUT]], [[SHAPE_IN]]) {only_set_shape, output_bounds = [1, 1, 500, 2], output_shape = [1, 1, -9223372036854775808, 2]} : tensor<1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 2]> : tensor<3xsi64>, order = #CHW}>, tensor<4xsi32> -> tensor<1x1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[RESHAPE_IN]], [[FQ_CONST]], [[FQ_CONST]], [[FQ_CONST]], [[FQ_CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:    [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[FQ]], [[SHAPE_OUT]]) {only_set_shape, output_bounds = [1, 500, 2], output_shape = [1, -9223372036854775808, 2]} : tensor<1x1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>, tensor<3xsi32> -> tensor<1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 2]> : tensor<3xsi64>, order = #CHW}>

    // CHECK:    return [[RESHAPE_OUT]] : tensor<1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 500, 2]> : tensor<3xsi64>, order = #CHW}>
}

// -----

// CHECK-LABEL: @Convert3DSoftplus
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x512x1024xf16>
func.func @Convert3DSoftplus(%arg0: tensor<1x512x1024xf16>) -> tensor<1x512x1024xf16> {
    %0 = IE.SoftPlus(%arg0) : tensor<1x512x1024xf16> -> tensor<1x512x1024xf16>
    return %0 : tensor<1x512x1024xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 512, 1024]} : tensor<1x512x1024xf16> -> tensor<1x1x512x1024xf16>
    // CHECK:       [[SOFTPLUS:%.+]] = IE.SoftPlus([[RESHAPE_IN]]) : tensor<1x1x512x1024xf16> -> tensor<1x1x512x1024xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[SOFTPLUS]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [2]], shape_value = [1, 512, 1024]} : tensor<1x1x512x1024xf16> -> tensor<1x512x1024xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x512x1024xf16>
}

// -----

// CHECK-LABEL: @Convert5DSoftplus
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x6x80x38x38xf16>
func.func @Convert5DSoftplus(%arg0: tensor<1x6x80x38x38xf16>) -> tensor<1x6x80x38x38xf16> {
    %0 = IE.SoftPlus(%arg0) : tensor<1x6x80x38x38xf16> -> tensor<1x6x80x38x38xf16>
    return %0 : tensor<1x6x80x38x38xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [1], [2], [3], [3]], shape_value = [1, 6, 80, 1444]} : tensor<1x6x80x38x38xf16> -> tensor<1x6x80x1444xf16>
    // CHECK:       [[SOFTPLUS:%.+]] = IE.SoftPlus([[RESHAPE_IN]]) : tensor<1x6x80x1444xf16> -> tensor<1x6x80x1444xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[SOFTPLUS]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [1], [2], [3, 4]], shape_value = [1, 6, 80, 38, 38]} : tensor<1x6x80x1444xf16> -> tensor<1x6x80x38x38xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x6x80x38x38xf16>
}

// -----

// CHECK-LABEL: @Convert3DNormalizeL2
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x801x28xf16>
func.func @Convert3DNormalizeL2(%arg0: tensor<1x801x28xf16>) -> tensor<1x801x28xf16> {
    %0 = IE.NormalizeL2(%arg0) {axes_value = [2], eps = 9.999999960041972E-13 : f64, eps_mode = #IE.eps_mode<ADD>} : tensor<1x801x28xf16> -> tensor<1x801x28xf16>

    return %0 : tensor<1x801x28xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 801, 28]} : tensor<1x801x28xf16> -> tensor<1x1x801x28xf16>
    // CHECK:       [[NORMALIZE:%.+]] = IE.NormalizeL2([[RESHAPE_IN]]) {axes_value = [3], eps = 9.999999960041972E-13 : f64, eps_mode = #IE.eps_mode<ADD>} : tensor<1x1x801x28xf16> -> tensor<1x1x801x28xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[NORMALIZE]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [0], [1], [2]], shape_value = [1, 801, 28]} : tensor<1x1x801x28xf16> -> tensor<1x801x28xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x801x28xf16>
}

// -----

// CHECK-LABEL: @Convert5DNormalizeL2
// CHECK-SAME:      [[INPUT:%.+]]: tensor<384x768x7x7x7xf16>
func.func @Convert5DNormalizeL2(%arg0: tensor<384x768x7x7x7xf16>) -> tensor<384x768x7x7x7xf16> {
    %0 = IE.NormalizeL2(%arg0) {axes_value = [1], eps = 9.999999960041972E-13 : f64, eps_mode = #IE.eps_mode<ADD>} : tensor<384x768x7x7x7xf16> -> tensor<384x768x7x7x7xf16>
    return %0 : tensor<384x768x7x7x7xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1], [2], [3], [3], [3]], shape_value = [1, 384, 768, 343]} : tensor<384x768x7x7x7xf16> -> tensor<1x384x768x343xf16>
    // CHECK:       [[NORMALIZE:%.+]] = IE.NormalizeL2([[RESHAPE_IN]]) {axes_value = [2], eps = 9.999999960041972E-13 : f64, eps_mode = #IE.eps_mode<ADD>} : tensor<1x384x768x343xf16> -> tensor<1x384x768x343xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[NORMALIZE]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [384, 768, 7, 7, 7]} : tensor<1x384x768x343xf16> -> tensor<384x768x7x7x7xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<384x768x7x7x7xf16>
}
