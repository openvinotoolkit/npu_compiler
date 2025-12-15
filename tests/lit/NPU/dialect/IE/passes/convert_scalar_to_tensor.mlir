//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-scalar-to-tensor %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @Gather
func.func @Gather(%arg0: tensor<18x8x72x64xf16>) -> tensor<8x72x64xf16> {
    %cst = const.Declare tensor<si32> = dense<1> : tensor<si32>

    %0 = IE.Gather(%arg0, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64}
            : tensor<18x8x72x64xf16>, tensor<si32> -> tensor<8x72x64xf16>

    return %0 : tensor<8x72x64xf16>

    // CHECK-DAG:       [[VAL0:%.*]] = const.Declare tensor<1xsi32> = dense<1> : tensor<si32>, [#const.Reshape<[1]>]
    // CHECK:       [[VAL1:%.*]] = IE.Gather(%arg0, [[VAL0]]) {axis_value = 0 : i64, batch_dims = 0 : i64}
    // CHECK-SAME:      : tensor<18x8x72x64xf16>, tensor<1xsi32> -> tensor<1x8x72x64xf16>
    // CHECK:       [[VAL2:%.*]] = IE.Reshape([[VAL1]]) {shape_value = [8, 72, 64]} : tensor<1x8x72x64xf16> -> tensor<8x72x64xf16>
    // CHECK:       return [[VAL2]]
}

// CHECK-LABEL: @GatherAxisCase
func.func @GatherAxisCase(%arg0: tensor<16x3x3x3xf16>) -> tensor<27x3x3x3xf16> {
    %cst = const.Declare tensor<27xsi64> = dense<1> : tensor<27xsi64>
    %cst_0 = const.Declare tensor<si64> = dense<0> : tensor<si64>
    %0 = IE.Gather(%arg0, %cst, %cst_0) {batch_dims = 0 : i64} : tensor<16x3x3x3xf16>, tensor<27xsi64>, tensor<si64> -> tensor<27x3x3x3xf16>
    return %0 : tensor<27x3x3x3xf16>

    // CHECK-DAG:       [[CST:%.*]] = const.Declare tensor<27xsi64> = dense<1> : tensor<27xsi64>
    // CHECK-DAG:       [[CST0:%.*]] = const.Declare tensor<si64> = dense<0> : tensor<si64>
    // CHECK:       [[VAL0:%.*]] = IE.Reshape([[CST0]]) {shape_value = [1], special_zero} : tensor<si64> -> tensor<1xsi64>
    // CHECK:       [[VAL1:%.*]] = IE.Gather(%arg0, [[CST]], [[VAL0]]) {batch_dims = 0 : i64} : tensor<16x3x3x3xf16>, tensor<27xsi64>, tensor<1xsi64> -> tensor<27x3x3x3xf16>
    // CHECK:       [[VAL2:%.*]] = IE.Reshape([[VAL1]]) {shape_value = [27, 3, 3, 3]} : tensor<27x3x3x3xf16> -> tensor<27x3x3x3xf16>
    // CHECK:       return [[VAL2]]
  }

// CHECK-LABEL: @TopK
func.func @TopK(%arg0: tensor<6x12x10x24xf16>) -> (tensor<6x3x10x24xf16>, tensor<6x3x10x24xi64>) {
    %cst = const.Declare tensor<si32> = dense<3> : tensor<si32>

    %0:2 = IE.TopK(%arg0, %cst) {axis = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_VALUES>, element_type = i64}
            : tensor<6x12x10x24xf16>, tensor<si32> -> tensor<6x3x10x24xf16>, tensor<6x3x10x24xi64>

    return %0#0, %0#1 : tensor<6x3x10x24xf16>, tensor<6x3x10x24xi64>

    // CHECK-DAG:       [[CST0:%.*]] = const.Declare tensor<1xsi32> = dense<3> : tensor<si32>, [#const.Reshape<[1]>]
    // CHECK:       [[VAL1:%.*]], [[VAL2:%.*]] = IE.TopK(%arg0, [[CST0]])
    // CHECK-SAME:      {axis = 1 : i64, element_type = i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_VALUES>}
    // CHECK-SAME:      : tensor<6x12x10x24xf16>, tensor<1xsi32> -> tensor<6x3x10x24xf16>, tensor<6x3x10x24xi64>
    // CHECK:       return [[VAL1]], [[VAL2]]
}

// CHECK-LABEL: @Multiply
func.func @Multiply(%arg0: tensor<f16>, %arg1: tensor<1x16x32xf16>) -> tensor<1x16x32xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<f16>, tensor<1x16x32xf16> -> tensor<1x16x32xf16>

    return %0 : tensor<1x16x32xf16>

    // CHECK:       [[VAL0:%.*]] = IE.Reshape(%arg0) {shape_value = [1]} : tensor<f16> -> tensor<1xf16>
    // CHECK:       [[VAL1:%.*]] = IE.Multiply([[VAL0]], %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf16>, tensor<1x16x32xf16> -> tensor<1x16x32xf16>
    // CHECK:       return [[VAL1]]
}

// CHECK-LABEL: @AddResultRank0
func.func @AddResultRank0(%arg0: tensor<f16>, %arg1: tensor<f16>) -> tensor<f16> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<f16>, tensor<f16> -> tensor<f16>

    return %0 : tensor<f16>

    // CHECK:       [[VAL1:%.*]] = IE.Reshape(%arg1) {shape_value = [1]} : tensor<f16> -> tensor<1xf16>
    // CHECK:       [[VAL0:%.*]] = IE.Reshape(%arg0) {shape_value = [1]} : tensor<f16> -> tensor<1xf16>
    // CHECK:       [[VAL2:%.*]] = IE.Add([[VAL0]], [[VAL1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf16>, tensor<1xf16> -> tensor<1xf16>
    // CHECK:       [[VAL3:%.*]] = IE.Reshape([[VAL2]]) {shape_value = []} : tensor<1xf16> -> tensor<f16>
    // CHECK:       return [[VAL3]]
}

// CHECK-LABEL: @ReduceMax
func.func @ReduceMax(%arg0: tensor<10xf32>) -> tensor<1xf32> {
    %cst = const.Declare tensor<1xsi32> = dense<0> : tensor<1xsi32>
    %0 = IE.ReduceMax(%arg0, %cst) : tensor<10xf32>, tensor<1xsi32> -> tensor<1xf32>
    return %0 : tensor<1xf32>

    // CHECK:       [[CST:%.*]] = const.Declare tensor<1xsi32> = dense<0> : tensor<1xsi32>
    // CHECK:       [[VAL0:%.*]] = IE.ReduceMax(%arg0, %cst) : tensor<10xf32>, tensor<1xsi32> -> tensor<1xf32>
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: @ReduceProd
func.func @ReduceProd(%arg0: tensor<10xf32>) -> tensor<1xf32> {
    %AXIS = const.Declare tensor<si32> = dense<0> : tensor<si32>
    %REDUCE = IE.ReduceProd(%arg0, %AXIS) : tensor<10xf32>, tensor<si32> -> tensor<1xf32>
    return %REDUCE : tensor<1xf32>

    // CHECK:   [[AXIS:%.*]] = const.Declare tensor<1xsi32> =
    // CHECK-SAME:      dense<0> : tensor<si32>, [#const.Reshape<[1]>]

    // CHECK:   [[REDUCE:%.*]] = IE.ReduceProd(%arg0, [[AXIS]]) :
    // CHECK-SAME:      tensor<10xf32>, tensor<1xsi32> -> tensor<1xf32>

    // CHECK:   return [[REDUCE]] : tensor<1xf32>
}

// -----

// CHECK-LABEL: @ReduceMaxScalar
func.func @ReduceMaxScalar(%arg0: tensor<10xf32>) -> tensor<1xf32> {
    %AXIS = const.Declare tensor<si32> = dense<0> : tensor<si32>
    %REDUCE = IE.ReduceMax(%arg0, %AXIS) : tensor<10xf32>, tensor<si32> -> tensor<1xf32>
    return %REDUCE : tensor<1xf32>

    // CHECK:   [[AXIS:%.*]] = const.Declare tensor<1xsi32> =
    // CHECK-SAME:      dense<0> : tensor<si32>, [#const.Reshape<[1]>]

    // CHECK:   [[REDUCE:%.*]] = IE.ReduceMax(%arg0, [[AXIS]]) :
    // CHECK-SAME:      tensor<10xf32>, tensor<1xsi32> -> tensor<1xf32>

    // CHECK:   return [[REDUCE]] : tensor<1xf32>
}

// -----

// CHECK-LABEL: @ReduceMean
func.func @ReduceMean(%arg0: tensor<10xf32>) -> tensor<1xf32> {
    %AXIS = const.Declare tensor<si32> = dense<0> : tensor<si32>
    %REDUCE = IE.ReduceMean(%arg0, %AXIS) : tensor<10xf32>, tensor<si32> -> tensor<1xf32>
    return %REDUCE : tensor<1xf32>

    // CHECK:   [[AXIS:%.*]] = const.Declare tensor<1xsi32> =
    // CHECK-SAME:      dense<0> : tensor<si32>, [#const.Reshape<[1]>]

    // CHECK:   [[REDUCE:%.*]] = IE.ReduceMean(%arg0, [[AXIS]]) :
    // CHECK-SAME:      tensor<10xf32>, tensor<1xsi32> -> tensor<1xf32>

    // CHECK:   return [[REDUCE]] : tensor<1xf32>
}

// -----

// CHECK-LABEL: @ReduceLogicalOr
func.func @ReduceLogicalOr(%arg0: tensor<10xf32>) -> tensor<1xf32> {
    %AXIS = const.Declare tensor<si32> = dense<0> : tensor<si32>
    %REDUCE = IE.ReduceLogicalOr(%arg0, %AXIS) : tensor<10xf32>, tensor<si32> -> tensor<1xf32>
    return %REDUCE : tensor<1xf32>

    // CHECK:   [[AXIS:%.*]] = const.Declare tensor<1xsi32> =
    // CHECK-SAME:      dense<0> : tensor<si32>, [#const.Reshape<[1]>]

    // CHECK:   [[REDUCE:%.*]] = IE.ReduceLogicalOr(%arg0, [[AXIS]]) :
    // CHECK-SAME:      tensor<10xf32>, tensor<1xsi32> -> tensor<1xf32>

    // CHECK:   return [[REDUCE]] : tensor<1xf32>
}

// -----

// CHECK-LABEL: @ReduceLogicalAnd
func.func @ReduceLogicalAnd(%arg0: tensor<10xf32>) -> tensor<1xf32> {
    %AXIS = const.Declare tensor<si32> = dense<0> : tensor<si32>
    %REDUCE = IE.ReduceLogicalAnd(%arg0, %AXIS) : tensor<10xf32>, tensor<si32> -> tensor<1xf32>
    return %REDUCE : tensor<1xf32>

    // CHECK:   [[AXIS:%.*]] = const.Declare tensor<1xsi32> =
    // CHECK-SAME:      dense<0> : tensor<si32>, [#const.Reshape<[1]>]

    // CHECK:   [[REDUCE:%.*]] = IE.ReduceLogicalAnd(%arg0, [[AXIS]]) :
    // CHECK-SAME:      tensor<10xf32>, tensor<1xsi32> -> tensor<1xf32>

    // CHECK:   return [[REDUCE]] : tensor<1xf32>
}

// -----

// CHECK-LABEL: @ReduceSum
func.func @ReduceSum(%arg0: tensor<10xf32>) -> tensor<1xf32> {
    %AXIS = const.Declare tensor<si32> = dense<0> : tensor<si32>
    %REDUCE = IE.ReduceSum(%arg0, %AXIS) : tensor<10xf32>, tensor<si32> -> tensor<1xf32>
    return %REDUCE : tensor<1xf32>

    // CHECK:   [[AXIS:%.*]] = const.Declare tensor<1xsi32> =
    // CHECK-SAME:      dense<0> : tensor<si32>, [#const.Reshape<[1]>]

    // CHECK:   [[REDUCE:%.*]] = IE.ReduceSum(%arg0, [[AXIS]]) :
    // CHECK-SAME:      tensor<10xf32>, tensor<1xsi32> -> tensor<1xf32>

    // CHECK:   return [[REDUCE]] : tensor<1xf32>
}

// -----

// CHECK-LABEL: @ReduceMin
func.func @ReduceMin(%arg0: tensor<10xf32>) -> tensor<1xf32> {
    %AXIS = const.Declare tensor<si32> = dense<0> : tensor<si32>
    %REDUCE = IE.ReduceMin(%arg0, %AXIS) : tensor<10xf32>, tensor<si32> -> tensor<1xf32>
    return %REDUCE : tensor<1xf32>

    // CHECK:   [[AXIS:%.*]] = const.Declare tensor<1xsi32> =
    // CHECK-SAME:      dense<0> : tensor<si32>, [#const.Reshape<[1]>]

    // CHECK:   [[REDUCE:%.*]] = IE.ReduceMin(%arg0, [[AXIS]]) :
    // CHECK-SAME:      tensor<10xf32>, tensor<1xsi32> -> tensor<1xf32>

    // CHECK:   return [[REDUCE]] : tensor<1xf32>
}

// -----

// CHECK-LABEL: @ReduceL1
func.func @ReduceL1(%arg0: tensor<10xf32>) -> tensor<1xf32> {
    %AXIS = const.Declare tensor<si32> = dense<0> : tensor<si32>
    %REDUCE = IE.ReduceL1(%arg0, %AXIS) : tensor<10xf32>, tensor<si32> -> tensor<1xf32>
    return %REDUCE : tensor<1xf32>

    // CHECK:   [[AXIS:%.*]] = const.Declare tensor<1xsi32> =
    // CHECK-SAME:      dense<0> : tensor<si32>, [#const.Reshape<[1]>]

    // CHECK:   [[REDUCE:%.*]] = IE.ReduceL1(%arg0, [[AXIS]]) :
    // CHECK-SAME:      tensor<10xf32>, tensor<1xsi32> -> tensor<1xf32>

    // CHECK:   return [[REDUCE]] : tensor<1xf32>
}

// -----

// CHECK-LABEL: @ReduceL2
func.func @ReduceL2(%arg0: tensor<10xf32>) -> tensor<1xf32> {
    %AXIS = const.Declare tensor<si32> = dense<0> : tensor<si32>
    %REDUCE = IE.ReduceL2(%arg0, %AXIS) : tensor<10xf32>, tensor<si32> -> tensor<1xf32>
    return %REDUCE : tensor<1xf32>

    // CHECK:   [[AXIS:%.*]] = const.Declare tensor<1xsi32> =
    // CHECK-SAME:      dense<0> : tensor<si32>, [#const.Reshape<[1]>]

    // CHECK:   [[REDUCE:%.*]] = IE.ReduceL2(%arg0, [[AXIS]]) :
    // CHECK-SAME:      tensor<10xf32>, tensor<1xsi32> -> tensor<1xf32>

    // CHECK:   return [[REDUCE]] : tensor<1xf32>
}
