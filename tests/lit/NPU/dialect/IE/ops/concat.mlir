//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
!qElemType = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1}>

// CHECK-LABEL: @ConcatLargeOffsetStride
func.func @ConcatLargeOffsetStride(%arg0: tensor<1x2x3x4x!qElemType>, %arg1: tensor<1x2x3x4x!qElemType>) -> tensor<2x2x3x4x!qElemType> {
    %0 = IE.Concat(%arg0, %arg1) {per_axis = #IE.Concat<axis = 0, offset = 1, stride = 2>} : tensor<1x2x3x4x!qElemType>, tensor<1x2x3x4x!qElemType> -> tensor<2x2x3x4x!qElemType>
    return %0 : tensor<2x2x3x4x!qElemType>

    // The operation should be parsed and verified successfully
    // CHECK: IE.Concat
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0000000000000000E-1>

// CHECK-LABEL: @PerTensorQuant
func.func @PerTensorQuant(%arg0: tensor<1x2x3x4x!qElemType>, %arg1: tensor<1x2x3x4x!qElemType>) -> tensor<1x4x3x4x!qElemType> {
    %0 = IE.Concat(%arg0, %arg1) {per_axis = #IE.Concat<axis = 1>} : tensor<1x2x3x4x!qElemType>, tensor<1x2x3x4x!qElemType> -> tensor<1x4x3x4x!qElemType>
    return %0 : tensor<1x4x3x4x!qElemType>

    // The operation should be parsed and verified successfully
    // CHECK: IE.Concat
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1}>

// CHECK-LABEL: @PerAxisQuantOtherAxis
func.func @PerAxisQuantOtherAxis(%arg0: tensor<1x2x3x4x!qElemType>, %arg1: tensor<1x2x3x4x!qElemType>) -> tensor<1x2x6x4x!qElemType> {
    %0 = IE.Concat(%arg0, %arg1) {per_axis = #IE.Concat<axis = 2>} : tensor<1x2x3x4x!qElemType>, tensor<1x2x3x4x!qElemType> -> tensor<1x2x6x4x!qElemType>
    return %0 : tensor<1x2x6x4x!qElemType>

    // The operation should be parsed and verified successfully
    // CHECK: IE.Concat
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1}>

// CHECK-LABEL: @PerAxisQuantOtherAxisOffsets
func.func @PerAxisQuantOtherAxisOffsets(%arg0: tensor<1x2x3x4x!qElemType>, %arg1: tensor<1x2x3x4x!qElemType>) -> tensor<1x2x6x4x!qElemType> {
    %0 = IE.Concat(%arg0, %arg1) {
        static_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]
    } : tensor<1x2x3x4x!qElemType>, tensor<1x2x3x4x!qElemType> -> tensor<1x2x6x4x!qElemType>
    return %0 : tensor<1x2x6x4x!qElemType>

    // The operation should be parsed and verified successfully
    // CHECK: IE.Concat
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1}>
!qElemType1 = !quant.uniform<u8:f16:1, {3.0000000000000000E-1, 4.0000000000000000E-1}>
!qElemType2 = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1, 3.0000000000000000E-1, 4.0000000000000000E-1}>

// CHECK-LABEL: @PerAxisQuantSameAxis
func.func @PerAxisQuantSameAxis(%arg0: tensor<1x2x3x4x!qElemType>, %arg1: tensor<1x2x3x4x!qElemType1>) -> tensor<1x4x3x4x!qElemType2> {
    %0 = IE.Concat(%arg0, %arg1) {per_axis = #IE.Concat<axis = 1>} : tensor<1x2x3x4x!qElemType>, tensor<1x2x3x4x!qElemType1> -> tensor<1x4x3x4x!qElemType2>
    return %0 : tensor<1x4x3x4x!qElemType2>

    // The operation should be parsed and verified successfully
    // CHECK: IE.Concat
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1}>
!qElemType1 = !quant.uniform<u8:f16:1, {3.0000000000000000E-1, 4.0000000000000000E-1}>
!qElemType2 = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1, 3.0000000000000000E-1, 4.0000000000000000E-1}>

// CHECK-LABEL: @PerAxisQuantSameAxisOffsets
func.func @PerAxisQuantSameAxisOffsets(%arg0: tensor<1x2x3x4x!qElemType>, %arg1: tensor<1x2x3x4x!qElemType1>) -> tensor<1x4x3x4x!qElemType2> {
    %0 = IE.Concat(%arg0, %arg1) {
        static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0]]
    } : tensor<1x2x3x4x!qElemType>, tensor<1x2x3x4x!qElemType1> -> tensor<1x4x3x4x!qElemType2>
    return %0 : tensor<1x4x3x4x!qElemType2>

    // The operation should be parsed and verified successfully
    // CHECK: IE.Concat
}

// -----

// CHECK-LABEL: @ConvertPerAxisToOffsets
func.func @ConvertPerAxisToOffsets(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>) -> tensor<1x4x3x4xf32> {
    %0 = IE.Concat(%arg0, %arg1) {per_axis = #IE.Concat<axis = 1>} : tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x4x3x4xf32>
    return %0: tensor<1x4x3x4xf32>

    // CHECK:     [[VAL_0:%.*]] = IE.Concat(%arg0, %arg1)
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0]]}
    // CHECK-SAME:     tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x4x3x4xf32>
    // CHECK:     return [[VAL_0]] : tensor<1x4x3x4xf32>
}

// -----

// CHECK-LABEL: @FuseConcatWithOffsetsAndOtherOp
func.func @FuseConcatWithOffsetsAndOtherOp(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>,
                                      %arg2: tensor<1x2x3x4xf32>, %arg3: tensor<1x2x3x4xf32>,
                                      %arg4: tensor<1x2x4x3xf32>) -> tensor<1x10x3x4xf32> {
    %0 = IE.Concat(%arg0, %arg1) {
        static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0]]
    } : tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x4x3x4xf32>
    %1 = IE.Concat(%arg2, %arg3) {per_axis = #IE.Concat<axis = 1>} : tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x4x3x4xf32>
    %2 = IE.Reshape(%arg4) { shape_value = [1, 2, 3, 4] } : tensor<1x2x4x3xf32> -> tensor<1x2x3x4xf32>
    %3 = IE.Concat(%0, %1, %2) {
        static_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0]]
    } : tensor<1x4x3x4xf32>, tensor<1x4x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x10x3x4xf32>
    return %3: tensor<1x10x3x4xf32>

    // CHECK-DAG:     [[RES_0:%.*]] = IE.Reshape(%arg4) {shape_value = [1, 2, 3, 4]} : tensor<1x2x4x3xf32> -> tensor<1x2x3x4xf32>
    // CHECK:     [[VAL_0:%.*]] = IE.Concat(%arg0, %arg1, %arg2, %arg3, [[RES_0]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0], [0, 8, 0, 0]]}
    // CHECK-SAME:     tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x10x3x4xf32>
    // CHECK:     return [[VAL_0]] : tensor<1x10x3x4xf32>
}

// -----

// CHECK-LABEL: @FuseConcatWithPerAxis
func.func @FuseConcatWithPerAxis(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>,
                            %arg2: tensor<1x2x3x4xf32>, %arg3: tensor<1x2x3x4xf32>,
                            %arg4: tensor<1x2x4x3xf32>) -> tensor<1x10x3x4xf32> {
    %0 = IE.Concat(%arg0, %arg1) {
        static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0]]
    } : tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x4x3x4xf32>
    %1 = IE.Concat(%arg2, %arg3) {per_axis = #IE.Concat<axis = 1>} : tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x4x3x4xf32>
    %2 = IE.Reshape(%arg4) { shape_value = [1, 2, 3, 4] } : tensor<1x2x4x3xf32> -> tensor<1x2x3x4xf32>
    %3 = IE.Concat(%0, %1, %2) {per_axis = #IE.Concat<axis = 1>} : tensor<1x4x3x4xf32>, tensor<1x4x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x10x3x4xf32>
    return %3 : tensor<1x10x3x4xf32>

    // CHECK-DAG:     [[RES_0:%.*]] = IE.Reshape(%arg4) {shape_value = [1, 2, 3, 4]} : tensor<1x2x4x3xf32> -> tensor<1x2x3x4xf32>
    // CHECK:     [[VAL_0:%.*]] = IE.Concat(%arg0, %arg1, %arg2, %arg3, [[RES_0]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0], [0, 8, 0, 0]]}
    // CHECK-SAME:     tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x10x3x4xf32>
    // CHECK:     return [[VAL_0]] : tensor<1x10x3x4xf32>
}

// -----

// CHECK-LABEL: @OneInputFold
func.func @OneInputFold(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = IE.Concat(%arg0) {per_axis = #IE.Concat<axis = 1>} : tensor<4x4xf32> -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>

    // CHECK-NOT: IE.Concat
    // CHECK:     return %arg0
}

// -----

// CHECK-LABEL: @ConstInputsFold
func.func @ConstInputsFoldforNCHW() -> tensor<1x6x8x1xf16> {
    %cst_0 = const.Declare tensor<1x3x8x1xf16> = dense<[1.0, 2.0, 3.0]> : tensor<3xf16>, [#const.Reshape<[1, 3, 1, 1]>, #const.Broadcast<2 : i64, 8 : i64>]
    %cst_1 = const.Declare tensor<1x3x8x1xf16> = dense<[4.0, 5.0, 6.0]> : tensor<3xf16>, [#const.Reshape<[1, 3, 1, 1]>, #const.Broadcast<2 : i64, 8 : i64>]
    %0 = IE.Concat(%cst_0, %cst_1) {static_offsets = [[0, 0, 0, 0], [0, 3, 0, 0]]} : tensor<1x3x8x1xf16>, tensor<1x3x8x1xf16> -> tensor<1x6x8x1xf16>
    return %0 : tensor<1x6x8x1xf16>

    // CHECK: [[cst:%.+]] = const.Declare tensor<1x6x8x1xf16> = dense<
    // CHECK-SAME{LITERAL}:     [[[[1.000000e+00], [1.000000e+00], [1.000000e+00], [1.000000e+00], [1.000000e+00], [1.000000e+00], [1.000000e+00], [1.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[2.000000e+00], [2.000000e+00], [2.000000e+00], [2.000000e+00], [2.000000e+00], [2.000000e+00], [2.000000e+00], [2.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[3.000000e+00], [3.000000e+00], [3.000000e+00], [3.000000e+00], [3.000000e+00], [3.000000e+00], [3.000000e+00], [3.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[4.000000e+00], [4.000000e+00], [4.000000e+00], [4.000000e+00], [4.000000e+00], [4.000000e+00], [4.000000e+00], [4.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[5.000000e+00], [5.000000e+00], [5.000000e+00], [5.000000e+00], [5.000000e+00], [5.000000e+00], [5.000000e+00], [5.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[6.000000e+00], [6.000000e+00], [6.000000e+00], [6.000000e+00], [6.000000e+00], [6.000000e+00], [6.000000e+00], [6.000000e+00]]]]>
    // CHECK-SAME:                tensor<1x6x8x1xf16>
    // CHECK-NOT: IE.Concat
    // CHECK:     return [[cst]]
}

// -----

// CHECK-LABEL: @ConstInputsFoldWithDifferentDimValueForNCHW
func.func @ConstInputsFoldWithDifferentDimValueForNCHW() -> tensor<1x5x8x1xf16> {
    %cst_0 = const.Declare tensor<1x3x8x1xf16> = dense<[1.0, 2.0, 3.0]> : tensor<3xf16>, [#const.Reshape<[1, 3, 1, 1]>, #const.Broadcast<2 : i64, 8 : i64>]
    %cst_1 = const.Declare tensor<1x2x8x1xf16> = dense<[4.0, 5.0]> : tensor<2xf16>, [#const.Reshape<[1, 2, 1, 1]>, #const.Broadcast<2 : i64, 8 : i64>]
    %0 = IE.Concat(%cst_0, %cst_1) {static_offsets = [[0, 0, 0, 0], [0, 3, 0, 0]]} : tensor<1x3x8x1xf16>, tensor<1x2x8x1xf16> -> tensor<1x5x8x1xf16>
    return %0 : tensor<1x5x8x1xf16>

    // CHECK: [[cst:%.+]] = const.Declare tensor<1x5x8x1xf16> = dense<
    // CHECK-SAME{LITERAL}:     [[[[1.000000e+00], [1.000000e+00], [1.000000e+00], [1.000000e+00], [1.000000e+00], [1.000000e+00], [1.000000e+00], [1.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[2.000000e+00], [2.000000e+00], [2.000000e+00], [2.000000e+00], [2.000000e+00], [2.000000e+00], [2.000000e+00], [2.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[3.000000e+00], [3.000000e+00], [3.000000e+00], [3.000000e+00], [3.000000e+00], [3.000000e+00], [3.000000e+00], [3.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[4.000000e+00], [4.000000e+00], [4.000000e+00], [4.000000e+00], [4.000000e+00], [4.000000e+00], [4.000000e+00], [4.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[5.000000e+00], [5.000000e+00], [5.000000e+00], [5.000000e+00], [5.000000e+00], [5.000000e+00], [5.000000e+00], [5.000000e+00]]]]>
    // CHECK-SAME:                tensor<1x5x8x1xf16>
    // CHECK-NOT: IE.Concat
    // CHECK:     return [[cst]]
}

// -----

// CHECK-LABEL: @ConstInputsFoldAxisNotMostOuterForNCHW
func.func @ConstInputsFoldAxisNotMostOuterForNCHW() -> tensor<1x4x6x2xf16> {
    %cst_0 = const.Declare tensor<1x4x3x2xf16> = dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]> : tensor<6xf16>, [#const.Reshape<[1, 1, 3, 2]>, #const.Broadcast<1 : i64, 4 : i64>]
    %cst_1 = const.Declare tensor<1x4x3x2xf16> = dense<[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]> : tensor<6xf16>, [#const.Reshape<[1, 1, 3, 2]>, #const.Broadcast<1 : i64, 4 : i64>]
    %0 = IE.Concat(%cst_0, %cst_1) {static_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]} : tensor<1x4x3x2xf16>, tensor<1x4x3x2xf16> -> tensor<1x4x6x2xf16>
    return %0 : tensor<1x4x6x2xf16>

    // CHECK: [[cst:%.+]] = const.Declare tensor<1x4x6x2xf16> = dense<
    // CHECK-SAME{LITERAL}:      [[[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00], [9.000000e+00, 1.000000e+01], [1.100000e+01, 1.200000e+01]],
    // CHECK-SAME{LITERAL}:        [[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00], [9.000000e+00, 1.000000e+01], [1.100000e+01, 1.200000e+01]],
    // CHECK-SAME{LITERAL}:        [[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00], [9.000000e+00, 1.000000e+01], [1.100000e+01, 1.200000e+01]],
    // CHECK-SAME{LITERAL}:        [[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00], [9.000000e+00, 1.000000e+01], [1.100000e+01, 1.200000e+01]]]]>
    // CHECK-SAME{LITERAL}:        tensor<1x4x6x2xf16>
    // CHECK-NOT: IE.Concat
    // CHECK:     return [[cst]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @ConstInputsFoldForNHWC
func.func @ConstInputsFoldForNHWC() -> tensor<1x6x8x1xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<1x3x8x1xf16, {order = #NHWC}> = dense<[1.0, 2.0, 3.0]> : tensor<3xf16>, [#const.Reshape<[1, 3, 1, 1]>, #const.Broadcast<2 : i64, 8 : i64>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<1x3x8x1xf16, {order = #NHWC}> = dense<[4.0, 5.0, 6.0]> : tensor<3xf16>, [#const.Reshape<[1, 3, 1, 1]>, #const.Broadcast<2 : i64, 8 : i64>, #const.Reorder<#NHWC>]
    %0 = IE.Concat(%cst_0, %cst_1) {static_offsets = [[0, 0, 0, 0], [0, 3, 0, 0]]} : tensor<1x3x8x1xf16, {order = #NHWC}>, tensor<1x3x8x1xf16, {order = #NHWC}> -> tensor<1x6x8x1xf16, {order = #NHWC}>
    return %0 : tensor<1x6x8x1xf16, {order = #NHWC}>

    // CHECK: [[cst:%.+]] = const.Declare tensor<1x6x8x1xf16, {order = #NHWC}> = dense<
    // CHECK-SAME{LITERAL}:     [[[[1.000000e+00], [2.000000e+00], [3.000000e+00], [4.000000e+00], [5.000000e+00], [6.000000e+00], [1.000000e+00], [2.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[3.000000e+00], [4.000000e+00], [5.000000e+00], [6.000000e+00], [1.000000e+00], [2.000000e+00], [3.000000e+00], [4.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[5.000000e+00], [6.000000e+00], [1.000000e+00], [2.000000e+00], [3.000000e+00], [4.000000e+00], [5.000000e+00], [6.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[1.000000e+00], [2.000000e+00], [3.000000e+00], [4.000000e+00], [5.000000e+00], [6.000000e+00], [1.000000e+00], [2.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[3.000000e+00], [4.000000e+00], [5.000000e+00], [6.000000e+00], [1.000000e+00], [2.000000e+00], [3.000000e+00], [4.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[5.000000e+00], [6.000000e+00], [1.000000e+00], [2.000000e+00], [3.000000e+00], [4.000000e+00], [5.000000e+00], [6.000000e+00]]]]>
    // CHECK-SAME{LITERAL}:       tensor<1x6x8x1xf16, {order = #NHWC}>
    // CHECK-NOT: IE.Concat
    // CHECK:     return [[cst]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @ConstInputsFoldWithDifferentDimValueForNHWC
func.func @ConstInputsFoldWithDifferentDimValueForNHWC() -> tensor<1x5x8x1xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<1x3x8x1xf16, {order = #NHWC}> = dense<[1.0, 2.0, 3.0]> : tensor<3xf16>, [#const.Reshape<[1, 3, 1, 1]>, #const.Broadcast<2 : i64, 8 : i64>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<1x2x8x1xf16, {order = #NHWC}> = dense<[4.0, 5.0]> : tensor<2xf16>, [#const.Reshape<[1, 2, 1, 1]>, #const.Broadcast<2 : i64, 8 : i64>, #const.Reorder<#NHWC>]
    %0 = IE.Concat(%cst_0, %cst_1) {static_offsets = [[0, 0, 0, 0], [0, 3, 0, 0]]} : tensor<1x3x8x1xf16, {order = #NHWC}>, tensor<1x2x8x1xf16, {order = #NHWC}> -> tensor<1x5x8x1xf16, {order = #NHWC}>
    return %0 : tensor<1x5x8x1xf16, {order = #NHWC}>

    // CHECK: [[cst:%.+]] = const.Declare tensor<1x5x8x1xf16, {order = #NHWC}> = dense<
    // CHECK-SAME{LITERAL}:     [[[[1.000000e+00], [2.000000e+00], [3.000000e+00], [4.000000e+00], [5.000000e+00], [1.000000e+00], [2.000000e+00], [3.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[4.000000e+00], [5.000000e+00], [1.000000e+00], [2.000000e+00], [3.000000e+00], [4.000000e+00], [5.000000e+00], [1.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[2.000000e+00], [3.000000e+00], [4.000000e+00], [5.000000e+00], [1.000000e+00], [2.000000e+00], [3.000000e+00], [4.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[5.000000e+00], [1.000000e+00], [2.000000e+00], [3.000000e+00], [4.000000e+00], [5.000000e+00], [1.000000e+00], [2.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[3.000000e+00], [4.000000e+00], [5.000000e+00], [1.000000e+00], [2.000000e+00], [3.000000e+00], [4.000000e+00], [5.000000e+00]]]]>
    // CHECK-SAME:                tensor<1x5x8x1xf16, {order = #NHWC}>
    // CHECK-NOT: IE.Concat
    // CHECK:     return [[cst]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @ConstInputsFoldAxisNotMostOuterForNHWC
func.func @ConstInputsFoldAxisNotMostOuterForNHWC() -> tensor<1x4x6x2xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<1x4x3x2xf16, {order = #NHWC}> = dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]> : tensor<6xf16>, [#const.Reshape<[1, 1, 3, 2]>, #const.Broadcast<1 : i64, 4 : i64>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<1x4x3x2xf16, {order = #NHWC}> = dense<[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]> : tensor<6xf16>, [#const.Reshape<[1, 1, 3, 2]>, #const.Broadcast<1 : i64, 4 : i64>, #const.Reorder<#NHWC>]
    %0 = IE.Concat(%cst_0, %cst_1) {static_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]} : tensor<1x4x3x2xf16, {order = #NHWC}>, tensor<1x4x3x2xf16, {order = #NHWC}> -> tensor<1x4x6x2xf16, {order = #NHWC}>
    return %0 : tensor<1x4x6x2xf16, {order = #NHWC}>

    // CHECK: [[cst:%.+]] = const.Declare tensor<1x4x6x2xf16, {order = #NHWC}> = dense<
    // CHECK-SAME{LITERAL}:      [[[[1.000000e+00, 1.000000e+00], [1.000000e+00, 1.000000e+00], [2.000000e+00, 2.000000e+00], [2.000000e+00, 2.000000e+00], [3.000000e+00, 3.000000e+00], [3.000000e+00, 3.000000e+00]],
    // CHECK-SAME{LITERAL}:        [[4.000000e+00, 4.000000e+00], [4.000000e+00, 4.000000e+00], [5.000000e+00, 5.000000e+00], [5.000000e+00, 5.000000e+00], [6.000000e+00, 6.000000e+00], [6.000000e+00, 6.000000e+00]],
    // CHECK-SAME{LITERAL}:        [[7.000000e+00, 7.000000e+00], [7.000000e+00, 7.000000e+00], [8.000000e+00, 8.000000e+00], [8.000000e+00, 8.000000e+00], [9.000000e+00, 9.000000e+00], [9.000000e+00, 9.000000e+00]],
    // CHECK-SAME{LITERAL}:        [[1.000000e+01, 1.000000e+01], [1.000000e+01, 1.000000e+01], [1.100000e+01, 1.100000e+01], [1.100000e+01, 1.100000e+01], [1.200000e+01, 1.200000e+01], [1.200000e+01, 1.200000e+01]]]]>
    // CHECK-SAME{LITERAL}:        tensor<1x4x6x2xf16, {order = #NHWC}>
    // CHECK-NOT: IE.Concat
    // CHECK:     return [[cst]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: ConstInputsNotFoldForAxisSizeNot1
func.func @ConstInputsNotFoldForAxisSizeNot1() -> tensor<1x6x16x1xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<1x3x8x1xf16, {order = #NHWC}> = dense<[1.0, 2.0, 3.0]> : tensor<3xf16>, [#const.Reshape<[1, 3, 1, 1]>, #const.Broadcast<2 : i64, 8 : i64>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<1x3x8x1xf16, {order = #NHWC}> = dense<[4.0, 5.0, 6.0]> : tensor<3xf16>, [#const.Reshape<[1, 3, 1, 1]>, #const.Broadcast<2 : i64, 8 : i64>, #const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<1x6x8x1xf16, {order = #NHWC}> = dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]> : tensor<6xf16>, [#const.Reshape<[1, 6, 1, 1]>, #const.Broadcast<2 : i64, 8 : i64>, #const.Reorder<#NHWC>]
    %0 = IE.Concat(%cst_0, %cst_1, %cst_2) {static_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 0, 8, 0]]} : tensor<1x3x8x1xf16, {order = #NHWC}>, tensor<1x3x8x1xf16, {order = #NHWC}>, tensor<1x6x8x1xf16, {order = #NHWC}> -> tensor<1x6x16x1xf16, {order = #NHWC}>
    return %0 : tensor<1x6x16x1xf16, {order = #NHWC}>

    // CHECK: IE.Concat
}

// -----

// CHECK-LABEL: @NonConstInputsNotFold
func.func @NonConstInputsNotFold(%arg0 : tensor<1x3x8x1xf16>) -> tensor<1x6x8x1xf16> {
    %cst_0 = const.Declare tensor<1x3x8x1xf16> = dense<[1.0, 2.0, 3.0]> : tensor<3xf16>, [#const.Reshape<[1, 3, 1, 1]>, #const.Broadcast<2 : i64, 8 : i64>]
    %0 = IE.Concat(%cst_0, %arg0) {static_offsets = [[0, 0, 0, 0], [0, 3, 0, 0]]} : tensor<1x3x8x1xf16>, tensor<1x3x8x1xf16> -> tensor<1x6x8x1xf16>
    return %0 : tensor<1x6x8x1xf16>

    // CHECK: IE.Concat
}

// -----

// CHECK-LABEL: @ConstInputFoldWithDifferentInputAndOutputType
func.func @ConstInputFoldWithDifferentInputAndOutputType() -> tensor<1x6x8x1xf16> {
    %cst_0 = const.Declare tensor<1x3x8x1xf16> = dense<[1.0, 2.0, 3.0]> : tensor<3xf32>, [#const.Reshape<[1, 3, 1, 1]>, #const.Broadcast<2 : i64, 8 : i64>, #const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x3x8x1xf16> = dense<[4.0, 5.0, 6.0]> : tensor<3xf32>, [#const.Reshape<[1, 3, 1, 1]>, #const.Broadcast<2 : i64, 8 : i64>, #const.CastElemType<f16>]
    %0 = IE.Concat(%cst_0, %cst_1) {static_offsets = [[0, 0, 0, 0], [0, 3, 0, 0]]} : tensor<1x3x8x1xf16>, tensor<1x3x8x1xf16> -> tensor<1x6x8x1xf16>
    return %0 : tensor<1x6x8x1xf16>

    // CHECK: [[cst:%.+]] = const.Declare tensor<1x6x8x1xf16> = dense<
    // CHECK-SAME{LITERAL}:     [[[[1.000000e+00], [1.000000e+00], [1.000000e+00], [1.000000e+00], [1.000000e+00], [1.000000e+00], [1.000000e+00], [1.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[2.000000e+00], [2.000000e+00], [2.000000e+00], [2.000000e+00], [2.000000e+00], [2.000000e+00], [2.000000e+00], [2.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[3.000000e+00], [3.000000e+00], [3.000000e+00], [3.000000e+00], [3.000000e+00], [3.000000e+00], [3.000000e+00], [3.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[4.000000e+00], [4.000000e+00], [4.000000e+00], [4.000000e+00], [4.000000e+00], [4.000000e+00], [4.000000e+00], [4.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[5.000000e+00], [5.000000e+00], [5.000000e+00], [5.000000e+00], [5.000000e+00], [5.000000e+00], [5.000000e+00], [5.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[6.000000e+00], [6.000000e+00], [6.000000e+00], [6.000000e+00], [6.000000e+00], [6.000000e+00], [6.000000e+00], [6.000000e+00]]]]>
    // CHECK-SAME{LITERAL}:       tensor<1x6x8x1xf16>
    // CHECK-NOT: IE.Concat
    // CHECK:     return [[cst]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8<0:254>:f16, 0.0078740157480314959:127>
// CHECK-LABEL: @ConcatWithConstInputsFoldForQuantize
func.func @ConcatWithConstInputsFoldForQuantize() -> tensor<16x12x2x1x!qElemType, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<16x6x2x1x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<16x3x2x1xf16>, [#const.Add<1.270000e+02 : f64>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>, #const.Reshape<[16, 3, 2, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 3, 0, 0]>]
    %cst_1 = const.Declare tensor<16x6x2x1x!qElemType, {order = #NHWC}> = dense<-1.0> : tensor<16x3x2x1xf16>, [#const.Add<1.270000e+02 : f64>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>, #const.Reshape<[16, 3, 2, 1]>, #const.PadWithZero<[0, 3, 0, 0], [0, 0, 0, 0]>]

    %0 = IE.Concat(%cst_0, %cst_1) {per_axis = #IE.Concat<axis = 1>} : tensor<16x6x2x1x!qElemType, {order = #NHWC}>, tensor<16x6x2x1x!qElemType, {order = #NHWC}> -> tensor<16x12x2x1x!qElemType, {order = #NHWC}>
    return %0 : tensor<16x12x2x1x!qElemType, {order = #NHWC}>

    // CHECK-NOT: IE.Concat
    // CHECK: [[cst:%.+]] = const.Declare tensor<16x12x2x1x!qElemType, {order = #NHWC}> = dense<"
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E
    // CHECK-SAME                         8080807F7F7F7F7F7F7E7E7E"> : tensor<16x12x2x1xui8, {order = #NHWC}>, [#const.CastElemType<!qElemType>]
    // CHECK:     return [[cst]]
}

// -----

// CHECK-LABEL: @foldSliceConcat
func.func @foldSliceConcat(%arg0: tensor<1x128x96x64xf16>) -> tensor<1x128x96x64xf16> {
  %slice_0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
  %slice_1 = IE.Slice %arg0 [0, 64, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
  %ret = IE.Concat(%slice_0, %slice_1) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16> -> tensor<1x128x96x64xf16>
  return %ret : tensor<1x128x96x64xf16>

  // CHECK-NOT:     IE.Slice
  // CHECK-NOT:     IE.Slice
  // CHECK-NOT:     IE.Concat
  // CHECK:     return %arg0
}

// -----

// CHECK-LABEL: @foldSliceConcatWithMultInputs
func.func @foldSliceConcatWithMultInputs(%arg0: tensor<1x128x96x64xf16>, %arg1: tensor<1x64x96x64xf16>) -> tensor<1x192x96x64xf16> {
  %slice_0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
  %slice_1 = IE.Slice %arg0 [0, 64, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
  %ret = IE.Concat(%slice_0, %slice_1, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0]]} : tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16> -> tensor<1x192x96x64xf16>
  return %ret : tensor<1x192x96x64xf16>

  // CHECK:                 [[CONCAT_RET:%.+]] = IE.Concat(%arg0, %arg1)
  // CHECK-SAME{LITERAL}:       {static_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]} : tensor<1x128x96x64xf16>, tensor<1x64x96x64xf16> -> tensor<1x192x96x64xf16>
  // CHECK:                 return [[CONCAT_RET]]
}

// -----

// CHECK-LABEL: @foldSliceConcatWhenSliceHasDifferentParent
func.func @foldSliceConcatWhenSliceHasDifferentParent(%arg0: tensor<1x128x96x64xf16>, %arg1: tensor<1x128x96x64xf16>) -> tensor<1x256x96x64xf16> {
  %slice_0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
  %slice_1 = IE.Slice %arg0 [0, 64, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
  %slice_2 = IE.Slice %arg1 [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
  %slice_3 = IE.Slice %arg1 [0, 64, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
  %ret = IE.Concat(%slice_0, %slice_1, %slice_2, %slice_3) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 192, 0, 0]]} : tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16> -> tensor<1x256x96x64xf16>
  return %ret : tensor<1x256x96x64xf16>

  // CHECK:                 [[CONCAT_RET:%.+]] = IE.Concat(%arg0, %arg1)
  // CHECK-SAME{LITERAL}:       {static_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]} : tensor<1x128x96x64xf16>, tensor<1x128x96x64xf16> -> tensor<1x256x96x64xf16>
  // CHECK:                 return [[CONCAT_RET]]
}

// -----

// CHECK-LABEL: @foldSliceConcatWhenSliceHasDifferentParentAndBreaked
func.func @foldSliceConcatWhenSliceHasDifferentParentAndBreaked(%arg0: tensor<1x128x96x64xf16>, %arg1: tensor<1x64x96x64xf16>, %arg2: tensor<1x128x96x64xf16>) -> tensor<1x320x96x64xf16> {
  %slice_0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
  %slice_1 = IE.Slice %arg0 [0, 64, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
  %slice_2 = IE.Slice %arg2 [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
  %slice_3 = IE.Slice %arg2 [0, 64, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
  %ret = IE.Concat(%slice_0, %slice_1, %arg1, %slice_2, %slice_3) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 192, 0, 0], [0, 256, 0, 0]]} : tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16> -> tensor<1x320x96x64xf16>
  return %ret : tensor<1x320x96x64xf16>

  // CHECK:                 [[CONCAT_RET:%.+]] = IE.Concat(%arg0, %arg1, %arg2)
  // CHECK-SAME{LITERAL}:       {static_offsets = [[0, 0, 0, 0], [0, 128, 0, 0], [0, 192, 0, 0]]} : tensor<1x128x96x64xf16>, tensor<1x64x96x64xf16>, tensor<1x128x96x64xf16> -> tensor<1x320x96x64xf16>
  // CHECK:                 return [[CONCAT_RET]]
}

// -----

// CHECK-LABEL: @notFoldSliceConcatWhenSliceOverlapped
func.func @notFoldSliceConcatWhenSliceOverlapped(%arg0: tensor<1x128x96x64xf16>, %arg1: tensor<1x64x96x64xf16>) -> tensor<1x192x96x64xf16> {
  %slice_0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
  %slice_1 = IE.Slice %arg0 [0, 63, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
  %ret = IE.Concat(%slice_0, %slice_1, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0]]} : tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16> -> tensor<1x192x96x64xf16>
  return %ret : tensor<1x192x96x64xf16>

  // CHECK:     [[SLICE_0:%.+]] = IE.Slice
  // CHECK:     [[SLICE_1:%.+]] = IE.Slice
  // CHECK:     [[CONCAT_RET:%.+]] = IE.Concat
  // CHECK:     return [[CONCAT_RET]]
}

// -----

// CHECK-LABEL: @notFoldSliceConcatWhenSliceWithGap
func.func @notFoldSliceConcatWhenSliceWithGap(%arg0: tensor<1x129x96x64xf16>, %arg1: tensor<1x64x96x64xf16>) -> tensor<1x192x96x64xf16> {
  %slice_0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x129x96x64xf16> to tensor<1x64x96x64xf16>
  %slice_1 = IE.Slice %arg0 [0, 65, 0, 0] [1, 64, 96, 64] : tensor<1x129x96x64xf16> to tensor<1x64x96x64xf16>
  %ret = IE.Concat(%slice_0, %slice_1, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0]]} : tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16> -> tensor<1x192x96x64xf16>
  return %ret : tensor<1x192x96x64xf16>

  // CHECK:     [[SLICE_0:%.+]] = IE.Slice
  // CHECK:     [[SLICE_1:%.+]] = IE.Slice
  // CHECK:     [[CONCAT_RET:%.+]] = IE.Concat
  // CHECK:     return [[CONCAT_RET]]
}

// -----

// CHECK-LABEL: @notFoldSliceConcatWhenSliceInMultiAxes
func.func @notFoldSliceConcatWhenSliceInMultiAxes(%arg0: tensor<1x128x96x128xf16>) -> tensor<1x128x96x64xf16> {
  %slice_0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x128xf16> to tensor<1x64x96x64xf16>
  %slice_1 = IE.Slice %arg0 [0, 64, 0, 64] [1, 64, 96, 64] : tensor<1x128x96x128xf16> to tensor<1x64x96x64xf16>
  %ret = IE.Concat(%slice_0, %slice_1) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16> -> tensor<1x128x96x64xf16>
  return %ret : tensor<1x128x96x64xf16>

  // CHECK:     [[SLICE_0:%.+]] = IE.Slice
  // CHECK:     [[SLICE_1:%.+]] = IE.Slice
  // CHECK:     [[CONCAT_RET:%.+]] = IE.Concat
  // CHECK:     return [[CONCAT_RET]]
}

// -----

// CHECK-LABEL: @notFoldSliceConcatWhenNotRecoverParent
func.func @notFoldSliceConcatWhenNotRecoverParent(%arg0: tensor<1x128x96x64xf16>) -> tensor<1x128x96x64xf16> {
  %slice_0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
  %slice_1 = IE.Slice %arg0 [0, 64, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
  %ret = IE.Concat(%slice_1, %slice_0) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16> -> tensor<1x128x96x64xf16>
  return %ret : tensor<1x128x96x64xf16>

  // CHECK:     [[SLICE_0:%.+]] = IE.Slice
  // CHECK:     [[SLICE_1:%.+]] = IE.Slice
  // CHECK:     [[CONCAT_RET:%.+]] = IE.Concat
  // CHECK:     return [[CONCAT_RET]]
}

// CHECK-LABEL: @NotfoldSliceConcatWhenDifferentAxis
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x128x96x64xf16>
func.func @NotfoldSliceConcatWhenDifferentAxis(%arg0 : tensor<1x128x96x64xf16>) -> tensor<1x64x96x128xf16> {
    %slice_0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    %slice_1 = IE.Slice %arg0 [0, 64, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    %ret = IE.Concat(%slice_0, %slice_1) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 64]]} : tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16> -> tensor<1x64x96x128xf16>

    return %ret : tensor<1x64x96x128xf16>

    // CHECK:               [[SLICE_0:%.*]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    // CHECK:               [[SLICE_1:%.*]] = IE.Slice [[INPUT]] [0, 64, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    // CHECK:               [[CONCAT:%.*]] = IE.Concat([[SLICE_0]], [[SLICE_1]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 64]]} : tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16> -> tensor<1x64x96x128xf16>
    // CHECK:               return [[CONCAT]] : tensor<1x64x96x128xf16>
}

// CHECK-LABEL: @foldSliceConcatWhenDifferentAxis
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x128x96x64xf16>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x64x96x128xf16>
func.func @foldSliceConcatWhenDifferentAxis(%arg0 : tensor<1x128x96x64xf16>, %arg1 : tensor<1x64x96x128xf16>) -> tensor<1x64x96x256xf16> {
    %slice_0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    %slice_1 = IE.Slice %arg0 [0, 64, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    %slice_2 = IE.Slice %arg1 [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x64x96x128xf16> to tensor<1x64x96x64xf16>
    %slice_3 = IE.Slice %arg1 [0, 0, 0, 64] [1, 64, 96, 64] : tensor<1x64x96x128xf16> to tensor<1x64x96x64xf16>
    %ret = IE.Concat(%slice_0, %slice_1, %slice_2, %slice_3) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 64], [0, 0, 0, 128], [0, 0, 0, 192]]}
                :tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16> -> tensor<1x64x96x256xf16>

    return %ret : tensor<1x64x96x256xf16>

    // CHECK:               [[SLICE_0:%.*]] = IE.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    // CHECK:               [[SLICE_1:%.*]] = IE.Slice [[INPUT_0]] [0, 64, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    // CHECK:               [[CONCAT:%.*]] = IE.Concat([[SLICE_0]], [[SLICE_1]], [[INPUT_1]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 64], [0, 0, 0, 128]]} : tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>, tensor<1x64x96x128xf16> -> tensor<1x64x96x256xf16>
    // CHECK:               return [[CONCAT]] : tensor<1x64x96x256xf16>
}
