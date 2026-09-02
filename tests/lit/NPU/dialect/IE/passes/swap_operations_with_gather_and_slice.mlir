//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --swap-operations-with-gather-and-slice %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @MoveMultiplySubtractPostGather
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10xsi32>
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: tensor<10000x1xf16>
// CHECK-SAME:      [[INPUT_2:%arg[0-9]]]: tensor<10000x1xf16>
// CHECK-SAME:      [[INPUT_3:%arg[0-9]]]: tensor<10000x3584xui8>
func.func @MoveMultiplySubtractPostGather(%arg0: tensor<1x10xsi32>, %arg1: tensor<10000x1xf16>, %arg2: tensor<10000x1xf16>, %arg3: tensor<10000x3584xui8>) -> tensor<1x10x3584xf32> {
    %0 = IE.Convert(%arg3) {dstElemType = f16} : tensor<10000x3584xui8> -> tensor<10000x3584xf16>
    %1 = IE.Subtract(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<10000x3584xf16>, tensor<10000x1xf16> -> tensor<10000x3584xf16>
    %2 = IE.Multiply(%1, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<10000x3584xf16>, tensor<10000x1xf16> -> tensor<10000x3584xf16>
    %3 = IE.Convert(%2) {dstElemType = f32} : tensor<10000x3584xf16> -> tensor<10000x3584xf32>
    %4 = IE.Gather(%3, %arg0) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<10000x3584xf32>, tensor<1x10xsi32> -> tensor<1x10x3584xf32>
    return %4 : tensor<1x10x3584xf32>

    // CHECK:       [[GATHER_IN:%.+]] = IE.Gather([[INPUT_3]], [[INPUT_0]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<10000x3584xui8>, tensor<1x10xsi32> -> tensor<1x10x3584xui8>
    // CHECK:       [[CONVERT_IN:%.+]] = IE.Convert([[GATHER_IN]]) {dstElemType = f16} : tensor<1x10x3584xui8> -> tensor<1x10x3584xf16>
    // CHECK:       [[GATHER_SUB:%.+]]  = IE.Gather([[INPUT_1]], [[INPUT_0]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<10000x1xf16>, tensor<1x10xsi32> -> tensor<1x10x1xf16>
    // CHECK:       [[SUBTRACT:%.+]]  = IE.Subtract([[CONVERT_IN]], [[GATHER_SUB]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x3584xf16>, tensor<1x10x1xf16> -> tensor<1x10x3584xf16>
    // CHECK:       [[GATHER_MUL:%.+]]  = IE.Gather([[INPUT_2]], [[INPUT_0]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<10000x1xf16>, tensor<1x10xsi32> -> tensor<1x10x1xf16>
    // CHECK:       [[MULTIPLY:%.+]]  = IE.Multiply([[SUBTRACT]], [[GATHER_MUL]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x3584xf16>, tensor<1x10x1xf16> -> tensor<1x10x3584xf16>
    // CHECK:       [[CONVERT_OUT:%.+]]  = IE.Convert([[MULTIPLY]]) {dstElemType = f32} : tensor<1x10x3584xf16> -> tensor<1x10x3584xf32>
    // CHECK:       return [[CONVERT_OUT]] : tensor<1x10x3584xf32>
}



// -----

// CHECK-LABEL: @MoveMultiplyPostGather
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x10xsi32>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<10000x1xf16>
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<10000x3584xui8>
func.func @MoveMultiplyPostGather(%arg0: tensor<1x10xsi32>, %arg1: tensor<10000x1xf16>, %arg2: tensor<10000x3584xui8>) -> tensor<1x10x3584xf32> {
    %0 = IE.Convert(%arg2) {dstElemType = f16} : tensor<10000x3584xui8> -> tensor<10000x3584xf16>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<10000x3584xf16>, tensor<10000x1xf16> -> tensor<10000x3584xf16>
    %2 = IE.Convert(%1) {dstElemType = f32} : tensor<10000x3584xf16> -> tensor<10000x3584xf32>
    %3 = IE.Gather(%2, %arg0) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<10000x3584xf32>, tensor<1x10xsi32> -> tensor<1x10x3584xf32>
    return %3 : tensor<1x10x3584xf32>

    // CHECK:       [[GATHER_IN:%.+]] = IE.Gather([[INPUT_2]], [[INPUT_0]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<10000x3584xui8>, tensor<1x10xsi32> -> tensor<1x10x3584xui8>
    // CHECK:       [[CONVERT_IN:%.+]] = IE.Convert([[GATHER_IN]]) {dstElemType = f16} : tensor<1x10x3584xui8> -> tensor<1x10x3584xf16>
    // CHECK:       [[GATHER_MUL:%.+]]  = IE.Gather([[INPUT_1]], [[INPUT_0]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<10000x1xf16>, tensor<1x10xsi32> -> tensor<1x10x1xf16>
    // CHECK:       [[MULTIPLY:%.+]]  = IE.Multiply([[CONVERT_IN]], [[GATHER_MUL]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x3584xf16>, tensor<1x10x1xf16> -> tensor<1x10x3584xf16>
    // CHECK:       [[CONVERT_OUT:%.+]]  = IE.Convert([[MULTIPLY]]) {dstElemType = f32} : tensor<1x10x3584xf16> -> tensor<1x10x3584xf32>
    // CHECK:       return [[CONVERT_OUT]] : tensor<1x10x3584xf32>
}


// -----

// CHECK-LABEL: @MoveMultiplyPostGatherWithOutConvert
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x10xsi32>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<10000x1xf16>
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<10000x3584xui8>
func.func @MoveMultiplyPostGatherWithOutConvert(%arg0: tensor<1x10xsi32>, %arg1: tensor<10000x1xf16>, %arg2: tensor<10000x3584xui8>) -> tensor<1x10x3584xf16> {
    %0 = IE.Convert(%arg2) {dstElemType = f16} : tensor<10000x3584xui8> -> tensor<10000x3584xf16>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<10000x3584xf16>, tensor<10000x1xf16> -> tensor<10000x3584xf16>
    %2 = IE.Gather(%1, %arg0) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<10000x3584xf16>, tensor<1x10xsi32> -> tensor<1x10x3584xf16>
    return %2 : tensor<1x10x3584xf16>

    // CHECK:       [[GATHER_IN:%.+]] = IE.Gather([[INPUT_2]], [[INPUT_0]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<10000x3584xui8>, tensor<1x10xsi32> -> tensor<1x10x3584xui8>
    // CHECK:       [[CONVERT_IN:%.+]] = IE.Convert([[GATHER_IN]]) {dstElemType = f16} : tensor<1x10x3584xui8> -> tensor<1x10x3584xf16>
    // CHECK:       [[GATHER_MUL:%.+]]  = IE.Gather([[INPUT_1]], [[INPUT_0]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<10000x1xf16>, tensor<1x10xsi32> -> tensor<1x10x1xf16>
    // CHECK:       [[MULTIPLY:%.+]]  = IE.Multiply([[CONVERT_IN]], [[GATHER_MUL]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x3584xf16>, tensor<1x10x1xf16> -> tensor<1x10x3584xf16>
    // CHECK:       return [[MULTIPLY]] : tensor<1x10x3584xf16>
}

// -----

// CHECK-LABEL: @NotConvertForAxisNotZero
func.func @NotConvertForAxisNotZero(%arg0: tensor<1x10xsi32>, %arg1: tensor<10000x1xf16>, %arg2: tensor<10000x3584xui8>) -> tensor<10000x1x10xf16> {
    %0 = IE.Convert(%arg2) {dstElemType = f16} : tensor<10000x3584xui8> -> tensor<10000x3584xf16>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<10000x3584xf16>, tensor<10000x1xf16> -> tensor<10000x3584xf16>
    %2 = IE.Gather(%1, %arg0) {axis_value = 1 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<10000x3584xf16>, tensor<1x10xsi32> -> tensor<10000x1x10xf16>
    return %2 : tensor<10000x1x10xf16>

    // CHECK:       [[CONVERT_IN:%.+]] = IE.Convert
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply
    // CHECK:       [[GATHER:%.+]] = IE.Gather
    // CHECK:       return [[GATHER]] : tensor<10000x1x10xf16>
}

// -----

// CHECK-LABEL: @MoveINT8ConvertAfterGather
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<73440x1536xsi8>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1xsi64>
func.func @MoveINT8ConvertAfterGather(%arg0: tensor<73440x1536xsi8>, %arg1: tensor<1x1xsi64>) -> tensor<1x1x1536xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<73440x1536xsi8> -> tensor<73440x1536xf16>
    %1 = IE.Convert(%arg1) {dstElemType = si32} : tensor<1x1xsi64> -> tensor<1x1xsi32>
    %2 = IE.Gather(%0, %1) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<73440x1536xf16>, tensor<1x1xsi32> -> tensor<1x1x1536xf16>

    return %2 : tensor<1x1x1536xf16>

    // CHECK:       [[INDICES:%.+]] = IE.Convert([[INPUT_1]]) {dstElemType = si32} : tensor<1x1xsi64> -> tensor<1x1xsi32>
    // CHECK:       [[GATHER:%.+]] = IE.Gather([[INPUT_0]], [[INDICES]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<73440x1536xsi8>, tensor<1x1xsi32> -> tensor<1x1x1536xsi8>
    // CHECK:       [[CONVERT:%.+]] = IE.Convert([[GATHER]]) {dstElemType = f16} : tensor<1x1x1536xsi8> -> tensor<1x1x1536xf16>

    // CHECK:       return [[CONVERT]] : tensor<1x1x1536xf16>
}

// -----

!qElemType = !quant.uniform<i4:f32, 1.000000e+00>

// CHECK-LABEL: @MoveDynamicDQAfterGather
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<645632x224x!qElemType>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<645632x1xf16>
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x1024xsi32>
func.func @MoveDynamicDQAfterGather(%arg0: tensor<645632x224x!qElemType>, %arg1: tensor<645632x1xf16>, %arg2: tensor<1x1024xsi32>) -> tensor<1x1024x224xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<645632x224x!qElemType>, tensor<645632x1xf16> -> tensor<645632x224xf16>
    %1 = IE.Gather(%0, %arg2) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<645632x224xf16>, tensor<1x1024xsi32> -> tensor<1x1024x224xf16>

    return %1 : tensor<1x1024x224xf16>

    // CHECK:       [[GATHER0:%.+]] = IE.Gather([[INPUT_0]], [[INPUT_2]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<645632x224x!qElemType>, tensor<1x1024xsi32> -> tensor<1x1024x224x!qElemType>
    // CHECK:       [[GATHER1:%.+]] = IE.Gather([[INPUT_1]], [[INPUT_2]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<645632x1xf16>, tensor<1x1024xsi32> -> tensor<1x1024x1xf16>
    // CHECK:       [[DQ:%.+]] = IE.DynamicDequantize([[GATHER0]], [[GATHER1]]) {dstElemType = f16} : tensor<1x1024x224x!qElemType>, tensor<1x1024x1xf16> -> tensor<1x1024x224xf16>

    // CHECK:       return [[DQ]] : tensor<1x1024x224xf16>
}

// -----

!qElemType = !quant.uniform<i4:f32, 1.000000e+00>

// CHECK-LABEL: @NotMoveDynamicDQAfterGatherDueToAxisBroadcast
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<645632x224x!qElemType>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x224xf16>
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x1024xsi32>
func.func @NotMoveDynamicDQAfterGatherDueToAxisBroadcast(%arg0: tensor<645632x224x!qElemType>, %arg1: tensor<1x224xf16>, %arg2: tensor<1x1024xsi32>) -> tensor<1x1024x224xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<645632x224x!qElemType>, tensor<1x224xf16> -> tensor<645632x224xf16>
    %1 = IE.Gather(%0, %arg2) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<645632x224xf16>, tensor<1x1024xsi32> -> tensor<1x1024x224xf16>

    return %1 : tensor<1x1024x224xf16>

    // CHECK:       [[DQ:%.+]] = IE.DynamicDequantize([[INPUT_0]], [[INPUT_1]]) {dstElemType = f16} : tensor<645632x224x!qElemType>, tensor<1x224xf16> -> tensor<645632x224xf16>
    // CHECK:       [[GATHER:%.+]] = IE.Gather([[DQ]], [[INPUT_2]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<645632x224xf16>, tensor<1x1024xsi32> -> tensor<1x1024x224xf16>

    // CHECK:       return [[GATHER]] : tensor<1x1024x224xf16>
}

// -----

!qElemType = !quant.uniform<i4:f32, 1.000000e+00>

// CHECK-LABEL: @MoveTwoAxesDynamicDQAfterGather
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<256128x36x128x!qElemType>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<256128x36x1xf16>
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x1023xsi32>
func.func @MoveTwoAxesDynamicDQAfterGather(%arg0: tensor<256128x36x128x!qElemType>, %arg1: tensor<256128x36x1xf16>, %arg2: tensor<1x1023xsi32>) -> tensor<1x1023x36x128xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<256128x36x128x!qElemType>, tensor<256128x36x1xf16> -> tensor<256128x36x128xf16>
    %1 = IE.Gather(%0, %arg2) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<256128x36x128xf16>, tensor<1x1023xsi32> -> tensor<1x1023x36x128xf16>

    return %1 : tensor<1x1023x36x128xf16>

    // CHECK:       [[GATHER0:%.+]] = IE.Gather([[INPUT_0]], [[INPUT_2]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<256128x36x128x!qElemType>, tensor<1x1023xsi32> -> tensor<1x1023x36x128x!qElemType>
    // CHECK:       [[GATHER1:%.+]] = IE.Gather([[INPUT_1]], [[INPUT_2]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<256128x36x1xf16>, tensor<1x1023xsi32> -> tensor<1x1023x36x1xf16>
    // CHECK:       [[DQ:%.+]] = IE.DynamicDequantize([[GATHER0]], [[GATHER1]]) {dstElemType = f16} : tensor<1x1023x36x128x!qElemType>, tensor<1x1023x36x1xf16> -> tensor<1x1023x36x128xf16>

    // CHECK:       return [[DQ]] : tensor<1x1023x36x128xf16>
}

// -----

!qElemType = !quant.uniform<i4:f32, 1.000000e+00>

// CHECK-LABEL: @NotMoveDynamicDQWithZPAfterGather
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<645632x224x!qElemType>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<645632x1xf16>
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<645632x224xsi4>
// CHECK-SAME:      [[INPUT_3:%.+]]: tensor<1x1024xsi32>
func.func @NotMoveDynamicDQWithZPAfterGather(%arg0: tensor<645632x224x!qElemType>, %arg1: tensor<645632x1xf16>, %arg2: tensor<645632x224xsi4>, %arg3: tensor<1x1024xsi32>) -> tensor<1x1024x224xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1, %arg2) {dstElemType = f16} : tensor<645632x224x!qElemType>, tensor<645632x1xf16>, tensor<645632x224xsi4> -> tensor<645632x224xf16>
    %1 = IE.Gather(%0, %arg3) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<645632x224xf16>, tensor<1x1024xsi32> -> tensor<1x1024x224xf16>

    return %1 : tensor<1x1024x224xf16>

    // CHECK:       [[DQ:%.+]] = IE.DynamicDequantize([[INPUT_0]], [[INPUT_1]], [[INPUT_2]]) {dstElemType = f16} : tensor<645632x224x!qElemType>, tensor<645632x1xf16>, tensor<645632x224xsi4> -> tensor<645632x224xf16>
    // CHECK:       [[GATHER:%.+]] = IE.Gather([[DQ]], [[INPUT_3]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<645632x224xf16>, tensor<1x1024xsi32> -> tensor<1x1024x224xf16>

    // CHECK:       return [[GATHER]] : tensor<1x1024x224xf16>
}

// -----

!qElemType = !quant.uniform<i8:f32, 1.000000e+00>

// CHECK-LABEL: @MoveDynamicDQAfterGatherWithPerTensorScale
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<64xsi32>
func.func @MoveDynamicDQAfterGatherWithPerTensorScale(%arg0: tensor<64xsi32>) -> tensor<64x512xf32> {
    %cst = const.Declare tensor<4096x512x!qElemType> = dense<1> : tensor<4096x512xsi8>, [#const.CastElemType<f32>, #const.CastElemType<si8>, #const.CastElemType<!qElemType>]
    %cst_0 = const.Declare tensor<1x1xf16> = dense<3.921570e-03> : tensor<1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.DynamicDequantize(%cst, %cst_0) {dstElemType = f16} : tensor<4096x512x!qElemType>, tensor<1x1xf16> -> tensor<4096x512xf16>
    %1 = IE.Gather(%0, %arg0) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<4096x512xf16>, tensor<64xsi32> -> tensor<64x512xf16>
    %2 = IE.Convert(%1) {dstElemType = f32} : tensor<64x512xf16> -> tensor<64x512xf32>
    return %2 : tensor<64x512xf32>

    // Weights are gathered; per-tensor scale [1x1] is passed through unchanged.
    // CHECK-DAG:   [[WEIGHT:%.+]] = const.Declare tensor<4096x512x!qElemType>
    // CHECK-DAG:   [[SCALE:%.+]]  = const.Declare tensor<1x1xf16>
    // CHECK:       [[GATHER:%.+]] = IE.Gather([[WEIGHT]], [[INPUT_0]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<4096x512x!qElemType>, tensor<64xsi32> -> tensor<64x512x!qElemType>
    // CHECK:       [[DQ:%.+]]     = IE.DynamicDequantize([[GATHER]], [[SCALE]]) {dstElemType = f16} : tensor<64x512x!qElemType>, tensor<1x1xf16> -> tensor<64x512xf16>
    // CHECK:       [[CONVERT:%.+]] = IE.Convert([[DQ]]) {dstElemType = f32} : tensor<64x512xf16> -> tensor<64x512xf32>
    // CHECK:       return [[CONVERT]] : tensor<64x512xf32>
}

// -----

!qElemType = !quant.uniform<i8:f32, 1.000000e+00>

// CHECK-LABEL: @NotMoveDynamicDQAfterGatherPerColumnScaleRankMismatch
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<645632x224x!qElemType>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x224xf16>
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x1024xsi32>
func.func @NotMoveDynamicDQAfterGatherPerColumnScaleRankMismatch(%arg0: tensor<645632x224x!qElemType>, %arg1: tensor<1x224xf16>, %arg2: tensor<1x1024xsi32>) -> tensor<1x1024x224xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<645632x224x!qElemType>, tensor<1x224xf16> -> tensor<645632x224xf16>
    %1 = IE.Gather(%0, %arg2) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<645632x224xf16>, tensor<1x1024xsi32> -> tensor<1x1024x224xf16>
    return %1 : tensor<1x1024x224xf16>

    // CHECK:       [[DQ:%.+]]    = IE.DynamicDequantize([[INPUT_0]], [[INPUT_1]]) {dstElemType = f16} : tensor<645632x224x!qElemType>, tensor<1x224xf16> -> tensor<645632x224xf16>
    // CHECK:       [[GATHER:%.+]] = IE.Gather([[DQ]], [[INPUT_2]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<645632x224xf16>, tensor<1x1024xsi32> -> tensor<1x1024x224xf16>
    // CHECK:       return [[GATHER]] : tensor<1x1024x224xf16>
}

// -----

// CHECK-LABEL: @MoveConvertF32ToF16AfterGatherLargeReduction
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x1024x35x256xf32>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1xsi32>
func.func @MoveConvertF32ToF16AfterGatherLargeReduction(%arg0: tensor<1x1024x35x256xf32>, %arg1: tensor<1xsi32>) -> tensor<1x1024x256xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x1024x35x256xf32> -> tensor<1x1024x35x256xf16>
    %1 = IE.Gather(%0, %arg1) {axis_value = 2 : i64, batch_dims = 0 : i64, indices_rank = 0 : i64} : tensor<1x1024x35x256xf16>, tensor<1xsi32> -> tensor<1x1024x256xf16>
    return %1 : tensor<1x1024x256xf16>

    // CHECK:       [[GATHER:%.+]] = IE.Gather([[INPUT_0]], [[INPUT_1]]) {axis_value = 2 : i64, batch_dims = 0 : i64, indices_rank = 0 : i64} : tensor<1x1024x35x256xf32>, tensor<1xsi32> -> tensor<1x1024x256xf32>
    // CHECK:       [[CONVERT:%.+]] = IE.Convert([[GATHER]]) {dstElemType = f16} : tensor<1x1024x256xf32> -> tensor<1x1024x256xf16>
    // CHECK:       return [[CONVERT]] : tensor<1x1024x256xf16>
}

// -----

// CHECK-LABEL: @NotMoveConvertAfterGatherSmallReduction
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x1024x2x256xf32>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1xsi32>
func.func @NotMoveConvertAfterGatherSmallReduction(%arg0: tensor<1x1024x2x256xf32>, %arg1: tensor<1xsi32>) -> tensor<1x1024x256xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x1024x2x256xf32> -> tensor<1x1024x2x256xf16>
    %1 = IE.Gather(%0, %arg1) {axis_value = 2 : i64, batch_dims = 0 : i64, indices_rank = 0 : i64} : tensor<1x1024x2x256xf16>, tensor<1xsi32> -> tensor<1x1024x256xf16>
    return %1 : tensor<1x1024x256xf16>

    // CHECK:       [[CONVERT:%.+]] = IE.Convert([[INPUT_0]]) {dstElemType = f16} : tensor<1x1024x2x256xf32> -> tensor<1x1024x2x256xf16>
    // CHECK:       [[GATHER:%.+]] = IE.Gather([[CONVERT]], [[INPUT_1]]) {axis_value = 2 : i64, batch_dims = 0 : i64, indices_rank = 0 : i64} : tensor<1x1024x2x256xf16>, tensor<1xsi32> -> tensor<1x1024x256xf16>
    // CHECK:       return [[GATHER]] : tensor<1x1024x256xf16>
}

// -----

// CHECK-LABEL: @MoveSliceBeforeMultiplyBothInputs
// CHECK-SAME:      [[ARG0:%[^:]+]]: tensor<1x4x2x2xf16>, [[ARG1:%[^:]+]]: tensor<1x4x2x2xf16>
func.func @MoveSliceBeforeMultiplyBothInputs(%arg0: tensor<1x4x2x2xf16>, %arg1: tensor<1x4x2x2xf16>) -> tensor<1x1x2x2xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x2x2xf16>, tensor<1x4x2x2xf16> -> tensor<1x4x2x2xf16>
    %1 = IE.Slice %0 [0, 2, 0, 0] [1, 1, 2, 2] : tensor<1x4x2x2xf16> to tensor<1x1x2x2xf16>
    return %1 : tensor<1x1x2x2xf16>

    // CHECK-DAG: [[SLICE0:%.+]] = IE.Slice [[ARG0]] [0, 2, 0, 0] [1, 1, 2, 2] : tensor<1x4x2x2xf16> to tensor<1x1x2x2xf16>
    // CHECK-DAG: [[SLICE1:%.+]] = IE.Slice [[ARG1]] [0, 2, 0, 0] [1, 1, 2, 2] : tensor<1x4x2x2xf16> to tensor<1x1x2x2xf16>
    // CHECK:     [[MUL:%.+]] = IE.Multiply([[SLICE0]], [[SLICE1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x2xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x2x2xf16>
    // CHECK:     return [[MUL]]
}

// -----

// CHECK-LABEL: @NotMoveSliceBeforeMultiplyWithoutReduction
// CHECK-SAME:      [[ARG0:%[^:]+]]: tensor<1x4x2x2xf16>, [[ARG1:%[^:]+]]: tensor<1x4x2x2xf16>
func.func @NotMoveSliceBeforeMultiplyWithoutReduction(%arg0: tensor<1x4x2x2xf16>, %arg1: tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x2x2xf16>, tensor<1x4x2x2xf16> -> tensor<1x4x2x2xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 4, 2, 2] : tensor<1x4x2x2xf16> to tensor<1x4x2x2xf16>
    return %1 : tensor<1x4x2x2xf16>

    // CHECK: [[MUL:%.+]] = IE.Multiply([[ARG0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x2x2xf16>, tensor<1x4x2x2xf16> -> tensor<1x4x2x2xf16>
    // CHECK: return [[MUL]]
}

// -----

// CHECK-LABEL: @MoveSliceBeforeMultiplyWithScalarInput
// CHECK-SAME: [[ARG0:%[^:]+]]: tensor<1x4x2x2xf16>, [[ARG1:%[^:]+]]: tensor<1x1x1x1xf16>
func.func @MoveSliceBeforeMultiplyWithScalarInput(%arg0: tensor<1x4x2x2xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x1x2x2xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x2x2xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x2x2xf16>
    %1 = IE.Slice %0 [0, 2, 0, 0] [1, 1, 2, 2] : tensor<1x4x2x2xf16> to tensor<1x1x2x2xf16>
    return %1 : tensor<1x1x2x2xf16>

    // CHECK: [[SLICE0:%.+]] = IE.Slice [[ARG0]] [0, 2, 0, 0] [1, 1, 2, 2] : tensor<1x4x2x2xf16> to tensor<1x1x2x2xf16>
    // CHECK: [[MUL:%.+]] = IE.Multiply([[SLICE0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x2xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>
    // CHECK: return [[MUL]] : tensor<1x1x2x2xf16>
}

// -----

// CHECK-LABEL: @NotMoveSliceBeforeMultiplyMultipleUsers
// CHECK-SAME:      [[ARG0:%[^:]+]]: tensor<1x4x2x2xf16>, [[ARG1:%[^:]+]]: tensor<1x4x2x2xf16>
func.func @NotMoveSliceBeforeMultiplyMultipleUsers(%arg0: tensor<1x4x2x2xf16>, %arg1: tensor<1x4x2x2xf16>) -> (tensor<1x1x2x2xf16>, tensor<1x4x2x2xf16>) {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x2x2xf16>, tensor<1x4x2x2xf16> -> tensor<1x4x2x2xf16>
    %1 = IE.Slice %0 [0, 2, 0, 0] [1, 1, 2, 2] : tensor<1x4x2x2xf16> to tensor<1x1x2x2xf16>
    return %1, %0 : tensor<1x1x2x2xf16>, tensor<1x4x2x2xf16>

    // CHECK: [[MUL:%.+]] = IE.Multiply([[ARG0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x2x2xf16>, tensor<1x4x2x2xf16> -> tensor<1x4x2x2xf16>
    // CHECK: [[SLICE:%.+]] = IE.Slice [[MUL]] [0, 2, 0, 0] [1, 1, 2, 2] : tensor<1x4x2x2xf16> to tensor<1x1x2x2xf16>
    // CHECK: return [[SLICE]], [[MUL]]
}

// -----

// CHECK-LABEL: @MoveSliceBeforeMultiplyMultipleAxes
// CHECK-SAME:      [[ARG0:%[^:]+]]: tensor<2x4x2x2xf16>, [[ARG1:%[^:]+]]: tensor<2x4x2x2xf16>
func.func @MoveSliceBeforeMultiplyMultipleAxes(%arg0: tensor<2x4x2x2xf16>, %arg1: tensor<2x4x2x2xf16>) -> tensor<1x2x2x2xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x4x2x2xf16>, tensor<2x4x2x2xf16> -> tensor<2x4x2x2xf16>
    %1 = IE.Slice %0 [0, 1, 0, 0] [1, 2, 2, 2] : tensor<2x4x2x2xf16> to tensor<1x2x2x2xf16>
    return %1 : tensor<1x2x2x2xf16>

    // CHECK-DAG: [[SLICE0:%.+]] = IE.Slice [[ARG0]] [0, 1, 0, 0] [1, 2, 2, 2] : tensor<2x4x2x2xf16> to tensor<1x2x2x2xf16>
    // CHECK-DAG: [[SLICE1:%.+]] = IE.Slice [[ARG1]] [0, 1, 0, 0] [1, 2, 2, 2] : tensor<2x4x2x2xf16> to tensor<1x2x2x2xf16>
    // CHECK:     [[MUL:%.+]] = IE.Multiply([[SLICE0]], [[SLICE1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x2x2xf16>, tensor<1x2x2x2xf16> -> tensor<1x2x2x2xf16>
    // CHECK:     return [[MUL]]
}

// -----

// CHECK-LABEL: @NotMoveSliceBeforeMultiplyBroadcastMultiAxis
// CHECK-SAME:      [[ARG0:%[^:]+]]: tensor<2x4x2x2xf16>, [[ARG1:%[^:]+]]: tensor<1x1x2x2xf16>
func.func @NotMoveSliceBeforeMultiplyBroadcastMultiAxis(%arg0: tensor<2x4x2x2xf16>, %arg1: tensor<1x1x2x2xf16>) -> tensor<1x4x2x2xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x4x2x2xf16>, tensor<1x1x2x2xf16> -> tensor<2x4x2x2xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 4, 2, 2] : tensor<2x4x2x2xf16> to tensor<1x4x2x2xf16>
    return %1 : tensor<1x4x2x2xf16>

    // CHECK: [[MUL:%.+]] = IE.Multiply([[ARG0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x4x2x2xf16>, tensor<1x1x2x2xf16> -> tensor<2x4x2x2xf16>
    // CHECK: [[SLICE:%.+]] = IE.Slice [[MUL]] [0, 0, 0, 0] [1, 4, 2, 2] : tensor<2x4x2x2xf16> to tensor<1x4x2x2xf16>
    // CHECK: return [[SLICE]]
}

// -----

// CHECK-LABEL: @MoveSliceBeforeSwish
// CHECK-SAME:      [[ARG0:%[^:]+]]: tensor<1x4x16xf16>
func.func @MoveSliceBeforeSwish(%arg0: tensor<1x4x16xf16>) -> tensor<1x1x16xf16> {
    %0 = IE.Swish(%arg0) : tensor<1x4x16xf16> -> tensor<1x4x16xf16>
    %1 = IE.Slice %0 [0, 2, 0] [1, 1, 16] : tensor<1x4x16xf16> to tensor<1x1x16xf16>
    return %1 : tensor<1x1x16xf16>

    // CHECK: [[SLICE:%.+]] = IE.Slice [[ARG0]] [0, 2, 0] [1, 1, 16] : tensor<1x4x16xf16> to tensor<1x1x16xf16>
    // CHECK: [[SWISH:%.+]] = IE.Swish([[SLICE]]) : tensor<1x1x16xf16> -> tensor<1x1x16xf16>
    // CHECK: return [[SWISH]]
}

// -----

// CHECK-LABEL: @NotMoveSliceBeforeSwishWithoutReduction
// CHECK-SAME:      [[ARG0:%[^:]+]]: tensor<1x4x16xf16>
func.func @NotMoveSliceBeforeSwishWithoutReduction(%arg0: tensor<1x4x16xf16>) -> tensor<1x4x16xf16> {
    %0 = IE.Swish(%arg0) : tensor<1x4x16xf16> -> tensor<1x4x16xf16>
    %1 = IE.Slice %0 [0, 0, 0] [1, 4, 16] : tensor<1x4x16xf16> to tensor<1x4x16xf16>
    return %1 : tensor<1x4x16xf16>

    // CHECK: [[SWISH:%.+]] = IE.Swish([[ARG0]]) : tensor<1x4x16xf16> -> tensor<1x4x16xf16>
    // CHECK: return [[SWISH]]
}

// -----

// CHECK-LABEL: @NotMoveSliceBeforeSwishMultipleUsers
// CHECK-SAME:      [[ARG0:%[^:]+]]: tensor<1x4x16xf16>
func.func @NotMoveSliceBeforeSwishMultipleUsers(%arg0: tensor<1x4x16xf16>) -> (tensor<1x1x16xf16>, tensor<1x4x16xf16>) {
    %0 = IE.Swish(%arg0) : tensor<1x4x16xf16> -> tensor<1x4x16xf16>
    %1 = IE.Slice %0 [0, 2, 0] [1, 1, 16] : tensor<1x4x16xf16> to tensor<1x1x16xf16>
    return %1, %0 : tensor<1x1x16xf16>, tensor<1x4x16xf16>

    // CHECK: [[SWISH:%.+]] = IE.Swish([[ARG0]]) : tensor<1x4x16xf16> -> tensor<1x4x16xf16>
    // CHECK: [[SLICE:%.+]] = IE.Slice [[SWISH]] [0, 2, 0] [1, 1, 16] : tensor<1x4x16xf16> to tensor<1x1x16xf16>
    // CHECK: return [[SLICE]], [[SWISH]]
}

// -----

// CHECK-LABEL: @MoveSliceBeforeFullyConnectedDirect
// CHECK-SAME:      [[ARG0:%[^:]+]]: tensor<4x8xf16>
func.func @MoveSliceBeforeFullyConnectedDirect(%arg0: tensor<4x8xf16>) -> tensor<1x16xf16> {
    %weights = const.Declare tensor<16x8xf16> = dense<1.0> : tensor<16x8xf16>
    %0 = IE.FullyConnected(%arg0, %weights) : tensor<4x8xf16>, tensor<16x8xf16> -> tensor<4x16xf16>
    %1 = IE.Slice %0 [3, 0] [1, 16] : tensor<4x16xf16> to tensor<1x16xf16>
    return %1 : tensor<1x16xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<16x8xf16> = dense{{.*}}1.0{{.*}}
    // CHECK: [[SLICE:%.+]] = IE.Slice [[ARG0]] [3, 0] [1, 8] : tensor<4x8xf16> to tensor<1x8xf16>
    // CHECK: [[FC:%.+]] = IE.FullyConnected([[SLICE]], [[CST]]) : tensor<1x8xf16>, tensor<16x8xf16> -> tensor<1x16xf16>
    // CHECK: return [[FC]]
}

// -----

// CHECK-LABEL: @NotMoveSliceBeforeFullyConnectedWithoutReduction
// CHECK-SAME:      [[ARG0:%[^:]+]]: tensor<4x8xf16>
func.func @NotMoveSliceBeforeFullyConnectedWithoutReduction(%arg0: tensor<4x8xf16>) -> tensor<4x16xf16> {
    %weights = const.Declare tensor<16x8xf16> = dense<1.0> : tensor<16x8xf16>
    %0 = IE.FullyConnected(%arg0, %weights) : tensor<4x8xf16>, tensor<16x8xf16> -> tensor<4x16xf16>
    %1 = IE.Slice %0 [0, 0] [4, 16] : tensor<4x16xf16> to tensor<4x16xf16>
    return %1 : tensor<4x16xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<16x8xf16> = dense{{.*}}1.0{{.*}}
    // CHECK: [[FC:%.+]] = IE.FullyConnected([[ARG0]], [[CST]]) : tensor<4x8xf16>, tensor<16x8xf16> -> tensor<4x16xf16>
    // CHECK: return [[FC]]
}

// -----

// CHECK-LABEL: @NotMoveSliceBeforeFullyConnectedMultipleUsers
// CHECK-SAME:      [[ARG0:%[^:]+]]: tensor<4x8xf16>
func.func @NotMoveSliceBeforeFullyConnectedMultipleUsers(%arg0: tensor<4x8xf16>) -> (tensor<1x16xf16>, tensor<4x16xf16>) {
    %weights = const.Declare tensor<16x8xf16> = dense<1.0> : tensor<16x8xf16>
    %0 = IE.FullyConnected(%arg0, %weights) : tensor<4x8xf16>, tensor<16x8xf16> -> tensor<4x16xf16>
    %1 = IE.Slice %0 [3, 0] [1, 16] : tensor<4x16xf16> to tensor<1x16xf16>
    return %1, %0 : tensor<1x16xf16>, tensor<4x16xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<16x8xf16> = dense{{.*}}1.0{{.*}}
    // CHECK: [[FC:%.+]] = IE.FullyConnected([[ARG0]], [[CST]]) : tensor<4x8xf16>, tensor<16x8xf16> -> tensor<4x16xf16>
    // CHECK: [[SLICE:%.+]] = IE.Slice [[FC]] [3, 0] [1, 16] : tensor<4x16xf16> to tensor<1x16xf16>
    // CHECK: return [[SLICE]], [[FC]]
}

// -----

// CHECK-LABEL: @NotMoveSliceBeforeFullyConnectedChannelSlice
// CHECK-SAME:      [[ARG0:%[^:]+]]: tensor<4x8xf16>
func.func @NotMoveSliceBeforeFullyConnectedChannelSlice(%arg0: tensor<4x8xf16>) -> tensor<4x8xf16> {
    %weights = const.Declare tensor<16x8xf16> = dense<1.0> : tensor<16x8xf16>
    %0 = IE.FullyConnected(%arg0, %weights) : tensor<4x8xf16>, tensor<16x8xf16> -> tensor<4x16xf16>
    %1 = IE.Slice %0 [0, 4] [4, 8] : tensor<4x16xf16> to tensor<4x8xf16>
    return %1 : tensor<4x8xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<16x8xf16> = dense{{.*}}1.0{{.*}}
    // CHECK: [[FC:%.+]] = IE.FullyConnected([[ARG0]], [[CST]]) : tensor<4x8xf16>, tensor<16x8xf16> -> tensor<4x16xf16>
    // CHECK: [[SLICE:%.+]] = IE.Slice [[FC]] [0, 4] [4, 8] : tensor<4x16xf16> to tensor<4x8xf16>
    // CHECK: return [[SLICE]]
}

// -----

// CHECK-LABEL: @MoveSliceBeforeFullyConnectedThroughReshape
// CHECK-SAME:      [[ARG0:%[^:]+]]: tensor<4x8xf16>
func.func @MoveSliceBeforeFullyConnectedThroughReshape(%arg0: tensor<4x8xf16>) -> tensor<1x1x16xf16> {
    %weights = const.Declare tensor<16x8xf16> = dense<1.0> : tensor<16x8xf16>
    %0 = IE.FullyConnected(%arg0, %weights) : tensor<4x8xf16>, tensor<16x8xf16> -> tensor<4x16xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1]], shape_value = [1, 4, 16]} : tensor<4x16xf16> -> tensor<1x4x16xf16>
    %2 = IE.Slice %1 [0, 2, 0] [1, 1, 16] : tensor<1x4x16xf16> to tensor<1x1x16xf16>
    return %2 : tensor<1x1x16xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<16x8xf16> = dense{{.*}}1.0{{.*}}
    // CHECK: [[SLICE:%.+]] = IE.Slice [[ARG0]] [2, 0] [1, 8] : tensor<4x8xf16> to tensor<1x8xf16>
    // CHECK: [[FC:%.+]] = IE.FullyConnected([[SLICE]], [[CST]]) : tensor<1x8xf16>, tensor<16x8xf16> -> tensor<1x16xf16>
    // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[FC]]) {dim_mapping = {{.*}}, shape_value = [1, 1, 16]} : tensor<1x16xf16> -> tensor<1x1x16xf16>
    // CHECK: return [[RESHAPE]]
}

// -----

// CHECK-LABEL: @NotMoveSliceBeforeFullyConnectedReshapeMultipleUsers
// CHECK-SAME:      [[ARG0:%[^:]+]]: tensor<4x8xf16>
func.func @NotMoveSliceBeforeFullyConnectedReshapeMultipleUsers(%arg0: tensor<4x8xf16>) -> (tensor<1x1x16xf16>, tensor<1x4x16xf16>) {
    %weights = const.Declare tensor<16x8xf16> = dense<1.0> : tensor<16x8xf16>
    %0 = IE.FullyConnected(%arg0, %weights) : tensor<4x8xf16>, tensor<16x8xf16> -> tensor<4x16xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1]], shape_value = [1, 4, 16]} : tensor<4x16xf16> -> tensor<1x4x16xf16>
    %2 = IE.Slice %1 [0, 2, 0] [1, 1, 16] : tensor<1x4x16xf16> to tensor<1x1x16xf16>
    return %2, %1 : tensor<1x1x16xf16>, tensor<1x4x16xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<16x8xf16> = dense{{.*}}1.0{{.*}}
    // CHECK: [[FC:%.+]] = IE.FullyConnected([[ARG0]], [[CST]]) : tensor<4x8xf16>, tensor<16x8xf16> -> tensor<4x16xf16>
    // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[FC]]) {dim_mapping = {{.*}}, shape_value = [1, 4, 16]} : tensor<4x16xf16> -> tensor<1x4x16xf16>
    // CHECK: [[SLICE:%.+]] = IE.Slice [[RESHAPE]] [0, 2, 0] [1, 1, 16] : tensor<1x4x16xf16> to tensor<1x1x16xf16>
    // CHECK: return [[SLICE]], [[RESHAPE]]
}
