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

// CHECK-LABEL: @MoveSliceBeforeMultiplyBroadcastMultiAxis
// CHECK-SAME:      [[ARG0:%[^:]+]]: tensor<2x4x2x2xf16>, [[ARG1:%[^:]+]]: tensor<1x1x2x2xf16>
func.func @MoveSliceBeforeMultiplyBroadcastMultiAxis(%arg0: tensor<2x4x2x2xf16>, %arg1: tensor<1x1x2x2xf16>) -> tensor<1x2x2x2xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x4x2x2xf16>, tensor<1x1x2x2xf16> -> tensor<2x4x2x2xf16>
    %1 = IE.Slice %0 [0, 1, 0, 0] [1, 2, 2, 2] : tensor<2x4x2x2xf16> to tensor<1x2x2x2xf16>
    return %1 : tensor<1x2x2x2xf16>

    // CHECK: [[SLICE0:%.+]] = IE.Slice [[ARG0]] [0, 1, 0, 0] [1, 2, 2, 2] : tensor<2x4x2x2xf16> to tensor<1x2x2x2xf16>
    // CHECK: [[MUL:%.+]] = IE.Multiply([[SLICE0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x2x2xf16>, tensor<1x1x2x2xf16> -> tensor<1x2x2x2xf16>
    // CHECK: return [[MUL]]
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

// -----

// CHECK-LABEL: @MoveConvertBeforeGatherThroughReshape
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<16512x1xf16>
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: tensor<1x1x1x16512xsi32>
func.func @MoveConvertBeforeGatherThroughReshape(%arg0: tensor<16512x1xf16>, %arg1: tensor<1x1x1x16512xsi32>) -> tensor<1x1x1x16512xi8> {
    %0 = IE.Gather(%arg0, %arg1) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 4 : i64} : tensor<16512x1xf16>, tensor<1x1x1x16512xsi32> -> tensor<1x1x1x16512x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [2], [3], [3]], shape_value = [1, 1, 1, 16512]} : tensor<1x1x1x16512x1xf16> -> tensor<1x1x1x16512xf16>
    %2 = IE.Convert(%1) {dstElemType = i8} : tensor<1x1x1x16512xf16> -> tensor<1x1x1x16512xi8>
    return %2 : tensor<1x1x1x16512xi8>

    // Convert is hoisted before Gather; Gather and AffineReshape operate on i8.
    // CHECK:       [[CONVERT:%.+]] = IE.Convert([[INPUT_0]]) {dstElemType = i8} : tensor<16512x1xf16> -> tensor<16512x1xi8>
    // CHECK:       [[GATHER:%.+]]  = IE.Gather([[CONVERT]], [[INPUT_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 4 : i64} : tensor<16512x1xi8>, tensor<1x1x1x16512xsi32> -> tensor<1x1x1x16512x1xi8>
    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[GATHER]]) {dim_mapping = {{.*}}, shape_value = [1, 1, 1, 16512]} : tensor<1x1x1x16512x1xi8> -> tensor<1x1x1x16512xi8>
    // CHECK:       return [[RESHAPE]] : tensor<1x1x1x16512xi8>
}

// -----

// CHECK-LABEL: @MoveConvertBeforeGatherDirect
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1024x128xf16>
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: tensor<1024xsi32>
func.func @MoveConvertBeforeGatherDirect(%arg0: tensor<1024x128xf16>, %arg1: tensor<1024xsi32>) -> tensor<1024x128xi8> {
    %0 = IE.Gather(%arg0, %arg1) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<1024x128xf16>, tensor<1024xsi32> -> tensor<1024x128xf16>
    %1 = IE.Convert(%0) {dstElemType = i8} : tensor<1024x128xf16> -> tensor<1024x128xi8>
    return %1 : tensor<1024x128xi8>

    // Gather input and output have the same element count; hoisting is beneficial.
    // CHECK:       [[CONVERT:%.+]] = IE.Convert([[INPUT_0]]) {dstElemType = i8} : tensor<1024x128xf16> -> tensor<1024x128xi8>
    // CHECK:       [[GATHER:%.+]]  = IE.Gather([[CONVERT]], [[INPUT_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<1024x128xi8>, tensor<1024xsi32> -> tensor<1024x128xi8>
    // CHECK:       return [[GATHER]] : tensor<1024x128xi8>
}

// -----

// CHECK-LABEL: @NotMoveConvertBeforeGatherElementSizeIncreases
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1024x128xi8>
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: tensor<1024xsi32>
func.func @NotMoveConvertBeforeGatherElementSizeIncreases(%arg0: tensor<1024x128xi8>, %arg1: tensor<1024xsi32>) -> tensor<1024x128xf16> {
    %0 = IE.Gather(%arg0, %arg1) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<1024x128xi8>, tensor<1024xsi32> -> tensor<1024x128xi8>
    %1 = IE.Convert(%0) {dstElemType = f16} : tensor<1024x128xi8> -> tensor<1024x128xf16>
    return %1 : tensor<1024x128xf16>

    // Convert expands element byte size; hoisting is not beneficial.
    // CHECK:       [[GATHER:%.+]]  = IE.Gather([[INPUT_0]], [[INPUT_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<1024x128xi8>, tensor<1024xsi32> -> tensor<1024x128xi8>
    // CHECK:       [[CONVERT:%.+]] = IE.Convert([[GATHER]]) {dstElemType = f16} : tensor<1024x128xi8> -> tensor<1024x128xf16>
    // CHECK:       return [[CONVERT]] : tensor<1024x128xf16>
}

// -----

// CHECK-LABEL: @NotMoveConvertBeforeGatherGatherReducesElements
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<16512x128xf16>
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: tensor<10xsi32>
func.func @NotMoveConvertBeforeGatherGatherReducesElements(%arg0: tensor<16512x128xf16>, %arg1: tensor<10xsi32>) -> tensor<10x128xi8> {
    %0 = IE.Gather(%arg0, %arg1) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<16512x128xf16>, tensor<10xsi32> -> tensor<10x128xf16>
    %1 = IE.Convert(%0) {dstElemType = i8} : tensor<10x128xf16> -> tensor<10x128xi8>
    return %1 : tensor<10x128xi8>

    // Gather reduces element count from 2113536 to 1280; hoisting Convert would process far more
    // elements than the original Convert, so the transformation is skipped.
    // CHECK:       [[GATHER:%.+]]  = IE.Gather([[INPUT_0]], [[INPUT_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<16512x128xf16>, tensor<10xsi32> -> tensor<10x128xf16>
    // CHECK:       [[CONVERT:%.+]] = IE.Convert([[GATHER]]) {dstElemType = i8} : tensor<10x128xf16> -> tensor<10x128xi8>
    // CHECK:       return [[CONVERT]] : tensor<10x128xi8>
}

// -----

// CHECK-LABEL: @NotMoveConvertBeforeGatherMultipleGatherUsers
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1024x128xf16>
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: tensor<1024xsi32>
func.func @NotMoveConvertBeforeGatherMultipleGatherUsers(%arg0: tensor<1024x128xf16>, %arg1: tensor<1024xsi32>) -> (tensor<1024x128xi8>, tensor<1024x128xf16>) {
    %0 = IE.Gather(%arg0, %arg1) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<1024x128xf16>, tensor<1024xsi32> -> tensor<1024x128xf16>
    %1 = IE.Convert(%0) {dstElemType = i8} : tensor<1024x128xf16> -> tensor<1024x128xi8>
    return %1, %0 : tensor<1024x128xi8>, tensor<1024x128xf16>

    // Gather output has two users; the transformation is skipped.
    // CHECK:       [[GATHER:%.+]]  = IE.Gather([[INPUT_0]], [[INPUT_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<1024x128xf16>, tensor<1024xsi32> -> tensor<1024x128xf16>
    // CHECK:       [[CONVERT:%.+]] = IE.Convert([[GATHER]]) {dstElemType = i8} : tensor<1024x128xf16> -> tensor<1024x128xi8>
    // CHECK:       return [[CONVERT]], [[GATHER]] : tensor<1024x128xi8>, tensor<1024x128xf16>
}

// -----

// CHECK-LABEL: @MoveSliceBeforeAddWithScalarBias
// CHECK-SAME: [[ARG0:%[^:]+]]: tensor<1x4x2x2xf16>, [[ARG1:%[^:]+]]: tensor<1x1x1x1xf16>
func.func @MoveSliceBeforeAddWithScalarBias(%arg0: tensor<1x4x2x2xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x1x2x2xf16> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x2x2xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x2x2xf16>
    %1 = IE.Slice %0 [0, 2, 0, 0] [1, 1, 2, 2] : tensor<1x4x2x2xf16> to tensor<1x1x2x2xf16>
    return %1 : tensor<1x1x2x2xf16>

    // CHECK: [[SLICE0:%.+]] = IE.Slice [[ARG0]] [0, 2, 0, 0] [1, 1, 2, 2] : tensor<1x4x2x2xf16> to tensor<1x1x2x2xf16>
    // CHECK: [[ADD:%.+]] = IE.Add([[SLICE0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x2xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>
    // CHECK: return [[ADD]] : tensor<1x1x2x2xf16>
}

// -----

// CHECK-LABEL: @MoveSliceBeforeGelu
// CHECK-SAME:      [[ARG0:%[^:]+]]: tensor<1x4x16xf16>
func.func @MoveSliceBeforeGelu(%arg0: tensor<1x4x16xf16>) -> tensor<1x1x16xf16> {
    %0 = IE.Gelu(%arg0) : tensor<1x4x16xf16> -> tensor<1x4x16xf16>
    %1 = IE.Slice %0 [0, 2, 0] [1, 1, 16] : tensor<1x4x16xf16> to tensor<1x1x16xf16>
    return %1 : tensor<1x1x16xf16>

    // CHECK: [[SLICE:%.+]] = IE.Slice [[ARG0]] [0, 2, 0] [1, 1, 16] : tensor<1x4x16xf16> to tensor<1x1x16xf16>
    // CHECK: [[GELU:%.+]] = IE.Gelu([[SLICE]]) : tensor<1x1x16xf16> -> tensor<1x1x16xf16>
    // CHECK: return [[GELU]]
}

// -----

// CHECK-LABEL: @MoveSliceBeforeFakeQuantizePerTensor
// CHECK-SAME:      [[ARG0:%[^:]+]]: tensor<1x4x16xf16>
func.func @MoveSliceBeforeFakeQuantizePerTensor(%arg0: tensor<1x4x16xf16>) -> tensor<1x1x16xf16> {
    %input_low = const.Declare tensor<1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1xf16>
    %input_high = const.Declare tensor<1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1xf16>
    %output_low = const.Declare tensor<1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1xf16>
    %output_high = const.Declare tensor<1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1xf16>
    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x16xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16> -> tensor<1x4x16xf16>
    %1 = IE.Slice %0 [0, 2, 0] [1, 1, 16] : tensor<1x4x16xf16> to tensor<1x1x16xf16>
    return %1 : tensor<1x1x16xf16>

    // CHECK-DAG:   [[LOW:%.+]] = const.Declare tensor<1x1x1xf16> = dense<-1.000000e+00>
    // CHECK-DAG:   [[HIGH:%.+]] = const.Declare tensor<1x1x1xf16> = dense<1.000000e+00>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG0]] [0, 2, 0] [1, 1, 16] : tensor<1x4x16xf16> to tensor<1x1x16xf16>
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[SLICE]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x16xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16> -> tensor<1x1x16xf16>
    // CHECK:       return [[FQ]] : tensor<1x1x16xf16>
}

// -----

// CHECK-LABEL: @NotMoveSliceBeforeFakeQuantizePerChannel
// CHECK-SAME:      [[ARG0:%[^:]+]]: tensor<1x4x16xf16>
func.func @NotMoveSliceBeforeFakeQuantizePerChannel(%arg0: tensor<1x4x16xf16>) -> tensor<1x1x16xf16> {
    %input_low = const.Declare tensor<1x4x1xf16> = dense<-1.000000e+00> : tensor<1x4x1xf16>
    %input_high = const.Declare tensor<1x4x1xf16> = dense<1.000000e+00> : tensor<1x4x1xf16>
    %output_low = const.Declare tensor<1x4x1xf16> = dense<-1.000000e+00> : tensor<1x4x1xf16>
    %output_high = const.Declare tensor<1x4x1xf16> = dense<1.000000e+00> : tensor<1x4x1xf16>
    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x16xf16>, tensor<1x4x1xf16>, tensor<1x4x1xf16>, tensor<1x4x1xf16>, tensor<1x4x1xf16> -> tensor<1x4x16xf16>
    %1 = IE.Slice %0 [0, 2, 0] [1, 1, 16] : tensor<1x4x16xf16> to tensor<1x1x16xf16>
    return %1 : tensor<1x1x16xf16>

    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[ARG0]],
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[FQ]] [0, 2, 0] [1, 1, 16] : tensor<1x4x16xf16> to tensor<1x1x16xf16>
    // CHECK:       return [[SLICE]]
}

// -----

// CHECK-LABEL: @MoveSliceBeforeRMS
// CHECK-SAME:      [[ARG0:%[^:]+]]: tensor<1x4x16xf16>
func.func @MoveSliceBeforeRMS(%arg0: tensor<1x4x16xf16>) -> tensor<1x1x16xf16> {
    %gamma = const.Declare tensor<16xf16> = dense<1.000000e+00> : tensor<16xf16>
    %0 = IE.RMS(%arg0, %gamma) {eps = 9.9999997473787516E-6 : f64} : tensor<1x4x16xf16>, tensor<16xf16> -> tensor<1x4x16xf16>
    %1 = IE.Slice %0 [0, 2, 0] [1, 1, 16] : tensor<1x4x16xf16> to tensor<1x1x16xf16>
    return %1 : tensor<1x1x16xf16>

    // CHECK-DAG:   [[GAMMA:%.+]] = const.Declare tensor<16xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG0]] [0, 2, 0] [1, 1, 16] : tensor<1x4x16xf16> to tensor<1x1x16xf16>
    // CHECK:       [[RMS:%.+]] = IE.RMS([[SLICE]], [[GAMMA]]) {eps = 9.9999997473787516E-6 : f64} : tensor<1x1x16xf16>, tensor<16xf16> -> tensor<1x1x16xf16>
    // CHECK:       return [[RMS]]
}

// -----

// CHECK-LABEL: @NotMoveSliceBeforeRMSOnNormDim
// CHECK-SAME:      [[ARG0:%[^:]+]]: tensor<1x1x16xf16>
func.func @NotMoveSliceBeforeRMSOnNormDim(%arg0: tensor<1x1x16xf16>) -> tensor<1x1x4xf16> {
    %gamma = const.Declare tensor<16xf16> = dense<1.000000e+00> : tensor<16xf16>
    %0 = IE.RMS(%arg0, %gamma) {eps = 9.9999997473787516E-6 : f64} : tensor<1x1x16xf16>, tensor<16xf16> -> tensor<1x1x16xf16>
    %1 = IE.Slice %0 [0, 0, 0] [1, 1, 4] : tensor<1x1x16xf16> to tensor<1x1x4xf16>
    return %1 : tensor<1x1x4xf16>

    // CHECK:       [[RMS:%.+]] = IE.RMS([[ARG0]], {{%.+}}) {eps = 9.9999997473787516E-6 : f64} : tensor<1x1x16xf16>, tensor<16xf16> -> tensor<1x1x16xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[RMS]] [0, 0, 0] [1, 1, 4] : tensor<1x1x16xf16> to tensor<1x1x4xf16>
    // CHECK:       return [[SLICE]]
}

// -----

// CHECK-LABEL: @MoveSliceBeforeAffineReshape
// CHECK-SAME:      [[ARG0:%[^:]+]]: tensor<4x16xf16>
func.func @MoveSliceBeforeAffineReshape(%arg0: tensor<4x16xf16>) -> tensor<1x1x16xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2]], shape_value = [4, 1, 16]} : tensor<4x16xf16> -> tensor<4x1x16xf16>
    %1 = IE.Slice %0 [2, 0, 0] [1, 1, 16] : tensor<4x1x16xf16> to tensor<1x1x16xf16>
    return %1 : tensor<1x1x16xf16>

    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG0]] [2, 0] [1, 16] : tensor<4x16xf16> to tensor<1x16xf16>
    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[SLICE]]) {dim_mapping = {{.*}}, shape_value = [1, 1, 16]} : tensor<1x16xf16> -> tensor<1x1x16xf16>
    // CHECK:       return [[RESHAPE]]
}

// -----

// CHECK-LABEL: @NotMoveSliceBeforeAffineReshapeSplitAxis
// CHECK-SAME:      [[ARG0:%[^:]+]]: tensor<4x32xf16>
func.func @NotMoveSliceBeforeAffineReshapeSplitAxis(%arg0: tensor<4x32xf16>) -> tensor<4x1x8xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2]], shape_value = [4, 4, 8]} : tensor<4x32xf16> -> tensor<4x4x8xf16>
    %1 = IE.Slice %0 [0, 0, 0] [4, 1, 8] : tensor<4x4x8xf16> to tensor<4x1x8xf16>
    return %1 : tensor<4x1x8xf16>

    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[ARG0]]) {dim_mapping = {{.*}}, shape_value = [4, 4, 8]} : tensor<4x32xf16> -> tensor<4x4x8xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[RESHAPE]] [0, 0, 0] [4, 1, 8] : tensor<4x4x8xf16> to tensor<4x1x8xf16>
    // CHECK:       return [[SLICE]]
}

// -----

// CHECK-LABEL: @SwapSlicesDisjointAxes
// CHECK-SAME:      [[ARG0:%[^:]+]]: tensor<2x4x8xf16>
func.func @SwapSlicesDisjointAxes(%arg0: tensor<2x4x8xf16>) -> tensor<1x4x4xf16> {
    %0 = IE.Slice %arg0 [0, 0, 0] [2, 4, 4] : tensor<2x4x8xf16> to tensor<2x4x4xf16>
    %1 = IE.Slice %0 [1, 0, 0] [1, 4, 4] : tensor<2x4x4xf16> to tensor<1x4x4xf16>
    return %1 : tensor<1x4x4xf16>

    // CHECK:       [[LOW:%.+]] = IE.Slice [[ARG0]] [1, 0, 0] [1, 4, 8] : tensor<2x4x8xf16> to tensor<1x4x8xf16>
    // CHECK:       [[HIGH:%.+]] = IE.Slice [[LOW]] [0, 0, 0] [1, 4, 4] : tensor<1x4x8xf16> to tensor<1x4x4xf16>
    // CHECK:       return [[HIGH]]
}

// -----

// CHECK-LABEL: @MergeHorizontalSlices
// CHECK-SAME:      [[ARG0:%[^:]+]]: tensor<1x4x16xf16>
func.func @MergeHorizontalSlices(%arg0: tensor<1x4x16xf16>) -> (tensor<1x1x16xf16>, tensor<1x1x16xf16>) {
    %0 = IE.Slice %arg0 [0, 2, 0] [1, 1, 16] : tensor<1x4x16xf16> to tensor<1x1x16xf16>
    %1 = IE.Slice %arg0 [0, 2, 0] [1, 1, 16] : tensor<1x4x16xf16> to tensor<1x1x16xf16>
    return %0, %1 : tensor<1x1x16xf16>, tensor<1x1x16xf16>

    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG0]] [0, 2, 0] [1, 1, 16] : tensor<1x4x16xf16> to tensor<1x1x16xf16>
    // CHECK-NOT:   IE.Slice
    // CHECK:       return [[SLICE]], [[SLICE]]
}

// -----

// A residual Add whose second operand is a per-channel-scale Multiply with a rank-2 scale
// (down_proj output * dynamic-quant scale). Hoisting the tail Slice before the Add creates a
// Slice on the Multiply output; that Slice cannot be hoisted further because the scale operand
// has a lower rank than the Multiply output. The eltwise hoisting pattern must reject this case
// without leaving a partially-created Slice behind, otherwise the greedy driver never converges
// and the pass fails.

// CHECK-LABEL: @DownProjScaleResidualConverges
// CHECK-SAME:      [[A:%[^:]+]]: tensor<1x4x8xf16>
// CHECK-SAME:      [[B:%[^:]+]]: tensor<1x4x8xf16>
// CHECK-SAME:      [[FC:%[^:]+]]: tensor<1x4x8xf16>
// CHECK-SAME:      [[W:%[^:]+]]: tensor<1x8xf16>
// CHECK-SAME:      [[G:%[^:]+]]: tensor<8xf16>
func.func @DownProjScaleResidualConverges(%a: tensor<1x4x8xf16>, %b: tensor<1x4x8xf16>, %fc: tensor<1x4x8xf16>, %w: tensor<1x8xf16>, %g: tensor<8xf16>) -> (tensor<1x1x8xf16>, tensor<1x4x8xf16>) {
    %P = IE.Add(%a, %b) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x8xf16>, tensor<1x4x8xf16> -> tensor<1x4x8xf16>
    %rms0 = IE.RMS(%P, %g) {eps = 1.000000e-05 : f64} : tensor<1x4x8xf16>, tensor<8xf16> -> tensor<1x4x8xf16>
    %downMul = IE.Multiply(%fc, %w) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x8xf16>, tensor<1x8xf16> -> tensor<1x4x8xf16>
    %add1 = IE.Add(%P, %downMul) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x8xf16>, tensor<1x4x8xf16> -> tensor<1x4x8xf16>
    %rms1 = IE.RMS(%add1, %g) {eps = 1.000000e-05 : f64} : tensor<1x4x8xf16>, tensor<8xf16> -> tensor<1x4x8xf16>
    %slice = IE.Slice %rms1 [0, 3, 0] [1, 1, 8] : tensor<1x4x8xf16> to tensor<1x1x8xf16>
    return %slice, %rms0 : tensor<1x1x8xf16>, tensor<1x4x8xf16>

    // CHECK:       [[P:%.+]] = IE.Add([[A]], [[B]]) {{.*}} : tensor<1x4x8xf16>, tensor<1x4x8xf16> -> tensor<1x4x8xf16>
    // CHECK:       [[RMS0:%.+]] = IE.RMS([[P]], [[G]]) {{.*}} : tensor<1x4x8xf16>, tensor<8xf16> -> tensor<1x4x8xf16>
    // CHECK:       [[DOWNMUL:%.+]] = IE.Multiply([[FC]], [[W]]) {{.*}} : tensor<1x4x8xf16>, tensor<1x8xf16> -> tensor<1x4x8xf16>
    // CHECK-DAG:   [[SLICE_P:%.+]] = IE.Slice [[P]] [0, 3, 0] [1, 1, 8] : tensor<1x4x8xf16> to tensor<1x1x8xf16>
    // CHECK-DAG:   [[SLICE_DOWN:%.+]] = IE.Slice [[DOWNMUL]] [0, 3, 0] [1, 1, 8] : tensor<1x4x8xf16> to tensor<1x1x8xf16>
    // CHECK:       [[NEWADD:%.+]] = IE.Add([[SLICE_P]], [[SLICE_DOWN]]) {{.*}} : tensor<1x1x8xf16>, tensor<1x1x8xf16> -> tensor<1x1x8xf16>
    // CHECK:       [[RMS1:%.+]] = IE.RMS([[NEWADD]], [[G]]) {{.*}} : tensor<1x1x8xf16>, tensor<8xf16> -> tensor<1x1x8xf16>
    // CHECK:       return [[RMS1]], [[RMS0]]
}

// -----

// Aggregated end-to-end test: a full LLM prefill-tail decoder block (Phi-4 reasoning tail).
// A single last-token Slice at the network output must propagate backward through the whole
// residual/MLP tail.

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @Phi4ReasoningPrefillTail
// CHECK-SAME:      [[RESIDUAL:%[^:]+]]: tensor<1x1024x5120xf16>
// CHECK-SAME:      [[ATTN:%[^:]+]]: tensor<1x1024x5120xf16>
// CHECK-SAME:      [[O_PROJ_W:%[^:]+]]: tensor<5120x5120xsi4>
// CHECK-SAME:      [[O_PROJ_S:%[^:]+]]: tensor<5120x1xf16>
// CHECK-SAME:      [[POST_GAMMA:%[^:]+]]: tensor<1x1x5120xf32>
// CHECK-SAME:      [[GU_W:%[^:]+]]: tensor<35840x5120xsi4>
// CHECK-SAME:      [[GU_S:%[^:]+]]: tensor<35840x1xf16>
// CHECK-SAME:      [[AWQ_MUL:%[^:]+]]: tensor<1x1x17920xf32>
// CHECK-SAME:      [[DN_W:%[^:]+]]: tensor<5120x17920xsi4>
// CHECK-SAME:      [[DN_S:%[^:]+]]: tensor<5120x1xf16>
// CHECK-SAME:      [[FIN_GAMMA:%[^:]+]]: tensor<1x1x5120xf32>
func.func @Phi4ReasoningPrefillTail(%residual: tensor<1x1024x5120xf16>, %attn: tensor<1x1024x5120xf16>, %oProjW: tensor<5120x5120xsi4>, %oProjScale: tensor<5120x1xf16>, %postGamma: tensor<1x1x5120xf32>, %gateUpW: tensor<35840x5120xsi4>, %gateUpScale: tensor<35840x1xf16>, %awqMul: tensor<1x1x17920xf32>, %downW: tensor<5120x17920xsi4>, %downScale: tensor<5120x1xf16>, %finalGamma: tensor<1x1x5120xf32>) -> tensor<1x1x5120xf32> {
    // Attention output projection (o_proj).
    %oProjQC = IE.QuantizeCast(%oProjW) {dstElemType = !qElemType} : tensor<5120x5120xsi4> -> tensor<5120x5120x!qElemType>
    %oProjDQ = IE.DynamicDequantize(%oProjQC, %oProjScale) {dstElemType = f16} : tensor<5120x5120x!qElemType>, tensor<5120x1xf16> -> tensor<5120x5120xf16>
    %attnR = IE.AffineReshape(%attn) {dim_mapping = [[0], [0], [1]], shape_value = [1024, 5120]} : tensor<1x1024x5120xf16> -> tensor<1024x5120xf16>
    %oFC = IE.FullyConnected(%attnR, %oProjDQ) : tensor<1024x5120xf16>, tensor<5120x5120xf16> -> tensor<1024x5120xf16>
    %oR = IE.AffineReshape(%oFC) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1024, 5120]} : tensor<1024x5120xf16> -> tensor<1x1024x5120xf16>
    %add1 = IE.Add(%residual, %oR) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x5120xf16>, tensor<1x1024x5120xf16> -> tensor<1x1024x5120xf16>

    // Post-attention RMS normalization.
    %postCvt = IE.Convert(%postGamma) {dstElemType = f16} : tensor<1x1x5120xf32> -> tensor<1x1x5120xf16>
    %postG = IE.AffineReshape(%postCvt) {dim_mapping = [[0], [0], [0]], shape_value = [5120]} : tensor<1x1x5120xf16> -> tensor<5120xf16>
    %rms1 = IE.RMS(%add1, %postG) {eps = 9.9999997473787516E-6 : f64} : tensor<1x1024x5120xf16>, tensor<5120xf16> -> tensor<1x1024x5120xf16>

    // Gate/up projection, split, Swish and AWQ multiplies.
    %guQC = IE.QuantizeCast(%gateUpW) {dstElemType = !qElemType} : tensor<35840x5120xsi4> -> tensor<35840x5120x!qElemType>
    %guDQ = IE.DynamicDequantize(%guQC, %gateUpScale) {dstElemType = f16} : tensor<35840x5120x!qElemType>, tensor<35840x1xf16> -> tensor<35840x5120xf16>
    %rms1R = IE.AffineReshape(%rms1) {dim_mapping = [[0], [0], [1]], shape_value = [1024, 5120]} : tensor<1x1024x5120xf16> -> tensor<1024x5120xf16>
    %guFC = IE.FullyConnected(%rms1R, %guDQ) : tensor<1024x5120xf16>, tensor<35840x5120xf16> -> tensor<1024x35840xf16>
    %guR = IE.AffineReshape(%guFC) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1024, 35840]} : tensor<1024x35840xf16> -> tensor<1x1024x35840xf16>
    %gate = IE.Slice %guR [0, 0, 0] [1, 1024, 17920] : tensor<1x1024x35840xf16> to tensor<1x1024x17920xf16>
    %swish = IE.Swish(%gate) {beta_value = 1.000000e+00 : f64} : tensor<1x1024x17920xf16> -> tensor<1x1024x17920xf16>
    %awqCvt = IE.Convert(%awqMul) {dstElemType = f16} : tensor<1x1x17920xf32> -> tensor<1x1x17920xf16>
    %awq = IE.Multiply(%swish, %awqCvt) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x17920xf16>, tensor<1x1x17920xf16> -> tensor<1x1024x17920xf16>
    %up = IE.Slice %guR [0, 0, 17920] [1, 1024, 17920] : tensor<1x1024x35840xf16> to tensor<1x1024x17920xf16>
    %gated = IE.Multiply(%up, %awq) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x17920xf16>, tensor<1x1024x17920xf16> -> tensor<1x1024x17920xf16>

    // Down projection.
    %dnQC = IE.QuantizeCast(%downW) {dstElemType = !qElemType} : tensor<5120x17920xsi4> -> tensor<5120x17920x!qElemType>
    %dnDQ = IE.DynamicDequantize(%dnQC, %downScale) {dstElemType = f16} : tensor<5120x17920x!qElemType>, tensor<5120x1xf16> -> tensor<5120x17920xf16>
    %gatedR = IE.AffineReshape(%gated) {dim_mapping = [[0], [0], [1]], shape_value = [1024, 17920]} : tensor<1x1024x17920xf16> -> tensor<1024x17920xf16>
    %dnFC = IE.FullyConnected(%gatedR, %dnDQ) : tensor<1024x17920xf16>, tensor<5120x17920xf16> -> tensor<1024x5120xf16>
    %dnR = IE.AffineReshape(%dnFC) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1024, 5120]} : tensor<1024x5120xf16> -> tensor<1x1024x5120xf16>
    %add2 = IE.Add(%add1, %dnR) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x5120xf16>, tensor<1x1024x5120xf16> -> tensor<1x1024x5120xf16>

    // Final RMS normalization and last-token extraction.
    %finalCvt = IE.Convert(%finalGamma) {dstElemType = f16} : tensor<1x1x5120xf32> -> tensor<1x1x5120xf16>
    %finalG = IE.AffineReshape(%finalCvt) {dim_mapping = [[0], [0], [0]], shape_value = [5120]} : tensor<1x1x5120xf16> -> tensor<5120xf16>
    %rms2 = IE.RMS(%add2, %finalG) {eps = 9.9999997473787516E-6 : f64} : tensor<1x1024x5120xf16>, tensor<5120xf16> -> tensor<1x1024x5120xf16>
    %slice = IE.Slice %rms2 [0, 1023, 0] [1, 1, 5120] : tensor<1x1024x5120xf16> to tensor<1x1x5120xf16>
    %result = IE.Convert(%slice) {dstElemType = f32} : tensor<1x1x5120xf16> -> tensor<1x1x5120xf32>
    return %result : tensor<1x1x5120xf32>

    // CHECK:       [[O_DQ:%.+]] = IE.DynamicDequantize
    // CHECK:       [[POST_G:%.+]] = IE.AffineReshape
    // CHECK:       [[GU_DQ:%.+]] = IE.DynamicDequantize
    // CHECK:       [[AWQ_CVT:%.+]] = IE.Convert([[AWQ_MUL]]) {dstElemType = f16} : tensor<1x1x17920xf32> -> tensor<1x1x17920xf16>
    // CHECK:       [[DN_DQ:%.+]] = IE.DynamicDequantize
    // CHECK:       [[FIN_G:%.+]] = IE.AffineReshape

    // CHECK:       [[SLICE_RES:%.+]] = IE.Slice [[RESIDUAL]] [0, 1023, 0] [1, 1, 5120] : tensor<1x1024x5120xf16> to tensor<1x1x5120xf16>
    // CHECK:       [[SLICE_ATTN:%.+]] = IE.Slice [[ATTN]] [0, 1023, 0] [1, 1, 5120] : tensor<1x1024x5120xf16> to tensor<1x1x5120xf16>
    // CHECK:       [[ATTN_G:%.+]] = IE.AffineReshape([[SLICE_ATTN]])
    // CHECK:       [[O_FC:%.+]] = IE.FullyConnected([[ATTN_G]], [[O_DQ]]) : tensor<1x5120xf16>, tensor<5120x5120xf16> -> tensor<1x5120xf16>
    // CHECK:       [[O_R:%.+]] = IE.AffineReshape([[O_FC]])
    // CHECK:       [[ADD1:%.+]] = IE.Add([[SLICE_RES]], [[O_R]]) {{.*}} : tensor<1x1x5120xf16>, tensor<1x1x5120xf16> -> tensor<1x1x5120xf16>
    // CHECK:       [[RMS1:%.+]] = IE.RMS([[ADD1]], [[POST_G]]) {{.*}} : tensor<1x1x5120xf16>, tensor<5120xf16> -> tensor<1x1x5120xf16>
    // CHECK:       [[RMS1_G:%.+]] = IE.AffineReshape([[RMS1]])
    // CHECK:       [[GU_FC:%.+]] = IE.FullyConnected([[RMS1_G]], [[GU_DQ]]) : tensor<1x5120xf16>, tensor<35840x5120xf16> -> tensor<1x35840xf16>
    // CHECK:       [[GU_R:%.+]] = IE.AffineReshape([[GU_FC]])
    // CHECK:       [[UP:%.+]] = IE.Slice [[GU_R]] [0, 0, 17920] [1, 1, 17920] : tensor<1x1x35840xf16> to tensor<1x1x17920xf16>
    // CHECK:       [[GATE:%.+]] = IE.Slice [[GU_R]] [0, 0, 0] [1, 1, 17920] : tensor<1x1x35840xf16> to tensor<1x1x17920xf16>
    // CHECK:       [[SWISH:%.+]] = IE.Swish([[GATE]]) {{.*}} : tensor<1x1x17920xf16> -> tensor<1x1x17920xf16>
    // CHECK:       [[AWQ:%.+]] = IE.Multiply([[SWISH]], [[AWQ_CVT]]) {{.*}} : tensor<1x1x17920xf16>, tensor<1x1x17920xf16> -> tensor<1x1x17920xf16>
    // CHECK:       [[GATED:%.+]] = IE.Multiply([[UP]], [[AWQ]]) {{.*}} : tensor<1x1x17920xf16>, tensor<1x1x17920xf16> -> tensor<1x1x17920xf16>
    // CHECK:       [[GATED_G:%.+]] = IE.AffineReshape([[GATED]])
    // CHECK:       [[DN_FC:%.+]] = IE.FullyConnected([[GATED_G]], [[DN_DQ]]) : tensor<1x17920xf16>, tensor<5120x17920xf16> -> tensor<1x5120xf16>
    // CHECK:       [[DN_R:%.+]] = IE.AffineReshape([[DN_FC]])
    // CHECK:       [[ADD2:%.+]] = IE.Add([[ADD1]], [[DN_R]]) {{.*}} : tensor<1x1x5120xf16>, tensor<1x1x5120xf16> -> tensor<1x1x5120xf16>
    // CHECK:       [[RMS2:%.+]] = IE.RMS([[ADD2]], [[FIN_G]]) {{.*}} : tensor<1x1x5120xf16>, tensor<5120xf16> -> tensor<1x1x5120xf16>
    // CHECK:       [[RESULT:%.+]] = IE.Convert([[RMS2]]) {dstElemType = f32} : tensor<1x1x5120xf16> -> tensor<1x1x5120xf32>
    // CHECK:       return [[RESULT]] : tensor<1x1x5120xf32>
}

// -----

// CHECK-LABEL: @SplitFullyConnectedQKVWithBias
// CHECK-SAME:      [[ARG0:%[^:]+]]: tensor<4x8xf16>
func.func @SplitFullyConnectedQKVWithBias(%arg0: tensor<4x8xf16>) -> (tensor<4x1x8xf16>, tensor<4x1x8xf16>, tensor<4x1x8xf16>) {
    %weights = const.Declare tensor<24x8xf16> = dense<1.000000e+00> : tensor<24x8xf16>
    %bias = const.Declare tensor<24xf16> = dense<1.000000e+00> : tensor<24xf16>
    %fc = IE.FullyConnected(%arg0, %weights, %bias) : tensor<4x8xf16>, tensor<24x8xf16>, tensor<24xf16> -> tensor<4x24xf16>
    %reshape = IE.AffineReshape(%fc) {dim_mapping = [[0], [1, 2]], shape_value = [4, 3, 8]} : tensor<4x24xf16> -> tensor<4x3x8xf16>
    %q = IE.Slice %reshape [0, 0, 0] [4, 1, 8] : tensor<4x3x8xf16> to tensor<4x1x8xf16>
    %k = IE.Slice %reshape [0, 1, 0] [4, 1, 8] : tensor<4x3x8xf16> to tensor<4x1x8xf16>
    %v = IE.Slice %reshape [0, 2, 0] [4, 1, 8] : tensor<4x3x8xf16> to tensor<4x1x8xf16>
    return %q, %k, %v : tensor<4x1x8xf16>, tensor<4x1x8xf16>, tensor<4x1x8xf16>

    // Slices on the constant weights/bias fold directly into sliced const.Declare ops.
    // CHECK-DAG:   [[W_Q:%.+]] = const.Declare tensor<8x8xf16> = dense<1.000000e+00> : tensor<24x8xf16>, [#const.SubView<[0, 0], [8, 8]>]
    // CHECK-DAG:   [[B_Q:%.+]] = const.Declare tensor<8xf16> = dense<1.000000e+00> : tensor<24xf16>, [#const.SubView<[0], [8]>]
    // CHECK-DAG:   [[W_K:%.+]] = const.Declare tensor<8x8xf16> = dense<1.000000e+00> : tensor<24x8xf16>, [#const.SubView<[8, 0], [8, 8]>]
    // CHECK-DAG:   [[B_K:%.+]] = const.Declare tensor<8xf16> = dense<1.000000e+00> : tensor<24xf16>, [#const.SubView<[8], [8]>]
    // CHECK-DAG:   [[W_V:%.+]] = const.Declare tensor<8x8xf16> = dense<1.000000e+00> : tensor<24x8xf16>, [#const.SubView<[16, 0], [8, 8]>]
    // CHECK-DAG:   [[B_V:%.+]] = const.Declare tensor<8xf16> = dense<1.000000e+00> : tensor<24xf16>, [#const.SubView<[16], [8]>]

    // The original wide FullyConnected/AffineReshape/Slice chain must not survive.
    // CHECK-NOT:   IE.Slice

    // CHECK-DAG:   [[FC_Q:%.+]] = IE.FullyConnected([[ARG0]], [[W_Q]], [[B_Q]]) : tensor<4x8xf16>, tensor<8x8xf16>, tensor<8xf16> -> tensor<4x8xf16>
    // CHECK-DAG:   [[Q:%.+]] = IE.AffineReshape([[FC_Q]]){{.*}}: tensor<4x8xf16> -> tensor<4x1x8xf16>

    // CHECK-DAG:   [[FC_K:%.+]] = IE.FullyConnected([[ARG0]], [[W_K]], [[B_K]]) : tensor<4x8xf16>, tensor<8x8xf16>, tensor<8xf16> -> tensor<4x8xf16>
    // CHECK-DAG:   [[K:%.+]] = IE.AffineReshape([[FC_K]]){{.*}}: tensor<4x8xf16> -> tensor<4x1x8xf16>

    // CHECK-DAG:   [[FC_V:%.+]] = IE.FullyConnected([[ARG0]], [[W_V]], [[B_V]]) : tensor<4x8xf16>, tensor<8x8xf16>, tensor<8xf16> -> tensor<4x8xf16>
    // CHECK-DAG:   [[V:%.+]] = IE.AffineReshape([[FC_V]]){{.*}}: tensor<4x8xf16> -> tensor<4x1x8xf16>

    // CHECK:       return [[Q]], [[K]], [[V]] : tensor<4x1x8xf16>, tensor<4x1x8xf16>, tensor<4x1x8xf16>
}

// -----

//
// Positive: fused QKV FullyConnected (non-constant weights/bias) decomposed via plain Reshape.
// Pattern: FullyConnected(act, W[48x32], bias[48]) -> Reshape([4,3,16]) -> Slice x3 (axis=1)
//

// CHECK-LABEL: @SplitFullyConnectedQKVPlainReshape
// CHECK-SAME:      [[ACT:%.+]]: tensor<4x32xf16>, [[W:%.+]]: tensor<48x32xf16>, [[B:%.+]]: tensor<48xf16>
func.func @SplitFullyConnectedQKVPlainReshape(%act: tensor<4x32xf16>, %w: tensor<48x32xf16>, %b: tensor<48xf16>)
        -> (tensor<4x1x16xf16>, tensor<4x1x16xf16>, tensor<4x1x16xf16>) {
    %fc = IE.FullyConnected(%act, %w, %b) : tensor<4x32xf16>, tensor<48x32xf16>, tensor<48xf16> -> tensor<4x48xf16>
    %rs = IE.Reshape(%fc) {shape_value = [4, 3, 16]} : tensor<4x48xf16> -> tensor<4x3x16xf16>
    %q = IE.Slice %rs [0, 0, 0] [4, 1, 16] : tensor<4x3x16xf16> to tensor<4x1x16xf16>
    %k = IE.Slice %rs [0, 1, 0] [4, 1, 16] : tensor<4x3x16xf16> to tensor<4x1x16xf16>
    %v = IE.Slice %rs [0, 2, 0] [4, 1, 16] : tensor<4x3x16xf16> to tensor<4x1x16xf16>
    return %q, %k, %v : tensor<4x1x16xf16>, tensor<4x1x16xf16>, tensor<4x1x16xf16>

    // The original wide FullyConnected/Reshape/Slice chain must not survive.
    // CHECK-NOT:   IE.FullyConnected([[ACT]], [[W]], [[B]])

    // CHECK-DAG:   [[W_Q:%.+]] = IE.Slice [[W]] [0, 0] [16, 32] : tensor<48x32xf16> to tensor<16x32xf16>
    // CHECK-DAG:   [[B_Q:%.+]] = IE.Slice [[B]] [0] [16] : tensor<48xf16> to tensor<16xf16>
    // CHECK-DAG:   [[FC_Q:%.+]] = IE.FullyConnected([[ACT]], [[W_Q]], [[B_Q]]) : tensor<4x32xf16>, tensor<16x32xf16>, tensor<16xf16> -> tensor<4x16xf16>
    // CHECK-DAG:   [[Q:%.+]] = IE.Reshape([[FC_Q]]) {shape_value = [4, 1, 16]} : tensor<4x16xf16> -> tensor<4x1x16xf16>

    // CHECK-DAG:   [[W_K:%.+]] = IE.Slice [[W]] [16, 0] [16, 32] : tensor<48x32xf16> to tensor<16x32xf16>
    // CHECK-DAG:   [[B_K:%.+]] = IE.Slice [[B]] [16] [16] : tensor<48xf16> to tensor<16xf16>
    // CHECK-DAG:   [[FC_K:%.+]] = IE.FullyConnected([[ACT]], [[W_K]], [[B_K]]) : tensor<4x32xf16>, tensor<16x32xf16>, tensor<16xf16> -> tensor<4x16xf16>
    // CHECK-DAG:   [[K:%.+]] = IE.Reshape([[FC_K]]) {shape_value = [4, 1, 16]} : tensor<4x16xf16> -> tensor<4x1x16xf16>

    // CHECK-DAG:   [[W_V:%.+]] = IE.Slice [[W]] [32, 0] [16, 32] : tensor<48x32xf16> to tensor<16x32xf16>
    // CHECK-DAG:   [[B_V:%.+]] = IE.Slice [[B]] [32] [16] : tensor<48xf16> to tensor<16xf16>
    // CHECK-DAG:   [[FC_V:%.+]] = IE.FullyConnected([[ACT]], [[W_V]], [[B_V]]) : tensor<4x32xf16>, tensor<16x32xf16>, tensor<16xf16> -> tensor<4x16xf16>
    // CHECK-DAG:   [[V:%.+]] = IE.Reshape([[FC_V]]) {shape_value = [4, 1, 16]} : tensor<4x16xf16> -> tensor<4x1x16xf16>

    // CHECK:       return [[Q]], [[K]], [[V]] : tensor<4x1x16xf16>, tensor<4x1x16xf16>, tensor<4x1x16xf16>
}

// -----

//
// Positive: fused QKV FullyConnected (non-constant weights/bias) decomposed via AffineReshape.
// Pattern: FullyConnected(act, W[48x32], bias[48]) -> AffineReshape([4,3,16]) -> Slice x3 (axis=1)
//

// CHECK-LABEL: @SplitFullyConnectedQKVAffineReshape
// CHECK-SAME:      [[ACT:%.+]]: tensor<4x32xf16>, [[W:%.+]]: tensor<48x32xf16>, [[B:%.+]]: tensor<48xf16>
func.func @SplitFullyConnectedQKVAffineReshape(%act: tensor<4x32xf16>, %w: tensor<48x32xf16>, %b: tensor<48xf16>)
        -> (tensor<4x1x16xf16>, tensor<4x1x16xf16>, tensor<4x1x16xf16>) {
    %fc = IE.FullyConnected(%act, %w, %b) : tensor<4x32xf16>, tensor<48x32xf16>, tensor<48xf16> -> tensor<4x48xf16>
    %rs = IE.AffineReshape(%fc) {dim_mapping = [[0], [1, 2]], shape_value = [4, 3, 16]} : tensor<4x48xf16> -> tensor<4x3x16xf16>
    %q = IE.Slice %rs [0, 0, 0] [4, 1, 16] : tensor<4x3x16xf16> to tensor<4x1x16xf16>
    %k = IE.Slice %rs [0, 1, 0] [4, 1, 16] : tensor<4x3x16xf16> to tensor<4x1x16xf16>
    %v = IE.Slice %rs [0, 2, 0] [4, 1, 16] : tensor<4x3x16xf16> to tensor<4x1x16xf16>
    return %q, %k, %v : tensor<4x1x16xf16>, tensor<4x1x16xf16>, tensor<4x1x16xf16>

    // The original wide FullyConnected/AffineReshape/Slice chain must not survive.
    // CHECK-NOT:   IE.FullyConnected([[ACT]], [[W]], [[B]])

    // CHECK-DAG:   [[W_Q:%.+]] = IE.Slice [[W]] [0, 0] [16, 32] : tensor<48x32xf16> to tensor<16x32xf16>
    // CHECK-DAG:   [[B_Q:%.+]] = IE.Slice [[B]] [0] [16] : tensor<48xf16> to tensor<16xf16>
    // CHECK-DAG:   [[FC_Q:%.+]] = IE.FullyConnected([[ACT]], [[W_Q]], [[B_Q]]) : tensor<4x32xf16>, tensor<16x32xf16>, tensor<16xf16> -> tensor<4x16xf16>
    // CHECK-DAG:   [[Q:%.+]] = IE.AffineReshape([[FC_Q]]){{.*}}: tensor<4x16xf16> -> tensor<4x1x16xf16>

    // CHECK-DAG:   [[W_K:%.+]] = IE.Slice [[W]] [16, 0] [16, 32] : tensor<48x32xf16> to tensor<16x32xf16>
    // CHECK-DAG:   [[B_K:%.+]] = IE.Slice [[B]] [16] [16] : tensor<48xf16> to tensor<16xf16>
    // CHECK-DAG:   [[FC_K:%.+]] = IE.FullyConnected([[ACT]], [[W_K]], [[B_K]]) : tensor<4x32xf16>, tensor<16x32xf16>, tensor<16xf16> -> tensor<4x16xf16>
    // CHECK-DAG:   [[K:%.+]] = IE.AffineReshape([[FC_K]]){{.*}}: tensor<4x16xf16> -> tensor<4x1x16xf16>

    // CHECK-DAG:   [[W_V:%.+]] = IE.Slice [[W]] [32, 0] [16, 32] : tensor<48x32xf16> to tensor<16x32xf16>
    // CHECK-DAG:   [[B_V:%.+]] = IE.Slice [[B]] [32] [16] : tensor<48xf16> to tensor<16xf16>
    // CHECK-DAG:   [[FC_V:%.+]] = IE.FullyConnected([[ACT]], [[W_V]], [[B_V]]) : tensor<4x32xf16>, tensor<16x32xf16>, tensor<16xf16> -> tensor<4x16xf16>
    // CHECK-DAG:   [[V:%.+]] = IE.AffineReshape([[FC_V]]){{.*}}: tensor<4x16xf16> -> tensor<4x1x16xf16>

    // CHECK:       return [[Q]], [[K]], [[V]] : tensor<4x1x16xf16>, tensor<4x1x16xf16>, tensor<4x1x16xf16>
}

// -----

//
// Negative: Slice restricts the batch axis (axis=0) of the reshaped output — the output-channel
// interior-axis guard rejects splitting the FullyConnected, since axis 0 is never the output
// channel dim.
//

// CHECK-LABEL: @NotSplitFullyConnectedOnBatchAxis
// CHECK-SAME:      [[ACT:%.+]]: tensor<4x32xf16>, [[W:%.+]]: tensor<12x32xf16>, [[B:%.+]]: tensor<12xf16>
func.func @NotSplitFullyConnectedOnBatchAxis(%act: tensor<4x32xf16>, %w: tensor<12x32xf16>, %b: tensor<12xf16>)
        -> (tensor<2x3x4xf16>, tensor<2x3x4xf16>) {
    %fc = IE.FullyConnected(%act, %w, %b) : tensor<4x32xf16>, tensor<12x32xf16>, tensor<12xf16> -> tensor<4x12xf16>
    %rs = IE.Reshape(%fc) {shape_value = [4, 3, 4]} : tensor<4x12xf16> -> tensor<4x3x4xf16>
    %s0 = IE.Slice %rs [0, 0, 0] [2, 3, 4] : tensor<4x3x4xf16> to tensor<2x3x4xf16>
    %s1 = IE.Slice %rs [2, 0, 0] [2, 3, 4] : tensor<4x3x4xf16> to tensor<2x3x4xf16>
    return %s0, %s1 : tensor<2x3x4xf16>, tensor<2x3x4xf16>

    // CHECK:       [[FC:%.+]] = IE.FullyConnected([[ACT]], [[W]], [[B]]) : tensor<4x32xf16>, tensor<12x32xf16>, tensor<12xf16> -> tensor<4x12xf16>
    // CHECK:       [[RS:%.+]] = IE.Reshape([[FC]]) {shape_value = [4, 3, 4]} : tensor<4x12xf16> -> tensor<4x3x4xf16>
    // CHECK-DAG:   [[S0:%.+]] = IE.Slice [[RS]] [0, 0, 0] [2, 3, 4] : tensor<4x3x4xf16> to tensor<2x3x4xf16>
    // CHECK-DAG:   [[S1:%.+]] = IE.Slice [[RS]] [2, 0, 0] [2, 3, 4] : tensor<4x3x4xf16> to tensor<2x3x4xf16>
    // CHECK:       return [[S0]], [[S1]] : tensor<2x3x4xf16>, tensor<2x3x4xf16>
}

// -----

//
// Negative: Slice restricts the innermost axis while a non-trivial ("num_heads"-like) dim of the
// output-channel group precedes it, so the flattened FullyConnected output-channel slice would be
// non-contiguous. The contiguity guard rejects the split.
//

// CHECK-LABEL: @NotSplitFullyConnectedNonContiguousAxis
// CHECK-SAME:      [[ACT:%.+]]: tensor<4x32xf16>, [[W:%.+]]: tensor<48x32xf16>, [[B:%.+]]: tensor<48xf16>
func.func @NotSplitFullyConnectedNonContiguousAxis(%act: tensor<4x32xf16>, %w: tensor<48x32xf16>, %b: tensor<48xf16>)
        -> (tensor<4x3x8xf16>, tensor<4x3x8xf16>) {
    %fc = IE.FullyConnected(%act, %w, %b) : tensor<4x32xf16>, tensor<48x32xf16>, tensor<48xf16> -> tensor<4x48xf16>
    %rs = IE.Reshape(%fc) {shape_value = [4, 3, 16]} : tensor<4x48xf16> -> tensor<4x3x16xf16>
    %s0 = IE.Slice %rs [0, 0, 0] [4, 3, 8] : tensor<4x3x16xf16> to tensor<4x3x8xf16>
    %s1 = IE.Slice %rs [0, 0, 8] [4, 3, 8] : tensor<4x3x16xf16> to tensor<4x3x8xf16>
    return %s0, %s1 : tensor<4x3x8xf16>, tensor<4x3x8xf16>

    // CHECK:       [[FC:%.+]] = IE.FullyConnected([[ACT]], [[W]], [[B]]) : tensor<4x32xf16>, tensor<48x32xf16>, tensor<48xf16> -> tensor<4x48xf16>
    // CHECK:       [[RS:%.+]] = IE.Reshape([[FC]]) {shape_value = [4, 3, 16]} : tensor<4x48xf16> -> tensor<4x3x16xf16>
    // CHECK-DAG:   [[S0:%.+]] = IE.Slice [[RS]] [0, 0, 0] [4, 3, 8] : tensor<4x3x16xf16> to tensor<4x3x8xf16>
    // CHECK-DAG:   [[S1:%.+]] = IE.Slice [[RS]] [0, 0, 8] [4, 3, 8] : tensor<4x3x16xf16> to tensor<4x3x8xf16>
    // CHECK:       return [[S0]], [[S1]] : tensor<4x3x8xf16>, tensor<4x3x8xf16>
}

// -----

// Stacked projection block (base + a two-FC A -> B chain) ending in a single-row Slice. Hoisting
// the Slice onto FC B must re-enqueue FC A in the same iteration so the whole chain converges.
// CHECK-LABEL: @PropagateSliceThroughStackedFcBlock
// CHECK-SAME:      [[RES:%[^:]+]]: tensor<1x4x8xf16>
// CHECK-SAME:      [[ACT:%[^:]+]]: tensor<1x4x8xf16>
func.func @PropagateSliceThroughStackedFcBlock(
        %res: tensor<1x4x8xf16>, %act: tensor<1x4x8xf16>, %baseW: tensor<8x8xf16>,
        %aW: tensor<2x8xf16>, %bW: tensor<8x2xf16>, %scale: tensor<1x1x1xf16>,
        %gamma: tensor<8xf16>) -> tensor<1x1x8xf16> {
    %actRs = IE.AffineReshape(%act) {dim_mapping = [[0], [0], [1]], shape_value = [4, 8]} : tensor<1x4x8xf16> -> tensor<4x8xf16>
    %base = IE.FullyConnected(%actRs, %baseW) : tensor<4x8xf16>, tensor<8x8xf16> -> tensor<4x8xf16>
    %baseRs = IE.AffineReshape(%base) {dim_mapping = [[0, 1], [2]], shape_value = [1, 4, 8]} : tensor<4x8xf16> -> tensor<1x4x8xf16>
    %fcA = IE.FullyConnected(%actRs, %aW) : tensor<4x8xf16>, tensor<2x8xf16> -> tensor<4x2xf16>
    %fcB = IE.FullyConnected(%fcA, %bW) : tensor<4x2xf16>, tensor<8x2xf16> -> tensor<4x8xf16>
    %fcBRs = IE.AffineReshape(%fcB) {dim_mapping = [[0, 1], [2]], shape_value = [1, 4, 8]} : tensor<4x8xf16> -> tensor<1x4x8xf16>
    %scaled = IE.Multiply(%fcBRs, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x8xf16>, tensor<1x1x1xf16> -> tensor<1x4x8xf16>
    %proj = IE.Add(%baseRs, %scaled) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x8xf16>, tensor<1x4x8xf16> -> tensor<1x4x8xf16>
    %resAdd = IE.Add(%res, %proj) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x8xf16>, tensor<1x4x8xf16> -> tensor<1x4x8xf16>
    %rms = IE.RMS(%resAdd, %gamma) {eps = 9.9999997473787516E-6 : f64} : tensor<1x4x8xf16>, tensor<8xf16> -> tensor<1x4x8xf16>
    %slice = IE.Slice %rms [0, 3, 0] [1, 1, 8] : tensor<1x4x8xf16> to tensor<1x1x8xf16>
    return %slice : tensor<1x1x8xf16>

    // Every FC runs on a 1-row input instead of 4 rows.
    // CHECK:       [[RES_SLICE:%.+]] = IE.Slice [[RES]] [0, 3, 0] [1, 1, 8] : tensor<1x4x8xf16> to tensor<1x1x8xf16>
    // CHECK:       [[ACT_SLICE:%.+]] = IE.Slice [[ACT]] [0, 3, 0] [1, 1, 8] : tensor<1x4x8xf16> to tensor<1x1x8xf16>
    // CHECK:       [[ACT_RS:%.+]] = IE.AffineReshape([[ACT_SLICE]])
    // CHECK-SAME:      tensor<1x1x8xf16> -> tensor<1x8xf16>
    // CHECK:       [[BASE:%.+]] = IE.FullyConnected([[ACT_RS]], {{%.+}}) : tensor<1x8xf16>, tensor<8x8xf16> -> tensor<1x8xf16>
    // CHECK:       [[BASE_RS:%.+]] = IE.AffineReshape([[BASE]])
    // CHECK:       [[FC_A:%.+]] = IE.FullyConnected([[ACT_RS]], {{%.+}}) : tensor<1x8xf16>, tensor<2x8xf16> -> tensor<1x2xf16>
    // CHECK:       [[FC_B:%.+]] = IE.FullyConnected([[FC_A]], {{%.+}}) : tensor<1x2xf16>, tensor<8x2xf16> -> tensor<1x8xf16>
    // CHECK:       [[FC_B_RS:%.+]] = IE.AffineReshape([[FC_B]])
    // CHECK:       [[SCALED:%.+]] = IE.Multiply([[FC_B_RS]], {{%.+}})
    // CHECK:       [[PROJ:%.+]] = IE.Add([[BASE_RS]], [[SCALED]])
    // CHECK:       [[RES_ADD:%.+]] = IE.Add([[RES_SLICE]], [[PROJ]])
    // CHECK:       [[RMS:%.+]] = IE.RMS([[RES_ADD]], {{%.+}}) {eps = 9.9999997473787516E-6 : f64} : tensor<1x1x8xf16>, tensor<8xf16> -> tensor<1x1x8xf16>
    // CHECK:       return [[RMS]] : tensor<1x1x8xf16>
}
