//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --swap-operation-with-gather %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

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
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<645632x224xi4>
// CHECK-SAME:      [[INPUT_3:%.+]]: tensor<1x1024xsi32>
func.func @NotMoveDynamicDQWithZPAfterGather(%arg0: tensor<645632x224x!qElemType>, %arg1: tensor<645632x1xf16>, %arg2: tensor<645632x224xi4>, %arg3: tensor<1x1024xsi32>) -> tensor<1x1024x224xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1, %arg2) {dstElemType = f16} : tensor<645632x224x!qElemType>, tensor<645632x1xf16>, tensor<645632x224xi4> -> tensor<645632x224xf16>
    %1 = IE.Gather(%0, %arg3) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<645632x224xf16>, tensor<1x1024xsi32> -> tensor<1x1024x224xf16>

    return %1 : tensor<1x1024x224xf16>

    // CHECK:       [[DQ:%.+]] = IE.DynamicDequantize([[INPUT_0]], [[INPUT_1]], [[INPUT_2]]) {dstElemType = f16} : tensor<645632x224x!qElemType>, tensor<645632x1xf16>, tensor<645632x224xi4> -> tensor<645632x224xf16>
    // CHECK:       [[GATHER:%.+]] = IE.Gather([[DQ]], [[INPUT_3]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<645632x224xf16>, tensor<1x1024xsi32> -> tensor<1x1024x224xf16>

    // CHECK:       return [[GATHER]] : tensor<1x1024x224xf16>
}
