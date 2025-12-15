//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-expand --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.01>

// CHECK-LABEL: @ExpandEndF8E4M3FN
// CHECK-SAME:      [[INPUT:%.+]]: memref<1x3x4x4x!qElemType>
// CHECK-SAME: -> memref<1x8x4x4x!qElemType>
func.func @ExpandEndF8E4M3FN(%arg0: memref<1x3x4x4x!qElemType>) -> memref<1x8x4x4x!qElemType> {
    %0 = memref.alloc() : memref<1x8x4x4x!qElemType>
    %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 5, 0, 0]} inputs(%arg0 : memref<1x3x4x4x!qElemType>) outputs(%0 : memref<1x8x4x4x!qElemType>) -> memref<1x8x4x4x!qElemType>

    return %1 : memref<1x8x4x4x!qElemType>

    // CHECK-DAG:    [[CST_END:%.+]] = const.Declare memref<1x5x4x4x!qElemType> = dense<0.000000e+00> : tensor<80xf8E4M3FN>, [#const.Reshape<[1, 5, 4, 4]>]
    // CHECK:        [[OUT_BUFFER:%.+]] = memref.alloc() : memref<1x8x4x4x!qElemType>

    // CHECK:        [[VIEW_0:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 0, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:       : memref<1x8x4x4x!qElemType> to memref<1x3x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:        [[COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:       inputs([[INPUT]] : memref<1x3x4x4x!qElemType>) outputs([[VIEW_0]] : memref<1x3x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>) -> memref<1x3x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>

    // CHECK:        [[VIEW_1:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 3, 0, 0] [1, 5, 4, 4]
    // CHECK-SAME:       : memref<1x8x4x4x!qElemType> to memref<1x5x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:        [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:       inputs([[CST_END]] : memref<1x5x4x4x!qElemType>) outputs([[VIEW_1]] : memref<1x5x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>) -> memref<1x5x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>

    // CHECK:        [[OUT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:       inputs([[COPY_0]], [[COPY_1]] : memref<1x3x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>, memref<1x5x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>) outputs(%alloc : memref<1x8x4x4x!qElemType>) -> memref<1x8x4x4x!qElemType>

    // CHECK: return [[OUT]] : memref<1x8x4x4x!qElemType>
}

// -----

!qElemType = !quant.uniform<f8E5M2:f16, 0.01>

// CHECK-LABEL: @ExpandBeginF8E5M2
// CHECK-SAME:      [[INPUT:%.+]]: memref<1x3x4x4x!qElemType>
// CHECK-SAME: -> memref<1x8x4x4x!qElemType>
func.func @ExpandBeginF8E5M2(%arg0: memref<1x3x4x4x!qElemType>) -> memref<1x8x4x4x!qElemType> {
    %0 = memref.alloc() : memref<1x8x4x4x!qElemType>
    %1 = VPUIP.Expand {pads_begin = [0, 5, 0, 0], pads_end = [0, 0, 0, 0]} inputs(%arg0 : memref<1x3x4x4x!qElemType>) outputs(%0 : memref<1x8x4x4x!qElemType>) -> memref<1x8x4x4x!qElemType>

    return %1 : memref<1x8x4x4x!qElemType>

    // CHECK-DAG:    [[CST_BEGIN:%.+]] = const.Declare memref<1x5x4x4x!qElemType> = dense<0.000000e+00> : tensor<80xf8E5M2>, [#const.Reshape<[1, 5, 4, 4]>]
    // CHECK:        [[OUT_BUFFER:%.+]] = memref.alloc() : memref<1x8x4x4x!qElemType>

    // CHECK:        [[VIEW_0:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 0, 0, 0] [1, 5, 4, 4]
    // CHECK-SAME:       : memref<1x8x4x4x!qElemType> to memref<1x5x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:        [[COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:       inputs([[CST_BEGIN]] : memref<1x5x4x4x!qElemType>) outputs([[VIEW_0]] : memref<1x5x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>) -> memref<1x5x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>

    // CHECK:        [[VIEW_1:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 5, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:       : memref<1x8x4x4x!qElemType> to memref<1x3x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:        [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:       inputs([[INPUT]] : memref<1x3x4x4x!qElemType>) outputs([[VIEW_1]] : memref<1x3x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>) -> memref<1x3x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>

    // CHECK:        [[OUT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:       inputs([[COPY_0]], [[COPY_1]] : memref<1x5x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>, memref<1x3x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>) outputs(%alloc : memref<1x8x4x4x!qElemType>) -> memref<1x8x4x4x!qElemType>

    // CHECK: return [[OUT]] : memref<1x8x4x4x!qElemType>
}

// -----

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.01>

// CHECK-LABEL: @ExpandBeginAndEndF8E4M3FN
// CHECK-SAME:      [[INPUT:%.+]]: memref<1x3x4x4x!qElemType>
// CHECK-SAME: -> memref<1x8x4x4x!qElemType>
func.func @ExpandBeginAndEndF8E4M3FN(%arg0: memref<1x3x4x4x!qElemType>) -> memref<1x8x4x4x!qElemType> {
    %0 = memref.alloc() : memref<1x8x4x4x!qElemType>
    %1 = VPUIP.Expand {pads_begin = [0, 3, 0, 0], pads_end = [0, 2, 0, 0]} inputs(%arg0 : memref<1x3x4x4x!qElemType>) outputs(%0 : memref<1x8x4x4x!qElemType>) -> memref<1x8x4x4x!qElemType>

    return %1 : memref<1x8x4x4x!qElemType>

    // CHECK-DAG:    [[CST_END:%.+]] = const.Declare memref<1x2x4x4x!qElemType> = dense<0.000000e+00> : tensor<80xf8E4M3FN>, [#const.SubView<[0], [32]>, #const.Reshape<[1, 2, 4, 4]>]
    // CHECK-DAG:    [[CST_BEGIN:%.+]] = const.Declare memref<1x3x4x4x!qElemType> = dense<0.000000e+00> : tensor<80xf8E4M3FN>, [#const.SubView<[0], [48]>, #const.Reshape<[1, 3, 4, 4]>]
    // CHECK:        [[OUT_BUFFER:%.+]] = memref.alloc() : memref<1x8x4x4x!qElemType>

    // CHECK:        [[VIEW_0:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 0, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:       : memref<1x8x4x4x!qElemType> to memref<1x3x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:        [[COPY_0:%.+]] = VPUIP.Copy

    // CHECK-SAME:       inputs([[CST_BEGIN]] : memref<1x3x4x4x!qElemType>) outputs([[VIEW_0]] : memref<1x3x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>) -> memref<1x3x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:        [[VIEW_1:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 3, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:       : memref<1x8x4x4x!qElemType> to memref<1x3x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:        [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:       inputs([[INPUT]] : memref<1x3x4x4x!qElemType>) outputs([[VIEW_1]] : memref<1x3x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>) -> memref<1x3x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>

    // CHECK:        [[VIEW_2:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 6, 0, 0] [1, 2, 4, 4]
    // CHECK-SAME:       : memref<1x8x4x4x!qElemType> to memref<1x2x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:        [[COPY_2:%.+]] = VPUIP.Copy
    // CHECK-SAME:       inputs([[CST_END]] : memref<1x2x4x4x!qElemType>) outputs([[VIEW_2]] : memref<1x2x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>) -> memref<1x2x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>

    // CHECK:        [[OUT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:       inputs([[COPY_0]], [[COPY_1]], [[COPY_2]] : memref<1x3x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>, memref<1x3x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>, memref<1x2x4x4x!qElemType, {order = #NCHW, strides = [128, 16, 4, 1]}>) outputs(%alloc : memref<1x8x4x4x!qElemType>) -> memref<1x8x4x4x!qElemType>

    // CHECK: return [[OUT]] : memref<1x8x4x4x!qElemType>
}

// -----

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.01>

// CHECK-LABEL: @MultiplePrecisionExpands
// CHECK-SAME:      [[INPUT_0:%.+]]: memref<1x3x9x4xf16>
// CHECK-SAME:      [[INPUT_1:%.+]]: memref<1x9x4x4x!qElemType>
// CHECK-SAME: -> memref<1x9x9x4x!qElemType>
func.func @MultiplePrecisionExpands(%arg0: memref<1x3x9x4xf16>, %arg1: memref<1x9x4x4x!qElemType>) -> memref<1x9x9x4x!qElemType> {
    %0 = memref.alloc() : memref<1x9x9x4xf16>
    %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 5, 0]} inputs(%arg0 : memref<1x3x9x4xf16>) outputs(%0 : memref<1x9x9x4xf16>) -> memref<1x9x9x4xf16>

    %2 = memref.alloc() : memref<1x9x9x4x!qElemType>
    %3 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 5, 0]} inputs(%arg1 : memref<1x9x4x4x!qElemType>) outputs(%2 : memref<1x9x9x4x!qElemType>) -> memref<1x9x9x4x!qElemType>

    return %3 : memref<1x9x9x4x!qElemType>

    // CHECK-DAG:    [[CST_END_0:%.+]] = const.Declare memref<1x9x5x4x!qElemType> = dense<0.000000e+00> : tensor<180xf8E4M3FN>, [#const.Reshape<[1, 9, 5, 4]>]
    // CHECK-DAG:    [[CST_END_1:%.+]] = const.Declare memref<1x3x5x4xf16> = dense<0.000000e+00> : tensor<216xf16>, [#const.SubView<[0], [60]>, #const.Reshape<[1, 3, 5, 4]>]

    // CHECK:        [[OUT_BUFFER_0:%.+]] = memref.alloc() : memref<1x9x9x4xf16>
    // CHECK:        [[VIEW_0:%.+]] = VPUIP.SubView [[OUT_BUFFER_0]] [0, 0, 0, 0] [1, 3, 9, 4]
    // CHECK-SAME:       : memref<1x9x9x4xf16> to memref<1x3x9x4xf16, {order = #NCHW, strides = [324, 36, 4, 1]}>
    // CHECK:        [[COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:       inputs([[INPUT_0]] : memref<1x3x9x4xf16>) outputs(%0 : memref<1x3x9x4xf16, {order = #NCHW, strides = [324, 36, 4, 1]}>) -> memref<1x3x9x4xf16, {order = #NCHW, strides = [324, 36, 4, 1]}>
    // CHECK:        [[VIEW_1:%.+]] = VPUIP.SubView [[OUT_BUFFER_0]] [0, 3, 0, 0] [1, 3, 5, 4]
    // CHECK-SAME:       : memref<1x9x9x4xf16> to memref<1x3x5x4xf16, {order = #NCHW, strides = [324, 36, 4, 1]}>
    // CHECK:        [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:       inputs([[CST_END_1]] : memref<1x3x5x4xf16>) outputs(%2 : memref<1x3x5x4xf16, {order = #NCHW, strides = [324, 36, 4, 1]}>) -> memref<1x3x5x4xf16, {order = #NCHW, strides = [324, 36, 4, 1]}>

    // CHECK:        [[OUT_BUFFER_1:%.+]] = memref.alloc() : memref<1x9x9x4x!qElemType>
    // CHECK:        [[VIEW_2:%.+]] = VPUIP.SubView [[OUT_BUFFER_1]] [0, 0, 0, 0] [1, 9, 4, 4]
    // CHECK-SAME:       : memref<1x9x9x4x!qElemType> to memref<1x9x4x4x!qElemType, {order = #NCHW, strides = [324, 36, 4, 1]}>
    // CHECK:        [[COPY_2:%.+]] = VPUIP.Copy
    // CHECK-SAME:       inputs([[INPUT_1]] : memref<1x9x4x4x!qElemType>) outputs(%4 : memref<1x9x4x4x!qElemType, {order = #NCHW, strides = [324, 36, 4, 1]}>) -> memref<1x9x4x4x!qElemType, {order = #NCHW, strides = [324, 36, 4, 1]}>
    // CHECK:        [[VIEW_3:%.+]] = VPUIP.SubView [[OUT_BUFFER_1]] [0, 0, 4, 0] [1, 9, 5, 4]
    // CHECK-SAME:       : memref<1x9x9x4x!qElemType> to memref<1x9x5x4x!qElemType, {order = #NCHW, strides = [324, 36, 4, 1]}>
    // CHECK:        [[COPY_3:%.+]] = VPUIP.Copy
    // CHECK-SAME:       inputs([[CST_END_0]] : memref<1x9x5x4x!qElemType>) outputs(%6 : memref<1x9x5x4x!qElemType, {order = #NCHW, strides = [324, 36, 4, 1]}>) -> memref<1x9x5x4x!qElemType, {order = #NCHW, strides = [324, 36, 4, 1]}>
    // CHECK:        [[OUT:%.+]] = VPUIP.ConcatView inputs(%5, %7 : memref<1x9x4x4x!qElemType, {order = #NCHW, strides = [324, 36, 4, 1]}>, memref<1x9x5x4x!qElemType, {order = #NCHW, strides = [324, 36, 4, 1]}>) outputs(%alloc_1 : memref<1x9x9x4x!qElemType>) -> memref<1x9x9x4x!qElemType>

    // CHECK: return [[OUT]] : memref<1x9x9x4x!qElemType>
}
