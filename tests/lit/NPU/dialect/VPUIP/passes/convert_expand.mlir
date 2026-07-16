//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-expand --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @Expand
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x3x4x4xf16>)
func.func @Expand(%arg0: memref<1x3x4x4xf16>) -> (memref<1x8x4x4xf16>) {
    %0 = memref.alloc() : memref<1x8x4x4xf16>
    %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 5, 0, 0]} inputs(%arg0 : memref<1x3x4x4xf16>) outputs(%0 : memref<1x8x4x4xf16>) -> memref<1x8x4x4xf16>
    return %1 : memref<1x8x4x4xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare memref<1x5x4x4xf16> = dense<0.000000e+00> : tensor<80xf16>, [#const.Reshape<[1, 5, 4, 4]>]
    // CHECK:       [[OUT_BUFFER:%.+]] = memref.alloc() : memref<1x8x4x4xf16>

    // CHECK:       [[VIEW1:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 0, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:      : memref<1x8x4x4xf16> to memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x3x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW1]] : memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)

    // CHECK:       [[VIEW2:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 3, 0, 0] [1, 5, 4, 4]
    // CHECK-SAME:      : memref<1x8x4x4xf16> to memref<1x5x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x5x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW2]] : memref<1x5x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)

    // CHECK:       [[OUT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY1]], [[COPY2]] :
    // CHECK-SAME:          memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>,
    // CHECK-SAME:          memref<1x5x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)
    // CHECK-SAME:      outputs([[OUT_BUFFER]] : memref<1x8x4x4xf16>) -> memref<1x8x4x4xf16>

    // CHECK:       return [[OUT]] : memref<1x8x4x4xf16>
}

// -----

// CHECK-LABEL: func.func @ExpandToSubviewWithoutTail
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x4x4x4xf16>)
func.func @ExpandToSubviewWithoutTail(%arg0: memref<1x4x4x4xf16>) -> memref<1x8x4x4xf16> {
    %0 = memref.alloc() : memref<1x8x4x4xf16>
    %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 4, 0, 0]} inputs(%arg0 : memref<1x4x4x4xf16>) outputs(%0 : memref<1x8x4x4xf16>) -> memref<1x8x4x4xf16>
    return %1 : memref<1x8x4x4xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare memref<1x4x4x4xf16> = dense<0.000000e+00> : tensor<64xf16>, [#const.Reshape<[1, 4, 4, 4]>]
    // CHECK:       [[OUT_BUFFER:%.+]] = memref.alloc() : memref<1x8x4x4xf16>

    // CHECK:       [[VIEW1:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 0, 0, 0] [1, 4, 4, 4]
    // CHECK-SAME:      : memref<1x8x4x4xf16> to memref<1x4x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x4x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW1]] : memref<1x4x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)

    // CHECK:       [[VIEW2:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 4, 0, 0] [1, 4, 4, 4]
    // CHECK-SAME:      : memref<1x8x4x4xf16> to memref<1x4x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x4x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW2]] : memref<1x4x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)

    // CHECK:       [[OUT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY1]], [[COPY2]] :
    // CHECK-SAME:          memref<1x4x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>,
    // CHECK-SAME:          memref<1x4x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)
    // CHECK-SAME:      outputs([[OUT_BUFFER]] : memref<1x8x4x4xf16>) -> memref<1x8x4x4xf16>

    // CHECK:       return [[OUT]] : memref<1x8x4x4xf16>
}

// -----

// CHECK-LABEL: func.func @ExpandToSubviewOnlyWithTail
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x5x4x4xf16>)
func.func @ExpandToSubviewOnlyWithTail(%arg0: memref<1x5x4x4xf16>) -> memref<1x8x4x4xf16> {
    %0 = memref.alloc() : memref<1x8x4x4xf16>
    %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} inputs(%arg0 : memref<1x5x4x4xf16>) outputs(%0 : memref<1x8x4x4xf16>) -> memref<1x8x4x4xf16>
    return %1 : memref<1x8x4x4xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare memref<1x3x4x4xf16> = dense<0.000000e+00> : tensor<48xf16>, [#const.Reshape<[1, 3, 4, 4]>]
    // CHECK:       [[OUT_BUFFER:%.+]] = memref.alloc() : memref<1x8x4x4xf16>

    // CHECK:       [[VIEW1:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 0, 0, 0] [1, 5, 4, 4]
    // CHECK-SAME:      : memref<1x8x4x4xf16> to memref<1x5x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x5x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW1]] : memref<1x5x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)

    // CHECK:       [[VIEW2:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 5, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:      : memref<1x8x4x4xf16> to memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x3x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW2]] : memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)

    // CHECK:       [[OUT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY1]], [[COPY2]] :
    // CHECK-SAME:          memref<1x5x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>,
    // CHECK-SAME:          memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)
    // CHECK-SAME:      outputs([[OUT_BUFFER]] : memref<1x8x4x4xf16>) -> memref<1x8x4x4xf16>

    // CHECK:       return [[OUT]] : memref<1x8x4x4xf16>
}

// -----

// CHECK-LABEL: func.func @ExpandOverWidth
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x3x4x4xf16>)
func.func @ExpandOverWidth(%arg0: memref<1x3x4x4xf16>) -> memref<1x3x4x9xf16> {
    %0 = memref.alloc() : memref<1x3x4x9xf16>
    %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 5]} inputs(%arg0 : memref<1x3x4x4xf16>) outputs(%0 : memref<1x3x4x9xf16>) -> memref<1x3x4x9xf16>
    return %1 : memref<1x3x4x9xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare memref<1x3x4x5xf16> = dense<0.000000e+00> : tensor<60xf16>, [#const.Reshape<[1, 3, 4, 5]>]
    // CHECK:       [[BUFFER:%.+]] = memref.alloc() : memref<1x3x4x9xf16>

    // CHECK:       [[VIEW1:%.+]] = VPUIP.SubView [[BUFFER]] [0, 0, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:      : memref<1x3x4x9xf16> to memref<1x3x4x4xf16, {order = #NCHW, strides = [108, 36, 9, 1]}>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x3x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW1]] : memref<1x3x4x4xf16, {order = #NCHW, strides = [108, 36, 9, 1]}>)

    // CHECK:       [[VIEW2:%.+]] = VPUIP.SubView [[BUFFER]] [0, 0, 0, 4] [1, 3, 4, 5]
    // CHECK-SAME:      : memref<1x3x4x9xf16> to memref<1x3x4x5xf16, {order = #NCHW, strides = [108, 36, 9, 1]}>
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x3x4x5xf16>)
    // CHECK-SAME:      outputs([[VIEW2]] : memref<1x3x4x5xf16, {order = #NCHW, strides = [108, 36, 9, 1]}>)

    // CHECK:       [[OUT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY1]], [[COPY2]] :
    // CHECK-SAME:          memref<1x3x4x4xf16, {order = #NCHW, strides = [108, 36, 9, 1]}>,
    // CHECK-SAME:          memref<1x3x4x5xf16, {order = #NCHW, strides = [108, 36, 9, 1]}>)
    // CHECK-SAME:      outputs([[BUFFER]] : memref<1x3x4x9xf16>) -> memref<1x3x4x9xf16>

    // CHECK:       return [[OUT]] : memref<1x3x4x9xf16>
}

// -----

// CHECK-LABEL: func.func @ExpandOverHeight
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x3x4x4xf16>)
func.func @ExpandOverHeight(%arg0: memref<1x3x4x4xf16>) -> memref<1x3x9x4xf16> {
    %0 = memref.alloc() : memref<1x3x9x4xf16>
    %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 5, 0]} inputs(%arg0 : memref<1x3x4x4xf16>) outputs(%0 : memref<1x3x9x4xf16>) -> memref<1x3x9x4xf16>
    return %1 : memref<1x3x9x4xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare memref<1x3x5x4xf16> = dense<0.000000e+00> : tensor<60xf16>, [#const.Reshape<[1, 3, 5, 4]>]
    // CHECK:       [[BUFFER:%.+]] = memref.alloc() : memref<1x3x9x4xf16>

    // CHECK:       [[VIEW1:%.+]] = VPUIP.SubView [[BUFFER]] [0, 0, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:      : memref<1x3x9x4xf16> to memref<1x3x4x4xf16, {order = #NCHW, strides = [108, 36, 4, 1]}>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x3x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW1]] : memref<1x3x4x4xf16, {order = #NCHW, strides = [108, 36, 4, 1]}>)

    // CHECK:       [[VIEW2:%.+]] = VPUIP.SubView [[BUFFER]] [0, 0, 4, 0] [1, 3, 5, 4]
    // CHECK-SAME:      : memref<1x3x9x4xf16> to memref<1x3x5x4xf16, {order = #NCHW, strides = [108, 36, 4, 1]}>
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x3x5x4xf16>)
    // CHECK-SAME:      outputs([[VIEW2]] : memref<1x3x5x4xf16, {order = #NCHW, strides = [108, 36, 4, 1]}>)

    // CHECK:       [[OUT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY1]], [[COPY2]] :
    // CHECK-SAME:          memref<1x3x4x4xf16, {order = #NCHW, strides = [108, 36, 4, 1]}>,
    // CHECK-SAME:          memref<1x3x5x4xf16, {order = #NCHW, strides = [108, 36, 4, 1]}>)
    // CHECK-SAME:      outputs([[BUFFER]] : memref<1x3x9x4xf16>) -> memref<1x3x9x4xf16>

    // CHECK:       return [[OUT]] : memref<1x3x9x4xf16>
}

// -----

// CHECK-LABEL: func.func @ExpandPadsBeginFullCopy
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x3x4x4xf16>)
func.func @ExpandPadsBeginFullCopy(%arg0: memref<1x3x4x4xf16>) -> memref<1x6x4x4xf16> {
    %0 = memref.alloc() : memref<1x6x4x4xf16>
    %1 = VPUIP.Expand {pads_begin = [0, 3, 0, 0], pads_end = [0, 0, 0, 0]} inputs(%arg0 : memref<1x3x4x4xf16>) outputs(%0 : memref<1x6x4x4xf16>) -> memref<1x6x4x4xf16>

    return %1 : memref<1x6x4x4xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare memref<1x3x4x4xf16> = dense<0.000000e+00> : tensor<48xf16>, [#const.Reshape<[1, 3, 4, 4]>]
    // CHECK:       [[OUT_BUFFER:%.+]] = memref.alloc() : memref<1x6x4x4xf16>

    // CHECK:       [[VIEW1:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 0, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:      : memref<1x6x4x4xf16> to memref<1x3x4x4xf16, {order = #NCHW, strides = [96, 16, 4, 1]}>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x3x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW1]] : memref<1x3x4x4xf16, {order = #NCHW, strides = [96, 16, 4, 1]}>)

    // CHECK:       [[VIEW2:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 3, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:      : memref<1x6x4x4xf16> to memref<1x3x4x4xf16, {order = #NCHW, strides = [96, 16, 4, 1]}>
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x3x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW2]] : memref<1x3x4x4xf16, {order = #NCHW, strides = [96, 16, 4, 1]}>)

    // CHECK:       [[OUT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY1]], [[COPY2]] :
    // CHECK-SAME:          memref<1x3x4x4xf16, {order = #NCHW, strides = [96, 16, 4, 1]}>,
    // CHECK-SAME:          memref<1x3x4x4xf16, {order = #NCHW, strides = [96, 16, 4, 1]}>)
    // CHECK-SAME:      outputs([[OUT_BUFFER]] : memref<1x6x4x4xf16>) -> memref<1x6x4x4xf16>

    // CHECK:       return [[OUT]] : memref<1x6x4x4xf16>
}

// -----

// CHECK-LABEL: func.func @ExpandPadsBeginSliceCopy
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x3x4x4xf16>)
func.func @ExpandPadsBeginSliceCopy(%arg0: memref<1x3x4x4xf16>) -> memref<1x5x4x4xf16> {
    %0 = memref.alloc() : memref<1x5x4x4xf16>
    %1 = VPUIP.Expand {pads_begin = [0, 2, 0, 0], pads_end = [0, 0, 0, 0]} inputs(%arg0 : memref<1x3x4x4xf16>) outputs(%0 : memref<1x5x4x4xf16>) -> memref<1x5x4x4xf16>

    return %1 : memref<1x5x4x4xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare memref<1x2x4x4xf16> = dense<0.000000e+00> : tensor<32xf16>, [#const.Reshape<[1, 2, 4, 4]>]
    // CHECK:       [[OUT_BUFFER:%.+]] = memref.alloc() : memref<1x5x4x4xf16>

    // CHECK:       [[VIEW1:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 0, 0, 0] [1, 2, 4, 4]
    // CHECK-SAME:      : memref<1x5x4x4xf16> to memref<1x2x4x4xf16, {order = #NCHW, strides = [80, 16, 4, 1]}>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x2x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW1]] : memref<1x2x4x4xf16, {order = #NCHW, strides = [80, 16, 4, 1]}>)

    // CHECK:       [[VIEW2:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 2, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:      : memref<1x5x4x4xf16> to memref<1x3x4x4xf16, {order = #NCHW, strides = [80, 16, 4, 1]}>
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x3x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW2]] : memref<1x3x4x4xf16, {order = #NCHW, strides = [80, 16, 4, 1]}>)

    // CHECK:       [[OUT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY1]], [[COPY2]] :
    // CHECK-SAME:          memref<1x2x4x4xf16, {order = #NCHW, strides = [80, 16, 4, 1]}>,
    // CHECK-SAME:          memref<1x3x4x4xf16, {order = #NCHW, strides = [80, 16, 4, 1]}>)
    // CHECK-SAME:      outputs([[OUT_BUFFER]] : memref<1x5x4x4xf16>) -> memref<1x5x4x4xf16>

    // CHECK:       return [[OUT]] : memref<1x5x4x4xf16>
}

// -----

// CHECK-LABEL: func.func @ExpandPadsBeginCopiesWithTail
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x3x4x4xf16>)
func.func @ExpandPadsBeginCopiesWithTail(%arg0: memref<1x3x4x4xf16>) -> memref<1x11x4x4xf16> {
    %0 = memref.alloc() : memref<1x11x4x4xf16>
    %1 = VPUIP.Expand {pads_begin = [0, 8, 0, 0], pads_end = [0, 0, 0, 0]} inputs(%arg0 : memref<1x3x4x4xf16>) outputs(%0 : memref<1x11x4x4xf16>) -> memref<1x11x4x4xf16>

    return %1 : memref<1x11x4x4xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare memref<1x8x4x4xf16> = dense<0.000000e+00> : tensor<128xf16>, [#const.Reshape<[1, 8, 4, 4]>]
    // CHECK:       [[OUT_BUFFER:%.+]] = memref.alloc() : memref<1x11x4x4xf16>

    // CHECK:       [[VIEW1:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 0, 0, 0] [1, 8, 4, 4]
    // CHECK-SAME:      : memref<1x11x4x4xf16> to memref<1x8x4x4xf16, {order = #NCHW, strides = [176, 16, 4, 1]}>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x8x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW1]] : memref<1x8x4x4xf16, {order = #NCHW, strides = [176, 16, 4, 1]}>)

    // CHECK:       [[VIEW2:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 8, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:      : memref<1x11x4x4xf16> to memref<1x3x4x4xf16, {order = #NCHW, strides = [176, 16, 4, 1]}>
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x3x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW2]] : memref<1x3x4x4xf16, {order = #NCHW, strides = [176, 16, 4, 1]}>)

    // CHECK:       [[OUT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY1]], [[COPY2]] :
    // CHECK-SAME:          memref<1x8x4x4xf16, {order = #NCHW, strides = [176, 16, 4, 1]}>,
    // CHECK-SAME:          memref<1x3x4x4xf16, {order = #NCHW, strides = [176, 16, 4, 1]}>)
    // CHECK-SAME:      outputs([[OUT_BUFFER]] : memref<1x11x4x4xf16>) -> memref<1x11x4x4xf16>

    // CHECK:       return [[OUT]] : memref<1x11x4x4xf16>
}

// -----

// CHECK-LABEL: func.func @ExpandBeginPadsWithEndPads
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x3x4x4xf16>)
func.func @ExpandBeginPadsWithEndPads(%arg0: memref<1x3x4x4xf16>) -> memref<1x9x4x4xf16> {
    %0 = memref.alloc() : memref<1x9x4x4xf16>
    %1 = VPUIP.Expand {pads_begin = [0, 3, 0, 0], pads_end = [0, 3, 0, 0]} inputs(%arg0 : memref<1x3x4x4xf16>) outputs(%0 : memref<1x9x4x4xf16>) -> memref<1x9x4x4xf16>

    return %1 : memref<1x9x4x4xf16>

    // CHECK:       [[CST:%.+]] =  const.Declare memref<1x3x4x4xf16> = dense<0.000000e+00> : tensor<96xf16>, [#const.SubView<[0], [48]>, #const.Reshape<[1, 3, 4, 4]>]
    // CHECK:       [[OUT_BUFFER:%.+]] = memref.alloc() : memref<1x9x4x4xf16>

    // CHECK:       [[VIEW1:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 0, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:      : memref<1x9x4x4xf16> to memref<1x3x4x4xf16, {order = #NCHW, strides = [144, 16, 4, 1]}>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x3x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW1]] : memref<1x3x4x4xf16, {order = #NCHW, strides = [144, 16, 4, 1]}>)

    // CHECK:       [[VIEW2:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 3, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:      : memref<1x9x4x4xf16> to memref<1x3x4x4xf16, {order = #NCHW, strides = [144, 16, 4, 1]}>
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x3x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW2]] : memref<1x3x4x4xf16, {order = #NCHW, strides = [144, 16, 4, 1]}>)

    // CHECK:       [[VIEW3:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 6, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:      : memref<1x9x4x4xf16> to memref<1x3x4x4xf16, {order = #NCHW, strides = [144, 16, 4, 1]}>
    // CHECK:       [[COPY3:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x3x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW3]] : memref<1x3x4x4xf16, {order = #NCHW, strides = [144, 16, 4, 1]}>)

    // CHECK:       [[OUT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY1]], [[COPY2]], [[COPY3]] :
    // CHECK-SAME:          memref<1x3x4x4xf16, {order = #NCHW, strides = [144, 16, 4, 1]}>,
    // CHECK-SAME:          memref<1x3x4x4xf16, {order = #NCHW, strides = [144, 16, 4, 1]}>,
    // CHECK-SAME:          memref<1x3x4x4xf16, {order = #NCHW, strides = [144, 16, 4, 1]}>)
    // CHECK-SAME:      outputs([[OUT_BUFFER]] : memref<1x9x4x4xf16>) -> memref<1x9x4x4xf16>

    // CHECK:       return [[OUT]] : memref<1x9x4x4xf16>
}

// -----

// CHECK-LABEL: func.func @TwoExpandsAndReuseConstant
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x3x4x4xf16>)
func.func @TwoExpandsAndReuseConstant(%arg0: memref<1x3x4x4xf16>) -> memref<1x9x9x4xf16> {
    %0 = memref.alloc() : memref<1x9x4x4xf16>
    %1 = VPUIP.Expand {pads_begin = [0, 3, 0, 0], pads_end = [0, 3, 0, 0]} inputs(%arg0 : memref<1x3x4x4xf16>) outputs(%0 : memref<1x9x4x4xf16>) -> memref<1x9x4x4xf16>

    %2 = memref.alloc() : memref<1x9x9x4xf16>
    %3 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 5, 0]} inputs(%1 : memref<1x9x4x4xf16>) outputs(%2 : memref<1x9x9x4xf16>) -> memref<1x9x9x4xf16>
    return %3 : memref<1x9x9x4xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare memref<1x9x5x4xf16> = dense<0.000000e+00> : tensor<180xf16>, [#const.Reshape<[1, 9, 5, 4]>]
    // CHECK-DAG:       [[CST_0:%.+]] = const.Declare memref<1x3x4x4xf16> = dense<0.000000e+00> : tensor<180xf16>, [#const.SubView<[0], [48]>, #const.Reshape<[1, 3, 4, 4]>]

    // CHECK:       [[OUT_BUFFER_0:%.+]] = memref.alloc() : memref<1x9x4x4xf16>

    // CHECK:       [[VIEW1:%.+]] = VPUIP.SubView [[OUT_BUFFER_0]] [0, 0, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:      : memref<1x9x4x4xf16> to memref<1x3x4x4xf16, {order = #NCHW, strides = [144, 16, 4, 1]}>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[CST_0]] : memref<1x3x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW1]] : memref<1x3x4x4xf16, {order = #NCHW, strides = [144, 16, 4, 1]}>)

    // CHECK:       [[VIEW2:%.+]] = VPUIP.SubView [[OUT_BUFFER_0]] [0, 3, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:      : memref<1x9x4x4xf16> to memref<1x3x4x4xf16, {order = #NCHW, strides = [144, 16, 4, 1]}>
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x3x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW2]] : memref<1x3x4x4xf16, {order = #NCHW, strides = [144, 16, 4, 1]}>)

    // CHECK:       [[VIEW3:%.+]] = VPUIP.SubView [[OUT_BUFFER_0]] [0, 6, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:      : memref<1x9x4x4xf16> to memref<1x3x4x4xf16, {order = #NCHW, strides = [144, 16, 4, 1]}>
    // CHECK:       [[COPY3:%.+]] = VPUIP.Copy inputs([[CST_0]] : memref<1x3x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW3]] : memref<1x3x4x4xf16, {order = #NCHW, strides = [144, 16, 4, 1]}>)

    // CHECK:       [[EXPAND_0:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY1]], [[COPY2]], [[COPY3]] :
    // CHECK-SAME:          memref<1x3x4x4xf16, {order = #NCHW, strides = [144, 16, 4, 1]}>,
    // CHECK-SAME:          memref<1x3x4x4xf16, {order = #NCHW, strides = [144, 16, 4, 1]}>,
    // CHECK-SAME:          memref<1x3x4x4xf16, {order = #NCHW, strides = [144, 16, 4, 1]}>)
    // CHECK-SAME:      outputs([[OUT_BUFFER_0]] : memref<1x9x4x4xf16>) -> memref<1x9x4x4xf16>

    // CHECK:       [[OUT_BUFFER_1:%.+]] = memref.alloc() : memref<1x9x9x4xf16>

    // CHECK:       [[VIEW4:%.+]] = VPUIP.SubView [[OUT_BUFFER_1]] [0, 0, 0, 0] [1, 9, 4, 4]
    // CHECK-SAME:      : memref<1x9x9x4xf16> to memref<1x9x4x4xf16, {order = #NCHW, strides = [324, 36, 4, 1]}>
    // CHECK:       [[COPY4:%.+]] = VPUIP.Copy inputs([[EXPAND_0]] : memref<1x9x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW4]] : memref<1x9x4x4xf16, {order = #NCHW, strides = [324, 36, 4, 1]}>)

    // CHECK:       [[VIEW5:%.+]] = VPUIP.SubView [[OUT_BUFFER_1]] [0, 0, 4, 0] [1, 9, 5, 4]
    // CHECK-SAME:      : memref<1x9x9x4xf16> to memref<1x9x5x4xf16, {order = #NCHW, strides = [324, 36, 4, 1]}>
    // CHECK:       [[COPY5:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x9x5x4xf16>)
    // CHECK-SAME:      outputs([[VIEW5]] : memref<1x9x5x4xf16, {order = #NCHW, strides = [324, 36, 4, 1]}>)

    // CHECK:       [[EXPAND_1:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY4]], [[COPY5]] :
    // CHECK-SAME:          memref<1x9x4x4xf16, {order = #NCHW, strides = [324, 36, 4, 1]}>,
    // CHECK-SAME:          memref<1x9x5x4xf16, {order = #NCHW, strides = [324, 36, 4, 1]}>)
    // CHECK-SAME:      outputs([[OUT_BUFFER_1]] : memref<1x9x9x4xf16>) -> memref<1x9x9x4xf16>

    // CHECK:       return [[EXPAND_1]] : memref<1x9x9x4xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0069734788408466414>

// CHECK-LABEL: func.func @QuantizedExpandWithPadsBegin
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x1x128x200x!qElemType>)
func.func @QuantizedExpandWithPadsBegin(%arg0: memref<1x1x128x200x!qElemType>) -> memref<1x1x128x202x!qElemType> {
    %alloc_0 = memref.alloc() : memref<1x1x128x202x!qElemType>

    %0 = VPUIP.Expand {pads_begin = [0, 0, 0, 1], pads_end = [0, 0, 0, 1]} inputs(%arg0 : memref<1x1x128x200x!qElemType>) outputs(%alloc_0 : memref<1x1x128x202x!qElemType>) -> memref<1x1x128x202x!qElemType>

    return %0 : memref<1x1x128x202x!qElemType>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare memref<1x1x128x1x!qElemType> = dense<0> : tensor<256xi8>, [#const.SubView<[0], [128]>, #const.Reshape<[1, 1, 128, 1]>]
    // CHECK:           [[OUT_BUFFER_0:%.+]] = memref.alloc() : memref<1x1x128x202x!qElemType>
    // CHECK:       [[VIEW1:%.+]] = VPUIP.SubView [[OUT_BUFFER_0]] [0, 0, 0, 0] [1, 1, 128, 1]
    // CHECK-SAME:      : memref<1x1x128x202x!qElemType> to memref<1x1x128x1x!qElemType, {order = #NCHW, strides = [25856, 25856, 202, 1]}>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x1x128x1x!qElemType>)
    // CHECK-SAME:      outputs([[VIEW1]] : memref<1x1x128x1x!qElemType, {order = #NCHW, strides = [25856, 25856, 202, 1]}>)

    // CHECK:       [[VIEW2:%.+]] = VPUIP.SubView [[OUT_BUFFER_0]] [0, 0, 0, 1] [1, 1, 128, 200]
    // CHECK-SAME:      : memref<1x1x128x202x!qElemType> to memref<1x1x128x200x!qElemType, {order = #NCHW, strides = [25856, 25856, 202, 1]}>
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x1x128x200x!qElemType>)
    // CHECK-SAME:      outputs([[VIEW2]] : memref<1x1x128x200x!qElemType, {order = #NCHW, strides = [25856, 25856, 202, 1]}>)

    // CHECK:       [[VIEW3:%.+]] = VPUIP.SubView [[OUT_BUFFER_0]] [0, 0, 0, 201] [1, 1, 128, 1]
    // CHECK-SAME:      : memref<1x1x128x202x!qElemType> to memref<1x1x128x1x!qElemType, {order = #NCHW, strides = [25856, 25856, 202, 1]}>
    // CHECK:       [[COPY3:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x1x128x1x!qElemType>)
    // CHECK-SAME:      outputs([[VIEW3]] : memref<1x1x128x1x!qElemType, {order = #NCHW, strides = [25856, 25856, 202, 1]}>)

    // CHECK:       [[OUT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY1]], [[COPY2]], [[COPY3]] :
    // CHECK-SAME:          memref<1x1x128x1x!qElemType, {order = #NCHW, strides = [25856, 25856, 202, 1]}>,
    // CHECK-SAME:          memref<1x1x128x200x!qElemType, {order = #NCHW, strides = [25856, 25856, 202, 1]}>,
    // CHECK-SAME:          memref<1x1x128x1x!qElemType, {order = #NCHW, strides = [25856, 25856, 202, 1]}>)
    // CHECK-SAME:      outputs([[OUT_BUFFER_0]] : memref<1x1x128x202x!qElemType>) -> memref<1x1x128x202x!qElemType>

    // CHECK:       return [[OUT]] : memref<1x1x128x202x!qElemType>

}

// -----

!qElemType = !quant.uniform<i8:f16, 0.0069734788408466414>

// CHECK-LABEL: func.func @SignedQuantizedExpandWithPadsBegin
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x1x128x200x!qElemType>)
func.func @SignedQuantizedExpandWithPadsBegin(%arg0: memref<1x1x128x200x!qElemType>) -> memref<1x1x128x202x!qElemType> {
    %alloc_0 = memref.alloc() : memref<1x1x128x202x!qElemType>

    %0 = VPUIP.Expand {pads_begin = [0, 0, 0, 1], pads_end = [0, 0, 0, 1]} inputs(%arg0 : memref<1x1x128x200x!qElemType>) outputs(%alloc_0 : memref<1x1x128x202x!qElemType>) -> memref<1x1x128x202x!qElemType>

    return %0 : memref<1x1x128x202x!qElemType>

    // CHECK-NOT:       VPUIP.Expand
    // CHECK-DAG:       [[CST:%.+]] = const.Declare memref<1x1x128x1x!qElemType> = dense<0> : tensor<256xi8>, [#const.SubView<[0], [128]>, #const.Reshape<[1, 1, 128, 1]>]
    // CHECK:           [[OUT_BUFFER_0:%.+]] = memref.alloc() : memref<1x1x128x202x!qElemType>
    // CHECK:       [[VIEW1:%.+]] = VPUIP.SubView [[OUT_BUFFER_0]] [0, 0, 0, 0] [1, 1, 128, 1]
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x1x128x1x!qElemType>)

    // CHECK:       [[VIEW2:%.+]] = VPUIP.SubView [[OUT_BUFFER_0]] [0, 0, 0, 1] [1, 1, 128, 200]
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x1x128x200x!qElemType>)

    // CHECK:       [[VIEW3:%.+]] = VPUIP.SubView [[OUT_BUFFER_0]] [0, 0, 0, 201] [1, 1, 128, 1]
    // CHECK:       [[COPY3:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x1x128x1x!qElemType>)

    // CHECK:       [[OUT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY1]], [[COPY2]], [[COPY3]]
    // CHECK-SAME:      outputs([[OUT_BUFFER_0]] : memref<1x1x128x202x!qElemType>) -> memref<1x1x128x202x!qElemType>
    // CHECK:       return [[OUT]] : memref<1x1x128x202x!qElemType>
}

// -----

!qElemType = !quant.uniform<i8:f16, 0.005:5>

// Non-zero zero-points require stored-zero-point padding; keep Expand for the DMA path.
// CHECK-LABEL: func.func @SignedQuantizedExpandNonZeroZeroPoint
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x3x4x4x!qElemType>)
func.func @SignedQuantizedExpandNonZeroZeroPoint(%arg0: memref<1x3x4x4x!qElemType>) -> memref<1x4x4x4x!qElemType> {
    %alloc_0 = memref.alloc() : memref<1x4x4x4x!qElemType>

    %0 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} inputs(%arg0 : memref<1x3x4x4x!qElemType>) outputs(%alloc_0 : memref<1x4x4x4x!qElemType>) -> memref<1x4x4x4x!qElemType>

    return %0 : memref<1x4x4x4x!qElemType>

    // CHECK-NOT:       const.Declare
    // CHECK:           [[ALLOC_0:%.+]] = memref.alloc() : memref<1x4x4x4x!qElemType>
    // CHECK:           [[EXPAND:%.+]] = VPUIP.Expand
    // CHECK-SAME:      inputs([[ARG_0]] : memref<1x3x4x4x!qElemType>)
    // CHECK-SAME:      outputs([[ALLOC_0]] : memref<1x4x4x4x!qElemType>) -> memref<1x4x4x4x!qElemType>
    // CHECK:           return [[EXPAND]] : memref<1x4x4x4x!qElemType>
}

// -----

!qElemType = !quant.uniform<i8:f16:1, {0.005, 0.006, 0.007}>

// Per-axis parameters require axis-specific padding values; keep Expand for the DMA path.
// CHECK-LABEL: func.func @PerAxisQuantizedExpand
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x3x4x4x!qElemType>)
func.func @PerAxisQuantizedExpand(%arg0: memref<1x3x4x4x!qElemType>) -> memref<1x3x4x5x!qElemType> {
    %alloc_0 = memref.alloc() : memref<1x3x4x5x!qElemType>

    %0 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 1]} inputs(%arg0 : memref<1x3x4x4x!qElemType>) outputs(%alloc_0 : memref<1x3x4x5x!qElemType>) -> memref<1x3x4x5x!qElemType>

    return %0 : memref<1x3x4x5x!qElemType>

    // CHECK-NOT:       const.Declare
    // CHECK:           [[ALLOC_0:%.+]] = memref.alloc() : memref<1x3x4x5x!qElemType>
    // CHECK:           [[EXPAND:%.+]] = VPUIP.Expand
    // CHECK-SAME:      inputs([[ARG_0]] : memref<1x3x4x4x!qElemType>)
    // CHECK-SAME:      outputs([[ALLOC_0]] : memref<1x3x4x5x!qElemType>) -> memref<1x3x4x5x!qElemType>
    // CHECK:           return [[EXPAND]] : memref<1x3x4x5x!qElemType>
}

// -----

!qElemType = !quant.uniform<si8:f16, 0.005>
!qElemType1 = !quant.uniform<ui8:f16, 0.0038725490663565842>

// Both signed and unsigned i8 zp=0 with zero `pads_begin` are decomposed here under the
// default `defer-to-expand-dma=false`. One shared zero constant is emitted per signedness
// bucket. The deferral path is covered separately in convert_expand_defer_to_expand_dma.mlir.
// CHECK-LABEL: func.func @MixedSignedAndUnsignedQuantizedExpand
// CHECK-SAME: ([[ARG_I8:%[^:]+]]: memref<1x3x4x4x!qElemType>, [[ARG_U8:%[^:]+]]: memref<1x3x6x6x!qElemType1>)
func.func @MixedSignedAndUnsignedQuantizedExpand(
        %arg0: memref<1x3x4x4x!qElemType>,
        %arg1: memref<1x3x6x6x!qElemType1>) ->
        (memref<1x4x4x4x!qElemType>, memref<1x4x6x6x!qElemType1>) {
    %alloc_i8 = memref.alloc() : memref<1x4x4x4x!qElemType>
    %alloc_u8 = memref.alloc() : memref<1x4x6x6x!qElemType1>

    %0 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} inputs(%arg0 : memref<1x3x4x4x!qElemType>) outputs(%alloc_i8 : memref<1x4x4x4x!qElemType>) -> memref<1x4x4x4x!qElemType>
    %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} inputs(%arg1 : memref<1x3x6x6x!qElemType1>) outputs(%alloc_u8 : memref<1x4x6x6x!qElemType1>) -> memref<1x4x6x6x!qElemType1>

    return %0, %1 : memref<1x4x4x4x!qElemType>, memref<1x4x6x6x!qElemType1>

    // CHECK-NOT:   VPUIP.Expand
    // CHECK-DAG:   [[CST_U8:%.+]] = const.Declare memref<1x1x6x6x!qElemType1> = dense<0> : tensor<36xui8>
    // CHECK-DAG:   [[CST_I8:%.+]] = const.Declare memref<1x1x4x4x!qElemType> = dense<0> : tensor<16xsi8>
    // CHECK:       VPUIP.Copy inputs([[ARG_I8]]
    // CHECK:       VPUIP.Copy inputs([[CST_I8]]
    // CHECK:       [[OUT_I8:%.+]] = VPUIP.ConcatView
    // CHECK:       VPUIP.Copy inputs([[ARG_U8]]
    // CHECK:       VPUIP.Copy inputs([[CST_U8]]
    // CHECK:       [[OUT_U8:%.+]] = VPUIP.ConcatView
    // CHECK:       return [[OUT_I8]], [[OUT_U8]]
}
