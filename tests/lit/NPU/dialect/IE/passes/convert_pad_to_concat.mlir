//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-pad-to-concat %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @convertPadToConcatWithN
func.func @convertPadToConcatWithN(%arg0: tensor<1x8x16x16xf16>) -> tensor<4x8x16x16xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 1.000000e+00 : f64, pads_begin_attr = [1, 0, 0, 0], pads_end_attr = [2, 0, 0, 0]}
                        : tensor<1x8x16x16xf16> -> tensor<4x8x16x16xf16>
    return %0 : tensor<4x8x16x16xf16>

    // CHECK-DAG:      [[CST0:%.*]] = const.Declare tensor<1x8x16x16xf16> = dense<1.000000e+00
    // CHECK-DAG:      [[CST1:%.*]] = const.Declare tensor<2x8x16x16xf16> = dense<1.000000e+00
    // CHECK-NOT:      IE.Pad
    // CHECK:       [[VAR0:%.*]] = IE.Concat([[CST0]], %arg0, [[CST1]]) {per_axis = #IE.Concat<axis = 0 : i64>}
    // CHECK-SAME:       tensor<1x8x16x16xf16>, tensor<1x8x16x16xf16>, tensor<2x8x16x16xf16> -> tensor<4x8x16x16xf16>
    // CHECK:       return [[VAR0]] : tensor<4x8x16x16xf16>
}

// CHECK-LABEL: @convertPadToConcatWithHW
func.func @convertPadToConcatWithHW(%arg0: tensor<1x8x16x16xf16>) -> tensor<1x8x19x19xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 1.000000e+00 : f64, pads_begin_attr = [0, 0, 1, 2], pads_end_attr = [0, 0, 2, 1]}
                        : tensor<1x8x16x16xf16> -> tensor<1x8x19x19xf16>
    return %0 : tensor<1x8x19x19xf16>

    // CHECK-DAG:      [[CST0:%.*]] = const.Declare tensor<1x8x16x2xf16> = dense<1.000000e+00
    // CHECK-DAG:      [[CST1:%.*]] = const.Declare tensor<1x8x16x1xf16> = dense<1.000000e+00
    // CHECK-DAG:      [[CST2:%.*]] = const.Declare tensor<1x8x1x19xf16> = dense<1.000000e+00
    // CHECK-DAG:      [[CST3:%.*]] = const.Declare tensor<1x8x2x19xf16> = dense<1.000000e+00
    // CHECK-NOT:      IE.Pad
    // CHECK:       [[VAR0:%.*]] = IE.Concat([[CST0]], %arg0, [[CST1]]) {per_axis = #IE.Concat<axis = 3 : i64>}
    // CHECK-SAME:       tensor<1x8x16x2xf16>, tensor<1x8x16x16xf16>, tensor<1x8x16x1xf16> -> tensor<1x8x16x19xf16>
    // CHECK:       [[VAR1:%.*]] = IE.Concat([[CST2]], [[VAR0]], [[CST3]]) {per_axis = #IE.Concat<axis = 2 : i64>}
    // CHECK-SAME:       tensor<1x8x1x19xf16>, tensor<1x8x16x19xf16>, tensor<1x8x2x19xf16> -> tensor<1x8x19x19xf16>
    // CHECK:       return [[VAR1]] : tensor<1x8x19x19xf16>
}

// CHECK-LABEL: @convertPadToConcatWithNCHW
func.func @convertPadToConcatWithNCHW(%arg0: tensor<1x8x16x16xf16>) -> tensor<6x13x21x21xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [4, 3, 2, 1], pads_end_attr = [1, 2, 3, 4]}
                        : tensor<1x8x16x16xf16> -> tensor<6x13x21x21xf16>
    return %0 : tensor<6x13x21x21xf16>

    // CHECK-DAG:      [[CST0:%.*]] = const.Declare tensor<1x8x16x1xf16> = dense<0.000000e+00
    // CHECK-DAG:      [[CST1:%.*]] = const.Declare tensor<1x8x16x4xf16> = dense<0.000000e+00
    // CHECK-DAG:      [[CST2:%.*]] = const.Declare tensor<1x8x2x21xf16> = dense<0.000000e+00
    // CHECK-DAG:      [[CST3:%.*]] = const.Declare tensor<1x8x3x21xf16> = dense<0.000000e+00
    // CHECK-DAG:      [[CST4:%.*]] = const.Declare tensor<1x3x21x21xf16> = dense<0.000000e+00
    // CHECK-DAG:      [[CST5:%.*]] = const.Declare tensor<1x2x21x21xf16> = dense<0.000000e+00
    // CHECK-DAG:      [[CST6:%.*]] = const.Declare tensor<4x13x21x21xf16> = dense<0.000000e+00
    // CHECK-DAG:      [[CST7:%.*]] = const.Declare tensor<1x13x21x21xf16> = dense<0.000000e+00
    // CHECK-NOT:      IE.Pad
    // CHECK:       [[VAR0:%.*]] = IE.Concat([[CST0]], %arg0, [[CST1]]) {per_axis = #IE.Concat<axis = 3 : i64>}
    // CHECK-SAME:       tensor<1x8x16x1xf16>, tensor<1x8x16x16xf16>, tensor<1x8x16x4xf16> -> tensor<1x8x16x21xf16>
    // CHECK:       [[VAR1:%.*]] = IE.Concat([[CST2]], [[VAR0]], [[CST3]]) {per_axis = #IE.Concat<axis = 2 : i64>}
    // CHECK-SAME:       tensor<1x8x2x21xf16>, tensor<1x8x16x21xf16>, tensor<1x8x3x21xf16> -> tensor<1x8x21x21xf16>
    // CHECK:       [[VAR2:%.*]] = IE.Concat([[CST4]], [[VAR1]], [[CST5]]) {per_axis = #IE.Concat<axis = 1 : i64>}
    // CHECK-SAME:       tensor<1x3x21x21xf16>, tensor<1x8x21x21xf16>, tensor<1x2x21x21xf16> -> tensor<1x13x21x21xf16>
    // CHECK:       [[VAR3:%.*]] = IE.Concat([[CST6]], [[VAR2]], [[CST7]]) {per_axis = #IE.Concat<axis = 0 : i64>}
    // CHECK-SAME:       tensor<4x13x21x21xf16>, tensor<1x13x21x21xf16>, tensor<1x13x21x21xf16> -> tensor<6x13x21x21xf16>
    // CHECK:       return [[VAR3]] : tensor<6x13x21x21xf16>
}

// CHECK-LABEL: @convertPadToConcatWithMultiUser
func.func @convertPadToConcatWithMultiUser(%arg0: tensor<1x3x297x297xf16>) -> (tensor<1x16x300x300xf16>, tensor<1x3x300x300xf16>) {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 1, 2], pads_end_attr = [0, 0, 2, 1]}
                        : tensor<1x3x297x297xf16> -> tensor<1x3x300x300xf16>
    %filters = const.Declare tensor<16x3x3x3xf16> = dense<1.0> : tensor<16x3x3x3xf16>
    %1 = IE.Convolution(%0, %filters)
        {
            strides = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            dilations = [1, 1]
        } :
        tensor<1x3x300x300xf16>, tensor<16x3x3x3xf16> -> tensor<1x16x300x300xf16>

    %2 = const.Declare tensor<1x3x1x1xf16> = dense<1.0> : tensor<1x3x1x1xf16>
    %3 = IE.Add(%0, %2)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x300x300xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x300x300xf16>
    return %1, %3 : tensor<1x16x300x300xf16>, tensor<1x3x300x300xf16>

    // CHECK-DAG:      [[CST0:%.*]] = const.Declare tensor<1x3x1x1xf16> = dense<1.000000e+00
    // CHECK-DAG:      [[CST1:%.*]] = const.Declare tensor<16x3x3x3xf16> = dense<1.000000e+00
    // CHECK-DAG:      [[CST2:%.*]] = const.Declare tensor<1x3x297x2xf16> = dense<0.000000e+00
    // CHECK-DAG:      [[CST3:%.*]] = const.Declare tensor<1x3x297x1xf16> = dense<0.000000e+00
    // CHECK-DAG:      [[CST4:%.*]] = const.Declare tensor<1x3x1x300xf16> = dense<0.000000e+00
    // CHECK-DAG:      [[CST5:%.*]] = const.Declare tensor<1x3x2x300xf16> = dense<0.000000e+00
    // CHECK-NOT:      IE.Pad
    // CHECK:       [[VAR0:%.*]] = IE.Concat([[CST2]], %arg0, [[CST3]]) {per_axis = #IE.Concat<axis = 3 : i64>}
    // CHECK-SAME:       tensor<1x3x297x2xf16>, tensor<1x3x297x297xf16>, tensor<1x3x297x1xf16> -> tensor<1x3x297x300xf16>
    // CHECK:       [[VAR1:%.*]] = IE.Concat([[CST4]], [[VAR0]], [[CST5]]) {per_axis = #IE.Concat<axis = 2 : i64>}
    // CHECK-SAME:       tensor<1x3x1x300xf16>, tensor<1x3x297x300xf16>, tensor<1x3x2x300xf16> -> tensor<1x3x300x300xf16>
    // CHECK:       [[VAR2:%.*]] = IE.Convolution([[VAR1]], [[CST1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:       tensor<1x3x300x300xf16>, tensor<16x3x3x3xf16> -> tensor<1x16x300x300xf16>
    // CHECK:       [[VAR3:%.*]] = IE.Add([[VAR1]], [[CST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x300x300xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x300x300xf16>
    // CHECK:       return [[VAR2]], [[VAR3]] : tensor<1x16x300x300xf16>, tensor<1x3x300x300xf16>
}

// CHECK-LABEL: @convertPadToConcatWith3DShape
func.func @convertPadToConcatWith3DShape(%arg0: tensor<1x96x21499xf16>) -> tensor<1x96x21500xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0], pads_end_attr = [0, 0, 1]}
                        : tensor<1x96x21499xf16> -> tensor<1x96x21500xf16>

    return %0 : tensor<1x96x21500xf16>

    // CHECK-DAG:      [[CST0:%.*]] = const.Declare tensor<1x96x1xf16> = dense<0.000000e+00> : tensor<1x96x1xf32>, [#const.CastElemType<f16>]
    // CHECK-NOT:      IE.Pad
    // CHECK:       [[VAR0:%.*]] = IE.Concat(%arg0, [[CST0]]) {per_axis = #IE.Concat<axis = 2 : i64>}
    // CHECK-SAME:       tensor<1x96x21499xf16>, tensor<1x96x1xf16> -> tensor<1x96x21500xf16>
    // CHECK:       return [[VAR0]] : tensor<1x96x21500xf16>
}

// CHECK-LABEL: @convertPadToConcatWith5DShape
func.func @convertPadToConcatWith5DShape(%arg0: tensor<1x16x32x64x128xf16>) -> tensor<1x17x32x64x129xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 1.000000e+00 : f64, pads_begin_attr = [0, 1, 0, 0, 0], pads_end_attr = [0, 0, 0, 0, 1]}
                        : tensor<1x16x32x64x128xf16> -> tensor<1x17x32x64x129xf16>

    return %0 : tensor<1x17x32x64x129xf16>

    // CHECK-DAG:      [[CST0:%.*]] = const.Declare tensor<1x16x32x64x1xf16> = dense<1.000000e+00> : tensor<1x16x32x64x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:      [[CST1:%.*]] = const.Declare tensor<1x1x32x64x129xf16> = dense<1.000000e+00> : tensor<1x1x32x64x129xf32>, [#const.CastElemType<f16>]

    // CHECK-NOT:      IE.Pad
    // CHECK:       [[CONCAT0:%.*]] = IE.Concat(%arg0, [[CST0]]) {per_axis = #IE.Concat<axis = 4 : i64>}
    // CHECK-SAME:       tensor<1x16x32x64x128xf16>, tensor<1x16x32x64x1xf16> -> tensor<1x16x32x64x129xf16>
    // CHECK:       [[CONCAT1:%.*]] = IE.Concat([[CST1]], [[CONCAT0]]) {per_axis = #IE.Concat<axis = 1 : i64>}
    // CHECK-SAME:       tensor<1x1x32x64x129xf16>, tensor<1x16x32x64x129xf16> -> tensor<1x17x32x64x129xf16>
    // CHECK:       return [[CONCAT1]] : tensor<1x17x32x64x129xf16>
}

// -----

!qElemTypeIn = !quant.uniform<i8:f16:1, {0.00906,0.00819}>
!qElemTypeOut = !quant.uniform<i8:f16:1, {0.00906,0.00906,0.00819,0.00819}>
// CHECK-DAG: [[QTYPE:!.+]] = !quant.uniform<i8:f16:1, {9.060000e-03,8.190000e-03}>
// CHECK-DAG: [[QTYPE1:!.+]] = !quant.uniform<i8:f16:1, {9.060000e-03,9.060000e-03,8.190000e-03,8.190000e-03}>
// CHECK-DAG: [[QTYPE2:!.+]] = !quant.uniform<i8:f16:1, {9.060000e-03}>
// CHECK-DAG: [[QTYPE3:!.+]] = !quant.uniform<i8:f16:1, {8.190000e-03}>

// Note: CHECK-LABEL must NOT be used: it resets quantization checks above such
//       that [[QTYPE*]] captured variables become undefined.
// CHECK: @convertQuantPadToConcat
// CHECK-SAME: [[ARG0:%.+]]: tensor<8x2x16x16x[[QTYPE]]>)
func.func @convertQuantPadToConcat(%arg0: tensor<8x2x16x16x!qElemTypeIn>) -> tensor<8x4x16x16x!qElemTypeOut> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 1.000000e+00 : f64, pads_begin_attr = [0, 1, 0, 0], pads_end_attr = [0, 1, 0, 0]}
                        : tensor<8x2x16x16x!qElemTypeIn> -> tensor<8x4x16x16x!qElemTypeOut>
    return %0 : tensor<8x4x16x16x!qElemTypeOut>

    // CHECK:   [[CST0:%.*]] = const.Declare tensor<8x1x16x16x[[QTYPE2]]> = dense<1.000000e+00> : tensor<8x1x16x16xf32>, [#const.CastElemType<[[QTYPE2]]>]
    // CHECK:   [[CST1:%.*]] = const.Declare tensor<8x1x16x16x[[QTYPE3]]> = dense<1.000000e+00> : tensor<8x1x16x16xf32>, [#const.CastElemType<[[QTYPE3]]>]

    // CHECK-NOT:      IE.Pad
    // CHECK:       [[CONCAT0:%.*]] = IE.Concat([[CST0]], [[ARG0]], [[CST1]]) {per_axis = #IE.Concat<axis = 1 : i64>}
    // CHECK-SAME:       tensor<8x1x16x16x[[QTYPE2]]>, tensor<8x2x16x16x[[QTYPE]]>, tensor<8x1x16x16x[[QTYPE3]]> -> tensor<8x4x16x16x[[QTYPE1]]>
    // CHECK:       return [[CONCAT0]] : tensor<8x4x16x16x[[QTYPE1]]>
}

// -----

!qElemTypeIn = !quant.uniform<i8:f16:1, {0.00906,0.00819}>
!qElemTypeOut = !quant.uniform<i8:f16:1, {0.00906,0.00906,0.00906,0.00819}>

// CHECK-DAG: [[QTYPE:!.+]] = !quant.uniform<i8:f16:1, {9.060000e-03,8.190000e-03}>
// CHECK-DAG: [[QTYPE1:!.+]] = !quant.uniform<i8:f16:1, {9.060000e-03,9.060000e-03,9.060000e-03,8.190000e-03}>
// CHECK-DAG: [[QTYPE2:!.+]] = !quant.uniform<i8:f16:1, {9.060000e-03,9.060000e-03}>

// Note: CHECK-LABEL must NOT be used: it resets quantization checks above such
//       that [[QTYPE*]] captured variables become undefined.
// CHECK: @convertQuantPadToConcatLeft
// CHECK-SAME: [[ARG0:%.+]]: tensor<8x2x16x16x[[QTYPE]]>)
func.func @convertQuantPadToConcatLeft(%arg0: tensor<8x2x16x16x!qElemTypeIn>) -> tensor<8x4x16x16x!qElemTypeOut> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 1.000000e+00 : f64, pads_begin_attr = [0, 2, 0, 0], pads_end_attr = [0, 0, 0, 0]}
                        : tensor<8x2x16x16x!qElemTypeIn> -> tensor<8x4x16x16x!qElemTypeOut>
    return %0 : tensor<8x4x16x16x!qElemTypeOut>

    // CHECK:   [[CST0:%.*]] = const.Declare tensor<8x2x16x16x[[QTYPE2]]> = dense<1.000000e+00> : tensor<8x2x16x16xf32>, [#const.CastElemType<[[QTYPE2]]>]

    // CHECK-NOT:      IE.Pad
    // CHECK:       [[CONCAT0:%.*]] = IE.Concat([[CST0]], [[ARG0]]) {per_axis = #IE.Concat<axis = 1 : i64>}
    // CHECK-SAME:       tensor<8x2x16x16x[[QTYPE2]]>, tensor<8x2x16x16x[[QTYPE]]> -> tensor<8x4x16x16x[[QTYPE1]]>
    // CHECK:       return [[CONCAT0]] : tensor<8x4x16x16x[[QTYPE1]]>
}

// -----

!qElemTypeIn = !quant.uniform<i8:f16:1, {0.00906,0.00819}>
!qElemTypeOut = !quant.uniform<i8:f16:1, {0.00906,0.00819,0.00819,0.00819}>

// CHECK-DAG: [[QTYPE:!.+]] = !quant.uniform<i8:f16:1, {9.060000e-03,8.190000e-03}>
// CHECK-DAG: [[QTYPE1:!.+]] = !quant.uniform<i8:f16:1, {9.060000e-03,8.190000e-03,8.190000e-03,8.190000e-03}>
// CHECK-DAG: [[QTYPE2:!.+]] = !quant.uniform<i8:f16:1, {8.190000e-03,8.190000e-03}>

// Note: CHECK-LABEL must NOT be used: it resets quantization checks above such
//       that [[QTYPE*]] captured variables become undefined.
// CHECK: @convertQuantPadToConcatRight
// CHECK-SAME: [[ARG0:%.+]]: tensor<8x2x16x16x[[QTYPE]]>)
func.func @convertQuantPadToConcatRight(%arg0: tensor<8x2x16x16x!qElemTypeIn>) -> tensor<8x4x16x16x!qElemTypeOut> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 1.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 2, 0, 0]}
                        : tensor<8x2x16x16x!qElemTypeIn> -> tensor<8x4x16x16x!qElemTypeOut>
    return %0 : tensor<8x4x16x16x!qElemTypeOut>

    // CHECK:   [[CST0:%.*]] = const.Declare tensor<8x2x16x16x[[QTYPE2]]> = dense<1.000000e+00> : tensor<8x2x16x16xf32>, [#const.CastElemType<[[QTYPE2]]>]

    // CHECK-NOT:      IE.Pad
    // CHECK:       [[CONCAT0:%.*]] = IE.Concat([[ARG0]], [[CST0]]) {per_axis = #IE.Concat<axis = 1 : i64>}
    // CHECK-SAME:       tensor<8x2x16x16x[[QTYPE]]>, tensor<8x2x16x16x[[QTYPE2]]> -> tensor<8x4x16x16x[[QTYPE1]]>
    // CHECK:       return [[CONCAT0]] : tensor<8x4x16x16x[[QTYPE1]]>
}

// -----

!qElemTypeIn = !quant.uniform<i8:f16:1, {1.00236,2.00906,3.00819,4.00768}>
!qElemTypeOut = !quant.uniform<i8:f16:1, {1.00236,1.00236,1.00236,2.00906,3.00819,4.00768,4.00768,4.00768}>
// CHECK-DAG: [[QTYPE:!.+]] = !quant.uniform<i8:f16:1, {1.002360e+00,2.009060e+00,3.008190e+00,4.007680e+00}>
// CHECK-DAG: [[QTYPE1:!.+]] = !quant.uniform<i8:f16:1, {1.002360e+00,1.002360e+00,1.002360e+00,2.009060e+00,3.008190e+00,4.007680e+00,4.007680e+00,4.007680e+00}>
// CHECK-DAG: [[QTYPE2:!.+]] = !quant.uniform<i8:f16:1, {1.002360e+00,1.002360e+00}>
// CHECK-DAG: [[QTYPE3:!.+]] = !quant.uniform<i8:f16:1, {4.007680e+00,4.007680e+00}>

// Note: CHECK-LABEL must NOT be used: it resets quantization checks above such
//       that [[QTYPE*]] captured variables become undefined.
// CHECK: @convertQuantPadToConcatBeginEndEqualToTwo
// CHECK-SAME: [[ARG0:%.+]]: tensor<8x4x16x16x[[QTYPE]]>)
func.func @convertQuantPadToConcatBeginEndEqualToTwo(%arg0: tensor<8x4x16x16x!qElemTypeIn>) -> tensor<8x8x16x16x!qElemTypeOut> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 1.000000e+00 : f64, pads_begin_attr = [0, 2, 0, 0], pads_end_attr = [0, 2, 0, 0]}
                        : tensor<8x4x16x16x!qElemTypeIn> -> tensor<8x8x16x16x!qElemTypeOut>
    return %0 : tensor<8x8x16x16x!qElemTypeOut>

    // CHECK:   [[CST0:%.*]] = const.Declare tensor<8x2x16x16x[[QTYPE2]]> = dense<1.000000e+00> : tensor<8x2x16x16xf32>, [#const.CastElemType<[[QTYPE2]]>]
    // CHECK:   [[CST1:%.*]] = const.Declare tensor<8x2x16x16x[[QTYPE3]]> = dense<1.000000e+00> : tensor<8x2x16x16xf32>, [#const.CastElemType<[[QTYPE3]]>]

    // CHECK-NOT:      IE.Pad
    // CHECK:       [[CONCAT0:%.*]] = IE.Concat([[CST0]], [[ARG0]], [[CST1]]) {per_axis = #IE.Concat<axis = 1 : i64>}
    // CHECK-SAME:       tensor<8x2x16x16x[[QTYPE2]]>, tensor<8x4x16x16x[[QTYPE]]>, tensor<8x2x16x16x[[QTYPE3]]> -> tensor<8x8x16x16x[[QTYPE1]]>
    // CHECK:       return [[CONCAT0]] : tensor<8x8x16x16x[[QTYPE1]]>
}

// -----

!qElemType = !quant.uniform<i8:f16, 0.5:120>
// CHECK-DAG: [[QTYPE:!.+]] = !quant.uniform<i8:f16, 5.000000e-01:120>

// Note: CHECK-LABEL must NOT be used: it resets quantization checks above such
//       that [[QTYPE*]] captured variables become undefined.
// CHECK: @convertQuantPadToConcatPerTensor
// CHECK-SAME: [[ARG0:%.+]]: tensor<8x2x16x16x[[QTYPE]]>)
func.func @convertQuantPadToConcatPerTensor(%arg0: tensor<8x2x16x16x!qElemType>) -> tensor<8x4x16x16x!qElemType> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 1.000000e+00 : f64, pads_begin_attr = [0, 1, 0, 0], pads_end_attr = [0, 1, 0, 0]}
                        : tensor<8x2x16x16x!qElemType> -> tensor<8x4x16x16x!qElemType>
    return %0 : tensor<8x4x16x16x!qElemType>

    // CHECK:   [[CST0:%.*]] = const.Declare tensor<8x1x16x16x[[QTYPE]]> = dense<1.000000e+00> : tensor<8x1x16x16xf32>, [#const.CastElemType<[[QTYPE]]>]

    // CHECK-NOT:      IE.Pad
    // CHECK:       [[CONCAT0:%.*]] = IE.Concat([[CST0]], [[ARG0]], [[CST0]]) {per_axis = #IE.Concat<axis = 1 : i64>}
    // CHECK-SAME:       tensor<8x1x16x16x[[QTYPE]]>, tensor<8x2x16x16x[[QTYPE]]>, tensor<8x1x16x16x[[QTYPE]]> -> tensor<8x4x16x16x[[QTYPE]]>
    // CHECK:       return [[CONCAT0]] : tensor<8x4x16x16x[[QTYPE]]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i8:f16, 0.5:120>
// CHECK-DAG: [[QTYPE:!.+]] = !quant.uniform<i8:f16, 5.000000e-01:120>

// Note: CHECK-LABEL must NOT be used: it resets quantization checks above such
//       that [[QTYPE*]] captured variables become undefined.
// CHECK: @convertQuantPadToConcatPerTensorNonDefaultLayout
// CHECK-SAME: [[ARG0:%.+]]: tensor<8x2x16x16x[[QTYPE]], {order = #NHWC}>)
func.func @convertQuantPadToConcatPerTensorNonDefaultLayout(%arg0: tensor<8x2x16x16x!qElemType, {order = #NHWC}>) -> tensor<8x4x16x16x!qElemType, {order = #NHWC}> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 1.000000e+00 : f64, pads_begin_attr = [0, 1, 0, 0], pads_end_attr = [0, 1, 0, 0]}
                        : tensor<8x2x16x16x!qElemType, {order = #NHWC}> -> tensor<8x4x16x16x!qElemType, {order = #NHWC}>
    return %0 : tensor<8x4x16x16x!qElemType, {order = #NHWC}>

    // CHECK:   [[CST0:%.*]] = const.Declare tensor<8x1x16x16x[[QTYPE]], {order = #NHWC}> = dense<1.000000e+00> : tensor<8x1x16x16xf32>, [#const.CastElemType<[[QTYPE]]>, #const.Reorder<#NHWC>]

    // CHECK-NOT:      IE.Pad
    // CHECK:       [[CONCAT0:%.*]] = IE.Concat([[CST0]], [[ARG0]], [[CST0]]) {per_axis = #IE.Concat<axis = 1 : i64>}
    // CHECK-SAME:       tensor<8x1x16x16x[[QTYPE]], {order = #NHWC}>, tensor<8x2x16x16x[[QTYPE]], {order = #NHWC}>, tensor<8x1x16x16x[[QTYPE]], {order = #NHWC}> -> tensor<8x4x16x16x[[QTYPE]], {order = #NHWC}>
    // CHECK:       return [[CONCAT0]] : tensor<8x4x16x16x[[QTYPE]], {order = #NHWC}>
}
