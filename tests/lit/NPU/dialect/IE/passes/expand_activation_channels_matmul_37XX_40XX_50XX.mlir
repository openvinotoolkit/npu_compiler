//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --expand-activation-channels %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

!input1ElemType = !quant.uniform<u8:f16, 0.1:1>
!input2ElemType = !quant.uniform<u8:f16, 0.2:2>
!outputElemType = !quant.uniform<u8:f16, 0.3:3>
// CHECK: [[INPUT1_ELEM_TYPE:!.+]] = !quant.uniform<u8:f16, 1.000000e-01:1>
// CHECK: [[INPUT2_ELEM_TYPE:!.+]] = !quant.uniform<u8:f16, 2.000000e-01:2>
// CHECK: [[OUTPUT_ELEM_TYPE:!.+]] = !quant.uniform<u8:f16, 3.000000e-01:3>

// CHECK:       func.func @ExpandOCMatMulQuant([[INPUT1:%.+]]: tensor<1x32x75x16x[[INPUT1_ELEM_TYPE]]>,
// CHECK-SAME:  [[INPUT2:%.+]]: tensor<1x32x75x16x[[INPUT2_ELEM_TYPE]]>) ->  tensor<1x32x75x75x[[OUTPUT_ELEM_TYPE]]> {
func.func @ExpandOCMatMulQuant(%input1: tensor<1x32x75x16x!input1ElemType>, %input2: tensor<1x32x75x16x!input2ElemType>) -> tensor<1x32x75x75x!outputElemType> {
    %matmul = IE.MatMul(%input1, %input2) {transpose_b} : tensor<1x32x75x16x!input1ElemType>, tensor<1x32x75x16x!input2ElemType> -> tensor<1x32x75x75x!outputElemType>

    return %matmul : tensor<1x32x75x75x!outputElemType>
    // CHECK:           [[EXPAND:%.+]] = IE.Expand([[INPUT2]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 5, 0]}
    // CHECK-SAME:      : tensor<1x32x75x16x[[INPUT2_ELEM_TYPE]]> -> tensor<1x32x80x16x[[INPUT2_ELEM_TYPE]]>
    // CHECK:           [[MATMUL:%.+]] = IE.MatMul([[INPUT1]], [[EXPAND]]) {transpose_b}
    // CHECK-SAME:      : tensor<1x32x75x16x[[INPUT1_ELEM_TYPE]]>, tensor<1x32x80x16x[[INPUT2_ELEM_TYPE]]> -> tensor<1x32x75x80x[[OUTPUT_ELEM_TYPE]]>
    // CHECK:           [[SLICE:%.+]] = IE.Slice [[MATMUL]] [0, 0, 0, 0] [1, 32, 75, 75]
    // CHECK-SAME:      : tensor<1x32x75x80x[[OUTPUT_ELEM_TYPE]]> to tensor<1x32x75x75x[[OUTPUT_ELEM_TYPE]]>
    // CHECK:           return [[SLICE]] : tensor<1x32x75x75x[[OUTPUT_ELEM_TYPE]]>
}
