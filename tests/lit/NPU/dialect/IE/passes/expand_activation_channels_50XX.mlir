//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --expand-activation-channels --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX

// CHECK-LABEL: @FlashSDPA
// CHECK-SAME: [[QUERY:%[^, ]+]]: tensor<1x16x117x85xf16>,
// CHECK-SAME: [[KEY:%[^, ]+]]: tensor<1x16x249x85xf16>,
// CHECK-SAME: [[VALUE:%[^, ]+]]: tensor<1x16x87x249xf16>,
// CHECK-SAME: [[ATTENTION_MASK:%[^, ]+]]: tensor<1x16x117x249xf16>,
// CHECK-SAME: [[SCALE:%[^, ]+]]: tensor<1x1x1x1xf16>
func.func @FlashSDPA(%arg0: tensor<1x16x117x85xf16>, %arg1: tensor<1x16x249x85xf16>, %arg2: tensor<1x16x87x249xf16>, %arg3: tensor<1x16x117x249xf16>, %arg4: tensor<1x1x1x1xf16>) -> tensor<1x16x117x87xf16> {
    %0 = IE.FlashSDPA(%arg0, %arg1, %arg2, %arg3, %arg4) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>, source_seq_len_pad_size = 0 : i64}
            : tensor<1x16x117x85xf16>, tensor<1x16x249x85xf16>, tensor<1x16x87x249xf16>, tensor<1x16x117x249xf16>, tensor<1x1x1x1xf16>
            -> tensor<1x16x117x87xf16>

    return %0 : tensor<1x16x117x87xf16>

    // CHECK-DAG:   [[QUERY_PAD:%.+]] = IE.Expand([[QUERY]]) {
    // CHECK-SAME:          pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 11]
    // CHECK-SAME:      } : tensor<1x16x117x85xf16> -> tensor<1x16x117x96xf16>


    // CHECK-DAG:   [[KEY_PAD_SEQ_LEN:%.+]] = IE.Expand([[KEY]]) {
    // CHECK-SAME:          pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 7, 0]
    // CHECK-SAME:      } : tensor<1x16x249x85xf16> -> tensor<1x16x256x85xf16>

    // CHECK-DAG:   [[KEY_PAD:%.+]] = IE.Expand([[KEY_PAD_SEQ_LEN]])
    // CHECK-SAME:          pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 11]
    // CHECK-SAME:      } : tensor<1x16x256x85xf16> -> tensor<1x16x256x96xf16>


    // CHECK-DAG:   [[VALUE_PAD_SEQ_LEN:%.+]] = IE.Expand([[VALUE]]) {
    // CHECK-SAME:          pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 9, 0]
    // CHECK-SAME:      } : tensor<1x16x87x249xf16> -> tensor<1x16x96x249xf16>

    // CHECK-DAG:   [[VALUE_PAD:%.+]] = IE.Expand([[VALUE_PAD_SEQ_LEN]]) {
    // CHECK-SAME:          pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 7]
    // CHECK-SAME:      } : tensor<1x16x96x249xf16> -> tensor<1x16x96x256xf16>


    // CHECK-DAG:   [[ATTENTION_MASK_PAD:%.+]] = IE.Expand([[ATTENTION_MASK]]) {
    // CHECK-SAME:          pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 7]
    // CHECK-SAME:      } : tensor<1x16x117x249xf16> -> tensor<1x16x117x256xf16>


    // CHECK:       [[FLASH_SDPA:%.+]] = IE.FlashSDPA([[QUERY_PAD]], [[KEY_PAD]], [[VALUE_PAD]], [[ATTENTION_MASK_PAD]], [[SCALE]])
    // CHECK-SAME:          operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>,
    // CHECK-SAME:          source_seq_len_pad_size = 7 : i64
    // CHECK-SAME:      -> tensor<1x16x117x96xf16>


    // CHECK:       [[RESULT:%.+]] = IE.Slice [[FLASH_SDPA]]
    // CHECK-SAME:          [0, 0, 0, 0] [1, 16, 117, 87]
    // CHECK-SAME:      : tensor<1x16x117x96xf16> to tensor<1x16x117x87xf16>

    // CHECK:       return [[RESULT]]
}
