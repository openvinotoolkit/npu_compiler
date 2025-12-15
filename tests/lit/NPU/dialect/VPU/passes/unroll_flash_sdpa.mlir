//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --unroll-flash-sdpa %s | FileCheck %s
// REQUIRES: arch-NPU50XX

// CHECK-LABEL: @FlashSDPA8kSeqLenOneQueryTile
// CHECK-SAME: [[QUERY:%[^, ]+]]: tensor<1x1x128x32xf16>,
// CHECK-SAME: [[KEY:%[^, ]+]]: tensor<1x1x4321x32xf16>,
// CHECK-SAME: [[VALUE:%[^, ]+]]: tensor<1x1x64x4321xf16>,
// CHECK-SAME: [[ATTENTION_MASK:%[^, ]+]]: tensor<1x1x128x4321xf16>,
// CHECK-SAME: [[SCALE:%[^, ]+]]: tensor<1x1x1x1xf16>
func.func @FlashSDPA8kSeqLenOneQueryTile(%arg0: tensor<1x1x128x32xf16>, %arg1: tensor<1x1x4321x32xf16>, %arg2: tensor<1x1x64x4321xf16>, %arg3: tensor<1x1x128x4321xf16>, %arg4: tensor<1x1x1x1xf16>) -> tensor<1x1x128x64xf16> {
    %cst = const.Declare tensor<1x1x64x4xsi32> = dense<0> : tensor<1x1x64x4xsi32>
    %cst_0 = const.Declare tensor<1x1x4336x4xsi32> = dense<0> : tensor<1x1x4336x4xsi32>
    %cst_1 = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
    %cst_2 = const.Declare tensor<1x1x128x4336xf16> = dense<0.000000e+00> : tensor<1x1x128x4336xf16>
    %cst_3 = const.Declare tensor<1x1x128x1xf16> = dense<0.000000e+00> : tensor<1x1x128x1xf16>
    %cst_4 = const.Declare tensor<1x1x128x1xf16> = dense<0xFC00> : tensor<1x1x128x1xf16>
    %cst_5 = const.Declare tensor<1x1x128x64xf16> = dense<0.000000e+00> : tensor<1x1x128x64xf16>

    %key_padded = VPU.Expand(%arg1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 15, 0]} : tensor<1x1x4321x32xf16> -> tensor<1x1x4336x32xf16>
    %value_padded = VPU.Expand(%arg2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 15]} : tensor<1x1x64x4321xf16> -> tensor<1x1x64x4336xf16>
    %mask_padded = VPU.Expand(%arg3) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 15]} : tensor<1x1x128x4321xf16> -> tensor<1x1x128x4336xf16>

    %result_running_output, %result_running_max, %result_running_sum, %result_query =
        VPU.FlashSDPA(%arg0, %key_padded, %value_padded, %cst_2, %cst_1, %cst_0, %cst, %cst_5, %cst_4, %cst_3, %mask_padded, %arg4) {
            is_head = true,
            is_tail = true,
            kv_num_blocks = 2 : i64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>,
            source_seq_len_pad_size = 15 : i64
        } : tensor<1x1x128x32xf16>, tensor<1x1x4336x32xf16>, tensor<1x1x64x4336xf16>,
            tensor<1x1x128x4336xf16>, tensor<1x1x2x256xsi32>, tensor<1x1x4336x4xsi32>,
            tensor<1x1x64x4xsi32>, tensor<1x1x128x64xf16>, tensor<1x1x128x1xf16>,
            tensor<1x1x128x1xf16>, tensor<1x1x128x4336xf16>, tensor<1x1x1x1xf16>
        -> tensor<1x1x128x64xf16>, tensor<1x1x128x1xf16>, tensor<1x1x128x1xf16>, tensor<1x1x128x32xf16>

    return %result_running_output : tensor<1x1x128x64xf16>

    // CHECK-DAG:       [[IN_SUM0:%.+]] = const.Declare tensor<1x1x128x1xf16> = dense<0.000000e+00> : tensor<1x1x128x1xf16>
    // CHECK-DAG:       [[IN_MAX0:%.+]] = const.Declare tensor<1x1x128x1xf16> = dense<0xFC00> : tensor<1x1x128x1xf16>
    // CHECK-DAG:       [[IN_OUT0:%.+]] = const.Declare tensor<1x1x128x64xf16> = dense<0.000000e+00> : tensor<1x1x128x64xf16>

    // CHECK:           [[KEY_PADDED:%.+]] = VPU.Expand([[KEY]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 15, 0]} : tensor<1x1x4321x32xf16> -> tensor<1x1x4336x32xf16>
    // CHECK:           [[VALUE_PADDED:%.+]] = VPU.Expand([[VALUE]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 15]} : tensor<1x1x64x4321xf16> -> tensor<1x1x64x4336xf16>
    // CHECK:           [[ATTENTION_MASK_PADDED:%.+]] = VPU.Expand([[ATTENTION_MASK]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 15]} : tensor<1x1x128x4321xf16> -> tensor<1x1x128x4336xf16>

    // CHECK:           [[KEY_SLICE0:%.+]] = VPU.Slice [[KEY_PADDED]] [0, 0, 0, 0] [1, 1, 2176, 32] : tensor<1x1x4336x32xf16> to tensor<1x1x2176x32xf16>
    // CHECK:           [[VALUE_SLICE0:%.+]] = VPU.Slice [[VALUE_PADDED]] [0, 0, 0, 0] [1, 1, 64, 2176] : tensor<1x1x64x4336xf16> to tensor<1x1x64x2176xf16>
    // CHECK:           [[ATTENTION_MASK_SLICE0:%.+]] = VPU.Slice [[ATTENTION_MASK_PADDED]] [0, 0, 0, 0] [1, 1, 128, 2176] : tensor<1x1x128x4336xf16> to tensor<1x1x128x2176xf16>

    // CHECK:           [[IN_AUX0:%.+]] = const.Declare tensor<1x1x128x2176xf16> = dense<0.000000e+00> : tensor<1x1x128x2176xf16>
    // CHECK:           [[DPU_DESCRIPTORS_BUF0:%.+]] = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
    // CHECK:           [[WEIGHTS_TABLE0_0:%.+]] = const.Declare tensor<1x1x2176x4xsi32> = dense
    // CHECK:           [[WEIGHTS_TABLE1_0:%.+]] = const.Declare tensor<1x1x64x4xsi32> = dense

    // CHECK:           [[RES_OUT0:%[^, ]+]], [[RES_MAX0:%[^, ]+]], [[RES_SUM0:%[^, ]+]], [[RES_QUERY0:%[^, ]+]] =
    // CHECK-SAME:              VPU.FlashSDPA([[QUERY]], [[KEY_SLICE0]], [[VALUE_SLICE0]], [[IN_AUX0]],
    // CHECK-SAME:                            [[DPU_DESCRIPTORS_BUF0]], [[WEIGHTS_TABLE0_0]], [[WEIGHTS_TABLE1_0]],
    // CHECK-SAME:                            [[IN_OUT0]], [[IN_MAX0]], [[IN_SUM0]], [[ATTENTION_MASK_SLICE0]], [[SCALE]]) {
    // CHECK-SAME:                      is_head = true,
    // CHECK-SAME:                      is_tail = false,
    // CHECK-NOT:                       kv_num_blocks
    // CHECK-SAME:                      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:                      operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>
    // CHECK-SAME:                      source_seq_len_pad_size = 0 : i64
    // CHECK-SAME:                  ->  tensor<1x1x128x64xf16>, tensor<1x1x128x1xf16>, tensor<1x1x128x1xf16>, tensor<1x1x128x32xf16>

    // CHECK:           [[KEY_SLICE1:%.+]] = VPU.Slice [[KEY_PADDED]] [0, 0, 2176, 0] [1, 1, 2160, 32] : tensor<1x1x4336x32xf16> to tensor<1x1x2160x32xf16>
    // CHECK:           [[VALUE_SLICE1:%.+]] = VPU.Slice [[VALUE_PADDED]] [0, 0, 0, 2176] [1, 1, 64, 2160] : tensor<1x1x64x4336xf16> to tensor<1x1x64x2160xf16>
    // CHECK:           [[ATTENTION_MASK_SLICE1:%.+]] = VPU.Slice [[ATTENTION_MASK_PADDED]] [0, 0, 0, 2176] [1, 1, 128, 2160] : tensor<1x1x128x4336xf16> to tensor<1x1x128x2160xf16>

    // CHECK-DAG:       [[IN_AUX1:%.+]] = const.Declare tensor<1x1x128x2160xf16> = dense<0.000000e+00> : tensor<1x1x128x2160xf16>
    // CHECK-DAG:       [[DPU_DESCRIPTORS_BUF1:%.+]] = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
    // CHECK-DAG:       [[WEIGHTS_TABLE0_1:%.+]] = const.Declare tensor<1x1x2160x4xsi32> = dense
    // CHECK-DAG:       [[WEIGHTS_TABLE1_1:%.+]] = const.Declare tensor<1x1x64x4xsi32> = dense

    // CHECK:           [[RES_OUT1:%[^, ]+]], [[RES_MAX1:%[^, ]+]], [[RES_SUM1:%[^, ]+]], [[RES_QUERY1:%[^, ]+]] =
    // CHECK-SAME:              VPU.FlashSDPA([[RES_QUERY0]], [[KEY_SLICE1]], [[VALUE_SLICE1]], [[IN_AUX1]],
    // CHECK-SAME:                            [[DPU_DESCRIPTORS_BUF1]], [[WEIGHTS_TABLE0_1]], [[WEIGHTS_TABLE1_1]],
    // CHECK-SAME:                            [[RES_OUT0]], [[RES_MAX0]], [[RES_SUM0]], [[ATTENTION_MASK_SLICE1]], [[SCALE]]) {
    // CHECK-SAME:                      is_head = false,
    // CHECK-SAME:                      is_tail = true,
    // CHECK-NOT:                       kv_num_blocks
    // CHECK-SAME:                      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:                      source_seq_len_pad_size = 15 : i64
    // CHECK-SAME:                  ->  tensor<1x1x128x64xf16>, tensor<1x1x128x1xf16>, tensor<1x1x128x1xf16>, tensor<1x1x128x32xf16>

    // CHECK:           return [[RES_OUT1]] : tensor<1x1x128x64xf16>
}
