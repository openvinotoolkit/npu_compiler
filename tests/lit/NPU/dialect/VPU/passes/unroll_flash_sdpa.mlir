//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --unroll-flash-sdpa %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @FlashSDPA8kSeqLenOneQueryTile
// CHECK-SAME: [[QUERY:%[^, ]+]]: tensor<1x1x8192x32xf16>,
// CHECK-SAME: [[KEY:%[^, ]+]]: tensor<1x1x128x32xf16>,
// CHECK-SAME: [[VALUE:%[^, ]+]]: tensor<1x1x128x64xf16>,
// CHECK-SAME: [[ATTENTION_MASK:%[^, ]+]]: tensor<1x1x8192x128xf16>,
// CHECK-SAME: [[SCALE:%[^, ]+]]: tensor<1x1x1x1xf16>
func.func @FlashSDPA8kSeqLenOneQueryTile(%arg0: tensor<1x1x8192x32xf16>, %arg1: tensor<1x1x128x32xf16>, %arg2: tensor<1x1x128x64xf16>, %arg3: tensor<1x1x8192x128xf16>, %arg4: tensor<1x1x1x1xf16>)
                                  -> tensor<1x1x2731x64xf16> {
    %cst_7 = const.Declare tensor<1x1x1x2731xf32> = dense<0.000000e+00> : tensor<1x1x1x8192xf32>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 2731]>]
    %cst_8 = const.Declare tensor<1x1x1x2731xf32> = dense<0xFF800000> : tensor<1x1x1x8192xf32>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 2731]>]
    %cst_9 = const.Declare tensor<1x1x2731x64xf16> = dense<0.000000e+00> : tensor<1x1x8192x64xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 2731, 64]>]
    %cst_10 = const.Declare tensor<1x1x2731x128xf16> = dense<0.000000e+00> : tensor<1x1x8192x128xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 2731, 128]>]

    %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 1, 2731, 32] : tensor<1x1x8192x32xf16> to tensor<1x1x2731x32xf16>
    %1 = VPU.Slice %arg3 [0, 0, 0, 0] [1, 1, 2731, 128] : tensor<1x1x8192x128xf16> to tensor<1x1x2731x128xf16>
    %result_running_output, %result_running_max, %result_running_sum, %result_query = VPU.FlashSDPA(%0, %arg1, %arg2, %cst_10, %cst_9, %cst_8, %cst_7, %1, %arg4)
            {is_head = true, is_tail = true, kv_num_blocks = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 1>}
            : tensor<1x1x2731x32xf16>, tensor<1x1x128x32xf16>, tensor<1x1x128x64xf16>, tensor<1x1x2731x128xf16>, tensor<1x1x2731x64xf16>,
              tensor<1x1x1x2731xf32>, tensor<1x1x1x2731xf32>, tensor<1x1x2731x128xf16>, tensor<1x1x1x1xf16>
            -> tensor<1x1x2731x64xf16>, tensor<1x1x1x2731xf32>, tensor<1x1x1x2731xf32>, tensor<1x1x2731x32xf16>

    return %result_running_output : tensor<1x1x2731x64xf16>

    // CHECK-DAG:       [[IN_SUM0:%.+]] = const.Declare tensor<1x1x1x2731xf32> = dense<0.000000e+00> : tensor<1x1x1x8192xf32>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 2731]>]
    // CHECK-DAG:       [[IN_MAX0:%.+]] = const.Declare tensor<1x1x1x2731xf32> = dense<0xFF800000> : tensor<1x1x1x8192xf32>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 2731]>]
    // CHECK-DAG:       [[IN_OUT0:%.+]] = const.Declare tensor<1x1x2731x64xf16> = dense<0.000000e+00> : tensor<1x1x8192x64xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 2731, 64]>]
    // CHECK-DAG:       [[IN_AUX_LEFTOVER:%.+]] = const.Declare tensor<1x1x2731x128xf16> = dense<0.000000e+00> : tensor<1x1x8192x128xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 2731, 128]>]

    // CHECK:           [[QUERY_SLICE:%.+]] = VPU.Slice [[QUERY]] [0, 0, 0, 0] [1, 1, 2731, 32] : tensor<1x1x8192x32xf16> to tensor<1x1x2731x32xf16>
    // CHECK:           [[ATTENTION_MASK_SLICE:%.+]] = VPU.Slice [[ATTENTION_MASK]] [0, 0, 0, 0] [1, 1, 2731, 128] : tensor<1x1x8192x128xf16> to tensor<1x1x2731x128xf16>

    // CHECK:           [[KEY0:%.+]] = VPU.Slice [[KEY]] [0, 0, 0, 0] [1, 1, 64, 32] : tensor<1x1x128x32xf16> to tensor<1x1x64x32xf16>
    // CHECK:           [[VALUE0:%.+]] = VPU.Slice [[VALUE]] [0, 0, 0, 0] [1, 1, 64, 64] : tensor<1x1x128x64xf16> to tensor<1x1x64x64xf16>
    // CHECK:           [[ATTENTION_MASK0:%.+]] = VPU.Slice [[ATTENTION_MASK_SLICE]] [0, 0, 0, 0] [1, 1, 2731, 64] : tensor<1x1x2731x128xf16> to tensor<1x1x2731x64xf16>
    // CHECK:           [[IN_AUX0:%.+]] = const.Declare tensor<1x1x2731x64xf16> = dense<0.000000e+00> : tensor<1x1x2731x64xf16>

    // CHECK:           [[RES_OUT0:%.+]], [[RES_MAX0:%.+]], [[RES_SUM0:%.+]], [[RES_QUERY0:%.+]] = VPU.FlashSDPA
    // CHECK-SAME:          ([[QUERY_SLICE]], [[KEY0]], [[VALUE0]], [[IN_AUX0]], [[IN_OUT0]], [[IN_MAX0]], [[IN_SUM0]], [[ATTENTION_MASK0]], [[SCALE]])
    // CHECK-SAME:          {is_head = true, is_tail = false
    // CHECK-NOT:           kv_num_blocks
    // CHECK-SAME:          -> tensor<1x1x2731x64xf16>, tensor<1x1x1x2731xf32>, tensor<1x1x1x2731xf32>, tensor<1x1x2731x32xf16>

    // CHECK:           [[KEY1:%.+]] = VPU.Slice [[KEY]] [0, 0, 64, 0] [1, 1, 64, 32] : tensor<1x1x128x32xf16> to tensor<1x1x64x32xf16>
    // CHECK:           [[VALUE1:%.+]] = VPU.Slice [[VALUE]] [0, 0, 64, 0] [1, 1, 64, 64] : tensor<1x1x128x64xf16> to tensor<1x1x64x64xf16>
    // CHECK:           [[ATTENTION_MASK1:%.+]] = VPU.Slice [[ATTENTION_MASK_SLICE]] [0, 0, 0, 64] [1, 1, 2731, 64] : tensor<1x1x2731x128xf16> to tensor<1x1x2731x64xf16>
    // CHECK:           [[IN_AUX1:%.+]] = const.Declare tensor<1x1x2731x64xf16> = dense<0.000000e+00> : tensor<1x1x2731x64xf16>

    // CHECK:           [[RES_OUT1:%.+]], [[RES_MAX1:%.+]], [[RES_SUM1:%.+]], [[RES_QUERY1:%.+]] = VPU.FlashSDPA
    // CHECK-SAME:          ([[RES_QUERY0]], [[KEY1]], [[VALUE1]], [[IN_AUX1]], [[RES_OUT0]], [[RES_MAX0]], [[RES_SUM0]], [[ATTENTION_MASK1]], [[SCALE]])
    // CHECK-SAME:          {is_head = false, is_tail = true
    // CHECK-NOT:           kv_num_blocks
    // CHECK-SAME:          -> tensor<1x1x2731x64xf16>, tensor<1x1x1x2731xf32>, tensor<1x1x1x2731xf32>, tensor<1x1x2731x32xf16>

    // CHECK:           return [[RES_OUT1]]
}
