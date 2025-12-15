//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --flash-sdpa-tiling-strategy-estimation %s | FileCheck %s
// REQUIRES: arch-NPU50XX

// CHECK-LABEL: @FlashSDPA8kSeqLen
func.func @FlashSDPA8kSeqLen(%arg0: tensor<1x8x8192x32xf16>, %arg1: tensor<1x8x128x32xf16>, %arg2: tensor<1x8x64x128xf16>, %arg3: tensor<1x8x8192x128xf16>, %arg4: tensor<1x1x1x1xf16>)
                                  -> tensor<1x8x8192x64xf16> {
    %cst = const.Declare tensor<1x1x64x4xsi32> = dense<0> : tensor<1x1x64x4xsi32>
    %cst_0 = const.Declare tensor<1x1x128x4xsi32> = dense<0> : tensor<1x1x128x4xsi32>
    %cst_1 = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
    %cst_2 = const.Declare tensor<1x8x8192x128xf16> = dense<0.000000e+00> : tensor<1x8x8192x128xf16>
    %cst_3 = const.Declare tensor<1x8x8192x1xf16> = dense<0.000000e+00> : tensor<1x8x8192x1xf16>
    %cst_4 = const.Declare tensor<1x8x8192x1xf16> = dense<0xFC00> : tensor<1x8x8192x1xf16>
    %cst_5 = const.Declare tensor<1x8x8192x64xf16> = dense<0.000000e+00> : tensor<1x8x8192x64xf16>

    %result_running_output, %result_running_max, %result_running_sum, %result_query =
        VPU.FlashSDPA(%arg0, %arg1, %arg2, %cst_2, %cst_1, %cst_0, %cst, %cst_5, %cst_4, %cst_3, %arg3, %arg4) {
            is_head = true,
            is_tail = true,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>,
            source_seq_len_pad_size = 0 : i64
        } : tensor<1x8x8192x32xf16>, tensor<1x8x128x32xf16>, tensor<1x8x64x128xf16>,
            tensor<1x8x8192x128xf16>, tensor<1x1x2x256xsi32>, tensor<1x1x128x4xsi32>,
            tensor<1x1x64x4xsi32>, tensor<1x8x8192x64xf16>, tensor<1x8x8192x1xf16>,
            tensor<1x8x8192x1xf16>, tensor<1x8x8192x128xf16>, tensor<1x1x1x1xf16>
        -> tensor<1x8x8192x64xf16>, tensor<1x8x8192x1xf16>, tensor<1x8x8192x1xf16>, tensor<1x8x8192x32xf16>

    return %result_running_output : tensor<1x8x8192x64xf16>

    // CHECK:       VPU.FlashSDPA
    // CHECK-SAME:  kv_num_blocks = 8
}

// -----

// CHECK-LABEL: @FlashSDPAKeyValueTilingWithNoQueryTiling
func.func @FlashSDPAKeyValueTilingWithNoQueryTiling(%arg0: tensor<1x1x1x128xf16>, %arg1: tensor<1x1x4096x128xf16>, %arg2: tensor<1x1x128x4096xf16>, %arg3: tensor<1x1x1x4096xf16>) -> tensor<1x1x1x128xf16> {
    %cst = const.Declare tensor<1x1x128x4xsi32> = dense<0> : tensor<1x1x128x4xsi32>
    %cst_0 = const.Declare tensor<1x1x4096x4xsi32> = dense<0> : tensor<1x1x4096x4xsi32>
    %cst_1 = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
    %cst_2 = const.Declare tensor<1x1x1x4096xf16> = dense<0.000000e+00> : tensor<1x1x1x4096xf16>
    %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_4 = const.Declare tensor<1x1x1x1xf16> = dense<0xFC00> : tensor<1x1x1x1xf16>
    %cst_5 = const.Declare tensor<1x1x1x128xf16> = dense<0.000000e+00> : tensor<1x1x1x128xf16>

    %result_running_output, %result_running_max, %result_running_sum, %result_query =
        VPU.FlashSDPA(%arg0, %arg1, %arg2, %cst_2, %cst_1, %cst_0, %cst, %cst_5, %cst_4, %cst_3, %arg3) {
            is_head = true,
            is_tail = true,
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
            operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0>,
            source_seq_len_pad_size = 0 : i64
        } : tensor<1x1x1x128xf16>, tensor<1x1x4096x128xf16>, tensor<1x1x128x4096xf16>,
            tensor<1x1x1x4096xf16>, tensor<1x1x2x256xsi32>, tensor<1x1x4096x4xsi32>,
            tensor<1x1x128x4xsi32>, tensor<1x1x1x128xf16>, tensor<1x1x1x1xf16>,
            tensor<1x1x1x1xf16>, tensor<1x1x1x4096xf16>
        -> tensor<1x1x1x128xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x128xf16>

    return %result_running_output : tensor<1x1x1x128xf16>

    // CHECK:       VPU.FlashSDPA
    // CHECK-SAME:  kv_num_blocks = 2
}
