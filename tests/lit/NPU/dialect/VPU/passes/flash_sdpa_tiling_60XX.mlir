//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --flash-sdpa-tiling %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

config.Resources 4 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1997824 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @FlashSDPA8kSeqLen
func.func @FlashSDPA8kSeqLen(%arg0: tensor<1x8x8192x32xf16>, %arg1: tensor<1x8x128x32xf16>, %arg2: tensor<1x8x128x64xf16>, %arg3: tensor<1x8x8192x128xf16>)
                                  -> tensor<1x8x8192x64xf16> {
    %cst = const.Declare tensor<1x1x64x4xsi32> = dense<0> : tensor<1x1x64x4xsi32>
    %cst_0 = const.Declare tensor<1x1x128x4xsi32> = dense<0> : tensor<1x1x128x4xsi32>
    %cst_1 = const.Declare tensor<1x1x2x240xsi32> = dense<0> : tensor<1x1x2x240xsi32>
    %cst_2 = const.Declare tensor<1x2x8192x128xf16> = dense<0.000000e+00> : tensor<1x2x8192x128xf16>
    %cst_3 = const.Declare tensor<1x8x8192x1xf32> = dense<0.000000e+00> : tensor<1x8x8192x1xf32>
    %cst_4 = const.Declare tensor<1x8x8192x1xf16> = dense<0xFC00> : tensor<1x8x8192x1xf16>
    %cst_5 = const.Declare tensor<1x8x8192x64xf16> = dense<0.000000e+00> : tensor<1x8x8192x64xf16>

    %value_reordered = VPU.MemPermute(%arg2) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x8x128x64xf16> -> tensor<1x8x128x64xf16, {order = #NCWH}>

    %result_running_output, %result_running_max, %result_running_sum =
        VPU.FlashSDPA(%arg0, %arg1, %value_reordered, %cst_2, %cst_1, %cst_0, %cst, %cst_5, %cst_4, %cst_3, %arg3) {
            is_head = true,
            is_tail = true,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            source_seq_len_pad_size = 0 : i64
        } : tensor<1x8x8192x32xf16>, tensor<1x8x128x32xf16>, tensor<1x8x128x64xf16, {order = #NCWH}>,
            tensor<1x2x8192x128xf16>, tensor<1x1x2x240xsi32>, tensor<1x1x128x4xsi32>,
            tensor<1x1x64x4xsi32>, tensor<1x8x8192x64xf16>, tensor<1x8x8192x1xf16>,
            tensor<1x8x8192x1xf32>, tensor<1x8x8192x128xf16>
        -> tensor<1x8x8192x64xf16>, tensor<1x8x8192x1xf16>, tensor<1x8x8192x1xf32>

    return %result_running_output : tensor<1x8x8192x64xf16>

     // 60XX CMX allows head tiling plus 2 query-seq tiles of 4096 rows per head,
     // for 16 total FlashSDPA ops.
     // Head-major emission: both seq tiles of head 0 first, then head 1, etc.

    // Head 0
    // CHECK-DAG:   [[VALUE_REORDERED:%.+]] = VPU.MemPermute([[ARG:%.+]]) {dst_order = #NCWH, mem_perm = #NCWH}
    // CHECK:       [[HEAD0_Q:%.+]] = VPU.Slice [[QUERY:%.+]] [0, 0, 0, 0] [1, 1, 8192, 32] : tensor<1x8x8192x32xf16>
    // CHECK:       [[KEY0:%.+]] = VPU.Slice [[KEY:%.+]] [0, 0, 0, 0] [1, 1, 128, 32]
    // CHECK:       [[VAL0:%.+]] = VPU.Slice [[VALUE_REORDERED]] [0, 0, 0, 0] [1, 1, 128, 64]
    // CHECK:       [[Q0:%.+]] = VPU.Slice [[HEAD0_Q]] [0, 0, 0, 0] [1, 1, 4096, 32]
    // CHECK:       VPU.FlashSDPA
    // CHECK-SAME:      loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 0, 0>}
    // CHECK-SAME:      tiling_loop_index = 0
    // CHECK-SAME:      -> tensor<1x1x4096x64xf16>, tensor<1x1x4096x1xf16>, tensor<1x1x4096x1xf32>

    // Heads 0-7, 2 seq tiles each = 16 total (1 checked above, 15 remaining)
    // CHECK-COUNT-15: VPU.FlashSDPA

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat{{.*}}-> tensor<1x8x8192x64xf16>
    // CHECK:       return [[CONCAT]]
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

config.Resources 4 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1997824 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @FlashSDPAKeyValueTilingWithNoQueryTiling
func.func @FlashSDPAKeyValueTilingWithNoQueryTiling(%arg0: tensor<1x1x1x128xf16>, %arg1: tensor<1x1x4096x128xf16>, %arg2: tensor<1x1x4096x128xf16>, %arg3: tensor<1x1x1x4096xf16>) -> tensor<1x1x1x128xf16> {
    %cst = const.Declare tensor<1x1x128x4xsi32> = dense<0> : tensor<1x1x128x4xsi32>
    %cst_0 = const.Declare tensor<1x1x4096x4xsi32> = dense<0> : tensor<1x1x4096x4xsi32>
    %cst_1 = const.Declare tensor<1x1x2x240xsi32> = dense<0> : tensor<1x1x2x240xsi32>
    %cst_2 = const.Declare tensor<1x1x1x4096xf16> = dense<0.000000e+00> : tensor<1x1x1x4096xf16>
    %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %cst_4 = const.Declare tensor<1x1x1x1xf16> = dense<0xFC00> : tensor<1x1x1x1xf16>
    %cst_5 = const.Declare tensor<1x1x1x128xf16> = dense<0.000000e+00> : tensor<1x1x1x128xf16>

    %value_reordered = VPU.MemPermute(%arg2) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x1x4096x128xf16> -> tensor<1x1x4096x128xf16, {order = #NCWH}>

    %result_running_output, %result_running_max, %result_running_sum =
        VPU.FlashSDPA(%arg0, %arg1, %value_reordered, %cst_2, %cst_1, %cst_0, %cst, %cst_5, %cst_4, %cst_3, %arg3) {
            is_head = true,
            is_tail = true,
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
            source_seq_len_pad_size = 0 : i64
        } : tensor<1x1x1x128xf16>, tensor<1x1x4096x128xf16>, tensor<1x1x4096x128xf16, {order = #NCWH}>,
            tensor<1x1x1x4096xf16>, tensor<1x1x2x240xsi32>, tensor<1x1x4096x4xsi32>,
            tensor<1x1x128x4xsi32>, tensor<1x1x1x128xf16>, tensor<1x1x1x1xf16>,
            tensor<1x1x1x1xf32>, tensor<1x1x1x4096xf16>
        -> tensor<1x1x1x128xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf32>

    return %result_running_output : tensor<1x1x1x128xf16>

    // No head or query seq tiling needed. In the decode case (target SL 1) the KV sequence is
    // unrolled into 3 tiles instead of the minimal 2, reserving one extra KV block of
    // Key/Value/mask so the scheduler can prefetch the next block while the current one computes.
    // Tiles are chained through running output/max/sum.

    // First KV tile: key/value/mask sliced on H/W, is_head=true, is_tail=false
    // CHECK:       [[KEY0:%.+]] = VPU.Slice {{%.+}} [0, 0, 0, 0] [1, 1, 1376, 128]
    // CHECK:       [[VAL0:%.+]] = VPU.Slice {{%.+}} [0, 0, 0, 0] [1, 1, 1376, 128]
    // CHECK:       [[MASK0:%.+]] = VPU.Slice {{%.+}} [0, 0, 0, 0] [1, 1, 1, 1376]
    // CHECK:       [[OUT0:%.+]], [[MAX0:%.+]], [[SUM0:%.+]] = VPU.FlashSDPA({{%.+}}, [[KEY0]], [[VAL0]],
    // CHECK-SAME:      is_head = true
    // CHECK-SAME:      is_tail = false
    // CHECK-SAME:      loop_tile_coord = {axes = array<i64>, indices = array<i64>}
    // CHECK-SAME:      tiling_loop_index = [[LOOP_INDEX_0:[0-9]+]]
    // CHECK-SAME:      -> tensor<1x1x1x128xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf32>

    // Second KV tile: key/value/mask sliced on H/W at offset 1376, takes running OUT0/MAX0/SUM0
    // CHECK:       [[KEY1:%.+]] = VPU.Slice {{%.+}} [0, 0, 1376, 0] [1, 1, 1376, 128]
    // CHECK:       [[VAL1:%.+]] = VPU.Slice {{%.+}} [0, 0, 1376, 0] [1, 1, 1376, 128]
    // CHECK:       [[MASK1:%.+]] = VPU.Slice {{%.+}} [0, 0, 0, 1376] [1, 1, 1, 1376]
    // CHECK:       [[OUT1:%.+]], [[MAX1:%.+]], [[SUM1:%.+]] = VPU.FlashSDPA{{.*}}[[OUT0]], [[MAX0]], [[SUM0]]
    // CHECK-SAME:      is_head = false
    // CHECK-SAME:      is_tail = false
    // CHECK-SAME:      loop_tile_coord = {axes = array<i64>, indices = array<i64>}
    // CHECK-SAME:      tiling_loop_index = [[LOOP_INDEX_1:[0-9]+]]
    // CHECK-SAME:      -> tensor<1x1x1x128xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf32>

    // Third KV tile: key/value/mask sliced on H/W at offset 2752 (size 1344), takes running OUT1/MAX1/SUM1
    // CHECK:       [[KEY2:%.+]] = VPU.Slice {{%.+}} [0, 0, 2752, 0] [1, 1, 1344, 128]
    // CHECK:       [[VAL2:%.+]] = VPU.Slice {{%.+}} [0, 0, 2752, 0] [1, 1, 1344, 128]
    // CHECK:       [[MASK2:%.+]] = VPU.Slice {{%.+}} [0, 0, 0, 2752] [1, 1, 1, 1344]
    // CHECK:       VPU.FlashSDPA{{.*}}[[OUT1]], [[MAX1]], [[SUM1]]
    // CHECK-SAME:      is_head = false
    // CHECK-SAME:      is_tail = true
    // CHECK-SAME:      loop_tile_coord = {axes = array<i64>, indices = array<i64>}
    // CHECK-SAME:      tiling_loop_index = 0
    // CHECK-SAME:      -> tensor<1x1x1x128xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf32>
    // CHECK:       return
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

config.Resources 4 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1997824 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// Same KV unroll as above, but with a non-zero source_seq_len_pad_size on the
// original op. The padding describes trailing junk in the (already expanded) KV
// sequence, so it must be applied only on the final KV tile: every earlier tile
// computes over a full, real key/value slice and therefore must carry
// source_seq_len_pad_size = 0. Only the last tile keeps the original padding.

// CHECK-LABEL: @FlashSDPAKeyValueTilingPadSizePropagation
func.func @FlashSDPAKeyValueTilingPadSizePropagation(%arg0: tensor<1x1x1x128xf16>, %arg1: tensor<1x1x4096x128xf16>, %arg2: tensor<1x1x4096x128xf16>, %arg3: tensor<1x1x1x4096xf16>) -> tensor<1x1x1x128xf16> {
    %cst = const.Declare tensor<1x1x128x4xsi32> = dense<0> : tensor<1x1x128x4xsi32>
    %cst_0 = const.Declare tensor<1x1x4096x4xsi32> = dense<0> : tensor<1x1x4096x4xsi32>
    %cst_1 = const.Declare tensor<1x1x2x240xsi32> = dense<0> : tensor<1x1x2x240xsi32>
    %cst_2 = const.Declare tensor<1x1x1x4096xf16> = dense<0.000000e+00> : tensor<1x1x1x4096xf16>
    %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %cst_4 = const.Declare tensor<1x1x1x1xf16> = dense<0xFC00> : tensor<1x1x1x1xf16>
    %cst_5 = const.Declare tensor<1x1x1x128xf16> = dense<0.000000e+00> : tensor<1x1x1x128xf16>

    %value_reordered = VPU.MemPermute(%arg2) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x1x4096x128xf16> -> tensor<1x1x4096x128xf16, {order = #NCWH}>

    %result_running_output, %result_running_max, %result_running_sum =
        VPU.FlashSDPA(%arg0, %arg1, %value_reordered, %cst_2, %cst_1, %cst_0, %cst, %cst_5, %cst_4, %cst_3, %arg3) {
            is_head = true,
            is_tail = true,
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
            source_seq_len_pad_size = 15 : i64
        } : tensor<1x1x1x128xf16>, tensor<1x1x4096x128xf16>, tensor<1x1x4096x128xf16, {order = #NCWH}>,
            tensor<1x1x1x4096xf16>, tensor<1x1x2x240xsi32>, tensor<1x1x4096x4xsi32>,
            tensor<1x1x128x4xsi32>, tensor<1x1x1x128xf16>, tensor<1x1x1x1xf16>,
            tensor<1x1x1x1xf32>, tensor<1x1x1x4096xf16>
        -> tensor<1x1x1x128xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf32>

    return %result_running_output : tensor<1x1x1x128xf16>

    // In decode the KV sequence is unrolled into 3 tiles chained through running output/max/sum.
    // The padding describes trailing junk in the (already expanded) KV sequence, so only the
    // final tile keeps source_seq_len_pad_size = 15; every earlier tile computes over a full,
    // real key/value slice and must carry source_seq_len_pad_size = 0.

    // First (head) tile is a full, real KV slice: source_seq_len_pad_size must be 0.
    // CHECK:       VPU.FlashSDPA
    // CHECK-SAME:      is_head = true
    // CHECK-SAME:      is_tail = false
    // CHECK-SAME:      source_seq_len_pad_size = 0 : i64
    // CHECK-SAME:      -> tensor<1x1x1x128xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf32>

    // Middle tile is also a full, real KV slice: source_seq_len_pad_size must be 0.
    // CHECK:       VPU.FlashSDPA
    // CHECK-SAME:      is_head = false
    // CHECK-SAME:      is_tail = false
    // CHECK-SAME:      source_seq_len_pad_size = 0 : i64
    // CHECK-SAME:      -> tensor<1x1x1x128xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf32>

    // Last (tail) tile keeps the original padding.
    // CHECK:       VPU.FlashSDPA
    // CHECK-SAME:      is_head = false
    // CHECK-SAME:      is_tail = true
    // CHECK-SAME:      source_seq_len_pad_size = 15 : i64
    // CHECK-SAME:      -> tensor<1x1x1x128xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf32>
    // CHECK:       return
}

// -----

// GQA: 32 Q heads, 8 KV heads => group size = 4
// SeqLen=1024, QKEmbed=128, VEmbed=128, broadcasted attention mask

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

config.Resources 4 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1798041 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1997824 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @FlashSDPA_GQA_1024
func.func @FlashSDPA_GQA_1024(%arg0: tensor<1x32x1024x128xf16>, %arg1: tensor<1x8x1024x128xf16>, %arg2: tensor<1x8x1024x128xf16>, %arg3: tensor<1x1x1024x1024xf16>)
                                  -> tensor<1x32x1024x128xf16> {
    %cst = const.Declare tensor<1x1x128x4xsi32> = dense<0> : tensor<1x1x128x4xsi32>
    %cst_0 = const.Declare tensor<1x1x1024x4xsi32> = dense<0> : tensor<1x1x1024x4xsi32>
    %cst_1 = const.Declare tensor<1x1x2x240xsi32> = dense<0> : tensor<1x1x2x240xsi32>
    %cst_2 = const.Declare tensor<1x2x1024x1024xf16> = dense<0.000000e+00> : tensor<1x2x1024x1024xf16>
    %cst_3 = const.Declare tensor<1x32x1024x1xf32> = dense<0.000000e+00> : tensor<1x32x1024x1xf32>
    %cst_4 = const.Declare tensor<1x32x1024x1xf16> = dense<0xFC00> : tensor<1x32x1024x1xf16>
    %cst_5 = const.Declare tensor<1x32x1024x128xf16> = dense<0.000000e+00> : tensor<1x32x1024x128xf16>

    %value_reordered = VPU.MemPermute(%arg2) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x8x1024x128xf16> -> tensor<1x8x1024x128xf16, {order = #NCWH}>

    %result_running_output, %result_running_max, %result_running_sum =
        VPU.FlashSDPA(%arg0, %arg1, %value_reordered, %cst_2, %cst_1, %cst_0, %cst, %cst_5, %cst_4, %cst_3, %arg3) {
            is_head = false,
            is_tail = true,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            source_seq_len_pad_size = 0 : i64
        } : tensor<1x32x1024x128xf16>, tensor<1x8x1024x128xf16>, tensor<1x8x1024x128xf16, {order = #NCWH}>,
            tensor<1x2x1024x1024xf16>, tensor<1x1x2x240xsi32>, tensor<1x1x1024x4xsi32>,
            tensor<1x1x128x4xsi32>, tensor<1x32x1024x128xf16>, tensor<1x32x1024x1xf16>,
            tensor<1x32x1024x1xf32>, tensor<1x1x1024x1024xf16>
        -> tensor<1x32x1024x128xf16>, tensor<1x32x1024x1xf16>, tensor<1x32x1024x1xf32>

    return %result_running_output : tensor<1x32x1024x128xf16>

    // GQA head tiling: 8 KV groups (4 Q heads each), seq tiles 2x384 + 1x256.
    // Head-major emission: all 3 seq tiles of KV group 0 first, then KV group 1, etc.

    // First KV group (Q heads 0-3, KV head 0): full-head slice, then first seq tile
    // CHECK-DAG:   [[VALUE_REORDERED:%.+]] = VPU.MemPermute([[ARG2:%.+]]) {dst_order = #NCWH, mem_perm = #NCWH}
    // CHECK:       [[HEAD0_Q:%.+]] = VPU.Slice {{%.+}} [0, 0, 0, 0] [1, 4, 1024, 128] : tensor<1x32x1024x128xf16>
    // CHECK:       [[KEY0:%.+]] = VPU.Slice [[KEY:%.+]] [0, 0, 0, 0] [1, 1, 1024, 128]
    // CHECK:       [[VAL0:%.+]] = VPU.Slice [[VALUE_REORDERED]] [0, 0, 0, 0] [1, 1, 1024, 128]
    // CHECK:       [[Q0:%.+]] = VPU.Slice [[HEAD0_Q]] [0, 0, 0, 0] [1, 4, 384, 128]
    // CHECK:       VPU.FlashSDPA([[Q0]], [[KEY0]], [[VAL0]],
    // CHECK-SAME:      tiling_loop_index = 0
    // CHECK-SAME:      -> tensor<1x4x384x128xf16>, tensor<1x4x384x1xf16>, tensor<1x4x384x1xf32>

    // Total: 8 KV groups x 3 seq tiles = 24 FlashSDPA ops (1 checked above, 23 remaining)
    // CHECK-COUNT-23: VPU.FlashSDPA

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat{{.*}}-> tensor<1x32x1024x128xf16>
    // CHECK:       return [[CONCAT]]
}

// -----

// MHA with SplitOverKernel: 27 Q heads = 27 KV heads, TargetSeqLen=1
// Kernel contract: each cluster invocation must receive exactly 1 KV head.
// alignment[C] is capped to numTiles (clusters) = 4 for the SOK strategy on
// 60XX. headsPerTile = groupSize=1 * alignmentC=4 = 4, so 27 heads become
// 6 full 4-head tiles plus one residual 3-head tile.

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

config.Resources 4 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1798041 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1997824 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @FlashSDPA_MHA_SplitOverKernel_27Heads
func.func @FlashSDPA_MHA_SplitOverKernel_27Heads(%arg0: tensor<1x27x1x64xf16>, %arg1: tensor<1x27x1024x64xf16>, %arg2: tensor<1x27x1024x64xf16>, %arg3: tensor<1x1x1x1024xf16>) -> tensor<1x27x1x64xf16> {
    %cst = const.Declare tensor<1x1x64x4xsi32> = dense<0> : tensor<1x1x64x4xsi32>
    %cst_0 = const.Declare tensor<1x1x1024x4xsi32> = dense<0> : tensor<1x1x1024x4xsi32>
    %cst_1 = const.Declare tensor<1x1x2x240xsi32> = dense<0> : tensor<1x1x2x240xsi32>
    %cst_2 = const.Declare tensor<1x2x1x1024xf16> = dense<0.000000e+00> : tensor<1x2x1x1024xf16>
    %cst_3 = const.Declare tensor<1x27x1x1xf32> = dense<0.000000e+00> : tensor<1x27x1x1xf32>
    %cst_4 = const.Declare tensor<1x27x1x1xf16> = dense<0xFC00> : tensor<1x27x1x1xf16>
    %cst_5 = const.Declare tensor<1x27x1x64xf16> = dense<0.000000e+00> : tensor<1x27x1x64xf16>

    %value_reordered = VPU.MemPermute(%arg2) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x27x1024x64xf16> -> tensor<1x27x1024x64xf16, {order = #NCWH}>

    %result_running_output, %result_running_max, %result_running_sum =
        VPU.FlashSDPA(%arg0, %arg1, %value_reordered, %cst_2, %cst_1, %cst_0, %cst, %cst_5, %cst_4, %cst_3, %arg3) {
            is_head = true,
            is_tail = true,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            source_seq_len_pad_size = 0 : i64
        } : tensor<1x27x1x64xf16>, tensor<1x27x1024x64xf16>, tensor<1x27x1024x64xf16, {order = #NCWH}>,
            tensor<1x2x1x1024xf16>, tensor<1x1x2x240xsi32>, tensor<1x1x1024x4xsi32>,
            tensor<1x1x64x4xsi32>, tensor<1x27x1x64xf16>, tensor<1x27x1x1xf16>,
            tensor<1x27x1x1xf32>, tensor<1x1x1x1024xf16>
        -> tensor<1x27x1x64xf16>, tensor<1x27x1x1xf16>, tensor<1x27x1x1xf32>

    return %result_running_output : tensor<1x27x1x64xf16>

    // 6 head tiles of 4 heads each plus one residual 3-head tile.
    // Same result with or without pipelining (TargetSeqLen=1, no query seq tiling)

    // First tile: 4 heads
    // CHECK-DAG:   [[VALUE_REORDERED:%.+]] = VPU.MemPermute([[ARG:%.+]]) {dst_order = #NCWH, mem_perm = #NCWH}
    // CHECK:       [[Q0:%.+]] = VPU.Slice [[QUERY:%.+]] [0, 0, 0, 0] [1, 4, 1, 64] : tensor<1x27x1x64xf16> to tensor<1x4x1x64xf16>
    // CHECK:       [[KEY0:%.+]] = VPU.Slice [[KEY:%.+]] [0, 0, 0, 0] [1, 4, 1024, 64]
    // CHECK:       [[VAL0:%.+]] = VPU.Slice [[VALUE_REORDERED]] [0, 0, 0, 0] [1, 4, 1024, 64]
    // CHECK:       VPU.FlashSDPA
    // CHECK-SAME:      tiling_loop_index = 0
    // CHECK-SAME:      -> tensor<1x4x1x64xf16>, tensor<1x4x1x1xf16>, tensor<1x4x1x1xf32>

    // Second tile receives a distinct loop index.
    // CHECK:       VPU.FlashSDPA{{.*}}tiling_loop_index = 1 : i64

    // CHECK:       VPU.FlashSDPA{{.*}}tiling_loop_index = 2 : i64{{.*}}-> tensor<1x4x1x64xf16>, tensor<1x4x1x1xf16>, tensor<1x4x1x1xf32>
    // CHECK:       VPU.FlashSDPA{{.*}}tiling_loop_index = 3 : i64{{.*}}-> tensor<1x4x1x64xf16>, tensor<1x4x1x1xf16>, tensor<1x4x1x1xf32>
    // CHECK:       VPU.FlashSDPA{{.*}}tiling_loop_index = 4 : i64{{.*}}-> tensor<1x4x1x64xf16>, tensor<1x4x1x1xf16>, tensor<1x4x1x1xf32>
    // CHECK:       VPU.FlashSDPA{{.*}}tiling_loop_index = 5 : i64{{.*}}-> tensor<1x4x1x64xf16>, tensor<1x4x1x1xf16>, tensor<1x4x1x1xf32>
    // CHECK:       VPU.FlashSDPA{{.*}}tiling_loop_index = 6 : i64{{.*}}-> tensor<1x3x1x64xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf32>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat(
    // CHECK-SAME:      -> tensor<1x27x1x64xf16>
    // CHECK:       return [[CONCAT]]
}

// -----

// GQA with Clustering and TargetSeqLen=1: 32 Q heads, 8 KV heads (groupSize=4).
// With Clustering strategy, alignment[C] = 1, so headsPerTile = groupSize = 4
// and headTiles = 32 / 4 = 8. Each tile carries exactly 1 KV head.

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

config.Resources 4 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1798041 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1997824 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @FlashSDPA_GQA_Clustering_TargetSeqLen1
func.func @FlashSDPA_GQA_Clustering_TargetSeqLen1(%arg0: tensor<1x32x1x128xf16>, %arg1: tensor<1x8x1024x128xf16>, %arg2: tensor<1x8x1024x128xf16>, %arg3: tensor<1x1x1x1024xf16>)
                                  -> tensor<1x32x1x128xf16> {
    %cst = const.Declare tensor<1x1x128x4xsi32> = dense<0> : tensor<1x1x128x4xsi32>
    %cst_0 = const.Declare tensor<1x1x1024x4xsi32> = dense<0> : tensor<1x1x1024x4xsi32>
    %cst_1 = const.Declare tensor<1x1x2x240xsi32> = dense<0> : tensor<1x1x2x240xsi32>
    %cst_2 = const.Declare tensor<1x4x1x1024xf16> = dense<0.000000e+00> : tensor<1x4x1x1024xf16>
    %cst_3 = const.Declare tensor<1x32x1x1xf32> = dense<0.000000e+00> : tensor<1x32x1x1xf32>
    %cst_4 = const.Declare tensor<1x32x1x1xf16> = dense<0xFC00> : tensor<1x32x1x1xf16>
    %cst_5 = const.Declare tensor<1x32x1x128xf16> = dense<0.000000e+00> : tensor<1x32x1x128xf16>

    %value_reordered = VPU.MemPermute(%arg2) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x8x1024x128xf16> -> tensor<1x8x1024x128xf16, {order = #NCWH}>

    %result_running_output, %result_running_max, %result_running_sum =
        VPU.FlashSDPA(%arg0, %arg1, %value_reordered, %cst_2, %cst_1, %cst_0, %cst, %cst_5, %cst_4, %cst_3, %arg3) {
            is_head = true,
            is_tail = true,
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
            source_seq_len_pad_size = 0 : i64
        } : tensor<1x32x1x128xf16>, tensor<1x8x1024x128xf16>, tensor<1x8x1024x128xf16, {order = #NCWH}>,
            tensor<1x4x1x1024xf16>, tensor<1x1x2x240xsi32>, tensor<1x1x1024x4xsi32>,
            tensor<1x1x128x4xsi32>, tensor<1x32x1x128xf16>, tensor<1x32x1x1xf16>,
            tensor<1x32x1x1xf32>, tensor<1x1x1x1024xf16>
        -> tensor<1x32x1x128xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x1xf32>

    return %result_running_output : tensor<1x32x1x128xf16>

    // 8 head tiles of 4 Q heads (= 1 KV head) each.
    // CHECK-DAG:   [[VALUE_REORDERED:%.+]] = VPU.MemPermute([[ARG:%.+]]) {dst_order = #NCWH, mem_perm = #NCWH}
    // CHECK:       [[Q0:%.+]] = VPU.Slice [[QUERY:%.+]] [0, 0, 0, 0] [1, 4, 1, 128] : tensor<1x32x1x128xf16> to tensor<1x4x1x128xf16>
    // CHECK:       [[KEY0:%.+]] = VPU.Slice [[KEY:%.+]] [0, 0, 0, 0] [1, 1, 1024, 128]
    // CHECK:       [[VAL0:%.+]] = VPU.Slice [[VALUE_REORDERED]] [0, 0, 0, 0] [1, 1, 1024, 128]
    // CHECK:       VPU.FlashSDPA
    // CHECK-SAME:      -> tensor<1x4x1x128xf16>, tensor<1x4x1x1xf16>, tensor<1x4x1x1xf32>

    // CHECK-COUNT-7: -> tensor<1x4x1x128xf16>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat(
    // CHECK-SAME:      -> tensor<1x32x1x128xf16>
    // CHECK:       return [[CONCAT]]
}

// -----

// GQA with SplitOverKernel and TargetSeqLen=1: 32 Q heads, 8 KV heads (groupSize=4),
// numClusters=4 on 60XX. With SOK, alignment[C] = numClusters = 4, so
// headsPerTile = groupSize * numClusters = 16 and the shape becomes two equal
// tiles of 16 Q heads / 4 KV heads.

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

config.Resources 4 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1798041 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1997824 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @FlashSDPA_GQA_SplitOverKernel_ResidualTile
func.func @FlashSDPA_GQA_SplitOverKernel_ResidualTile(%arg0: tensor<1x32x1x128xf16>, %arg1: tensor<1x8x1024x128xf16>, %arg2: tensor<1x8x1024x128xf16>, %arg3: tensor<1x1x1x1024xf16>)
                                  -> tensor<1x32x1x128xf16> {
    %cst = const.Declare tensor<1x1x128x4xsi32> = dense<0> : tensor<1x1x128x4xsi32>
    %cst_0 = const.Declare tensor<1x1x1024x4xsi32> = dense<0> : tensor<1x1x1024x4xsi32>
    %cst_1 = const.Declare tensor<1x1x2x240xsi32> = dense<0> : tensor<1x1x2x240xsi32>
    %cst_2 = const.Declare tensor<1x4x1x1024xf16> = dense<0.000000e+00> : tensor<1x4x1x1024xf16>
    %cst_3 = const.Declare tensor<1x32x1x1xf32> = dense<0.000000e+00> : tensor<1x32x1x1xf32>
    %cst_4 = const.Declare tensor<1x32x1x1xf16> = dense<0xFC00> : tensor<1x32x1x1xf16>
    %cst_5 = const.Declare tensor<1x32x1x128xf16> = dense<0.000000e+00> : tensor<1x32x1x128xf16>

    %value_reordered = VPU.MemPermute(%arg2) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x8x1024x128xf16> -> tensor<1x8x1024x128xf16, {order = #NCWH}>

    %result_running_output, %result_running_max, %result_running_sum =
        VPU.FlashSDPA(%arg0, %arg1, %value_reordered, %cst_2, %cst_1, %cst_0, %cst, %cst_5, %cst_4, %cst_3, %arg3) {
            is_head = true,
            is_tail = true,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            source_seq_len_pad_size = 0 : i64
        } : tensor<1x32x1x128xf16>, tensor<1x8x1024x128xf16>, tensor<1x8x1024x128xf16, {order = #NCWH}>,
            tensor<1x4x1x1024xf16>, tensor<1x1x2x240xsi32>, tensor<1x1x1024x4xsi32>,
            tensor<1x1x128x4xsi32>, tensor<1x32x1x128xf16>, tensor<1x32x1x1xf16>,
            tensor<1x32x1x1xf32>, tensor<1x1x1x1024xf16>
        -> tensor<1x32x1x128xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x1xf32>

    return %result_running_output : tensor<1x32x1x128xf16>

    // First tile: 16 Q heads / 4 KV heads.
    // CHECK-DAG:   [[VALUE_REORDERED:%.+]] = VPU.MemPermute([[ARG:%.+]]) {dst_order = #NCWH, mem_perm = #NCWH}
    // CHECK:       VPU.Slice {{%.+}} [0, 0, 0, 0] [1, 16, 1, 128]
    // CHECK:       VPU.Slice {{%.+}} [0, 0, 0, 0] [1, 4, 1024, 128]
    // CHECK:       VPU.FlashSDPA
    // CHECK-SAME:      SplitOverKernel
    // CHECK-SAME:      -> tensor<1x16x1x128xf16>, tensor<1x16x1x1xf16>, tensor<1x16x1x1xf32>

    // Second tile: 16 Q heads / 4 KV heads.
    // CHECK:       VPU.Slice {{%.+}} [0, 16, 0, 0] [1, 16, 1, 128]
    // CHECK:       VPU.Slice {{%.+}} [0, 4, 0, 0] [1, 4, 1024, 128]
    // CHECK:       VPU.FlashSDPA
    // CHECK-SAME:      SplitOverKernel
    // CHECK-SAME:      -> tensor<1x16x1x128xf16>, tensor<1x16x1x1xf16>, tensor<1x16x1x1xf32>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat(
    // CHECK-SAME:      -> tensor<1x32x1x128xf16>
    // CHECK:       return [[CONCAT]]
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

config.Resources 4 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1997824 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// Four KV blocks are already materialized before FlashSDPA tiling. Their
// running states are chained and all blocks belong to loop 0.
// CHECK-LABEL: @FlashSDPAPreassignedLoopIndex
func.func @FlashSDPAPreassignedLoopIndex(
        %query: tensor<1x32x1x128xf16>,
        %key0: tensor<1x4x1024x128xf16>, %key1: tensor<1x4x1024x128xf16>,
        %key2: tensor<1x4x1024x128xf16>, %key3: tensor<1x4x1024x128xf16>,
        %value0: tensor<1x4x1024x128xf16, {order = #NCWH}>,
        %value1: tensor<1x4x1024x128xf16, {order = #NCWH}>,
        %value2: tensor<1x4x1024x128xf16, {order = #NCWH}>,
        %value3: tensor<1x4x1024x128xf16, {order = #NCWH}>,
        %aux: tensor<1x8x1x1024xf16>,
        %descriptors: tensor<1x1x2x240xsi32>,
        %weights0: tensor<1x1x1024x4xsi32>,
        %weights1: tensor<1x1x128x4xsi32>,
        %initial_output: tensor<1x32x1x128xf16>,
        %initial_max: tensor<1x32x1x1xf16>,
        %initial_sum: tensor<1x32x1x1xf32>,
        %mask0: tensor<1x1x1x1024xf16>, %mask1: tensor<1x1x1x1024xf16>,
        %mask2: tensor<1x1x1x1024xf16>, %mask3: tensor<1x1x1x1024xf16>)
        -> tensor<1x32x1x128xf16> {
    // CHECK: [[OUT0:%.+]], [[MAX0:%.+]], [[SUM0:%.+]] = VPU.FlashSDPA
    // CHECK-SAME: is_head = true, is_tail = false
    // CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME: tiling_loop_index = 0 : i64
    %out0, %max0, %sum0 = VPU.FlashSDPA(%query, %key0, %value0, %aux, %descriptors, %weights0, %weights1,
                                          %initial_output, %initial_max, %initial_sum, %mask0) {
        is_head = true, is_tail = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        source_seq_len_pad_size = 0 : i64, tiling_loop_index = 0 : i64
    } : tensor<1x32x1x128xf16>, tensor<1x4x1024x128xf16>, tensor<1x4x1024x128xf16, {order = #NCWH}>,
        tensor<1x8x1x1024xf16>, tensor<1x1x2x240xsi32>, tensor<1x1x1024x4xsi32>, tensor<1x1x128x4xsi32>,
        tensor<1x32x1x128xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x1xf32>, tensor<1x1x1x1024xf16>
        -> tensor<1x32x1x128xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x1xf32>

    // CHECK: [[OUT1:%.+]], [[MAX1:%.+]], [[SUM1:%.+]] = VPU.FlashSDPA{{.*}}[[OUT0]], [[MAX0]], [[SUM0]]
    // CHECK-SAME: is_head = false, is_tail = false
    // CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME: tiling_loop_index = 0 : i64
    %out1, %max1, %sum1 = VPU.FlashSDPA(%query, %key1, %value1, %aux, %descriptors, %weights0, %weights1,
                                          %out0, %max0, %sum0, %mask1) {
        is_head = false, is_tail = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        source_seq_len_pad_size = 0 : i64, tiling_loop_index = 0 : i64
    } : tensor<1x32x1x128xf16>, tensor<1x4x1024x128xf16>, tensor<1x4x1024x128xf16, {order = #NCWH}>,
        tensor<1x8x1x1024xf16>, tensor<1x1x2x240xsi32>, tensor<1x1x1024x4xsi32>, tensor<1x1x128x4xsi32>,
        tensor<1x32x1x128xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x1xf32>, tensor<1x1x1x1024xf16>
        -> tensor<1x32x1x128xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x1xf32>

    // CHECK: [[OUT2:%.+]], [[MAX2:%.+]], [[SUM2:%.+]] = VPU.FlashSDPA{{.*}}[[OUT1]], [[MAX1]], [[SUM1]]
    // CHECK-SAME: is_head = false, is_tail = false
    // CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME: tiling_loop_index = 0 : i64
    %out2, %max2, %sum2 = VPU.FlashSDPA(%query, %key2, %value2, %aux, %descriptors, %weights0, %weights1,
                                          %out1, %max1, %sum1, %mask2) {
        is_head = false, is_tail = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        source_seq_len_pad_size = 0 : i64, tiling_loop_index = 0 : i64
    } : tensor<1x32x1x128xf16>, tensor<1x4x1024x128xf16>, tensor<1x4x1024x128xf16, {order = #NCWH}>,
        tensor<1x8x1x1024xf16>, tensor<1x1x2x240xsi32>, tensor<1x1x1024x4xsi32>, tensor<1x1x128x4xsi32>,
        tensor<1x32x1x128xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x1xf32>, tensor<1x1x1x1024xf16>
        -> tensor<1x32x1x128xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x1xf32>

    // CHECK: [[OUT3:%.+]], [[MAX3:%.+]], [[SUM3:%.+]] = VPU.FlashSDPA{{.*}}[[OUT2]], [[MAX2]], [[SUM2]]
    // CHECK-SAME: is_head = false, is_tail = true
    // CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME: tiling_loop_index = 0 : i64
    %out3, %max3, %sum3 = VPU.FlashSDPA(%query, %key3, %value3, %aux, %descriptors, %weights0, %weights1,
                                          %out2, %max2, %sum2, %mask3) {
        is_head = false, is_tail = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        source_seq_len_pad_size = 0 : i64, tiling_loop_index = 0 : i64
    } : tensor<1x32x1x128xf16>, tensor<1x4x1024x128xf16>, tensor<1x4x1024x128xf16, {order = #NCWH}>,
        tensor<1x8x1x1024xf16>, tensor<1x1x2x240xsi32>, tensor<1x1x1024x4xsi32>, tensor<1x1x128x4xsi32>,
        tensor<1x32x1x128xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x1xf32>, tensor<1x1x1x1024xf16>
        -> tensor<1x32x1x128xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x1xf32>

    return %out3 : tensor<1x32x1x128xf16>
}

// -----

// MQA prefill that has to split a GQA group to afford Q-tile pipelining: 16 Q heads and 1 KV head, so
// one whole group spans all 16 heads. With QKEmbed=VEmbed=1024 the per-Q-tile buffers
// (query + input/result running output) cost 3 x 16 heads x 16 rows x 1024 x 2 = 1572864 bytes per
// cluster at the smallest query tile (alignment[H] = clusters * shaves * 8 = 64, i.e. 16 rows per
// cluster), so keeping a second copy free for prefetching exceeds the 1997824 usable CMX bytes no
// matter how far the Key/Value sequence is split. Halving the head tile to 8 Q heads halves the Q side
// and fits with the reservation - the single KV head is still read whole by both head tiles.

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

config.Resources 4 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1798041 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1997824 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @FlashSDPA_MQA_SubGroupHeadTiling
func.func @FlashSDPA_MQA_SubGroupHeadTiling(%arg0: tensor<1x16x256x1024xf16>, %arg1: tensor<1x1x512x1024xf16>, %arg2: tensor<1x1x512x1024xf16>, %arg3: tensor<1x16x256x512xf16>)
                                  -> tensor<1x16x256x1024xf16> {
    %cst = const.Declare tensor<1x1x1024x4xsi32> = dense<0> : tensor<1x1x1024x4xsi32>
    %cst_0 = const.Declare tensor<1x1x512x4xsi32> = dense<0> : tensor<1x1x512x4xsi32>
    %cst_1 = const.Declare tensor<1x1x2x240xsi32> = dense<0> : tensor<1x1x2x240xsi32>
    %cst_2 = const.Declare tensor<1x2x256x512xf16> = dense<0.000000e+00> : tensor<1x2x256x512xf16>
    %cst_3 = const.Declare tensor<1x16x256x1xf32> = dense<0.000000e+00> : tensor<1x16x256x1xf32>
    %cst_4 = const.Declare tensor<1x16x256x1xf16> = dense<0xFC00> : tensor<1x16x256x1xf16>
    %cst_5 = const.Declare tensor<1x16x256x1024xf16> = dense<0.000000e+00> : tensor<1x16x256x1024xf16>

    %value_reordered = VPU.MemPermute(%arg2) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x1x512x1024xf16> -> tensor<1x1x512x1024xf16, {order = #NCWH}>

    %result_running_output, %result_running_max, %result_running_sum =
        VPU.FlashSDPA(%arg0, %arg1, %value_reordered, %cst_2, %cst_1, %cst_0, %cst, %cst_5, %cst_4, %cst_3, %arg3) {
            is_head = false,
            is_tail = true,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            source_seq_len_pad_size = 0 : i64
        } : tensor<1x16x256x1024xf16>, tensor<1x1x512x1024xf16>, tensor<1x1x512x1024xf16, {order = #NCWH}>,
            tensor<1x2x256x512xf16>, tensor<1x1x2x240xsi32>, tensor<1x1x512x4xsi32>,
            tensor<1x1x1024x4xsi32>, tensor<1x16x256x1024xf16>, tensor<1x16x256x1xf16>,
            tensor<1x16x256x1xf32>, tensor<1x16x256x512xf16>
        -> tensor<1x16x256x1024xf16>, tensor<1x16x256x1xf16>, tensor<1x16x256x1xf32>

    return %result_running_output : tensor<1x16x256x1024xf16>

    // 2 head tiles x 7 KV blocks x 4 query tiles = 56 FlashSDPA ops.
    // Head tiles: 2 x 8 (half of the 16-head GQA group), both against the single KV head.
    // KV blocks: 6 x 80 + 1 x 32 (alignValUp(divUp(512, 7), 16) = 80). Query tiles: 4 x 64.

    // First head tile, first KV block, first query tile.
    // CHECK-DAG:   [[VALUE_REORDERED:%.+]] = VPU.MemPermute({{%.+}}) {dst_order = #NCWH, mem_perm = #NCWH}
    // CHECK:       [[HEAD0_Q:%.+]] = VPU.Slice {{%.+}} [0, 0, 0, 0] [1, 8, 256, 1024] : tensor<1x16x256x1024xf16>
    // CHECK:       [[HEAD0_MASK:%.+]] = VPU.Slice {{%.+}} [0, 0, 0, 0] [1, 8, 256, 512] : tensor<1x16x256x512xf16>
    // CHECK:       [[KEY0:%.+]] = VPU.Slice {{%.+}} [0, 0, 0, 0] [1, 1, 80, 1024]
    // CHECK:       [[VAL0:%.+]] = VPU.Slice [[VALUE_REORDERED]] [0, 0, 0, 0] [1, 1, 80, 1024]
    // CHECK:       [[HEAD0_KV_MASK:%.+]] = VPU.Slice [[HEAD0_MASK]] [0, 0, 0, 0] [1, 8, 256, 80]
    // CHECK:       [[Q0:%.+]] = VPU.Slice [[HEAD0_Q]] [0, 0, 0, 0] [1, 8, 64, 1024]
    // CHECK:       [[MASK0:%.+]] = VPU.Slice [[HEAD0_KV_MASK]] [0, 0, 0, 0] [1, 8, 64, 80]
    // CHECK:       VPU.FlashSDPA([[Q0]], [[KEY0]], [[VAL0]]{{.*}}[[MASK0]])
    // CHECK-SAME:      -> tensor<1x8x64x1024xf16>, tensor<1x8x64x1xf16>, tensor<1x8x64x1xf32>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat{{.*}}-> tensor<1x16x256x1024xf16>
    // CHECK:       return [[CONCAT]]
}
