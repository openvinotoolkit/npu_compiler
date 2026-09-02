//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --flash-sdpa-tiling %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

config.Resources 3 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @FlashSDPA8kSeqLen
func.func @FlashSDPA8kSeqLen(%arg0: tensor<1x8x8192x32xf16>, %arg1: tensor<1x8x128x32xf16>, %arg2: tensor<1x8x128x64xf16>, %arg3: tensor<1x8x8192x128xf16>)
                                  -> tensor<1x8x8192x64xf16> {
    %cst = const.Declare tensor<1x1x64x4xsi32> = dense<0> : tensor<1x1x64x4xsi32>
    %cst_0 = const.Declare tensor<1x1x128x4xsi32> = dense<0> : tensor<1x1x128x4xsi32>
    %cst_1 = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
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
            tensor<1x2x8192x128xf16>, tensor<1x1x2x256xsi32>, tensor<1x1x128x4xsi32>,
            tensor<1x1x64x4xsi32>, tensor<1x8x8192x64xf16>, tensor<1x8x8192x1xf16>,
            tensor<1x8x8192x1xf32>, tensor<1x8x8192x128xf16>
        -> tensor<1x8x8192x64xf16>, tensor<1x8x8192x1xf16>, tensor<1x8x8192x1xf32>

    return %result_running_output : tensor<1x8x8192x64xf16>

    // 8 heads x 3 seq tiles = 24 FlashSDPA ops.
    // Seq tiles: 2736, 2736, 2720
    // Head-major emission: all 3 seq tiles of head 0 first, then head 1, etc.

    // Head 0: full-head slice, then seq tile 0
    // CHECK-DAG:   [[VALUE_REORDERED:%.+]] = VPU.MemPermute([[ARG:%.+]]) {dst_order = #NCWH, mem_perm = #NCWH}
    // CHECK:       [[HEAD0_Q:%.+]] = VPU.Slice [[QUERY:%.+]] [0, 0, 0, 0] [1, 1, 8192, 32] : tensor<1x8x8192x32xf16>
    // CHECK:       [[KEY0:%.+]] = VPU.Slice [[KEY:%.+]] [0, 0, 0, 0] [1, 1, 128, 32]
    // CHECK:       [[VAL0:%.+]] = VPU.Slice [[VALUE_REORDERED]] [0, 0, 0, 0] [1, 1, 128, 64]
    // CHECK:       [[Q0:%.+]] = VPU.Slice [[HEAD0_Q]] [0, 0, 0, 0] [1, 1, 2736, 32]
    // CHECK:       VPU.FlashSDPA
    // CHECK-SAME:      tiling_loop_index = 0
    // CHECK-SAME:      -> tensor<1x1x2736x64xf16>, tensor<1x1x2736x1xf16>, tensor<1x1x2736x1xf32>

    // Seq tile 1 of head 0, then heads 1-7 with 3 seq tiles each
    // CHECK-COUNT-23: VPU.FlashSDPA

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat{{.*}}-> tensor<1x8x8192x64xf16>
    // CHECK:       return [[CONCAT]]
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

config.Resources 3 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @FlashSDPAKeyValueTilingWithNoQueryTiling
func.func @FlashSDPAKeyValueTilingWithNoQueryTiling(%arg0: tensor<1x1x1x128xf16>, %arg1: tensor<1x1x4096x128xf16>, %arg2: tensor<1x1x4096x128xf16>, %arg3: tensor<1x1x1x4096xf16>) -> tensor<1x1x1x128xf16> {
    %cst = const.Declare tensor<1x1x128x4xsi32> = dense<0> : tensor<1x1x128x4xsi32>
    %cst_0 = const.Declare tensor<1x1x4096x4xsi32> = dense<0> : tensor<1x1x4096x4xsi32>
    %cst_1 = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
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
            tensor<1x1x1x4096xf16>, tensor<1x1x2x256xsi32>, tensor<1x1x4096x4xsi32>,
            tensor<1x1x128x4xsi32>, tensor<1x1x1x128xf16>, tensor<1x1x1x1xf16>,
            tensor<1x1x1x1xf32>, tensor<1x1x1x4096xf16>
        -> tensor<1x1x1x128xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf32>

    return %result_running_output : tensor<1x1x1x128xf16>

    // No head or query seq tiling needed, KV unrolled into 2 tiles chained through running output/max/sum.
    // Same result with or without pipelining for this small query.

    // First KV tile: key/value/mask sliced on H/W, is_head=true, is_tail=false
    // CHECK:       [[KEY0:%.+]] = VPU.Slice {{%.+}} [0, 0, 0, 0] [1, 1, 2048, 128]
    // CHECK:       [[VAL0:%.+]] = VPU.Slice {{%.+}} [0, 0, 0, 0] [1, 1, 2048, 128]
    // CHECK:       [[MASK0:%.+]] = VPU.Slice {{%.+}} [0, 0, 0, 0] [1, 1, 1, 2048]
    // CHECK:       [[OUT0:%.+]], [[MAX0:%.+]], [[SUM0:%.+]] = VPU.FlashSDPA({{%.+}}, [[KEY0]], [[VAL0]],
    // CHECK-SAME:      is_head = true
    // CHECK-SAME:      is_tail = false
    // CHECK-SAME:      tiling_loop_index = 0
    // CHECK-SAME:      -> tensor<1x1x1x128xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf32>

    // Second KV tile: key/value/mask sliced on H/W at offset 2048, takes running OUT0/MAX0/SUM0
    // CHECK:       [[KEY1:%.+]] = VPU.Slice {{%.+}} [0, 0, 2048, 0] [1, 1, 2048, 128]
    // CHECK:       [[VAL1:%.+]] = VPU.Slice {{%.+}} [0, 0, 2048, 0] [1, 1, 2048, 128]
    // CHECK:       [[MASK1:%.+]] = VPU.Slice {{%.+}} [0, 0, 0, 2048] [1, 1, 1, 2048]
    // CHECK:       VPU.FlashSDPA{{.*}}[[OUT0]], [[MAX0]], [[SUM0]]
    // CHECK-SAME:      is_head = false
    // CHECK-SAME:      is_tail = true
    // CHECK-SAME:      tiling_loop_index = 1
    // CHECK-SAME:      -> tensor<1x1x1x128xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf32>
    // CHECK:       return
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

config.Resources 3 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
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
    %cst_1 = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
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
            tensor<1x1x1x4096xf16>, tensor<1x1x2x256xsi32>, tensor<1x1x4096x4xsi32>,
            tensor<1x1x128x4xsi32>, tensor<1x1x1x128xf16>, tensor<1x1x1x1xf16>,
            tensor<1x1x1x1xf32>, tensor<1x1x1x4096xf16>
        -> tensor<1x1x1x128xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf32>

    return %result_running_output : tensor<1x1x1x128xf16>

    // KV unrolled into 2 tiles chained through running output/max/sum.
    // First (head) tile is a full, real KV slice: source_seq_len_pad_size must be 0.
    // CHECK:       VPU.FlashSDPA
    // CHECK-SAME:      is_head = true
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

config.Resources 3 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @FlashSDPA_GQA_1024
func.func @FlashSDPA_GQA_1024(%arg0: tensor<1x32x1024x128xf16>, %arg1: tensor<1x8x1024x128xf16>, %arg2: tensor<1x8x1024x128xf16>, %arg3: tensor<1x1x1024x1024xf16>)
                                  -> tensor<1x32x1024x128xf16> {
    %cst = const.Declare tensor<1x1x128x4xsi32> = dense<0> : tensor<1x1x128x4xsi32>
    %cst_0 = const.Declare tensor<1x1x1024x4xsi32> = dense<0> : tensor<1x1x1024x4xsi32>
    %cst_1 = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
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
            tensor<1x2x1024x1024xf16>, tensor<1x1x2x256xsi32>, tensor<1x1x1024x4xsi32>,
            tensor<1x1x128x4xsi32>, tensor<1x32x1024x128xf16>, tensor<1x32x1024x1xf16>,
            tensor<1x32x1024x1xf32>, tensor<1x1x1024x1024xf16>
        -> tensor<1x32x1024x128xf16>, tensor<1x32x1024x1xf16>, tensor<1x32x1024x1xf32>

    return %result_running_output : tensor<1x32x1024x128xf16>


    // GQA head tiling: 8 KV groups (4 Q heads each), seq tiles 5x192 + 1x64.
    // Head-major emission: all 6 seq tiles of KV group 0 first, then KV group 1, etc.

    // First KV group (Q heads 0-3, KV head 0): full-head slice, then first seq tile
    // CHECK-DAG:   [[VALUE_REORDERED:%.+]] = VPU.MemPermute([[ARG2:%.+]]) {dst_order = #NCWH, mem_perm = #NCWH}
    // CHECK:       [[HEAD0_Q:%.+]] = VPU.Slice {{%.+}} [0, 0, 0, 0] [1, 4, 1024, 128] : tensor<1x32x1024x128xf16>
    // CHECK:       [[KEY0:%.+]] = VPU.Slice {{%.+}} [0, 0, 0, 0] [1, 1, 1024, 128]
    // CHECK:       [[VAL0:%.+]] = VPU.Slice [[VALUE_REORDERED]] [0, 0, 0, 0] [1, 1, 1024, 128]
    // CHECK:       [[Q0:%.+]] = VPU.Slice [[HEAD0_Q]] [0, 0, 0, 0] [1, 4, 192, 128]
    // CHECK:       VPU.FlashSDPA([[Q0]], [[KEY0]], [[VAL0]],
    // CHECK-SAME:      tiling_loop_index = 0
    // CHECK-SAME:      -> tensor<1x4x192x128xf16>, tensor<1x4x192x1xf16>, tensor<1x4x192x1xf32>

    // Total: 8 KV groups x 6 seq tiles = 48 FlashSDPA ops (1 checked above, 47 remaining)
    // CHECK-COUNT-47: VPU.FlashSDPA

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat{{.*}}-> tensor<1x32x1024x128xf16>
    // CHECK:       return [[CONCAT]]
}

// -----

// MHA with SplitOverKernel: 27 Q heads = 27 KV heads, TargetSeqLen=1
// Kernel contract: each cluster invocation must receive exactly 1 KV head.
// alignment[C] is capped to numTiles (clusters) = 3 for the SOK strategy.
// headsPerTile = groupSize=1 * alignmentC=3 = 3
// headTiles = divUp(27, 3) = 9
// Tiles: 9 tiles of 3 heads each. MC SOK splits each tile across 3 clusters
// so each cluster handles 1 KV head.

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

config.Resources 3 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @FlashSDPA_MHA_SplitOverKernel_27Heads
func.func @FlashSDPA_MHA_SplitOverKernel_27Heads(%arg0: tensor<1x27x1x64xf16>, %arg1: tensor<1x27x1024x64xf16>, %arg2: tensor<1x27x1024x64xf16>, %arg3: tensor<1x1x1x1024xf16>) -> tensor<1x27x1x64xf16> {
    %cst = const.Declare tensor<1x1x64x4xsi32> = dense<0> : tensor<1x1x64x4xsi32>
    %cst_0 = const.Declare tensor<1x1x1024x4xsi32> = dense<0> : tensor<1x1x1024x4xsi32>
    %cst_1 = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
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
            tensor<1x2x1x1024xf16>, tensor<1x1x2x256xsi32>, tensor<1x1x1024x4xsi32>,
            tensor<1x1x64x4xsi32>, tensor<1x27x1x64xf16>, tensor<1x27x1x1xf16>,
            tensor<1x27x1x1xf32>, tensor<1x1x1x1024xf16>
        -> tensor<1x27x1x64xf16>, tensor<1x27x1x1xf16>, tensor<1x27x1x1xf32>

    return %result_running_output : tensor<1x27x1x64xf16>

    // 9 head tiles of 3 heads each.
    // Same result with or without pipelining (TargetSeqLen=1, no query seq tiling)

    // First tile: 3 heads
    // CHECK-DAG:   [[VALUE_REORDERED:%.+]] = VPU.MemPermute([[ARG:%.+]]) {dst_order = #NCWH, mem_perm = #NCWH}
    // CHECK:       [[Q0:%.+]] = VPU.Slice [[QUERY:%.+]] [0, 0, 0, 0] [1, 3, 1, 64] : tensor<1x27x1x64xf16> to tensor<1x3x1x64xf16>
    // CHECK:       [[KEY0:%.+]] = VPU.Slice [[KEY:%.+]] [0, 0, 0, 0] [1, 3, 1024, 64]
    // CHECK:       [[VAL0:%.+]] = VPU.Slice [[VALUE_REORDERED]] [0, 0, 0, 0] [1, 3, 1024, 64]
    // CHECK:       VPU.FlashSDPA
    // CHECK-SAME:      tiling_loop_index = 0
    // CHECK-SAME:      -> tensor<1x3x1x64xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf32>

    // Tiles 2-9: 3 heads each
    // CHECK-COUNT-8: -> tensor<1x3x1x64xf16>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat(
    // CHECK-SAME:      -> tensor<1x27x1x64xf16>
    // CHECK:       return [[CONCAT]]
}

// -----

// GQA with Clustering and TargetSeqLen=1: 32 Q heads, 8 KV heads (groupSize=4).
// With Clustering strategy, alignment[C] = 1, so headsPerTile = groupSize = 4
// and headTiles = 32 / 4 = 8. Each tile carries exactly 1 KV head.

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

config.Resources 3 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @FlashSDPA_GQA_Clustering_TargetSeqLen1
func.func @FlashSDPA_GQA_Clustering_TargetSeqLen1(%arg0: tensor<1x32x1x128xf16>, %arg1: tensor<1x8x1024x128xf16>, %arg2: tensor<1x8x1024x128xf16>, %arg3: tensor<1x1x1x1024xf16>)
                                  -> tensor<1x32x1x128xf16> {
    %cst = const.Declare tensor<1x1x128x4xsi32> = dense<0> : tensor<1x1x128x4xsi32>
    %cst_0 = const.Declare tensor<1x1x1024x4xsi32> = dense<0> : tensor<1x1x1024x4xsi32>
    %cst_1 = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
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
            tensor<1x4x1x1024xf16>, tensor<1x1x2x256xsi32>, tensor<1x1x1024x4xsi32>,
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
// numClusters=3. With SOK, alignment[C] = numClusters = 3, so headsPerTile =
// groupSize * numClusters = 12 and headTiles = ceil(32 / 12) = 3, producing tiles of
// 12, 12, 8 Q heads => 3, 3, 2 KV heads. All tiles keep SOK; the residual tile (2 KV
// heads) runs SOK on fewer clusters via the kvHeads cap in getOptimalNumClusters.

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

config.Resources 3 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @FlashSDPA_GQA_SplitOverKernel_ResidualTile
func.func @FlashSDPA_GQA_SplitOverKernel_ResidualTile(%arg0: tensor<1x32x1x128xf16>, %arg1: tensor<1x8x1024x128xf16>, %arg2: tensor<1x8x1024x128xf16>, %arg3: tensor<1x1x1x1024xf16>)
                                  -> tensor<1x32x1x128xf16> {
    %cst = const.Declare tensor<1x1x128x4xsi32> = dense<0> : tensor<1x1x128x4xsi32>
    %cst_0 = const.Declare tensor<1x1x1024x4xsi32> = dense<0> : tensor<1x1x1024x4xsi32>
    %cst_1 = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
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
            tensor<1x4x1x1024xf16>, tensor<1x1x2x256xsi32>, tensor<1x1x1024x4xsi32>,
            tensor<1x1x128x4xsi32>, tensor<1x32x1x128xf16>, tensor<1x32x1x1xf16>,
            tensor<1x32x1x1xf32>, tensor<1x1x1x1024xf16>
        -> tensor<1x32x1x128xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x1xf32>

    return %result_running_output : tensor<1x32x1x128xf16>

    // First tile: 12 Q heads / 3 KV heads, SOK on 3 clusters.
    // CHECK-DAG:   [[VALUE_REORDERED:%.+]] = VPU.MemPermute([[ARG:%.+]]) {dst_order = #NCWH, mem_perm = #NCWH}
    // CHECK:       VPU.Slice {{%.+}} [0, 0, 0, 0] [1, 12, 1, 128]
    // CHECK:       VPU.Slice {{%.+}} [0, 0, 0, 0] [1, 3, 1024, 128]
    // CHECK:       VPU.FlashSDPA
    // CHECK-SAME:      SplitOverKernel
    // CHECK-SAME:      -> tensor<1x12x1x128xf16>, tensor<1x12x1x1xf16>, tensor<1x12x1x1xf32>

    // Second tile: 12 Q heads / 3 KV heads, SOK on 3 clusters.
    // CHECK:       VPU.Slice {{%.+}} [0, 12, 0, 0] [1, 12, 1, 128]
    // CHECK:       VPU.Slice {{%.+}} [0, 3, 0, 0] [1, 3, 1024, 128]
    // CHECK:       VPU.FlashSDPA
    // CHECK-SAME:      SplitOverKernel
    // CHECK-SAME:      -> tensor<1x12x1x128xf16>, tensor<1x12x1x1xf16>, tensor<1x12x1x1xf32>

    // Residual tile: 8 Q heads / 2 KV heads, SOK on 2 clusters (cluster count is
    // resolved later by getOptimalNumClusters; the strategy attribute remains SOK).
    // CHECK:       VPU.Slice {{%.+}} [0, 24, 0, 0] [1, 8, 1, 128]
    // CHECK:       VPU.Slice {{%.+}} [0, 6, 0, 0] [1, 2, 1024, 128]
    // CHECK:       VPU.FlashSDPA
    // CHECK-SAME:      SplitOverKernel
    // CHECK-SAME:      -> tensor<1x8x1x128xf16>, tensor<1x8x1x1xf16>, tensor<1x8x1x1xf32>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat(
    // CHECK-SAME:      -> tensor<1x32x1x128xf16>
    // CHECK:       return [[CONCAT]]
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

config.Resources 3 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @FlashSDPA_GQA_SplitOverKernel_GroupSize8_3Clusters
func.func @FlashSDPA_GQA_SplitOverKernel_GroupSize8_3Clusters(
        %arg0: tensor<1x32x1x128xf16>,
        %arg1: tensor<1x4x1024x128xf16>,
        %arg2: tensor<1x4x1024x128xf16>,
        %arg3: tensor<1x1x1x1024xf16>) -> tensor<1x32x1x128xf16> {
    %cst   = const.Declare tensor<1x1x128x4xsi32>  = dense<0> : tensor<1x1x128x4xsi32>
    %cst_0 = const.Declare tensor<1x1x1024x4xsi32> = dense<0> : tensor<1x1x1024x4xsi32>
    %cst_1 = const.Declare tensor<1x1x2x256xsi32>  = dense<0> : tensor<1x1x2x256xsi32>
    // aux_buffer: groupSize=8 scratch rows (batched GQA decode path, tSL=1), sSL=1024
    %cst_2 = const.Declare tensor<1x8x1x1024xf16>  = dense<0.000000e+00> : tensor<1x8x1x1024xf16>
    %cst_3 = const.Declare tensor<1x32x1x1xf32>    = dense<0.000000e+00> : tensor<1x32x1x1xf32>
    %cst_4 = const.Declare tensor<1x32x1x1xf16>    = dense<0xFC00>       : tensor<1x32x1x1xf16>
    %cst_5 = const.Declare tensor<1x32x1x128xf16>  = dense<0.000000e+00> : tensor<1x32x1x128xf16>

    %value_reordered = VPU.MemPermute(%arg2) {dst_order = #NCWH, mem_perm = #NCWH}
        : tensor<1x4x1024x128xf16> -> tensor<1x4x1024x128xf16, {order = #NCWH}>

    %result_running_output, %result_running_max, %result_running_sum =
        VPU.FlashSDPA(%arg0, %arg1, %value_reordered, %cst_2, %cst_1, %cst_0, %cst,
                      %cst_5, %cst_4, %cst_3, %arg3) {
            is_head = true,
            is_tail = true,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            source_seq_len_pad_size = 0 : i64
        } : tensor<1x32x1x128xf16>, tensor<1x4x1024x128xf16>,
            tensor<1x4x1024x128xf16, {order = #NCWH}>,
            tensor<1x8x1x1024xf16>, tensor<1x1x2x256xsi32>, tensor<1x1x1024x4xsi32>,
            tensor<1x1x128x4xsi32>, tensor<1x32x1x128xf16>, tensor<1x32x1x1xf16>,
            tensor<1x32x1x1xf32>, tensor<1x1x1x1024xf16>
        -> tensor<1x32x1x128xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x1xf32>

    return %result_running_output : tensor<1x32x1x128xf16>

    // CHECK-DAG:   [[VALUE_REORDERED:%.+]] = VPU.MemPermute([[ARG:%.+]]) {dst_order = #NCWH, mem_perm = #NCWH}
    // CHECK:       [[Q0:%.+]] = VPU.Slice [[QUERY:%.+]] [0, 0, 0, 0] [1, 24, 1, 128]
    // CHECK-SAME:      tensor<1x32x1x128xf16> to tensor<1x24x1x128xf16>
    // CHECK:       [[K0:%.+]] = VPU.Slice [[KEY:%.+]] [0, 0, 0, 0] [1, 3, 1024, 128]
    // CHECK-SAME:      tensor<1x4x1024x128xf16> to tensor<1x3x1024x128xf16>
    // CHECK:       [[V0:%.+]] = VPU.Slice [[VALUE_REORDERED]] [0, 0, 0, 0] [1, 3, 1024, 128]
    // CHECK:       VPU.FlashSDPA([[Q0]], [[K0]], [[V0]],
    // CHECK-SAME:      SplitOverKernel
    // CHECK-SAME:      -> tensor<1x24x1x128xf16>, tensor<1x24x1x1xf16>, tensor<1x24x1x1xf32>

    // CHECK:       [[Q1:%.+]] = VPU.Slice [[QUERY]] [0, 24, 0, 0] [1, 8, 1, 128]
    // CHECK-SAME:      tensor<1x32x1x128xf16> to tensor<1x8x1x128xf16>
    // CHECK:       [[K1:%.+]] = VPU.Slice [[KEY]] [0, 3, 0, 0] [1, 1, 1024, 128]
    // CHECK-SAME:      tensor<1x4x1024x128xf16> to tensor<1x1x1024x128xf16>
    // CHECK:       [[V1:%.+]] = VPU.Slice [[VALUE_REORDERED]] [0, 3, 0, 0] [1, 1, 1024, 128]

    // CHECK:       VPU.FlashSDPA([[Q1]], [[K1]], [[V1]],
    // CHECK-SAME:      SplitOverKernel
    // CHECK-SAME:      -> tensor<1x8x1x128xf16>, tensor<1x8x1x1xf16>, tensor<1x8x1x1xf32>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat(
    // CHECK-SAME:      -> tensor<1x32x1x128xf16>
    // CHECK:       return [[CONCAT]]
}

// -----

// MQA prefill that has to split a GQA group to afford Q-tile pipelining: 16 Q heads and 1 KV head,
// so one whole group spans all 16 heads. With QKEmbed=VEmbed=512 the per-Q-tile buffers
// (query + input/result running output) then cost 3 x 16 heads x 16 rows x 512 x 2 = 786432 bytes per
// cluster at the smallest query tile (alignment[H] = clusters * shaves * 8 = 48, i.e. 16 rows per
// cluster), so keeping a second copy free for prefetching exceeds the 1470720 usable CMX bytes no
// matter how far the Key/Value sequence is split. Halving the head tile to 8 Q heads halves the Q side
// and fits with the reservation - the single KV head is still read whole by both head tiles.

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

config.Resources 3 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @FlashSDPA_MQA_SubGroupHeadTiling
func.func @FlashSDPA_MQA_SubGroupHeadTiling(%arg0: tensor<1x16x1024x512xf16>, %arg1: tensor<1x1x1024x512xf16>, %arg2: tensor<1x1x1024x512xf16>, %arg3: tensor<1x16x1024x1024xf16>)
                                  -> tensor<1x16x1024x512xf16> {
    %cst = const.Declare tensor<1x1x512x4xsi32> = dense<0> : tensor<1x1x512x4xsi32>
    %cst_0 = const.Declare tensor<1x1x1024x4xsi32> = dense<0> : tensor<1x1x1024x4xsi32>
    %cst_1 = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
    %cst_2 = const.Declare tensor<1x2x1024x1024xf16> = dense<0.000000e+00> : tensor<1x2x1024x1024xf16>
    %cst_3 = const.Declare tensor<1x16x1024x1xf32> = dense<0.000000e+00> : tensor<1x16x1024x1xf32>
    %cst_4 = const.Declare tensor<1x16x1024x1xf16> = dense<0xFC00> : tensor<1x16x1024x1xf16>
    %cst_5 = const.Declare tensor<1x16x1024x512xf16> = dense<0.000000e+00> : tensor<1x16x1024x512xf16>

    %value_reordered = VPU.MemPermute(%arg2) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x1x1024x512xf16> -> tensor<1x1x1024x512xf16, {order = #NCWH}>

    %result_running_output, %result_running_max, %result_running_sum =
        VPU.FlashSDPA(%arg0, %arg1, %value_reordered, %cst_2, %cst_1, %cst_0, %cst, %cst_5, %cst_4, %cst_3, %arg3) {
            is_head = false,
            is_tail = true,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            source_seq_len_pad_size = 0 : i64
        } : tensor<1x16x1024x512xf16>, tensor<1x1x1024x512xf16>, tensor<1x1x1024x512xf16, {order = #NCWH}>,
            tensor<1x2x1024x1024xf16>, tensor<1x1x2x256xsi32>, tensor<1x1x1024x4xsi32>,
            tensor<1x1x512x4xsi32>, tensor<1x16x1024x512xf16>, tensor<1x16x1024x1xf16>,
            tensor<1x16x1024x1xf32>, tensor<1x16x1024x1024xf16>
        -> tensor<1x16x1024x512xf16>, tensor<1x16x1024x1xf16>, tensor<1x16x1024x1xf32>

    return %result_running_output : tensor<1x16x1024x512xf16>

    // 2 head tiles x 5 KV blocks x 22 query tiles = 220 FlashSDPA ops.
    // Head tiles: 2 x 8 (half of the 16-head GQA group), both against the single KV head.
    // KV blocks: 4 x 208 + 1 x 192 (alignValUp(divUp(1024, 5), 16) = 208)
    // Query tiles: 21 x 48 + 1 x 16

    // First head tile, first KV block, first query tile
    // CHECK-DAG:   [[VALUE_REORDERED:%.+]] = VPU.MemPermute({{%.+}}) {dst_order = #NCWH, mem_perm = #NCWH}
    // CHECK:       [[HEAD0_Q:%.+]] = VPU.Slice {{%.+}} [0, 0, 0, 0] [1, 8, 1024, 512] : tensor<1x16x1024x512xf16>
    // CHECK:       [[HEAD0_MASK:%.+]] = VPU.Slice {{%.+}} [0, 0, 0, 0] [1, 8, 1024, 1024] : tensor<1x16x1024x1024xf16>
    // CHECK:       [[KEY0:%.+]] = VPU.Slice {{%.+}} [0, 0, 0, 0] [1, 1, 208, 512]
    // CHECK:       [[VAL0:%.+]] = VPU.Slice [[VALUE_REORDERED]] [0, 0, 0, 0] [1, 1, 208, 512]
    // CHECK:       [[MASK0:%.+]] = VPU.Slice [[HEAD0_MASK]] [0, 0, 0, 0] [1, 8, 1024, 208]
    // CHECK:       [[Q0:%.+]] = VPU.Slice [[HEAD0_Q]] [0, 0, 0, 0] [1, 8, 48, 512]
    // CHECK:       VPU.FlashSDPA([[Q0]], [[KEY0]], [[VAL0]],
    // CHECK-SAME:      -> tensor<1x8x48x512xf16>, tensor<1x8x48x1xf16>, tensor<1x8x48x1xf32>

    // CHECK-COUNT-219: VPU.FlashSDPA

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat{{.*}}-> tensor<1x16x1024x512xf16>
    // CHECK:       return [[CONCAT]]
}

// -----

// GQA prefill that has to split a GQA group to afford Q-tile pipelining: 32 Q heads and 2 KV heads,
// so one whole group spans 16 Q heads. As in the MQA case above, QKEmbed=VEmbed=512 makes a 16-head
// tile too expensive to keep a prefetch copy of, so the estimator halves it to 8 Q heads. Unlike the
// MQA case, the tiles then start at C offsets that are not multiples of the group size (8 and 24), and
// each one still has to map onto the single KV head that owns it: Q heads 0-15 read KV head 0 and
// Q heads 16-31 read KV head 1.

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

config.Resources 3 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @FlashSDPA_GQA_SubGroupHeadTiling
func.func @FlashSDPA_GQA_SubGroupHeadTiling(%arg0: tensor<1x32x192x512xf16>, %arg1: tensor<1x2x1024x512xf16>, %arg2: tensor<1x2x1024x512xf16>, %arg3: tensor<1x32x192x1024xf16>)
                                  -> tensor<1x32x192x512xf16> {
    %cst = const.Declare tensor<1x1x512x4xsi32> = dense<0> : tensor<1x1x512x4xsi32>
    %cst_0 = const.Declare tensor<1x1x1024x4xsi32> = dense<0> : tensor<1x1x1024x4xsi32>
    %cst_1 = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
    %cst_2 = const.Declare tensor<1x2x192x1024xf16> = dense<0.000000e+00> : tensor<1x2x192x1024xf16>
    %cst_3 = const.Declare tensor<1x32x192x1xf32> = dense<0.000000e+00> : tensor<1x32x192x1xf32>
    %cst_4 = const.Declare tensor<1x32x192x1xf16> = dense<0xFC00> : tensor<1x32x192x1xf16>
    %cst_5 = const.Declare tensor<1x32x192x512xf16> = dense<0.000000e+00> : tensor<1x32x192x512xf16>

    %value_reordered = VPU.MemPermute(%arg2) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x2x1024x512xf16> -> tensor<1x2x1024x512xf16, {order = #NCWH}>

    %result_running_output, %result_running_max, %result_running_sum =
        VPU.FlashSDPA(%arg0, %arg1, %value_reordered, %cst_2, %cst_1, %cst_0, %cst, %cst_5, %cst_4, %cst_3, %arg3) {
            is_head = false,
            is_tail = true,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            source_seq_len_pad_size = 0 : i64
        } : tensor<1x32x192x512xf16>, tensor<1x2x1024x512xf16>, tensor<1x2x1024x512xf16, {order = #NCWH}>,
            tensor<1x2x192x1024xf16>, tensor<1x1x2x256xsi32>, tensor<1x1x1024x4xsi32>,
            tensor<1x1x512x4xsi32>, tensor<1x32x192x512xf16>, tensor<1x32x192x1xf16>,
            tensor<1x32x192x1xf32>, tensor<1x32x192x1024xf16>
        -> tensor<1x32x192x512xf16>, tensor<1x32x192x1xf16>, tensor<1x32x192x1xf32>

    return %result_running_output : tensor<1x32x192x512xf16>

    // 4 head tiles x 5 KV blocks x 4 query tiles = 80 FlashSDPA ops.
    // Head tiles: 4 x 8 Q heads at C offsets 0, 8, 16, 24 - half of the 16-head GQA group.
    // KV blocks: 4 x 208 + 1 x 192. Query tiles: 4 x 48.

    // First head tile (Q heads 0-7, KV head 0), first KV block, first query tile
    // CHECK-DAG:   [[VALUE_REORDERED:%.+]] = VPU.MemPermute({{%.+}}) {dst_order = #NCWH, mem_perm = #NCWH}
    // CHECK:       [[HEAD0_Q:%.+]] = VPU.Slice {{%.+}} [0, 0, 0, 0] [1, 8, 192, 512] : tensor<1x32x192x512xf16>
    // CHECK:       [[HEAD0_KEY:%.+]] = VPU.Slice {{%.+}} [0, 0, 0, 0] [1, 1, 1024, 512] : tensor<1x2x1024x512xf16>
    // CHECK:       [[HEAD0_VAL:%.+]] = VPU.Slice [[VALUE_REORDERED]] [0, 0, 0, 0] [1, 1, 1024, 512]
    // CHECK:       [[HEAD0_MASK:%.+]] = VPU.Slice {{%.+}} [0, 0, 0, 0] [1, 8, 192, 1024] : tensor<1x32x192x1024xf16>
    // CHECK:       [[KEY0:%.+]] = VPU.Slice [[HEAD0_KEY]] [0, 0, 0, 0] [1, 1, 208, 512]
    // CHECK:       [[VAL0:%.+]] = VPU.Slice [[HEAD0_VAL]] [0, 0, 0, 0] [1, 1, 208, 512]
    // CHECK:       [[MASK0:%.+]] = VPU.Slice [[HEAD0_MASK]] [0, 0, 0, 0] [1, 8, 192, 208]
    // CHECK:       [[Q0:%.+]] = VPU.Slice [[HEAD0_Q]] [0, 0, 0, 0] [1, 8, 48, 512]
    // CHECK:       VPU.FlashSDPA([[Q0]], [[KEY0]], [[VAL0]],
    // CHECK-SAME:      -> tensor<1x8x48x512xf16>, tensor<1x8x48x1xf16>, tensor<1x8x48x1xf32>

    // The second head tile starts inside the same GQA group, so it reads KV head 0 as well
    // CHECK:       [[HEAD1_Q:%.+]] = VPU.Slice {{%.+}} [0, 8, 0, 0] [1, 8, 192, 512] : tensor<1x32x192x512xf16>
    // CHECK:       [[HEAD1_MASK:%.+]] = VPU.Slice {{%.+}} [0, 8, 0, 0] [1, 8, 192, 1024] : tensor<1x32x192x1024xf16>

    // The third head tile crosses into the second group and switches to KV head 1
    // CHECK:       [[HEAD2_Q:%.+]] = VPU.Slice {{%.+}} [0, 16, 0, 0] [1, 8, 192, 512] : tensor<1x32x192x512xf16>
    // CHECK:       [[HEAD2_KEY:%.+]] = VPU.Slice {{%.+}} [0, 1, 0, 0] [1, 1, 1024, 512] : tensor<1x2x1024x512xf16>

    // CHECK:       [[HEAD3_Q:%.+]] = VPU.Slice {{%.+}} [0, 24, 0, 0] [1, 8, 192, 512] : tensor<1x32x192x512xf16>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat{{.*}}-> tensor<1x32x192x512xf16>
    // CHECK:       return [[CONCAT]]
}
