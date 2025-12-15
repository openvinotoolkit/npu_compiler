//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% weights-table-reuse-mode=ENABLED" --set-zero-offset-weights-table %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotSetIsZeroOffsetWTForNCEClusterTask
func.func @DoNotSetIsZeroOffsetWTForNCEClusterTask(%input: !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>>,
                        %weights: !VPUIP.SparseBuffer<data=memref<64x32x3x3xf16, #NHWC, @CMX_NN>, sparsity_map=memref<64x1x1x384xi1, @CMX_NN>, is_weights>,
                        %weight_table: memref<64x1x1x4xsi32, @CMX_NN>)
                    -> !VPUIP.SparseBuffer<data=memref<1x64x14x14xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x64x14x14xi1, #NHWC, @CMX_NN>> {
    %outputs = memref.alloc() : memref<1x64x14x14xf16, #NHWC, @CMX_NN>
    %output_sparsity_map = memref.alloc() : memref<1x64x14x14xi1, #NHWC, @CMX_NN>
    %in_data, %in_sparsityMap = VPUIP.UngroupSparseBuffer(%input) {resultSegmentSizes = array<i32: 1, 1, 0>}
            -> memref<1x32x16x16xf16, #NHWC, @CMX_NN>, memref<1x32x16x16xi1, #NHWC, @CMX_NN>
    %w_data, %w_sparsityMap = VPUIP.UngroupSparseBuffer(%weights) {resultSegmentSizes = array<i32: 1, 1, 0>}
            -> memref<64x32x3x3xf16, #NHWC, @CMX_NN>, memref<64x1x1x384xi1, @CMX_NN>
    %nce_task:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                                        kernel_size = [3, 3], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
        input(%in_data : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
        input_sparsity_map(%in_sparsityMap : memref<1x32x16x16xi1, #NHWC, @CMX_NN>)
        weights(%w_data : memref<64x32x3x3xf16, #NHWC, @CMX_NN>)
        weights_sparsity_map(%w_sparsityMap : memref<64x1x1x384xi1, @CMX_NN>)
        weight_table(%weight_table : memref<64x1x1x4xsi32, @CMX_NN>)
        parent_input(%in_data : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
        parent_input_sparsity_map(%in_sparsityMap : memref<1x32x16x16xi1, #NHWC, @CMX_NN>)
        parent_output(%outputs : memref<1x64x14x14xf16, #NHWC, @CMX_NN>)
        parent_output_sparsity_map(%output_sparsity_map : memref<1x64x14x14xi1, #NHWC, @CMX_NN>)
        outputs(%outputs : memref<1x64x14x14xf16, #NHWC, @CMX_NN>)
        output_sparsity_map(%output_sparsity_map : memref<1x64x14x14xi1, #NHWC, @CMX_NN>)
        -> memref<1x64x14x14xf16, #NHWC, @CMX_NN>, memref<1x64x14x14xi1, #NHWC, @CMX_NN> variants : {
            DPUTask {mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 15, 31], outStart = [0, 0, 0],
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        } PPE : {
            PPETask {ppe = #VPU.PPEStub<>}
    }
    %out = VPUIP.GroupSparseBuffer(%nce_task#0, %nce_task#1)
            -> !VPUIP.SparseBuffer<data=memref<1x64x14x14xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x64x14x14xi1, #NHWC, @CMX_NN>>
    return %out : !VPUIP.SparseBuffer<data=memref<1x64x14x14xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x64x14x14xi1, #NHWC, @CMX_NN>>

    // CHECK:       VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      kernel_size = [3, 3], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:      input({{[^:]+}} : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      input_sparsity_map({{[^:]+}} : memref<1x32x16x16xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weights({{[^:]+}} : memref<64x32x3x3xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weights_sparsity_map({{[^:]+}} : memref<64x1x1x384xi1, @CMX_NN>)
    // CHECK-SAME:      weight_table({{[^:]+}} : memref<64x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      parent_input({{[^:]+}} : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_input_sparsity_map({{[^:]+}} : memref<1x32x16x16xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_output({{[^:]+}} : memref<1x64x14x14xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_output_sparsity_map({{[^:]+}} : memref<1x64x14x14xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs({{[^:]+}} : memref<1x64x14x14xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      output_sparsity_map({{[^:]+}} : memref<1x64x14x14xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:      -> memref<1x64x14x14xf16, #NHWC, @CMX_NN>, memref<1x64x14x14xi1, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SetIsZeroOffsetWTForNCEClusterTask
func.func @SetIsZeroOffsetWTForNCEClusterTask(%input: memref<1x32x16x16xf16, #NHWC, @CMX_NN>) -> memref<1x64x14x14xf16, #NHWC, @CMX_NN> {
    %weights = const.Declare memref<64x32x3x3xf16, #NHWC, @CMX_NN> = dense<1.000000e+00> : tensor<64x32x3x3xf16, {mem_space = @CMX_NN}>, [#const.Reorder<#NHWC>]
    %weight_table = const.Declare memref<64x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>
    %output = memref.alloc() : memref<1x64x14x14xf16, #NHWC, @CMX_NN>
    %nce_task = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                                        kernel_size = [3, 3], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%input : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    weights(%weights : memref<64x32x3x3xf16, #NHWC, @CMX_NN>)
    weight_table(%weight_table : memref<64x1x1x4xsi32, @CMX_NN>)
    parent_input(%input : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    parent_output(%output : memref<1x64x14x14xf16, #NHWC, @CMX_NN>)
    outputs(%output : memref<1x64x14x14xf16, #NHWC, @CMX_NN>)
    -> memref<1x64x14x14xf16, #NHWC, @CMX_NN> variants : {
      DPUTask {mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 15, 31], outStart = [0, 0, 0],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    return %nce_task : memref<1x64x14x14xf16, #NHWC, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask {is_zero_offset_weights_table,
    // CHECK-SAME:      kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      kernel_size = [3, 3], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:      input({{[^:]+}} : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weights({{[^:]+}} : memref<64x32x3x3xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weight_table({{[^:]+}} : memref<64x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      parent_input({{[^:]+}} : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_output({{[^:]+}} : memref<1x64x14x14xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs({{[^:]+}} : memref<1x64x14x14xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      -> memref<1x64x14x14xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x32x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    32x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!WeightsTableStub_CMX = memref<32x1x1x4xsi32, #NCHW, @CMX_NN>

// CHECK-LABEL: @SetIsZeroOffsetWTForNCEClusterTask
func.func @SetIsZeroOffsetWTForNCEClusterTask(%input: !InputDistributed, %weights: !WeightsDistributed) -> !OutputDistributed {
    %weights_table_cst = const.Declare memref<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %output = VPURT.AllocDistributed -> !OutputDistributed
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed

    %weights_table = VPUIP.Copy inputs(%weights_table_cst: memref<32x1x1x4xsi32>) outputs(%weights_table_cmx: !WeightsTableDistributed) -> !WeightsTableDistributed
    %nce_task = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
        } input(%input : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%weights_table : !WeightsTableDistributed)
        parent_input(%input : !InputDistributed)
        parent_output(%output : !OutputDistributed)
        outputs(%output : !OutputDistributed)
        -> !OutputDistributed variants :  {
        DPUTask {
                outStart = [0, 0, 0], outEnd = [31, 31, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                cluster_id = 0 : i64
        }
        DPUTask {
                outStart = [0, 0, 16], outEnd = [31, 31, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                cluster_id = 1 : i64
        }
        } PPE :  {
    }

    return %nce_task : !OutputDistributed

    // CHECK:   VPUIP.NCEClusterTask {is_zero_offset_weights_table,
    // CHECK-SAME:      kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:      input({{[^:]+}} : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:      weights({{[^:]+}} : !VPUIP.DistributedBuffer<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:      weight_table({{[^:]+}} : !VPUIP.DistributedBuffer<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:      parent_input({{[^:]+}} : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:      parent_output({{[^:]+}} : !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:      outputs({{[^:]+}} : !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    8x1x64x3x1xf16, #GNHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [4, 1, 1, 1, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[2, 1, 64, 3, 1], [2, 1, 64, 3, 1], [2, 1, 64, 3, 1], [2, 1, 64, 3, 1]],
    compute_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [6, 0, 0, 0, 0]],
    memory_shapes = [[2, 1, 64, 3, 1], [2, 1, 64, 3, 1], [2, 1, 64, 3, 1], [2, 1, 64, 3, 1]],
    memory_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [6, 0, 0, 0, 0]]
}>

// CHECK-LABEL: @SetZeroOffsetWTFor5DShape
func.func @SetZeroOffsetWTFor5DShape(%input: memref<8x1x1024x3x1xf16, #GNHWC, @CMX_NN>, %weights: memref<8x64x1024x1x1xf16, #GNHWC, @CMX_NN>, %weights_table: memref<8x64x1x1x4xsi32, @CMX_NN>) -> !OutputDistributed {
    %alloc = VPURT.AllocDistributed -> !OutputDistributed
    %nce_cluster_tiling = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}
        input(%input : memref<8x1x1024x3x1xf16, #GNHWC, @CMX_NN>)
        weights(%weights : memref<8x64x1024x1x1xf16, #GNHWC, @CMX_NN>)
        weight_table(%weights_table : memref<8x64x1x1x4xsi32, @CMX_NN>)
        parent_input(%input : memref<8x1x1024x3x1xf16, #GNHWC, @CMX_NN>)
        parent_output(%alloc : !OutputDistributed)
        outputs(%alloc : !OutputDistributed)
    ->  !OutputDistributed variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 2, 1023], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [0, 2, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 2, 1023], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [0, 2, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 2 : i64, inEnd = [0, 2, 1023], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [0, 2, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 3 : i64, inEnd = [0, 2, 1023], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [0, 2, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }

    return %nce_cluster_tiling : !OutputDistributed

    // CHECK:   VPUIP.NCEClusterTask {is_zero_offset_weights_table,
    // CHECK-SAME:      kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:      input({{[^:]+}} : memref<8x1x1024x3x1xf16, #GNHWC, @CMX_NN>)
    // CHECK-SAME:      weights({{[^:]+}} : memref<8x64x1024x1x1xf16, #GNHWC, @CMX_NN>)
    // CHECK-SAME:      weight_table({{[^:]+}} : memref<8x64x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      parent_input({{[^:]+}} : memref<8x1x1024x3x1xf16, #GNHWC, @CMX_NN>)
    // CHECK-SAME:      parent_output({{[^:]+}} : !VPUIP.DistributedBuffer<8x1x64x3x1xf16
    // CHECK-SAME:      outputs({{[^:]+}} : !VPUIP.DistributedBuffer<8x1x64x3x1xf16
    // CHECK-SAME{LITERAL}:      -> !VPUIP.DistributedBuffer<8x1x64x3x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[2, 1, 64, 3, 1], [2, 1, 64, 3, 1], [2, 1, 64, 3, 1], [2, 1, 64, 3, 1]], compute_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [6, 0, 0, 0, 0]], memory_shapes = [[2, 1, 64, 3, 1], [2, 1, 64, 3, 1], [2, 1, 64, 3, 1], [2, 1, 64, 3, 1]], memory_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [6, 0, 0, 0, 0]]}>
}
