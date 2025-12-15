//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --mlir-print-elementsattrs-with-hex-if-larger=-1 --compute-se-base-ptrs %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x4x4xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 16, 2, 4], [1, 16, 2, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]],
    memory_shapes = [[1, 16, 3, 4], [1, 16, 3, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0]]
}>

!InputSMDistributed = !VPUIP.DistributedBuffer<
    1x16x8x8xi1, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 16, 4, 8], [1, 16, 4, 8]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]],
    memory_shapes = [[1, 16, 4, 8], [1, 16, 4, 8]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]
}>

!InputSEDistributed = !VPUIP.DistributedBuffer<
    1x1x8x8xi32, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 1, 4, 8], [1, 1, 4, 8]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]],
    memory_shapes = [[1, 1, 4, 8], [1, 1, 4, 8]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x8x8xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 16, 4, 8], [1, 16, 4, 8]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]],
    memory_shapes = [[1, 16, 4, 8], [1, 16, 4, 8]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsSMDistributed = !VPUIP.DistributedBuffer<
    16x1x1x128xi1, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x16x4x4xf16, #NHWC>
!InputSM_DDR = memref<1x16x8x8xi1, #NHWC>
!InputSE_DDR = memref<1x1x8x8xi32, #NHWC, @DDR>
!Weights_DDR = memref<16x16x1x1xf16, #NHWC>
!WeightsSM_DDR = memref<16x1x1x128xi1>

!Input_CMX = memref<1x16x4x4xf16, #NHWC, @CMX_NN>
!InputSM_CMX = memref<1x16x8x8xi1, #NHWC, @CMX_NN>
!InputSE_CMX = memref<1x1x8x8xi32, #NHWC, @CMX_NN>
!Weights_CMX = memref<16x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsSM_CMX = memref<16x1x1x128xi1, @CMX_NN>
!Output_CMX = memref<1x16x8x8xf16, #NHWC, @CMX_NN>

// CHECK-LABEL:  func.func @SEPaddingWithLargePadSize
func.func @SEPaddingWithLargePadSize(%input_data: !Input_DDR, %input_sm: !InputSM_DDR) -> !OutputDistributed {
    %input_se = VPUIP.StorageElementTable {
                dataElemType = f16, dataShape = [1, 16, 4, 4], seAttr = #VPU.SEPadding<mode = <REFLECT>,
                padding = [2, 2, 2, 2]>, seDepth = 1 : i64, seSize = [16]
            } -> memref<1x1x8x8xi32, #NHWC, @DDR>
    %input_sparse = VPUIP.GroupSparseBuffer (%input_data, %input_sm, %input_se)
        -> !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR>

    %input_data_cmx = VPURT.AllocDistributed -> !InputDistributed
    %input_sm_cmx = VPURT.AllocDistributed -> !InputSMDistributed
    %input_se_cmx = VPURT.AllocDistributed -> !InputSEDistributed
    %input_sparse_cmx = VPUIP.GroupSparseBuffer (%input_data_cmx, %input_sm_cmx, %input_se_cmx)
        -> !VPUIP.SparseBuffer<data=!InputDistributed, sparsity_map=!InputSMDistributed, storage_element_table=!InputSEDistributed>
    %input = VPUIP.Copy
        inputs(%input_sparse : !VPUIP.SparseBuffer<data = !Input_DDR, sparsity_map = !InputSM_DDR, storage_element_table = !InputSE_DDR>)
        outputs(%input_sparse_cmx : !VPUIP.SparseBuffer<data = !InputDistributed, sparsity_map = !InputSMDistributed, storage_element_table = !InputSEDistributed>)  -> !VPUIP.SparseBuffer<data = !InputDistributed, sparsity_map = !InputSMDistributed, storage_element_table = !InputSEDistributed>

    %cst_weights = const.Declare !Weights_DDR = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %cst_weights_sm = const.Declare !WeightsSM_DDR = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPUIP.GroupSparseBuffer (%cst_weights, %cst_weights_sm) {is_weights}
        -> !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=!WeightsSM_DDR, is_weights>

    %weights_data_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %weights_sm_cmx = VPURT.AllocDistributed -> !WeightsSMDistributed
    %weights_sparse_cmx = VPUIP.GroupSparseBuffer (%weights_data_cmx, %weights_sm_cmx) {is_weights}
        -> !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights>
    %weights = VPUIP.Copy
        inputs(%weights_sparse : !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=!WeightsSM_DDR, is_weights>)
        outputs(%weights_sparse_cmx : !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights>)  -> !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights>

    %in_data, %in_sm, %in_se = VPUIP.UngroupSparseBuffer(%input) {resultSegmentSizes = array<i32: 1, 1, 1>}
        -> !InputDistributed, !InputSMDistributed, !InputSEDistributed
    %w_data, %w_sm = VPUIP.UngroupSparseBuffer(%weights)  {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> !WeightsDistributed, !WeightsSMDistributed
    %out_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %conv_out = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
        input(%in_data : !InputDistributed)
        input_sparsity_map(%in_sm : !InputSMDistributed)
        input_storage_element_table(%in_se : !InputSEDistributed)
        weights(%w_data : !WeightsDistributed)
        weights_sparsity_map(%w_sm : !WeightsSMDistributed)
        parent_input(%in_data : !InputDistributed)
        parent_input_sparsity_map(%in_sm : !InputSMDistributed)
        parent_input_storage_element_table(%in_se : !InputSEDistributed)
        parent_output(%out_cmx : !OutputDistributed)
        outputs(%out_cmx : !OutputDistributed)
    -> !OutputDistributed variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [7, 3, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [7, 3, 15], outStart = [0, 4, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }

    return %conv_out : !OutputDistributed

    // CHECK:       VPUIP.StorageElementTable {
    // CHECK-SAME:    basePtrs = dense<[0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-SAME:                      0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-SAME:                      0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-SAME:                      0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-SAME:                      0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-SAME:                      1, 1, 1, 1, 1, 1, 1, 1,
    // CHECK-SAME:                      1, 1, 1, 1, 1, 1, 1, 1,
    // CHECK-SAME:                      1, 1, 1, 1, 1, 1, 1, 1]> : tensor<64xi32>,
    // CHECK-SAME:    dataElemType = f16, dataShape = [1, 16, 4, 4],
    // CHECK-SAME:    seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [2, 2, 2, 2]>, seDepth = 1 : i64, seSize = [16]
    // CHECK-SAME:  } -> memref<1x1x8x8xi32, #NHWC, @DDR>
}
