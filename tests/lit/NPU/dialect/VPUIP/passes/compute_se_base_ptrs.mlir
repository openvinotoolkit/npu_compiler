//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --mlir-print-elementsattrs-with-hex-if-larger=-1 --compute-se-base-ptrs %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_DDR = memref<1x16x3x3xf16, #NHWC>
!InputSM_DDR = memref<1x16x3x3xi1, #NHWC>
!InputSE_DDR = memref<1x1x3x3xi32, #NHWC>
!Weights_DDR = memref<16x16x1x1xf16, #NHWC>
!WeightsSM_DDR = memref<16x1x1x128xi1>
!WeightsTable_DDR = memref<16x1x1x4xsi32>

!Input_CMX = memref<1x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
!InputSM_CMX = memref<1x16x3x3xi1, #NHWC, [@CMX_NN, 0]>
!InputSE_CMX = memref<1x1x3x3xi32, #NHWC, [@CMX_NN, 0]>
!Weights_CMX = memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
!WeightsSM_CMX = memref<16x1x1x128xi1, [@CMX_NN, 0]>
!WeightsTable_CMX = memref<16x1x1x4xsi32, [@CMX_NN, 0]>
!Output_CMX = memref<1x16x3x3xf16, #NHWC, [@CMX_NN, 0]>

// CHECK-LABEL:  func.func @SETableSingleCluster
func.func @SETableSingleCluster(%input_data: !Input_DDR, %input_sm: !InputSM_DDR) -> !Output_CMX {
    %input_se = VPUIP.StorageElementTable {
                dataShape = [1, 16, 3, 3], dataElemType = f16,
                seSize = 16, seDepth = 1
            } -> !InputSE_DDR
    %input_sparse = VPUIP.GroupSparseBuffer (%input_data, %input_sm, %input_se)
        -> !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR>

    %input_data_cmx = memref.alloc() : !Input_CMX
    %input_sm_cmx = memref.alloc() : !InputSM_CMX
    %input_se_cmx = memref.alloc() : !InputSE_CMX
    %input_sparse_cmx = VPUIP.GroupSparseBuffer (%input_data_cmx, %input_sm_cmx, %input_se_cmx)
        -> !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX>

    %input = VPUIP.Copy inputs(%input_sparse: !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR>)
                   outputs(%input_sparse_cmx: !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX>)
        -> !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX>

    %cst_weights = const.Declare !Weights_DDR = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %cst_weights_sm = const.Declare !WeightsSM_DDR = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPUIP.GroupSparseBuffer (%cst_weights, %cst_weights_sm) {is_weights}
        -> !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=!WeightsSM_DDR, is_weights>

    %weights_data_cmx = memref.alloc() : !Weights_CMX
    %weights_sm_cmx = memref.alloc() : !WeightsSM_CMX
    %weights_sparse_cmx = VPUIP.GroupSparseBuffer (%weights_data_cmx, %weights_sm_cmx) {is_weights}
        -> !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=!WeightsSM_CMX, is_weights>

    %weights = VPUIP.Copy inputs(%weights_sparse: !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=!WeightsSM_DDR, is_weights>)
                     outputs(%weights_sparse_cmx: !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=!WeightsSM_CMX, is_weights>)
        -> !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=!WeightsSM_CMX, is_weights>

    %cst_weights_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<16x1x1x4xsi32>
    %weights_table_cmx = memref.alloc() : !WeightsTable_CMX

    %weights_table = VPUIP.Copy inputs(%cst_weights_table: !WeightsTable_DDR) outputs(%weights_table_cmx: !WeightsTable_CMX) -> !WeightsTable_CMX

    %in_data, %in_sm, %in_se = VPUIP.UngroupSparseBuffer(%input) {resultSegmentSizes = array<i32: 1, 1, 1>}
        -> !Input_CMX, !InputSM_CMX, !InputSE_CMX
    %w_data, %w_sm = VPUIP.UngroupSparseBuffer(%weights)  {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> !Weights_CMX, !WeightsSM_CMX
    %out_cmx = memref.alloc() : !Output_CMX

    %conv_out = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%in_data : !Input_CMX)
        input_sparsity_map(%in_sm : !InputSM_CMX)
        input_storage_element_table(%in_se : !InputSE_CMX)
        weights(%w_data : !Weights_CMX)
        weights_sparsity_map(%w_sm : !WeightsSM_CMX)
        weight_table(%weights_table : !WeightsTable_CMX)
        parent_input(%in_data : !Input_CMX)
        parent_input_sparsity_map(%in_sm : !InputSM_CMX)
        parent_input_storage_element_table(%in_se : !InputSE_CMX)
        parent_output(%out_cmx : !Output_CMX)
        outputs(%out_cmx : !Output_CMX)
            -> !Output_CMX
        variants : {
            DPUTask {cluster_id = 0 : i64, outStart = [0, 0, 0], outEnd = [16, 3, 3], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                     pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        } PPE : {
        }

    return %conv_out : !Output_CMX

    // CHECK:       VPUIP.StorageElementTable {
    // CHECK-SAME:    basePtrs = dense<0> : tensor<9xi32>,
    // CHECK-SAME:    dataElemType = f16, dataShape = [1, 16, 3, 3],
    // CHECK-SAME:    seDepth = 1 : i64, seSize = 16 : i64
    // CHECK-SAME:  } -> memref<1x1x3x3xi32, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!InputSMDistributed = !VPUIP.DistributedBuffer<
    1x16x3x3xi1, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!InputSEDistributed = !VPUIP.DistributedBuffer<
    1x1x3x3xi32, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
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

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x16x3x3xf16, #NHWC>
!InputSM_DDR = memref<1x16x3x3xi1, #NHWC>
!InputSE_DDR = memref<1x1x3x3xi32, #NHWC>
!Weights_DDR = memref<16x16x1x1xf16, #NHWC>
!WeightsSM_DDR = memref<16x1x1x128xi1>
!WeightsTable_DDR = memref<16x1x1x4xsi32>

!Input_CMX = memref<1x16x3x3xf16, #NHWC, @CMX_NN>
!InputSM_CMX = memref<1x16x3x3xi1, #NHWC, @CMX_NN>
!InputSE_CMX = memref<1x1x3x3xi32, #NHWC, @CMX_NN>
!Weights_CMX = memref<16x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsSM_CMX = memref<16x1x1x128xi1, @CMX_NN>
!WeightsTable_CMX = memref<16x1x1x4xsi32, @CMX_NN>
!Output_CMX = memref<1x16x3x3xf16, #NHWC, @CMX_NN>

// CHECK-LABEL:  func.func @SETableMultiCluster
func.func @SETableMultiCluster(%input_data: !Input_DDR, %input_sm: !InputSM_DDR) -> !OutputDistributed {
    %input_se = VPUIP.StorageElementTable {
                dataShape = [1, 16, 3, 3], dataElemType = f16,
                seSize = 16, seDepth = 1
            } -> !InputSE_DDR
    %input_sparse = VPUIP.GroupSparseBuffer (%input_data, %input_sm, %input_se)
        -> !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR>

    %input_data_cmx = VPURT.AllocDistributed -> !InputDistributed
    %input_sm_cmx = VPURT.AllocDistributed -> !InputSMDistributed
    %input_se_cmx = VPURT.AllocDistributed -> !InputSEDistributed
    %input_sparse_cmx = VPUIP.GroupSparseBuffer (%input_data_cmx, %input_sm_cmx, %input_se_cmx)
        -> !VPUIP.SparseBuffer<data=!InputDistributed, sparsity_map=!InputSMDistributed, storage_element_table=!InputSEDistributed>

    %input = VPUIP.NCEClusterTiling inputs(%input_sparse as %arg3: !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR>)
                               outputs(%input_sparse_cmx as %arg4: !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX>)
            -> !VPUIP.SparseBuffer<data=!InputDistributed, sparsity_map=!InputSMDistributed, storage_element_table=!InputSEDistributed> {
        %0 = VPUIP.Copy inputs(%arg3: !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR>)
                       outputs(%arg4: !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX>)
              -> !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX>
    }

    %cst_weights = const.Declare !Weights_DDR = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %cst_weights_sm = const.Declare !WeightsSM_DDR = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPUIP.GroupSparseBuffer (%cst_weights, %cst_weights_sm) {is_weights}
        -> !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=!WeightsSM_DDR, is_weights>

    %weights_data_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %weights_sm_cmx = VPURT.AllocDistributed -> !WeightsSMDistributed
    %weights_sparse_cmx = VPUIP.GroupSparseBuffer (%weights_data_cmx, %weights_sm_cmx) {is_weights}
        -> !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights>

    %weights = VPUIP.NCEClusterTiling inputs(%input_sparse as %arg3: !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=memref<!WeightsSM_DDR>, is_weights>)
                                 outputs(%input_sparse_cmx as %arg4: !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>)
            -> !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights> {
        %0 = VPUIP.Copy inputs(%arg3: !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=memref<!WeightsSM_DDR>, is_weights>)
                       outputs(%arg4: !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>)
              -> !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>
    }

    %cst_weights_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<16x1x1x4xsi32>
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed

    %weights_table = VPUIP.NCEClusterTiling inputs(%cst_weights_table as %arg3: !WeightsTable_DDR)
                                           outputs(%weights_table_cmx as %arg4: !WeightsTable_CMX)
            -> !WeightsTableDistributed {
        %0 = VPUIP.Copy inputs(%arg3: !WeightsTable_DDR) outputs(%arg4: !WeightsTable_CMX) -> !WeightsTable_CMX
    }

    %in_data, %in_sm, %in_se = VPUIP.UngroupSparseBuffer(%input) {resultSegmentSizes = array<i32: 1, 1, 1>}
        -> !InputDistributed, !InputSMDistributed, !InputSEDistributed
    %w_data, %w_sm = VPUIP.UngroupSparseBuffer(%weights)  {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> !WeightsDistributed, !WeightsSMDistributed
    %out_cmx = VPURT.AllocDistributed -> !OutputDistributed

    %conv_out = VPUIP.NCEClusterTiling inputs(%in_data as %arg3: !Input_CMX,
                                              %in_sm as %arg4: !InputSM_CMX,
                                              %in_se as %arg5: !InputSE_CMX,
                                              %w_data as %arg6: !Weights_CMX,
                                              %w_sm as %arg7: !WeightsSM_CMX,
                                              %weights_table as %arg8: !WeightsTable_CMX)
                                      outputs(%out_cmx as %arg9: !Output_CMX)
            -> !OutputDistributed {
        %0 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%arg3 : !Input_CMX)
            input_sparsity_map(%arg4 : !InputSM_CMX)
            input_storage_element_table(%arg5 : !InputSE_CMX)
            weights(%arg6 : !Weights_CMX)
            weights_sparsity_map(%arg7 : !WeightsSM_CMX)
            weight_table(%arg8 : !WeightsTable_CMX)
            parent_input(%arg3 : !Input_CMX)
            parent_input_sparsity_map(%arg4 : !InputSM_CMX)
            parent_input_storage_element_table(%arg5 : !InputSE_CMX)
            parent_output(%arg9 : !Output_CMX)
            outputs(%arg9 : !Output_CMX)
                -> !Output_CMX
            variants :  {
                DPUTask {cluster_id = 0 : i64, outStart = [0, 0, 0], outEnd = [16, 3, 3], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            } PPE :  {
            }
    }

    return %conv_out : !OutputDistributed

    // CHECK:       VPUIP.StorageElementTable {
    // CHECK-SAME:    basePtrs = dense<[0, 0, 0,
    // CHECK-SAME:                      0, 0, 0,
    // CHECK-SAME:                      1, 1, 1]> : tensor<9xi32>,
    // CHECK-SAME:    dataElemType = f16, dataShape = [1, 16, 3, 3],
    // CHECK-SAME:    seDepth = 1 : i64, seSize = 16 : i64
    // CHECK-SAME:  } -> memref<1x1x3x3xi32, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!InputSMDistributed = !VPUIP.DistributedBuffer<
    1x16x6x6xi1, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!InputSEDistributed = !VPUIP.DistributedBuffer<
    1x1x6x6xi32, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x6x6xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
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

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x16x3x3xf16, #NHWC>
!InputSM_DDR = memref<1x16x6x6xi1, #NHWC>
!InputSE_DDR = memref<1x1x6x6xi32, #NHWC>
!Weights_DDR = memref<16x16x1x1xf16, #NHWC>
!WeightsSM_DDR = memref<16x1x1x128xi1>
!WeightsTable_DDR = memref<16x1x1x4xsi32>

!Input_CMX = memref<1x16x3x3xf16, #NHWC, @CMX_NN>
!InputSM_CMX = memref<1x16x6x6xi1, #NHWC, @CMX_NN>
!InputSE_CMX = memref<1x1x6x6xi32, #NHWC, @CMX_NN>
!Weights_CMX = memref<16x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsSM_CMX = memref<16x1x1x128xi1, @CMX_NN>
!WeightsTable_CMX = memref<16x1x1x4xsi32, @CMX_NN>
!Output_CMX = memref<1x16x6x6xf16, #NHWC, @CMX_NN>

// CHECK-LABEL:  func.func @Interpolate
func.func @Interpolate(%input_data: !Input_DDR, %input_sm: !InputSM_DDR) -> !OutputDistributed {
    %input_se = VPUIP.StorageElementTable {
                dataShape = [1, 16, 3, 3], dataElemType = f16,
                seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                            nearest_mode = <FLOOR>,  offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>,
                seSize = 16, seDepth = 1
            } -> !InputSE_DDR
    %input_sparse = VPUIP.GroupSparseBuffer (%input_data, %input_sm, %input_se) {
            seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                        nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>
        } -> !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR,
                                 #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                                    nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>>

    %input_data_cmx = VPURT.AllocDistributed -> !InputDistributed
    %input_sm_cmx = VPURT.AllocDistributed -> !InputSMDistributed
    %input_se_cmx = VPURT.AllocDistributed -> !InputSEDistributed
    %input_sparse_cmx = VPUIP.GroupSparseBuffer (%input_data_cmx, %input_sm_cmx, %input_se_cmx) {
            seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                        nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>
        } -> !VPUIP.SparseBuffer<data=!InputDistributed, sparsity_map=!InputSMDistributed, storage_element_table=!InputSEDistributed,
                                 #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                                    nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>>

    %input = VPUIP.NCEClusterTiling inputs(%input_sparse as %arg3: !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR>)
                               outputs(%input_sparse_cmx as %arg4: !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX>)
            -> !VPUIP.SparseBuffer<data=!InputDistributed, sparsity_map=!InputSMDistributed, storage_element_table=!InputSEDistributed,
                                   #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                                      nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>> {
        %0 = VPUIP.Copy inputs(%arg3: !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR>)
                       outputs(%arg4: !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX>)
              -> !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX>
    }

    %cst_weights = const.Declare !Weights_DDR = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %cst_weights_sm = const.Declare !WeightsSM_DDR = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPUIP.GroupSparseBuffer (%cst_weights, %cst_weights_sm) {is_weights}
        -> !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=!WeightsSM_DDR, is_weights>

    %weights_data_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %weights_sm_cmx = VPURT.AllocDistributed -> !WeightsSMDistributed
    %weights_sparse_cmx = VPUIP.GroupSparseBuffer (%weights_data_cmx, %weights_sm_cmx) {is_weights}
        -> !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights>

    %weights = VPUIP.NCEClusterTiling inputs(%input_sparse as %arg3: !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=memref<!WeightsSM_DDR>, is_weights>)
                                 outputs(%input_sparse_cmx as %arg4: !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>)
            -> !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights> {
        %0 = VPUIP.Copy inputs(%arg3: !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=memref<!WeightsSM_DDR>, is_weights>)
                       outputs(%arg4: !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>)
              -> !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>
    }

    %cst_weights_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<16x1x1x4xsi32>
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed

    %weights_table = VPUIP.NCEClusterTiling inputs(%cst_weights_table as %arg3: !WeightsTable_DDR)
                                           outputs(%weights_table_cmx as %arg4: !WeightsTable_CMX)
            -> !WeightsTableDistributed {
        %0 = VPUIP.Copy inputs(%arg3: !WeightsTable_DDR) outputs(%arg4: !WeightsTable_CMX) -> !WeightsTable_CMX
    }

    %in_data, %in_sm, %in_se = VPUIP.UngroupSparseBuffer(%input) {resultSegmentSizes = array<i32: 1, 1, 1>}
        -> !InputDistributed, !InputSMDistributed, !InputSEDistributed
    %w_data, %w_sm = VPUIP.UngroupSparseBuffer(%weights)  {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> !WeightsDistributed, !WeightsSMDistributed
    %out_cmx = VPURT.AllocDistributed -> !OutputDistributed

    %conv_out = VPUIP.NCEClusterTiling inputs(%in_data as %arg3: !Input_CMX,
                                              %in_sm as %arg4: !InputSM_CMX,
                                              %in_se as %arg5: !InputSE_CMX,
                                              %w_data as %arg6: !Weights_CMX,
                                              %w_sm as %arg7: !WeightsSM_CMX,
                                              %weights_table as %arg8: !WeightsTable_CMX)
                                      outputs(%out_cmx as %arg9: !Output_CMX)
            -> !OutputDistributed {
        %0 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%arg3 : !Input_CMX)
            input_sparsity_map(%arg4 : !InputSM_CMX)
            input_storage_element_table(%arg5 : !InputSE_CMX)
            weights(%arg6 : !Weights_CMX)
            weights_sparsity_map(%arg7 : !WeightsSM_CMX)
            weight_table(%arg8 : !WeightsTable_CMX)
            parent_input(%arg3 : !Input_CMX)
            parent_input_sparsity_map(%arg4 : !InputSM_CMX)
            parent_input_storage_element_table(%arg5 : !InputSE_CMX)
            parent_output(%arg9 : !Output_CMX)
            outputs(%arg9 : !Output_CMX)
                -> !Output_CMX
            variants :  {
                DPUTask {cluster_id = 0 : i64, outStart = [0, 0, 0], outEnd = [16, 6, 6], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            } PPE :  {
            }
    }

    return %conv_out : !OutputDistributed

    // CHECK:       VPUIP.StorageElementTable {
    // CHECK-SAME:    basePtrs = dense<[0, 0, 0, 0, 0, 0,
    // CHECK-SAME:                      0, 0, 0, 0, 0, 0,
    // CHECK-SAME:                      0, 0, 0, 0, 0, 0,
    // CHECK-SAME:                      0, 0, 0, 0, 0, 0,
    // CHECK-SAME:                      1, 1, 1, 1, 1, 1,
    // CHECK-SAME:                      1, 1, 1, 1, 1, 1]> : tensor<36xi32>,
    // CHECK-SAME:    dataElemType = f16, dataShape = [1, 16, 3, 3],
    // CHECK-SAME:    seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                                nearest_mode = <FLOOR>,
    // CHECK-SAME:                                offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>,
    // CHECK-SAME:    seDepth = 1 : i64, seSize = 16 : i64
    // CHECK-SAME:  } -> memref<1x1x6x6xi32, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x32x3x3xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!InputSMDistributed = !VPUIP.DistributedBuffer<
    1x32x6x6xi1, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!InputSEDistributed = !VPUIP.DistributedBuffer<
    1x2x6x6xi32, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x32x6x6xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    32x32x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsSMDistributed = !VPUIP.DistributedBuffer<
    32x1x1x128xi1, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x32x3x3xf16, #NHWC>
!InputSM_DDR = memref<1x32x6x6xi1, #NHWC>
!InputSE_DDR = memref<1x2x6x6xi32, #NHWC>
!Weights_DDR = memref<32x32x1x1xf16, #NHWC>
!WeightsSM_DDR = memref<32x1x1x128xi1>
!WeightsTable_DDR = memref<32x1x1x4xsi32>

!Input_CMX = memref<1x32x3x3xf16, #NHWC, @CMX_NN>
!InputSM_CMX = memref<1x32x6x6xi1, #NHWC, @CMX_NN>
!InputSE_CMX = memref<1x2x6x6xi32, #NHWC, @CMX_NN>
!Weights_CMX = memref<32x32x1x1xf16, #NHWC, @CMX_NN>
!WeightsSM_CMX = memref<32x1x1x128xi1, @CMX_NN>
!WeightsTable_CMX = memref<32x1x1x4xsi32, @CMX_NN>
!Output_CMX = memref<1x32x6x6xf16, #NHWC, @CMX_NN>

// CHECK-LABEL:  func.func @InterpolateSESize
func.func @InterpolateSESize(%input_data: !Input_DDR, %input_sm: !InputSM_DDR) -> !OutputDistributed {
    %input_se = VPUIP.StorageElementTable {
                dataShape = [1, 32, 3, 3], dataElemType = f16,
                seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                            nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>,
                seSize = 16, seDepth = 2
            } -> !InputSE_DDR
    %input_sparse = VPUIP.GroupSparseBuffer (%input_data, %input_sm, %input_se) {
            seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                        nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>
        } -> !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR,
                                 #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                                    nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>>

    %input_data_cmx = VPURT.AllocDistributed -> !InputDistributed
    %input_sm_cmx = VPURT.AllocDistributed -> !InputSMDistributed
    %input_se_cmx = VPURT.AllocDistributed -> !InputSEDistributed
    %input_sparse_cmx = VPUIP.GroupSparseBuffer (%input_data_cmx, %input_sm_cmx, %input_se_cmx) {
            seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                        nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>
        } -> !VPUIP.SparseBuffer<data=!InputDistributed, sparsity_map=!InputSMDistributed, storage_element_table=!InputSEDistributed,
                                 #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                                    nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>>

    %input = VPUIP.NCEClusterTiling inputs(%input_sparse as %arg3: !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR,
                                            #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                                               nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>>)
                               outputs(%input_sparse_cmx as %arg4: !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX,
                                            #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                                               nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>>)
            -> !VPUIP.SparseBuffer<data=!InputDistributed, sparsity_map=!InputSMDistributed, storage_element_table=!InputSEDistributed,
                                            #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                                                nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>> {
        %0 = VPUIP.Copy inputs(%arg3: !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR,
                                            #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                                                nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>>)
                       outputs(%arg4: !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX,
                                            #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                                                nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>>)
              -> !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX,
                                            #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                                            nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>>
    }

    %cst_weights = const.Declare !Weights_DDR = dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %cst_weights_sm = const.Declare !WeightsSM_DDR = dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPUIP.GroupSparseBuffer (%cst_weights, %cst_weights_sm) {is_weights}
        -> !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=!WeightsSM_DDR, is_weights>

    %weights_data_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %weights_sm_cmx = VPURT.AllocDistributed -> !WeightsSMDistributed
    %weights_sparse_cmx = VPUIP.GroupSparseBuffer (%weights_data_cmx, %weights_sm_cmx) {is_weights}
        -> !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights>

    %weights = VPUIP.NCEClusterTiling inputs(%input_sparse as %arg3: !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=memref<!WeightsSM_DDR>, is_weights>)
                                 outputs(%input_sparse_cmx as %arg4: !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>)
            -> !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights> {
        %0 = VPUIP.Copy inputs(%arg3: !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=memref<!WeightsSM_DDR>, is_weights>)
                       outputs(%arg4: !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>)
              -> !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>
    }

    %cst_weights_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<32x1x1x4xsi32>
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed

    %weights_table = VPUIP.NCEClusterTiling inputs(%cst_weights_table as %arg3: !WeightsTable_DDR)
                                           outputs(%weights_table_cmx as %arg4: !WeightsTable_CMX)
            -> !WeightsTableDistributed {
        %0 = VPUIP.Copy inputs(%arg3: !WeightsTable_DDR) outputs(%arg4: !WeightsTable_CMX) -> !WeightsTable_CMX
    }

    %in_data, %in_sm, %in_se = VPUIP.UngroupSparseBuffer(%input) {resultSegmentSizes = array<i32: 1, 1, 1>}
        -> !InputDistributed, !InputSMDistributed, !InputSEDistributed
    %w_data, %w_sm = VPUIP.UngroupSparseBuffer(%weights)  {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> !WeightsDistributed, !WeightsSMDistributed
    %out_cmx = VPURT.AllocDistributed -> !OutputDistributed

    %conv_out = VPUIP.NCEClusterTiling inputs(%in_data as %arg3: !Input_CMX,
                                              %in_sm as %arg4: !InputSM_CMX,
                                              %in_se as %arg5: !InputSE_CMX,
                                              %w_data as %arg6: !Weights_CMX,
                                              %w_sm as %arg7: !WeightsSM_CMX,
                                              %weights_table as %arg8: !WeightsTable_CMX)
                                      outputs(%out_cmx as %arg9: !Output_CMX)
            -> !OutputDistributed {
        %0 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%arg3 : !Input_CMX)
            input_sparsity_map(%arg4 : !InputSM_CMX)
            input_storage_element_table(%arg5 : !InputSE_CMX)
            weights(%arg6 : !Weights_CMX)
            weights_sparsity_map(%arg7 : !WeightsSM_CMX)
            weight_table(%arg8 : !WeightsTable_CMX)
            parent_input(%arg3 : !Input_CMX)
            parent_input_sparsity_map(%arg4 : !InputSM_CMX)
            parent_input_storage_element_table(%arg5 : !InputSE_CMX)
            parent_output(%arg9 : !Output_CMX)
            outputs(%arg9 : !Output_CMX)
                -> !Output_CMX
            variants :  {
                DPUTask {cluster_id = 0 : i64, outStart = [0, 0, 0], outEnd = [16, 6, 6], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            } PPE :  {
            }
    }

    return %conv_out : !OutputDistributed

    // CHECK:       VPUIP.StorageElementTable {
    // CHECK-SAME:    basePtrs = dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-SAME:                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-SAME:                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-SAME:                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-SAME:                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // CHECK-SAME:                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]> : tensor<72xi32>,
    // CHECK-SAME:    dataElemType = f16, dataShape = [1, 32, 3, 3],
    // CHECK-SAME:    seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                                nearest_mode = <FLOOR>,
    // CHECK-SAME:                                offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>,
    // CHECK-SAME:    seDepth = 2 : i64, seSize = 16 : i64
    // CHECK-SAME:  } -> memref<1x2x6x6xi32, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!InputSMDistributed = !VPUIP.DistributedBuffer<
    1x16x4x6xi1, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!InputSEDistributed = !VPUIP.DistributedBuffer<
    1x1x4x6xi32, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x4x6xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
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

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x16x3x3xf16, #NHWC>
!InputSM_DDR = memref<1x16x4x6xi1, #NHWC>
!InputSE_DDR = memref<1x1x4x6xi32, #NHWC>
!Weights_DDR = memref<16x16x1x1xf16, #NHWC>
!WeightsSM_DDR = memref<16x1x1x128xi1>
!WeightsTable_DDR = memref<16x1x1x4xsi32>

!Input_CMX = memref<1x16x3x3xf16, #NHWC, @CMX_NN>
!InputSM_CMX = memref<1x16x4x6xi1, #NHWC, @CMX_NN>
!InputSE_CMX = memref<1x1x4x6xi32, #NHWC, @CMX_NN>
!Weights_CMX = memref<16x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsSM_CMX = memref<16x1x1x128xi1, @CMX_NN>
!WeightsTable_CMX = memref<16x1x1x4xsi32, @CMX_NN>
!Output_CMX = memref<1x16x4x6xf16, #NHWC, @CMX_NN>

// CHECK-LABEL:  func.func @InterpolateOutputOffsets
func.func @InterpolateOutputOffsets(%input_data: !Input_DDR, %input_sm: !InputSM_DDR) -> !OutputDistributed {
    %input_se = VPUIP.StorageElementTable {
                dataShape = [1, 16, 3, 3], dataElemType = f16,
                seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                            nearest_mode = <FLOOR>,  offsets = [0, 0, 1, 0], sizes = [1, 32, 4, 6]>,
                seSize = 16, seDepth = 1
            } -> !InputSE_DDR
    %input_sparse = VPUIP.GroupSparseBuffer (%input_data, %input_sm, %input_se) {
            seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                        nearest_mode = <FLOOR>, offsets = [0, 0, 1, 0], sizes = [1, 32, 4, 6]>
        } -> !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR,
                                 #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                                    nearest_mode = <FLOOR>, offsets = [0, 0, 1, 0], sizes = [1, 32, 4, 6]>>

    %input_data_cmx = VPURT.AllocDistributed -> !InputDistributed
    %input_sm_cmx = VPURT.AllocDistributed -> !InputSMDistributed
    %input_se_cmx = VPURT.AllocDistributed -> !InputSEDistributed
    %input_sparse_cmx = VPUIP.GroupSparseBuffer (%input_data_cmx, %input_sm_cmx, %input_se_cmx) {
            seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                        nearest_mode = <FLOOR>, offsets = [0, 0, 1, 0], sizes = [1, 32, 4, 6]>
        } -> !VPUIP.SparseBuffer<data=!InputDistributed, sparsity_map=!InputSMDistributed, storage_element_table=!InputSEDistributed,
                                 #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                                    nearest_mode = <FLOOR>, offsets = [0, 0, 1, 0], sizes = [1, 32, 4, 6]>>

    %input = VPUIP.NCEClusterTiling inputs(%input_sparse as %arg3: !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR>)
                               outputs(%input_sparse_cmx as %arg4: !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX>)
            -> !VPUIP.SparseBuffer<data=!InputDistributed, sparsity_map=!InputSMDistributed, storage_element_table=!InputSEDistributed,
                                   #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                                      nearest_mode = <FLOOR>, offsets = [0, 0, 1, 0], sizes = [1, 32, 4, 6]>> {
        %0 = VPUIP.Copy inputs(%arg3: !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR>)
                       outputs(%arg4: !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX>)
              -> !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX>
    }

    %cst_weights = const.Declare !Weights_DDR = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %cst_weights_sm = const.Declare !WeightsSM_DDR = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPUIP.GroupSparseBuffer (%cst_weights, %cst_weights_sm) {is_weights}
        -> !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=!WeightsSM_DDR, is_weights>

    %weights_data_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %weights_sm_cmx = VPURT.AllocDistributed -> !WeightsSMDistributed
    %weights_sparse_cmx = VPUIP.GroupSparseBuffer (%weights_data_cmx, %weights_sm_cmx) {is_weights}
        -> !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights>

    %weights = VPUIP.NCEClusterTiling inputs(%input_sparse as %arg3: !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=memref<!WeightsSM_DDR>, is_weights>)
                                 outputs(%input_sparse_cmx as %arg4: !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>)
            -> !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights> {
        %0 = VPUIP.Copy inputs(%arg3: !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=memref<!WeightsSM_DDR>, is_weights>)
                       outputs(%arg4: !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>)
              -> !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>
    }

    %cst_weights_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<16x1x1x4xsi32>
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed

    %weights_table = VPUIP.NCEClusterTiling inputs(%cst_weights_table as %arg3: !WeightsTable_DDR)
                                           outputs(%weights_table_cmx as %arg4: !WeightsTable_CMX)
            -> !WeightsTableDistributed {
        %0 = VPUIP.Copy inputs(%arg3: !WeightsTable_DDR) outputs(%arg4: !WeightsTable_CMX) -> !WeightsTable_CMX
    }

    %in_data, %in_sm, %in_se = VPUIP.UngroupSparseBuffer(%input) {resultSegmentSizes = array<i32: 1, 1, 1>}
        -> !InputDistributed, !InputSMDistributed, !InputSEDistributed
    %w_data, %w_sm = VPUIP.UngroupSparseBuffer(%weights)  {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> !WeightsDistributed, !WeightsSMDistributed
    %out_cmx = VPURT.AllocDistributed -> !OutputDistributed

    %conv_out = VPUIP.NCEClusterTiling inputs(%in_data as %arg3: !Input_CMX,
                                              %in_sm as %arg4: !InputSM_CMX,
                                              %in_se as %arg5: !InputSE_CMX,
                                              %w_data as %arg6: !Weights_CMX,
                                              %w_sm as %arg7: !WeightsSM_CMX,
                                              %weights_table as %arg8: !WeightsTable_CMX)
                                      outputs(%out_cmx as %arg9: !Output_CMX)
            -> !OutputDistributed {
        %0 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%arg3 : !Input_CMX)
            input_sparsity_map(%arg4 : !InputSM_CMX)
            input_storage_element_table(%arg5 : !InputSE_CMX)
            weights(%arg6 : !Weights_CMX)
            weights_sparsity_map(%arg7 : !WeightsSM_CMX)
            weight_table(%arg8 : !WeightsTable_CMX)
            parent_input(%arg3 : !Input_CMX)
            parent_input_sparsity_map(%arg4 : !InputSM_CMX)
            parent_input_storage_element_table(%arg5 : !InputSE_CMX)
            parent_output(%arg9 : !Output_CMX)
            outputs(%arg9 : !Output_CMX)
                -> !Output_CMX
            variants :  {
                DPUTask {cluster_id = 0 : i64, outStart = [0, 0, 0], outEnd = [16, 4, 6], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            } PPE :  {
            }
    }

    return %conv_out : !OutputDistributed

    // CHECK:       VPUIP.StorageElementTable {
    // CHECK-SAME:    basePtrs = dense<[0, 0, 0, 0, 0, 0,
    // CHECK-SAME:                      0, 0, 0, 0, 0, 0,
    // CHECK-SAME:                      0, 0, 0, 0, 0, 0,
    // CHECK-SAME:                      1, 1, 1, 1, 1, 1]> : tensor<24xi32>,
    // CHECK-SAME:    dataElemType = f16, dataShape = [1, 16, 3, 3],
    // CHECK-SAME:    seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                                nearest_mode = <FLOOR>,
    // CHECK-SAME:                                offsets = [0, 0, 1, 0], sizes = [1, 32, 4, 6]>,
    // CHECK-SAME:    seDepth = 1 : i64, seSize = 16 : i64
    // CHECK-SAME:  } -> memref<1x1x4x6xi32, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x4x6xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!InputSMDistributed = !VPUIP.DistributedBuffer<
    1x16x14x20xi1, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!InputSEDistributed = !VPUIP.DistributedBuffer<
    1x1x14x20xi32, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x9x18xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    16x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsSMDistributed = !VPUIP.DistributedBuffer<
    16x1x1x256xi1, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x16x6x16xf16, #NHWC, @DDR>
!Input_DDR_Tile = memref<1x16x4x6xf16, #NHWC, @DDR>
!InputSM_DDR_Tile = memref<1x16x11x20xi1, #NHWC, @DDR>
!InputSE_DDR_Tile = memref<1x1x11x20xi32, #NHWC, @DDR>
!Weights_DDR = memref<16x16x3x3xf16, #NHWC, @DDR>
!WeightsSM_DDR = memref<16x1x1x256xi1, @DDR>
!WeightsTable_DDR = memref<16x1x1x4xsi32, @DDR>
!Output_DDR = memref<1x16x18x18xf16, #NHWC, @DDR>

!Input_CMX_Tile = memref<1x16x4x6xf16, #NHWC, @CMX_NN>
!InputSM_CMX_Tile = memref<1x16x11x20xi1, #NHWC, @CMX_NN>
!InputSE_CMX_Tile = memref<1x1x11x20xi32, #NHWC, @CMX_NN>
!Weights_CMX = memref<16x16x3x3xf16, #NHWC, @CMX_NN>
!WeightsSM_CMX = memref<16x1x1x256xi1, @CMX_NN>
!WeightsTable_CMX = memref<16x1x1x4xsi32, @CMX_NN>
!Output_CMX_Tile = memref<1x16x9x18xf16, #NHWC, @CMX_NN>

// CHECK-LABEL:  func.func @InterpolateTileAndMultiCluster
func.func @InterpolateTileAndMultiCluster(%input_data: !Input_DDR, %input_sm_0: !InputSM_DDR_Tile, %input_sm_1: !InputSM_DDR_Tile) -> !Output_DDR {
    %input_se_0 = VPUIP.StorageElementTable {
                  dataShape = [1, 16, 4, 6], dataElemType = f16,
                  seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
                                              offsets = [0, 0, 0, 0], sizes = [1, 16, 11, 20], initial_input_shape = [1, 16, 6, 6], initial_output_shape = [1, 16, 18, 18]>,
                  seSize = 16, seDepth = 1
              } -> !InputSE_DDR_Tile
    %input_se_1 = VPUIP.StorageElementTable {
                  dataShape = [1, 16, 4, 6], dataElemType = f16,
                  seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
                                              offsets = [0, 0, 3, 0], sizes = [1, 16, 11, 20], initial_input_shape = [1, 16, 6, 6], initial_output_shape = [1, 16, 18, 18]>,
                  seSize = 16, seDepth = 1
              } -> !InputSE_DDR_Tile

    %input_sub_1 = VPUIP.SubView %input_data [0, 0, 2, 0] [1, 16, 4, 6]
                   : memref<1x16x6x16xf16, #NHWC, @DDR> to memref<1x16x4x6xf16, {order = #NHWC, strides = [1536, 1, 256, 16]}, @DDR>
    %input_alloc_1 = memref.alloc() : memref<1x16x4x6xf16, #NHWC, @DDR>
    %input_copy_1 = VPUIP.Copy inputs(%input_sub_1 : memref<1x16x4x6xf16, {order = #NHWC, strides = [1536, 1, 256, 16]}, @DDR>)
                               outputs(%input_alloc_1 : memref<1x16x4x6xf16, #NHWC, @DDR>) -> memref<1x16x4x6xf16, #NHWC, @DDR>
    %input_sparse_1 = VPUIP.GroupSparseBuffer (%input_copy_1, %input_sm_1, %input_se_1) {
                   seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
                                offsets = [0, 0, 3, 0], sizes = [1, 16, 11, 20], initial_input_shape = [1, 16, 6, 6], initial_output_shape = [1, 16, 18, 18]>
              } -> !VPUIP.SparseBuffer<data=!Input_DDR_Tile, sparsity_map=!InputSM_DDR_Tile, storage_element_table=!InputSE_DDR_Tile,
                            #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
                                offsets = [0, 0, 3, 0], sizes = [1, 16, 11, 20], initial_input_shape = [1, 16, 6, 6], initial_output_shape = [1, 16, 18, 18]>>

    %input_sub_0 = VPUIP.SubView %input_data [0, 0, 0, 0] [1, 16, 4, 6]
                   : memref<1x16x6x16xf16, #NHWC, @DDR> to memref<1x16x4x6xf16, {order = #NHWC, strides = [1536, 1, 256, 16]}, @DDR>
    %input_alloc_0 = memref.alloc() : memref<1x16x4x6xf16, #NHWC, @DDR>
    %input_copy_0 = VPUIP.Copy inputs(%input_sub_0 : memref<1x16x4x6xf16, {order = #NHWC, strides = [1536, 1, 256, 16]}, @DDR>)
                               outputs(%input_alloc_0 : memref<1x16x4x6xf16, #NHWC, @DDR>) -> memref<1x16x4x6xf16, #NHWC, @DDR>
    %input_sparse_0 = VPUIP.GroupSparseBuffer (%input_copy_0, %input_sm_0, %input_se_0) {
                   seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
                                offsets = [0, 0, 0, 0], sizes = [1, 16, 11, 20], initial_input_shape = [1, 16, 6, 6], initial_output_shape = [1, 16, 18, 18]>
              } -> !VPUIP.SparseBuffer<data=!Input_DDR_Tile, sparsity_map=!InputSM_DDR_Tile, storage_element_table=!InputSE_DDR_Tile,
                            #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
                                offsets = [0, 0, 0, 0], sizes = [1, 16, 11, 20], initial_input_shape = [1, 16, 6, 6], initial_output_shape = [1, 16, 18, 18]>>

    %input_data_cmx_0 = VPURT.AllocDistributed -> !InputDistributed
    %input_sm_cmx_0 = VPURT.AllocDistributed -> !InputSMDistributed
    %input_se_cmx_0 = VPURT.AllocDistributed -> !InputSEDistributed
    %input_sparse_cmx_0 = VPUIP.GroupSparseBuffer (%input_data_cmx_0, %input_sm_cmx_0, %input_se_cmx_0) {
                   seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
                                offsets = [0, 0, 0, 0], sizes = [1, 16, 11, 20], initial_input_shape = [1, 16, 6, 6], initial_output_shape = [1, 16, 18, 18]>
        } -> !VPUIP.SparseBuffer<data=!InputDistributed, sparsity_map=!InputSMDistributed, storage_element_table=!InputSEDistributed,
                            #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
                                offsets = [0, 0, 0, 0], sizes = [1, 16, 11, 20], initial_input_shape = [1, 16, 6, 6], initial_output_shape = [1, 16, 18, 18]>>

    %input_0 = VPUIP.NCEClusterTiling inputs(%input_sparse_0 as %arg3: !VPUIP.SparseBuffer<data=!Input_DDR_Tile, sparsity_map=!InputSM_DDR_Tile, storage_element_table=!InputSE_DDR_Tile>)
                                      outputs(%input_sparse_cmx_0 as %arg4: !VPUIP.SparseBuffer<data=!Input_CMX_Tile, sparsity_map=!InputSM_CMX_Tile, storage_element_table=!InputSE_CMX_Tile>)
              -> !VPUIP.SparseBuffer<data=!InputDistributed, sparsity_map=!InputSMDistributed, storage_element_table=!InputSEDistributed,
                            #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
                                offsets = [0, 0, 0, 0], sizes = [1, 16, 11, 20], initial_input_shape = [1, 16, 6, 6], initial_output_shape = [1, 16, 18, 18]>> {
        %0 = VPUIP.Copy inputs(%arg3: !VPUIP.SparseBuffer<data=!Input_DDR_Tile, sparsity_map=!InputSM_DDR_Tile, storage_element_table=!InputSE_DDR_Tile>)
                        outputs(%arg4: !VPUIP.SparseBuffer<data=!Input_CMX_Tile, sparsity_map=!InputSM_CMX_Tile, storage_element_table=!InputSE_CMX_Tile>)
              -> !VPUIP.SparseBuffer<data=!Input_CMX_Tile, sparsity_map=!InputSM_CMX_Tile, storage_element_table=!InputSE_CMX_Tile>
    }

    %cst_weights_0 = const.Declare !Weights_DDR = dense<1.0> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %cst_weights_sm_0 = const.Declare !WeightsSM_DDR = dense<1.0> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse_0 = VPUIP.GroupSparseBuffer (%cst_weights_0, %cst_weights_sm_0) {is_weights}
              -> !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=!WeightsSM_DDR, is_weights>

    %weights_data_cmx_0 = VPURT.AllocDistributed -> !WeightsDistributed
    %weights_sm_cmx_0 = VPURT.AllocDistributed -> !WeightsSMDistributed
    %weights_sparse_cmx_0 = VPUIP.GroupSparseBuffer (%weights_data_cmx_0, %weights_sm_cmx_0) {is_weights}
              -> !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights>

    %weights_0 = VPUIP.NCEClusterTiling inputs(%weights_sparse_0 as %arg3: !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=memref<!WeightsSM_DDR>, is_weights>)
                                      outputs(%weights_sparse_cmx_0 as %arg4: !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>)
            -> !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights> {
        %0 = VPUIP.Copy inputs(%arg3: !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=memref<!WeightsSM_DDR>, is_weights>)
                       outputs(%arg4: !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>)
              -> !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>
    }

    %cst_weights_table_0 = const.Declare !WeightsTable_DDR = dense<1> : tensor<16x1x1x4xsi32>
    %weights_table_cmx_0 = VPURT.AllocDistributed -> !WeightsTableDistributed
    %weights_table_0 = VPUIP.NCEClusterTiling inputs(%cst_weights_table_0 as %arg3: !WeightsTable_DDR)
                                           outputs(%weights_table_cmx_0 as %arg4: !WeightsTable_CMX)
            -> !WeightsTableDistributed {
        %0 = VPUIP.Copy inputs(%arg3: !WeightsTable_DDR) outputs(%arg4: !WeightsTable_CMX) -> !WeightsTable_CMX
    }

    %in_data_0, %in_sm_0, %in_se_0 = VPUIP.UngroupSparseBuffer(%input_0) {resultSegmentSizes = array<i32: 1, 1, 1>}
        -> !InputDistributed, !InputSMDistributed, !InputSEDistributed
    %w_data_0, %w_sm_0 = VPUIP.UngroupSparseBuffer(%weights_0)  {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> !WeightsDistributed, !WeightsSMDistributed
    %out_cmx_0 = VPURT.AllocDistributed -> !OutputDistributed

    %conv_out_0 = VPUIP.NCEClusterTiling inputs(%in_data_0 as %arg3: !Input_CMX_Tile,
                                                %in_sm_0 as %arg4: !InputSM_CMX_Tile,
                                                %in_se_0 as %arg5: !InputSE_CMX_Tile,
                                                %w_data_0 as %arg6: !Weights_CMX,
                                                %w_sm_0 as %arg7: !WeightsSM_CMX,
                                                %weights_table_0 as %arg8: !WeightsTable_CMX)
                                      outputs(%out_cmx_0 as %arg9: !Output_CMX_Tile)
            -> !OutputDistributed {
        %0 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [3, 3],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%arg3 : !Input_CMX_Tile)
            input_sparsity_map(%arg4 : !InputSM_CMX_Tile)
            input_storage_element_table(%arg5 : !InputSE_CMX_Tile)
            weights(%arg6 : !Weights_CMX)
            weights_sparsity_map(%arg7 : !WeightsSM_CMX)
            weight_table(%arg8 : !WeightsTable_CMX)
            parent_input(%arg3 : !Input_CMX_Tile)
            parent_input_sparsity_map(%arg4 : !InputSM_CMX_Tile)
            parent_input_storage_element_table(%arg5 : !InputSE_CMX_Tile)
            parent_output(%arg9 : !Output_CMX_Tile)
            outputs(%arg9 : !Output_CMX_Tile)
                -> !Output_CMX_Tile
            variants :  {
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [17, 4, 15], outStart = [0, 0, 0],
                         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [17, 8, 15], outStart = [0, 5, 0],
                         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            } PPE :  {
            }
    }

    %output_alloc = memref.alloc() : !Output_DDR
    %output_0 = VPUIP.SubView %output_alloc [0, 0, 0, 0] [1, 16, 9, 18] : !Output_DDR to memref<1x16x9x18xf16, {order = #NHWC, strides = [5184, 1, 288, 16]}, @DDR>
    %copy_output_0 = VPUIP.NCEClusterTiling inputs(%conv_out_0 as %arg2: memref<1x16x9x18xf16, #NHWC, @CMX_NN>) outputs(%output_0 as %arg3: memref<1x16x9x18xf16, #NHWC>) -> memref<1x16x9x18xf16, {order = #NHWC, strides = [5184, 1, 288, 16]}, @DDR> {
        %0 = VPUIP.Copy inputs(%arg2 : memref<1x16x9x18xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x16x9x18xf16, #NHWC>) -> memref<1x16x9x18xf16, #NHWC>
    }

    %input_data_cmx_1 = VPURT.AllocDistributed -> !InputDistributed
    %input_sm_cmx_1 = VPURT.AllocDistributed -> !InputSMDistributed
    %input_se_cmx_1 = VPURT.AllocDistributed -> !InputSEDistributed
    %input_sparse_cmx_1 = VPUIP.GroupSparseBuffer (%input_data_cmx_1, %input_sm_cmx_1, %input_se_cmx_1) {
                   seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
                                offsets = [0, 0, 3, 0], sizes = [1, 16, 11, 20], initial_input_shape = [1, 16, 6, 6], initial_output_shape = [1, 16, 18, 18]>
        } -> !VPUIP.SparseBuffer<data=!InputDistributed, sparsity_map=!InputSMDistributed, storage_element_table=!InputSEDistributed,
                            #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
                                offsets = [0, 0, 3, 0], sizes = [1, 16, 11, 20], initial_input_shape = [1, 16, 6, 6], initial_output_shape = [1, 16, 18, 18]>>

    %input_1 = VPUIP.NCEClusterTiling inputs(%input_sparse_1 as %arg3: !VPUIP.SparseBuffer<data=!Input_DDR_Tile, sparsity_map=!InputSM_DDR_Tile, storage_element_table=!InputSE_DDR_Tile>)
                                      outputs(%input_sparse_cmx_1 as %arg4: !VPUIP.SparseBuffer<data=!Input_CMX_Tile, sparsity_map=!InputSM_CMX_Tile, storage_element_table=!InputSE_CMX_Tile>)
              -> !VPUIP.SparseBuffer<data=!InputDistributed, sparsity_map=!InputSMDistributed, storage_element_table=!InputSEDistributed,
                            #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
                                offsets = [0, 0, 3, 0], sizes = [1, 16, 11, 20], initial_input_shape = [1, 16, 6, 6], initial_output_shape = [1, 16, 18, 18]>> {
        %0 = VPUIP.Copy inputs(%arg3: !VPUIP.SparseBuffer<data=!Input_DDR_Tile, sparsity_map=!InputSM_DDR_Tile, storage_element_table=!InputSE_DDR_Tile>)
                        outputs(%arg4: !VPUIP.SparseBuffer<data=!Input_CMX_Tile, sparsity_map=!InputSM_CMX_Tile, storage_element_table=!InputSE_CMX_Tile>)
              -> !VPUIP.SparseBuffer<data=!Input_CMX_Tile, sparsity_map=!InputSM_CMX_Tile, storage_element_table=!InputSE_CMX_Tile>
    }

    %cst_weights_1 = const.Declare !Weights_DDR = dense<1.0> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %cst_weights_sm_1 = const.Declare !WeightsSM_DDR = dense<1.0> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse_1 = VPUIP.GroupSparseBuffer (%cst_weights_1, %cst_weights_sm_1) {is_weights}
              -> !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=!WeightsSM_DDR, is_weights>

    %weights_data_cmx_1 = VPURT.AllocDistributed -> !WeightsDistributed
    %weights_sm_cmx_1 = VPURT.AllocDistributed -> !WeightsSMDistributed
    %weights_sparse_cmx_1 = VPUIP.GroupSparseBuffer (%weights_data_cmx_1, %weights_sm_cmx_1) {is_weights}
              -> !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights>

    %weights_1 = VPUIP.NCEClusterTiling inputs(%weights_sparse_1 as %arg3: !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=memref<!WeightsSM_DDR>, is_weights>)
                                      outputs(%weights_sparse_cmx_1 as %arg4: !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>)
            -> !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights> {
        %0 = VPUIP.Copy inputs(%arg3: !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=memref<!WeightsSM_DDR>, is_weights>)
                       outputs(%arg4: !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>)
              -> !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>
    }

    %cst_weights_table_1 = const.Declare !WeightsTable_DDR = dense<1> : tensor<16x1x1x4xsi32>
    %weights_table_cmx_1 = VPURT.AllocDistributed -> !WeightsTableDistributed
    %weights_table_1 = VPUIP.NCEClusterTiling inputs(%cst_weights_table_1 as %arg3: !WeightsTable_DDR)
                                           outputs(%weights_table_cmx_1 as %arg4: !WeightsTable_CMX)
            -> !WeightsTableDistributed {
        %0 = VPUIP.Copy inputs(%arg3: !WeightsTable_DDR) outputs(%arg4: !WeightsTable_CMX) -> !WeightsTable_CMX
    }

    %in_data_1, %in_sm_1, %in_se_1 = VPUIP.UngroupSparseBuffer(%input_1) {resultSegmentSizes = array<i32: 1, 1, 1>}
        -> !InputDistributed, !InputSMDistributed, !InputSEDistributed
    %w_data_1, %w_sm_1 = VPUIP.UngroupSparseBuffer(%weights_1)  {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> !WeightsDistributed, !WeightsSMDistributed
    %out_cmx_1 = VPURT.AllocDistributed -> !OutputDistributed

    %conv_out_1 = VPUIP.NCEClusterTiling inputs(%in_data_1 as %arg3: !Input_CMX_Tile,
                                                %in_sm_1 as %arg4: !InputSM_CMX_Tile,
                                                %in_se_1 as %arg5: !InputSE_CMX_Tile,
                                                %w_data_1 as %arg6: !Weights_CMX,
                                                %w_sm_1 as %arg7: !WeightsSM_CMX,
                                                %weights_table_1 as %arg8: !WeightsTable_CMX)
                                      outputs(%out_cmx_1 as %arg9: !Output_CMX_Tile)
            -> !OutputDistributed {
        %0 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [3, 3],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%arg3 : !Input_CMX_Tile)
            input_sparsity_map(%arg4 : !InputSM_CMX_Tile)
            input_storage_element_table(%arg5 : !InputSE_CMX_Tile)
            weights(%arg6 : !Weights_CMX)
            weights_sparsity_map(%arg7 : !WeightsSM_CMX)
            weight_table(%arg8 : !WeightsTable_CMX)
            parent_input(%arg3 : !Input_CMX_Tile)
            parent_input_sparsity_map(%arg4 : !InputSM_CMX_Tile)
            parent_input_storage_element_table(%arg5 : !InputSE_CMX_Tile)
            parent_output(%arg9 : !Output_CMX_Tile)
            outputs(%arg9 : !Output_CMX_Tile)
                -> !Output_CMX_Tile
            variants :  {
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [17, 4, 15], outStart = [0, 0, 0],
                         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [17, 8, 15], outStart = [0, 5, 0],
                         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            } PPE :  {
            }
    }

    %output_1 = VPUIP.SubView %output_alloc [0, 0, 3, 0] [1, 16, 9, 18] : !Output_DDR to memref<1x16x9x18xf16, {order = #NHWC, strides = [5184, 1, 288, 16]}, @DDR>
    %copy_output_1 = VPUIP.NCEClusterTiling inputs(%conv_out_1 as %arg2: memref<1x16x9x18xf16, #NHWC, @CMX_NN>) outputs(%output_1 as %arg3: memref<1x16x9x18xf16, #NHWC>) -> memref<1x16x9x18xf16, {order = #NHWC, strides = [5184, 1, 288, 16]}, @DDR> {
        %0 = VPUIP.Copy inputs(%arg2 : memref<1x16x9x18xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x16x9x18xf16, #NHWC>) -> memref<1x16x9x18xf16, #NHWC>
    }

    return %output_alloc : !Output_DDR

    // CHECK:       VPUIP.StorageElementTable {
    // CHECK-SAME:    basePtrs = dense<[
    // CHECK-SAME:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-SAME:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-SAME:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-SAME:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-SAME:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-SAME:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-SAME:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-SAME:      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // CHECK-SAME:      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // CHECK-SAME:      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // CHECK-SAME:      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    // CHECK-SAME:      ]> : tensor<220xi32>,
    // CHECK-SAME:    dataElemType = f16, dataShape = [1, 16, 4, 6],
    // CHECK-SAME:    seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                                scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
    // CHECK-SAME:                                offsets = [0, 0, 0, 0], sizes = [1, 16, 11, 20], initial_input_shape = [1, 16, 6, 6], initial_output_shape = [1, 16, 18, 18]>,
    // CHECK-SAME:    seDepth = 1 : i64, seSize = 16 : i64
    // CHECK-SAME:  } -> memref<1x1x11x20xi32, #NHWC, @DDR>

    // CHECK:       VPUIP.StorageElementTable {
    // CHECK-SAME:    basePtrs = dense<[
    // CHECK-SAME:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-SAME:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-SAME:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-SAME:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-SAME:      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // CHECK-SAME:      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // CHECK-SAME:      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // CHECK-SAME:      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // CHECK-SAME:      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // CHECK-SAME:      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // CHECK-SAME:      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    // CHECK-SAME:      ]> : tensor<220xi32>,
    // CHECK-SAME:    dataElemType = f16, dataShape = [1, 16, 4, 6],
    // CHECK-SAME:    seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                                scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
    // CHECK-SAME:                                offsets = [0, 0, 3, 0], sizes = [1, 16, 11, 20], initial_input_shape = [1, 16, 6, 6], initial_output_shape = [1, 16, 18, 18]>,
    // CHECK-SAME:    seDepth = 1 : i64, seSize = 16 : i64
    // CHECK-SAME:  } -> memref<1x1x11x20xi32, #NHWC, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x2x2xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!InputSMDistributed = !VPUIP.DistributedBuffer<
    1x16x5x5xi1, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!InputSEDistributed = !VPUIP.DistributedBuffer<
    1x1x5x5xi32, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
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

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x16x2x2xf16, #NHWC>
!InputSM_DDR = memref<1x16x5x5xi1, #NHWC>
!InputSE_DDR = memref<1x1x5x5xi32, #NHWC>
!Weights_DDR = memref<16x16x3x3xf16, #NHWC>
!WeightsSM_DDR = memref<16x1x1x256xi1>
!WeightsTable_DDR = memref<16x1x1x4xsi32>

!Input_CMX = memref<1x16x2x2xf16, #NHWC, @CMX_NN>
!InputSM_CMX = memref<1x16x5x5xi1, #NHWC, @CMX_NN>
!InputSE_CMX = memref<1x1x5x5xi32, #NHWC, @CMX_NN>
!Weights_CMX = memref<16x16x3x3xf16, #NHWC, @CMX_NN>
!WeightsSM_CMX = memref<16x1x1x256xi1, @CMX_NN>
!WeightsTable_CMX = memref<16x1x1x4xsi32, @CMX_NN>
!Output_CMX = memref<1x16x3x3xf16, #NHWC, @CMX_NN>

// CHECK-LABEL:  func.func @InterpolateBilinearAlignCornersOutputOffsets
func.func @InterpolateBilinearAlignCornersOutputOffsets(%input_data: !Input_DDR, %input_sm: !InputSM_DDR) -> !OutputDistributed {
    %input_se = VPUIP.StorageElementTable {
                dataShape = [1, 16, 2, 2], dataElemType = f16,
                seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ALIGN_CORNERS>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
                                            offsets = [0, 0, 1, 1], sizes = [1, 16, 5, 5], initial_input_shape = [1, 16, 3, 3], initial_output_shape = [1, 16, 7, 7]>,
                seSize = 16, seDepth = 1
            } -> !InputSE_DDR
    %input_sparse = VPUIP.GroupSparseBuffer (%input_data, %input_sm, %input_se) {
            seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ALIGN_CORNERS>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
                                        offsets = [0, 0, 1, 1], sizes = [1, 16, 5, 5], initial_input_shape = [1, 16, 3, 3], initial_output_shape = [1, 16, 7, 7]>
        } -> !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR,
                     #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ALIGN_CORNERS>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
                                        offsets = [0, 0, 1, 1], sizes = [1, 16, 5, 5], initial_input_shape = [1, 16, 3, 3], initial_output_shape = [1, 16, 7, 7]>>

    %input_data_cmx = VPURT.AllocDistributed -> !InputDistributed
    %input_sm_cmx = VPURT.AllocDistributed -> !InputSMDistributed
    %input_se_cmx = VPURT.AllocDistributed -> !InputSEDistributed
    %input_sparse_cmx = VPUIP.GroupSparseBuffer (%input_data_cmx, %input_sm_cmx, %input_se_cmx) {
            seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ALIGN_CORNERS>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
                                        offsets = [0, 0, 1, 1], sizes = [1, 16, 5, 5], initial_input_shape = [1, 16, 3, 3], initial_output_shape = [1, 16, 7, 7]>
        } -> !VPUIP.SparseBuffer<data=!InputDistributed, sparsity_map=!InputSMDistributed, storage_element_table=!InputSEDistributed,
                                 #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ALIGN_CORNERS>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
                                        offsets = [0, 0, 1, 1], sizes = [1, 16, 5, 5], initial_input_shape = [1, 16, 3, 3], initial_output_shape = [1, 16, 7, 7]>>

    %input = VPUIP.NCEClusterTiling inputs(%input_sparse as %arg3: !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR>)
                               outputs(%input_sparse_cmx as %arg4: !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX>)
            -> !VPUIP.SparseBuffer<data=!InputDistributed, sparsity_map=!InputSMDistributed, storage_element_table=!InputSEDistributed,
                                   #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ALIGN_CORNERS>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
                                        offsets = [0, 0, 1, 1], sizes = [1, 16, 5, 5], initial_input_shape = [1, 16, 3, 3], initial_output_shape = [1, 16, 7, 7]>> {
        %0 = VPUIP.Copy inputs(%arg3: !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR>)
                       outputs(%arg4: !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX>)
              -> !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX>
    }

    %cst_weights = const.Declare !Weights_DDR = dense<1.0> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %cst_weights_sm = const.Declare !WeightsSM_DDR = dense<1.0> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPUIP.GroupSparseBuffer (%cst_weights, %cst_weights_sm) {is_weights}
        -> !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=!WeightsSM_DDR, is_weights>

    %weights_data_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %weights_sm_cmx = VPURT.AllocDistributed -> !WeightsSMDistributed
    %weights_sparse_cmx = VPUIP.GroupSparseBuffer (%weights_data_cmx, %weights_sm_cmx) {is_weights}
        -> !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights>

    %weights = VPUIP.NCEClusterTiling inputs(%input_sparse as %arg3: !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=memref<!WeightsSM_DDR>, is_weights>)
                                 outputs(%input_sparse_cmx as %arg4: !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>)
            -> !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights> {
        %0 = VPUIP.Copy inputs(%arg3: !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=memref<!WeightsSM_DDR>, is_weights>)
                       outputs(%arg4: !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>)
              -> !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>
    }

    %cst_weights_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<16x1x1x4xsi32>
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed

    %weights_table = VPUIP.NCEClusterTiling inputs(%cst_weights_table as %arg3: !WeightsTable_DDR)
                                           outputs(%weights_table_cmx as %arg4: !WeightsTable_CMX)
            -> !WeightsTableDistributed {
        %0 = VPUIP.Copy inputs(%arg3: !WeightsTable_DDR) outputs(%arg4: !WeightsTable_CMX) -> !WeightsTable_CMX
    }

    %in_data, %in_sm, %in_se = VPUIP.UngroupSparseBuffer(%input) {resultSegmentSizes = array<i32: 1, 1, 1>}
        -> !InputDistributed, !InputSMDistributed, !InputSEDistributed
    %w_data, %w_sm = VPUIP.UngroupSparseBuffer(%weights)  {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> !WeightsDistributed, !WeightsSMDistributed
    %out_cmx = VPURT.AllocDistributed -> !OutputDistributed

    %conv_out = VPUIP.NCEClusterTiling inputs(%in_data as %arg3: !Input_CMX,
                                              %in_sm as %arg4: !InputSM_CMX,
                                              %in_se as %arg5: !InputSE_CMX,
                                              %w_data as %arg6: !Weights_CMX,
                                              %w_sm as %arg7: !WeightsSM_CMX,
                                              %weights_table as %arg8: !WeightsTable_CMX)
                                      outputs(%out_cmx as %arg9: !Output_CMX)
            -> !OutputDistributed {
        %0 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [3, 3],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%arg3 : !Input_CMX)
            input_sparsity_map(%arg4 : !InputSM_CMX)
            input_storage_element_table(%arg5 : !InputSE_CMX)
            weights(%arg6 : !Weights_CMX)
            weights_sparsity_map(%arg7 : !WeightsSM_CMX)
            weight_table(%arg8 : !WeightsTable_CMX)
            parent_input(%arg3 : !Input_CMX)
            parent_input_sparsity_map(%arg4 : !InputSM_CMX)
            parent_input_storage_element_table(%arg5 : !InputSE_CMX)
            parent_output(%arg9 : !Output_CMX)
            outputs(%arg9 : !Output_CMX)
                -> !Output_CMX
            variants :  {
                DPUTask {cluster_id = 0 : i64, outStart = [0, 0, 0], outEnd = [16, 4, 6], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            } PPE :  {
            }
    }

    return %conv_out : !OutputDistributed

    // CHECK:       VPUIP.StorageElementTable {
    // CHECK-SAME:    basePtrs = dense<[0, 0, 0, 0, 0,
    // CHECK-SAME:                      0, 0, 0, 0, 0,
    // CHECK-SAME:                      1, 1, 1, 1, 1,
    // CHECK-SAME:                      1, 1, 1, 1, 1,
    // CHECK-SAME:                      1, 1, 1, 1, 1]> : tensor<25xi32>,
    // CHECK-SAME:    dataElemType = f16, dataShape = [1, 16, 2, 2],
    // CHECK-SAME:    seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ALIGN_CORNERS>,
    // CHECK-SAME:                                scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
    // CHECK-SAME:                                offsets = [0, 0, 1, 1], sizes = [1, 16, 5, 5],
    // CHECK-SAME:                                initial_input_shape = [1, 16, 3, 3], initial_output_shape = [1, 16, 7, 7]>,
    // CHECK-SAME:    seDepth = 1 : i64, seSize = 16 : i64
    // CHECK-SAME:  } -> memref<1x1x5x5xi32, #NHWC>
}
