//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --run-ungroup-buffer-section-rewriters="rewriter=ungroup-sparse-buffer" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK:       func.func @SparseConv([[ARG0:%.+]]: memref<1x16x64x64xf16, {order = #NHWC}>, [[ARG1:%.+]]: memref<1x16x64x64xi1, {order = #NHWC}>,
// CHECK-SAME:                    [[ARG2:%.+]]: memref<1x32x64x64xf16, {order = #NHWC}>, [[ARG3:%.+]]: memref<1x32x64x64xi1, {order = #NHWC}>)
// CHECK-SAME:      -> (memref<1x32x64x64xf16, {order = #NHWC}>, memref<1x32x64x64xi1, {order = #NHWC}>)
func.func @SparseConv(%arg0: memref<1x16x64x64xf16, {order = #NHWC}>, %arg1: memref<1x16x64x64xi1, {order = #NHWC}>,
                 %arg2: memref<1x32x64x64xf16, {order = #NHWC}>, %arg3: memref<1x32x64x64xi1, {order = #NHWC}>)
        -> (memref<1x32x64x64xf16, {order = #NHWC}>, memref<1x32x64x64xi1, {order = #NHWC}>) {
    %input_sparse = VPUIP.GroupSparseBuffer (%arg0, %arg1)
        -> !VPUIP.SparseBuffer<data=memref<1x16x64x64xf16, {order = #NHWC}>, sparsity_map=memref<1x16x64x64xi1, {order = #NHWC}>>

    %input_data_cmx = memref.alloc() : memref<1x16x64x64xf16, {order = #NHWC}, @CMX_NN>
    %input_sm_cmx = memref.alloc() : memref<1x16x64x64xi1, {order = #NHWC}, @CMX_NN>
    %input_sparse_cmx = VPUIP.GroupSparseBuffer (%input_data_cmx, %input_sm_cmx)
        -> !VPUIP.SparseBuffer<data=memref<1x16x64x64xf16, {order = #NHWC}, @CMX_NN>, sparsity_map=memref<1x16x64x64xi1, {order = #NHWC}, @CMX_NN>>

    %input = VPUIP.Copy inputs(%input_sparse : !VPUIP.SparseBuffer<data=memref<1x16x64x64xf16, {order = #NHWC}>, sparsity_map=memref<1x16x64x64xi1, {order = #NHWC}>>)
                        outputs(%input_sparse_cmx : !VPUIP.SparseBuffer<data=memref<1x16x64x64xf16, {order = #NHWC}, @CMX_NN>, sparsity_map=memref<1x16x64x64xi1, {order = #NHWC}, @CMX_NN>>)
        -> !VPUIP.SparseBuffer<data=memref<1x16x64x64xf16, {order = #NHWC}, @CMX_NN>, sparsity_map=memref<1x16x64x64xi1, {order = #NHWC}, @CMX_NN>>

    %cst_weights = const.Declare memref<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %cst_weights_sm = const.Declare memref<32x1x1x256xi1> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPUIP.GroupSparseBuffer (%cst_weights, %cst_weights_sm) <{is_weights}>
        -> !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16, {order = #NHWC}>, sparsity_map=memref<32x1x1x256xi1>, is_weights>

    %weights_data_cmx = memref.alloc() : memref<32x16x3x3xf16, {order = #NHWC}, @CMX_NN>
    %weights_sm_cmx = memref.alloc() : memref<32x1x1x256xi1, @CMX_NN>
    %weights_sparse_cmx = VPUIP.GroupSparseBuffer (%weights_data_cmx, %weights_sm_cmx) <{is_weights}>
        -> !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16, {order = #NHWC}, @CMX_NN>, sparsity_map=memref<32x1x1x256xi1, @CMX_NN>, is_weights>

    %weights = VPUIP.Copy inputs(%weights_sparse : !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16, {order = #NHWC}>, sparsity_map=memref<32x1x1x256xi1>, is_weights>)
                          outputs(%weights_sparse_cmx : !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16, {order = #NHWC}, @CMX_NN>, sparsity_map=memref<32x1x1x256xi1, @CMX_NN>, is_weights>)
        -> !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16, {order = #NHWC}, @CMX_NN>, sparsity_map=memref<32x1x1x256xi1, @CMX_NN>, is_weights>

    %in_data, %in_sm = VPUIP.UngroupSparseBuffer(%input) {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> memref<1x16x64x64xf16, {order = #NHWC}, @CMX_NN>, memref<1x16x64x64xi1, {order = #NHWC}, @CMX_NN>
    %w_data, %w_sm = VPUIP.UngroupSparseBuffer(%weights)  {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> memref<32x16x3x3xf16, {order = #NHWC}, @CMX_NN>, memref<32x1x1x256xi1, @CMX_NN>
    %out_data = memref.alloc() : memref<1x32x64x64xf16, {order = #NHWC}, @CMX_NN>
    %out_sm = memref.alloc() : memref<1x32x64x64xi1, {order = #NHWC}, @CMX_NN>

    %conv_out:2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>} <{
            kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            kernel_size = [3, 3],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }>
        input(%in_data : memref<1x16x64x64xf16, {order = #NHWC}, @CMX_NN>)
        input_sparsity_map(%in_sm : memref<1x16x64x64xi1, {order = #NHWC}, @CMX_NN>)
        weights(%w_data : memref<32x16x3x3xf16, {order = #NHWC}, @CMX_NN>)
        weights_sparsity_map(%w_sm : memref<32x1x1x256xi1, @CMX_NN>)
        parent_input(%in_data : memref<1x16x64x64xf16, {order = #NHWC}, @CMX_NN>)
        parent_input_sparsity_map(%in_sm : memref<1x16x64x64xi1, {order = #NHWC}, @CMX_NN>)
        parent_output(%out_data : memref<1x32x64x64xf16, {order = #NHWC}, @CMX_NN>)
        parent_output_sparsity_map(%out_sm : memref<1x32x64x64xi1, {order = #NHWC}, @CMX_NN>)
        outputs(%out_data : memref<1x32x64x64xf16, {order = #NHWC}, @CMX_NN>)
        output_sparsity_map(%out_sm : memref<1x32x64x64xi1, {order = #NHWC}, @CMX_NN>)
            -> memref<1x32x64x64xf16, {order = #NHWC}, @CMX_NN>, memref<1x32x64x64xi1, {order = #NHWC}, @CMX_NN>
        variants :  {
            DPUTask {cluster_id = 0 : i64, outEnd = [32, 64, 64], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
        }

    %output_sparse = VPUIP.GroupSparseBuffer(%conv_out#0, %conv_out#1)
        -> !VPUIP.SparseBuffer<data=memref<1x32x64x64xf16, {order = #NHWC}, @CMX_NN>, sparsity_map=memref<1x32x64x64xi1, {order = #NHWC}, @CMX_NN>>

    %output_sparse_ddr = VPUIP.GroupSparseBuffer(%arg2, %arg3)
        -> !VPUIP.SparseBuffer<data=memref<1x32x64x64xf16, {order = #NHWC}>, sparsity_map=memref<1x32x64x64xi1, {order = #NHWC}>>

    %output_sparse_copy = VPUIP.Copy inputs(%output_sparse : !VPUIP.SparseBuffer<data=memref<1x32x64x64xf16, {order = #NHWC}, @CMX_NN>, sparsity_map=memref<1x32x64x64xi1, {order = #NHWC}, @CMX_NN>>)
                    outputs(%output_sparse_ddr : !VPUIP.SparseBuffer<data=memref<1x32x64x64xf16, {order = #NHWC}>, sparsity_map=memref<1x32x64x64xi1, {order = #NHWC}>>)
        -> !VPUIP.SparseBuffer<data=memref<1x32x64x64xf16, {order = #NHWC}>, sparsity_map=memref<1x32x64x64xi1, {order = #NHWC}>>

    %output_data, %output_sm = VPUIP.UngroupSparseBuffer(%output_sparse_copy) {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> memref<1x32x64x64xf16, {order = #NHWC}>, memref<1x32x64x64xi1, {order = #NHWC}>

    return %output_data, %output_sm : memref<1x32x64x64xf16, {order = #NHWC}>, memref<1x32x64x64xi1, {order = #NHWC}>

    // CHECK-DAG:   [[CST_WEIGHTS_SM:%.+]] = const.Declare memref<32x1x1x256xi1> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare memref<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]

    // CHECK:       [[INPUT_DATA_CMX:%.+]] = memref.alloc() : memref<1x16x64x64xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[INPUT_SM_CMX:%.+]] = memref.alloc() : memref<1x16x64x64xi1, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[INPUT_DATA:%.+]] = VPUIP.Copy inputs([[ARG0]]
    // CHECK-SAME:                                  outputs([[INPUT_DATA_CMX]]
    // CHECK-SAME:      -> memref<1x16x64x64xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[INPUT_SM:%.+]] = VPUIP.Copy inputs([[ARG1]]
    // CHECK-SAME:                                outputs([[INPUT_SM_CMX]]
    // CHECK-SAME:      -> memref<1x16x64x64xi1, {order = #NHWC}, @CMX_NN>

    // CHECK:       [[WEIGHTS_DATA_CMX:%.+]] = memref.alloc() : memref<32x16x3x3xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[WEIGHTS_SM_CMX:%.+]] = memref.alloc() : memref<32x1x1x256xi1, @CMX_NN>
    // CHECK:       [[WEIGHTS_DATA:%.+]] = VPUIP.Copy inputs([[CST_WEIGHTS]]
    // CHECK-SAME:                                    outputs([[WEIGHTS_DATA_CMX]]
    // CHECK-SAME:      -> memref<32x16x3x3xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[WEIGHTS_SM:%.+]] = VPUIP.Copy inputs([[CST_WEIGHTS_SM]]
    // CHECK-SAME:                                  outputs([[WEIGHTS_SM_CMX]]
    // CHECK-SAME:      -> memref<32x1x1x256xi1, @CMX_NN>

    // CHECK:       [[OUT_DATA:%.+]] = memref.alloc() : memref<1x32x64x64xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[OUT_SM:%.+]] = memref.alloc() : memref<1x32x64x64xi1, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[CONV_OUT:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[INPUT_DATA]]
    // CHECK-SAME:      input_sparsity_map([[INPUT_SM]]
    // CHECK-SAME:      weights([[WEIGHTS_DATA]]
    // CHECK-SAME:      weights_sparsity_map([[WEIGHTS_SM]]
    // CHECK-SAME:      parent_input([[INPUT_DATA]]
    // CHECK-SAME:      parent_input_sparsity_map([[INPUT_SM]]
    // CHECK-SAME:      parent_output([[OUT_DATA]]
    // CHECK-SAME:      parent_output_sparsity_map([[OUT_SM]]
    // CHECK-SAME:      outputs([[OUT_DATA]]
    // CHECK-SAME:      output_sparsity_map([[OUT_SM]]
    // CHECK-SAME:          -> memref<1x32x64x64xf16, {order = #NHWC}, @CMX_NN>, memref<1x32x64x64xi1, {order = #NHWC}, @CMX_NN>
    // CHECK-SAME:      variants : {
    // CHECK:               DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [32, 64, 64], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>}
    // CHECK:           } PPE : {
    // CHECK:           }

    // CHECK:       [[OUT_DATA_COPY:%.+]] = VPUIP.Copy inputs([[CONV_OUT]]#0 : memref<1x32x64x64xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:                                     outputs([[ARG2]] : memref<1x32x64x64xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> memref<1x32x64x64xf16, {order = #NHWC}>
    // CHECK:       [[OUT_SM_COPY:%.+]] = VPUIP.Copy inputs([[CONV_OUT]]#1 : memref<1x32x64x64xi1, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:                                   outputs([[ARG3]] : memref<1x32x64x64xi1, {order = #NHWC}>)
    // CHECK-SAME:      -> memref<1x32x64x64xi1, {order = #NHWC}>

    // CHECK:       return [[OUT_DATA_COPY]], [[OUT_SM_COPY]] : memref<1x32x64x64xf16, {order = #NHWC}>, memref<1x32x64x64xi1, {order = #NHWC}>
}
// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IODistributed = !VPUIP.DistributedBuffer<
    1x16x64x64xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!IOSMDistributed = !VPUIP.DistributedBuffer<
    1x16x64x64xi1, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    32x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsSMDistributed = !VPUIP.DistributedBuffer<
    32x1x1x256xi1, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!IOBuffer = memref<1x16x64x64xf16, {order = #NHWC}, @CMX_NN>
!IOSMBuffer = memref<1x16x64x64xi1, {order = #NHWC}, @CMX_NN>
!WeightsBuffer = memref<32x16x3x3xf16, {order = #NHWC}, @CMX_NN>
!WeightsSMBuffer = memref<32x1x1x256xi1, @CMX_NN>

// CHECK:       func.func @SparseConvDistributed(
// CHECK-SAME:      [[ARG0:%.+]]: !VPUIP.DistributedBuffer<1x16x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
// CHECK-SAME:      [[ARG1:%.+]]: !VPUIP.DistributedBuffer<1x16x64x64xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
// CHECK-SAME:      [[ARG2:%.+]]: !VPUIP.DistributedBuffer<32x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
// CHECK-SAME:      [[ARG3:%.+]]: !VPUIP.DistributedBuffer<32x1x1x256xi1, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
// CHECK-SAME:      -> (!VPUIP.DistributedBuffer<1x16x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
// CHECK-SAME:          !VPUIP.DistributedBuffer<1x16x64x64xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
func.func @SparseConvDistributed(%arg0: !IODistributed, %arg1: !IOSMDistributed, %arg2: !WeightsDistributed, %arg3: !WeightsSMDistributed)
        -> (!IODistributed, !IOSMDistributed) {
    %input_sparse = VPUIP.GroupSparseBuffer (%arg0, %arg1)
        -> !VPUIP.SparseBuffer<data=!IODistributed, sparsity_map=!IOSMDistributed>

    %weights_sparse = VPUIP.GroupSparseBuffer (%arg2, %arg3) <{is_weights}>
        -> !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights>

    %out_data = VPURT.AllocDistributed -> !IODistributed
    %out_sm = VPURT.AllocDistributed -> !IOSMDistributed

    %in_data, %in_sm = VPUIP.UngroupSparseBuffer(%input_sparse) {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> !IODistributed, !IOSMDistributed
    %w_data, %w_sm = VPUIP.UngroupSparseBuffer(%weights_sparse) {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> !WeightsDistributed, !WeightsSMDistributed

    %conv_output:2  = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>} <{
        kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        kernel_size = [3, 3],
        kernel_strides = [1, 1],
        task_type = #VPUIP.nce_task_type<CONV>
    }>
        input(%in_data : !IODistributed)
        input_sparsity_map(%in_sm : !IOSMDistributed)
        weights(%w_data : !WeightsDistributed)
        weights_sparsity_map(%w_sm : !WeightsSMDistributed)
        parent_input(%in_data : !IODistributed)
        parent_input_sparsity_map(%in_sm : !IOSMDistributed)
        parent_output(%out_data : !IODistributed)
        parent_output_sparsity_map(%out_sm : !IOSMDistributed)
        outputs(%out_data : !IODistributed)
        output_sparsity_map(%out_sm : !IOSMDistributed)
            -> !IODistributed, !IOSMDistributed
    variants :  {
        DPUTask {cluster_id = 0 : i64, outEnd = [32, 64, 64], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, outStart = [0, 0, 0]}
    } PPE :  {
    }

    %output_sparse = VPUIP.GroupSparseBuffer(%conv_output#0, %conv_output#1)
        -> !VPUIP.SparseBuffer<data=!IODistributed, sparsity_map=!IOSMDistributed>

    %conv_out_data, %conv_out_sm = VPUIP.UngroupSparseBuffer(%output_sparse) {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> !IODistributed, !IOSMDistributed

    return %conv_out_data, %conv_out_sm : !IODistributed, !IOSMDistributed

    // CHECK:       [[OUT_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[OUT_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x64xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[CONV_OUT:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[ARG0]]
    // CHECK-SAME:      input_sparsity_map([[ARG1]]
    // CHECK-SAME:      weights([[ARG2]]
    // CHECK-SAME:      weights_sparsity_map([[ARG3]]
    // CHECK-SAME:      parent_input([[ARG0]]
    // CHECK-SAME:      parent_input_sparsity_map([[ARG1]]
    // CHECK-SAME:      parent_output([[OUT_DATA]]
    // CHECK-SAME:      parent_output_sparsity_map([[OUT_SM]]
    // CHECK-SAME:      outputs([[OUT_DATA]]
    // CHECK-SAME:      output_sparsity_map([[OUT_SM]]
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x16x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:        !VPUIP.DistributedBuffer<1x16x64x64xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:      variants : {
    // CHECK:           DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [32, 64, 64], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>}
    // CHECK:         } PPE : {
    // CHECK:       }
    // CHECK:       return [[CONV_OUT]]#0, [[CONV_OUT]]#1
}
