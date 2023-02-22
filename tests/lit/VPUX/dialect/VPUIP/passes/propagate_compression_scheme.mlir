//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-compression-scheme %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK:  func @SparseConvWeights([[ARG0:%.+]]: memref<1x16x64x64xf16, #NHWC>, [[ARG1:%.+]]: memref<1x32x64x64xf16, #NHWC>) -> memref<1x32x64x64xf16, #NHWC>
func @SparseConvWeights(%arg0: memref<1x16x64x64xf16, #NHWC>, %arg1: memref<1x32x64x64xf16, #NHWC>) -> memref<1x32x64x64xf16, #NHWC> {
    %input_cmx = memref.alloc() : memref<1x16x64x64xf16, #NHWC, @CMX_NN>
    %input = VPUIP.Copy inputs(%arg0 : memref<1x16x64x64xf16, #NHWC>)
                        outputs(%input_cmx : memref<1x16x64x64xf16, #NHWC, @CMX_NN>)
        -> memref<1x16x64x64xf16, #NHWC, @CMX_NN>

    %cst_weights = const.Declare memref<32x16x3x3xf16, #NHWC> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false, dense<1> : tensor<32xi64>>]
    %cst_weights_sm = const.Declare memref<32x1x1x256xi1> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %cst_weights_sparse = VPUIP.GroupSparseBuffer (%cst_weights, %cst_weights_sm) {compression_scheme = #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>, is_weights}
        -> !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16, #NHWC>, sparsity_map=memref<32x1x1x256xi1>, is_weights, #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>>

    %weights_data_cmx = memref.alloc() : memref<32x16x3x3xf16, #NHWC, @CMX_NN>
    %weights_sm_cmx = memref.alloc() : memref<32x1x1x256xi1, @CMX_NN>
    %weights_sparse_cmx = VPUIP.GroupSparseBuffer (%weights_data_cmx, %weights_sm_cmx) {compression_scheme = #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>, is_weights}
        -> !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16, #NHWC, @CMX_NN>, sparsity_map=memref<32x1x1x256xi1, @CMX_NN>, is_weights, #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>>

    %weights = VPUIP.Copy inputs(%cst_weights_sparse : !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16, #NHWC>, sparsity_map=memref<32x1x1x256xi1>, is_weights, #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>>)
                          outputs(%weights_sparse_cmx : !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16, #NHWC, @CMX_NN>, sparsity_map=memref<32x1x1x256xi1, @CMX_NN>, is_weights, #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>>)
        -> !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16, #NHWC, @CMX_NN>, sparsity_map=memref<32x1x1x256xi1, @CMX_NN>, is_weights, #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>>

    %cst_weights_table = const.Declare memref<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %weights_table_cmx = memref.alloc() : memref<32x1x1x4xsi32, @CMX_NN>
    %weights_table = VPUIP.Copy inputs(%cst_weights_table : memref<32x1x1x4xsi32>)
                                outputs(%weights_table_cmx : memref<32x1x1x4xsi32, @CMX_NN>)
        -> memref<32x1x1x4xsi32, @CMX_NN>

    %weights_data, %weights_sm = VPUIP.UngroupSparseBuffer(%weights) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>}
        -> memref<32x16x3x3xf16, #NHWC, @CMX_NN>, memref<32x1x1x256xi1, @CMX_NN>

    %output_cmx = memref.alloc() : memref<1x32x64x64xf16, #NHWC, @CMX_NN>
    %conv_out = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
            kernel_size = [3, 3],
            kernel_strides = [1, 1],
            task_type = "CONV"
        }
        input(%input : memref<1x16x64x64xf16, #NHWC, @CMX_NN>)
        weights(%weights_data : memref<32x16x3x3xf16, #NHWC, @CMX_NN>)
        weights_sparsity_map(%weights_sm : memref<32x1x1x256xi1, @CMX_NN>)
        weight_table(%weights_table : memref<32x1x1x4xsi32, @CMX_NN>)
        parent_input(%input : memref<1x16x64x64xf16, #NHWC, @CMX_NN>)
        parent_output(%output_cmx : memref<1x32x64x64xf16, #NHWC, @CMX_NN>)
        outputs(%output_cmx : memref<1x32x64x64xf16, #NHWC, @CMX_NN>)
            -> memref<1x32x64x64xf16, #NHWC, @CMX_NN>
        variants :  {
            DPUTask {cluster_id = 0 : i64, mpe_mode = "VECTOR_FP16", outEnd = [32, 64, 64], outStart = [0, 0, 0], pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}}
        } PPE :  {
        }

    %output = VPUIP.Copy inputs(%conv_out : memref<1x32x64x64xf16, #NHWC, @CMX_NN>)
                    outputs(%arg1 : memref<1x32x64x64xf16, #NHWC>)
        -> memref<1x32x64x64xf16, #NHWC>

    return %output : memref<1x32x64x64xf16, #NHWC>

    // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare memref<32x16x3x3xf16, {compressionScheme = #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>, order = #NHWC}>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false, dense<1> : tensor<32xi64>>]
    // CHECK-DAG:   [[CST_WEIGHTS_SM:%.+]] = const.Declare memref<32x1x1x256xi1> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK:       [[CST_WEIGHTS_SPARSE:%.+]] = VPUIP.GroupSparseBuffer([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]]) {
    // CHECK-SAME:          compression_scheme = #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>, is_weights
    // CHECK-SAME:      } -> !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16, {compressionScheme = #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>, order = #NHWC}>,
    // CHECK-SAME:                               sparsity_map=memref<32x1x1x256xi1>, is_weights, #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>>

    // CHECK:       [[WEIGHTS_DATA_CMX:%.+]] = memref.alloc() : memref<32x16x3x3xf16, {compressionScheme = #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>, order = #NHWC}, @CMX_NN>
    // CHECK:       [[WEIGHTS_SM_CMX:%.+]] = memref.alloc() : memref<32x1x1x256xi1, @CMX_NN>
    // CHECK:       [[WEIGHTS_SPARSE_CMX:%.+]] = VPUIP.GroupSparseBuffer([[WEIGHTS_DATA_CMX]], [[WEIGHTS_SM_CMX]]) {
    // CHECK-SAME:          compression_scheme = #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>, is_weights
    // CHECK-SAME:      } -> !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16, {compressionScheme = #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>, order = #NHWC}, @CMX_NN>,
    // CHECK-SAME:                               sparsity_map=memref<32x1x1x256xi1, @CMX_NN>, is_weights, #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>>

    // CHECK:       [[WEIGHTS:%.+]] = VPUIP.Copy inputs([[CST_WEIGHTS_SPARSE]]
    // CHECK-SAME:                               outputs([[WEIGHTS_SPARSE_CMX]]

    // CHECK:       [[WEIGHTS_DATA:%.+]], [[WEIGHTS_SM:%.+]] = VPUIP.UngroupSparseBuffer([[WEIGHTS]])
    // CHECK-SAME:      -> memref<32x16x3x3xf16, {compressionScheme = #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>, order = #NHWC}, @CMX_NN>,
    // CHEKC-SAME:         memref<32x1x1x256xi1, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      weights([[WEIGHTS_DATA]]
    // CHECK-SAME:      weights_sparsity_map([[WEIGHTS_SM]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IODistributed = type !VPUIP.DistributedBuffer<
    1x16x64x64xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    32x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsSMDistributed = type !VPUIP.DistributedBuffer<
    32x1x1x256xi1, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsBufferDDR = type memref<32x16x3x3xf16, #NHWC>
!WeightsSMBufferDDR = type memref<32x1x1x256xi1>

!IOBuffer = type memref<1x16x64x64xf16, #NHWC, @CMX_NN>
!WeightsBuffer = type memref<32x16x3x3xf16, #NHWC, @CMX_NN>
!WeightsSMBuffer = type memref<32x1x1x256xi1, @CMX_NN>
!WeightsTableBuffer = type memref<32x1x1x4xsi32, @CMX_NN>

// CHECK:       func @SparseConvWeightsDistributed(
// CHECK-SAME:      [[ARG0:%.+]]: !VPUIP.DistributedBuffer<1x16x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
// CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x16x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
func @SparseConvWeightsDistributed(%arg0: !IODistributed) -> !IODistributed {
    %cst_weights = const.Declare !WeightsBufferDDR = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false, dense<1> : tensor<32xi64>>]
    %cst_weights_sm = const.Declare !WeightsSMBufferDDR = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %cst_weights_sparse = VPUIP.GroupSparseBuffer (%cst_weights, %cst_weights_sm) {compression_scheme = #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>, is_weights}
        -> !VPUIP.SparseBuffer<data=!WeightsBufferDDR, sparsity_map=!WeightsSMBufferDDR, is_weights, #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>>

    %weights_data_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %weights_sm_cmx = VPURT.AllocDistributed -> !WeightsSMDistributed
    %weights_sparse_cmx = VPUIP.GroupSparseBuffer (%weights_data_cmx, %weights_sm_cmx) {compression_scheme = #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>, is_weights}
        -> !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights, #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>>

    %weights = VPUIP.NCEClusterTiling inputs(%cst_weights_sparse as %arg1: !VPUIP.SparseBuffer<data=!WeightsBufferDDR, sparsity_map=!WeightsSMBufferDDR, is_weights, #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>>)
                                      outputs(%weights_sparse_cmx as %arg2: !VPUIP.SparseBuffer<data=!WeightsBuffer, sparsity_map=!WeightsSMBuffer, is_weights, #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>>)
            -> !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights, #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>> {
      %0 = VPUIP.Copy inputs(%arg1 : !VPUIP.SparseBuffer<data=!WeightsBufferDDR, sparsity_map=!WeightsSMBufferDDR, is_weights, #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>>)
                      outputs(%arg2 : !VPUIP.SparseBuffer<data=!WeightsBuffer, sparsity_map=!WeightsSMBuffer, is_weights, #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>>)
                -> !VPUIP.SparseBuffer<data=!WeightsBuffer, sparsity_map=!WeightsSMBuffer, is_weights, #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>>
    }

    %cst_weights_table = const.Declare !WeightsTableBuffer = dense<1> : tensor<32x1x1x4xsi32>

    %output_data = VPURT.AllocDistributed -> !IODistributed

    %weights_data, %weights_sm = VPUIP.UngroupSparseBuffer(%weights) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>}
        -> !WeightsDistributed, !WeightsSMDistributed

    %output = VPUIP.NCEClusterTiling
        inputs(%arg0 as %arg1: !IOBuffer,
               %weights_data as %arg2: !WeightsBuffer,
               %weights_sm as %arg3: !WeightsSMBuffer,
               %cst_weights_table as %arg4: !WeightsTableBuffer)
        outputs(%output_data as %arg5: !IOBuffer) -> !IODistributed {

        %conv_out = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
            kernel_size = [3, 3],
            kernel_strides = [1, 1],
            task_type = "CONV"
        }
            input(%arg1 : !IOBuffer)
            weights(%arg2 : !WeightsBuffer)
            weights_sparsity_map(%arg3 : !WeightsSMBuffer)
            weight_table(%arg4 : !WeightsTableBuffer)
            parent_input(%arg1 : !IOBuffer)
            parent_output(%arg5 : !IOBuffer)
            outputs(%arg5 : !IOBuffer) -> !IOBuffer
        variants :  {
            DPUTask {cluster_id = 0 : i64, mpe_mode = "VECTOR_FP16", outEnd = [32, 64, 64], outStart = [0, 0, 0], pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}}
        } PPE :  {
        }
    }

    return %output : !IODistributed

    // CHECK:       [[CST_WEIGHTS:%.+]] = const.Declare memref<32x16x3x3xf16, {compressionScheme = #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>, order = #NHWC}>
    // CHECK-SAME:     = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false, dense<1> : tensor<32xi64>>]
    // CHECK:       [[CST_WEIGHTS_SM:%.+]] = const.Declare memref<32x1x1x256xi1> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK:       [[CST_WEIGHTS_SPARSE:%.+]] = VPUIP.GroupSparseBuffer([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]]) {
    // CHECK-SAME:          compression_scheme = #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>,
    // CHECK-SAME:          is_weights
    // CHECK-SAME:      } -> !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16, {compressionScheme = #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>, order = #NHWC}>,
    // CHECK-SAME:                               sparsity_map=memref<32x1x1x256xi1>, is_weights, #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>>

    // CHECK:       [[WEIGHTS_DATA_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<32x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64},
    // CHECK-SAME:      #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>>
    // CHECK:       [[WEIGHTS_SM_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<32x1x1x256xi1, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN,
    // CHECK-SAME:      {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[WEIGHTS_SPARSE_CMX:%.+]] = VPUIP.GroupSparseBuffer([[WEIGHTS_DATA_CMX]], [[WEIGHTS_SM_CMX]]) {
    // CHECK-SAME:          compression_scheme = #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>,
    // CHECK-SAME:          is_weights
    // CHECK-SAME:      } -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<32x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64},
    // CHECK-SAME:                                    #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>>,
    // CHECK-SAME:                               sparsity_map=!VPUIP.DistributedBuffer<32x1x1x256xi1, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN,
    // CHECK-SAME:                                            {mode = "DUPLICATED", num_clusters = 2 : i64}>, is_weights, #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>>

    // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[CST_WEIGHTS_SPARSE]] as [[INNER_IN:[^:]+]]:
    // CHECK-SAME:             !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16, {compressionScheme = #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>, order = #NHWC}>,
    // CHECK-SAME:                                 sparsity_map=memref<32x1x1x256xi1>,
    // CHECK-SAME:                                 is_weights,
    // CHECK-SAME:                                 #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>>)
    // CHECK-SAME:      outputs([[WEIGHTS_SPARSE_CMX]] as [[INNER_OUT:[^:]+]]:
    // CHECK-SAME:              !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16, {compressionScheme = #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>, order = #NHWC}, @CMX_NN>,
    // CHECK-SAME:                                  sparsity_map=memref<32x1x1x256xi1, @CMX_NN>,
    // CHECK-SAME:                                  is_weights,
    // CHECK-SAME:                                  #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>>)
    // CHECK-SAME:      -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<32x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}, #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>>,
    // CHECK-SAME:                             sparsity_map=!VPUIP.DistributedBuffer<32x1x1x256xi1, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
    // CHECK-SAME:                             is_weights, #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>> {
    // CHECK:           VPUIP.Copy inputs([[INNER_IN]]
    // CHECK-SAME:                 outputs([[INNER_OUT]]
    // CHECK:       }

    // CHECK:       [[WEIGHTS_DATA:%.+]], [[WEIGHTS_SM:%.+]] = VPUIP.UngroupSparseBuffer([[WEIGHTS_SPARSE]])
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<32x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}, #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>>,
    // CHECK-SAME:         !VPUIP.DistributedBuffer<32x1x1x256xi1, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       VPUIP.NCEClusterTiling inputs(
    // CHECK-SAME:          [[WEIGHTS_DATA]] as [[INNER_W:[^:]+]]: memref<32x16x3x3xf16, {compressionScheme = #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>, order = #NHWC}, @CMX_NN>,
    // CHECK-SAME:          [[WEIGHTS_SM]] as [[INNER_W_SM:[^:]+]]: memref<32x1x1x256xi1, @CMX_NN>,
    // CHECK:         VPUIP.NCEClusterTask
    // CHECK-SAME:      weights([[INNER_W]]
    // CHECK-SAME:      weights_sparsity_map([[INNER_W_SM]]
    // CHECK:         }
    // CHECK:       }
}
