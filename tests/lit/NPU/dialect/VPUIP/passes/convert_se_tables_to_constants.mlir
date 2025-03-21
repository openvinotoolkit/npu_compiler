//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --mlir-print-elementsattrs-with-hex-if-larger=-1 --convert-se-tables-to-constants %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_DDR = memref<1x32x3x3xf16, #NHWC>
!InputSM_DDR = memref<1x32x3x3xi1, #NHWC>
!InputSE_DDR = memref<1x1x6x6xi32, #NHWC>

!Input_CMX = memref<1x32x3x3xf16, #NHWC, @CMX_NN>
!InputSM_CMX = memref<1x32x3x3xi1, #NHWC, @CMX_NN>
!InputSE_CMX = memref<1x1x6x6xi32, #NHWC, @CMX_NN>
!Weights_CMX = memref<32x32x1x1xf16, #NHWC, @CMX_NN>
!WeightsTable_CMX = memref<32x1x1x4xsi32, @CMX_NN>
!Output_CMX = memref<1x32x6x6xf16, #NHWC, @CMX_NN>

!SparseBufferDDR = !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR,
    #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                       nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>>

!SparseBufferCMX = !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX,
    #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                       nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>>

// CHECK-LABEL:  func.func @SETableInterpolateNearestSingleCluster
func.func @SETableInterpolateNearestSingleCluster(%input_data: !Input_DDR, %input_sm: !InputSM_DDR,
                                     %weights_data: !Weights_CMX, %weights_table: !WeightsTable_CMX
        ) -> !Output_CMX {
    %input_se = VPUIP.StorageElementTable {
                basePtrs = dense<0> : tensor<36xi32>,
                dataShape = [1, 32, 3, 3], dataElemType = f16,
                seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                            nearest_mode = <FLOOR>,  offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>,
                seSize = 32, seDepth = 1
            } -> !InputSE_DDR
    %input_sparse = VPUIP.GroupSparseBuffer (%input_data, %input_sm, %input_se) {
            seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                        nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>
        } -> !SparseBufferDDR

    %input_data_cmx = memref.alloc() : !Input_CMX
    %input_sm_cmx = memref.alloc() : !InputSM_CMX
    %input_se_cmx = memref.alloc() : !InputSE_CMX
    %input_sparse_cmx = VPUIP.GroupSparseBuffer (%input_data_cmx, %input_sm_cmx, %input_se_cmx) {
            seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                        nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>
        } -> !SparseBufferCMX

    %input = VPUIP.Copy inputs(%input_sparse: !SparseBufferDDR) outputs(%input_sparse_cmx: !SparseBufferCMX) -> !SparseBufferCMX

    %in_data, %in_sm, %in_se = VPUIP.UngroupSparseBuffer(%input) {resultSegmentSizes = array<i32: 1, 1, 1>}
        -> !Input_CMX, !InputSM_CMX, !InputSE_CMX
    %out_cmx = memref.alloc() : !Output_CMX

    %conv_out = VPUIP.NCEClusterTiling inputs(%in_data as %arg3: !Input_CMX,
                                              %in_sm as %arg4: !InputSM_CMX,
                                              %in_se as %arg5: !InputSE_CMX,
                                              %weights_data as %arg6: !Weights_CMX,
                                              %weights_table as %arg7: !WeightsTable_CMX)
                                      outputs(%out_cmx as %arg8: !Output_CMX)
            -> !Output_CMX {
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
            weight_table(%arg7 : !WeightsTable_CMX)
            parent_input(%arg3 : !Input_CMX)
            parent_input_sparsity_map(%arg4 : !InputSM_CMX)
            parent_input_storage_element_table(%arg5 : !InputSE_CMX)
            parent_output(%arg8 : !Output_CMX)
            outputs(%arg8 : !Output_CMX)
                -> !Output_CMX
            variants :  {
                DPUTask {cluster_id = 0 : i64, outStart = [0, 0, 0], outEnd = [16, 6, 6], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            } PPE :  {
            }
    }

    return %conv_out : !Output_CMX

    // Pointers have the following offsets:
    //     0   0  64  64 128 128
    //     0   0  64  64 128 128
    //   192 192 256 256 320 320
    //   192 192 256 256 320 320
    //   384 384 448 448 512 512
    //   384 384 448 448 512 512
    // Without the last 4 bits:
    //      0  0  4  4  8  8
    //      0  0  4  4  8  8
    //     12 12 16 16 20 20
    //     12 12 16 16 20 20
    //     24 24 28 28 32 32
    //     24 24 28 28 32 32
    // Shifted left 9 times:
    //         0     0  2048  2048  4096  4096
    //         0     0  2048  2048  4096  4096
    //      6144  6144  8192  8192 10240 10240
    //      6144  6144  8192  8192 10240 10240
    //     12288 12288 14336 14336 16384 16384
    //     12288 12288 14336 14336 16384 16384

    // CHECK-NOT:            VPUIP.StorageElementTable
    // CHECK:                const.Declare memref<1x1x6x6xi32, #NHWC> = dense<
    // CHECK-SAME{LITERAL}:    [[[[0, 0, 2048, 2048, 4096, 4096],
    // CHECK-SAME{LITERAL}:       [0, 0, 2048, 2048, 4096, 4096],
    // CHECK-SAME{LITERAL}:       [6144, 6144, 8192, 8192, 10240, 10240],
    // CHECK-SAME{LITERAL}:       [6144, 6144, 8192, 8192, 10240, 10240],
    // CHECK-SAME{LITERAL}:       [12288, 12288, 14336, 14336, 16384, 16384],
    // CHECK-SAME{LITERAL}:       [12288, 12288, 14336, 14336, 16384, 16384]]]]> : tensor<1x1x6x6xi32, {order = #NHWC}>
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
    1x1x6x6xi32, #NHWC, @CMX_NN, {
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

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x32x3x3xf16, #NHWC>
!InputSM_DDR = memref<1x32x3x3xi1, #NHWC>
!InputSE_DDR = memref<1x1x6x6xi32, #NHWC>

!Input_CMX = memref<1x32x3x3xf16, #NHWC, @CMX_NN>
!InputSM_CMX = memref<1x32x3x3xi1, #NHWC, @CMX_NN>
!InputSE_CMX = memref<1x1x6x6xi32, #NHWC, @CMX_NN>
!Weights_CMX = memref<32x32x1x1xf16, #NHWC, @CMX_NN>
!WeightsTable_CMX = memref<32x1x1x4xsi32, @CMX_NN>
!Output_CMX = memref<1x32x6x6xf16, #NHWC, @CMX_NN>

!SparseBufferDDR = !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR,
    #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                       nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>>

!SparseBufferCMX = !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX,
    #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                       nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>>

!SparseBufferDistributed = !VPUIP.SparseBuffer<data=!InputDistributed, sparsity_map=!InputSMDistributed, storage_element_table=!InputSEDistributed,
    #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                       nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>>

// CHECK-LABEL:  func.func @SETableInterpolateNearest
func.func @SETableInterpolateNearest(%input_data: !Input_DDR, %input_sm: !InputSM_DDR,
                                     %weights_data: !WeightsDistributed, %weights_table: !WeightsTableDistributed
        ) -> !OutputDistributed {
    %input_se = VPUIP.StorageElementTable {
                basePtrs = dense<[0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0,
                                  1, 1, 1, 1, 1, 1,
                                  1, 1, 1, 1, 1, 1]> : tensor<36xi32>,
                dataShape = [1, 32, 3, 3], dataElemType = f16,
                seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                            nearest_mode = <FLOOR>,  offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>,
                seSize = 32, seDepth = 1
            } -> !InputSE_DDR
    %input_sparse = VPUIP.GroupSparseBuffer (%input_data, %input_sm, %input_se) {
            seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                        nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>
        } -> !SparseBufferDDR

    %input_data_cmx = VPURT.AllocDistributed -> !InputDistributed
    %input_sm_cmx = VPURT.AllocDistributed -> !InputSMDistributed
    %input_se_cmx = VPURT.AllocDistributed -> !InputSEDistributed
    %input_sparse_cmx = VPUIP.GroupSparseBuffer (%input_data_cmx, %input_sm_cmx, %input_se_cmx) {
            seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                        nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>
        } -> !SparseBufferDistributed

    %input = VPUIP.NCEClusterTiling inputs(%input_sparse as %arg3: !SparseBufferDDR) outputs(%input_sparse_cmx as %arg4: !SparseBufferCMX) -> !SparseBufferDistributed {
        %0 = VPUIP.Copy inputs(%arg3: !SparseBufferDDR) outputs(%arg4: !SparseBufferCMX) -> !SparseBufferCMX
    }

    %in_data, %in_sm, %in_se = VPUIP.UngroupSparseBuffer(%input) {resultSegmentSizes = array<i32: 1, 1, 1>}
        -> !InputDistributed, !InputSMDistributed, !InputSEDistributed
    %out_cmx = VPURT.AllocDistributed -> !OutputDistributed

    %conv_out = VPUIP.NCEClusterTiling inputs(%in_data as %arg3: !Input_CMX,
                                              %in_sm as %arg4: !InputSM_CMX,
                                              %in_se as %arg5: !InputSE_CMX,
                                              %weights_data as %arg6: !Weights_CMX,
                                              %weights_table as %arg7: !WeightsTable_CMX)
                                      outputs(%out_cmx as %arg8: !Output_CMX)
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
            weight_table(%arg7 : !WeightsTable_CMX)
            parent_input(%arg3 : !Input_CMX)
            parent_input_sparsity_map(%arg4 : !InputSM_CMX)
            parent_input_storage_element_table(%arg5 : !InputSE_CMX)
            parent_output(%arg8 : !Output_CMX)
            outputs(%arg8 : !Output_CMX)
                -> !Output_CMX
            variants :  {
                DPUTask {cluster_id = 0 : i64, outStart = [0, 0, 0], outEnd = [16, 6, 6], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            } PPE :  {
            }
    }

    return %conv_out : !OutputDistributed

    // Pointers have the following offsets:
    //     0   0  64  64 128 128
    //     0   0  64  64 128 128
    //   192 192 256 256 320 320
    //   192 192 256 256 320 320
    //   384 384 448 448 512 512
    //   384 384 448 448 512 512
    // The offsets are reset for separate clusters:
    //     0   0  64  64 128 128
    //     0   0  64  64 128 128
    //   192 192 256 256 320 320
    //   192 192 256 256 320 320
    //     0   0  64  64 128 128
    //     0   0  64  64 128 128
    // Without the last 4 bits:
    //      0  0  4  4  8  8
    //      0  0  4  4  8  8
    //     12 12 16 16 20 20
    //     12 12 16 16 20 20
    //      0  0  4  4  8  8
    //      0  0  4  4  8  8
    // Shifted left 9 times:
    //        0     0  2048  2048  4096  4096
    //        0     0  2048  2048  4096  4096
    //     6144  6144  8192  8192 10240 10240
    //     6144  6144  8192  8192 10240 10240
    //        0     0  2048  2048  4096  4096
    //        0     0  2048  2048  4096  4096
    // Then, the base_ptrs values are added to the last 9 bits.

    // CHECK-NOT:            VPUIP.StorageElementTable
    // CHECK:                const.Declare memref<1x1x6x6xi32, #NHWC> = dense<
    // CHECK-SAME{LITERAL}:    [[[[0, 0, 2048, 2048, 4096, 4096],
    // CHECK-SAME{LITERAL}:       [0, 0, 2048, 2048, 4096, 4096],
    // CHECK-SAME{LITERAL}:       [6144, 6144, 8192, 8192, 10240, 10240],
    // CHECK-SAME{LITERAL}:       [6144, 6144, 8192, 8192, 10240, 10240],
    // CHECK-SAME{LITERAL}:       [1, 1, 2049, 2049, 4097, 4097],
    // CHECK-SAME{LITERAL}:       [1, 1, 2049, 2049, 4097, 4097]]]]> : tensor<1x1x6x6xi32, {order = #NHWC}>
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
    1x32x7x7xi1, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!InputSEDistributed = !VPUIP.DistributedBuffer<
    1x1x7x7xi32, #NHWC, @CMX_NN, {
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
    32x32x2x2xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x32x3x3xf16, #NHWC>
!InputSM_DDR = memref<1x32x7x7xi1, #NHWC>
!InputSE_DDR = memref<1x1x7x7xi32, #NHWC>

!Input_CMX = memref<1x32x3x3xf16, #NHWC, @CMX_NN>
!InputSM_CMX = memref<1x32x7x7xi1, #NHWC, @CMX_NN>
!InputSE_CMX = memref<1x1x7x7xi32, #NHWC, @CMX_NN>
!Weights_CMX = memref<32x32x2x2xf16, #NHWC, @CMX_NN>
!WeightsTable_CMX = memref<32x1x1x4xsi32, @CMX_NN>
!Output_CMX = memref<1x32x6x6xf16, #NHWC, @CMX_NN>

!SparseBufferDDR = !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR,
    #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                       offsets = [0, 0, 0, 0], sizes = [1, 32, 7, 7]>>

!SparseBufferCMX = !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX,
    #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                       offsets = [0, 0, 0, 0], sizes = [1, 32, 7, 7]>>

!SparseBufferDistributed = !VPUIP.SparseBuffer<data=!InputDistributed, sparsity_map=!InputSMDistributed, storage_element_table=!InputSEDistributed,
    #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                       offsets = [0, 0, 0, 0], sizes = [1, 32, 7, 7]>>

// CHECK-LABEL:  func.func @SETableInterpolateBilinear
func.func @SETableInterpolateBilinear(%input_data: !Input_DDR, %input_sm: !InputSM_DDR,
                                     %weights_data: !WeightsDistributed, %weights_table: !WeightsTableDistributed
        ) -> !OutputDistributed {
    %input_se = VPUIP.StorageElementTable {
                basePtrs = dense<[0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0,
                                  1, 1, 1, 1, 1, 1, 1,
                                  1, 1, 1, 1, 1, 1, 1,
                                  1, 1, 1, 1, 1, 1, 1]> : tensor<49xi32>,
                dataShape = [1, 32, 3, 3], dataElemType = f16,
                seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                            offsets = [0, 0, 0, 0], sizes = [1, 32, 7, 7]>,
                seSize = 32, seDepth = 1
            } -> !InputSE_DDR
    %input_sparse = VPUIP.GroupSparseBuffer (%input_data, %input_sm, %input_se) {
            seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                        offsets = [0, 0, 0, 0], sizes = [1, 32, 7, 7]>
        } -> !SparseBufferDDR

    %input_data_cmx = VPURT.AllocDistributed -> !InputDistributed
    %input_sm_cmx = VPURT.AllocDistributed -> !InputSMDistributed
    %input_se_cmx = VPURT.AllocDistributed -> !InputSEDistributed
    %input_sparse_cmx = VPUIP.GroupSparseBuffer (%input_data_cmx, %input_sm_cmx, %input_se_cmx) {
            seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                        offsets = [0, 0, 0, 0], sizes = [1, 32, 7, 7]>
        } -> !SparseBufferDistributed

    %input = VPUIP.NCEClusterTiling inputs(%input_sparse as %arg3: !SparseBufferDDR) outputs(%input_sparse_cmx as %arg4: !SparseBufferCMX) -> !SparseBufferDistributed {
        %0 = VPUIP.Copy inputs(%arg3: !SparseBufferDDR) outputs(%arg4: !SparseBufferCMX) -> !SparseBufferCMX
    }

    %in_data, %in_sm, %in_se = VPUIP.UngroupSparseBuffer(%input) {resultSegmentSizes = array<i32: 1, 1, 1>}
        -> !InputDistributed, !InputSMDistributed, !InputSEDistributed
    %out_cmx = VPURT.AllocDistributed -> !OutputDistributed

    %conv_out = VPUIP.NCEClusterTiling inputs(%in_data as %arg3: !Input_CMX,
                                              %in_sm as %arg4: !InputSM_CMX,
                                              %in_se as %arg5: !InputSE_CMX,
                                              %weights_data as %arg6: !Weights_CMX,
                                              %weights_table as %arg7: !WeightsTable_CMX)
                                      outputs(%out_cmx as %arg8: !Output_CMX)
            -> !OutputDistributed {
        %0 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [2, 2],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%arg3 : !Input_CMX)
            input_sparsity_map(%arg4 : !InputSM_CMX)
            input_storage_element_table(%arg5 : !InputSE_CMX)
            weights(%arg6 : !Weights_CMX)
            weight_table(%arg7 : !WeightsTable_CMX)
            parent_input(%arg3 : !Input_CMX)
            parent_input_sparsity_map(%arg4 : !InputSM_CMX)
            parent_input_storage_element_table(%arg5 : !InputSE_CMX)
            parent_output(%arg8 : !Output_CMX)
            outputs(%arg8 : !Output_CMX)
                -> !Output_CMX
            variants :  {
                DPUTask {cluster_id = 0 : i64, outStart = [0, 0, 0], outEnd = [16, 6, 6], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            } PPE :  {
            }
    }

    return %conv_out : !OutputDistributed

    // Pointers have the following offsets:
    //     0   0  64  64 128 128 128
    //     0   0  64  64 128 128 128
    //   192 192 256 256 320 320 320
    //   192 192 256 256 320 320 320
    //   384 384 448 448 512 512 512
    //   384 384 448 448 512 512 512
    //   384 384 448 448 512 512 512
    // The offsets are reset for separate clusters:
    //     0   0  64  64 128 128 128
    //     0   0  64  64 128 128 128
    //   192 192 256 256 320 320 320
    //   192 192 256 256 320 320 320
    //     0   0  64  64 128 128 128
    //     0   0  64  64 128 128 128
    //     0   0  64  64 128 128 128
    // Without the last 4 bits:
    //      0  0  4  4  8  8  8
    //      0  0  4  4  8  8  8
    //     12 12 16 16 20 20 20
    //     12 12 16 16 20 20 20
    //      0  0  4  4  8  8  8
    //      0  0  4  4  8  8  8
    //      0  0  4  4  8  8  8
    // Shifted left 9 times:
    //        0     0  2048  2048  4096  4096  4096
    //        0     0  2048  2048  4096  4096  4096
    //     6144  6144  8192  8192 10240 10240 10240
    //     6144  6144  8192  8192 10240 10240 10240
    //        0     0  2048  2048  4096  4096  4096
    //        0     0  2048  2048  4096  4096  4096
    //        0     0  2048  2048  4096  4096  4096
    // Then, the base_ptrs values are added to the last 9 bits.


    // CHECK-NOT:            VPUIP.StorageElementTable
    // CHECK:                const.Declare memref<1x1x7x7xi32, #NHWC> = dense<
    // CHECK-SAME{LITERAL}:    [[[[0, 0, 2048, 2048, 4096, 4096, 4096],
    // CHECK-SAME{LITERAL}:       [0, 0, 2048, 2048, 4096, 4096, 4096],
    // CHECK-SAME{LITERAL}:       [6144, 6144, 8192, 8192, 10240, 10240, 10240],
    // CHECK-SAME{LITERAL}:       [6144, 6144, 8192, 8192, 10240, 10240, 10240],
    // CHECK-SAME{LITERAL}:       [1, 1, 2049, 2049, 4097, 4097, 4097],
    // CHECK-SAME{LITERAL}:       [1, 1, 2049, 2049, 4097, 4097, 4097],
    // CHECK-SAME{LITERAL}:       [1, 1, 2049, 2049, 4097, 4097, 4097]]]]> : tensor<1x1x7x7xi32, {order = #NHWC}>
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
                  basePtrs = dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]> : tensor<220xi32>,
                  dataShape = [1, 16, 4, 6], dataElemType = f16,
                  seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
                                              offsets = [0, 0, 0, 0], sizes = [1, 16, 11, 20], initial_input_shape = [1, 16, 6, 6], initial_output_shape = [1, 16, 18, 18]>,
                  seSize = 16, seDepth = 1
              } -> !InputSE_DDR_Tile
    %input_se_1 = VPUIP.StorageElementTable {
                  basePtrs = dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]> : tensor<220xi32>,
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

    %output_1 = VPUIP.SubView %output_alloc [0, 0, 9, 0] [1, 16, 9, 18] : !Output_DDR to memref<1x16x9x18xf16, {order = #NHWC, strides = [5184, 1, 288, 16]}, @DDR>
    %copy_output_1 = VPUIP.NCEClusterTiling inputs(%conv_out_1 as %arg2: memref<1x16x9x18xf16, #NHWC, @CMX_NN>) outputs(%output_1 as %arg3: memref<1x16x9x18xf16, #NHWC>) -> memref<1x16x9x18xf16, {order = #NHWC, strides = [5184, 1, 288, 16]}, @DDR> {
        %0 = VPUIP.Copy inputs(%arg2 : memref<1x16x9x18xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x16x9x18xf16, #NHWC>) -> memref<1x16x9x18xf16, #NHWC>
    }

    return %output_alloc : !Output_DDR

    // Step1: Pointers have the following offsets (Tile 0):
    //     0   0   0   0  32  32  32  64  64  64  96  96  96 128 128 128 160 160 160 160
    //     0   0   0   0  32  32  32  64  64  64  96  96  96 128 128 128 160 160 160 160
    //     0   0   0   0  32  32  32  64  64  64  96  96  96 128 128 128 160 160 160 160
    //     0   0   0   0  32  32  32  64  64  64  96  96  96 128 128 128 160 160 160 160
    //   192 192 192 192 224 224 224 256 256 256 288 288 288 320 320 320 352 352 352 352
    //   192 192 192 192 224 224 224 256 256 256 288 288 288 320 320 320 352 352 352 352
    //   192 192 192 192 224 224 224 256 256 256 288 288 288 320 320 320 352 352 352 352
    //   355 355 355 355 387 387 387 390 390 390 422 422 422 454 454 454 486 486 486 486
    //   355 355 355 355 387 387 387 390 390 390 422 422 422 454 454 454 486 486 486 486
    //   355 355 355 355 387 387 387 390 390 390 422 422 422 454 454 454 486 486 486 486
    //   518 518 518 518 550 550 550 582 582 582 614 614 614 646 646 646 678 678 678 678

    // Step2: The offsets are reset for separate clusters (Tile 0):
    //     0   0   0   0  32  32  32  64  64  64  96  96  96 128 128 128 160 160 160 160
    //     0   0   0   0  32  32  32  64  64  64  96  96  96 128 128 128 160 160 160 160
    //     0   0   0   0  32  32  32  64  64  64  96  96  96 128 128 128 160 160 160 160
    //     0   0   0   0  32  32  32  64  64  64  96  96  96 128 128 128 160 160 160 160
    //   192 192 192 192 224 224 224 256 256 256 288 288 288 320 320 320 352 352 352 352
    //   192 192 192 192 224 224 224 256 256 256 288 288 288 320 320 320 352 352 352 352
    //   192 192 192 192 224 224 224 256 256 256 288 288 288 320 320 320 352 352 352 352
    //     0   0   0   0  32  32  32  64  64  64  96  96  96 128 128 128 160 160 160 160
    //     0   0   0   0  32  32  32  64  64  64  96  96  96 128 128 128 160 160 160 160
    //     0   0   0   0  32  32  32  64  64  64  96  96  96 128 128 128 160 160 160 160
    //   192 192 192 192 224 224 224 256 256 256 288 288 288 320 320 320 352 352 352 352

    // Step3: Without the last 4 bits (Tile 0):
    //    0  0  0  0  2  2  2  4  4  4  6  6  6  8  8  8 10 10 10 10
    //    0  0  0  0  2  2  2  4  4  4  6  6  6  8  8  8 10 10 10 10
    //    0  0  0  0  2  2  2  4  4  4  6  6  6  8  8  8 10 10 10 10
    //    0  0  0  0  2  2  2  4  4  4  6  6  6  8  8  8 10 10 10 10
    //   12 12 12 12 14 14 14 16 16 16 18 18 18 20 20 20 22 22 22 22
    //   12 12 12 12 14 14 14 16 16 16 18 18 18 20 20 20 22 22 22 22
    //   12 12 12 12 14 14 14 16 16 16 18 18 18 20 20 20 22 22 22 22
    //    0  0  0  0  2  2  2  4  4  4  6  6  6  8  8  8 10 10 10 10
    //    0  0  0  0  2  2  2  4  4  4  6  6  6  8  8  8 10 10 10 10
    //    0  0  0  0  2  2  2  4  4  4  6  6  6  8  8  8 10 10 10 10
    //   12 12 12 12 14 14 14 16 16 16 18 18 18 20 20 20 22 22 22 22

    // Step4: Shifted left 9 times (Tile 0):
    //      0    0    0    0 1024 1024 1024 2048 2048 2048 3072 3072 3072  4096  4096  4096  5120  5120  5120  5120
    //      0    0    0    0 1024 1024 1024 2048 2048 2048 3072 3072 3072  4096  4096  4096  5120  5120  5120  5120
    //      0    0    0    0 1024 1024 1024 2048 2048 2048 3072 3072 3072  4096  4096  4096  5120  5120  5120  5120
    //      0    0    0    0 1024 1024 1024 2048 2048 2048 3072 3072 3072  4096  4096  4096  5120  5120  5120  5120
    //   6144 6144 6144 6144 7168 7168 7168 8192 8192 8192 9216 9216 9216 10240 10240 10240 11264 11264 11264 11264
    //   6144 6144 6144 6144 7168 7168 7168 8192 8192 8192 9216 9216 9216 10240 10240 10240 11264 11264 11264 11264
    //   6144 6144 6144 6144 7168 7168 7168 8192 8192 8192 9216 9216 9216 10240 10240 10240 11264 11264 11264 11264
    //      0    0    0    0 1024 1024 1024 2048 2048 2048 3072 3072 3072  4096  4096  4096  5120  5120  5120  5120
    //      0    0    0    0 1024 1024 1024 2048 2048 2048 3072 3072 3072  4096  4096  4096  5120  5120  5120  5120
    //      0    0    0    0 1024 1024 1024 2048 2048 2048 3072 3072 3072  4096  4096  4096  5120  5120  5120  5120
    //   6144 6144 6144 6144 7168 7168 7168 8192 8192 8192 9216 9216 9216 10240 10240 10240 11264 11264 11264 11264

    // Step5: The base_ptrs values are added to the last 9 bits (Tile 0):
    // CHECK-NOT:   VPUIP.StorageElementTable
    // CHECK:       const.Declare memref<1x1x11x20xi32, #NHWC, @DDR> = dense<
    // CHECK-SAME{LITERAL}:    [0, 0, 0, 0, 1024, 1024, 1024, 2048, 2048, 2048, 3072, 3072, 3072, 4096, 4096, 4096, 5120, 5120, 5120, 5120],
    // CHECK-SAME{LITERAL}:    [0, 0, 0, 0, 1024, 1024, 1024, 2048, 2048, 2048, 3072, 3072, 3072, 4096, 4096, 4096, 5120, 5120, 5120, 5120],
    // CHECK-SAME{LITERAL}:    [0, 0, 0, 0, 1024, 1024, 1024, 2048, 2048, 2048, 3072, 3072, 3072, 4096, 4096, 4096, 5120, 5120, 5120, 5120],
    // CHECK-SAME{LITERAL}:    [0, 0, 0, 0, 1024, 1024, 1024, 2048, 2048, 2048, 3072, 3072, 3072, 4096, 4096, 4096, 5120, 5120, 5120, 5120],
    // CHECK-SAME{LITERAL}:    [6144, 6144, 6144, 6144, 7168, 7168, 7168, 8192, 8192, 8192, 9216, 9216, 9216, 10240, 10240, 10240, 11264, 11264, 11264, 11264],
    // CHECK-SAME{LITERAL}:    [6144, 6144, 6144, 6144, 7168, 7168, 7168, 8192, 8192, 8192, 9216, 9216, 9216, 10240, 10240, 10240, 11264, 11264, 11264, 11264],
    // CHECK-SAME{LITERAL}:    [6144, 6144, 6144, 6144, 7168, 7168, 7168, 8192, 8192, 8192, 9216, 9216, 9216, 10240, 10240, 10240, 11264, 11264, 11264, 11264],
    // CHECK-SAME{LITERAL}:    [1, 1, 1, 1, 1025, 1025, 1025, 2049, 2049, 2049, 3073, 3073, 3073, 4097, 4097, 4097, 5121, 5121, 5121, 5121],
    // CHECK-SAME{LITERAL}:    [1, 1, 1, 1, 1025, 1025, 1025, 2049, 2049, 2049, 3073, 3073, 3073, 4097, 4097, 4097, 5121, 5121, 5121, 5121],
    // CHECK-SAME{LITERAL}:    [1, 1, 1, 1, 1025, 1025, 1025, 2049, 2049, 2049, 3073, 3073, 3073, 4097, 4097, 4097, 5121, 5121, 5121, 5121],
    // CHECK-SAME{LITERAL}:    [6145, 6145, 6145, 6145, 7169, 7169, 7169, 8193, 8193, 8193, 9217, 9217, 9217, 10241, 10241, 10241, 11265, 11265, 11265, 11265]
    // CHECK-SAME:  : tensor<1x1x11x20xi32, {order = #NHWC}>

    // Step4: Shifted left 9 times (Tile 1):
    //      0    0    0    0 1024 1024 1024 2048 2048 2048 3072 3072 3072  4096  4096  4096  5120  5120  5120  5120
    //   6144 6144 6144 6144 7168 7168 7168 8192 8192 8192 9216 9216 9216 10240 10240 10240 11264 11264 11264 11264
    //   6144 6144 6144 6144 7168 7168 7168 8192 8192 8192 9216 9216 9216 10240 10240 10240 11264 11264 11264 11264
    //   6144 6144 6144 6144 7168 7168 7168 8192 8192 8192 9216 9216 9216 10240 10240 10240 11264 11264 11264 11264
    //      0    0    0    0 1024 1024 1024 2048 2048 2048 3072 3072 3072  4096  4096  4096  5120  5120  5120  5120
    //      0    0    0    0 1024 1024 1024 2048 2048 2048 3072 3072 3072  4096  4096  4096  5120  5120  5120  5120
    //      0    0    0    0 1024 1024 1024 2048 2048 2048 3072 3072 3072  4096  4096  4096  5120  5120  5120  5120
    //   6144 6144 6144 6144 7168 7168 7168 8192 8192 8192 9216 9216 9216 10240 10240 10240 11264 11264 11264 11264
    //   6144 6144 6144 6144 7168 7168 7168 8192 8192 8192 9216 9216 9216 10240 10240 10240 11264 11264 11264 11264
    //   6144 6144 6144 6144 7168 7168 7168 8192 8192 8192 9216 9216 9216 10240 10240 10240 11264 11264 11264 11264
    //   6144 6144 6144 6144 7168 7168 7168 8192 8192 8192 9216 9216 9216 10240 10240 10240 11264 11264 11264 11264

    // Step5: The base_ptrs values are added to the last 9 bits (Tile 1):
    // CHECK-NOT:   VPUIP.StorageElementTable
    // CHECK:       const.Declare memref<1x1x11x20xi32, #NHWC, @DDR> = dense<
    // CHECK-SAME{LITERAL}:    [0, 0, 0, 0, 1024, 1024, 1024, 2048, 2048, 2048, 3072, 3072, 3072, 4096, 4096, 4096, 5120, 5120, 5120, 5120],
    // CHECK-SAME{LITERAL}:    [6144, 6144, 6144, 6144, 7168, 7168, 7168, 8192, 8192, 8192, 9216, 9216, 9216, 10240, 10240, 10240, 11264, 11264, 11264, 11264],
    // CHECK-SAME{LITERAL}:    [6144, 6144, 6144, 6144, 7168, 7168, 7168, 8192, 8192, 8192, 9216, 9216, 9216, 10240, 10240, 10240, 11264, 11264, 11264, 11264],
    // CHECK-SAME{LITERAL}:    [6144, 6144, 6144, 6144, 7168, 7168, 7168, 8192, 8192, 8192, 9216, 9216, 9216, 10240, 10240, 10240, 11264, 11264, 11264, 11264],
    // CHECK-SAME{LITERAL}:    [1, 1, 1, 1, 1025, 1025, 1025, 2049, 2049, 2049, 3073, 3073, 3073, 4097, 4097, 4097, 5121, 5121, 5121, 5121],
    // CHECK-SAME{LITERAL}:    [1, 1, 1, 1, 1025, 1025, 1025, 2049, 2049, 2049, 3073, 3073, 3073, 4097, 4097, 4097, 5121, 5121, 5121, 5121],
    // CHECK-SAME{LITERAL}:    [1, 1, 1, 1, 1025, 1025, 1025, 2049, 2049, 2049, 3073, 3073, 3073, 4097, 4097, 4097, 5121, 5121, 5121, 5121],
    // CHECK-SAME{LITERAL}:    [6145, 6145, 6145, 6145, 7169, 7169, 7169, 8193, 8193, 8193, 9217, 9217, 9217, 10241, 10241, 10241, 11265, 11265, 11265, 11265],
    // CHECK-SAME{LITERAL}:    [6145, 6145, 6145, 6145, 7169, 7169, 7169, 8193, 8193, 8193, 9217, 9217, 9217, 10241, 10241, 10241, 11265, 11265, 11265, 11265],
    // CHECK-SAME{LITERAL}:    [6145, 6145, 6145, 6145, 7169, 7169, 7169, 8193, 8193, 8193, 9217, 9217, 9217, 10241, 10241, 10241, 11265, 11265, 11265, 11265],
    // CHECK-SAME{LITERAL}:    [6145, 6145, 6145, 6145, 7169, 7169, 7169, 8193, 8193, 8193, 9217, 9217, 9217, 10241, 10241, 10241, 11265, 11265, 11265, 11265]
    // CHECK-SAME:  : tensor<1x1x11x20xi32, {order = #NHWC}>
}
