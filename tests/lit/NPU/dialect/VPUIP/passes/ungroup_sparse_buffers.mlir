//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --run-ungroup-buffer-section-rewriters="rewriter=ungroup-sparse-buffer" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK:       func.func @SparseCopy([[ARG0:%.+]]: memref<32x16x3x3xf16>, [[ARG1:%.+]]: memref<32x16x3x3xi1>)
// CHECK-SAME:      -> (memref<32x16x3x3xf16, @CMX_NN>, memref<32x16x3x3xi1, @CMX_NN>)
func.func @SparseCopy(%arg0: memref<32x16x3x3xf16>, %arg1: memref<32x16x3x3xi1>) -> (memref<32x16x3x3xf16, @CMX_NN>, memref<32x16x3x3xi1, @CMX_NN>) {
    %0 = VPUIP.GroupSparseBuffer (%arg0, %arg1)
        -> !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16>, sparsity_map=memref<32x16x3x3xi1>>
    %1 = memref.alloc() : memref<32x16x3x3xf16, @CMX_NN>
    %2 = memref.alloc() : memref<32x16x3x3xi1, @CMX_NN>
    %3 = VPUIP.GroupSparseBuffer(%1, %2)
        -> !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16, @CMX_NN>, sparsity_map=memref<32x16x3x3xi1, @CMX_NN>>
    %4 = VPUIP.Copy inputs(%0 : !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16>, sparsity_map=memref<32x16x3x3xi1>>)
                    outputs(%3 : !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16, @CMX_NN>, sparsity_map=memref<32x16x3x3xi1, @CMX_NN>>)
        -> !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16, @CMX_NN>, sparsity_map=memref<32x16x3x3xi1, @CMX_NN>>
    %5, %6 = VPUIP.UngroupSparseBuffer(%4) {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> memref<32x16x3x3xf16, @CMX_NN>, memref<32x16x3x3xi1, @CMX_NN>

    return %5, %6 : memref<32x16x3x3xf16, @CMX_NN>, memref<32x16x3x3xi1, @CMX_NN>

    // CHECK:       [[VAR0:%.+]] = memref.alloc() : memref<32x16x3x3xf16, @CMX_NN>
    // CHECK:       [[VAR1:%.+]] = memref.alloc() : memref<32x16x3x3xi1, @CMX_NN>
    // CHECK:       [[VAR2:%.+]] = VPUIP.Copy inputs([[ARG0]] : memref<32x16x3x3xf16>)
    // CHECK-SAME:                            outputs([[VAR0]] : memref<32x16x3x3xf16, @CMX_NN>)
    // CHECK-SAME:                 -> memref<32x16x3x3xf16, @CMX_NN>
    // CHECK:       [[VAR3:%.+]] = VPUIP.Copy inputs([[ARG1]] : memref<32x16x3x3xi1>)
    // CHECK-SAME:                            outputs([[VAR1]] : memref<32x16x3x3xi1, @CMX_NN>)
    // CHECK-SAME:                 -> memref<32x16x3x3xi1, @CMX_NN>
    // CHECK:       return [[VAR2]], [[VAR3]] : memref<32x16x3x3xf16, @CMX_NN>, memref<32x16x3x3xi1, @CMX_NN>
}
// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Data_Distributed = !VPUIP.DistributedBuffer<
  32x16x3x3xf16, #NHWC, @CMX_NN, {
  mode = "DUPLICATED",
  num_clusters = 2 : i64
}>

!SM_Distributed = !VPUIP.DistributedBuffer<
  32x1x1x256xi1, #NCHW, @CMX_NN, {
  mode = "DUPLICATED",
  num_clusters = 2 : i64
}>

!Data_DDR = memref<32x16x3x3xf16, {order = #NHWC}>
!SM_DDR = memref<32x1x1x256xi1>

!Data_CMX = memref<32x16x3x3xf16, {order = #NHWC}, @CMX_NN>
!SM_CMX = memref<32x1x1x256xi1, @CMX_NN>

// CHECK:       func.func @SparseCopyDistributed([[ARG0:%.+]]: memref<32x16x3x3xf16, {order = #NHWC}>, [[ARG1:%.+]]: memref<32x1x1x256xi1>)
// CHECK-SAME:      -> (!VPUIP.DistributedBuffer<32x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
// CHECK-SAME:          !VPUIP.DistributedBuffer<32x1x1x256xi1, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
func.func @SparseCopyDistributed(%arg0: !Data_DDR, %arg1: !SM_DDR) -> (!Data_Distributed, !SM_Distributed) {
    %0 = VPUIP.GroupSparseBuffer (%arg0, %arg1) <{is_weights}> -> !VPUIP.SparseBuffer<data=!Data_DDR, sparsity_map=!SM_DDR, is_weights>

    %1 = VPURT.AllocDistributed -> !Data_Distributed
    %2 = VPURT.AllocDistributed -> !SM_Distributed
    %3 = VPUIP.GroupSparseBuffer(%1, %2) <{is_weights}> -> !VPUIP.SparseBuffer<data=!Data_Distributed, sparsity_map=!SM_Distributed, is_weights>

    %4 = VPUIP.Copy inputs(%0 : !VPUIP.SparseBuffer<data=!Data_DDR, sparsity_map=!SM_DDR, is_weights>)
                               outputs(%3 : !VPUIP.SparseBuffer<data=!Data_Distributed, sparsity_map=!SM_Distributed, is_weights>)
            -> !VPUIP.SparseBuffer<data=!Data_Distributed, sparsity_map=!SM_Distributed, is_weights>

    %5, %6 = VPUIP.UngroupSparseBuffer(%4) {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> !Data_Distributed, !SM_Distributed

    return %5, %6 : !Data_Distributed, !SM_Distributed

    // CHECK:       [[ALLOC_DATA:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<32x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[ALLOC_SM:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<32x1x1x256xi1, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[OUT_DATA:%.+]] = VPUIP.Copy inputs([[ARG0]]
    // CHECK-SAME:                                outputs([[ALLOC_DATA]]
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<32x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[OUT_SM:%.+]] = VPUIP.Copy inputs([[ARG1]]
    // CHECK-SAME:                              outputs([[ALLOC_SM]]
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<32x1x1x256xi1, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       return [[OUT_DATA]], [[OUT_SM]]
}
