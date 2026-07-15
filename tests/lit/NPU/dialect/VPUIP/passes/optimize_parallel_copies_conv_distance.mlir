//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --optimize-parallel-copies %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// Test cost-distance based copy fusion with mixed CONV + ELTWISE siblings.
// getSiblingEltwise returns true if any sibling is in-place eltwise, enabling cost-distance check.
// Copies within cost distance are fused; copies beyond cost distance are kept separate.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!buffDistrType = !VPUIP.DistributedBuffer<1x64x250x32xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                        compute_shapes = [[1, 64, 63, 32], [1, 64, 63, 32], [1, 64, 62, 32], [1, 64, 62, 32]],
                        compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0], [0, 0, 126, 0], [0, 0, 188, 0]],
                        memory_shapes = [[1, 64, 63, 32], [1, 64, 63, 32], [1, 64, 62, 32], [1, 64, 62, 32]],
                        memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0], [0, 0, 126, 0], [0, 0, 188, 0]]}>
!buffDistrType1 = !VPUIP.DistributedBuffer<1x64x250x32xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                    compute_shapes = [[1, 64, 63, 32], [1, 64, 63, 32], [1, 64, 62, 32], [1, 64, 62, 32]],
                    compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0], [0, 0, 126, 0], [0, 0, 188, 0]],
                    memory_shapes = [[1, 64, 63, 32], [1, 64, 63, 32], [1, 64, 62, 32], [1, 64, 62, 32]],
                    memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0], [0, 0, 126, 0], [0, 0, 188, 0]]}>
!Weights_CMX = memref<64x64x1x1xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @OptimizeCopiesConsiderDistanceForConv
func.func @OptimizeCopiesConsiderDistanceForConv() -> !buffDistrType {
    %weights = memref.alloc() : !Weights_CMX

    %0 = memref.alloc() : memref<1x64x250x250xf16, #NHWC, @DDR>
    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 64, 250, 32] : memref<1x64x250x250xf16, #NHWC, @DDR> to memref<1x64x250x32xf16, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>
    %2 = VPURT.AllocDistributed -> !buffDistrType1
    %3 = VPUIP.ViewOp %2 : !buffDistrType1 to !buffDistrType
    %4 = VPURT.AllocDistributed -> !buffDistrType1
    %5 = VPUIP.Copy inputs(%1 : memref<1x64x250x32xf16, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>) outputs(%4 : !buffDistrType1) -> !buffDistrType1

    %6 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
    input(%5 : !buffDistrType1) weights(%weights : !Weights_CMX) parent_input(%5 : !buffDistrType1) parent_output(%3 : !buffDistrType) outputs(%3 : !buffDistrType) -> !buffDistrType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 3 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }

    %7 = VPURT.AllocDistributed -> !buffDistrType1
    %8 = VPUIP.ViewOp %7 : !buffDistrType1 to !buffDistrType
    %9 = VPURT.AllocDistributed -> !buffDistrType1
    %10 = VPURT.AllocDistributed -> !buffDistrType1
    // %11 will be fused to %5, since it is within cost distance
    %11 = VPUIP.Copy inputs(%1 : memref<1x64x250x32xf16, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>) outputs(%10 : !buffDistrType1) -> !buffDistrType1
    %12 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
    <{is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}> input(%9 : !buffDistrType1) weights(%11 : !buffDistrType1) parent_input(%9 : !buffDistrType1) parent_output(%8 : !buffDistrType) outputs(%8 : !buffDistrType) -> !buffDistrType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 3 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }

    %13 = VPURT.AllocDistributed -> !buffDistrType1
    %14 = VPUIP.ViewOp %13 : !buffDistrType1 to !buffDistrType
    %15 = VPURT.AllocDistributed -> !buffDistrType1
    %16 = VPURT.AllocDistributed -> !buffDistrType1
    // %17 will be fused to %5, since it is within cost distance
    %17 = VPUIP.Copy inputs(%1 : memref<1x64x250x32xf16, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>) outputs(%16 : !buffDistrType1) -> !buffDistrType1
    %18 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
    <{is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}> input(%15 : !buffDistrType1) weights(%17 : !buffDistrType1) parent_input(%15 : !buffDistrType1) parent_output(%14 : !buffDistrType) outputs(%14 : !buffDistrType) -> !buffDistrType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 3 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }

    %19 = VPURT.AllocDistributed -> !buffDistrType1
    %20 = VPUIP.ViewOp %19 : !buffDistrType1 to !buffDistrType
    %21 = VPURT.AllocDistributed -> !buffDistrType1
    %22 = VPURT.AllocDistributed -> !buffDistrType1
    // %23 will not be fused, since it is beyond cost distance
    %23 = VPUIP.Copy inputs(%1 : memref<1x64x250x32xf16, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>) outputs(%22 : !buffDistrType1) -> !buffDistrType1
    %24 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}> input(%21 : !buffDistrType1) weights(%23 : !buffDistrType1) parent_input(%21 : !buffDistrType1) parent_output(%8 : !buffDistrType) outputs(%8 : !buffDistrType) -> !buffDistrType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 3 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }

    %25 = VPURT.AllocDistributed -> !buffDistrType1
    %26 = VPUIP.ViewOp %25 : !buffDistrType1 to !buffDistrType
    %27 = VPURT.AllocDistributed -> !buffDistrType1
    %28 = VPURT.AllocDistributed -> !buffDistrType1
    // %29 will be fused to %23, since it is within cost distance
    %29 = VPUIP.Copy inputs(%1 : memref<1x64x250x32xf16, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>) outputs(%28 : !buffDistrType1) -> !buffDistrType1
    %30 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}> input(%27 : !buffDistrType1) weights(%29 : !buffDistrType1) parent_input(%27 : !buffDistrType1) parent_output(%26 : !buffDistrType) outputs(%26 : !buffDistrType) -> !buffDistrType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 3 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    return %30 : !buffDistrType

    // On non-NPU50XX: any_of returns true (eltwise sibling exists), cost-distance check applies.
    // Copies within cost distance fuse together, copies beyond distance start a new group.
    // Result: 2 copies (COPY for first group, COPY_1 for second group).

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x64x250x250xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 64, 250, 32]

    // CHECK: [[COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<1x64x250x32xf16, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>)

    // CHECK: [[NCE:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1],
    // CHECK:     input([[COPY]] : !VPUIP.DistributedBuffer

    // CHECK: [[NCE_1:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64}
    // CHECK:     <{is_inplace = true
    // CHECK:     weights([[COPY]] : !VPUIP.DistributedBuffer

    // CHECK: [[NCE_2:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64}
    // CHECK:     <{is_inplace = true
    // CHECK:     weights([[COPY]] : !VPUIP.DistributedBuffer

    // CHECK: [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<1x64x250x32xf16, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>)
    // CHECK: [[NCE_3:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64}
    // CHECK:     <{is_inplace = true
    // CHECK:     weights([[COPY_1]] : !VPUIP.DistributedBuffer

    // CHECK: [[NCE_4:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64}
    // CHECK:     <{is_inplace = true
    // CHECK:     weights([[COPY_1]] : !VPUIP.DistributedBuffer
}
