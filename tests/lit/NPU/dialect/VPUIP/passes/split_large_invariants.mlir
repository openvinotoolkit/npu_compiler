//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --platform=%platform% --split-large-invariants="max-variant-count=8"  %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DataType = memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
!WeightsType = memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
!WeightTableType = memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>

// CHECK-LABEL: @SplitIntoTwoTasks
module @SplitLargeInvariants attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {

    func.func @SplitIntoTwoTasks(%input: !DataType, %weights: !WeightsType, %weight_table: !WeightTableType,
                                 %output: !DataType) {
    %wait = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %update = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task waits(%wait: !VPURT.Barrier) updates(%update: !VPURT.Barrier) {
        %0 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }>
            input(%input : !DataType)
            weights(%weights : !WeightsType)
            weight_table(%weight_table : !WeightTableType)
            parent_input(%input : !DataType)
            parent_output(%output : !DataType)
            outputs(%output : !DataType) -> !DataType
            variants : {
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [1, 1, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 1, 31], outStart = [2, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [5, 1, 31], outStart = [4, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [7, 1, 31], outStart = [6, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [9, 1, 31], outStart = [8, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [11, 1, 31], outStart = [10, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 1, 31], outStart = [12, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 1, 31], outStart = [14, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            } PPE : {
            }
    }
    // CHECK: VPURT.Task waits(%{{.+}} : !VPURT.Barrier) updates([[CHAIN:%[^ ]+]] : !VPURT.Barrier) {
    // CHECK: VPUIP.NCEClusterTask
    // CHECK: variants : {
    // CHECK-NEXT: DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [1, 1, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NEXT: DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 1, 31], outStart = [2, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NEXT: DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [5, 1, 31], outStart = [4, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NEXT: DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [7, 1, 31], outStart = [6, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NOT: DPUTask
    // CHECK: } PPE : {
    // CHECK: VPURT.Task waits(%{{.+}} : !VPURT.Barrier) updates(%{{.+}} : !VPURT.Barrier) {
    // CHECK: VPUIP.NCEClusterTask
    // CHECK: variants : {
    // CHECK-NEXT: DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [9, 1, 31], outStart = [8, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NEXT: DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [11, 1, 31], outStart = [10, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NEXT: DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 1, 31], outStart = [12, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NEXT: DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 1, 31], outStart = [14, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NOT: DPUTask
    // CHECK: } PPE : {

        return
    }
}
