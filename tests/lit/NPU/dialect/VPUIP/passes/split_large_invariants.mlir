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

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DataType = memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
!WeightsType = memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
!WeightTableType = memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>

!ProfType = memref<32xui64, [@CMX_NN, 0]>

// CHECK-LABEL: @SplitIntoTwoTasksWithProfiling
module @SplitLargeInvariantsWithProfiling attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {

    func.func @SplitIntoTwoTasksWithProfiling(%input: !DataType, %weights: !WeightsType, %weight_table: !WeightTableType,
                                              %output: !DataType) {
    %prof = VPURT.DeclareBuffer <CMX_NN> [0] <256> -> !ProfType
    // CHECK: [[PROF_CHUNK_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <256> -> memref<16xui64, [@CMX_NN, 0]>
    %wait = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %update = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task waits(%wait: !VPURT.Barrier) updates(%update: !VPURT.Barrier) {
        %0:2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 1, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>,
                profilingMetadata = #VPUIP.DpuProfilingMetadataAttr<bufferId = 1 : i64, taskId = 2 : i64, maxVariants = 8 : i64, numVariants = 8 : i64, clusterId = 0 : i64>
            }>
            input(%input : !DataType)
            weights(%weights : !WeightsType)
            weight_table(%weight_table : !WeightTableType)
            parent_input(%input : !DataType)
            parent_output(%output : !DataType)
            outputs(%output : !DataType)
            profiling_data(%prof : !ProfType)
            -> !DataType, !ProfType
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
    // CHECK-SAME: profilingMetadata = #VPUIP.DpuProfilingMetadataAttr<bufferId = 1 : i64, taskId = 2 : i64, maxVariants = 4 : i64, numVariants = 4 : i64, clusterId = 0 : i64>
    // CHECK-SAME: profiling_data([[PROF_CHUNK_0]] : memref<16xui64, [@CMX_NN, 0]>)
    // CHECK: variants : {
    // CHECK-NEXT: DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [1, 1, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NEXT: DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 1, 31], outStart = [2, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NEXT: DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [5, 1, 31], outStart = [4, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NEXT: DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [7, 1, 31], outStart = [6, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NOT: DPUTask
    // CHECK: } PPE : {

    // CHECK: [[PROF_CHUNK_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <384> -> memref<16xui64, [@CMX_NN, 0]>
    // CHECK: VPURT.Task waits([[CHAIN]] : !VPURT.Barrier) updates(%{{.+}} : !VPURT.Barrier) {
    // CHECK: VPUIP.NCEClusterTask
    // CHECK-SAME: profilingMetadata = #VPUIP.DpuProfilingMetadataAttr<bufferId = 1 : i64, taskId = 3 : i64, maxVariants = 4 : i64, numVariants = 4 : i64, clusterId = 0 : i64>
    // CHECK-SAME: profiling_data([[PROF_CHUNK_1]] : memref<16xui64, [@CMX_NN, 0]>)
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

// -----

// When the NCEClusterTask uses sprLookupTable, InsertDelayDPUVariant has prepended a dummy DPUTask
// at variants[0] (loc tagged with "dummy"). SplitLargeInvariantsPass must keep that dummy at the
// front of every chunk it produces — chunk 0 already gets it via the normal [0, K) range, but
// chunks with begin > 0 lose the dummy under the legacy implementation. That breaks the LUT-load
// FSM (sprLutRead/forceInvRead would then be consumed on a real variant) and was observed as a
// SIGSEGV in NCEClusterTaskRewriter

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DataType = memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
!WeightsType = memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
!WeightTableType = memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
!SprLutType = memref<256xui16, [@CMX_NN, 0]>

// CHECK-LABEL: @KeepDummyVariantInEveryChunkForSprLut
module @SplitLargeInvariantsKeepDummy attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {

    func.func @KeepDummyVariantInEveryChunkForSprLut(%input: !DataType, %weights: !WeightsType,
                                                     %weight_table: !WeightTableType,
                                                     %spr_lut: !SprLutType, %output: !DataType) {
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
            spr_lookup_table(%spr_lut : !SprLutType)
            parent_input(%input : !DataType)
            parent_output(%output : !DataType)
            outputs(%output : !DataType) -> !DataType
            variants : {
                // The dummy variant inserted by InsertDelayDPUVariant is identified by being at
                // variants[0] in front of the real variants; its workload size is also typically
                // smaller than the real ones (here outEnd=[0,0,15]).
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [0, 0, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
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

    // The 9 input variants (1 dummy + 8 real) get split with max-variant-count=8 → variantsPerChunk = 4.
    // numChunks = ceil(9 / 4) = 3 chunks. Chunk 0 keeps [0,4) which already includes the dummy as
    // variants[0]. Chunk 1 keeps [4,8) plus a preserved dummy at the front → 5 variants. Chunk 2
    // keeps [8,9) plus a preserved dummy at the front → 2 variants.

    // CHECK: VPURT.Task waits(%{{.+}} : !VPURT.Barrier) updates([[CHAIN0:%[^ ]+]] : !VPURT.Barrier) {
    // CHECK: VPUIP.NCEClusterTask
    // CHECK: spr_lookup_table
    // CHECK: variants : {
    // CHECK-NEXT: DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [0, 0, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NEXT: DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [1, 1, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NEXT: DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 1, 31], outStart = [2, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NEXT: DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [5, 1, 31], outStart = [4, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NOT: DPUTask
    // CHECK: } PPE : {

    // Chunk 1 must lead with the preserved dummy variant (outEnd=[0,0,15] outStart=[0,0,0]).
    // CHECK: VPURT.Task waits([[CHAIN0]] : !VPURT.Barrier) updates([[CHAIN1:%[^ ]+]] : !VPURT.Barrier) {
    // CHECK: VPUIP.NCEClusterTask
    // CHECK: spr_lookup_table
    // CHECK: variants : {
    // CHECK-NEXT: DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [0, 0, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NEXT: DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [7, 1, 31], outStart = [6, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NEXT: DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [9, 1, 31], outStart = [8, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NEXT: DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [11, 1, 31], outStart = [10, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NEXT: DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 1, 31], outStart = [12, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NOT: DPUTask
    // CHECK: } PPE : {

    // Chunk 2 (the tail) must also lead with the preserved dummy variant.
    // CHECK: VPURT.Task waits([[CHAIN1]] : !VPURT.Barrier) updates(%{{.+}} : !VPURT.Barrier) {
    // CHECK: VPUIP.NCEClusterTask
    // CHECK: spr_lookup_table
    // CHECK: variants : {
    // CHECK-NEXT: DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [0, 0, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NEXT: DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 1, 31], outStart = [14, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NOT: DPUTask
    // CHECK: } PPE : {

        return
    }
}
