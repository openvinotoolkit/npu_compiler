//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --optimize-copies %s | FileCheck %s
// REQUIRES: platform-NPU6010

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!InputDistributedType1 = !VPUIP.DistributedBuffer<
    1x256x46x20xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 256, 16, 20], [1, 256, 15, 20], [1, 256, 15, 20]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 31, 0]],
    memory_shapes = [[1, 256, 16, 20], [1, 256, 15, 20], [1, 256, 15, 20]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 31, 0]]
}>

!SubviewDistributedType = !VPUIP.DistributedBuffer<
    1x512x46x20xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 512, 16, 20], [1, 512, 15, 20], [1, 512, 15, 20]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 31, 0]],
    memory_shapes = [[1, 512, 16, 20], [1, 512, 15, 20], [1, 512, 15, 20]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 31, 0]]
}>

!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x256x46x20xf16, {order = #NHWC, strides = [471040, 1, 10240, 512]}, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 256, 16, 20], [1, 256, 15, 20], [1, 256, 15, 20]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 31, 0]],
    memory_shapes = [[1, 256, 16, 20], [1, 256, 15, 20], [1, 256, 15, 20]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 31, 0]]
}>

// CHECK-LABEL: func.func @DoNotOptimizeCMXToCMXCopySequenceWithTwoNCEUsers
func.func @DoNotOptimizeCMXToCMXCopySequenceWithTwoNCEUsers(%arg0 : !InputDistributedType1) {
    %0 = VPURT.AllocDistributed -> !InputDistributedType1

    %1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 4536 : i64} <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, eltwise_type = #VPU.eltwise_type<ADD>, is_inplace = true, mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}>
        input(%arg0 : !InputDistributedType1)
        weights(%arg0 : !InputDistributedType1)
        parent_input(%arg0 : !InputDistributedType1)
        parent_output(%0 : !InputDistributedType1)
        outputs(%0 : !InputDistributedType1)
        -> !InputDistributedType1 variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [19, 15, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [19, 15, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [19, 14, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [19, 14, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 2 : i64, inEnd = [19, 14, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [19, 14, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }

    %2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 4536 : i64} <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, eltwise_type = #VPU.eltwise_type<ADD>, is_inplace = true, mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}>
        input(%arg0 : !InputDistributedType1)
        weights(%arg0 : !InputDistributedType1)
        parent_input(%arg0 : !InputDistributedType1)
        parent_output(%0 : !InputDistributedType1)
        outputs(%0 : !InputDistributedType1)
        -> !InputDistributedType1 variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [19, 15, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [19, 15, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [19, 14, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [19, 14, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 2 : i64, inEnd = [19, 14, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [19, 14, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }

    %3 = VPURT.AllocDistributed -> !SubviewDistributedType
    %4 = VPUIP.SubView %3 [0, 256, 0, 0] [1, 256, 46, 20] : !SubviewDistributedType to !OutputDistributedType
    %5 = VPUIP.Copy inputs(%2 : !InputDistributedType1) outputs(%4 : !OutputDistributedType) -> !OutputDistributedType

    return

    // CHECK:      VPURT.AllocDistributed
    // CHECK:      VPUIP.NCEClusterTask
    // CHECK:      [[NCE_TASK:%.+]] = VPUIP.NCEClusterTask

    // CHECK:      [[ALLOC:%.+]] = VPURT.AllocDistributed
    // CHECK:      VPUIP.SubView [[ALLOC:%.+]] [0, 256, 0, 0] [1, 256, 46, 20]
    // CHECK:      VPUIP.Copy inputs([[NCE_TASK]]
}
