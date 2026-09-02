//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --one-shot-bufferize-VPU-to-VPUIP --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 0.78431372549019607>

// CHECK-LABEL: @SuperdenseNCEEltwise
func.func @SuperdenseNCEEltwise(%arg0: tensor<1x32x16x320xf16, {order = #NHWC}>) -> tensor<1x32x16x320x!qElemType, {order = #NWCH}> {
    %1 = VPU.Copy(%arg0) { out_mem_space = @CMX_NN } : tensor<1x32x16x320xf16, {order = #NHWC}> -> !VPU.DistributedTensor<1x32x16x320xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 4, 320], [1, 32, 4, 320], [1, 32, 4, 320], [1, 32, 4, 320]], compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]], memory_shapes = [[1, 32, 4, 320], [1, 32, 4, 320], [1, 32, 4, 320], [1, 32, 4, 320]], memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]]}>

    %2 = VPU.NCE.Eltwise(%1, %1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, minimumHardwareExecutionCost = 3052 : i64, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}-> !VPU.DistributedTensor<1x32x16x320x!qElemType, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 4, 320], [1, 32, 4, 320], [1, 32, 4, 320], [1, 32, 4, 320]], compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]], memory_shapes = [[1, 32, 4, 320], [1, 32, 4, 320], [1, 32, 4, 320], [1, 32, 4, 320]], memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]]}> {
            VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 32, 4, 320] outOffsets [0, 0, 0, 0] outSizes [1, 32, 4, 320] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 32, 4, 320] outOffsets [0, 0, 0, 0] outSizes [1, 32, 4, 320] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 1 : i64}
            VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 32, 4, 320] outOffsets [0, 0, 0, 0] outSizes [1, 32, 4, 320] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 2 : i64}
            VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 32, 4, 320] outOffsets [0, 0, 0, 0] outSizes [1, 32, 4, 320] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 3 : i64}
        }
    %3 = VPU.Copy(%2) : !VPU.DistributedTensor<1x32x16x320x!qElemType, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 4, 320], [1, 32, 4, 320], [1, 32, 4, 320], [1, 32, 4, 320]], compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]], memory_shapes = [[1, 32, 4, 320], [1, 32, 4, 320], [1, 32, 4, 320], [1, 32, 4, 320]], memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]]}> -> tensor<1x32x16x320x!qElemType, {order = #NWCH}>

    return %3 : tensor<1x32x16x320x!qElemType, {order = #NWCH}>

    // CHECK:       VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3052 : i64} <{
    // CHECK-SAME:      is_superdense,
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:  }
}
