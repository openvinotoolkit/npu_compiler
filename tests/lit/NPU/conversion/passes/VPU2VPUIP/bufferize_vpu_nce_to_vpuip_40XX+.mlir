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

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

!qElemType = !quant.uniform<!quant.quantile<ui2:f16, {0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00}>:f16, 1.000000e+00>

// CHECK-LABEL: @MatMulQuantileTypeWeights
func.func @MatMulQuantileTypeWeights(%arg0: tensor<43x1x32x16x4xf16, {mem_space = @CMX_NN, order = #GNHWC}>,
                                      %arg1: tensor<43x64x32x1x1x!qElemType, {mem_space = @CMX_NN, order = #GNHWC}>,
                                      %arg2: tensor<43x64x1x1x4xsi32, {mem_space = @CMX_NN}>)
                                      -> tensor<43x1x64x16x4xf16, {mem_space = @CMX_NN, order = #GNHWC}> {

    %0 = VPU.NCE.MatMul(%arg0, %arg1, %arg2) rawFilterShape [43, 64, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEStub<>,
                strides = [1, 1]
            } -> tensor<43x1x64x16x4xf16, {mem_space = @CMX_NN, order = #GNHWC}> {
        VPU.DPU.Workload inOffsets [0, 0, 0, 0, 0] inSizes [43, 1, 32, 16, 4] outOffsets [0, 0, 0, 0, 0] outSizes [43, 1, 64, 16, 4] pad [0, 0, 0, 0] <CUBOID_16x16>
    }

    return %0 : tensor<43x1x64x16x4xf16, {mem_space = @CMX_NN, order = #GNHWC}>

    // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<43x1x64x16x4xf16, {order = #GNHWC}, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:  input({{%[^ ]+}} : memref<43x1x32x16x4xf16, {order = #GNHWC}, @CMX_NN>)
    // CHECK-SAME:  weights({{%[^ ]+}} : memref<43x64x32x1x1x!qElemType, {order = #GNHWC}, @CMX_NN>)
    // CHECK-SAME:  weight_table({{%[^ ]+}} : memref<43x64x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:  parent_input({{%[^ ]+}} : memref<43x1x32x16x4xf16, {order = #GNHWC}, @CMX_NN>)
    // CHECK-SAME:  parent_output([[OUT_BUF]] : memref<43x1x64x16x4xf16, {order = #GNHWC}, @CMX_NN>)
    // CHECK-SAME:  outputs([[OUT_BUF]] : memref<43x1x64x16x4xf16, {order = #GNHWC}, @CMX_NN>)

    // CHECK: return
    // CHECK-SAME: memref<43x1x64x16x4xf16, {order = #GNHWC}, @CMX_NN>
}
